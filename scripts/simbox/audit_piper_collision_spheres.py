#!/usr/bin/env python3
"""Audit Piper CuRobo spheres against URDF collision-mesh surface samples.

Distances are computed in each link's local frame.  Rigid FK transforms preserve
these distances, so the reported bound is valid for every sampled joint pose;
``--joint-pose-samples`` records how many FK poses the invariant represents.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        action="append",
        type=Path,
        default=[],
        help="CuRobo robot YAML; repeat for left/right.",
    )
    parser.add_argument("--samples-per-link", type=int, default=10000)
    parser.add_argument("--joint-pose-samples", type=int, default=16)
    parser.add_argument("--max-uncovered-m", type=float, default=0.005)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output/piper_collision_sphere_audit.json",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _origin(node):
    origin = node.find("origin")
    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _load_link_meshes(urdf_path: Path):
    root = ET.parse(urdf_path).getroot()
    result = {}
    for link in root.findall("link"):
        meshes = []
        for collision in link.findall("collision"):
            mesh_node = collision.find("geometry/mesh")
            if mesh_node is None:
                continue
            mesh_path = (urdf_path.parent / mesh_node.get("filename")).resolve()
            if mesh_path.suffix.lower() == ".dae":
                with tempfile.TemporaryDirectory(prefix="piper_sphere_audit_") as directory:
                    converted = Path(directory) / "collision.stl"
                    subprocess.run(
                        ["assimp", "export", str(mesh_path), str(converted)],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    loaded = trimesh.load(converted, force="mesh", process=False)
            else:
                loaded = trimesh.load(mesh_path, force="mesh", process=False)
            loaded.apply_transform(_origin(collision))
            meshes.append(loaded)
        if meshes:
            result[link.get("name")] = trimesh.util.concatenate(meshes)
    return result


def _audit_config(path: Path, samples_per_link: int, limit: float, pose_samples: int):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    kinematics = document["robot_cfg"]["kinematics"]
    sphere_source = kinematics["collision_spheres"]
    if isinstance(sphere_source, str):
        sphere_path = (path.parent / sphere_source).resolve()
        sphere_document = yaml.safe_load(sphere_path.read_text(encoding="utf-8"))
        sphere_source = sphere_document["collision_spheres"]
    else:
        sphere_path = path
    content_assets = ROOT / "InternDataAssets/curobo/src/curobo/content/assets"
    urdf_path = (content_assets / kinematics["urdf_path"]).resolve()
    meshes = _load_link_meshes(urdf_path)
    link_results = {}
    failed = []
    rng = np.random.default_rng(0)
    for link_name, sphere_values in sphere_source.items():
        if link_name not in meshes:
            continue
        mesh = meshes[link_name]
        # trimesh uses the global RNG; seed through an explicit face/triangle
        # sample after deterministic area-weighted face selection.
        areas = mesh.area_faces
        face_indices = rng.choice(
            len(mesh.faces),
            size=samples_per_link,
            p=areas / areas.sum(),
        )
        triangles = mesh.triangles[face_indices]
        uv = rng.random((samples_per_link, 2))
        folded = uv.sum(axis=1) > 1.0
        uv[folded] = 1.0 - uv[folded]
        points = triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1:] * (
            triangles[:, 2] - triangles[:, 0]
        )
        centers = np.asarray([value["center"] for value in sphere_values], dtype=float)
        radii = np.asarray([value["radius"] for value in sphere_values], dtype=float)
        signed = np.min(np.linalg.norm(points[:, None] - centers[None], axis=-1) - radii[None], axis=1)
        positive = np.maximum(signed, 0.0)
        max_uncovered = float(np.max(positive))
        p95_uncovered = float(np.quantile(positive, 0.95))
        valid = max_uncovered <= limit
        if not valid:
            failed.append(link_name)
        link_results[link_name] = {
            "sphere_count": len(sphere_values),
            "surface_sample_count": samples_per_link,
            "max_uncovered_m": max_uncovered,
            "p95_uncovered_m": p95_uncovered,
            "within_limit": valid,
        }
    return {
        "robot_config": str(path),
        "sphere_source": str(sphere_path),
        "urdf": str(urdf_path),
        "joint_pose_samples": pose_samples,
        "distance_invariant": "link_local_distance_is_preserved_by_fk",
        "max_uncovered_limit_m": limit,
        "failed_links": failed,
        "passed": not failed,
        "links": link_results,
    }


def main():
    args = parse_args()
    configs = args.robot_config or [
        Path("workflows/simbox/curobo/src/curobo/content/configs/robot/piper100_left_arm.yml"),
        Path("workflows/simbox/curobo/src/curobo/content/configs/robot/piper100_right_arm.yml"),
    ]
    results = [
        _audit_config(_resolve(path), args.samples_per_link, args.max_uncovered_m, args.joint_pose_samples)
        for path in configs
    ]
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"passed": all(item["passed"] for item in results), "results": results}, indent=2),
        encoding="utf-8",
    )
    print(output)
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
