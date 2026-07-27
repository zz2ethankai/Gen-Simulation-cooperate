#!/usr/bin/env python3
"""Fit and audit conservative Piper collision spheres from URDF collision meshes.

The output is a standalone CuRobo sphere YAML.  It is intentionally generated
separately from the robot configuration so that the original inline sphere set
can remain in the robot YAML as a commented LEGACY block.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COUNTS = {
    "arm_base": 12,
    "link1": 10,
    "link2": 28,
    "link3": 24,
    "link4": 10,
    "link5": 12,
    "link6": 28,
    "link7": 10,
    "link8": 10,
}


def _load_audit_helpers():
    path = Path(__file__).with_name("audit_piper_collision_spheres.py")
    spec = importlib.util.spec_from_file_location("piper_sphere_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path(
            "workflows/simbox/curobo/src/curobo/content/configs/robot/"
            "piper100_right_arm.yml"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "workflows/simbox/curobo/src/curobo/content/configs/robot/spheres/"
            "piper100_collision_audited_20260720.yml"
        ),
    )
    # Fit to 4 mm so an independent 5 mm audit has 1 mm sampling reserve.
    parser.add_argument("--coverage-margin-m", type=float, default=0.004)
    parser.add_argument("--validation-samples", type=int, default=30000)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sample_surface(mesh, count: int, rng: np.random.Generator):
    areas = mesh.area_faces
    face_indices = rng.choice(len(mesh.faces), size=count, p=areas / areas.sum())
    triangles = mesh.triangles[face_indices]
    uv = rng.random((count, 2))
    folded = uv.sum(axis=1) > 1.0
    uv[folded] = 1.0 - uv[folded]
    return triangles[:, 0] + uv[:, :1] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1:] * (
        triangles[:, 2] - triangles[:, 0]
    )


def _inflate_to_coverage(points, centers, radii, margin: float):
    """Inflate fitted radii only as much as needed for the sampled 5 mm bound."""

    distances = np.linalg.norm(points[:, None] - centers[None], axis=-1)
    nearest = np.argmin(distances - radii[None], axis=1)
    signed = distances[np.arange(len(points)), nearest] - radii[nearest]
    for sphere_index in range(len(radii)):
        assigned = signed[nearest == sphere_index]
        if assigned.size:
            radii[sphere_index] += max(0.0, float(np.max(assigned)) - margin)
    return radii


def _fit_surface_clusters(points: np.ndarray, count: int, rng: np.random.Generator):
    """Fit fixed-count conservative spheres using farthest seeds and Lloyd means."""

    centers = [points[int(rng.integers(len(points)))]]
    nearest_sq = np.sum((points - centers[0]) ** 2, axis=1)
    for _ in range(1, count):
        index = int(np.argmax(nearest_sq))
        centers.append(points[index])
        nearest_sq = np.minimum(nearest_sq, np.sum((points - points[index]) ** 2, axis=1))
    centers = np.asarray(centers, dtype=float)
    for _ in range(20):
        distance_sq = np.sum((points[:, None] - centers[None]) ** 2, axis=-1)
        labels = np.argmin(distance_sq, axis=1)
        updated = centers.copy()
        for index in range(count):
            members = points[labels == index]
            if len(members):
                updated[index] = np.mean(members, axis=0)
        if np.max(np.linalg.norm(updated - centers, axis=1)) < 1e-6:
            break
        centers = updated
    return centers


def main():
    args = parse_args()
    helper = _load_audit_helpers()
    config_path = _resolve(args.robot_config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    kinematics = config["robot_cfg"]["kinematics"]
    asset_root = ROOT / "workflows/simbox/curobo/src/curobo/content/assets"
    meshes = helper._load_link_meshes((asset_root / kinematics["urdf_path"]).resolve())
    rng = np.random.default_rng(20260720)
    result = {}
    audit = {}
    for link_name, sphere_count in DEFAULT_COUNTS.items():
        mesh = meshes[link_name]
        points = _sample_surface(mesh, args.validation_samples, rng)
        centers = _fit_surface_clusters(points, sphere_count, rng)
        radii = np.zeros(len(centers), dtype=float)
        radii = _inflate_to_coverage(points, centers, radii, args.coverage_margin_m)
        signed = np.min(
            np.linalg.norm(points[:, None] - centers[None], axis=-1) - radii[None], axis=1
        )
        result[link_name] = [
            {
                "center": [round(float(value), 6) for value in center],
                "radius": round(float(radius), 6),
            }
            for center, radius in zip(centers, radii)
        ]
        audit[link_name] = {
            "sphere_count": len(radii),
            "max_uncovered_m": float(np.maximum(signed, 0.0).max()),
            "max_radius_m": float(radii.max()),
        }

    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# ROBOT-001 audited Piper collision spheres\n"
        f"# Generated from: {config_path}\n"
        "# Generated: 2026-07-20; validation samples are deterministic.\n"
        "# Original inline parameters remain in the robot YAML LEGACY block.\n"
    )
    output.write_text(
        header + yaml.safe_dump({"collision_spheres": result}, sort_keys=False),
        encoding="utf-8",
    )
    print(yaml.safe_dump({"output": str(output), "audit": audit}, sort_keys=False))
    return 0 if all(
        value["max_uncovered_m"] <= args.coverage_margin_m + 1e-9 for value in audit.values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
