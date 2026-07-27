#!/usr/bin/env python3
"""Generate project-compatible sparse grasp annotations with GraspGenX."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

# Project integration always resolves external assets explicitly.  Set this
# before any graspgenx import so importing the CLI cannot auto-clone models.
os.environ.setdefault("GRASPGENX_DISABLE_AUTO_SETUP", "1")

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra

from graspgenx.exporters.interndata import (
    R_GRASPGENX_FROM_GRASPNET,
    RobotProfile,
    export_interndata_grasps,
    resolve_gripper_name,
)
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers.planner import run_planner_on_object
from graspgenx.utils.checkpoint_io import load_model_cfg


def _load_mesh_points(
    mesh_path: Path, *, scale: float, num_points: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    loaded = trimesh.load(mesh_path, force="scene")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Mesh scene is empty: {mesh_path}")
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
        raise ValueError(f"Could not load a triangle mesh: {mesh_path}")
    if num_points <= 0:
        raise ValueError("num_sample_points must be positive")

    mesh = mesh.copy()
    mesh.apply_scale(scale)
    np.random.seed(seed)
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    points = np.asarray(points, dtype=np.float32)
    center_transform = tra.translation_matrix(-points.mean(axis=0))
    centered = tra.transform_points(points, center_transform).astype(np.float32)
    return centered, center_transform


def _infer(
    points: np.ndarray,
    sampler: GraspGenXSampler,
    *,
    planner: str,
    count: int,
    num_grasps: int,
    topk: int,
    moe_num_yaws: int,
    moe_z_offsets_cm: tuple[float, ...],
    moe_obb_density: str,
    moe_obb_position_spacing_cm: float,
) -> tuple[np.ndarray, np.ndarray, list[str], dict | None]:
    if planner == "diffusion":
        poses_t, confidences_t = GraspGenXSampler.run_inference(
            points,
            sampler,
            grasp_threshold=-1.0,
            num_grasps=num_grasps,
            topk_num_grasps=topk,
            min_grasps=count,
            remove_outliers=False,
        )
        if len(poses_t) == 0:
            return (
                np.zeros((0, 4, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
                None,
            )
        poses = poses_t.detach().cpu().numpy()
        confidences = confidences_t.detach().cpu().numpy()
        return poses, confidences, ["diff"] * len(poses), None

    poses, confidences, branch_tags, obb = run_planner_on_object(
        points,
        sampler,
        planner="graspmoe",
        grasp_threshold=-1.0,
        num_grasps=num_grasps,
        topk_num_grasps=topk,
        moe_num_yaws=moe_num_yaws,
        moe_z_offsets_cm=moe_z_offsets_cm,
        moe_obb_density=moe_obb_density,
        moe_obb_position_spacing_cm=moe_obb_position_spacing_cm,
    )
    return (
        np.asarray(poses),
        np.asarray(confidences).reshape(-1),
        list(branch_tags),
        obb,
    )


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated numbers, got {value!r}"
        ) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--robot-config", required=True, type=Path)
    parser.add_argument(
        "--models-dir",
        required=True,
        type=Path,
        help="GraspGenXModel checkout containing <version>/{gen,dis}",
    )
    parser.add_argument(
        "--gripper-descriptions-dir",
        required=True,
        type=Path,
        help="gripper_descriptions checkout root",
    )
    parser.add_argument(
        "--checkpoint-version",
        default="release",
        help="Subdirectory below --models-dir (default: release)",
    )
    parser.add_argument(
        "--gripper-name",
        default="auto",
        help="Expert override; default resolves from the project robot config",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--num-grasps", type=int, default=512)
    parser.add_argument("--num-sample-points", type=int, default=3500)
    parser.add_argument(
        "--planner", choices=("diffusion", "graspmoe"), default="graspmoe"
    )
    parser.add_argument(
        "--moe-num-yaws", type=int, default=36, help="GraspMoE OBB yaw samples"
    )
    parser.add_argument(
        "--moe-z-offsets-cm",
        type=_parse_float_tuple,
        default=(-2.0, 0.0),
        help="Comma-separated OBB Z offsets in centimetres",
    )
    parser.add_argument(
        "--moe-obb-density",
        choices=("sparse", "dense", "dense-topandside"),
        default="dense-topandside",
    )
    parser.add_argument("--moe-obb-position-spacing-cm", type=float, default=1.0)
    parser.add_argument("--unit", choices=("m", "mm"), default="m")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.num_grasps <= 0:
        raise ValueError("num_grasps must be positive")

    mesh_path = args.mesh.expanduser().resolve()
    models_dir = args.models_dir.expanduser().resolve()
    gripper_descriptions_dir = args.gripper_descriptions_dir.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh does not exist: {mesh_path}")
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory does not exist: {models_dir}")
    if not gripper_descriptions_dir.is_dir():
        raise FileNotFoundError(
            f"Gripper descriptions directory does not exist: {gripper_descriptions_dir}"
        )

    os.environ["GRASPGENX_GRIPPER_CFG_DIR"] = str(gripper_descriptions_dir)
    os.environ["GRASPGENX_CHECKPOINT_DIR"] = str(models_dir)

    profile = RobotProfile.from_project_config(args.robot_config)
    gripper_name = resolve_gripper_name(profile, args.gripper_name)
    descriptor_dir = (
        gripper_descriptions_dir
        / "gripper_descriptions"
        / "assets"
        / "x_grippers"
        / gripper_name
    )
    if not descriptor_dir.is_dir():
        raise FileNotFoundError(
            f"Gripper descriptor {gripper_name!r} does not exist: {descriptor_dir}"
        )

    checkpoint_root = models_dir / args.checkpoint_version
    cfg = load_model_cfg(
        str(checkpoint_root / "gen"), str(checkpoint_root / "dis")
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sampler = GraspGenXSampler(cfg, gripper_name=gripper_name)
    scale = 0.001 if args.unit == "mm" else 1.0
    points, center_transform = _load_mesh_points(
        mesh_path,
        scale=scale,
        num_points=args.num_sample_points,
        seed=args.seed,
    )

    topk = max(args.count, 100)
    inference_started = time.perf_counter()
    poses, confidences, branch_tags, obb = _infer(
        points,
        sampler,
        planner=args.planner,
        count=args.count,
        num_grasps=args.num_grasps,
        topk=topk,
        moe_num_yaws=args.moe_num_yaws,
        moe_z_offsets_cm=args.moe_z_offsets_cm,
        moe_obb_density=args.moe_obb_density,
        moe_obb_position_spacing_cm=args.moe_obb_position_spacing_cm,
    )
    inference_seconds = time.perf_counter() - inference_started
    if len(poses) == 0:
        raise RuntimeError("GraspGenX returned no grasp candidates")

    inverse_center = tra.inverse_matrix(center_transform)
    poses = np.asarray([inverse_center @ pose for pose in poses])
    annotations = export_interndata_grasps(
        poses,
        confidences,
        profile,
        tool_tcp_transform=sampler.gripper.tool_tcp_transform,
        count=args.count,
    )
    if annotations.shape != (args.count, 17) or annotations.dtype != np.float32:
        raise RuntimeError(
            f"Invalid exported annotation contract: {annotations.shape} {annotations.dtype}"
        )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else mesh_path.with_name("Aligned_grasp_sparse.npy")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, annotations)

    metadata = {
        "format": "interndata_sparse_grasp_nx17",
        "generator": "GraspGenX",
        "upstream_commit": "b9429097728cb1c430dd78b92edf17ba318aad03",
        "mesh": str(mesh_path),
        "mesh_unit": args.unit,
        "output": str(output_path),
        "shape": list(annotations.shape),
        "dtype": str(annotations.dtype),
        "planner": args.planner,
        "planner_parameters": {
            "num_grasps": args.num_grasps,
            "topk_num_grasps": topk,
            "moe_num_yaws": args.moe_num_yaws,
            "moe_z_offsets_cm": list(args.moe_z_offsets_cm),
            "moe_obb_density": args.moe_obb_density,
            "moe_obb_position_spacing_cm": args.moe_obb_position_spacing_cm,
        },
        "requested_count": args.count,
        "raw_candidate_count": int(len(poses)),
        "raw_branch_counts": dict(Counter(branch_tags)),
        "inference_seconds": inference_seconds,
        "confidence_min": float(np.min(confidences)),
        "confidence_max": float(np.max(confidences)),
        "confidence_mean": float(np.mean(confidences)),
        "legacy_score_min": float(np.min(annotations[:, 0])),
        "legacy_score_max": float(np.max(annotations[:, 0])),
        "seed": args.seed,
        "checkpoint_root": str(checkpoint_root),
        "generator_checkpoint": str(cfg.eval.gen_checkpoint),
        "discriminator_checkpoint": str(cfg.eval.dis_checkpoint),
        "gripper_name": gripper_name,
        "gripper_descriptor": str(descriptor_dir),
        "descriptor_bbox_width": float(sampler.gripper.width),
        "tool_tcp_transform": np.asarray(
            sampler.gripper.tool_tcp_transform, dtype=float
        ).tolist(),
        "coordinate_conversion": {
            "graspgenx_frame": "+X closing, +Z approach",
            "interndata_graspnet_frame": "x approach, y closing, z height",
            "R_graspgenx_from_graspnet": R_GRASPGENX_FROM_GRASPNET.tolist(),
            "score_semantics": "0.1 + 0.9 * (1 - confidence); lower is better",
            "tcp_center": "T_object_graspgenx @ T_graspgenx_tool_tcp",
        },
        "robot_profile": profile.as_metadata(),
        "obb": None
        if obb is None
        else {
            key: np.asarray(value).tolist()
            for key, value in obb.items()
            if key in {"center", "half_extent", "R"}
        },
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(output_path)
    print(metadata_path)
    print(f"robot={profile.name} gripper={gripper_name}")
    print(f"shape={annotations.shape} dtype={annotations.dtype}")
    print(f"raw_candidates={len(poses)} branches={dict(Counter(branch_tags))}")
    print(f"inference_seconds={inference_seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
