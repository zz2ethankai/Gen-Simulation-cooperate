#!/usr/bin/env python3
"""Generate project-compatible sparse grasp annotations with GraspGen."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import trimesh.transformations as tra

from grasp_gen.exporters.interndata import (
    RobotProfile,
    export_interndata_grasps,
    load_source_gripper_geometry,
    resolve_model_config,
)
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.samplers import run_graspmoe


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
    sampler: GraspGenSampler,
    *,
    planner: str,
    num_grasps: int,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    if planner == "graspmoe":
        result = run_graspmoe(
            points,
            sampler,
            grasp_threshold=-1.0,
            num_grasps=num_grasps,
            topk_num_grasps=topk,
        )
        poses = np.concatenate([result["grasps_diff"], result["grasps_obb"]], axis=0)
        confidences = np.concatenate(
            [result["scores_diff"], result["scores_obb"]], axis=0
        )
    else:
        poses_t, confidences_t = GraspGenSampler.run_inference(
            points,
            sampler,
            grasp_threshold=-1.0,
            num_grasps=num_grasps,
            topk_num_grasps=topk,
            min_grasps=topk,
            remove_outliers=False,
        )
        poses = poses_t.detach().cpu().numpy()
        confidences = confidences_t.detach().cpu().numpy()
    return np.asarray(poses), np.asarray(confidences).reshape(-1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--robot-config", required=True, type=Path)
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--models-dir", type=Path)
    model.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--source-gripper",
        default="auto",
        help="Checkpoint embodiment when --models-dir is used (default: auto)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--num-grasps", type=int, default=2000)
    parser.add_argument("--num-sample-points", type=int, default=2000)
    parser.add_argument("--planner", choices=("diffusion", "graspmoe"), default="diffusion")
    parser.add_argument("--unit", choices=("m", "mm"), default="m")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mesh_path = args.mesh.expanduser().resolve()
    profile = RobotProfile.from_project_config(args.robot_config)
    if args.model_config is not None:
        model_config = args.model_config.expanduser().resolve()
    else:
        model_config = resolve_model_config(
            args.models_dir, profile, source_gripper=args.source_gripper
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    grasp_cfg = load_grasp_cfg(str(model_config))
    source_gripper = str(grasp_cfg.data.gripper_name)
    source_geometry = load_source_gripper_geometry(source_gripper)
    sampler = GraspGenSampler(grasp_cfg)

    scale = 0.001 if args.unit == "mm" else 1.0
    points, center_transform = _load_mesh_points(
        mesh_path,
        scale=scale,
        num_points=args.num_sample_points,
        seed=args.seed,
    )
    inference_started = time.perf_counter()
    poses, confidences = _infer(
        points,
        sampler,
        planner=args.planner,
        num_grasps=args.num_grasps,
        topk=max(args.count, 100),
    )
    inference_seconds = time.perf_counter() - inference_started
    if len(poses) == 0:
        raise RuntimeError("GraspGen returned no grasp candidates")

    inverse_center = tra.inverse_matrix(center_transform)
    poses = np.asarray([inverse_center @ pose for pose in poses])
    annotations = export_interndata_grasps(
        poses,
        confidences,
        profile,
        source_gripper_depth=source_geometry["depth"],
        count=args.count,
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
        "mesh": str(mesh_path),
        "mesh_unit": args.unit,
        "output": str(output_path),
        "shape": list(annotations.shape),
        "dtype": str(annotations.dtype),
        "planner": args.planner,
        "requested_count": args.count,
        "raw_candidate_count": int(len(poses)),
        "inference_seconds": inference_seconds,
        "confidence_min": float(np.min(confidences)),
        "confidence_max": float(np.max(confidences)),
        "seed": args.seed,
        "model_config": str(model_config),
        "source_gripper": source_gripper,
        "source_gripper_geometry": source_geometry,
        "robot_profile": profile.as_metadata(),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(output_path)
    print(metadata_path)
    print(f"shape={annotations.shape} dtype={annotations.dtype}")
    print(f"inference_seconds={inference_seconds:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
