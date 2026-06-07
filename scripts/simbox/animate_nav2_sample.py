#!/usr/bin/env python3
"""Render a Nav2 sample trajectory overlay as an MP4.

The input sample directory is one of the exported
``output/ros_bridge/skills/split_aloha_nav2_goal_*`` folders.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import yaml


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_snapshot_path(sample_dir: Path) -> Path:
    for name in ("failure_snapshot.json", "success_snapshot.json", "shutdown_snapshot.json"):
        path = sample_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No snapshot found in {sample_dir}")


def _resolve_repo_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def _load_map(sample_dir: Path, snapshot: dict, repo_root: Path):
    map_info = snapshot.get("map_info", {}) or {}
    map_yaml_path = _resolve_repo_path(str(map_info["yaml_path"]), repo_root)
    map_yaml = _load_yaml(map_yaml_path)
    map_img = np.asarray(Image.open(map_yaml_path.parent / map_yaml["image"]))
    resolution = float(map_yaml["resolution"])
    origin_x = float(map_yaml["origin"][0])
    origin_y = float(map_yaml["origin"][1])
    return map_img, resolution, origin_x, origin_y


def _load_footprint(snapshot: dict, repo_root: Path) -> np.ndarray:
    params_path = _resolve_repo_path(str(snapshot["params_path"]), repo_root)
    params = _load_yaml(params_path)
    footprint_raw = params["global_costmap"]["global_costmap"]["ros__parameters"]["footprint"]
    return np.asarray(json.loads(footprint_raw), dtype=np.float32)


def _world_to_pixel(x: float, y: float, *, height: int, resolution: float, origin_x: float, origin_y: float):
    col = int(round((x - origin_x) / resolution))
    row = height - 1 - int(round((y - origin_y) / resolution))
    return col, row


def _polyline(points: list[tuple[float, float]], *, height: int, resolution: float, origin_x: float, origin_y: float):
    if not points:
        return np.empty((0, 1, 2), dtype=np.int32)
    pixels = [_world_to_pixel(x, y, height=height, resolution=resolution, origin_x=origin_x, origin_y=origin_y) for x, y in points]
    return np.asarray(pixels, dtype=np.int32).reshape((-1, 1, 2))


def _rotated_footprint(points: np.ndarray, x: float, y: float, yaw: float) -> list[tuple[float, float]]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    result = []
    for px, py in points:
        result.append((x + c * float(px) - s * float(py), y + s * float(px) + c * float(py)))
    return result


def _draw_marker(frame: np.ndarray, xy: tuple[float, float], color: tuple[int, int, int], *, height: int, resolution: float, origin_x: float, origin_y: float, label: str):
    col, row = _world_to_pixel(xy[0], xy[1], height=height, resolution=resolution, origin_x=origin_x, origin_y=origin_y)
    cv2.circle(frame, (col, row), 7, color, -1)
    cv2.putText(frame, label, (col + 8, row - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def render_video(sample_dir: Path, output_path: Path, fps: int, tail_frames: int):
    repo_root = Path.cwd()
    snapshot = _load_json(_resolve_snapshot_path(sample_dir))
    planned = _load_json(sample_dir / "planned_path.json")
    trajectory = _load_json(sample_dir / "actual_trajectory.json")

    map_img, resolution, origin_x, origin_y = _load_map(sample_dir, snapshot, repo_root)
    height, width = map_img.shape
    footprint = _load_footprint(snapshot, repo_root)

    # Occupied cells are black in PGM. Convert to a light BGR canvas.
    free = np.where(map_img == 0, 35, 235).astype(np.uint8)
    base = cv2.cvtColor(free, cv2.COLOR_GRAY2BGR)

    planned_xy = [(float(p["x"]), float(p["y"])) for p in planned["path"]["poses"]]
    actual_xy = [(float(p["x"]), float(p["y"])) for p in trajectory]
    goal_xy = (float(snapshot["goal"]["x"]), float(snapshot["goal"]["y"]))
    start_xy = actual_xy[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    planned_px = _polyline(planned_xy, height=height, resolution=resolution, origin_x=origin_x, origin_y=origin_y)
    total_frames = len(trajectory) + max(0, tail_frames)
    try:
        for frame_idx in range(total_frames):
            idx = min(frame_idx, len(trajectory) - 1)
            pose = trajectory[idx]
            frame = base.copy()
            if len(planned_px) >= 2:
                cv2.polylines(frame, [planned_px], isClosed=False, color=(200, 90, 20), thickness=2)
            actual_px = _polyline(actual_xy[: idx + 1], height=height, resolution=resolution, origin_x=origin_x, origin_y=origin_y)
            if len(actual_px) >= 2:
                cv2.polylines(frame, [actual_px], isClosed=False, color=(30, 30, 220), thickness=2)

            _draw_marker(frame, start_xy, (30, 150, 30), height=height, resolution=resolution, origin_x=origin_x, origin_y=origin_y, label="start")
            _draw_marker(frame, goal_xy, (180, 40, 180), height=height, resolution=resolution, origin_x=origin_x, origin_y=origin_y, label="goal")

            fp = _rotated_footprint(footprint, float(pose["x"]), float(pose["y"]), float(pose["yaw"]))
            fp_px = _polyline(fp, height=height, resolution=resolution, origin_x=origin_x, origin_y=origin_y)
            if len(fp_px) >= 3:
                cv2.polylines(frame, [fp_px], isClosed=True, color=(0, 140, 255), thickness=2)
            _draw_marker(frame, (float(pose["x"]), float(pose["y"])), (0, 140, 255), height=height, resolution=resolution, origin_x=origin_x, origin_y=origin_y, label="robot")

            dist = math.hypot(float(pose["x"]) - goal_xy[0], float(pose["y"]) - goal_xy[1])
            cv2.putText(frame, f"{sample_dir.name}  frame={idx + 1}/{len(trajectory)}  dist={dist:.2f}m", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(frame, "blue=planned red=actual orange=footprint", (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()


def main():
    parser = argparse.ArgumentParser(description="Render an MP4 overlay for one Nav2 sample.")
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--tail-frames", type=int, default=24)
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    if args.output is None:
        output = Path.cwd() / "output" / "analysis" / f"{sample_dir.name}_trajectory.mp4"
    else:
        output = args.output.resolve()
    render_video(sample_dir=sample_dir, output_path=output, fps=args.fps, tail_frames=args.tail_frames)
    print(output)


if __name__ == "__main__":
    main()
