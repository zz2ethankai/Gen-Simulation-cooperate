#!/usr/bin/env python3
"""Export a static-map overlay for one Nav2 dynamic-goal attempt."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image
import yaml


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rotated_footprint(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return np.stack(
        [
            x + cos_yaw * points[:, 0] - sin_yaw * points[:, 1],
            y + sin_yaw * points[:, 0] + cos_yaw * points[:, 1],
        ],
        axis=1,
    )


def add_footprint(axis, footprint: np.ndarray, *, color: str, label: str) -> None:
    axis.add_patch(
        Polygon(footprint, closed=True, fill=False, edgecolor=color, linewidth=1.8, label=label)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Nav2 dynamic approach points on the exported map.")
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    candidates = load_json(sample_dir / "dynamic_goal_candidates.json")
    snapshot = load_json(sample_dir / "failure_snapshot.json")
    map_path = Path(snapshot["map_info"]["yaml_path"])
    if not map_path.is_absolute():
        map_path = Path.cwd() / map_path
    params_path = Path(snapshot["params_path"])
    if not params_path.is_absolute():
        params_path = Path.cwd() / params_path

    map_yaml = load_yaml(map_path)
    map_image = np.asarray(Image.open(map_path.parent / map_yaml["image"]))
    height, width = map_image.shape
    resolution = float(map_yaml["resolution"])
    origin_x, origin_y = (float(value) for value in map_yaml["origin"][:2])
    extent = [origin_x, origin_x + width * resolution, origin_y, origin_y + height * resolution]
    params = load_yaml(params_path)
    footprint = np.asarray(
        json.loads(params["global_costmap"]["global_costmap"]["ros__parameters"]["footprint"]), dtype=float
    )

    selected = candidates["selected"]
    target = candidates["approach"]["target_pose"]
    effective_goal = selected["effective_goal"]
    final_xy = np.asarray(snapshot["world_xy"], dtype=float)
    final_yaw = float(snapshot["world_yaw"])
    selected_xy = np.asarray([selected["x"], selected["y"]], dtype=float)
    selected_yaw = float(selected["yaw"])
    goal_xy = np.asarray([effective_goal["x"], effective_goal["y"]], dtype=float)

    figure, (global_axis, zoom_axis) = plt.subplots(1, 2, figsize=(16, 7), dpi=180)
    free_space = np.where(np.flipud(map_image) == 0, 0.0, 1.0)
    for axis in (global_axis, zoom_axis):
        axis.imshow(free_space, cmap="gray", origin="lower", extent=extent, interpolation="nearest")
        for candidate in candidates["candidates"]:
            color = "#5B5B5B" if candidate["static_ok"] else "#BDBDBD"
            marker = "." if candidate["path_ok"] else "x"
            axis.scatter(candidate["x"], candidate["y"], color=color, marker=marker, s=22, alpha=0.7)
        axis.scatter(target["x"], target["y"], c="#D32F2F", marker="o", s=58, label="orange")
        axis.scatter(selected_xy[0], selected_xy[1], c="#1565C0", marker="o", s=48, label="selected request")
        axis.scatter(goal_xy[0], goal_xy[1], c="#6A1B9A", marker="*", s=115, label="effective Nav2 goal")
        axis.scatter(final_xy[0], final_xy[1], c="#EF6C00", marker="X", s=68, label="failure pose")
        add_footprint(axis, rotated_footprint(footprint, *selected_xy, selected_yaw), color="#1565C0", label="selected footprint")
        add_footprint(axis, rotated_footprint(footprint, *final_xy, final_yaw), color="#EF6C00", label="failure footprint")
        axis.set_aspect("equal")
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")
        axis.grid(alpha=0.2)

    global_axis.set_title("Configured Dynamic Goal Points")
    global_axis.legend(loc="upper left", fontsize=8)
    zoom_axis.set_title("Selected Goal and Failure Pose")
    zoom_axis.set_xlim(min(selected_xy[0], final_xy[0], goal_xy[0]) - 1.1, max(selected_xy[0], final_xy[0], goal_xy[0]) + 1.1)
    zoom_axis.set_ylim(min(selected_xy[1], final_xy[1], goal_xy[1]) - 0.9, max(selected_xy[1], final_xy[1], goal_xy[1]) + 0.9)
    figure.suptitle(
        "PandaOmron: nav_to_pick_orange_0_id9009\n"
        f"failure XY distance to effective goal: {snapshot['world_dist']:.3f} m",
        fontsize=12,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
