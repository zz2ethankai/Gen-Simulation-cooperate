#!/usr/bin/env python3
"""Visualize a dynamic-approach failure with no reachable candidate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nav2.runtime.dynamic_goal import check_footprint_static_collision, load_static_map


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rotated_footprint(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return np.stack(
        [
            x + cos_yaw * points[:, 0] - sin_yaw * points[:, 1],
            y + sin_yaw * points[:, 0] + cos_yaw * points[:, 1],
        ],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a Nav2 dynamic approach candidate failure.")
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    dynamic = load_json(sample_dir / "dynamic_goal_candidates.json")
    snapshot = load_json(sample_dir / "workflow_incomplete_snapshot.json")
    params = yaml.safe_load((sample_dir / "debug_inputs/panda_omron_nav2_skill_params.yaml").read_text())
    map_yaml_path = sample_dir / "debug_inputs/map.yaml"
    map_yaml = yaml.safe_load(map_yaml_path.read_text())
    map_image = np.asarray(Image.open(map_yaml_path.parent / map_yaml["image"]))
    map_info = load_static_map(str(map_yaml_path))
    footprint = np.asarray(
        json.loads(params["global_costmap"]["global_costmap"]["ros__parameters"]["footprint"]), dtype=float
    )
    padding = float(dynamic["approach"]["footprint_padding_m"])
    start_x, start_y = (float(value) for value in snapshot["nav_xy"])
    start_yaw = float(snapshot["nav_yaw"])
    start_static = check_footprint_static_collision(
        static_map=map_info,
        footprint_points=footprint.tolist(),
        x=start_x,
        y=start_y,
        yaw=start_yaw,
        footprint_padding_m=padding,
    )
    target = dynamic["approach"]["target_pose"]

    height, width = map_image.shape
    resolution = float(map_yaml["resolution"])
    origin_x, origin_y = (float(value) for value in map_yaml["origin"][:2])
    extent = [origin_x, origin_x + width * resolution, origin_y, origin_y + height * resolution]
    map_plot = np.where(np.flipud(map_image) == 0, 0.0, 1.0)
    candidates = dynamic["candidates"]
    static_collision = [candidate for candidate in candidates if candidate.get("static_reason") == "static_footprint_collision"]
    out_of_bounds = [candidate for candidate in candidates if candidate.get("static_reason") == "footprint_out_of_bounds"]
    path_failed = [candidate for candidate in candidates if candidate.get("static_ok") and not candidate.get("path_ok")]

    figure, (global_axis, zoom_axis) = plt.subplots(1, 2, figsize=(16, 7), dpi=180)
    for axis in (global_axis, zoom_axis):
        axis.imshow(map_plot, cmap="gray", origin="lower", extent=extent, interpolation="nearest")
        axis.scatter(
            [candidate["x"] for candidate in static_collision],
            [candidate["y"] for candidate in static_collision],
            color="#9E9E9E",
            marker="x",
            s=18,
            alpha=0.65,
            label="static footprint collision",
        )
        axis.scatter(
            [candidate["x"] for candidate in out_of_bounds],
            [candidate["y"] for candidate in out_of_bounds],
            color="#D32F2F",
            marker="x",
            s=18,
            alpha=0.65,
            label="footprint out of bounds",
        )
        axis.scatter(
            [candidate["x"] for candidate in path_failed],
            [candidate["y"] for candidate in path_failed],
            color="#1565C0",
            marker="o",
            s=22,
            alpha=0.85,
            label="static clear, Nav2 plan failed",
        )
        axis.scatter(target["x"], target["y"], color="#C62828", marker="*", s=135, label="tray target")
        axis.scatter(start_x, start_y, color="#EF6C00", marker="X", s=70, label="Nav2 start")
        axis.add_patch(
            Polygon(
                rotated_footprint(footprint, start_x, start_y, start_yaw),
                closed=True,
                fill=False,
                edgecolor="#EF6C00",
                linewidth=2.0,
                label="start footprint",
            )
        )
        axis.set_aspect("equal")
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")
        axis.grid(alpha=0.2)

    global_axis.set_title("All Dynamic Approach Candidates")
    global_axis.legend(loc="upper left", fontsize=8)
    zoom_axis.set_title("Tray Candidates and Nav2 Start")
    all_x = [candidate["x"] for candidate in candidates] + [start_x, float(target["x"])]
    all_y = [candidate["y"] for candidate in candidates] + [start_y, float(target["y"])]
    zoom_axis.set_xlim(min(all_x) - 0.25, max(all_x) + 0.25)
    zoom_axis.set_ylim(min(all_y) - 0.25, max(all_y) + 0.25)

    figure.suptitle(
        "nav_to_place_orange: no reachable tray approach candidate\n"
        f"static candidates: {len(path_failed)}; Nav2 path successes: 0; "
        f"start static check: {'clear' if start_static['ok'] else start_static['reason']}",
        fontsize=12,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
