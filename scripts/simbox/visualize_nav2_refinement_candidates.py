#!/usr/bin/env python3
"""Render selected dynamic-approach refinements on the failed Nav2 map."""

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
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rotated_footprint(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.stack(
        [
            x + cosine * points[:, 0] - sine * points[:, 1],
            y + sine * points[:, 0] + cosine * points[:, 1],
        ],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize dynamic-approach refinement selections.")
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sample_dir = args.sample_dir.resolve()
    dynamic = load_json(sample_dir / "dynamic_goal_candidates.json")
    failure = load_json(sample_dir / "failure_snapshot.json")
    map_path = Path(failure["map_info"]["yaml_path"])
    if not map_path.is_absolute():
        map_path = Path.cwd() / map_path
    params_path = Path(failure["params_path"])
    if not params_path.is_absolute():
        params_path = Path.cwd() / params_path

    map_yaml = load_yaml(map_path)
    map_image = np.asarray(Image.open(map_path.parent / map_yaml["image"]))
    params = load_yaml(params_path)
    footprint = np.asarray(
        json.loads(params["global_costmap"]["global_costmap"]["ros__parameters"]["footprint"]),
        dtype=float,
    )
    height, width = map_image.shape
    resolution = float(map_yaml["resolution"])
    origin_x, origin_y = (float(value) for value in map_yaml["origin"][:2])
    extent = [origin_x, origin_x + width * resolution, origin_y, origin_y + height * resolution]

    selections = [entry["selected"] for entry in dynamic["refinement_history"]]
    selections.append(dynamic["selected"])
    colors = ["#1565C0", "#00897B", "#8E24AA"]
    target = dynamic["approach"]["target_pose"]
    final_xy = np.asarray(failure["world_xy"], dtype=float)
    final_yaw = float(failure["world_yaw"])
    effective_goal = dynamic["selected"]["effective_goal"]

    figure, (global_axis, zoom_axis) = plt.subplots(1, 2, figsize=(16, 7), dpi=180)
    for axis in (global_axis, zoom_axis):
        axis.imshow(np.where(np.flipud(map_image) == 0, 0.0, 1.0), cmap="gray", origin="lower", extent=extent)
        candidates = dynamic["candidates"]
        axis.scatter(
            [point["x"] for point in candidates if point["static_ok"]],
            [point["y"] for point in candidates if point["static_ok"]],
            color="#9E9E9E",
            marker=".",
            s=18,
            alpha=0.6,
            label="final refinement static-clear candidates",
        )
        axis.scatter(target["x"], target["y"], color="#D32F2F", marker="*", s=130, label="tray")
        for index, (selection, color) in enumerate(zip(selections, colors)):
            axis.scatter(selection["x"], selection["y"], color=color, marker="o", s=65, label=f"refinement {index} selected")
            axis.add_patch(
                Polygon(
                    rotated_footprint(footprint, selection["x"], selection["y"], selection["yaw"]),
                    closed=True,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.6,
                )
            )
        axis.scatter(effective_goal["x"], effective_goal["y"], color="#6A1B9A", marker="P", s=90, label="final Nav2 goal")
        axis.scatter(final_xy[0], final_xy[1], color="#EF6C00", marker="X", s=82, label="timeout pose")
        axis.add_patch(Polygon(rotated_footprint(footprint, *final_xy, final_yaw), closed=True, fill=False, edgecolor="#EF6C00", linewidth=2.0))
        axis.set_aspect("equal")
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")
        axis.grid(alpha=0.2)

    global_axis.set_title("Final Refinement Candidate Set")
    global_axis.legend(loc="upper left", fontsize=8)
    zoom_axis.set_title("Three Selected Refinement Goals")
    focus_x = [target["x"], final_xy[0], effective_goal["x"]] + [point["x"] for point in selections]
    focus_y = [target["y"], final_xy[1], effective_goal["y"]] + [point["y"] for point in selections]
    zoom_axis.set_xlim(min(focus_x) - 0.55, max(focus_x) + 0.55)
    zoom_axis.set_ylim(min(focus_y) - 0.55, max(focus_y) + 0.55)
    figure.suptitle(
        "nav_to_place_orange_0_id9009: refinement selections and timeout pose\n"
        f"final distance to Nav2 goal: {failure['world_dist']:.3f} m",
        fontsize=12,
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
