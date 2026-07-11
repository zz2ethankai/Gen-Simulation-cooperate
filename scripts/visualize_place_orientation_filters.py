#!/usr/bin/env python3
"""Render the PandaOmron place-orientation filters as a 3D technical figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation


BASE_COLORS = ("#d1495b", "#2a9d6f", "#277da1")
EE_COLORS = ("#e76f51", "#43aa8b", "#4361ee")


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _basis_around(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = _unit(axis)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(axis, reference))
    v = np.cross(axis, u)
    return u, v


def _draw_arrow(ax, start, vector, color, label, *, length=1.0, linewidth=2.4):
    start = np.asarray(start, dtype=float)
    vector = np.asarray(vector, dtype=float) * length
    ax.quiver(
        start[0],
        start[1],
        start[2],
        vector[0],
        vector[1],
        vector[2],
        color=color,
        arrow_length_ratio=0.13,
        linewidth=linewidth,
    )
    end = start + vector
    ax.text(end[0], end[1], end[2], label, color=color, fontsize=9, fontweight="bold")


def _draw_frame(ax, origin, rotation, *, prefix, length, colors):
    for index, axis_name in enumerate(("x", "y", "z")):
        _draw_arrow(
            ax,
            origin,
            rotation[:, index],
            colors[index],
            f"{prefix} {axis_name}",
            length=length,
        )


def _draw_sphere(ax, *, radius=1.0, alpha=0.16):
    u = np.linspace(0.0, 2.0 * np.pi, 42)
    v = np.linspace(0.0, np.pi, 22)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color="#9aa0a6", alpha=alpha, linewidth=0.45)


def _draw_cone(ax, axis, half_angle_deg, color, *, length=1.0, alpha=0.11):
    axis = _unit(np.asarray(axis, dtype=float))
    u, v = _basis_around(axis)
    phi = np.linspace(0.0, 2.0 * np.pi, 80)
    distance = np.linspace(0.0, length, 20)
    phi_grid, distance_grid = np.meshgrid(phi, distance)
    half_angle = np.deg2rad(half_angle_deg)
    directions = (
        np.cos(half_angle) * axis[None, None, :]
        + np.sin(half_angle)
        * (
            np.cos(phi_grid)[..., None] * u[None, None, :]
            + np.sin(phi_grid)[..., None] * v[None, None, :]
        )
    )
    surface = distance_grid[..., None] * directions
    ax.plot_surface(
        surface[..., 0],
        surface[..., 1],
        surface[..., 2],
        color=color,
        alpha=alpha,
        linewidth=0,
        shade=False,
    )
    boundary = directions[-1]
    ax.plot(boundary[:, 0], boundary[:, 1], boundary[:, 2], color=color, linewidth=1.5, alpha=0.9)


def _configure_axis(ax, title, *, limit=1.08, elev=23, azim=-52, show_labels=True):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    if show_labels:
        ax.set_xlabel("base X / forward", fontsize=8, labelpad=2)
        ax.set_ylabel("base Y / left", fontsize=8, labelpad=2)
        ax.set_zlabel("base Z / up", fontsize=8, labelpad=2)
        ax.tick_params(labelsize=7, pad=0)
    else:
        ax.set_axis_off()


def _previous_mask(rotations: np.ndarray) -> np.ndarray:
    return (
        (rotations[:, 0, 0] <= np.cos(np.deg2rad(110.0)))
        & (rotations[:, 2, 1] <= np.cos(np.deg2rad(120.0)))
        & (rotations[:, 0, 2] >= np.cos(np.deg2rad(70.0)))
    )


def _current_mask(rotations: np.ndarray) -> np.ndarray:
    return (rotations[:, 0, 0] >= np.cos(np.deg2rad(45.0))) & (
        rotations[:, 2, 2] <= np.cos(np.deg2rad(150.0))
    )


def _angle_deg(dot_products: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.arccos(np.clip(dot_products, -1.0, 1.0)))


def _sample_rows(values: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    if len(values) <= count:
        return values
    return values[rng.choice(len(values), size=count, replace=False)]


def _draw_panel_coordinate_frames(ax):
    ax.set_title("A. Base frame and a natural top-down EE pose", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(-0.1, 1.0)
    ax.set_ylim(-0.55, 0.55)
    ax.set_zlim(0.0, 0.95)
    ax.set_box_aspect((1.1, 1.0, 0.95))
    ax.view_init(elev=24, azim=-55)
    ax.set_xlabel("base X / forward", fontsize=8)
    ax.set_ylabel("base Y / left", fontsize=8)
    ax.set_zlabel("base Z / up", fontsize=8)
    ax.tick_params(labelsize=7)

    _draw_frame(ax, np.zeros(3), np.eye(3), prefix="base", length=0.32, colors=BASE_COLORS)

    ee_origin = np.array([0.58, 0.0, 0.68])
    desired_rotation = np.diag([1.0, -1.0, -1.0])
    _draw_frame(ax, ee_origin, desired_rotation, prefix="EE", length=0.25, colors=EE_COLORS)

    tray_x = np.array([0.38, 0.78, 0.78, 0.38, 0.38])
    tray_y = np.array([-0.24, -0.24, 0.24, 0.24, -0.24])
    tray_z = np.full_like(tray_x, 0.05)
    ax.plot(tray_x, tray_y, tray_z, color="#5f6368", linewidth=2.0)
    ax.plot_trisurf(
        [0.38, 0.78, 0.78, 0.38],
        [-0.24, -0.24, 0.24, 0.24],
        [0.05] * 4,
        triangles=[[0, 1, 2], [0, 2, 3]],
        color="#d9d9d9",
        alpha=0.35,
        shade=False,
    )
    ax.text(0.58, -0.32, 0.03, "tray / placement plane", fontsize=8, color="#444444")

    finger_offset = desired_rotation[:, 1] * 0.055
    finger_direction = desired_rotation[:, 2] * 0.18
    for sign in (-1.0, 1.0):
        start = ee_origin + sign * finger_offset
        end = start + finger_direction
        ax.plot(*zip(start, end), color="#333333", linewidth=4.0, solid_capstyle="round")
    ax.plot(
        *zip(ee_origin - finger_offset, ee_origin + finger_offset),
        color="#333333",
        linewidth=5.0,
        solid_capstyle="round",
    )
    ax.text(0.65, 0.12, 0.48, "local z is the tool axis", fontsize=8, color=EE_COLORS[2])


def _draw_panel_previous(ax, rotations, rng):
    valid = rotations[_previous_mask(rotations)]
    shown = _sample_rows(valid, 900, rng)
    tool_z = shown[:, :, 2]
    angles = _angle_deg(-tool_z[:, 2])

    _configure_axis(ax, "B. Previous filters: tool z is mostly sideways")
    _draw_sphere(ax)
    _draw_cone(ax, np.array([1.0, 0.0, 0.0]), 70.0, "#f4a261", alpha=0.08)
    colors = cm.turbo(np.clip(angles / 180.0, 0.0, 1.0))
    ax.scatter(tool_z[:, 0], tool_z[:, 1], tool_z[:, 2], s=8, c=colors, alpha=0.55, depthshade=False)
    for vector in _sample_rows(tool_z, 28, rng):
        ax.plot([0.0, vector[0]], [0.0, vector[1]], [0.0, vector[2]], color="#e76f51", alpha=0.23, linewidth=0.8)
    _draw_arrow(ax, np.zeros(3), np.array([0.0, 0.0, -1.0]), "#333333", "down", length=0.92, linewidth=1.8)
    ax.text2D(
        0.02,
        0.02,
        "old: x backward 110 deg, y downward 120 deg, z forward 70 deg\n"
        f"tool-z/down angle: median {np.median(angles):.1f} deg",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
    )


def _draw_panel_current(ax, rotations, rng):
    valid = rotations[_current_mask(rotations)]
    shown = _sample_rows(valid, 900, rng)
    ee_x = shown[:, :, 0]
    tool_z = shown[:, :, 2]

    _configure_axis(ax, "C. Current filters: downward tool cone + forward wrist")
    _draw_sphere(ax)
    _draw_cone(ax, np.array([0.0, 0.0, -1.0]), 30.0, EE_COLORS[2], alpha=0.12)
    _draw_cone(ax, np.array([1.0, 0.0, 0.0]), 45.0, EE_COLORS[0], alpha=0.08)
    ax.scatter(tool_z[:, 0], tool_z[:, 1], tool_z[:, 2], s=9, color=EE_COLORS[2], alpha=0.5, depthshade=False)
    ax.scatter(ee_x[:, 0], ee_x[:, 1], ee_x[:, 2], s=9, color=EE_COLORS[0], alpha=0.42, depthshade=False)
    for vector in _sample_rows(tool_z, 22, rng):
        ax.plot([0.0, vector[0]], [0.0, vector[1]], [0.0, vector[2]], color=EE_COLORS[2], alpha=0.24, linewidth=0.8)
    _draw_arrow(ax, np.zeros(3), np.array([0.0, 0.0, -1.0]), "#222222", "down", length=0.92, linewidth=1.8)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=EE_COLORS[2], label="EE local z / tool axis"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=EE_COLORS[0], label="EE local x axis"),
        ],
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )
    ax.text2D(
        0.02,
        0.02,
        "current: x forward 45 deg, z downward 150 deg\n"
        "tool z stays within 30 deg of straight down",
        transform=ax.transAxes,
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#cccccc"},
    )


def _draw_angle_arc(ax, start_deg, end_deg, radius, color):
    theta = np.deg2rad(np.linspace(start_deg, end_deg, 100))
    x = radius * np.sin(theta)
    y = np.zeros_like(theta)
    z = radius * np.cos(theta)
    ax.plot(x, y, z, color=color, linewidth=2.4)


def _draw_panel_angle_encoding(ax):
    _configure_axis(ax, "D. Why downward 150 deg means a 30 deg down-cone", elev=14, azim=-68, show_labels=False)
    _draw_sphere(ax, alpha=0.1)
    up = np.array([0.0, 0.0, 1.0])
    down = np.array([0.0, 0.0, -1.0])
    boundary = np.array([0.5, 0.0, -np.sqrt(3.0) / 2.0])
    _draw_arrow(ax, np.zeros(3), up, BASE_COLORS[2], "+Z / up", length=0.98, linewidth=2.3)
    _draw_arrow(ax, np.zeros(3), down, "#333333", "-Z / down", length=0.98, linewidth=2.3)
    _draw_arrow(ax, np.zeros(3), boundary, EE_COLORS[2], "allowed boundary", length=1.0, linewidth=3.0)
    _draw_cone(ax, down, 30.0, EE_COLORS[2], alpha=0.12)
    _draw_angle_arc(ax, 0.0, 150.0, 0.62, "#8e44ad")
    _draw_angle_arc(ax, 150.0, 180.0, 0.78, "#1b998b")
    ax.text2D(
        0.60,
        0.59,
        "config angle = 150 deg",
        transform=ax.transAxes,
        color="#8e44ad",
        fontsize=10,
        fontweight="bold",
    )
    ax.text2D(
        0.50,
        0.19,
        "actual down tolerance = 30 deg",
        transform=ax.transAxes,
        color="#1b998b",
        fontsize=10,
        fontweight="bold",
    )
    ax.text2D(
        0.05,
        0.04,
        "negative directions are encoded against the positive axis:\n"
        "down tolerance = 180 deg - configured value",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
    )


def render(output: Path, *, samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rotations = Rotation.random(samples, random_state=rng).as_matrix()

    previous = rotations[_previous_mask(rotations)]
    current = rotations[_current_mask(rotations)]
    if len(previous) == 0 or len(current) == 0:
        raise RuntimeError("No valid orientations were sampled; increase --samples")

    figure = plt.figure(figsize=(16, 11), constrained_layout=True)
    figure.suptitle(
        "PandaOmron place orientation filters: rotation-matrix geometry",
        fontsize=17,
        fontweight="bold",
    )
    axes = [figure.add_subplot(2, 2, index, projection="3d") for index in range(1, 5)]
    _draw_panel_coordinate_frames(axes[0])
    _draw_panel_previous(axes[1], rotations, rng)
    _draw_panel_current(axes[2], rotations, rng)
    _draw_panel_angle_encoding(axes[3])

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/panda_omron_place_orientation_filters_3d.png"),
        help="Output PNG path",
    )
    parser.add_argument("--samples", type=int, default=200_000, help="Number of random rotations")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()
    if args.samples < 1_000:
        parser.error("--samples must be at least 1000")
    render(args.output, samples=args.samples, seed=args.seed)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
