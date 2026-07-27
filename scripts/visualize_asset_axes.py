#!/usr/bin/env python3
"""Render a headless PNG showing asset geometry and local coordinate axes.

The tool intentionally uses matplotlib instead of a GUI viewer so it can run on
headless servers.  It supports SimBox-style USD assets, full USD scenes, and OBJ
meshes.  For USD files, pass --prim to inspect a specific object transform.
"""

from __future__ import annotations

import argparse
import math
import os
import signal
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
signal.signal(signal.SIGPIPE, signal.SIG_DFL)


AXIS_COLORS = {
    "x": "#d62728",
    "y": "#2ca02c",
    "z": "#1f77b4",
}

VIEW_ANGLES = {
    "iso": (24, -58),
    "front": (0, -90),
    "side": (0, 0),
    "top": (90, -90),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create PNG previews with bbox and local x/y/z axes for USD or OBJ assets."
    )
    parser.add_argument("asset", type=Path, help="Path to .usd/.usda/.usdc or .obj")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path")
    parser.add_argument(
        "--prim",
        help=(
            "USD prim to inspect. If omitted, /World/Aligned is preferred, then "
            "the stage default prim, then the first root prim."
        ),
    )
    parser.add_argument(
        "--list-prims",
        action="store_true",
        help="List USD prim paths and exit. Useful before choosing --prim in a full scene.",
    )
    parser.add_argument(
        "--view",
        nargs="+",
        default=["iso"],
        choices=[*VIEW_ANGLES.keys(), "all"],
        help="Camera view(s) for PNG output.",
    )
    parser.add_argument(
        "--axis-origin",
        choices=["center", "prim"],
        default="center",
        help="Draw local axes at the geometry bbox center or the prim/object origin.",
    )
    parser.add_argument(
        "--axis-scale",
        type=float,
        default=0.35,
        help="Axis length as a fraction of the geometry bbox diagonal.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=60000,
        help="Maximum geometry vertices drawn as point cloud.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.35,
        help="Matplotlib point size for geometry vertices.",
    )
    return parser.parse_args()


def matrix_to_numpy(matrix) -> np.ndarray:
    return np.array([[matrix[i][j] for j in range(4)] for i in range(4)], dtype=float)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.reshape(0, 3)
    homogeneous = np.ones((len(points), 4), dtype=float)
    homogeneous[:, :3] = points
    transformed = homogeneous @ matrix
    return transformed[:, :3]


def bbox_edges(minimum: np.ndarray, maximum: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    corners = np.array(
        [
            [minimum[0], minimum[1], minimum[2]],
            [maximum[0], minimum[1], minimum[2]],
            [minimum[0], maximum[1], minimum[2]],
            [maximum[0], maximum[1], minimum[2]],
            [minimum[0], minimum[1], maximum[2]],
            [maximum[0], minimum[1], maximum[2]],
            [minimum[0], maximum[1], maximum[2]],
            [maximum[0], maximum[1], maximum[2]],
        ],
        dtype=float,
    )
    pairs = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    return [(corners[a], corners[b]) for a, b in pairs]


def choose_views(raw_views: Iterable[str]) -> list[str]:
    views: list[str] = []
    for view in raw_views:
        if view == "all":
            for candidate in VIEW_ANGLES:
                if candidate not in views:
                    views.append(candidate)
        elif view not in views:
            views.append(view)
    return views or ["iso"]


def default_output_path(asset: Path, view: str, multiple: bool) -> Path:
    suffix = f"__{view}" if multiple else ""
    return asset.with_name(f"{asset.stem}_axes{suffix}.png")


def downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    return points[::step][:max_points]


def list_usd_prims(asset: Path) -> None:
    from pxr import Usd

    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {asset}")
    for prim in stage.Traverse():
        type_name = prim.GetTypeName() or "-"
        default = " default" if stage.GetDefaultPrim() == prim else ""
        print(f"{prim.GetPath()}  {type_name}{default}")


def choose_usd_prim(stage, requested_path: str | None):
    if requested_path:
        prim = stage.GetPrimAtPath(requested_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"USD prim not found: {requested_path}")
        return prim

    aligned = stage.GetPrimAtPath("/World/Aligned")
    if aligned and aligned.IsValid():
        return aligned

    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return default_prim

    roots = list(stage.GetPseudoRoot().GetChildren())
    if not roots:
        raise ValueError("USD stage has no root prims")
    return roots[0]


def load_usd_geometry(asset: Path, prim_path: str | None):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {asset}")

    target_prim = choose_usd_prim(stage, prim_path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points_chunks = []
    mesh_count = 0

    for prim in Usd.PrimRange(target_prim):
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        local_points = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=float)
        world_matrix = matrix_to_numpy(cache.GetLocalToWorldTransform(prim))
        points_chunks.append(transform_points(local_points, world_matrix))
        mesh_count += 1

    if not points_chunks:
        raise RuntimeError(f"No Mesh geometry found under prim {target_prim.GetPath()}")

    points = np.concatenate(points_chunks, axis=0)
    target_matrix = matrix_to_numpy(cache.GetLocalToWorldTransform(target_prim))
    return points, target_matrix, str(target_prim.GetPath()), f"{mesh_count} mesh prim(s)"


def load_obj_geometry(asset: Path):
    import trimesh

    loaded = trimesh.load(str(asset), force="scene")
    geometries = []
    if hasattr(loaded, "geometry"):
        for name, geometry in loaded.geometry.items():
            transform = loaded.graph.get(name)[0]
            vertices = np.asarray(geometry.vertices, dtype=float)
            geometries.append(transform_points(vertices, np.asarray(transform, dtype=float).T))
    else:
        geometries.append(np.asarray(loaded.vertices, dtype=float))

    if not geometries:
        raise RuntimeError(f"No geometry found in OBJ: {asset}")
    points = np.concatenate(geometries, axis=0)
    return points, np.eye(4), "/", f"{len(geometries)} OBJ geometry part(s)"


def load_geometry(asset: Path, prim_path: str | None):
    suffix = asset.suffix.lower()
    if suffix in {".usd", ".usda", ".usdc"}:
        return load_usd_geometry(asset, prim_path)
    if suffix == ".obj":
        return load_obj_geometry(asset)
    raise ValueError(f"Unsupported asset format: {asset.suffix}")


def equalize_axes(ax, minimum: np.ndarray, maximum: np.ndarray) -> None:
    center = (minimum + maximum) * 0.5
    span = float(np.max(maximum - minimum))
    if not math.isfinite(span) or span <= 1e-9:
        span = 1.0
    radius = span * 0.58
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def draw_axes(ax, origin: np.ndarray, matrix: np.ndarray, length: float) -> None:
    axis_specs = {
        "x": np.array([length, 0.0, 0.0]),
        "y": np.array([0.0, length, 0.0]),
        "z": np.array([0.0, 0.0, length]),
    }
    base = transform_points(np.array([[0.0, 0.0, 0.0]]), matrix)[0]
    for name, local_endpoint in axis_specs.items():
        world_endpoint = transform_points(local_endpoint.reshape(1, 3), matrix)[0]
        direction = world_endpoint - base
        norm = np.linalg.norm(direction)
        if norm > 1e-12:
            direction = direction / norm * length
        endpoint = origin + direction
        ax.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            color=AXIS_COLORS[name],
            linewidth=2.2,
            arrow_length_ratio=0.12,
            normalize=False,
        )
        ax.text(endpoint[0], endpoint[1], endpoint[2], name.upper(), color=AXIS_COLORS[name], fontsize=11)


def render_png(
    asset: Path,
    output: Path,
    points: np.ndarray,
    target_matrix: np.ndarray,
    target_label: str,
    geometry_label: str,
    view: str,
    axis_origin: str,
    axis_scale: float,
    max_points: int,
    point_size: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]
    if len(points) == 0:
        raise RuntimeError("Geometry contains no finite points")

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    diagonal = float(np.linalg.norm(maximum - minimum))
    if not math.isfinite(diagonal) or diagonal <= 1e-9:
        diagonal = 1.0
    axis_length = max(diagonal * axis_scale, 1e-6)
    local_origin = transform_points(np.array([[0.0, 0.0, 0.0]]), target_matrix)[0]
    origin = center if axis_origin == "center" else local_origin

    display_points = downsample(points, max_points)
    fig = plt.figure(figsize=(8, 8), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        display_points[:, 0],
        display_points[:, 1],
        display_points[:, 2],
        s=point_size,
        c="#7a7f87",
        alpha=0.52,
        depthshade=False,
    )
    for start, end in bbox_edges(minimum, maximum):
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="#111111",
            linewidth=0.7,
            alpha=0.55,
        )

    draw_axes(ax, origin, target_matrix, axis_length)
    ax.scatter([local_origin[0]], [local_origin[1]], [local_origin[2]], c="#000000", s=16, depthshade=False)
    ax.text(local_origin[0], local_origin[1], local_origin[2], " prim origin", color="#000000", fontsize=8)

    elevation, azimuth = VIEW_ANGLES[view]
    ax.view_init(elev=elevation, azim=azimuth)
    equalize_axes(ax, minimum, maximum)
    ax.set_xlabel("world X")
    ax.set_ylabel("world Y")
    ax.set_zlabel("world Z")
    ax.set_title(
        f"{asset.name}\nprim/object: {target_label} | {geometry_label} | view: {view}",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    asset = args.asset.expanduser().resolve()
    if not asset.exists():
        raise FileNotFoundError(asset)

    if args.list_prims:
        list_usd_prims(asset)
        return

    points, target_matrix, target_label, geometry_label = load_geometry(asset, args.prim)
    views = choose_views(args.view)
    for view in views:
        output = args.output
        if output is None:
            output = default_output_path(asset, view, multiple=len(views) > 1)
        elif len(views) > 1:
            output = output.with_name(f"{output.stem}__{view}{output.suffix or '.png'}")
        render_png(
            asset=asset,
            output=output.resolve(),
            points=points,
            target_matrix=target_matrix,
            target_label=target_label,
            geometry_label=geometry_label,
            view=view,
            axis_origin=args.axis_origin,
            axis_scale=args.axis_scale,
            max_points=args.max_points,
            point_size=args.point_size,
        )
        print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
