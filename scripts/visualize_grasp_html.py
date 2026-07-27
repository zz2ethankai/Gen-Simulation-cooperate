#!/usr/bin/env python3
"""Export sparse grasp annotations as an interactive Plotly HTML viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
import plotly.graph_objects as go


REPO_ROOT = Path(__file__).resolve().parents[1]
GRASP_TOOL_DIR = REPO_ROOT / "workflows" / "simbox" / "tools" / "grasp"
sys.path.insert(0, str(GRASP_TOOL_DIR))

from vis_grasp import R1, create_franka_gripper_o3d  # noqa: E402


def _load_object_points(obj_path: Path, *, unit: str, max_points: int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(str(obj_path))
    if mesh.is_empty():
        raise ValueError(f"Object mesh is empty: {obj_path}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.size == 0:
        raise ValueError(f"Object mesh has no vertices: {obj_path}")
    if unit == "mm":
        vertices = vertices / 1000.0
        mesh.vertices = o3d.utility.Vector3dVector(vertices)

    sample_count = min(max(int(max_points), 1), max(len(vertices) * 20, 1))
    try:
        pcd = mesh.sample_points_uniformly(number_of_points=sample_count)
        points = np.asarray(pcd.points, dtype=np.float64)
    except Exception:
        points = vertices

    if len(points) > max_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), size=max_points, replace=False)]
    return points


def _mesh_edges(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.size == 0 or triangles.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    edges = set()
    for tri in triangles:
        a, b, c = [int(v) for v in tri]
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))

    rows = []
    for a, b in sorted(edges):
        rows.append(vertices[a])
        rows.append(vertices[b])
        rows.append([np.nan, np.nan, np.nan])
    return np.asarray(rows, dtype=np.float64)


def _gripper_trace(gripper_mesh: o3d.geometry.TriangleMesh, *, name: str, color: str, hover: str) -> go.Scatter3d:
    edges = _mesh_edges(gripper_mesh)
    return go.Scatter3d(
        x=edges[:, 0],
        y=edges[:, 1],
        z=edges[:, 2],
        mode="lines",
        name=name,
        line={"color": color, "width": 3},
        hovertemplate=hover,
        showlegend=False,
    )


def _score_color(score: float, lo: float, hi: float) -> str:
    if hi <= lo:
        t = 0.5
    else:
        t = (float(score) - lo) / (hi - lo)
    # README says lower is better: green for lower scores, red for higher scores.
    r = int(40 + 210 * t)
    g = int(200 - 150 * t)
    b = int(70 + 40 * (1.0 - t))
    return f"rgb({r},{g},{b})"


def export_grasp_html(
    *,
    obj_path: Path,
    grasp_path: Path,
    output_path: Path,
    unit: str,
    count: int,
    max_points: int,
) -> None:
    obj_points = _load_object_points(obj_path, unit=unit, max_points=max_points)
    grasps = np.load(grasp_path, allow_pickle=True)
    if grasps.ndim != 2 or grasps.shape[1] < 16:
        raise ValueError(f"Expected grasp array with shape [N, >=16], got {grasps.shape}: {grasp_path}")

    n = min(max(int(count), 1), len(grasps))
    chosen = grasps[:n]
    scores = chosen[:, 0].astype(float)
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))

    traces: list[go.BaseTraceType] = [
        go.Scatter3d(
            x=obj_points[:, 0],
            y=obj_points[:, 1],
            z=obj_points[:, 2],
            mode="markers",
            name="object mesh samples",
            marker={"size": 2, "color": "rgba(255,128,0,0.55)"},
            hovertemplate="object point<br>x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f}<extra></extra>",
        )
    ]

    for idx, grasp in enumerate(chosen):
        score = float(grasp[0])
        width = float(grasp[1])
        height = float(grasp[2])
        depth = float(grasp[3])
        grasp_rot = np.asarray(grasp[4:13], dtype=np.float64).reshape(3, 3) @ R1.T
        center = np.asarray(grasp[13:16], dtype=np.float64)
        gripper = create_franka_gripper_o3d(center, grasp_rot, width, depth, score=score)
        color = _score_color(score, score_min, score_max)
        hover = (
            f"grasp #{idx}<br>"
            f"score={score:.4f}<br>"
            f"width={width:.4f} m<br>"
            f"height={height:.4f} m<br>"
            f"depth={depth:.4f} m<br>"
            f"center=({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})"
            "<extra></extra>"
        )
        traces.append(_gripper_trace(gripper, name=f"grasp {idx}", color=color, hover=hover))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{obj_path.parent.name}: sparse grasp annotations ({n}/{len(grasps)})",
        scene={
            "xaxis_title": "x (m)",
            "yaxis_title": "y (m)",
            "zaxis_title": "z (m)",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 42, "b": 0},
        template="plotly_white",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj-path", required=True, type=Path, help="Path to Aligned_obj.obj")
    parser.add_argument("--grasp-path", type=Path, help="Path to Aligned_grasp_sparse.npy")
    parser.add_argument("--output", required=True, type=Path, help="Output HTML path")
    parser.add_argument("--unit", choices=["mm", "m"], default="m", help="Mesh unit for OBJ coordinates")
    parser.add_argument("--count", type=int, default=300, help="Number of grasps to visualize")
    parser.add_argument("--max-points", type=int, default=15000, help="Object point samples to show")
    args = parser.parse_args()

    obj_path = args.obj_path.resolve()
    grasp_path = args.grasp_path.resolve() if args.grasp_path else Path(str(obj_path).replace("_obj.obj", "_grasp_sparse.npy"))
    export_grasp_html(
        obj_path=obj_path,
        grasp_path=grasp_path,
        output_path=args.output.resolve(),
        unit=args.unit,
        count=args.count,
        max_points=args.max_points,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
