#!/usr/bin/env python3
"""Unified InterDataEngine visualization entrypoint.

Two modes, auto-selected from the task file:

* ``layout`` — lightweight, physics-free matplotlib top-down schematic of the
  scene (floor, walls, support surface, object candidate regions, robot spawn
  and facing), drawn straight from a SimBox ``simbox_task.yaml`` + its
  ``arena_file``. Runs anywhere Python + matplotlib exist; no Isaac Sim needed.
  This is the default for converted SimBox tasks.

* ``physics`` — photoreal Isaac Sim renders, run in-process via
  ``agent/visual_physics.py``. It consumes
  either a SimBox ``simbox_task.yaml`` or an interdata ``task.yaml``. Only runs
  on a machine with Isaac Sim.

Usage::

    python agent/visual.py --task InternDataAssets/.../simbox_task.yaml \
        --out-dir runs/tabletop_cup_cube/visual

    python agent/visual.py --scene-dir runs/tabletop_cup_cube \
        --out-dir runs/tabletop_cup_cube/visual --mode layout
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
LAYOUT_PNG = "layout.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", default="")
    parser.add_argument("--task", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("auto", "layout", "physics"),
        default="auto",
        help="auto: layout for simbox_task.yaml, physics otherwise",
    )
    parser.add_argument("--dpi", type=int, default=110)
    # physics-mode arguments
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--rt-subframes", type=int, default=16)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--gravity-mps2", type=float, default=-9.81)
    parser.add_argument("--single-view", default="")
    parser.add_argument("--include-robot", action="store_true")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _resolve_ref(ref: str) -> Path:
    """Resolve a repo-relative ref (e.g. arena_file) against cwd, then repo root."""
    p = Path(ref)
    if p.is_absolute():
        return p
    cand = Path.cwd() / p
    if cand.exists():
        return cand
    return REPO_ROOT / p


def choose_task(scene_dir: Path, explicit_task: Path | None) -> Path:
    if explicit_task is not None:
        task = explicit_task
    else:
        candidates = list(scene_dir.glob("assets/basic/*/simbox_task.yaml")) + [
            scene_dir / "simbox" / "simbox_task.yaml",
            scene_dir / "simbox_task.yaml",
            scene_dir / "task.yaml",
            scene_dir / "interdata" / "task.yaml",
        ]
        task = next((path for path in candidates if path.is_file()), candidates[0])
    if not task.is_file():
        raise FileNotFoundError(
            f"No renderable task YAML found for {scene_dir}: {task}"
        )
    return task.resolve()


def is_simbox_task(task_path: Path) -> bool:
    try:
        doc = _load_yaml(task_path)
    except Exception:
        return False
    tasks = doc.get("tasks") or []
    if not tasks or not isinstance(tasks[0], dict):
        return False
    return bool(tasks[0].get("arena_file"))


# --------------------------------------------------------------------------- #
# layout mode: matplotlib top-down schematic of a SimBox task
# --------------------------------------------------------------------------- #


def _fixture_xy(fixture: dict) -> list[float]:
    tr = fixture.get("translation") or [0.0, 0.0, 0.0]
    return [float(tr[0]), float(tr[1])]


def _region_target_xy(
    region: dict,
    fixtures_by_name: dict[str, dict],
) -> list[float]:
    for field in ("B", "target"):
        target_name = region.get(field)
        if target_name in fixtures_by_name:
            return _fixture_xy(fixtures_by_name[target_name])
    region_name = region.get("name") or region.get("object") or "<unnamed>"
    raise ValueError(f"Region {region_name!r} has no resolvable B/target fixture")


def _find_support_fixture(fixtures: list[dict]) -> dict | None:
    for f in fixtures:
        if f.get("role") == "support_surface" or f.get("support_surface") is True:
            return f
    for f in fixtures:
        if f.get("asset_category") == "table":
            return f
    return None


def _wall_endpoints(fixture: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Horizontal (ground-plane) span of a vertical wall/glass panel.

    Arena walls are vertical planes whose length (``size[0]``) runs along world
    X when ``euler`` z is ~0/180 and along world Y when it is ~90/270.
    """
    size = fixture.get("size") or [0.0, 0.0, 0.0]
    eul = fixture.get("euler") or [0.0, 0.0, 0.0]
    cx, cy = _fixture_xy(fixture)
    length = float(size[0])
    ez = math.fmod(float(eul[2]), 180.0)
    along_x = (-45.0 <= ez <= 45.0) or (ez <= -135.0 or ez >= 135.0)
    half = length / 2.0
    if along_x:
        return (cx - half, cy), (cx + half, cy)
    return (cx, cy - half), (cx, cy + half)


def render_layout(task_path: Path, out_dir: Path, dpi: int) -> tuple[Path, dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    doc = _load_yaml(task_path)
    tasks = doc.get("tasks") or []
    if not tasks:
        raise ValueError(f"No `tasks:` block in {task_path}")
    t = tasks[0]
    arena_doc = (
        _load_yaml(_resolve_ref(t["arena_file"])) if t.get("arena_file") else {}
    )

    fixtures = arena_doc.get("fixtures") or []
    fixtures_by_name = {
        str(fixture["name"]): fixture
        for fixture in fixtures
        if isinstance(fixture, dict) and fixture.get("name")
    }
    robots = t.get("robots") or []
    regions = t.get("regions") or []
    source_regions = t.get("source_regions") or []

    table = _find_support_fixture(fixtures)
    table_center = _fixture_xy(table) if table else [0.0, 0.0]
    table_name = (table or {}).get("name")
    table_z = (table or {}).get("support_surface_z")
    robot_name = (robots[0].get("name") if robots else None) or "split_aloha"

    fig, ax = plt.subplots(figsize=(13, 10), dpi=dpi)
    ax.set_aspect("equal")
    ax.set_facecolor("#fafafa")

    # floor
    for f in fixtures:
        if f.get("name") != "floor":
            continue
        size = f.get("size") or [1.0, 1.0, 0.0]
        cx, cy = _fixture_xy(f)
        ax.add_patch(
            Rectangle(
                (cx - float(size[0]) / 2, cy - float(size[1]) / 2),
                float(size[0]),
                float(size[1]),
                facecolor="#ececec",
                edgecolor="none",
                zorder=0,
            )
        )

    # walls / glass / frames (vertical panels drawn as ground segments)
    for f in fixtures:
        if f.get("name") == "floor":
            continue
        role = str(f.get("role") or "")
        name = str(f.get("name") or "")
        wallish = role in ("wall", "window_glass", "window_frame") or name.startswith(
            ("wall_", "window_")
        )
        if not wallish:
            continue
        (x0, y0), (x1, y1) = _wall_endpoints(f)
        if role == "window_glass":
            color, lw = "#4a9ec4", 6
        elif role == "window_frame":
            color, lw = "#b9c0c7", 2
        else:
            color, lw = "#8a8f98", 6
        ax.plot(
            [x0, x1], [y0, y1], color=color, lw=lw,
            solid_capstyle="butt", zorder=1,
        )

    # support surface (e.g. central_work_table)
    if table is not None:
        ext = table.get("asset_world_extents") or table.get("size")
        cx, cy = table_center
        if ext and len(ext) >= 2:
            w, h = float(ext[0]), float(ext[1])
        else:
            size = table.get("size") or [1.0, 1.0]
            w, h = float(size[0]), float(size[1])
        ax.add_patch(
            Rectangle(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                facecolor="#d8c3a0",
                edgecolor="#7a5b2e",
                lw=1.5,
                zorder=2,
            )
        )
        ax.plot([cx], [cy], marker="x", color="#7a5b2e", ms=7, zorder=3)
        ax.text(
            cx, cy, table.get("name", "table"),
            ha="center", va="center", fontsize=9, color="#5c4322", zorder=3,
        )

    # object candidate regions + robot spawn
    robot_region: tuple[float, float, list[float]] | None = None
    robot_euler = 0.0
    if robots:
        eul = robots[0].get("euler") or [0.0, 0.0, 0.0]
        robot_euler = float(eul[2])

    region_info: list[dict] = []
    for r in regions:
        obj_name = str(r.get("object") or "")
        rc = r.get("random_config") or {}
        pr = rc.get("pos_range") or []
        if len(pr) != 2 or not pr[0] or not pr[1]:
            continue
        p0, p1 = pr[0], pr[1]
        is_robot = obj_name == robot_name
        ox, oy = _region_target_xy(r, fixtures_by_name)
        x0, y0 = ox + float(p0[0]), oy + float(p0[1])
        x1, y1 = ox + float(p1[0]), oy + float(p1[1])
        if is_robot:
            robot_region = (
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                list(rc.get("yaw_rotation") or [0.0, 0.0]),
            )
        else:
            region_info.append(
                {
                    "object": obj_name,
                    "world_xy": [round((x0 + x1) / 2, 4), round((y0 + y1) / 2, 4)],
                    "pos_range": pr,
                    "target": r.get("B"),
                }
            )
            w = abs(x1 - x0) or 0.02
            h = abs(y1 - y0) or 0.02
            ax.add_patch(
                Rectangle(
                    (min(x0, x1), min(y0, y1)),
                    w,
                    h,
                    facecolor="#9cc6e8",
                    alpha=0.35,
                    edgecolor="#3b6ea5",
                    lw=1.2,
                    zorder=2,
                )
            )
            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2, obj_name,
                ha="center", va="center", fontsize=8, color="#1d4d80", zorder=3,
            )

    if robot_region is not None:
        rx, ry, yaw_rng = robot_region
        # split_aloha yaw=0 faces +X (USD convention); final heading = euler z + region yaw
        yaw = (robot_euler + float(yaw_rng[0])) % 360.0
        rad = math.radians(yaw)
        dx, dy = math.cos(rad), math.sin(rad)
        base_r = 0.09
        ax.add_patch(
            Circle(
                (rx, ry), base_r,
                facecolor="#e27c3d", edgecolor="#7a3d12", lw=1.4, zorder=4,
            )
        )
        ax.annotate(
            "",
            xy=(rx + dx * base_r * 3.2, ry + dy * base_r * 3.2),
            xytext=(rx + dx * base_r * 1.1, ry + dy * base_r * 1.1),
            arrowprops=dict(arrowstyle="-|>", color="#7a3d12", lw=2.4),
            zorder=4,
        )
        ax.text(
            rx + dx * base_r * 3.6, ry + dy * base_r * 3.6,
            f"{robot_name}  heading={yaw:.0f} deg",
            ha="center", va="center", fontsize=9, color="#7a3d12", zorder=5,
        )
        robot_info = {
            "name": robot_name,
            "spawn_xy": [round(rx, 4), round(ry, 4)],
            "euler_z": robot_euler,
            "yaw_rotation": yaw_rng,
            "heading_deg": round(yaw, 1),
        }
    else:
        robot_info = None

    # source regions (robot initial region)
    for sr in source_regions:
        if sr.get("center_xyz"):
            c = sr["center_xyz"]
            cx, cy = float(c[0]), float(c[2])
        else:
            c = sr.get("center") or []
            if len(c) < 2:
                continue
            cx, cy = float(c[0]), float(c[1])
        ax.plot([cx], [cy], marker="o", color="#7a3d12", ms=5, zorder=4)
        ax.text(
            cx, cy + 0.02, str(sr.get("name", "")),
            ha="center", va="bottom", fontsize=8, color="#7a3d12", zorder=5,
        )

    # bounds
    xs: list[float] = []
    ys: list[float] = []
    for f in fixtures:
        tr = f.get("translation") or [0.0, 0.0, 0.0]
        size = f.get("size") or [1.0, 1.0, 0.0]
        xs += [float(tr[0]) - float(size[0]) / 2, float(tr[0]) + float(size[0]) / 2]
        ys += [float(tr[1]) - float(size[1]) / 2, float(tr[1]) + float(size[1]) / 2]
    if robot_region is not None:
        xs.append(robot_region[0])
        ys.append(robot_region[1])
    if xs:
        margin = 0.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.grid(True, linestyle=":", alpha=0.4, zorder=0)
    ax.set_title(f"{t.get('name')} — layout (world XY, top-down)", fontsize=12)

    info = [
        f"table: {table_name}  center XY: ({table_center[0]:.4f}, {table_center[1]:.4f})",
        f"support_surface_z: {table_z if table else '-'}",
        f"arena: {t.get('arena_file')}",
    ]
    ax.text(
        0.01, 0.99, "\n".join(info), transform=ax.transAxes,
        ha="left", va="top", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="#999999"),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / LAYOUT_PNG
    fig.savefig(out, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    manifest = {
        "mode": "layout",
        "task": str(task_path),
        "arena_file": t.get("arena_file"),
        "table": {
            "name": table_name,
            "center_xy": table_center,
            "support_surface_z": table_z,
        },
        "robot": robot_info,
        "regions": region_info,
        "output_png": LAYOUT_PNG,
    }
    return out, manifest


# --------------------------------------------------------------------------- #
# physics mode: in-process Isaac Sim rendering via agent/visual_physics.py
# --------------------------------------------------------------------------- #


def _load_visual_physics():
    """Import ``agent/visual_physics.py`` by file path, so it works whether this
    file is run as a script (``python agent/visual.py``) or as a module, and so
    the package ``__init__`` side effects are not triggered."""
    module_name = "agent.visual_physics"
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = Path(__file__).resolve().with_name("visual_physics.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_required_view_aliases(out_dir: Path) -> None:
    mapping = {
        "topdown.png": ("topdown",),
        "oblique.png": ("diagonal_overview",),
        "robot_head.png": ("doorway_interior", "south_interior"),
        "wrist_or_hand.png": (
            "room_interior",
            "window_overview",
            "east_interior",
        ),
    }
    for alias, view_dirs in mapping.items():
        src = next(
            (
                out_dir / view_dir / "rgb_0000.png"
                for view_dir in view_dirs
                if (out_dir / view_dir / "rgb_0000.png").is_file()
            ),
            None,
        )
        if src is not None:
            shutil.copy2(src, out_dir / alias)


def _run_physics(
    args: argparse.Namespace, scene_dir: Path, out_dir: Path, task: Path
) -> int:
    visual_physics = _load_visual_physics()
    exit_code = visual_physics.run_render(
        SimpleNamespace(
            task=task,
            output_dir=out_dir,
            width=args.width,
            height=args.height,
            rt_subframes=args.rt_subframes,
            settle_seconds=args.settle_seconds,
            gravity_mps2=args.gravity_mps2,
            include_robot=args.include_robot,
            single_view=args.single_view,
            renderer="RayTracedLighting",
        )
    )
    audit_path = out_dir / "physics_audit.json"
    physics_audit = {}
    if audit_path.is_file():
        physics_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    write_required_view_aliases(out_dir)
    images = sorted(
        path.relative_to(out_dir).as_posix() for path in out_dir.rglob("*.png")
    )
    manifest = {
        "method": args.method,
        "scene_dir": str(scene_dir),
        "task": str(task),
        "renderer": str(Path(visual_physics.__file__).resolve()),
        "physics_enabled": physics_audit.get("physics_enabled") is True,
        "gravity_mps2": physics_audit.get("gravity_mps2"),
        "settle_seconds": float(args.settle_seconds),
        "physics_audit": str(audit_path),
        "exit_code": exit_code,
        "image_count": len(images),
        "images": images,
    }
    (out_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if exit_code != 0:
        return exit_code
    if physics_audit.get("physics_enabled") is not True:
        print(
            f"Physics-enabled render audit missing or disabled: {audit_path}",
            file=sys.stderr,
        )
        return 1
    if not images:
        print(f"No PNG images produced in {out_dir}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    scene_dir = (args.scene_dir or Path.cwd()).resolve()
    out_dir = args.out_dir.resolve()
    task = choose_task(scene_dir, args.task).resolve()

    mode = args.mode
    if mode == "auto":
        mode = "layout" if is_simbox_task(task) else "physics"

    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "layout":
        try:
            out_png, manifest = render_layout(task, out_dir, args.dpi)
        except Exception as exc:
            print(f"Layout render failed: {exc}", file=sys.stderr)
            return 1
        (out_dir / "layout_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Layout saved to {out_png}")
        return 0

    return _run_physics(args, scene_dir, out_dir, task)


if __name__ == "__main__":
    raise SystemExit(main())
"""
agent/visual.py \
  --task InternDataAssets/Bench_2.1_isaacsim/scene_4/tabletop_cup_cube/assets/basic/tabletop_cup_cube/simbox_task.yaml \
  --out-dir runs/tabletop_cup_cube/visual \
  --mode physics \
  --include-robot
"""
