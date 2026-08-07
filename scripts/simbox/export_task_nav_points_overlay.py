#!/usr/bin/env python3
"""Export task navigation points over a top-down obstacle map.

The script is intentionally Isaac-free so it can run in lightweight conda
environments. It builds a configuration-level 2D obstacle map from arena YAML
fixtures and their delivered metadata sizes, then overlays task navigation
points and the configured mobile-base footprint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
import yaml


DEFAULT_TASK = Path(
    "workflows/simbox/assets/custom/scene_4/01_kitchen/assets/basic/"
    "kitchen_apple_to_tray/simbox_task.yaml"
)
DEFAULT_ARENA = Path(
    "workflows/simbox/assets/custom/scene_4/01_kitchen/assets/basic/"
    "kitchen_apple_to_tray/simbox_arena.yaml"
)
DEFAULT_BASE = Path("workflows/simbox/core/configs/bases/ranger_mini_v3.yaml")
DEFAULT_OUTPUT_DIR = Path("output/nav_debug_exports")


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_base_trajectory(lmdb_path: Path | None) -> list[tuple[float, float, float]] | None:
    """Load base pose trajectory [x, y, z, yaw] from LMDB states.base.pose."""
    if lmdb_path is None or not lmdb_path.exists():
        return None
    tmpdir = tempfile.mkdtemp(prefix="nav_overlay_lmdb_")
    try:
        for fname in ("data.mdb", "info.json", "lock.mdb"):
            src = lmdb_path / fname
            if src.exists():
                shutil.copy(src, tmpdir)
        import lmdb

        env = lmdb.open(tmpdir, readonly=True, lock=False)
        with env.begin() as txn:
            raw = txn.get(b"states.base.pose")
            if raw is None:
                return None
            poses = pickle.loads(raw)
        env.close()
        return [(float(p[0]), float(p[1]), float(p[3])) for p in poses]
    except Exception as exc:
        print(f"Warning: could not load base trajectory from {lmdb_path}: {exc}")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def to_float_pair(values: list[Any]) -> tuple[float, float]:
    return float(values[0]), float(values[1])


def rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y, s * x + c * y


def polygon_for_box(
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    yaw: float,
) -> list[tuple[float, float]]:
    corners = [
        (-sx / 2.0, -sy / 2.0),
        (sx / 2.0, -sy / 2.0),
        (sx / 2.0, sy / 2.0),
        (-sx / 2.0, sy / 2.0),
    ]
    return [(cx + rx, cy + ry) for x, y in corners for rx, ry in [rotate_xy(x, y, yaw)]]


def metadata_size_xy(meta: dict[str, Any]) -> tuple[float, float] | None:
    layout_pose = meta.get("layout_pose") or {}
    if isinstance(layout_pose, dict) and isinstance(layout_pose.get("size_xyz_m"), list):
        size = layout_pose["size_xyz_m"]
        if len(size) >= 3:
            return float(size[0]), float(size[2])

    size_m = meta.get("size_m")
    if isinstance(size_m, dict) and "x" in size_m and "y" in size_m:
        return float(size_m["x"]), float(size_m["y"])

    geom = meta.get("geometry_alignment") or {}
    if isinstance(geom, dict) and isinstance(geom.get("layout_size_xyz_m"), list):
        size = geom["layout_size_xyz_m"]
        if len(size) >= 3:
            return float(size[0]), float(size[2])

    files = meta.get("files") or {}
    if isinstance(files, dict) and isinstance(files.get("mesh_extent_xyz"), list):
        size = files["mesh_extent_xyz"]
        if len(size) >= 3:
            return float(size[0]), float(size[2])

    return None


def fixture_size_xy(
    fixture: dict[str, Any],
    arena_path: Path,
    task_root: Path,
) -> tuple[float, float] | None:
    size = fixture.get("size")
    if isinstance(size, list) and len(size) >= 2:
        return float(size[0]), float(size[1])

    source_metadata = fixture.get("source_metadata")
    candidate_paths: list[Path] = []
    if source_metadata:
        raw = Path(str(source_metadata))
        candidate_paths.append(raw if raw.is_absolute() else arena_path.parent / raw)
        if "fixtures" in raw.parts:
            fixture_index = raw.parts.index("fixtures")
            candidate_paths.append(task_root / Path(*raw.parts[fixture_index:]))

    path = fixture.get("path")
    if path:
        raw = Path(str(path))
        asset_path = raw if raw.is_absolute() else arena_path.parent / raw
        candidate_paths.append(asset_path.parent / "metadata.json")
        if "fixtures" in raw.parts:
            fixture_index = raw.parts.index("fixtures")
            candidate_paths.append(task_root / Path(*raw.parts[fixture_index:]).parent / "metadata.json")

    for candidate in candidate_paths:
        if candidate.is_file():
            meta = load_json(candidate)
            size_xy = metadata_size_xy(meta)
            if size_xy is not None:
                scale = fixture.get("scale") or [1.0, 1.0, 1.0]
                sx = float(scale[0]) if len(scale) > 0 else 1.0
                sy = float(scale[1]) if len(scale) > 1 else 1.0
                return size_xy[0] * sx, size_xy[1] * sy

    return None


class Canvas:
    def __init__(self, bounds: tuple[float, float, float, float], ppm: int = 240):
        min_x, max_x, min_y, max_y = bounds
        self.bounds = bounds
        self.margin = 90
        self.ppm = ppm
        self.width = int(round((max_x - min_x) * ppm)) + 2 * self.margin
        self.height = int(round((max_y - min_y) * ppm)) + 2 * self.margin
        self.image = Image.new("RGB", (self.width, self.height), (248, 249, 250))
        self.draw = ImageDraw.Draw(self.image)
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 18)
            self.small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
            self.title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
        except OSError:
            self.font = ImageFont.load_default()
            self.small_font = ImageFont.load_default()
            self.title_font = ImageFont.load_default()

    def xy(self, x: float, y: float) -> tuple[int, int]:
        min_x, _max_x, min_y, max_y = self.bounds
        px = self.margin + int(round((x - min_x) * self.ppm))
        py = self.margin + int(round((max_y - y) * self.ppm))
        return px, py

    def line_world(self, points: list[tuple[float, float]], fill, width: int = 2):
        self.draw.line([self.xy(x, y) for x, y in points], fill=fill, width=width)

    def polygon_world(self, points: list[tuple[float, float]], fill, outline, width: int = 2):
        pixels = [self.xy(x, y) for x, y in points]
        self.draw.polygon(pixels, fill=fill, outline=outline)
        if width > 1:
            self.draw.line(pixels + [pixels[0]], fill=outline, width=width)

    def text_world(self, x: float, y: float, text: str, fill=(20, 20, 20), font=None):
        self.draw.text(self.xy(x, y), text, fill=fill, font=font or self.small_font)


def draw_grid(canvas: Canvas, title: str):
    min_x, max_x, min_y, max_y = canvas.bounds
    for ix in range(math.floor(min_x), math.ceil(max_x) + 1):
        color = (214, 219, 224) if ix % 1 == 0 else (232, 235, 238)
        canvas.line_world([(ix, min_y), (ix, max_y)], fill=color, width=1)
        canvas.text_world(ix + 0.02, min_y + 0.03, f"x={ix}", fill=(88, 96, 105))
    for iy in range(math.floor(min_y), math.ceil(max_y) + 1):
        color = (214, 219, 224) if iy % 1 == 0 else (232, 235, 238)
        canvas.line_world([(min_x, iy), (max_x, iy)], fill=color, width=1)
        canvas.text_world(min_x + 0.03, iy + 0.03, f"y={iy}", fill=(88, 96, 105))
    canvas.draw.text((20, 18), title, fill=(10, 33, 55), font=canvas.title_font)


def draw_arrow(canvas: Canvas, x: float, y: float, yaw: float, length: float, fill, width: int = 5):
    end = (x + length * math.cos(yaw), y + length * math.sin(yaw))
    canvas.line_world([(x, y), end], fill=fill, width=width)
    head_len = 0.13
    left = (
        end[0] - head_len * math.cos(yaw - 0.45),
        end[1] - head_len * math.sin(yaw - 0.45),
    )
    right = (
        end[0] - head_len * math.cos(yaw + 0.45),
        end[1] - head_len * math.sin(yaw + 0.45),
    )
    canvas.polygon_world([end, left, right], fill=fill, outline=fill, width=1)


def transform_task_position(
    pos: dict[str, Any],
    floor_center: tuple[float, float],
) -> dict[str, float]:
    local_x = float(pos["x"])
    local_y = float(pos["y"])
    yaw = float(pos.get("yaw", 0.0))
    return {
        "local_x": local_x,
        "local_y": local_y,
        "world_x": floor_center[0] + local_x,
        "world_y": floor_center[1] + local_y,
        "yaw": yaw,
    }


def resolve_positions(task: dict[str, Any]) -> dict[str, Any]:
    if isinstance(task.get("positions"), dict):
        return task["positions"]
    if isinstance(task.get("task"), dict) and isinstance(task["task"].get("positions"), dict):
        return task["task"]["positions"]
    tasks = task.get("tasks")
    if isinstance(tasks, list):
        for item in tasks:
            if isinstance(item, dict) and isinstance(item.get("positions"), dict):
                return item["positions"]
    raise KeyError("Could not find positions in task YAML")


def infer_task_name(task: Any, task_path: Path) -> str:
    if isinstance(task, list):
        for item in task:
            if isinstance(item, dict) and item.get("name"):
                return str(item["name"])
    if isinstance(task, dict):
        if task.get("name"):
            return str(task["name"])
        nested = task.get("task")
        if isinstance(nested, dict) and nested.get("name"):
            return str(nested["name"])
        tasks = task.get("tasks")
        if isinstance(tasks, list):
            for item in tasks:
                if isinstance(item, dict) and item.get("name"):
                    return str(item["name"])
    if task_path.stem == "simbox_task":
        return task_path.parent.name
    return task_path.stem


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--arena", type=Path, default=DEFAULT_ARENA)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pixels-per-meter", type=int, default=240)
    parser.add_argument("--pick-target", type=str, default="",
                        help="Pick target world layout xy, e.g. '3.4,0.74'")
    parser.add_argument("--place-target", type=str, default="",
                        help="Place target world layout xy, e.g. '1.43,0.33'")
    parser.add_argument("--reachability-radius", type=float, default=0.9,
                        help="Arm reachability radius in meters to draw around nav points")
    parser.add_argument("--lmdb-path", type=Path, default=None,
                        help="Path to episode LMDB directory; if given, overlay the actual base trajectory")
    args = parser.parse_args()

    task = load_yaml(args.task)
    arena = load_yaml(args.arena)
    base = load_yaml(args.base)
    trajectory = load_base_trajectory(args.lmdb_path)

    # Resolve room bounds from either coordinate_frame or floor source_metadata
    if "coordinate_frame" in arena and "room_bounds_xz" in arena["coordinate_frame"]:
        room_bounds = arena["coordinate_frame"]["room_bounds_xz"]
    else:
        floor = next(f for f in arena["fixtures"] if f["name"] == "floor")
        meta = floor.get("source_metadata", {})
        room_bounds = meta.get("layout_extent_xz", floor.get("size", [0, 1, 0, 1]))
    bounds = (
        float(room_bounds[0]),
        float(room_bounds[1]),
        float(room_bounds[2]),
        float(room_bounds[3]),
    )

    floor = next(f for f in arena["fixtures"] if f["name"] == "floor")
    floor_center = to_float_pair(floor["translation"])
    positions = resolve_positions(task)
    nav_points = {
        name: transform_task_position(pos, floor_center)
        for name, pos in positions.items()
        if name.startswith("nav_to_")
    }
    if not nav_points:
        nav_points = {
            name: transform_task_position(pos, floor_center)
            for name, pos in positions.items()
        }

    footprint_points = [
        (float(x), float(y))
        for x, y in base["platform"]["local_navigation"]["footprint_points"]
    ]

    obstacles: list[dict[str, Any]] = []
    for fixture in arena.get("fixtures", []):
        if fixture.get("name") == "floor":
            continue
        if fixture.get("collision_enabled") is False:
            continue
        role = str(fixture.get("role", ""))
        name = str(fixture.get("name", ""))
        translation = fixture.get("translation")
        if not isinstance(translation, list) or len(translation) < 2:
            continue

        if role == "wall" or name.startswith("wall_"):
            thickness = float(fixture.get("collision_thickness", 0.02) or 0.02)
            if name == "wall_north":
                cx, cy, sx, sy, yaw = 2.0, bounds[3] - thickness / 2.0, 4.0, thickness, 0.0
            elif name == "wall_south":
                cx, cy, sx, sy, yaw = 2.0, bounds[2] + thickness / 2.0, 4.0, thickness, 0.0
            elif name == "wall_west":
                cx, cy, sx, sy, yaw = bounds[0] + thickness / 2.0, 1.5, thickness, 3.0, 0.0
            elif name == "wall_east":
                cx, cy, sx, sy, yaw = bounds[1] - thickness / 2.0, 1.5, thickness, 3.0, 0.0
            else:
                continue
        else:
            size_xy = fixture_size_xy(fixture, args.arena, args.task.parent)
            if size_xy is None:
                continue
            cx = float(translation[0])
            cy = float(translation[1])
            sx, sy = size_xy
            yaw = math.radians(float((fixture.get("euler") or [0.0, 0.0, 0.0])[2]))

        obstacles.append(
            {
                "name": name,
                "center": [cx, cy],
                "size": [sx, sy],
                "yaw": yaw,
                "polygon": polygon_for_box(cx, cy, sx, sy, yaw),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_name = infer_task_name(task, args.task)
    stem = f"{task_name}_with_path" if trajectory else task_name
    overlay_path = args.output_dir / f"{stem}_nav_points_obstacle_overlay.png"
    obstacle_path = args.output_dir / f"{stem}_obstacle_map.png"
    json_path = args.output_dir / f"{stem}_nav_points_overlay.json"

    title = f"{task_name} nav points over config obstacle map"
    canvas = Canvas(bounds, ppm=args.pixels_per_meter)
    draw_grid(canvas, title)

    min_x, max_x, min_y, max_y = bounds
    canvas.polygon_world(
        [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)],
        fill=None,
        outline=(44, 62, 80),
        width=3,
    )

    for obs in obstacles:
        is_wall = obs["name"].startswith("wall_")
        fill = (92, 99, 112) if is_wall else (174, 188, 205)
        outline = (45, 52, 63) if is_wall else (71, 91, 115)
        canvas.polygon_world(obs["polygon"], fill=fill, outline=outline, width=2)
        if not is_wall:
            cx, cy = obs["center"]
            label = obs["name"].replace("_0_id", "#")
            canvas.text_world(cx - 0.28, cy, label, fill=(29, 45, 62))

    # Draw floor-center coordinate frame explicitly.
    fc_x, fc_y = floor_center
    canvas.line_world([(fc_x - 0.12, fc_y), (fc_x + 0.12, fc_y)], fill=(30, 30, 30), width=3)
    canvas.line_world([(fc_x, fc_y - 0.12), (fc_x, fc_y + 0.12)], fill=(30, 30, 30), width=3)
    draw_arrow(canvas, fc_x, fc_y, 0.0, 0.45, fill=(0, 112, 192), width=4)
    draw_arrow(canvas, fc_x, fc_y, math.pi / 2.0, 0.45, fill=(32, 145, 74), width=4)
    canvas.text_world(fc_x + 0.48, fc_y - 0.05, "+X floor/local", fill=(0, 86, 148))
    canvas.text_world(fc_x + 0.05, fc_y + 0.5, "+Y floor/local", fill=(18, 104, 53))
    canvas.text_world(fc_x + 0.04, fc_y - 0.18, "floor center / local origin", fill=(30, 30, 30))

    point_colors = {
        "nav_to_pick": (214, 73, 51),
        "nav_to_place": (107, 76, 181),
    }
    for name, point in nav_points.items():
        color = point_colors.get(name, (222, 146, 38))
        wx = point["world_x"]
        wy = point["world_y"]
        yaw = point["yaw"]
        fp_world = [
            (wx + rx, wy + ry)
            for px, py in footprint_points
            for rx, ry in [rotate_xy(px, py, yaw)]
        ]
        canvas.polygon_world(fp_world, fill=None, outline=color, width=4)
        px, py = canvas.xy(wx, wy)
        radius = 8
        canvas.draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline=(20, 20, 20), width=2)
        draw_arrow(canvas, wx, wy, yaw, 0.38, fill=color, width=5)
        text = (
            f"{name}\n"
            f"floor=({point['local_x']:.2f}, {point['local_y']:.2f})\n"
            f"layout=({wx:.2f}, {wy:.2f}) yaw={yaw:.2f}"
        )
        canvas.text_world(wx + 0.08, wy + 0.12, text, fill=color, font=canvas.font)

    # Draw actual base trajectory from LMDB
    if trajectory:
        path_points = [(x, y) for x, y, _yaw in trajectory]
        canvas.line_world(path_points, fill=(220, 53, 69), width=3)
        # start marker
        sx, sy = path_points[0]
        px, py = canvas.xy(sx, sy)
        canvas.draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(40, 167, 69), outline=(20, 20, 20), width=2)
        canvas.text_world(sx + 0.06, sy + 0.06, "start", fill=(40, 167, 69), font=canvas.font)
        # end marker
        ex, ey = path_points[-1]
        px, py = canvas.xy(ex, ey)
        canvas.draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(220, 53, 69), outline=(20, 20, 20), width=2)
        canvas.text_world(ex + 0.06, ey + 0.06, "end", fill=(220, 53, 69), font=canvas.font)
        # heading arrows every N frames
        for i in range(0, len(trajectory), max(1, len(trajectory) // 20)):
            x, y, yaw = trajectory[i]
            draw_arrow(canvas, x, y, yaw, 0.20, fill=(220, 53, 69), width=3)

    # Draw reachability circles around nav points
    reach_r = args.reachability_radius
    for name, point in nav_points.items():
        wx = point["world_x"]
        wy = point["world_y"]
        # Approximate circle as polygon
        circle_pts = []
        for i in range(64):
            angle = 2 * math.pi * i / 64
            circle_pts.append((wx + reach_r * math.cos(angle), wy + reach_r * math.sin(angle)))
        canvas.line_world(circle_pts + [circle_pts[0]], fill=(255, 200, 50), width=2)

    # Draw pick / place targets
    def parse_target(s: str) -> tuple[float, float] | None:
        if not s:
            return None
        parts = s.split(",")
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
        return None

    pick_xy = parse_target(args.pick_target)
    place_xy = parse_target(args.place_target)

    def draw_target_circle(canvas: Canvas, wx: float, wy: float, radius_m: float, fill, outline, width: int):
        x0, y0 = canvas.xy(wx - radius_m, wy - radius_m)
        x1, y1 = canvas.xy(wx + radius_m, wy + radius_m)
        # Ensure correct ordering for PIL (y0 <= y1 in pixel coords)
        if y0 > y1:
            y0, y1 = y1, y0
        canvas.draw.ellipse((x0, y0, x1, y1), fill=fill, outline=outline, width=width)

    if pick_xy:
        px, py = pick_xy
        draw_target_circle(canvas, px, py, 0.06, (255, 215, 0), (180, 140, 0), 3)
        canvas.text_world(px + 0.05, py + 0.05, "pick_target", fill=(180, 140, 0), font=canvas.font)
        if "nav_to_pick" in nav_points:
            np = nav_points["nav_to_pick"]
            dist = math.hypot(px - np["world_x"], py - np["world_y"])
            mid_x = (px + np["world_x"]) / 2
            mid_y = (py + np["world_y"]) / 2
            canvas.text_world(mid_x, mid_y, f"d={dist:.2f}m", fill=(180, 140, 0), font=canvas.small_font)
            canvas.line_world([(np["world_x"], np["world_y"]), (px, py)], fill=(255, 215, 0), width=2)

    if place_xy:
        px, py = place_xy
        draw_target_circle(canvas, px, py, 0.06, (50, 205, 50), (20, 120, 20), 3)
        canvas.text_world(px + 0.05, py + 0.05, "place_target", fill=(20, 120, 20), font=canvas.font)
        if "nav_to_place" in nav_points:
            np = nav_points["nav_to_place"]
            dist = math.hypot(px - np["world_x"], py - np["world_y"])
            mid_x = (px + np["world_x"]) / 2
            mid_y = (py + np["world_y"]) / 2
            canvas.text_world(mid_x, mid_y, f"d={dist:.2f}m", fill=(20, 120, 20), font=canvas.small_font)
            canvas.line_world([(np["world_x"], np["world_y"]), (px, py)], fill=(50, 205, 50), width=2)

    legend_x = canvas.width - 430
    legend_y = 20
    legend_lines = [
        "Coordinates:",
        "layout/world x-y: arena room frame",
        "floor/local: origin at floor center",
        "nav point layout = floor center + task position",
        "outlined polygons: configured base footprint at yaw",
        "yellow circle: pick_target; green circle: place_target",
        "yellow dashed circle: reachability radius around nav point",
    ]
    canvas.draw.rectangle(
        (legend_x - 12, legend_y - 8, canvas.width - 18, legend_y + 122),
        fill=(255, 255, 255),
        outline=(160, 170, 180),
        width=1,
    )
    for i, line in enumerate(legend_lines):
        canvas.draw.text((legend_x, legend_y + i * 23), line, fill=(29, 45, 62), font=canvas.small_font)

    canvas.image.save(overlay_path)

    obstacle_canvas = Canvas(bounds, ppm=args.pixels_per_meter)
    draw_grid(obstacle_canvas, f"{task_name} config obstacle map")
    obstacle_canvas.polygon_world(
        [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)],
        fill=None,
        outline=(44, 62, 80),
        width=3,
    )
    for obs in obstacles:
        obstacle_canvas.polygon_world(obs["polygon"], fill=(55, 65, 78), outline=(20, 25, 32), width=2)
    obstacle_canvas.image.save(obstacle_path)

    payload = {
        "task": str(args.task),
        "arena": str(args.arena),
        "base": str(args.base),
        "coordinate_frame": {
            "arena_reference": arena.get("coordinate_frame", {}).get("reference"),
            "room_bounds_xz": room_bounds,
            "floor_center_layout_xy": [floor_center[0], floor_center[1]],
            "task_positions_interpretation": "floor-center relative; layout/world xy = floor_center + task xy",
        },
        "nav_points": nav_points,
        "footprint_points_base_frame": footprint_points,
        "obstacles": [
            {
                "name": obs["name"],
                "center_layout_xy": obs["center"],
                "size_xy": obs["size"],
                "yaw_rad": obs["yaw"],
            }
            for obs in obstacles
        ],
        "outputs": {
            "overlay_png": str(overlay_path),
            "obstacle_png": str(obstacle_path),
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"overlay_png={overlay_path}")
    print(f"obstacle_png={obstacle_path}")
    print(f"json={json_path}")
    for name, point in nav_points.items():
        print(
            f"{name}: floor=({point['local_x']:.3f},{point['local_y']:.3f}) "
            f"layout=({point['world_x']:.3f},{point['world_y']:.3f}) "
            f"yaw={point['yaw']:.6f}"
        )
    print(f"obstacles={len(obstacles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
