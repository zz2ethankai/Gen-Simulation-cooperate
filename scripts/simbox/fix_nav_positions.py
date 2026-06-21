#!/usr/bin/env python3
"""Fix scene-4 task nav positions to satisfy margin and front-distance constraints.

For each violating nav point the script first tries to nudge the robot away from
nearest obstacles while keeping the yaw pointed at the target. If that fails
(front distance leaves the [0.10, 0.55] m band), it moves the target object's
region a small amount toward a more open area and tries again.
"""
from __future__ import annotations

import json
import math
import yaml
from pathlib import Path
from typing import Any

ROOT = Path("workflows/simbox/assets/custom/scene_4")
REFERENCE = ROOT / "01_kitchen/assets/basic/kitchen_apple_to_tray/simbox_task.yaml"

# Ranger Mini V3 footprint in base frame (x forward, y left)
ROBOT_POLY = [
    (0.46, 0.24),
    (0.42, 0.29),
    (-0.32, 0.29),
    (-0.36, 0.24),
    (-0.36, -0.24),
    (-0.32, -0.29),
    (0.42, -0.29),
    (0.46, -0.24),
]

MIN_MARGIN = 0.08
MAX_FRONT = 0.55
MIN_FRONT = 0.10
PREFERRED_FRONT = 0.40


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, default_flow_style=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def polygon_at(nx: float, ny: float, yaw: float) -> list[tuple[float, float]]:
    c, s = math.cos(yaw), math.sin(yaw)
    return [(nx + x * c - y * s, ny + x * s + y * c) for (x, y) in ROBOT_POLY]


def polygons_intersect(poly_a: list[tuple[float, float]], poly_b: list[tuple[float, float]]) -> bool:
    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            edge = (x2 - x1, y2 - y1)
            length = math.hypot(*edge)
            if length < 1e-9:
                continue
            nx_ = -(y2 - y1) / length
            ny_ = (x2 - x1) / length
            dots_a = [p[0] * nx_ + p[1] * ny_ for p in poly_a]
            dots_b = [p[0] * nx_ + p[1] * ny_ for p in poly_b]
            if max(dots_a) < min(dots_b) or max(dots_b) < min(dots_a):
                return False
    return True


def polygon_distance(poly_a: list[tuple[float, float]], poly_b: list[tuple[float, float]]) -> float:
    if polygons_intersect(poly_a, poly_b):
        return 0.0
    min_dist = float("inf")
    for poly in (poly_a, poly_b):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            edge = (x2 - x1, y2 - y1)
            length = math.hypot(*edge)
            if length < 1e-9:
                continue
            nx_ = -(y2 - y1) / length
            ny_ = (x2 - x1) / length
            dots_a = [p[0] * nx_ + p[1] * ny_ for p in poly_a]
            dots_b = [p[0] * nx_ + p[1] * ny_ for p in poly_b]
            gap = max(min(dots_a) - max(dots_b), min(dots_b) - max(dots_a))
            if gap > 0:
                min_dist = min(min_dist, gap)
    return min_dist


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


def fixture_size_xy(fixture: dict[str, Any], arena_path: Path, task_root: Path) -> tuple[float, float] | None:
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


def polygon_for_box(cx: float, cy: float, sx: float, sy: float, yaw: float) -> list[tuple[float, float]]:
    corners = [(-sx / 2, -sy / 2), (sx / 2, -sy / 2), (sx / 2, sy / 2), (-sx / 2, sy / 2)]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + x * c - y * s, cy + x * s + y * c) for (x, y) in corners]


def get_obstacles(arena: dict[str, Any], arena_path: Path, task_root: Path):
    if "coordinate_frame" in arena and "room_bounds_xz" in arena["coordinate_frame"]:
        room_bounds = arena["coordinate_frame"]["room_bounds_xz"]
    else:
        floor = next(f for f in arena["fixtures"] if f["name"] == "floor")
        meta = floor.get("source_metadata", {})
        room_bounds = meta.get("layout_extent_xz", floor.get("size", [0, 1, 0, 1]))
    bounds = (float(room_bounds[0]), float(room_bounds[1]), float(room_bounds[2]), float(room_bounds[3]))

    floor = next(f for f in arena["fixtures"] if f["name"] == "floor")
    floor_center = (float(floor["translation"][0]), float(floor["translation"][1]))

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
            size_xy = fixture_size_xy(fixture, arena_path, task_root)
            if size_xy is None:
                continue
            cx = float(translation[0])
            cy = float(translation[1])
            sx, sy = size_xy
            yaw = math.radians(float((fixture.get("euler") or [0.0, 0.0, 0.0])[2]))

        obstacles.append({"name": name, "polygon": polygon_for_box(cx, cy, sx, sy, yaw)})
    return floor_center, bounds, obstacles


def find_target_region(task: dict[str, Any], target_name: str) -> tuple[dict[str, Any], str] | None:
    obj = next((o for o in task.get("objects", []) if o.get("name") == target_name), None)
    if obj is None:
        return None

    region_name = None
    regions = obj.get("regions")
    if isinstance(regions, list) and regions and isinstance(regions[0], dict):
        region_name = regions[0].get("spawn_region") or regions[0].get("region")
    if region_name is None:
        region_name = obj.get("spawn_region") or obj.get("placement", {}).get("spawn_region")
    if region_name is None:
        return None

    for region_list_name in ("source_regions", "regions"):
        for region in task.get(region_list_name, []):
            if region.get("name") == region_name:
                return region, region_list_name
    return None


def find_target_world_xy(task: dict[str, Any], target_name: str, _floor_center: tuple[float, float]) -> tuple[float, float] | None:
    region_info = find_target_region(task, target_name)
    if region_info is None:
        return None
    region, _ = region_info
    center = region.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return float(center[0]), float(center[1])
    return None


def set_target_region_pos(task: dict[str, Any], target_name: str, new_wx: float, new_wy: float) -> bool:
    region_info = find_target_region(task, target_name)
    if region_info is None:
        return False
    region, _ = region_info
    center = region.get("center")
    z = float(center[2]) if isinstance(center, (list, tuple)) and len(center) > 2 else 0.0
    region["center"] = [float(new_wx), float(new_wy), z]
    return True


def margin_at(wx: float, wy: float, yaw: float, obstacles: list[dict[str, Any]]) -> float:
    fp = polygon_at(wx, wy, yaw)
    return min(polygon_distance(fp, obs["polygon"]) for obs in obstacles)


def robot_pose_ok(wx: float, wy: float, yaw: float, tx: float, ty: float, obstacles: list[dict[str, Any]]) -> bool:
    if margin_at(wx, wy, yaw, obstacles) < MIN_MARGIN:
        return False
    dx = tx - wx
    dy = ty - wy
    front = dx * math.cos(yaw) + dy * math.sin(yaw)
    if not (MIN_FRONT <= front <= MAX_FRONT):
        return False
    return True


def score_pose(wx: float, wy: float, yaw: float, tx: float, ty: float, obstacles: list[dict[str, Any]]) -> float:
    margin = margin_at(wx, wy, yaw, obstacles)
    dx = tx - wx
    dy = ty - wy
    front = dx * math.cos(yaw) + dy * math.sin(yaw)
    if margin < MIN_MARGIN or front < MIN_FRONT or front > MAX_FRONT:
        return float("inf")
    return abs(front - PREFERRED_FRONT) + 0.5 * abs(margin - 0.12)


def nudge_robot(
    wx: float,
    wy: float,
    yaw: float,
    tx: float,
    ty: float,
    obstacles: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float] | None:
    """Gradient-like local search: move robot to improve margin while facing target."""
    best = (wx, wy, yaw)
    best_score = score_pose(wx, wy, yaw, tx, ty, obstacles)
    if math.isfinite(best_score):
        return best

    # Directions in world frame.
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (0.707, 0.707), (-0.707, 0.707), (0.707, -0.707), (-0.707, -0.707)]
    step = 0.02
    current = (wx, wy, yaw)
    for _ in range(60):
        improved = False
        for dx, dy in directions:
            nx = current[0] + dx * step
            ny = current[1] + dy * step
            if not (bounds[0] + 0.35 <= nx <= bounds[1] - 0.35 and bounds[2] + 0.35 <= ny <= bounds[3] - 0.35):
                continue
            nyaw = math.atan2(ty - ny, tx - nx)
            s = score_pose(nx, ny, nyaw, tx, ty, obstacles)
            if s < best_score:
                best_score = s
                best = (nx, ny, nyaw)
                improved = True
        if not improved:
            break
        current = best
    if math.isfinite(best_score):
        return best
    return None


def find_nav_pose(
    tx: float,
    ty: float,
    obstacles: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
    start: tuple[float, float, float] | None = None,
) -> tuple[float, float, float] | None:
    """Try local nudge from start, plus a sparse radial search."""
    if start is not None:
        result = nudge_robot(start[0], start[1], start[2], tx, ty, obstacles, bounds)
        if result is not None:
            return result

    best: tuple[float, float, float] | None = None
    best_score = float("inf")
    for radius in [r * 0.05 for r in range(2, 16)]:
        for n in range(32):
            angle = 2 * math.pi * n / 32
            wx = tx + radius * math.cos(angle)
            wy = ty + radius * math.sin(angle)
            if not (bounds[0] + 0.35 <= wx <= bounds[1] - 0.35 and bounds[2] + 0.35 <= wy <= bounds[3] - 0.35):
                continue
            yaw = math.atan2(ty - wy, tx - wx)
            s = score_pose(wx, wy, yaw, tx, ty, obstacles)
            if s < best_score:
                best_score = s
                best = (wx, wy, yaw)
    return best


def resolve_nav_to_target(task: dict[str, Any]) -> dict[str, str]:
    nav_targets: dict[str, str] = {}
    skills = task.get("skills", [])

    all_skills: list[dict[str, Any]] = []
    if isinstance(skills, list):
        for robot_entry in skills:
            if isinstance(robot_entry, dict):
                for arm_container in robot_entry.values():
                    if isinstance(arm_container, list):
                        for group_entry in arm_container:
                            if isinstance(group_entry, dict):
                                for group in group_entry.values():
                                    if isinstance(group, list):
                                        all_skills.extend(s for s in group if isinstance(s, dict))
    elif isinstance(skills, dict):
        all_skills = [s for s in skills.values() if isinstance(s, dict)]

    nav_by_id = {s["id"]: s for s in all_skills if s.get("name") == "navigate"}
    pick_place_by_dep: dict[str, dict[str, Any]] = {}
    for s in all_skills:
        if s.get("name") in ("pick", "place"):
            for dep in s.get("depends_on", []):
                pick_place_by_dep[dep] = s

    for nav_id, nav_skill in nav_by_id.items():
        target_skill = pick_place_by_dep.get(nav_id)
        if target_skill is None:
            continue
        if target_skill["name"] == "pick":
            target = target_skill.get("objects", [None])[0]
        else:
            target = target_skill.get("target_object") or target_skill.get("objects", [None, None])[1] or target_skill.get("objects", [None])[0]
        if target:
            nav_targets[nav_id] = target
    return nav_targets


def position_to_local(world_xy: tuple[float, float], floor_center: tuple[float, float]) -> tuple[float, float]:
    return world_xy[0] - floor_center[0], world_xy[1] - floor_center[1]


def fix_task(task_path: Path) -> list[str]:
    data = load_yaml(task_path)
    task = data["tasks"][0]
    arena_path = task_path.parent / "simbox_arena.yaml"
    arena = load_yaml(arena_path)
    floor_center, bounds, obstacles = get_obstacles(arena, arena_path, task_path.parent)
    positions = task.get("positions", {})
    nav_targets = resolve_nav_to_target(task)

    log: list[str] = []

    for nav_id, target_name in nav_targets.items():
        pos = positions.get(nav_id)
        if pos is None:
            continue
        txy = find_target_world_xy(task, target_name, floor_center)
        if txy is None:
            continue

        wx = floor_center[0] + float(pos["x"])
        wy = floor_center[1] + float(pos["y"])
        yaw = float(pos.get("yaw", 0.0))

        if robot_pose_ok(wx, wy, yaw, txy[0], txy[1], obstacles):
            continue

        new = find_nav_pose(txy[0], txy[1], obstacles, bounds, start=(wx, wy, yaw))
        if new is not None:
            nwx, nwy, nyaw = new
            nlx, nly = position_to_local((nwx, nwy), floor_center)
            positions[nav_id] = {"x": float(nlx), "y": float(nly), "yaw": float(nyaw)}
            old_margin = margin_at(wx, wy, yaw, obstacles)
            new_margin = margin_at(nwx, nwy, nyaw, obstacles)
            old_front = (txy[0] - wx) * math.cos(yaw) + (txy[1] - wy) * math.sin(yaw)
            new_front = (txy[0] - nwx) * math.cos(nyaw) + (txy[1] - nwy) * math.sin(nyaw)
            log.append(
                f"{nav_id} ({target_name}): robot moved "
                f"front {old_front:.3f}->{new_front:.3f}, margin {old_margin:.3f}->{new_margin:.3f}"
            )
            continue

        # Try moving target object toward room center.
        orig_tx, orig_ty = txy
        moved = False
        for step in range(1, 21):
            frac = step * 0.03
            room_cx = (bounds[0] + bounds[1]) / 2.0
            room_cy = (bounds[2] + bounds[3]) / 2.0
            new_tx = orig_tx + frac * (room_cx - orig_tx)
            new_ty = orig_ty + frac * (room_cy - orig_ty)
            set_target_region_pos(task, target_name, new_tx, new_ty)
            new = find_nav_pose(new_tx, new_ty, obstacles, bounds)
            if new is not None:
                nwx, nwy, nyaw = new
                nlx, nly = position_to_local((nwx, nwy), floor_center)
                positions[nav_id] = {"x": float(nlx), "y": float(nly), "yaw": float(nyaw)}
                old_margin = margin_at(wx, wy, yaw, obstacles)
                new_margin = margin_at(nwx, nwy, nyaw, obstacles)
                old_front = (orig_tx - wx) * math.cos(yaw) + (orig_ty - wy) * math.sin(yaw)
                new_front = (new_tx - nwx) * math.cos(nyaw) + (new_ty - nwy) * math.sin(nyaw)
                log.append(
                    f"{nav_id} ({target_name}): target moved {step*3}% toward room center "
                    f"({orig_tx:.2f},{orig_ty:.2f})->({new_tx:.2f},{new_ty:.2f}); "
                    f"front {old_front:.3f}->{new_front:.3f}, margin {old_margin:.3f}->{new_margin:.3f}"
                )
                moved = True
                break
        if not moved:
            set_target_region_pos(task, target_name, orig_tx, orig_ty)
            log.append(f"{nav_id} ({target_name}): could not find feasible robot pose or object relocation")

    if log:
        save_yaml(task_path, data)
    return log


def main() -> int:
    tasks = sorted(ROOT.rglob("simbox_task.yaml"))
    changed_any = False
    for task_path in tasks:
        if task_path == REFERENCE:
            continue
        log = fix_task(task_path)
        if log:
            changed_any = True
            print(f"\n{task_path.parent.name}:")
            for line in log:
                print(f"  {line}")
    if not changed_any:
        print("No nav positions needed adjustment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
