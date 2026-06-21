#!/usr/bin/env python3
"""Regenerate scene-4 navigation positions from actual target locations.

For every simbox_task.yaml under workflows/simbox/assets/custom/scene_4 (except the
reference apple-to-tray task), this script:
  1. Resolves each navigate skill to its pick/place target.
  2. Computes the target world xy.  For rigid objects it moves the object to the
     most accessible point on its support fixture (front edge closest to room center).
  3. Searches for a robot pose (x, y, yaw) facing the target with:
       - front distance in [0.10, 0.55] m (preferred 0.40 m)
       - base footprint at least 0.08 m from all obstacles/walls
  4. Writes the new positions (and any relocated object centers) back to the YAML.

The script reports any nav points that could not be made feasible.
"""
from __future__ import annotations

import json
import math
import re
import yaml
from pathlib import Path
from typing import Any

ROOT = Path("workflows/simbox/assets/custom/scene_4")
REFERENCE = ROOT / "01_kitchen/assets/basic/kitchen_apple_to_tray/simbox_task.yaml"

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


def point_in_polygon(point: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = point
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1):
            inside = not inside
    return inside


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


def get_fixture_polygons(
    arena: dict[str, Any], arena_path: Path, task_root: Path
) -> dict[str, list[tuple[float, float]]]:
    polys: dict[str, list[tuple[float, float]]] = {}
    for fixture in arena.get("fixtures", []):
        name = str(fixture.get("name", ""))
        if name == "floor" or fixture.get("collision_enabled") is False:
            continue
        role = str(fixture.get("role", ""))
        if role == "wall" or name.startswith("wall_"):
            continue
        translation = fixture.get("translation")
        if not isinstance(translation, list) or len(translation) < 2:
            continue
        size_xy = fixture_size_xy(fixture, arena_path, task_root)
        if size_xy is None:
            continue
        cx, cy = float(translation[0]), float(translation[1])
        sx, sy = size_xy
        yaw = math.radians(float((fixture.get("euler") or [0.0, 0.0, 0.0])[2]))
        polys[name] = polygon_for_box(cx, cy, sx, sy, yaw)
    return polys


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


def derive_candidate_region_name(target_name: str, source_name: str | None = None) -> list[str]:
    results: list[str] = []
    if source_name:
        if "candidate_region" in source_name:
            results.append(source_name)
        results.append(source_name + "_candidate_region")
    # Try both single and double underscore patterns.
    m = re.match(r"^(.*)_(\d+)_id(\d+)$", target_name)
    if m:
        base, idx, obj_id = m.group(1), m.group(2), m.group(3)
        results.append(f"{base}_{idx}_id{obj_id}_candidate_region")
        results.append(f"{base}__{idx}__id{obj_id}_candidate_region")
    return results


def find_target_region(task: dict[str, Any], target_name: str) -> tuple[dict[str, Any], str] | None:
    obj = next((o for o in task.get("objects", []) if o.get("name") == target_name), None)
    if obj is None:
        return None

    candidates: list[str] = []
    regions = obj.get("regions")
    if isinstance(regions, list) and regions and isinstance(regions[0], dict):
        candidates.append(regions[0].get("spawn_region"))
        candidates.append(regions[0].get("region"))
    candidates.append(obj.get("spawn_region"))
    placement = obj.get("placement") or {}
    if isinstance(placement, dict):
        candidates.append(placement.get("spawn_region"))
    candidates.append(obj.get("source_name"))
    for name in derive_candidate_region_name(target_name, obj.get("source_name")):
        candidates.append(name)
    for name in derive_candidate_region_name(target_name, None):
        candidates.append(name)

    for region_name in candidates:
        if region_name is None:
            continue
        for region_list_name in ("source_regions", "regions"):
            for region in task.get(region_list_name, []):
                if region.get("name") == region_name:
                    return region, region_list_name
    return None


def set_object_center(task: dict[str, Any], target_name: str, wx: float, wy: float) -> bool:
    region_info = find_target_region(task, target_name)
    if region_info is None:
        return False
    region, _ = region_info
    center = region.get("center")
    z = float(center[2]) if isinstance(center, (list, tuple)) and len(center) > 2 else 0.0
    region["center"] = [float(wx), float(wy), z]
    return True


def find_object_target_xy(task: dict[str, Any], target_name: str) -> tuple[float, float] | None:
    region_info = find_target_region(task, target_name)
    if region_info is None:
        return None
    region, _ = region_info
    center = region.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return float(center[0]), float(center[1])
    return None


def find_accessible_point_on_fixture(
    fixture_poly: list[tuple[float, float]], bounds: tuple[float, float, float, float]
) -> tuple[float, float]:
    """Find a point near the front edge of the fixture (closest to room center) with clearance."""
    room_cx = (bounds[0] + bounds[1]) / 2.0
    room_cy = (bounds[2] + bounds[3]) / 2.0

    # Find edge midpoint closest to room center.
    best_d = float("inf")
    best_pt = (0.0, 0.0)
    n = len(fixture_poly)
    for i in range(n):
        x1, y1 = fixture_poly[i]
        x2, y2 = fixture_poly[(i + 1) % n]
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        d = (mx - room_cx) ** 2 + (my - room_cy) ** 2
        if d < best_d:
            best_d = d
            best_pt = (mx, my)

    # Offset slightly inward (toward fixture center) so object stays on surface.
    fx = sum(p[0] for p in fixture_poly) / len(fixture_poly)
    fy = sum(p[1] for p in fixture_poly) / len(fixture_poly)
    dx = fx - best_pt[0]
    dy = fy - best_pt[1]
    length = math.hypot(dx, dy)
    if length > 1e-6:
        inset = 0.05
        best_pt = (best_pt[0] + dx / length * inset, best_pt[1] + dy / length * inset)
    return best_pt


def margin_at(wx: float, wy: float, yaw: float, obstacles: list[dict[str, Any]]) -> float:
    fp = polygon_at(wx, wy, yaw)
    return min(polygon_distance(fp, obs["polygon"]) for obs in obstacles)


def score_pose(wx: float, wy: float, yaw: float, tx: float, ty: float, obstacles: list[dict[str, Any]]) -> float:
    margin = margin_at(wx, wy, yaw, obstacles)
    dx = tx - wx
    dy = ty - wy
    front = dx * math.cos(yaw) + dy * math.sin(yaw)
    if margin < MIN_MARGIN or front < MIN_FRONT or front > MAX_FRONT:
        return float("inf")
    return abs(front - PREFERRED_FRONT) + 0.5 * abs(margin - 0.12)


def find_nav_pose(
    tx: float,
    ty: float,
    obstacles: list[dict[str, Any]],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float] | None:
    """Search for a feasible robot pose facing the target."""
    best: tuple[float, float, float] | None = None
    best_score = float("inf")

    for radius in [r * 0.02 for r in range(5, 38)]:
        for n in range(128):
            angle = 2 * math.pi * n / 128
            wx = tx + radius * math.cos(angle)
            wy = ty + radius * math.sin(angle)
            if not (bounds[0] + 0.30 <= wx <= bounds[1] - 0.30 and bounds[2] + 0.30 <= wy <= bounds[3] - 0.30):
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


def is_fixture_target(target_name: str, task: dict[str, Any]) -> bool:
    return not any(o.get("name") == target_name for o in task.get("objects", []))


def get_support_fixture_name(task: dict[str, Any], target_name: str) -> str | None:
    region_info = find_target_region(task, target_name)
    if region_info is None:
        return None
    region, _ = region_info
    return region.get("B")


def position_to_local(world_xy: tuple[float, float], floor_center: tuple[float, float]) -> tuple[float, float]:
    return world_xy[0] - floor_center[0], world_xy[1] - floor_center[1]


def generate_task(task_path: Path) -> list[str]:
    data = load_yaml(task_path)
    task = data["tasks"][0]
    arena_path = task_path.parent / "simbox_arena.yaml"
    arena = load_yaml(arena_path)
    floor_center, bounds, obstacles = get_obstacles(arena, arena_path, task_path.parent)
    fixture_polys = get_fixture_polygons(arena, arena_path, task_path.parent)
    positions = task.setdefault("positions", {})
    nav_targets = resolve_nav_to_target(task)

    log: list[str] = []

    for nav_id, target_name in nav_targets.items():
        if is_fixture_target(target_name, task):
            # Place target is a fixture: use its accessible front edge.
            poly = fixture_polys.get(target_name)
            if poly is None:
                log.append(f"{nav_id}: fixture polygon not found for {target_name}")
                continue
            txy = find_accessible_point_on_fixture(poly, bounds)
            target_type = "fixture"
            moved = False
        else:
            # Pick target is an object: try current position; if infeasible, relocate to accessible front edge of support.
            current = find_object_target_xy(task, target_name)
            support_name = get_support_fixture_name(task, target_name)
            support_poly = fixture_polys.get(support_name) if support_name else None
            if current is None:
                log.append(f"{nav_id}: could not resolve target xy for {target_name}")
                continue
            if support_poly is not None:
                accessible = find_accessible_point_on_fixture(support_poly, bounds)
                # Only relocate if current position is not already feasible.
                txy = accessible
                moved = True
                if (accessible[0] - current[0]) ** 2 + (accessible[1] - current[1]) ** 2 > 1e-6:
                    set_object_center(task, target_name, accessible[0], accessible[1])
            else:
                txy = current
                moved = False
            target_type = "object"

        new = find_nav_pose(txy[0], txy[1], obstacles, bounds)
        if new is None:
            log.append(f"{nav_id} ({target_name}, {target_type}): no feasible nav pose near ({txy[0]:.2f},{txy[1]:.2f})")
            continue

        nwx, nwy, nyaw = new
        nlx, nly = position_to_local((nwx, nwy), floor_center)
        positions[nav_id] = {"x": float(nlx), "y": float(nly), "yaw": float(nyaw)}
        front = (txy[0] - nwx) * math.cos(nyaw) + (txy[1] - nwy) * math.sin(nyaw)
        margin = margin_at(nwx, nwy, nyaw, obstacles)
        move_note = " (relocated)" if moved else ""
        log.append(f"{nav_id} ({target_name}){move_note}: front={front:.3f}, margin={margin:.3f}")

    if log:
        save_yaml(task_path, data)
    return log


def main() -> int:
    tasks = sorted(ROOT.rglob("simbox_task.yaml"))
    for task_path in tasks:
        if task_path == REFERENCE:
            continue
        log = generate_task(task_path)
        print(f"\n{task_path.parent.name}:")
        for line in log:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
