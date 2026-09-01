#!/usr/bin/env python3
"""Place a floor-mobile SimBox robot relative to a named arena fixture."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


AXIS_ALIASES = {
    "-x": "-x",
    "left": "-x",
    "west": "-x",
    "左": "-x",
    "左边": "-x",
    "左侧": "-x",
    "+x": "+x",
    "x": "+x",
    "right": "+x",
    "east": "+x",
    "右": "+x",
    "右边": "+x",
    "右侧": "+x",
    "-y": "-y",
    "bottom": "-y",
    "down": "-y",
    "south": "-y",
    "下": "-y",
    "下面": "-y",
    "地图下方": "-y",
    "南侧": "-y",
    "+y": "+y",
    "y": "+y",
    "top": "+y",
    "up": "+y",
    "north": "+y",
    "上": "+y",
    "上面": "+y",
    "地图上方": "+y",
    "北侧": "+y",
}
AXIS_VECTOR = {
    "-x": (-1.0, 0.0),
    "+x": (1.0, 0.0),
    "-y": (0.0, -1.0),
    "+y": (0.0, 1.0),
}
OPPOSITE_AXIS = {"-x": "+x", "+x": "-x", "-y": "+y", "+y": "-y"}
X_FRONT_YAW_DEG = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": -90.0}


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task", required=True, type=Path, help="SimBox task YAML")
    cli.add_argument("--arena", type=Path, help="Override the task's arena YAML")
    cli.add_argument("--target", required=True, help="Exact or uniquely normalized fixture name")
    cli.add_argument("--relation", required=True, help="-x/+x/-y/+y or left/right/top/bottom")
    cli.add_argument(
        "--facing",
        default="toward_target",
        help="-x/+x/-y/+y, toward_target, or away_from_target",
    )
    cli.add_argument(
        "--robot-profile",
        type=Path,
        default=Path("config/robots/split_aloha_runtime.yaml"),
    )
    cli.add_argument("--reference-image", help="Optional project-relative evidence path")
    cli.add_argument("--execute", action="store_true", help="Atomically update the task YAML")
    return cli


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def normalize_name(value: str) -> str:
    return "_".join("".join(ch.lower() if ch.isalnum() else " " for ch in value).split())


def normalize_axis(value: str) -> str:
    key = value.strip().lower().replace(" ", "")
    if key not in AXIS_ALIASES:
        raise ValueError(f"Unsupported horizontal axis/relation: {value!r}")
    return AXIS_ALIASES[key]


def resolve_facing(value: str, relation: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {
        "toward",
        "towards",
        "toward_target",
        "face_target",
        "面对",
        "面对目标",
        "朝向目标",
    }:
        return OPPOSITE_AXIS[relation]
    if key in {"away", "away_from_target", "back_to_target", "背对", "背对目标", "远离目标"}:
        return relation
    return normalize_axis(value)


def vector(values: Any, count: int, field: str) -> list[float]:
    if not isinstance(values, list) or len(values) < count:
        raise ValueError(f"Expected {field} to contain at least {count} values")
    result = [float(values[index]) for index in range(count)]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"Non-finite value in {field}")
    return result


def fixture_xy_bounds(fixture: dict[str, Any]) -> tuple[float, float, float, float]:
    translation = vector(fixture.get("translation", [0.0, 0.0, 0.0]), 2, "fixture.translation")
    raw_extents = fixture.get("asset_world_extents") or fixture.get("size")
    extents = vector(raw_extents, 2, f"{fixture.get('name')}.asset_world_extents/size")
    if extents[0] <= 0.0 or extents[1] <= 0.0:
        raise ValueError(f"Non-positive fixture extents: {fixture.get('name')}")
    half_x, half_y = 0.5 * extents[0], 0.5 * extents[1]
    return (
        translation[0] - half_x,
        translation[0] + half_x,
        translation[1] - half_y,
        translation[1] + half_y,
    )


def resolve_fixture(fixtures: list[Any], requested: str) -> dict[str, Any]:
    target_key = normalize_name(requested)
    matches = [
        fixture
        for fixture in fixtures
        if isinstance(fixture, dict) and normalize_name(str(fixture.get("name", ""))) == target_key
    ]
    if len(matches) != 1:
        names = [str(item.get("name")) for item in fixtures if isinstance(item, dict)]
        raise ValueError(f"Target fixture {requested!r} resolved to {len(matches)} matches; available={names}")
    return matches[0]


def rotated_robot_bounds(
    footprint: list[Any], x: float, y: float, yaw_deg: float
) -> tuple[float, float, float, float]:
    theta = math.radians(yaw_deg)
    cosine, sine = math.cos(theta), math.sin(theta)
    points: list[tuple[float, float]] = []
    for index, raw_point in enumerate(footprint):
        point = vector(raw_point, 2, f"base.footprint_xz_m[{index}]")
        points.append(
            (
                x + cosine * point[0] - sine * point[1],
                y + sine * point[0] + cosine * point[1],
            )
        )
    if len(points) < 3:
        raise ValueError("Robot footprint must contain at least three points")
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def overlaps(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[1] and a[1] > b[0] and a[2] < b[3] and a[3] > b[2]


def atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, sort_keys=False, allow_unicode=True, width=80)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parser().parse_args()
    project_root = Path(__file__).resolve().parents[4]
    task_path = args.task.resolve()
    profile_path = args.robot_profile
    if not profile_path.is_absolute():
        profile_path = (project_root / profile_path).resolve()

    document = load_yaml(task_path)
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1 or not isinstance(tasks[0], dict):
        raise ValueError("Expected exactly one task under tasks[]")
    task = tasks[0]

    arena_path = args.arena
    if arena_path is None:
        arena_ref = task.get("arena_file")
        if not isinstance(arena_ref, str) or not arena_ref:
            raise ValueError("Task has no arena_file")
        local_candidate = task_path.parent / arena_ref
        arena_path = local_candidate if local_candidate.is_file() else project_root / arena_ref
    elif not arena_path.is_absolute():
        arena_path = project_root / arena_path
    arena_path = arena_path.resolve()

    arena = load_yaml(arena_path)
    coordinate_frame = arena.get("coordinate_frame")
    if not isinstance(coordinate_frame, dict):
        raise ValueError("Arena has no coordinate_frame")
    if str(coordinate_frame.get("up_axis", "")).upper().lstrip("+") != "Z":
        raise ValueError("This workflow requires +Z up")
    if str(coordinate_frame.get("floor_plane", "")).upper() != "XY":
        raise ValueError("This workflow requires an XY floor plane")

    profile = load_yaml(profile_path)
    if profile.get("coordinate_frame") != "isaac_z_up_x_front":
        raise ValueError("Only isaac_z_up_x_front robot profiles are currently supported")
    robot_name = str(profile.get("name") or "")
    base = profile.get("base")
    manipulator = profile.get("manipulator")
    if not robot_name or not isinstance(base, dict) or not isinstance(manipulator, dict):
        raise ValueError("Robot profile is missing name, base, or manipulator data")

    fixtures = arena.get("fixtures")
    if not isinstance(fixtures, list):
        raise ValueError("Arena fixtures must be a list")
    target = resolve_fixture(fixtures, args.target)
    target_name = str(target["name"])
    target_bounds = fixture_xy_bounds(target)
    target_center_x = 0.5 * (target_bounds[0] + target_bounds[1])
    target_center_y = 0.5 * (target_bounds[2] + target_bounds[3])

    relation = normalize_axis(args.relation)
    facing = resolve_facing(args.facing, relation)
    yaw_deg = X_FRONT_YAW_DEG[facing]
    approach_offset = float(base.get("approach_offset_m"))
    if not math.isfinite(approach_offset) or approach_offset <= 0.0:
        raise ValueError("base.approach_offset_m must be positive")

    x, y = target_center_x, target_center_y
    if relation == "-x":
        x = target_bounds[0] - approach_offset
    elif relation == "+x":
        x = target_bounds[1] + approach_offset
    elif relation == "-y":
        y = target_bounds[2] - approach_offset
    else:
        y = target_bounds[3] + approach_offset

    footprint = base.get("footprint_xz_m")
    if not isinstance(footprint, list):
        raise ValueError("Robot profile has no base.footprint_xz_m")
    robot_bounds = rotated_robot_bounds(footprint, x, y, yaw_deg)

    floor = resolve_fixture(fixtures, "floor")
    floor_bounds = fixture_xy_bounds(floor)
    inside_floor = (
        robot_bounds[0] >= floor_bounds[0]
        and robot_bounds[1] <= floor_bounds[1]
        and robot_bounds[2] >= floor_bounds[2]
        and robot_bounds[3] <= floor_bounds[3]
    )

    collisions: list[str] = []
    for fixture in fixtures:
        if not isinstance(fixture, dict) or fixture is target or fixture is floor:
            continue
        name = str(fixture.get("name") or "")
        role = str(fixture.get("role") or "")
        if role == "wall" or name.startswith("wall_"):
            continue
        if not (fixture.get("asset_world_extents") or fixture.get("size")):
            continue
        if overlaps(robot_bounds, fixture_xy_bounds(fixture)):
            collisions.append(name)

    if relation == "-x":
        clearance = target_bounds[0] - robot_bounds[1]
    elif relation == "+x":
        clearance = robot_bounds[0] - target_bounds[1]
    elif relation == "-y":
        clearance = target_bounds[2] - robot_bounds[3]
    else:
        clearance = robot_bounds[2] - target_bounds[3]
    collision_activation = 0.05
    target_clear = clearance >= collision_activation
    horizontal_reach = float(manipulator.get("horizontal_reach_from_base_center_m"))
    target_reachable = approach_offset <= horizontal_reach

    checks = {
        "inside_floor": inside_floor,
        "fixture_collisions": collisions,
        "target_clearance_m": round(clearance, 6),
        "required_clearance_m": collision_activation,
        "target_clear": target_clear,
        "approach_offset_m": approach_offset,
        "horizontal_reach_m": horizontal_reach,
        "target_edge_within_reach": target_reachable,
    }
    passed = inside_floor and not collisions and target_clear and target_reachable
    result = {
        "status": "pass" if passed else "fail",
        "mode": "execute" if args.execute else "preview",
        "task": task_path.as_posix(),
        "arena": arena_path.as_posix(),
        "robot": robot_name,
        "target": target_name,
        "relation_world_axis": relation,
        "facing_world_axis": facing,
        "pose": {"translation": [round(x, 6), round(y, 6), 0.0], "euler": [0.0, 0.0, yaw_deg]},
        "target_xy_bounds": [round(value, 6) for value in target_bounds],
        "robot_xy_bounds": [round(value, 6) for value in robot_bounds],
        "checks": checks,
    }
    if not passed:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2
    if not args.execute:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    pose = [round(x, 6), round(y, 6), 0.0]
    euler = [0.0, 0.0, yaw_deg]
    region_name = f"{robot_name}_start_region"
    robot = {
        "name": robot_name,
        "robot_config_file": profile.get("interdata_config_file"),
        # `usd_asset` is the canonical repository-relative runtime path.  Keep
        # the legacy InterData field as a fallback for custom profiles, but do
        # not let a stale legacy value override the canonical path.
        "path": profile.get("usd_asset") or profile.get("interdata_task_usd_path"),
        "euler": euler,
        "ignore_substring": ["material", "floor", "wall", "scene"],
        "use_batch": True,
        "collision_activation_distance": collision_activation,
        "target_class": "SplitAloha",
    }
    existing_robots = task.get("robots")
    if not isinstance(existing_robots, list):
        existing_robots = []
    task["robots"] = [item for item in existing_robots if not isinstance(item, dict) or item.get("name") != robot_name]
    task["robots"].append(robot)

    floor_translation = vector(floor.get("translation", [0.0, 0.0, 0.0]), 2, "floor.translation")
    relative_x, relative_y = round(x - floor_translation[0], 6), round(y - floor_translation[1], 6)
    region = {
        "name": region_name,
        "type": "A_on_B_region_sampler",
        "A": robot_name,
        "B": "floor",
        "object": robot_name,
        "parent_fixture": "floor",
        "target": "floor",
        "center": [round(x, 6), round(y, 6)],
        "size": [0.05, 0.05],
        "support_surface_z": 0.0,
        "world_translation": list(pose),
        "world_euler": list(euler),
        "yaw_range": [0.0, 0.0],
        "random_type": "A_on_B_region_sampler",
        "random_config": {
            "pos_range": [[relative_x, relative_y, 0.0], [relative_x, relative_y, 0.0]],
            "yaw_rotation": [0.0, 0.0],
            "support_surface_z": 0.0,
        },
        "placement_mode": "fixed_from_region_pose",
        "sampling": {
            "mode": "fixed",
            "sampler": "A_on_B_region_sampler",
            "keep_upright": True,
            "reject_on_collision": True,
        },
        "robot_yaw_note": "Runtime start pose is region-owned; robots.euler mirrors its heading.",
    }
    regions = task.get("regions")
    if not isinstance(regions, list):
        regions = []
    task["regions"] = [
        item
        for item in regions
        if not isinstance(item, dict)
        or (item.get("name") != region_name and item.get("object") != robot_name)
    ]
    task["regions"].append(region)

    positions = task.get("positions")
    if not isinstance(positions, dict):
        positions = {}
    positions["wp_robot_start"] = {"x": pose[0], "y": pose[1], "yaw": yaw_deg}
    task["positions"] = positions

    metadata = task.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    existing_robot_placement = metadata.get("robot_placement")
    if not isinstance(existing_robot_placement, dict):
        existing_robot_placement = {}
    metadata["scene_only"] = False
    metadata["execution_scope"] = "robot_placement_only"
    metadata["robot_placement"] = {
        "target_fixture": target_name,
        "relation_world_axis": relation,
        "facing_world_axis": facing,
        "coordinate_frame": "isaac_sim_world_xyz",
        "robot_profile": profile_path.relative_to(project_root).as_posix()
        if profile_path.is_relative_to(project_root)
        else profile_path.as_posix(),
        "translation": list(pose),
        "euler": list(euler),
        "target_clearance_m": round(clearance, 6),
    }
    if args.reference_image:
        metadata["robot_placement"]["reference_image"] = args.reference_image
    elif existing_robot_placement.get("reference_image"):
        metadata["robot_placement"]["reference_image"] = existing_robot_placement[
            "reference_image"
        ]
    task["metadata"] = metadata

    atomic_write_yaml(task_path, document)
    result["written"] = True
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
