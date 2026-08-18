"""Pure deterministic camera templates shared by Agent and SimBox."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


ROBOT_TARGET_DIAGONAL_V1 = "robot_target_diagonal_v1"
ROBOT_TARGET_OVERHEAD_V1 = "robot_target_overhead_v1"
ROBOT_TARGET_DIAGONAL_DEFAULTS = {
    "behind_m": 0.75,
    "side_m": 0.85,
    "height_m": 1.20,
    "look_fraction": 0.65,
    "look_height_m": 0.20,
}
ROBOT_TARGET_OVERHEAD_DEFAULTS = {
    "height_m": 1.75,
    "look_fraction": 0.65,
    "look_height_m": 0.0,
}
CAMERA_TEMPLATE_DEFAULTS = {
    ROBOT_TARGET_DIAGONAL_V1: ROBOT_TARGET_DIAGONAL_DEFAULTS,
    ROBOT_TARGET_OVERHEAD_V1: ROBOT_TARGET_OVERHEAD_DEFAULTS,
}


def _point(values: Sequence[Any], name: str) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    point = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{name} must contain finite numbers")
    return point


def robot_target_diagonal_pose(
    robot_world_xyz: Sequence[Any],
    robot_yaw_deg: float,
    target_world_xyz: Sequence[Any],
    params: Mapping[str, Any] | None = None,
    room_bounds_xy: Sequence[Any] | None = None,
) -> dict[str, list[float]]:
    """Place an independent world camera behind and beside a robot-target pair."""

    robot = _point(robot_world_xyz, "robot_world_xyz")
    target = _point(target_world_xyz, "target_world_xyz")
    values = {**ROBOT_TARGET_DIAGONAL_DEFAULTS, **dict(params or {})}
    unknown = sorted(set(values) - set(ROBOT_TARGET_DIAGONAL_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown robot-target camera parameters: {unknown}")
    values = {key: float(value) for key, value in values.items()}
    yaw_deg = float(robot_yaw_deg)
    if not math.isfinite(yaw_deg) or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise ValueError("robot-target camera values must be finite")
    if values["behind_m"] < 0.0 or values["height_m"] <= values["look_height_m"]:
        raise ValueError(
            "robot-target camera requires behind_m >= 0 and height_m > look_height_m"
        )
    if not 0.0 <= values["look_fraction"] <= 1.0:
        raise ValueError("robot-target camera look_fraction must be in [0, 1]")

    yaw = math.radians(yaw_deg)
    forward = (math.cos(yaw), math.sin(yaw))
    left = (-forward[1], forward[0])
    anchor_z = max(robot[2], target[2])
    candidates = [
        [
            robot[0] + longitudinal * values["behind_m"] * forward[0]
            + lateral * values["side_m"] * left[0],
            robot[1] + longitudinal * values["behind_m"] * forward[1]
            + lateral * values["side_m"] * left[1],
            anchor_z + values["height_m"],
        ]
        for longitudinal, lateral in ((-1.0, 1.0), (-1.0, -1.0), (1.0, 1.0), (1.0, -1.0))
    ]
    eye = candidates[0]
    if room_bounds_xy is not None:
        bounds = tuple(float(value) for value in room_bounds_xy)
        if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("room_bounds_xy must contain four finite numbers")
        min_x, max_x, min_y, max_y = bounds
        if min_x >= max_x or min_y >= max_y:
            raise ValueError("room_bounds_xy min values must be below max values")

        def wall_clearance(candidate: Sequence[float]) -> float:
            return min(
                candidate[0] - min_x,
                max_x - candidate[0],
                candidate[1] - min_y,
                max_y - candidate[1],
            )

        eye = max(candidates, key=wall_clearance)
    look_at = [
        robot[0] + values["look_fraction"] * (target[0] - robot[0]),
        robot[1] + values["look_fraction"] * (target[1] - robot[1]),
        anchor_z + values["look_height_m"],
    ]
    return {"eye": eye, "target": look_at}


def robot_target_overhead_pose(
    robot_world_xyz: Sequence[Any],
    target_world_xyz: Sequence[Any],
    params: Mapping[str, Any] | None = None,
) -> dict[str, list[float]]:
    """Place an independent world camera above the robot-target work area."""

    robot = _point(robot_world_xyz, "robot_world_xyz")
    target = _point(target_world_xyz, "target_world_xyz")
    values = {**ROBOT_TARGET_OVERHEAD_DEFAULTS, **dict(params or {})}
    unknown = sorted(set(values) - set(ROBOT_TARGET_OVERHEAD_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown robot-target overhead parameters: {unknown}")
    values = {key: float(value) for key, value in values.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("robot-target overhead camera values must be finite")
    if values["height_m"] <= values["look_height_m"]:
        raise ValueError("overhead camera height_m must be above look_height_m")
    if not 0.0 <= values["look_fraction"] <= 1.0:
        raise ValueError("overhead camera look_fraction must be in [0, 1]")
    anchor_z = max(robot[2], target[2])
    look_at = [
        robot[0] + values["look_fraction"] * (target[0] - robot[0]),
        robot[1] + values["look_fraction"] * (target[1] - robot[1]),
        anchor_z + values["look_height_m"],
    ]
    return {
        "eye": [look_at[0], look_at[1], anchor_z + values["height_m"]],
        "target": look_at,
    }


def resolve_camera_template_pose(
    template: str,
    robot_world_xyz: Sequence[Any],
    robot_yaw_deg: float,
    target_world_xyz: Sequence[Any],
    params: Mapping[str, Any] | None = None,
    room_bounds_xy: Sequence[Any] | None = None,
) -> dict[str, list[float]]:
    if template == ROBOT_TARGET_OVERHEAD_V1:
        return robot_target_overhead_pose(robot_world_xyz, target_world_xyz, params)
    if template == ROBOT_TARGET_DIAGONAL_V1:
        return robot_target_diagonal_pose(
            robot_world_xyz,
            robot_yaw_deg,
            target_world_xyz,
            params,
            room_bounds_xy,
        )
    raise ValueError(f"unsupported camera template: {template}")


def room_bounds_xy_from_arena(arena: Mapping[str, Any]) -> list[float] | None:
    """Read world XY room bounds from an arena mapping without loading Isaac."""

    coordinate_frame = arena.get("coordinate_frame") or {}
    raw_bounds = coordinate_frame.get("room_bounds_xz")
    if isinstance(raw_bounds, Sequence) and not isinstance(raw_bounds, (str, bytes)):
        bounds = [float(value) for value in raw_bounds]
        if len(bounds) == 4 and all(math.isfinite(value) for value in bounds):
            return bounds
    for fixture in arena.get("fixtures") or []:
        if not isinstance(fixture, Mapping) or fixture.get("name") != "floor":
            continue
        size = fixture.get("size")
        translation = fixture.get("translation")
        if (
            isinstance(size, Sequence)
            and isinstance(translation, Sequence)
            and len(size) >= 2
            and len(translation) >= 2
        ):
            half_x, half_y = float(size[0]) * 0.5, float(size[1]) * 0.5
            center_x, center_y = float(translation[0]), float(translation[1])
            return [
                center_x - half_x,
                center_x + half_x,
                center_y - half_y,
                center_y + half_y,
            ]
    return None
