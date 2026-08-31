"""Dynamic approach-goal sampling and static map checks for local navigation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import random
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class StaticMap:
    """Immutable binary occupancy map in image coordinates."""

    occupancy: np.ndarray
    resolution: float
    origin: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.occupancy, np.ndarray):
            raise TypeError("StaticMap.occupancy must be a numpy.ndarray")
        if self.occupancy.ndim != 2:
            raise ValueError("StaticMap.occupancy must be a 2-D array")
        if self.occupancy.dtype != np.uint8:
            raise ValueError("StaticMap.occupancy must have dtype numpy.uint8")
        if not np.all((self.occupancy == 0) | (self.occupancy == 1)):
            raise ValueError("StaticMap.occupancy must contain only 0 (free) and 1 (occupied)")

        resolution = float(self.resolution)
        if not math.isfinite(resolution) or resolution <= 0.0:
            raise ValueError("StaticMap.resolution must be finite and positive")
        if not isinstance(self.origin, (tuple, list)) or len(self.origin) != 3:
            raise ValueError("StaticMap.origin must contain exactly three values")
        origin = tuple(float(value) for value in self.origin)
        if not all(math.isfinite(value) for value in origin):
            raise ValueError("StaticMap.origin must contain finite values")

        # Frozen dataclasses do not freeze ndarray buffers.  Own the input and
        # make the stored array read-only so cached maps cannot be changed by a
        # planner or a navigation skill.
        occupancy = np.array(self.occupancy, dtype=np.uint8, copy=True, order="C")
        occupancy.setflags(write=False)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "origin", origin)


@dataclass(frozen=True)
class ApproachConfig:
    target_name: str
    min_distance: float
    max_distance: float
    sample_count: int = 256
    sampling_random: bool = False
    sampling_seed: int | None = None
    arm: str | None = None
    object_armbase_xy: tuple[float, float] | None = None
    armbase_tolerance_m: float = 0.15
    max_refinements: int = 2


def wrap_to_pi(yaw: float) -> float:
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def parse_approach_config(
    cfg: dict[str, Any],
    navigation_cfg: dict[str, Any],
) -> ApproachConfig | None:
    target_name = str(cfg.get("approach", "") or "").strip()
    if not target_name:
        return None

    approach_cfg = navigation_cfg["approach"]
    min_distance = float(approach_cfg["min_distance"])
    max_distance = float(approach_cfg["max_distance"])
    sample_count = int(cfg.get("approach_sample_count", 512))
    sampling_random = _as_bool(cfg.get("approach_sampling_random", False))
    sampling_seed = cfg.get("approach_sampling_seed", None)
    sampling_seed = None if sampling_seed is None else int(sampling_seed)
    arm = _parse_arm_name(cfg.get("approach_arm", None))
    object_armbase_xy = _parse_optional_xy(cfg.get("approach_object_armbase_xy", None))
    armbase_tolerance_m = float(cfg.get("approach_armbase_tolerance", 0.15))
    max_refinements = int(cfg.get("approach_max_refinements", 2))
    if sampling_random and sampling_seed is None:
        sampling_seed = int.from_bytes(os.urandom(8), byteorder="big", signed=False)
    if min_distance <= 0.0:
        raise ValueError("base.local_navigation.approach.min_distance must be positive")
    if max_distance < min_distance:
        raise ValueError(
            "base.local_navigation.approach.max_distance must be >= "
            "base.local_navigation.approach.min_distance"
        )
    if sample_count <= 0:
        raise ValueError("approach_sample_count must be positive")
    if object_armbase_xy is not None and arm is None:
        raise ValueError("approach_object_armbase_xy requires approach_arm to be 'left' or 'right'")
    if armbase_tolerance_m < 0.0:
        raise ValueError("approach_armbase_tolerance must be non-negative")
    if max_refinements < 0:
        raise ValueError("approach_max_refinements must be non-negative")
    return ApproachConfig(
        target_name=target_name,
        min_distance=min_distance,
        max_distance=max_distance,
        sample_count=sample_count,
        sampling_random=sampling_random,
        sampling_seed=sampling_seed,
        arm=arm,
        object_armbase_xy=object_armbase_xy,
        armbase_tolerance_m=armbase_tolerance_m,
        max_refinements=max_refinements,
    )


def sample_approach_candidates(
    config: ApproachConfig,
    target_xy: tuple[float, float],
    armbase_target_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    angle_index_offset = 0
    if config.sampling_random:
        angle_index_offset = random.Random(int(config.sampling_seed)).randrange(config.sample_count)
    candidates = []
    for index in range(config.sample_count):
        radius = _candidate_radius(config, index)
        angle = _candidate_angle(index + angle_index_offset)
        x = target_x + float(radius) * math.cos(angle)
        y = target_y + float(radius) * math.sin(angle)
        yaw = wrap_to_pi(math.atan2(target_y - y, target_x - x))
        candidate = {
            "index": index,
            "x": float(x),
            "y": float(y),
            "yaw": float(yaw),
            "distance_to_target": float(math.hypot(target_x - x, target_y - y)),
            "angle": float(angle),
        }
        if armbase_target_context is not None:
            _apply_armbase_target_yaw(
                candidate,
                target_xy=(target_x, target_y),
                context=armbase_target_context,
            )
        candidates.append(candidate)
    return candidates


def _candidate_radius(config: ApproachConfig, index: int) -> float:
    min_distance = float(config.min_distance)
    max_distance = float(config.max_distance)
    if config.sample_count <= 1 or math.isclose(min_distance, max_distance, abs_tol=1e-9):
        return min_distance
    fraction = float(index) / float(config.sample_count - 1)
    return min_distance + (max_distance - min_distance) * fraction


def _candidate_angle(index: int) -> float:
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    return (float(index) * golden_angle) % (2.0 * math.pi)


def check_footprint_static_collision(
    *,
    static_map: StaticMap,
    x: float,
    y: float,
) -> dict[str, Any]:
    """Check the binary occupancy cell under the point robot center."""
    if not isinstance(static_map, StaticMap):
        raise TypeError("static_map must be a StaticMap")
    occupancy = static_map.occupancy
    height, width = occupancy.shape
    pixel_x, pixel_y = _world_to_image_pixel(
        float(x),
        float(y),
        origin_x=static_map.origin[0],
        origin_y=static_map.origin[1],
        resolution=static_map.resolution,
        height=height,
    )
    if not (0 <= pixel_x < width and 0 <= pixel_y < height):
        return {
            "ok": False,
            "reason": "center_out_of_bounds",
            "sampled_cells": 0,
            "blocked_cells": 0,
            "out_of_bounds_vertices": 1,
        }
    blocked_cells = int(occupancy[pixel_y, pixel_x] == 1)
    ok = blocked_cells == 0
    return {
        "ok": bool(ok),
        "reason": "" if ok else "static_center_collision",
        "sampled_cells": 1,
        "blocked_cells": blocked_cells,
        "out_of_bounds_vertices": 0,
    }


def check_path_static_collision(
    *,
    static_map: StaticMap,
    path_poses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check each discrete A* waypoint at the robot center."""
    if not isinstance(static_map, StaticMap):
        raise TypeError("static_map must be a StaticMap")
    blocked_results = []
    normalized_poses = [_path_pose(pose) for pose in path_poses or []]
    sampled_pose_count = 0

    def check_sample(
        pose: dict[str, float],
        *,
        index: int,
        segment_index: int | None,
        segment_fraction: float,
    ):
        nonlocal sampled_pose_count
        sampled_pose_count += 1
        result = check_footprint_static_collision(
            static_map=static_map,
            x=pose["x"],
            y=pose["y"],
        )
        if not bool(result.get("ok", False)):
            blocked_results.append(
                {
                    "index": int(index),
                    "pose": {
                        "x": pose["x"],
                        "y": pose["y"],
                        "yaw": pose["yaw"],
                    },
                    "segment_index": segment_index,
                    "segment_fraction": float(segment_fraction),
                    "reason": str(result.get("reason", "")),
                    "blocked_cells": int(result.get("blocked_cells", 0)),
                    "out_of_bounds_vertices": int(result.get("out_of_bounds_vertices", 0)),
                }
            )

    for index, pose in enumerate(normalized_poses):
        check_sample(
            pose,
            index=index,
            segment_index=None,
            segment_fraction=0.0,
        )

    return {
        "ok": len(blocked_results) == 0,
        "num_poses": int(len(normalized_poses)),
        "sampled_pose_count": int(sampled_pose_count),
        "blocked_pose_count": int(len(blocked_results)),
        "first_blocked_index": int(blocked_results[0]["index"]) if blocked_results else None,
        "first_blocked_result": blocked_results[0] if blocked_results else {},
        "blocked_summary": blocked_results[:20],
    }


def _path_pose(pose: dict[str, Any]) -> dict[str, float]:
    return {
        "x": float(pose.get("x", 0.0)),
        "y": float(pose.get("y", 0.0)),
        "yaw": wrap_to_pi(float(pose.get("yaw", 0.0))),
    }


def choose_best_reachable_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    reachable = [
        candidate
        for candidate in candidates
        if bool(candidate.get("static_ok", False)) and bool(candidate.get("path_ok", False))
    ]
    if not reachable:
        return None
    preferred_min_distance = _preferred_min_approach_distance(candidates)
    return min(
        reachable,
        key=lambda candidate: (
            _approach_rank_score(candidate, preferred_min_distance),
            float(candidate.get("distance_to_target", float("inf"))),
            float(candidate.get("path_length_m", float("inf"))),
            int(candidate.get("index", 0)),
        ),
    )


def sort_candidates_for_preflight(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred_min_distance = _preferred_min_approach_distance(candidates)
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            _approach_rank_score(candidate, preferred_min_distance),
            float(candidate.get("distance_to_target", float("inf"))),
            int(candidate.get("index", 0)),
        ),
    )
    for rank, candidate in enumerate(ordered):
        candidate["preflight_rank"] = int(rank)
        candidate["approach_rank_score"] = float(_approach_rank_score(candidate, preferred_min_distance))
        candidate["approach_distance_shortfall_penalty"] = float(
            _approach_distance_shortfall_penalty(candidate, preferred_min_distance)
        )
    return ordered


def _preferred_min_approach_distance(candidates: list[dict[str, Any]]) -> float:
    distances = [
        float(candidate["distance_to_target"])
        for candidate in candidates
        if _is_finite_number(candidate.get("distance_to_target"))
    ]
    if not distances:
        return 0.0
    min_distance = min(distances)
    max_distance = max(distances)
    return min_distance + 0.25 * max(max_distance - min_distance, 0.0)


def _approach_rank_score(candidate: dict[str, Any], preferred_min_distance: float) -> float:
    base_score = float(candidate.get("approach_score", candidate.get("distance_to_target", float("inf"))))
    return float(base_score + _approach_distance_shortfall_penalty(candidate, preferred_min_distance))


def _approach_distance_shortfall_penalty(candidate: dict[str, Any], preferred_min_distance: float) -> float:
    distance = float(candidate.get("distance_to_target", float("inf")))
    shortfall = max(float(preferred_min_distance) - distance, 0.0)
    return float(5.0 * shortfall * shortfall)


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def build_armbase_target_context(robot_cfg: dict[str, Any], config: ApproachConfig) -> dict[str, Any] | None:
    """Build arm-base target metadata from a robot config.

    The returned context is intentionally plain data so it can be serialized in
    dynamic-goal debug output and tested without Isaac imports.
    """

    if config.object_armbase_xy is None:
        return None
    if config.arm not in {"left", "right"}:
        raise ValueError("approach_arm must be 'left' or 'right' when approach_object_armbase_xy is set")
    robot_cfg = robot_cfg if isinstance(robot_cfg, dict) else {}
    prefix = "fl" if config.arm == "left" else "fr"
    translation = _parse_vec(robot_cfg.get(f"{prefix}_base_mount_translation"), 3, f"{prefix}_base_mount_translation")
    orientation = _parse_vec(
        robot_cfg.get(f"{prefix}_base_mount_orientation", [1.0, 0.0, 0.0, 0.0]),
        4,
        f"{prefix}_base_mount_orientation",
    )
    desired_mobile_xy = _armbase_xy_to_mobile_xy(
        config.object_armbase_xy,
        translation=translation,
        orientation=orientation,
    )
    target_angle = math.atan2(float(desired_mobile_xy[1]), float(desired_mobile_xy[0]))
    return {
        "arm": str(config.arm),
        "object_armbase_xy": [float(config.object_armbase_xy[0]), float(config.object_armbase_xy[1])],
        "mobile_to_armbase_translation": [float(value) for value in translation],
        "mobile_to_armbase_orientation": [float(value) for value in orientation],
        "mobile_to_armbase_yaw": float(_quat_wxyz_yaw(orientation)),
        "object_mobile_xy": [float(desired_mobile_xy[0]), float(desired_mobile_xy[1])],
        "object_mobile_angle": float(target_angle),
    }


def write_candidates_debug(path: str, payload: dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _parse_arm_name(value) -> str | None:
    if value is None:
        return None
    arm = str(value).strip().lower()
    if not arm:
        return None
    if arm in {"left", "right"}:
        return arm
    raise ValueError("approach_arm must be 'left' or 'right'")


def _parse_optional_xy(value) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("approach_object_armbase_xy must be a two-element list")
    return float(value[0]), float(value[1])


def _parse_vec(value, expected_len: int, field_name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_len:
        raise ValueError(f"{field_name} must be a {expected_len}-element list")
    return [float(item) for item in value]


def _quat_wxyz_yaw(orientation: list[float]) -> float:
    w, x, y, z = (float(value) for value in orientation[:4])
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _armbase_xy_to_mobile_xy(
    object_armbase_xy: tuple[float, float],
    *,
    translation: list[float],
    orientation: list[float],
) -> tuple[float, float]:
    mount_yaw = _quat_wxyz_yaw(orientation)
    cos_yaw = math.cos(mount_yaw)
    sin_yaw = math.sin(mount_yaw)
    arm_x, arm_y = float(object_armbase_xy[0]), float(object_armbase_xy[1])
    mobile_x = float(translation[0]) + arm_x * cos_yaw - arm_y * sin_yaw
    mobile_y = float(translation[1]) + arm_x * sin_yaw + arm_y * cos_yaw
    return mobile_x, mobile_y


def _apply_armbase_target_yaw(
    candidate: dict[str, Any],
    *,
    target_xy: tuple[float, float],
    context: dict[str, Any],
):
    object_mobile_xy = context.get("object_mobile_xy")
    if not isinstance(object_mobile_xy, (list, tuple)) or len(object_mobile_xy) != 2:
        return
    object_mobile_angle = math.atan2(float(object_mobile_xy[1]), float(object_mobile_xy[0]))
    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    dx = target_x - float(candidate["x"])
    dy = target_y - float(candidate["y"])
    yaw = wrap_to_pi(math.atan2(dy, dx) - object_mobile_angle)
    distance = float(math.hypot(dx, dy))
    achieved_mobile_x = distance * math.cos(object_mobile_angle)
    achieved_mobile_y = distance * math.sin(object_mobile_angle)
    achieved_arm_x, achieved_arm_y = _mobile_xy_to_armbase_xy(
        (achieved_mobile_x, achieved_mobile_y),
        translation=context.get("mobile_to_armbase_translation", [0.0, 0.0, 0.0]),
        yaw=float(context.get("mobile_to_armbase_yaw", 0.0)),
    )
    desired_arm_xy = context.get("object_armbase_xy")
    if isinstance(desired_arm_xy, (list, tuple)) and len(desired_arm_xy) == 2:
        score = (achieved_arm_x - float(desired_arm_xy[0])) ** 2 + (
            achieved_arm_y - float(desired_arm_xy[1])
        ) ** 2
    else:
        score = float(candidate.get("distance_to_target", distance))
    candidate["yaw"] = float(yaw)
    candidate["approach_yaw_strategy"] = "object_armbase_xy"
    candidate["approach_score"] = float(score)
    candidate["approach_armbase_prediction"] = {
        "object_armbase_xy": [float(achieved_arm_x), float(achieved_arm_y)],
        "object_mobile_xy": [float(achieved_mobile_x), float(achieved_mobile_y)],
        "target_object_mobile_xy": [float(object_mobile_xy[0]), float(object_mobile_xy[1])],
        "object_mobile_angle": float(object_mobile_angle),
    }


def _mobile_xy_to_armbase_xy(
    object_mobile_xy: tuple[float, float],
    *,
    translation,
    yaw: float,
) -> tuple[float, float]:
    translation = _parse_vec(translation, 3, "mobile_to_armbase_translation")
    rel_x = float(object_mobile_xy[0]) - float(translation[0])
    rel_y = float(object_mobile_xy[1]) - float(translation[1])
    cos_yaw = math.cos(-float(yaw))
    sin_yaw = math.sin(-float(yaw))
    arm_x = rel_x * cos_yaw - rel_y * sin_yaw
    arm_y = rel_x * sin_yaw + rel_y * cos_yaw
    return arm_x, arm_y


def _world_to_image_pixel(
    world_x: float,
    world_y: float,
    *,
    origin_x: float,
    origin_y: float,
    resolution: float,
    height: int,
) -> tuple[int, int]:
    col = int(round((float(world_x) - origin_x) / resolution))
    map_row = int(round((float(world_y) - origin_y) / resolution))
    return col, int(height - 1 - map_row)
