"""Dynamic approach-goal sampling and static map checks for Nav2 skills."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import random
from typing import Any

import numpy as np
import yaml

from workflows.simbox.core.mobile.platforms import get_mobile_base_platform


@dataclass(frozen=True)
class ApproachConfig:
    target_name: str
    min_distance: float = 0.85
    max_distance: float = 1.30
    sample_count: int = 256
    static_free_value_min: int = 250
    footprint_padding_m: float | None = None
    sampling_random: bool = False
    sampling_seed: int | None = None
    arm: str | None = None
    object_armbase_xy: tuple[float, float] | None = None


def wrap_to_pi(yaw: float) -> float:
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def parse_approach_config(cfg: dict[str, Any]) -> ApproachConfig | None:
    target_name = str(cfg.get("approach", "") or "").strip()
    if not target_name:
        return None

    min_distance = float(cfg.get("approach_min_distance", 0.55))
    max_distance = float(cfg.get("approach_max_distance", 0.85))
    sample_count = int(cfg.get("approach_sample_count", 256))
    footprint_padding_m = cfg.get("approach_footprint_padding", None)
    footprint_padding_m = None if footprint_padding_m is None else float(footprint_padding_m)
    sampling_random = _as_bool(cfg.get("approach_sampling_random", False))
    sampling_seed = cfg.get("approach_sampling_seed", None)
    sampling_seed = None if sampling_seed is None else int(sampling_seed)
    arm = _parse_arm_name(cfg.get("approach_arm", None))
    object_armbase_xy = _parse_optional_xy(cfg.get("approach_object_armbase_xy", None))
    if sampling_random and sampling_seed is None:
        sampling_seed = int.from_bytes(os.urandom(8), byteorder="big", signed=False)
    if min_distance <= 0.0:
        raise ValueError("approach_min_distance must be positive")
    if max_distance < min_distance:
        raise ValueError("approach_max_distance must be >= approach_min_distance")
    if sample_count <= 0:
        raise ValueError("approach_sample_count must be positive")
    if footprint_padding_m is not None and footprint_padding_m < 0.0:
        raise ValueError("approach_footprint_padding must be non-negative")
    if object_armbase_xy is not None and arm is None:
        raise ValueError("approach_object_armbase_xy requires approach_arm to be 'left' or 'right'")
    return ApproachConfig(
        target_name=target_name,
        min_distance=min_distance,
        max_distance=max_distance,
        sample_count=sample_count,
        footprint_padding_m=footprint_padding_m,
        sampling_random=sampling_random,
        sampling_seed=sampling_seed,
        arm=arm,
        object_armbase_xy=object_armbase_xy,
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


def resolve_nav2_footprint_points(base_cfg: dict[str, Any]) -> list[list[float]]:
    skill_cfg = base_cfg.get("nav2_skill", {}) if isinstance(base_cfg, dict) else {}
    points = _normalize_footprint_points(skill_cfg.get("footprint_points") if isinstance(skill_cfg, dict) else None)
    if points:
        return points
    return get_mobile_base_platform(base_cfg).default_nav2_footprint_points(base_cfg)


def resolve_approach_footprint_padding_m(base_cfg: dict[str, Any], config: ApproachConfig) -> float:
    if config.footprint_padding_m is not None:
        return float(config.footprint_padding_m)
    skill_cfg = base_cfg.get("nav2_skill", {}) if isinstance(base_cfg, dict) else {}
    shared_costmap = skill_cfg.get("costmap", {}) if isinstance(skill_cfg, dict) else {}
    if isinstance(shared_costmap, dict) and "footprint_padding" in shared_costmap:
        return max(float(shared_costmap.get("footprint_padding", 0.0)), 0.0)
    if isinstance(skill_cfg, dict) and "approach_footprint_padding" in skill_cfg:
        return max(float(skill_cfg.get("approach_footprint_padding", 0.0)), 0.0)
    local_costmap = skill_cfg.get("local_costmap", {}) if isinstance(skill_cfg, dict) else {}
    if isinstance(local_costmap, dict):
        return max(float(local_costmap.get("footprint_padding", 0.0)), 0.0)
    return 0.0


def load_static_map(map_yaml_path: str) -> dict[str, Any]:
    with open(map_yaml_path, "r", encoding="utf-8") as handle:
        map_yaml = yaml.safe_load(handle) or {}
    image_path = str(map_yaml.get("image", "")).strip()
    if not image_path:
        raise ValueError(f"map yaml {map_yaml_path} is missing image")
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)

    image = _load_pgm_image(image_path)
    origin = list(map_yaml.get("origin", [0.0, 0.0, 0.0]))
    return {
        "yaml_path": str(map_yaml_path),
        "image_path": image_path,
        "image": image,
        "resolution": float(map_yaml.get("resolution", 0.0)),
        "origin": [float(origin[0]), float(origin[1]), float(origin[2] if len(origin) > 2 else 0.0)],
        "occupied_thresh": float(map_yaml.get("occupied_thresh", 0.65)),
        "free_thresh": float(map_yaml.get("free_thresh", 0.25)),
    }


def check_footprint_static_collision(
    *,
    static_map: dict[str, Any],
    footprint_points: list[list[float]],
    x: float,
    y: float,
    yaw: float,
    free_value_min: int = 250,
    footprint_padding_m: float = 0.0,
) -> dict[str, Any]:
    image = np.asarray(static_map["image"])
    resolution = float(static_map["resolution"])
    origin = list(static_map["origin"])
    if resolution <= 0.0:
        raise ValueError("static map resolution must be positive")
    footprint_padding_m = max(float(footprint_padding_m), 0.0)
    footprint_padding_cells = int(math.ceil(footprint_padding_m / resolution)) if footprint_padding_m > 0.0 else 0
    height, width = image.shape[:2]
    world_polygon = transform_footprint_points(footprint_points, x=x, y=y, yaw=yaw)
    pixel_polygon = [
        _world_to_image_pixel(wx, wy, origin_x=float(origin[0]), origin_y=float(origin[1]), resolution=resolution, height=height)
        for wx, wy in world_polygon
    ]
    mask = _polygon_mask(height=height, width=width, polygon=pixel_polygon)
    if not mask.any():
        return {
            "ok": False,
            "reason": "footprint_out_of_bounds",
            "sampled_cells": 0,
            "blocked_cells": 0,
            "footprint_blocked_cells": 0,
            "padding_blocked_cells": 0,
            "unknown_cells": 0,
            "out_of_bounds_vertices": _out_of_bounds_vertices(pixel_polygon, width=width, height=height),
            "footprint_padding_m": float(footprint_padding_m),
            "footprint_padding_cells": int(footprint_padding_cells),
            "footprint_world": world_polygon,
        }

    out_of_bounds_vertices = _out_of_bounds_vertices(pixel_polygon, width=width, height=height)
    padding_out_of_bounds = _padding_out_of_bounds(pixel_polygon, width=width, height=height, padding_cells=footprint_padding_cells)
    padded_mask = _dilate_mask(mask, footprint_padding_cells)
    values = image[padded_mask]
    footprint_values = image[mask]
    padding_values = image[np.logical_and(padded_mask, np.logical_not(mask))]
    unknown_cells = int(np.count_nonzero(values < 0)) if np.issubdtype(values.dtype, np.signedinteger) else 0
    blocked_cells = int(np.count_nonzero(values < int(free_value_min)))
    footprint_blocked_cells = int(np.count_nonzero(footprint_values < int(free_value_min)))
    padding_blocked_cells = int(np.count_nonzero(padding_values < int(free_value_min)))
    ok = blocked_cells == 0 and unknown_cells == 0 and out_of_bounds_vertices == 0 and not padding_out_of_bounds
    reason = "" if ok else "static_footprint_collision"
    if out_of_bounds_vertices or padding_out_of_bounds:
        reason = "footprint_out_of_bounds"
    return {
        "ok": bool(ok),
        "reason": reason,
        "sampled_cells": int(values.size),
        "unpadded_sampled_cells": int(footprint_values.size),
        "blocked_cells": blocked_cells,
        "footprint_blocked_cells": footprint_blocked_cells,
        "padding_blocked_cells": padding_blocked_cells,
        "unknown_cells": unknown_cells,
        "out_of_bounds_vertices": int(out_of_bounds_vertices),
        "padding_out_of_bounds": bool(padding_out_of_bounds),
        "footprint_padding_m": float(footprint_padding_m),
        "footprint_padding_cells": int(footprint_padding_cells),
        "footprint_world": world_polygon,
    }


def check_path_static_collision(
    *,
    static_map: dict[str, Any],
    footprint_points: list[list[float]],
    path_poses: list[dict[str, Any]],
    free_value_min: int = 250,
    footprint_padding_m: float = 0.0,
) -> dict[str, Any]:
    """Check every path pose and the footprint sweep between adjacent poses."""
    blocked_results = []
    normalized_poses = [_path_pose(pose) for pose in path_poses or []]
    sampled_pose_count = 0
    interpolated_pose_count = 0

    def check_sample(
        pose: dict[str, float],
        *,
        index: int,
        segment_index: int | None,
        segment_fraction: float,
    ):
        nonlocal sampled_pose_count, interpolated_pose_count
        sampled_pose_count += 1
        if segment_fraction not in {0.0, 1.0}:
            interpolated_pose_count += 1
        result = check_footprint_static_collision(
            static_map=static_map,
            footprint_points=footprint_points,
            x=pose["x"],
            y=pose["y"],
            yaw=pose["yaw"],
            free_value_min=free_value_min,
            footprint_padding_m=footprint_padding_m,
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
                    "footprint_blocked_cells": int(result.get("footprint_blocked_cells", 0)),
                    "padding_blocked_cells": int(result.get("padding_blocked_cells", 0)),
                    "unknown_cells": int(result.get("unknown_cells", 0)),
                    "out_of_bounds_vertices": int(result.get("out_of_bounds_vertices", 0)),
                    "padding_out_of_bounds": bool(result.get("padding_out_of_bounds", False)),
                    "footprint_padding_m": float(result.get("footprint_padding_m", footprint_padding_m)),
                    "footprint_padding_cells": int(result.get("footprint_padding_cells", 0)),
                }
            )

    if normalized_poses:
        check_sample(
            normalized_poses[0],
            index=0,
            segment_index=None,
            segment_fraction=0.0,
        )
    for index in range(1, len(normalized_poses)):
        previous_pose = normalized_poses[index - 1]
        pose = normalized_poses[index]
        sample_count = _path_segment_sample_count(
            static_map=static_map,
            footprint_points=footprint_points,
            previous_pose=previous_pose,
            pose=pose,
        )
        yaw_delta = wrap_to_pi(pose["yaw"] - previous_pose["yaw"])
        for sample_index in range(1, sample_count + 1):
            fraction = float(sample_index) / float(sample_count)
            check_sample(
                {
                    "x": previous_pose["x"] + fraction * (pose["x"] - previous_pose["x"]),
                    "y": previous_pose["y"] + fraction * (pose["y"] - previous_pose["y"]),
                    "yaw": wrap_to_pi(previous_pose["yaw"] + fraction * yaw_delta),
                },
                index=index,
                segment_index=index - 1,
                segment_fraction=fraction,
            )

    return {
        "ok": len(blocked_results) == 0,
        "num_poses": int(len(normalized_poses)),
        "sampled_pose_count": int(sampled_pose_count),
        "interpolated_pose_count": int(interpolated_pose_count),
        "blocked_pose_count": int(len(blocked_results)),
        "first_blocked_index": int(blocked_results[0]["index"]) if blocked_results else None,
        "first_blocked_result": blocked_results[0] if blocked_results else {},
        "blocked_summary": blocked_results[:20],
        "free_value_min": int(free_value_min),
        "footprint_padding_m": float(max(float(footprint_padding_m), 0.0)),
    }


def _path_pose(pose: dict[str, Any]) -> dict[str, float]:
    return {
        "x": float(pose.get("x", 0.0)),
        "y": float(pose.get("y", 0.0)),
        "yaw": wrap_to_pi(float(pose.get("yaw", 0.0))),
    }


def _path_segment_sample_count(
    *,
    static_map: dict[str, Any],
    footprint_points: list[list[float]],
    previous_pose: dict[str, float],
    pose: dict[str, float],
) -> int:
    resolution = float(static_map["resolution"])
    if resolution <= 0.0:
        raise ValueError("static map resolution must be positive")
    translation_distance = math.hypot(pose["x"] - previous_pose["x"], pose["y"] - previous_pose["y"])
    yaw_delta = abs(wrap_to_pi(pose["yaw"] - previous_pose["yaw"]))
    footprint_radius = max((math.hypot(float(px), float(py)) for px, py in footprint_points), default=0.0)
    # A footprint vertex may travel both due to translation and rotation. Keep
    # that sweep below half a map cell so a gap between planner poses cannot
    # skip an occupied static-map cell.
    max_sweep_step_m = resolution * 0.5
    sweep_distance = translation_distance + footprint_radius * yaw_delta
    return max(1, int(math.ceil(sweep_distance / max_sweep_step_m)))


def transform_footprint_points(
    footprint_points: list[list[float]],
    *,
    x: float,
    y: float,
    yaw: float,
) -> list[list[float]]:
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    return [
        [
            float(x) + float(px) * cos_yaw - float(py) * sin_yaw,
            float(y) + float(px) * sin_yaw + float(py) * cos_yaw,
        ]
        for px, py in footprint_points
    ]


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


def _normalize_footprint_points(points) -> list[list[float]]:
    if not isinstance(points, (list, tuple)):
        return []
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        normalized.append([float(point[0]), float(point[1])])
    if len(normalized) < 3:
        return []
    return normalized


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


def _load_pgm_image(path: str) -> np.ndarray:
    try:
        import cv2  # type: ignore[import-not-found]

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is not None:
            if image.ndim == 3:
                image = image[:, :, 0]
            return np.asarray(image)
    except Exception:
        pass

    from PIL import Image

    return np.asarray(Image.open(path))


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


def _polygon_mask(*, height: int, width: int, polygon: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if len(polygon) < 3:
        return mask
    min_col = max(0, min(point[0] for point in polygon))
    max_col = min(width - 1, max(point[0] for point in polygon))
    min_row = max(0, min(point[1] for point in polygon))
    max_row = min(height - 1, max(point[1] for point in polygon))
    if min_col > max_col or min_row > max_row:
        return mask
    poly = [(float(col), float(row)) for col, row in polygon]
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            if _point_in_polygon(float(col) + 0.5, float(row) + 0.5, poly):
                mask[row, col] = True
    return mask


def _dilate_mask(mask: np.ndarray, padding_cells: int) -> np.ndarray:
    padding_cells = int(padding_cells)
    if padding_cells <= 0:
        return mask
    height, width = mask.shape[:2]
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return mask.copy()
    padded = mask.copy()
    for row, col in zip(rows, cols):
        min_row = max(0, int(row) - padding_cells)
        max_row = min(height - 1, int(row) + padding_cells)
        min_col = max(0, int(col) - padding_cells)
        max_col = min(width - 1, int(col) + padding_cells)
        for out_row in range(min_row, max_row + 1):
            row_delta_sq = (out_row - int(row)) * (out_row - int(row))
            for out_col in range(min_col, max_col + 1):
                delta_sq = row_delta_sq + (out_col - int(col)) * (out_col - int(col))
                if delta_sq <= padding_cells * padding_cells:
                    padded[out_row, out_col] = True
    return padded


def _padding_out_of_bounds(
    polygon: list[tuple[int, int]],
    *,
    width: int,
    height: int,
    padding_cells: int,
) -> bool:
    if padding_cells <= 0:
        return False
    return any(
        col - padding_cells < 0
        or col + padding_cells >= width
        or row - padding_cells < 0
        or row + padding_cells >= height
        for col, row in polygon
    )


def _point_in_polygon(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, point_i in enumerate(polygon):
        xi, yi = point_i
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) if not math.isclose(yj, yi) else 1.0e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def _out_of_bounds_vertices(polygon: list[tuple[int, int]], *, width: int, height: int) -> int:
    return sum(1 for col, row in polygon if col < 0 or col >= width or row < 0 or row >= height)
