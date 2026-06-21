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


def wrap_to_pi(yaw: float) -> float:
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def parse_approach_config(cfg: dict[str, Any]) -> ApproachConfig | None:
    target_name = str(cfg.get("approach", "") or "").strip()
    if not target_name:
        return None

    min_distance = float(cfg.get("approach_min_distance", 0.55))
    max_distance = float(cfg.get("approach_max_distance", 1.15))
    sample_count = int(cfg.get("approach_sample_count", 512))
    footprint_padding_m = cfg.get("approach_footprint_padding", None)
    footprint_padding_m = None if footprint_padding_m is None else float(footprint_padding_m)
    sampling_random = _as_bool(cfg.get("approach_sampling_random", False))
    sampling_seed = cfg.get("approach_sampling_seed", None)
    sampling_seed = None if sampling_seed is None else int(sampling_seed)
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
    return ApproachConfig(
        target_name=target_name,
        min_distance=min_distance,
        max_distance=max_distance,
        sample_count=sample_count,
        footprint_padding_m=footprint_padding_m,
        sampling_random=sampling_random,
        sampling_seed=sampling_seed,
    )


def sample_approach_candidates(config: ApproachConfig, target_xy: tuple[float, float]) -> list[dict[str, Any]]:
    target_x, target_y = float(target_xy[0]), float(target_xy[1])
    rng = random.Random(int(config.sampling_seed)) if config.sampling_random else None
    candidates = []
    for index in range(config.sample_count):
        radius = _candidate_radius(config, index, rng)
        angle = _candidate_angle(index, rng)
        x = target_x + float(radius) * math.cos(angle)
        y = target_y + float(radius) * math.sin(angle)
        yaw = wrap_to_pi(math.atan2(target_y - y, target_x - x))
        candidates.append(
            {
                "index": index,
                "x": float(x),
                "y": float(y),
                "yaw": float(yaw),
                "distance_to_target": float(math.hypot(target_x - x, target_y - y)),
                "angle": float(angle),
            }
        )
    return candidates


def _candidate_radius(config: ApproachConfig, index: int, rng: random.Random | None = None) -> float:
    min_distance = float(config.min_distance)
    max_distance = float(config.max_distance)
    if config.sample_count <= 1 or math.isclose(min_distance, max_distance, abs_tol=1e-9):
        return rng.uniform(min_distance, max_distance) if rng is not None else min_distance
    if rng is not None:
        return rng.uniform(min_distance, max_distance)
    fraction = float(index) / float(config.sample_count - 1)
    return min_distance + (max_distance - min_distance) * fraction


def _candidate_angle(index: int, rng: random.Random | None = None) -> float:
    if rng is not None:
        return rng.uniform(0.0, 2.0 * math.pi)
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
    return min(
        reachable,
        key=lambda candidate: (
            float(candidate.get("distance_to_target", float("inf"))),
            float(candidate.get("path_length_m", float("inf"))),
            int(candidate.get("index", 0)),
        ),
    )


def sort_candidates_for_preflight(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.get("distance_to_target", float("inf"))),
            int(candidate.get("index", 0)),
        ),
    )
    for rank, candidate in enumerate(ordered):
        candidate["preflight_rank"] = int(rank)
    return ordered


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
