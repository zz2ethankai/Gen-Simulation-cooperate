"""Pure NumPy geometry helpers for Pick/Place target visualization."""

from __future__ import annotations

from itertools import product

import numpy as np


def normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        return vector / norm
    if fallback is None:
        raise ValueError("cannot normalize a zero-length vector")
    return normalize(fallback)


def pose_matrix(position: np.ndarray, orientation_wxyz: np.ndarray) -> np.ndarray:
    """Build a column-vector transform from a scalar-first quaternion."""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    quaternion = np.asarray(orientation_wxyz, dtype=np.float64).reshape(4)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return points @ transform[:3, :3].T + transform[:3, 3]


def gripper_line_curves(
    ee_transform: np.ndarray,
    tool_head: np.ndarray,
    tool_tail: np.ndarray,
    tool_side: np.ndarray,
    gripper_max_width: float,
) -> list[np.ndarray]:
    """Create an AnyGrasp-style parallel-jaw outline from robot keypoints."""
    head = np.asarray(tool_head, dtype=np.float64).reshape(-1)[:3]
    tail = np.asarray(tool_tail, dtype=np.float64).reshape(-1)[:3]
    side = np.asarray(tool_side, dtype=np.float64).reshape(-1)[:3]
    approach = normalize(head - tail, np.array([0.0, 0.0, 1.0]))
    side_axis = normalize(side - head, np.array([1.0, 0.0, 0.0]))
    side_axis = normalize(
        side_axis - approach * float(np.dot(side_axis, approach)),
        np.array([1.0, 0.0, 0.0]),
    )
    finger_length = max(float(np.linalg.norm(head - tail)), 0.04)
    half_width = max(float(gripper_max_width) * 0.5, 0.01)
    front_center = head
    back_center = head - approach * finger_length
    back_left = back_center + side_axis * half_width
    back_right = back_center - side_axis * half_width
    front_left = front_center + side_axis * half_width
    front_right = front_center - side_axis * half_width
    local_curves = [
        np.stack([back_left, front_left]),
        np.stack([back_right, front_right]),
        np.stack([back_left, back_right]),
        np.stack([back_center, front_center]),
    ]
    return [transform_points(curve, ee_transform) for curve in local_curves]


def dashed_line_curves(
    start: np.ndarray,
    end: np.ndarray,
    dash_length_m: float = 0.018,
    gap_length_m: float = 0.010,
) -> list[np.ndarray]:
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-12:
        return []
    direction = delta / length
    curves = []
    distance = 0.0
    while distance < length:
        dash_end = min(distance + dash_length_m, length)
        curves.append(
            np.stack([start + direction * distance, start + direction * dash_end])
        )
        distance += dash_length_m + gap_length_m
    return curves


def ratio_box_corners(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    ratio_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> np.ndarray:
    """Return the exact eight corners of an axis-aligned ratio sub-volume."""
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    ratio_min = np.array([min(values) for values in ratio_ranges], dtype=np.float64)
    ratio_max = np.array([max(values) for values in ratio_ranges], dtype=np.float64)
    lower = bbox_min + ratio_min * (bbox_max - bbox_min)
    upper = bbox_min + ratio_max * (bbox_max - bbox_min)
    return np.asarray(
        [[xs, ys, zs] for xs, ys, zs in product(*zip(lower, upper))],
        dtype=np.float64,
    )


def plane_from_region_points(
    region_points: np.ndarray,
    normal: np.ndarray,
    min_display_extent_m: float,
    normal_offset_m: float = 0.0,
    tangent_hint: np.ndarray | None = None,
) -> dict[str, np.ndarray | bool]:
    """Flatten a target domain into a visible plane without losing true extents."""
    points = np.asarray(region_points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("region_points must not be empty")
    normal = normalize(normal, np.array([0.0, 0.0, 1.0]))
    if tangent_hint is None:
        tangent_hint = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(tangent_hint, normal))) > 0.95:
            tangent_hint = np.array([0.0, 1.0, 0.0])
    tangent_u = np.asarray(tangent_hint, dtype=np.float64).reshape(3)
    tangent_u -= normal * float(np.dot(tangent_u, normal))
    tangent_u = normalize(tangent_u, np.array([1.0, 0.0, 0.0]))
    tangent_v = normalize(np.cross(normal, tangent_u), np.array([0.0, 1.0, 0.0]))

    origin = points.mean(axis=0)
    u_values = (points - origin) @ tangent_u
    v_values = (points - origin) @ tangent_v
    u_min, u_max = float(u_values.min()), float(u_values.max())
    v_min, v_max = float(v_values.min()), float(v_values.max())
    true_extents = np.array([u_max - u_min, v_max - v_min], dtype=np.float64)
    padded = bool(np.any(true_extents < float(min_display_extent_m)))

    if u_max - u_min < min_display_extent_m:
        midpoint = 0.5 * (u_min + u_max)
        u_min, u_max = midpoint - min_display_extent_m / 2, midpoint + min_display_extent_m / 2
    if v_max - v_min < min_display_extent_m:
        midpoint = 0.5 * (v_min + v_max)
        v_min, v_max = midpoint - min_display_extent_m / 2, midpoint + min_display_extent_m / 2

    plane_origin = origin + normal * float(normal_offset_m)
    corners = np.asarray(
        [
            plane_origin + tangent_u * u_min + tangent_v * v_min,
            plane_origin + tangent_u * u_max + tangent_v * v_min,
            plane_origin + tangent_u * u_max + tangent_v * v_max,
            plane_origin + tangent_u * u_min + tangent_v * v_max,
        ],
        dtype=np.float64,
    )
    return {
        "corners": corners,
        "normal": normal,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
        "true_extents": true_extents,
        "display_extents": np.array([u_max - u_min, v_max - v_min]),
        "display_padded": padded,
    }
