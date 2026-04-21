"""Shared helpers for the split Nav2 runtime."""

from __future__ import annotations

import math
import time


def safe_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in str(value).strip())
    return cleaned or "robot"


def angle_diff_rad(target: float, current: float) -> float:
    return math.atan2(math.sin(float(target) - float(current)), math.cos(float(target) - float(current)))


def yaw_from_wxyz(q_wxyz) -> float:
    w = float(q_wxyz[0])
    x = float(q_wxyz[1])
    y = float(q_wxyz[2])
    z = float(q_wxyz[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = float(bx) - float(ax)
    aby = float(by) - float(ay)
    apx = float(px) - float(ax)
    apy = float(py) - float(ay)
    ab2 = abx * abx + aby * aby
    if ab2 <= 1.0e-12:
        return math.hypot(apx, apy)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))
    closest_x = float(ax) + t * abx
    closest_y = float(ay) + t * aby
    return math.hypot(float(px) - closest_x, float(py) - closest_y)


def footprint_inscribed_radius(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    radius = float("inf")
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        radius = min(
            radius,
            distance_point_to_segment(
                0.0,
                0.0,
                float(point[0]),
                float(point[1]),
                float(next_point[0]),
                float(next_point[1]),
            ),
        )
    return 0.0 if not math.isfinite(radius) else float(radius)


def time_monotonic() -> float:
    return time.monotonic()
