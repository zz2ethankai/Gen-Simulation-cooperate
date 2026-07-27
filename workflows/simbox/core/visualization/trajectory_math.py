"""Pure NumPy helpers for CuRobo trajectory visualization."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def uniform_sample_indices(length: int, max_count: int) -> np.ndarray:
    """Uniformly sample a sequence while always retaining unique endpoints."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    if length == 0:
        return np.empty((0,), dtype=np.int64)
    if length <= max_count:
        return np.arange(length, dtype=np.int64)
    indices = np.rint(np.linspace(0, length - 1, num=max_count)).astype(np.int64)
    indices[0] = 0
    indices[-1] = length - 1
    return np.unique(indices)


def distance_sample_indices(
    points: np.ndarray, min_spacing_m: float, max_count: int
) -> np.ndarray:
    """Sample ordered 3D points with a minimum adjacent center distance.

    The first and last points are retained. If the last point is too close to
    the latest interior sample, interior samples are removed until the spacing
    is satisfied. A trajectory shorter than ``min_spacing_m`` necessarily
    retains its two endpoints even though that single interval is shorter.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape Nx3, got {points.shape}")
    if min_spacing_m < 0.0:
        raise ValueError("min_spacing_m must be non-negative")
    if max_count <= 0:
        raise ValueError("max_count must be positive")
    length = points.shape[0]
    if length <= 1 or min_spacing_m == 0.0:
        return uniform_sample_indices(length, max_count)

    selected = [0]
    for index in range(1, length - 1):
        if np.linalg.norm(points[index] - points[selected[-1]]) >= min_spacing_m:
            selected.append(index)

    endpoint = length - 1
    while (
        len(selected) > 1
        and np.linalg.norm(points[endpoint] - points[selected[-1]]) < min_spacing_m
    ):
        selected.pop()
    if selected[-1] != endpoint:
        selected.append(endpoint)

    selected_array = np.asarray(selected, dtype=np.int64)
    if len(selected_array) > max_count:
        selected_array = selected_array[
            uniform_sample_indices(len(selected_array), max_count)
        ]
    return selected_array


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Transform Nx3 column-vector points with a conventional 4x4 matrix."""
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape Nx3, got {points.shape}")
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape 4x4, got {transform.shape}")
    return points @ transform[:3, :3].T + transform[:3, 3]


def valid_sphere_arrays(spheres: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    """Extract centers and positive physical radii from CuRobo Sphere objects."""
    centers: list[list[float]] = []
    radii: list[float] = []
    for sphere in spheres:
        radius = float(getattr(sphere, "radius"))
        if radius <= 0.0:
            continue
        pose = getattr(sphere, "pose")
        centers.append([float(pose[0]), float(pose[1]), float(pose[2])])
        radii.append(radius)
    return np.asarray(centers, dtype=np.float64).reshape(-1, 3), np.asarray(
        radii, dtype=np.float64
    )
