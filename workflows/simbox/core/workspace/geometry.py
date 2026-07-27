"""Pure deterministic geometry for target-centred robot placement."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .models import GeometryCandidate, RobotCollisionLayer, SamplingConfig


GOLDEN_ANGLE_RAD = math.pi * (3.0 - math.sqrt(5.0))


def yaw_to_target_deg(base_xy: Sequence[float], target_xy: Sequence[float]) -> float:
    dx = float(target_xy[0]) - float(base_xy[0])
    dy = float(target_xy[1]) - float(base_xy[1])
    if math.hypot(dx, dy) < 1e-9:
        raise ValueError("target and robot candidate share the same XY")
    return math.degrees(math.atan2(dy, dx))


def yaw_to_align_arm_base_deg(
    base_xy: Sequence[float],
    target_xy: Sequence[float],
    arm_base_xy_m: Sequence[float],
) -> float:
    """Rotate the chassis so the target lies on the selected arm's forward ray."""

    dx = float(target_xy[0]) - float(base_xy[0])
    dy = float(target_xy[1]) - float(base_xy[1])
    radius = math.hypot(dx, dy)
    lateral = float(arm_base_xy_m[1])
    if radius <= abs(lateral) + 1e-9:
        raise ValueError("target radius is too small to align the selected arm base")
    target_bearing = math.atan2(dy, dx)
    target_angle_in_robot = math.asin(lateral / radius)
    return math.degrees(target_bearing - target_angle_in_robot)


def sample_target_annulus(target_xy: Sequence[float], config: SamplingConfig) -> list[GeometryCandidate]:
    """Generate deterministic target-centred candidates.

    V1 preserves the original area-uniform golden-angle sequence.  V2 uses an
    explicit polar grid so radius and approach direction are independent; this
    avoids missing a narrow aisle merely because its useful angle happened to
    be paired with the wrong radius in V1.
    """
    config.validate()
    if config.planner == "target_annulus_v2":
        candidates: list[GeometryCandidate] = []
        assert config.radial_count is not None
        assert config.angular_count is not None
        radial_step = (config.max_radius_m - config.min_radius_m) / (config.radial_count - 1)
        angular_step = 2.0 * math.pi / config.angular_count
        for angle_index in range(config.angular_count):
            angle = angle_index * angular_step
            for radial_index in range(config.radial_count):
                radius = config.min_radius_m + radial_index * radial_step
                world_xy = (
                    float(target_xy[0]) + radius * math.cos(angle),
                    float(target_xy[1]) + radius * math.sin(angle),
                )
                base_yaw = yaw_to_target_deg(world_xy, target_xy)
                for yaw_index, yaw_offset in enumerate(config.yaw_offsets_deg):
                    candidates.append(
                        GeometryCandidate(
                            candidate_id=(
                                f"annulus_v2_a{angle_index:03d}_r{radial_index:03d}_"
                                f"y{yaw_index:02d}"
                            ),
                            world_xy=world_xy,
                            yaw_deg=base_yaw + float(yaw_offset),
                            radius_m=radius,
                            angle_deg=math.degrees(angle),
                            yaw_offset_deg=float(yaw_offset),
                        )
                    )
        return candidates

    min_sq = config.min_radius_m**2
    max_sq = config.max_radius_m**2
    candidates: list[GeometryCandidate] = []
    for index in range(config.candidate_count):
        u = (index + 0.5) / config.candidate_count
        radius = math.sqrt(min_sq + u * (max_sq - min_sq))
        angle = (index * GOLDEN_ANGLE_RAD) % (2.0 * math.pi)
        world_xy = (
            float(target_xy[0]) + radius * math.cos(angle),
            float(target_xy[1]) + radius * math.sin(angle),
        )
        candidates.append(
            GeometryCandidate(
                candidate_id=f"annulus_{index:03d}",
                world_xy=world_xy,
                yaw_deg=yaw_to_target_deg(world_xy, target_xy),
                radius_m=radius,
                angle_deg=math.degrees(angle),
            )
        )
    return candidates


def rectangle_corners(
    center: Sequence[float], size: Sequence[float], yaw_deg: float
) -> tuple[tuple[float, float], ...]:
    half_x, half_y = float(size[0]) / 2.0, float(size[1]) / 2.0
    yaw = math.radians(float(yaw_deg))
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return tuple(
        (
            float(center[0]) + local_x * cos_yaw - local_y * sin_yaw,
            float(center[1]) + local_x * sin_yaw + local_y * cos_yaw,
        )
        for local_x, local_y in ((-half_x, -half_y), (-half_x, half_y), (half_x, half_y), (half_x, -half_y))
    )


def _axes(corners: Sequence[Sequence[float]]) -> Iterable[tuple[float, float]]:
    for index in (0, 1):
        dx = float(corners[index + 1][0]) - float(corners[index][0])
        dy = float(corners[index + 1][1]) - float(corners[index][1])
        norm = math.hypot(dx, dy)
        if norm:
            yield (-dy / norm, dx / norm)


def rectangles_overlap(
    center_a: Sequence[float],
    size_a: Sequence[float],
    yaw_a: float,
    center_b: Sequence[float],
    size_b: Sequence[float],
    yaw_b: float,
    tolerance: float = 1e-6,
) -> bool:
    corners_a = rectangle_corners(center_a, size_a, yaw_a)
    corners_b = rectangle_corners(center_b, size_b, yaw_b)
    for axis in (*_axes(corners_a), *_axes(corners_b)):
        proj_a = [point[0] * axis[0] + point[1] * axis[1] for point in corners_a]
        proj_b = [point[0] * axis[0] + point[1] * axis[1] for point in corners_b]
        if max(proj_a) <= min(proj_b) + tolerance or max(proj_b) <= min(proj_a) + tolerance:
            return False
    return True

def inside_rect(
    candidate: GeometryCandidate,
    footprint: Sequence[float],
    region_center: Sequence[float],
    region_size: Sequence[float],
    region_yaw_deg: float = 0.0,
) -> bool:
    """Check if candidate's footprint lies entirely within a rotated rectangular region."""
    corners = rectangle_corners(candidate.world_xy, footprint, candidate.yaw_deg)
    yaw = math.radians(region_yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    half_w = float(region_size[0]) / 2.0
    half_d = float(region_size[1]) / 2.0
    rx, ry = float(region_center[0]), float(region_center[1])
    for cx, cy in corners:
        dx = cx - rx
        dy = cy - ry
        lx = dx * cos_y + dy * sin_y
        ly = -dx * sin_y + dy * cos_y
        if not (-half_w <= lx <= half_w and -half_d <= ly <= half_d):
            return False
    return True

def table_edge_centers(
    table_center: Sequence[float],
    table_size: Sequence[float],
    table_yaw_deg: float = 0.0,
) -> list[tuple[str, tuple[float, float], float]]:
    """Return (edge_name, center_xy, inward_yaw_deg) for the four edges of a table fixture."""
    tx, ty = float(table_center[0]), float(table_center[1])
    tw, td = float(table_size[0]), float(table_size[1])
    half_w, half_d = tw / 2.0, td / 2.0

    if abs(table_yaw_deg) < 1e-9:
        return [
            ("north", (tx, ty + half_d), -90.0),
            ("south", (tx, ty - half_d), 90.0),
            ("east", (tx + half_w, ty), 180.0),
            ("west", (tx - half_w, ty), 0.0),
        ]

    yaw = math.radians(table_yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    local_edges = [
        ("north", (0.0, half_d), -90.0),
        ("south", (0.0, -half_d), 90.0),
        ("east", (half_w, 0.0), 180.0),
        ("west", (-half_w, 0.0), 0.0),
    ]
    result: list[tuple[str, tuple[float, float], float]] = []
    for name, (lx, ly), base_yaw in local_edges:
        wx = tx + lx * cos_y - ly * sin_y
        wy = ty + lx * sin_y + ly * cos_y
        inward = (base_yaw + table_yaw_deg) % 360.0
        if inward > 180.0:
            inward -= 360.0
        result.append((name, (wx, wy), inward))
    return result

def _edge_endpoints(
    table_center: tuple[float, float],
    table_size: tuple[float, float],
    table_yaw_deg: float,
) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Compute world-space start/end points for each table edge."""
    tx, ty = table_center
    tw, td = table_size
    half_w, half_d = tw / 2.0, td / 2.0

    if abs(table_yaw_deg) < 1e-9:
        return {
            "north": ((tx - half_w, ty + half_d), (tx + half_w, ty + half_d)),
            "south": ((tx - half_w, ty - half_d), (tx + half_w, ty - half_d)),
            "east": ((tx + half_w, ty - half_d), (tx + half_w, ty + half_d)),
            "west": ((tx - half_w, ty - half_d), (tx - half_w, ty + half_d)),
        }

    yaw = math.radians(table_yaw_deg)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    local = {
        "north": ((-half_w, half_d), (half_w, half_d)),
        "south": ((-half_w, -half_d), (half_w, -half_d)),
        "east": ((half_w, -half_d), (half_w, half_d)),
        "west": ((-half_w, -half_d), (-half_w, half_d)),
    }
    result: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
    for name, ((sx, sy), (ex, ey)) in local.items():
        wsx = tx + sx * cos_y - sy * sin_y
        wsy = ty + sx * sin_y + sy * cos_y
        wex = tx + ex * cos_y - ey * sin_y
        wey = ty + ex * sin_y + ey * cos_y
        result[name] = ((wsx, wsy), (wex, wey))
    return result

def sample_table_edge(
    edge_name: str,
    edge_start: tuple[float, float],
    edge_end: tuple[float, float],
    inward_yaw_deg: float,
    footprint: tuple[float, float],
    count: int,
) -> list[GeometryCandidate]:
    """Generate candidates spaced along one table edge, inset onto the table surface."""
    edge_dx = edge_end[0] - edge_start[0]
    edge_dy = edge_end[1] - edge_start[1]
    edge_length = math.hypot(edge_dx, edge_dy)

    fw, fd = footprint
    margin = fw / 2.0
    usable = edge_length - fw

    if usable <= 0.0:
        return []

    inward_rad = math.radians(inward_yaw_deg)
    inset_x = (fd / 2.0) * math.cos(inward_rad)
    inset_y = (fd / 2.0) * math.sin(inward_rad)

    candidates: list[GeometryCandidate] = []
    for i in range(count):
        t = (i + 0.5) / count
        dist = margin + t * usable
        frac = dist / edge_length
        ex = edge_start[0] + frac * edge_dx
        ey = edge_start[1] + frac * edge_dy
        candidates.append(
            GeometryCandidate(
                candidate_id=f"{edge_name}_{i:03d}",
                world_xy=(ex + inset_x, ey + inset_y),
                yaw_deg=inward_yaw_deg,
                radius_m=0.0,
                angle_deg=0.0,
            )
        )
    return candidates

def fixture_collision_rect(fixture: Mapping[str, Any]) -> tuple[list[float], list[float], float] | None:
    name = str(fixture.get("name", ""))
    translation = fixture.get("translation")
    size = fixture.get("size")
    if not isinstance(translation, Sequence) or len(translation) < 2 or not isinstance(size, Sequence):
        return None
    if name == "floor" or "__support_plane_" in name or not fixture.get(
        "collision_enabled", fixture.get("collision", True)
    ):
        return None
    thickness = float(fixture.get("collision_thickness", 0.02))
    if name.startswith(("wall_north", "wall_south")):
        return [float(translation[0]), float(translation[1])], [float(size[0]), thickness], 0.0
    if name.startswith(("wall_east", "wall_west")):
        return [float(translation[0]), float(translation[1])], [thickness, float(size[0])], 0.0
    if len(size) < 2:
        return None
    euler = fixture.get("euler", fixture.get("rotation", [0.0, 0.0, 0.0]))
    yaw = float(euler[2]) if isinstance(euler, Sequence) and len(euler) >= 3 else 0.0
    return [float(translation[0]), float(translation[1])], [float(size[0]), float(size[1])], yaw


def colliding_fixture(
    candidate: GeometryCandidate, footprint: Sequence[float], fixtures: Sequence[Mapping[str, Any]]
) -> str | None:
    for fixture in fixtures:
        rect = fixture_collision_rect(fixture)
        if rect is None:
            continue
        center, size, yaw = rect
        if rectangles_overlap(candidate.world_xy, footprint, candidate.yaw_deg, center, size, yaw):
            return str(fixture.get("name", "unknown_fixture"))
    return None


def fixture_vertical_range(fixture: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return a conservative world-Z range for a collision fixture."""

    name = str(fixture.get("name", ""))
    translation = fixture.get("translation")
    if not isinstance(translation, Sequence) or len(translation) < 3:
        return None
    if name.startswith("wall_"):
        size = fixture.get("size")
        if isinstance(size, Sequence) and len(size) >= 2:
            half_height = float(size[1]) / 2.0
            return float(translation[2]) - half_height, float(translation[2]) + half_height
    size_xyz = fixture.get("size_xyz")
    if isinstance(size_xyz, Sequence) and len(size_xyz) >= 3:
        half_height = float(size_xyz[2]) / 2.0
        return float(translation[2]) - half_height, float(translation[2]) + half_height
    return None


def colliding_fixture_layer(
    candidate: GeometryCandidate,
    layer: RobotCollisionLayer,
    fixtures: Sequence[Mapping[str, Any]],
) -> str | None:
    """Check one measured robot home-pose layer against height-overlapping fixtures."""

    yaw = math.radians(candidate.yaw_deg)
    local_x, local_y = layer.center_xy_m
    center = (
        candidate.world_xy[0] + local_x * math.cos(yaw) - local_y * math.sin(yaw),
        candidate.world_xy[1] + local_x * math.sin(yaw) + local_y * math.cos(yaw),
    )
    for fixture in fixtures:
        rect = fixture_collision_rect(fixture)
        if rect is None:
            continue
        z_range = fixture_vertical_range(fixture)
        if z_range is None:
            # Unknown vertical size is not guessed here. GeometryObject inputs
            # are required to have size_xyz by the planner; PlaneObject support
            # surfaces are deliberately excluded from collision rectangles.
            continue
        if z_range[1] <= layer.min_z_m or layer.max_z_m <= z_range[0]:
            continue
        fixture_center, fixture_size, fixture_yaw = rect
        if rectangles_overlap(
            center,
            layer.size_xy_m,
            candidate.yaw_deg,
            fixture_center,
            fixture_size,
            fixture_yaw,
        ):
            return str(fixture.get("name", "unknown_fixture"))
    return None


def inside_floor(candidate: GeometryCandidate, footprint: Sequence[float], floor: Mapping[str, Any]) -> bool:
    translation = floor.get("translation")
    size = floor.get("size")
    if (
        not isinstance(translation, Sequence)
        or len(translation) < 2
        or not isinstance(size, Sequence)
        or len(size) < 2
    ):
        return True
    min_x = float(translation[0]) - float(size[0]) / 2.0
    max_x = float(translation[0]) + float(size[0]) / 2.0
    min_y = float(translation[1]) - float(size[1]) / 2.0
    max_y = float(translation[1]) + float(size[1]) / 2.0
    return all(
        min_x <= point[0] <= max_x and min_y <= point[1] <= max_y
        for point in rectangle_corners(candidate.world_xy, footprint, candidate.yaw_deg)
    )
