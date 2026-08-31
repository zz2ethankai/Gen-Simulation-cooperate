"""ROS-free static navigation primitives for SimBox.

The planner is intentionally small and deterministic: A* searches an
8-connected center-cell occupancy grid, and a
waypoint P controller emits one body-frame twist per physics cycle.  Dynamic
approach sampling is reused as plain-data geometry; no external action, TF,
clock, or controller process is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import importlib.util
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from .navigation_geometry import (
        ApproachConfig,
        build_armbase_target_context,
        check_footprint_static_collision,
        check_path_static_collision,
        choose_best_reachable_candidate,
        load_static_map,
        parse_approach_config,
        resolve_approach_footprint_padding_m,
        resolve_mobile_footprint_points as _resolve_legacy_footprint_points,
        sample_approach_candidates,
        sort_candidates_for_preflight,
        wrap_to_pi,
    )
except ImportError:
    # Focused tests load this file directly, outside the ``core.skills`` package.
    geometry_path = Path(__file__).with_name("navigation_geometry.py")
    geometry_spec = importlib.util.spec_from_file_location("simbox_navigation_geometry", geometry_path)
    if geometry_spec is None or geometry_spec.loader is None:
        raise ImportError(f"Unable to load local navigation geometry from {geometry_path}")
    geometry_module = importlib.util.module_from_spec(geometry_spec)
    sys.modules[geometry_spec.name] = geometry_module
    geometry_spec.loader.exec_module(geometry_module)
    ApproachConfig = geometry_module.ApproachConfig
    build_armbase_target_context = geometry_module.build_armbase_target_context
    check_footprint_static_collision = geometry_module.check_footprint_static_collision
    check_path_static_collision = geometry_module.check_path_static_collision
    choose_best_reachable_candidate = geometry_module.choose_best_reachable_candidate
    load_static_map = geometry_module.load_static_map
    parse_approach_config = geometry_module.parse_approach_config
    resolve_approach_footprint_padding_m = geometry_module.resolve_approach_footprint_padding_m
    _resolve_legacy_footprint_points = geometry_module.resolve_mobile_footprint_points
    sample_approach_candidates = geometry_module.sample_approach_candidates
    sort_candidates_for_preflight = geometry_module.sort_candidates_for_preflight
    wrap_to_pi = geometry_module.wrap_to_pi


def resolve_footprint_points(base_cfg: dict[str, Any]) -> list[list[float]]:
    local_cfg = base_cfg.get("local_navigation", {}) if isinstance(base_cfg, dict) else {}
    points = local_cfg.get("footprint_points") if isinstance(local_cfg, dict) else None
    if isinstance(points, (list, tuple)) and len(points) >= 3:
        return [[float(x), float(y)] for x, y in points]
    return _resolve_legacy_footprint_points(base_cfg)


@dataclass
class NavigationPlan:
    path: list[dict[str, float]]
    goal: tuple[float, float, float]
    collision_check: dict[str, Any]


class GridAStarPlanner:
    """Center-cell A* over a static map image, matching the example skill."""

    def __init__(self, *, resolution: float = 0.05, safety_distance_m: float = 0.35, proximity_weight: float = 2.0):
        self.resolution = float(resolution)
        self.safety_distance_m = max(float(safety_distance_m), 0.0)
        self.proximity_weight = max(float(proximity_weight), 0.0)
        self.static_map: dict[str, Any] | None = None
        self._grid: np.ndarray | None = None
        self._distance_field: np.ndarray | None = None
        self._origin = np.zeros(2, dtype=np.float32)
        self._footprint_points: list[list[float]] = []
        self._footprint_padding_m = 0.0

    def set_static_map(self, static_map: dict[str, Any], *, footprint_points: list[list[float]], footprint_padding_m: float = 0.0):
        image = np.asarray(static_map["image"])
        if image.ndim != 2 or float(static_map["resolution"]) <= 0.0:
            raise ValueError("static map must contain a 2-D image and positive resolution")
        self.static_map = static_map
        self.resolution = float(static_map["resolution"])
        self._origin = np.asarray(static_map["origin"][:2], dtype=np.float32)
        self._footprint_points = [[float(x), float(y)] for x, y in footprint_points]
        self._footprint_padding_m = max(float(footprint_padding_m), 0.0)
        # The map image uses the conventional top-left row origin.  A* uses
        # bottom-left rows so world y increases with the row index.
        occupied = np.flipud(image) < 250
        # The reference skill checks only the robot center against occupancy.
        # Keep footprint arguments for API/debug compatibility, but do not
        # inflate the map or reject cells based on the mobile-base volume.
        self._grid = occupied.astype(bool)
        self._distance_field = self._compute_distance_field(self._grid)

    @staticmethod
    def _compute_distance_field(grid: np.ndarray) -> np.ndarray | None:
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError:
            return None
        return distance_transform_edt(np.logical_not(grid))

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor((float(y) - float(self._origin[1])) / self.resolution)), int(math.floor((float(x) - float(self._origin[0])) / self.resolution))

    def _grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        return float(self._origin[0] + (col + 0.5) * self.resolution), float(self._origin[1] + (row + 0.5) * self.resolution)

    def _valid(self, node: tuple[int, int]) -> bool:
        return self._grid is not None and 0 <= node[0] < self._grid.shape[0] and 0 <= node[1] < self._grid.shape[1] and not bool(self._grid[node])

    def _nearest_valid(self, node: tuple[int, int], max_radius: int = 64) -> tuple[int, int] | None:
        if self._valid(node):
            return node
        for radius in range(1, max(int(max_radius), 1) + 1):
            candidates = []
            for dr in range(-radius, radius + 1):
                for dc in (-radius, radius):
                    candidates.append((node[0] + dr, node[1] + dc))
            for dc in range(-radius + 1, radius):
                for dr in (-radius, radius):
                    candidates.append((node[0] + dr, node[1] + dc))
            valid = [candidate for candidate in candidates if self._valid(candidate)]
            if valid:
                return min(valid, key=lambda candidate: (candidate[0] - node[0]) ** 2 + (candidate[1] - node[1]) ** 2)
        return None

    def plan(self, start_xy: tuple[float, float], goal_xy: tuple[float, float]) -> list[tuple[float, float]] | None:
        if self._grid is None:
            return [tuple(map(float, start_xy)), tuple(map(float, goal_xy))]
        requested_start = self._world_to_grid(*start_xy)
        requested_goal = self._world_to_grid(*goal_xy)
        start = self._nearest_valid(requested_start)
        goal = self._nearest_valid(requested_goal)
        if start is None or goal is None:
            return None
        frontier: list[tuple[float, int, tuple[int, int]]] = [(0.0, 0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost_so_far = {start: 0.0}
        counter = 0
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while frontier:
            _, _, current = heapq.heappop(frontier)
            if current == goal:
                break
            for dr, dc in neighbors:
                nxt = (current[0] + dr, current[1] + dc)
                if not self._valid(nxt):
                    continue
                if dr and dc and (not self._valid((current[0] + dr, current[1])) or not self._valid((current[0], current[1] + dc))):
                    continue
                step_cost = math.sqrt(2.0) if dr and dc else 1.0
                clearance_penalty = 0.0
                if self._distance_field is not None and self.safety_distance_m > 0.0:
                    clearance_m = float(self._distance_field[nxt]) * self.resolution
                    if clearance_m < self.safety_distance_m:
                        clearance_penalty = self.proximity_weight * (self.safety_distance_m - clearance_m) / self.resolution
                new_cost = cost_so_far[current] + step_cost + clearance_penalty
                if new_cost >= cost_so_far.get(nxt, float("inf")):
                    continue
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
                counter += 1
                heapq.heappush(frontier, (new_cost + heuristic, counter, nxt))
        if goal not in cost_so_far:
            return None
        nodes = [goal]
        while nodes[-1] != start:
            nodes.append(came_from[nodes[-1]])
        nodes.reverse()
        points = [self._grid_to_world(row, col) for row, col in nodes]
        if start != requested_start:
            points.insert(0, tuple(map(float, start_xy)))
        else:
            points[0] = tuple(map(float, start_xy))
        if goal != requested_goal:
            points.append(tuple(map(float, goal_xy)))
        else:
            points[-1] = tuple(map(float, goal_xy))
        # Keep every grid waypoint so the controller follows the A* route
        # rather than cutting directly across occupied center cells.
        return points

    def plan_to_goals(
        self,
        start_xy: tuple[float, float],
        goal_xy_list: list[tuple[float, float]],
        *,
        max_solutions: int = 10,
    ) -> dict[int, list[tuple[float, float]]]:
        """Run one A* search and stop after reaching enough goal cells."""
        if not goal_xy_list or max_solutions <= 0:
            return {}
        if self._grid is None:
            return {
                index: [tuple(map(float, start_xy)), (float(goal_xy[0]), float(goal_xy[1]))]
                for index, goal_xy in enumerate(goal_xy_list[:max_solutions])
            }
        requested_start = self._world_to_grid(*start_xy)
        start = self._nearest_valid(requested_start)
        if start is None:
            return {}
        # Approach candidates carry (x, y, yaw), while the occupancy grid is
        # purely planar.  Keep yaw for the final navigation goal, but only
        # project x/y into the A* grid.
        requested_goals = [self._world_to_grid(float(goal_xy[0]), float(goal_xy[1])) for goal_xy in goal_xy_list]
        goal_nodes: dict[tuple[int, int], list[int]] = {}
        for index, requested_goal in enumerate(requested_goals):
            goal = self._nearest_valid(requested_goal)
            if goal is not None:
                goal_nodes.setdefault(goal, []).append(index)
        if not goal_nodes:
            return {}
        frontier: list[tuple[float, int, tuple[int, int]]] = [(0.0, 0, start)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost_so_far = {start: 0.0}
        counter = 0
        found: dict[int, list[tuple[float, float]]] = {}
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while frontier and len(found) < min(int(max_solutions), len(goal_xy_list)):
            _, _, current = heapq.heappop(frontier)
            if current in goal_nodes:
                nodes = [current]
                while nodes[-1] != start:
                    nodes.append(came_from[nodes[-1]])
                nodes.reverse()
                grid_points = [self._grid_to_world(row, col) for row, col in nodes]
                for index in goal_nodes[current]:
                    points = list(grid_points)
                    requested_goal = goal_xy_list[index]
                    if start != requested_start:
                        points.insert(0, tuple(map(float, start_xy)))
                    else:
                        points[0] = tuple(map(float, start_xy))
                    if current != requested_goals[index]:
                        points.append((float(requested_goal[0]), float(requested_goal[1])))
                    else:
                        points[-1] = (float(requested_goal[0]), float(requested_goal[1]))
                    found[index] = points
                    if len(found) >= int(max_solutions):
                        break
                if len(found) >= int(max_solutions):
                    break
            for dr, dc in neighbors:
                nxt = (current[0] + dr, current[1] + dc)
                if not self._valid(nxt):
                    continue
                if dr and dc and (not self._valid((current[0] + dr, current[1])) or not self._valid((current[0], current[1] + dc))):
                    continue
                step_cost = math.sqrt(2.0) if dr and dc else 1.0
                clearance_penalty = 0.0
                if self._distance_field is not None and self.safety_distance_m > 0.0:
                    clearance_m = float(self._distance_field[nxt]) * self.resolution
                    if clearance_m < self.safety_distance_m:
                        clearance_penalty = self.proximity_weight * (self.safety_distance_m - clearance_m) / self.resolution
                new_cost = cost_so_far[current] + step_cost + clearance_penalty
                if new_cost >= cost_so_far.get(nxt, float("inf")):
                    continue
                cost_so_far[nxt] = new_cost
                came_from[nxt] = current
                heuristic = min(math.hypot(goal[0] - nxt[0], goal[1] - nxt[1]) for goal in goal_nodes)
                counter += 1
                heapq.heappush(frontier, (new_cost + heuristic, counter, nxt))
        return found

    @staticmethod
    def _simplify(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) <= 2:
            return points
        simplified = [points[0]]
        previous_heading = None
        for index in range(1, len(points)):
            dx = points[index][0] - points[index - 1][0]
            dy = points[index][1] - points[index - 1][1]
            heading = math.atan2(dy, dx)
            if previous_heading is not None and abs(wrap_to_pi(heading - previous_heading)) > 0.08:
                simplified.append(points[index - 1])
            previous_heading = heading
        simplified.append(points[-1])
        return simplified


class WaypointController:
    """Waypoint P controller that always uses the current pose yaw."""

    def __init__(self, *, max_linear_velocity: float = 0.35, max_angular_velocity: float = 0.8, waypoint_tolerance_m: float = 0.25, position_tolerance_m: float = 0.1, yaw_tolerance_rad: float = 0.1, rotate_first_error_rad: float = 0.2, linear_gain: float = 2.0, angular_gain: float = 2.0):
        self.max_linear_velocity = max(float(max_linear_velocity), 0.0)
        self.max_angular_velocity = max(float(max_angular_velocity), 0.0)
        self.waypoint_tolerance_m = max(float(waypoint_tolerance_m), 0.0)
        self.position_tolerance_m = max(float(position_tolerance_m), 0.0)
        self.yaw_tolerance_rad = max(float(yaw_tolerance_rad), 0.0)
        self.rotate_first_error_rad = max(float(rotate_first_error_rad), 0.0)
        self.linear_gain = max(float(linear_gain), 0.0)
        self.angular_gain = max(float(angular_gain), 0.0)
        self.path: list[dict[str, float]] = []
        self.waypoint_index = 0

    def reset(self, path: list[dict[str, float]]) -> None:
        self.path = list(path)
        self.waypoint_index = 0

    def command(self, pose: tuple[float, float, float], goal: tuple[float, float, float]) -> tuple[float, float, float, bool, dict[str, Any]]:
        if not self.path:
            return 0.0, 0.0, 0.0, False, {"reason": "empty_path"}
        x, y, yaw = [float(v) for v in pose]
        goal_x, goal_y, goal_yaw = [float(v) for v in goal]
        distance = math.hypot(goal_x - x, goal_y - y)
        yaw_error = wrap_to_pi(goal_yaw - yaw)
        if distance <= self.position_tolerance_m and abs(yaw_error) <= self.yaw_tolerance_rad:
            return 0.0, 0.0, 0.0, True, {"distance_to_goal": distance, "yaw_error": yaw_error}
        while self.waypoint_index < len(self.path) - 1:
            target = self.path[self.waypoint_index]
            if math.hypot(float(target["x"]) - x, float(target["y"]) - y) <= self.waypoint_tolerance_m:
                self.waypoint_index += 1
            else:
                break
        target = self.path[self.waypoint_index]
        dx, dy = float(target["x"]) - x, float(target["y"]) - y
        target_distance = math.hypot(dx, dy)
        if target_distance <= 1.0e-6:
            world_vx = world_vy = 0.0
        else:
            speed = min(self.max_linear_velocity, self.linear_gain * target_distance)
            world_vx, world_vy = speed * dx / target_distance, speed * dy / target_distance
        if distance > self.position_tolerance_m and abs(yaw_error) > self.rotate_first_error_rad:
            world_vx = world_vy = 0.0
        body_vx = world_vx * math.cos(yaw) + world_vy * math.sin(yaw)
        body_vy = -world_vx * math.sin(yaw) + world_vy * math.cos(yaw)
        wz = float(np.clip(self.angular_gain * yaw_error, -self.max_angular_velocity, self.max_angular_velocity))
        return body_vx, body_vy, wz, False, {"waypoint_index": self.waypoint_index, "waypoint_count": len(self.path), "distance_to_goal": distance, "yaw_error": yaw_error}


ControllerResult = tuple[float, float, float, bool, dict[str, Any]]


class PhasedWaypointController:
    """Track each segment with turn, lateral-align, and forward phases."""

    def __init__(
        self, *, goal: tuple[float, float, float], terminal_approach_distance_m: float,
        max_linear_velocity: float, terminal_max_linear_velocity: float, max_angular_velocity: float,
        waypoint_tolerance_m: float, position_tolerance_m: float, yaw_tolerance_rad: float,
        linear_gain: float, angular_gain: float, max_lateral_velocity: float = 0.20,
        lateral_gain: float = 2.0, lateral_alignment_enter_m: float = 0.08,
        lateral_alignment_exit_m: float = 0.04, turn_enter_heading_error_rad: float = 0.20,
        turn_exit_heading_error_rad: float = 0.10,
    ):
        self.goal = tuple(float(value) for value in goal)
        self.terminal_approach_distance_m = float(terminal_approach_distance_m)
        self.max_linear_velocity = max(float(max_linear_velocity), 0.0)
        self.terminal_max_linear_velocity = min(
            max(float(terminal_max_linear_velocity), 0.0),
            self.max_linear_velocity,
        )
        self.max_angular_velocity = max(float(max_angular_velocity), 0.0)
        self.waypoint_tolerance_m = max(float(waypoint_tolerance_m), 0.0)
        self.position_tolerance_m = max(float(position_tolerance_m), 0.0)
        self.yaw_tolerance_rad = max(float(yaw_tolerance_rad), 0.0)
        self.linear_gain = max(float(linear_gain), 0.0)
        self.angular_gain = max(float(angular_gain), 0.0)
        self.max_lateral_velocity = max(float(max_lateral_velocity), 0.0)
        self.lateral_gain = max(float(lateral_gain), 0.0)
        self.lateral_alignment_enter_m = max(float(lateral_alignment_enter_m), 0.0)
        self.lateral_alignment_exit_m = max(float(lateral_alignment_exit_m), 0.0)
        self.turn_enter_heading_error_rad = max(float(turn_enter_heading_error_rad), 0.0)
        self.turn_exit_heading_error_rad = max(float(turn_exit_heading_error_rad), 0.0)
        if (
            not all(math.isfinite(value) for value in self.goal)
            or not math.isfinite(self.terminal_approach_distance_m)
            or self.terminal_approach_distance_m <= 0.0
            or self.turn_exit_heading_error_rad > self.turn_enter_heading_error_rad
            or self.lateral_alignment_exit_m > self.lateral_alignment_enter_m
        ):
            raise ValueError("Phased waypoint goal and motion tolerances are invalid")
        self.path: list[dict[str, float]] = []
        self.waypoint_index = 0
        self.pre_goal_index = 0
        self.phase = "track_path"
        self.motion_mode = "turn_to_waypoint"
        self.translation_target: tuple[float, float, float] | None = None
        self.heading_error_to_waypoint = 0.0
        self.lateral_error_to_path = 0.0
        self.along_track_error = 0.0

    def reset(self, path: list[dict[str, float]]) -> None:
        self.path = [
            {key: float(waypoint[key]) for key in ("x", "y", "yaw")}
            for waypoint in path
        ]
        if not self.path:
            raise ValueError("Phased waypoint path must not be empty")
        goal_x, goal_y, goal_yaw = self.goal
        pre_goal = (
            goal_x - self.terminal_approach_distance_m * math.cos(goal_yaw),
            goal_y - self.terminal_approach_distance_m * math.sin(goal_yaw),
        )
        self.pre_goal_index = min(
            range(len(self.path)),
            key=lambda index: math.hypot(
                self.path[index]["x"] - pre_goal[0],
                self.path[index]["y"] - pre_goal[1],
            ),
        )
        self.waypoint_index = 0
        self.phase = "track_path"
        self.motion_mode = "turn_to_waypoint"
        self.translation_target = None
        self.heading_error_to_waypoint = 0.0
        self.lateral_error_to_path = 0.0
        self.along_track_error = 0.0

    def _segment_heading(self, index: int) -> float:
        target = self.path[index]
        if index == 0:
            return target["yaw"]
        start = self.path[index - 1]
        dx, dy = target["x"] - start["x"], target["y"] - start["y"]
        return target["yaw"] if math.hypot(dx, dy) <= 1.0e-9 else math.atan2(dy, dx)

    def _passed_waypoint(self, pose: tuple[float, float, float], index: int, tolerance: float) -> bool:
        x, y, _ = pose
        target = self.path[index]
        heading = self._segment_heading(index)
        dx, dy = target["x"] - x, target["y"] - y
        along = math.cos(heading) * dx + math.sin(heading) * dy
        lateral = -math.sin(heading) * dx + math.cos(heading) * dy
        return along <= 0.0 and abs(lateral) <= tolerance

    def _turn_command(self, yaw_error: float) -> tuple[float, float, float]:
        wz = np.clip(
            self.angular_gain * yaw_error,
            -self.max_angular_velocity,
            self.max_angular_velocity,
        )
        return 0.0, 0.0, float(wz)

    def _translation_command(
        self,
        pose: tuple[float, float, float],
        target: tuple[float, float],
        heading: float,
        *,
        max_velocity: float | None = None,
    ) -> tuple[float, float, float]:
        x, y, yaw = pose
        translation_target = (target[0], target[1], heading)
        if self.translation_target != translation_target:
            self.translation_target = translation_target
            self.motion_mode = "turn_to_waypoint"
        dx, dy = target[0] - x, target[1] - y
        cos_heading, sin_heading = math.cos(heading), math.sin(heading)
        self.along_track_error = cos_heading * dx + sin_heading * dy
        self.lateral_error_to_path = -sin_heading * dx + cos_heading * dy
        self.heading_error_to_waypoint = wrap_to_pi(heading - yaw)
        if self.motion_mode == "turn_to_waypoint":
            if abs(self.heading_error_to_waypoint) > self.turn_exit_heading_error_rad:
                return self._turn_command(self.heading_error_to_waypoint)
            self.motion_mode = "lateral_align"
        elif abs(self.heading_error_to_waypoint) > self.turn_enter_heading_error_rad:
            self.motion_mode = "turn_to_waypoint"
            return self._turn_command(self.heading_error_to_waypoint)
        if self.motion_mode == "lateral_align":
            if abs(self.lateral_error_to_path) > self.lateral_alignment_exit_m:
                velocity = np.clip(
                    self.lateral_gain * self.lateral_error_to_path,
                    -self.max_lateral_velocity,
                    self.max_lateral_velocity,
                )
                return 0.0, float(velocity), 0.0
            self.motion_mode = "walk_straight"
        elif abs(self.lateral_error_to_path) > self.lateral_alignment_enter_m:
            self.motion_mode = "lateral_align"
            velocity = np.clip(
                self.lateral_gain * self.lateral_error_to_path,
                -self.max_lateral_velocity,
                self.max_lateral_velocity,
            )
            return 0.0, float(velocity), 0.0
        velocity_limit = self.max_linear_velocity if max_velocity is None else max_velocity
        return min(velocity_limit, self.linear_gain * max(self.along_track_error, 0.0)), 0.0, 0.0

    def _result(self, command: tuple[float, float, float], distance: float, yaw_error: float) -> ControllerResult:
        debug = {
            "phase": self.phase,
            "motion_mode": self.motion_mode,
            "heading_error_to_waypoint": self.heading_error_to_waypoint,
            "lateral_error_to_path": self.lateral_error_to_path,
            "along_track_error": self.along_track_error,
            "waypoint_index": self.waypoint_index,
            "waypoint_count": len(self.path),
            "distance_to_goal": distance,
            "yaw_error": yaw_error,
        }
        return *command, False, debug

    def command(self, pose: tuple[float, float, float], goal: tuple[float, float, float]) -> ControllerResult:
        if not self.path:
            return 0.0, 0.0, 0.0, False, {"reason": "empty_path"}
        x, y, yaw = (float(value) for value in pose)
        goal_x, goal_y, goal_yaw = (float(value) for value in goal)
        distance = math.hypot(goal_x - x, goal_y - y)
        yaw_error = wrap_to_pi(goal_yaw - yaw)
        if distance <= self.position_tolerance_m and abs(yaw_error) <= self.yaw_tolerance_rad:
            self.phase = "done"
            _, _, _, _, debug = self._result((0.0, 0.0, 0.0), distance, yaw_error)
            return 0.0, 0.0, 0.0, True, debug

        pose_values = (x, y, yaw)
        for _ in range(3):
            if self.phase == "track_path":
                while self.waypoint_index < self.pre_goal_index:
                    target = self.path[self.waypoint_index]
                    target_distance = math.hypot(target["x"] - x, target["y"] - y)
                    if target_distance > self.waypoint_tolerance_m and not self._passed_waypoint(
                        pose_values, self.waypoint_index, self.waypoint_tolerance_m
                    ):
                        break
                    self.waypoint_index += 1
                target = self.path[self.waypoint_index]
                target_distance = math.hypot(target["x"] - x, target["y"] - y)
                if self.waypoint_index == self.pre_goal_index and (
                    target_distance <= self.waypoint_tolerance_m
                    or self._passed_waypoint(
                        pose_values, self.waypoint_index, self.waypoint_tolerance_m
                    )
                ):
                    self.phase = "align_final_approach"
                    continue
                command = self._translation_command(
                    pose_values,
                    (target["x"], target["y"]),
                    self._segment_heading(self.waypoint_index),
                )
                return self._result(command, distance, yaw_error)

            if self.phase == "align_final_approach":
                if abs(yaw_error) > self.yaw_tolerance_rad:
                    return self._result(self._turn_command(yaw_error), distance, yaw_error)
                self.phase = "final_approach"
                self.waypoint_index = min(self.pre_goal_index + 1, len(self.path) - 1)
                continue

            if self.phase == "final_approach":
                tolerance = min(self.waypoint_tolerance_m, self.position_tolerance_m)
                while self.waypoint_index < len(self.path) - 1:
                    target = self.path[self.waypoint_index]
                    target_distance = math.hypot(target["x"] - x, target["y"] - y)
                    if target_distance > tolerance and not self._passed_waypoint(
                        pose_values, self.waypoint_index, tolerance
                    ):
                        break
                    self.waypoint_index += 1
                target = self.path[self.waypoint_index]
                if self.waypoint_index == len(self.path) - 1 and distance <= self.position_tolerance_m:
                    self.phase = "align_final_pose"
                    continue
                command = self._translation_command(
                    pose_values,
                    (target["x"], target["y"]),
                    self._segment_heading(self.waypoint_index),
                    max_velocity=self.terminal_max_linear_velocity,
                )
                return self._result(command, distance, yaw_error)

            if self.phase == "align_final_pose":
                if distance <= self.position_tolerance_m:
                    return self._result(self._turn_command(yaw_error), distance, yaw_error)
                self.phase = "final_approach"
                self.waypoint_index = max(self.pre_goal_index + 1, len(self.path) - 2)
                self.waypoint_index = min(self.waypoint_index, len(self.path) - 1)
                target = self.path[self.waypoint_index]
                command = self._translation_command(
                    pose_values,
                    (target["x"], target["y"]),
                    self._segment_heading(self.waypoint_index),
                    max_velocity=self.terminal_max_linear_velocity,
                )
                return self._result(command, distance, yaw_error)

        return self._result((0.0, 0.0, 0.0), distance, yaw_error)


def build_navigation_controller(
    *,
    controller_cfg: dict[str, Any],
    planner_cfg: dict[str, Any],
    goal: tuple[float, float, float],
    waypoint_tolerance_m: float,
    position_tolerance_m: float,
    yaw_tolerance_rad: float,
):
    """Build the configured waypoint controller without changing the default."""

    controller_type = str(controller_cfg.get("controller_type", "waypoint")).strip()
    common = {
        "max_linear_velocity": float(controller_cfg.get("max_linear_velocity", 0.35)),
        "max_angular_velocity": float(controller_cfg.get("max_angular_velocity", 0.8)),
        "waypoint_tolerance_m": waypoint_tolerance_m,
        "position_tolerance_m": position_tolerance_m,
        "yaw_tolerance_rad": yaw_tolerance_rad,
        "linear_gain": float(controller_cfg.get("linear_gain", 2.0)),
        "angular_gain": float(controller_cfg.get("angular_gain", 2.0)),
    }
    if controller_type in {"", "waypoint"}:
        return WaypointController(
            **common,
            rotate_first_error_rad=float(
                controller_cfg.get("rotate_first_error_rad", 0.2)
            ),
        )
    if controller_type != "phased_waypoint":
        raise ValueError(f"Unsupported navigation controller_type: {controller_type}")
    return PhasedWaypointController(
        **common,
        goal=goal,
        terminal_approach_distance_m=float(
            planner_cfg.get("terminal_approach_distance_m", 0.0)
        ),
        terminal_max_linear_velocity=float(
            controller_cfg.get("terminal_max_linear_velocity", 0.15)
        ),
        max_lateral_velocity=float(controller_cfg.get("max_lateral_velocity", 0.20)),
        lateral_gain=float(controller_cfg.get("lateral_gain", 2.0)),
        lateral_alignment_enter_m=float(
            controller_cfg.get("lateral_alignment_enter_m", 0.08)
        ),
        lateral_alignment_exit_m=float(
            controller_cfg.get("lateral_alignment_exit_m", 0.04)
        ),
        turn_enter_heading_error_rad=float(
            controller_cfg.get("turn_enter_heading_error_rad", 0.20)
        ),
        turn_exit_heading_error_rad=float(
            controller_cfg.get("turn_exit_heading_error_rad", 0.10)
        ),
    )


def build_navigation_plan(*, start_pose: tuple[float, float, float], goal: tuple[float, float, float], static_map: dict[str, Any] | None, footprint_points: list[list[float]], footprint_padding_m: float = 0.0, planner_cfg: dict[str, Any] | None = None, planner: GridAStarPlanner | None = None, planned_points: list[tuple[float, float]] | None = None) -> NavigationPlan | None:
    planner_cfg = planner_cfg or {}
    if planner is None:
        planner = GridAStarPlanner(
            resolution=float(planner_cfg.get("map_resolution", static_map.get("resolution", 0.05) if static_map else 0.05)),
            safety_distance_m=float(planner_cfg.get("safety_distance_m", 0.35)),
            proximity_weight=float(planner_cfg.get("proximity_weight", 2.0)),
        )
        if static_map is not None:
            planner.set_static_map(static_map, footprint_points=footprint_points, footprint_padding_m=footprint_padding_m)
    approach_distance = float(planner_cfg.get("terminal_approach_distance_m", 0.0))
    if not math.isfinite(approach_distance) or approach_distance < 0.0:
        raise ValueError("terminal_approach_distance_m must be non-negative and finite")
    if planned_points is not None:
        points = planned_points
    elif approach_distance == 0.0:
        points = planner.plan((start_pose[0], start_pose[1]), (goal[0], goal[1]))
    else:
        approach_step = float(
            planner_cfg.get("terminal_approach_step_m", approach_distance)
        )
        if not math.isfinite(approach_step) or approach_step <= 0.0:
            raise ValueError("terminal_approach_step_m must be positive and finite")
        goal_x, goal_y, goal_yaw = goal
        pre_goal = (
            goal_x - approach_distance * math.cos(goal_yaw),
            goal_y - approach_distance * math.sin(goal_yaw),
        )
        points = planner.plan((start_pose[0], start_pose[1]), pre_goal)
        if points:
            points = list(points)
            sample_count = max(1, math.ceil(approach_distance / approach_step))
            points.extend(
                (
                    pre_goal[0] + (goal_x - pre_goal[0]) * index / sample_count,
                    pre_goal[1] + (goal_y - pre_goal[1]) * index / sample_count,
                )
                for index in range(1, sample_count + 1)
            )
    if not points:
        return None
    poses = []
    for index, (x, y) in enumerate(points):
        if index == 0:
            # The robot is already at this pose.  Preserve its measured yaw
            # so collision checks do not invent an in-place turn at startup.
            yaw = float(start_pose[2])
        elif index < len(points) - 1:
            next_x, next_y = points[index + 1]
            yaw = math.atan2(next_y - y, next_x - x)
        else:
            yaw = float(goal[2])
        poses.append({"x": float(x), "y": float(y), "yaw": wrap_to_pi(yaw)})
    if static_map is None:
        collision = {"ok": True, "reason": "no_static_map"}
    else:
        collision = check_path_static_collision(
            static_map=static_map,
            footprint_points=footprint_points,
            path_poses=poses,
            footprint_padding_m=footprint_padding_m,
            initial_padding_egress_distance_m=float(planner_cfg.get("initial_padding_egress_distance_m", 0.0)),
        )
        if not collision.get("ok", False):
            return None
    return NavigationPlan(path=poses, goal=tuple(float(v) for v in goal), collision_check=collision)


def load_or_export_static_map(*, workflow, robot, cfg: dict[str, Any], scene_name: str) -> dict[str, Any] | None:
    configured_path = str(cfg.get("occupancy_map_path", cfg.get("map_yaml_path", "")) or "").strip()
    if configured_path:
        if not os.path.exists(configured_path):
            raise FileNotFoundError(f"Configured occupancy map does not exist: {configured_path}")
        return load_static_map(configured_path)
    if workflow is None or robot is None:
        return None
    try:
        try:
            from .static_map_exporter import IsaacStaticMapExporter
        except ImportError:
            exporter_path = Path(__file__).with_name("static_map_exporter.py")
            exporter_spec = importlib.util.spec_from_file_location("simbox_static_map_exporter", exporter_path)
            if exporter_spec is None or exporter_spec.loader is None:
                raise ImportError(f"Unable to load local static-map exporter from {exporter_path}")
            exporter_module = importlib.util.module_from_spec(exporter_spec)
            sys.modules[exporter_spec.name] = exporter_module
            exporter_spec.loader.exec_module(exporter_module)
            IsaacStaticMapExporter = exporter_module.IsaacStaticMapExporter
    except ImportError:
        return None
    export = IsaacStaticMapExporter(workflow, robot, robot.get_base_interface()["base_cfg"], scene_name=scene_name)
    clear_center_xy = None
    pose_getter = getattr(robot, "get_nav_base_pose", None) or getattr(robot, "get_mobile_base_pose", None)
    if callable(pose_getter):
        translation, _ = pose_getter()
        clear_center_xy = (float(translation[0]), float(translation[1]))
    result = export.export_map(str(cfg.get("map_output_dir", "output/local_navigation_maps")), clear_center_xy=clear_center_xy)
    return load_static_map(result["yaml_path"])


def select_approach_goal(*, approach_config: ApproachConfig, target_xy: tuple[float, float], start_pose: tuple[float, float, float], static_map: dict[str, Any] | None, base_cfg: dict[str, Any], robot_cfg: dict[str, Any] | None = None, planner_cfg: dict[str, Any] | None = None) -> tuple[tuple[float, float, float] | None, dict[str, Any]]:
    robot_cfg = robot_cfg or {}
    planner_cfg = planner_cfg or {}
    footprint_points = resolve_footprint_points(base_cfg)
    padding = resolve_approach_footprint_padding_m(base_cfg, approach_config)
    context = build_armbase_target_context(robot_cfg, approach_config)
    candidates = sample_approach_candidates(approach_config, target_xy, context)
    planner = GridAStarPlanner(
        resolution=float(planner_cfg.get("map_resolution", static_map.get("resolution", 0.05) if static_map else 0.05)),
        safety_distance_m=float(planner_cfg.get("safety_distance_m", 0.35)),
        proximity_weight=float(planner_cfg.get("proximity_weight", 2.0)),
    )
    if static_map is not None:
        # Candidate preflight can evaluate hundreds of poses.  The occupancy
        # grid and distance field depend only on this navigation request.
        planner.set_static_map(static_map, footprint_points=footprint_points, footprint_padding_m=padding)
    checked = []
    eligible = []
    for candidate in candidates:
        goal = (float(candidate["x"]), float(candidate["y"]), float(candidate["yaw"]))
        static_result = {"ok": True}
        if static_map is not None:
            static_result = check_footprint_static_collision(static_map=static_map, footprint_points=footprint_points, x=goal[0], y=goal[1], yaw=goal[2], footprint_padding_m=padding)
        candidate = dict(candidate)
        candidate["static_ok"] = bool(static_result.get("ok", False))
        candidate["path_ok"] = False
        candidate["static_check"] = static_result
        checked.append(candidate)
        if static_result.get("ok", False):
            eligible.append((len(checked) - 1, goal))
    max_solutions = max(1, int(planner_cfg.get("max_approach_solutions", 10)))
    paths = planner.plan_to_goals(
        (start_pose[0], start_pose[1]),
        [goal for _, goal in eligible],
        max_solutions=max_solutions,
    )
    reachable = []
    for eligible_index, path in paths.items():
        checked_index, goal = eligible[eligible_index]
        plan = build_navigation_plan(start_pose=start_pose, goal=goal, static_map=static_map, footprint_points=footprint_points, footprint_padding_m=padding, planner_cfg=planner_cfg, planner=planner, planned_points=path)
        if plan is None:
            continue
        checked[checked_index]["path_ok"] = True
        checked[checked_index]["path"] = plan.path
        reachable.append(checked[checked_index])
    selected = min(reachable, key=lambda candidate: (float(candidate.get("distance_to_target", float("inf"))), int(candidate.get("index", 0)))) if reachable else None
    return (None if selected is None else (float(selected["x"]), float(selected["y"]), float(selected["yaw"]))), {"candidates": checked, "selected": selected}
