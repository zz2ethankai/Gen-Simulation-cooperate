"""ROS-free static navigation primitives for SimBox.

The planner is intentionally small and deterministic: an occupancy image is
inflated by the mobile footprint, A* searches an 8-connected grid, and a
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
    """Footprint-inflated A* over a static map image."""

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
        radius_m = max((math.hypot(x, y) for x, y in self._footprint_points), default=0.0) + self._footprint_padding_m
        radius_cells = int(math.ceil(radius_m / self.resolution))
        self._grid = self._inflate_grid(occupied, radius_cells)
        self._distance_field = self._compute_distance_field(self._grid)

    @staticmethod
    def _inflate_grid(occupied: np.ndarray, radius_cells: int) -> np.ndarray:
        if radius_cells <= 0:
            return occupied.astype(bool)
        try:
            from scipy.ndimage import binary_dilation

            axis = np.arange(-radius_cells, radius_cells + 1)
            xx, yy = np.meshgrid(axis, axis)
            structure = xx * xx + yy * yy <= radius_cells * radius_cells
            return binary_dilation(occupied, structure=structure)
        except ImportError:
            pass
        inflated = occupied.astype(bool).copy()
        occupied_indices = np.argwhere(occupied)
        for row, col in occupied_indices:
            row_min = max(0, int(row) - radius_cells)
            row_max = min(occupied.shape[0] - 1, int(row) + radius_cells)
            col_min = max(0, int(col) - radius_cells)
            col_max = min(occupied.shape[1] - 1, int(col) + radius_cells)
            for rr in range(row_min, row_max + 1):
                for cc in range(col_min, col_max + 1):
                    if (rr - int(row)) ** 2 + (cc - int(col)) ** 2 <= radius_cells ** 2:
                        inflated[rr, cc] = True
        return inflated

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
        return self._simplify(points)

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
                    found[index] = self._simplify(points)
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

    def __init__(self, *, max_linear_velocity: float = 0.35, max_angular_velocity: float = 0.8, waypoint_tolerance_m: float = 0.25, position_tolerance_m: float = 0.1, yaw_tolerance_rad: float = 0.1, rotate_first_error_rad: float = 0.2, linear_gain: float = 2.0, angular_gain: float = 2.0, holonomic: bool = True):
        self.max_linear_velocity = max(float(max_linear_velocity), 0.0)
        self.max_angular_velocity = max(float(max_angular_velocity), 0.0)
        self.waypoint_tolerance_m = max(float(waypoint_tolerance_m), 0.0)
        self.position_tolerance_m = max(float(position_tolerance_m), 0.0)
        self.yaw_tolerance_rad = max(float(yaw_tolerance_rad), 0.0)
        self.rotate_first_error_rad = max(float(rotate_first_error_rad), 0.0)
        self.linear_gain = max(float(linear_gain), 0.0)
        self.angular_gain = max(float(angular_gain), 0.0)
        self.holonomic = bool(holonomic)
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
        control_yaw_error = yaw_error
        if not self.holonomic and distance > self.position_tolerance_m and target_distance > 1.0e-6:
            control_yaw_error = wrap_to_pi(math.atan2(dy, dx) - yaw)
        if distance > self.position_tolerance_m and abs(control_yaw_error) > self.rotate_first_error_rad:
            world_vx = world_vy = 0.0
        body_vx = world_vx * math.cos(yaw) + world_vy * math.sin(yaw)
        body_vy = -world_vx * math.sin(yaw) + world_vy * math.cos(yaw)
        if not self.holonomic:
            body_vy = 0.0
        wz = float(np.clip(self.angular_gain * control_yaw_error, -self.max_angular_velocity, self.max_angular_velocity))
        return body_vx, body_vy, wz, False, {"waypoint_index": self.waypoint_index, "waypoint_count": len(self.path), "distance_to_goal": distance, "yaw_error": yaw_error}


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
    points = planned_points if planned_points is not None else planner.plan((start_pose[0], start_pose[1]), (goal[0], goal[1]))
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
        # Candidate preflight can evaluate hundreds of poses.  Map inflation
        # and the distance field depend only on this navigation request, not
        # on an individual candidate, so build them once.
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
