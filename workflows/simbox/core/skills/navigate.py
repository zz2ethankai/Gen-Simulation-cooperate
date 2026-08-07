"""Main-flow navigation skill backed by the local SimBox planner."""

from __future__ import annotations

import json
import importlib.util
import math
import os
from pathlib import Path
import sys

from core.skills.base_skill import BaseSkill, SKILL_DICT, register_skill
from omegaconf import DictConfig, OmegaConf
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask

try:
    from .local_navigation import (
        WaypointController,
        build_navigation_plan,
        load_or_export_static_map,
        parse_approach_config,
        resolve_footprint_points,
        select_approach_goal,
    )
except ImportError:
    # Some focused tests load this file without importing the skills package.
    module_path = Path(__file__).with_name("local_navigation.py")
    spec = importlib.util.spec_from_file_location("simbox_local_navigation", module_path)
    local_navigation = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = local_navigation
    spec.loader.exec_module(local_navigation)
    WaypointController = local_navigation.WaypointController
    build_navigation_plan = local_navigation.build_navigation_plan
    load_or_export_static_map = local_navigation.load_or_export_static_map
    parse_approach_config = local_navigation.parse_approach_config
    resolve_footprint_points = local_navigation.resolve_footprint_points
    select_approach_goal = local_navigation.select_approach_goal


def _wrap_to_pi(yaw: float) -> float:
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


@register_skill
class Navigate(BaseSkill):
    """Non-blocking local navigation state machine.

    ``update`` runs after every physics step.  It plans once, emits a body
    twist through the workflow's local base driver, and reports completion
    without creating an action client, ROS node, TF listener, or external
    process.
    """

    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.world = kwargs["world"]
        self.workflow = kwargs.get("workflow")
        self.skill_cfg = cfg
        cfg_container = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        base_cfg = getattr(self.robot, "base_cfg", {}) or {}
        self.local_navigation_cfg = dict(base_cfg.get("local_navigation", {})) if isinstance(base_cfg, dict) else {}
        self.planner_cfg = dict(self.local_navigation_cfg.get("planner", {}))
        self.map_cfg = dict(self.local_navigation_cfg.get("map", {}))
        self.controller_cfg = dict(self.local_navigation_cfg.get("controller", {}))
        self.planner_cfg.update({key: value for key, value in cfg_container.items() if key in self.planner_cfg})
        self.map_cfg.update(
            {
                key: value
                for key, value in cfg_container.items()
                if key in self.map_cfg or key in {"occupancy_map_path", "map_yaml_path", "map_output_dir"}
            }
        )
        self.controller_cfg.update(
            {
                key: value
                for key, value in cfg_container.items()
                if key
                in {
                    "max_linear_velocity",
                    "max_angular_velocity",
                    "waypoint_tolerance",
                    "rotate_first_error_rad",
                    "linear_gain",
                    "angular_gain",
                }
            }
        )
        nested_controller_cfg = cfg_container.get("local_navigation", {})
        if isinstance(nested_controller_cfg, dict):
            self.controller_cfg.update(nested_controller_cfg)
        self.approach_config = parse_approach_config(cfg_container)
        if self.approach_config is None:
            self.goal_x, self.goal_y, self.goal_yaw = self._resolve_goal_pose(task, cfg)
        else:
            self.goal_x = self.goal_y = self.goal_yaw = 0.0

        self.position_tolerance_m = float(
            cfg.get(
                "xy_goal_tolerance",
                cfg.get("skill_xy_goal_tolerance", self.controller_cfg.get("position_tolerance_m", 0.10)),
            )
        )
        self.yaw_tolerance_rad = float(
            cfg.get(
                "yaw_goal_tolerance",
                cfg.get("skill_yaw_goal_tolerance", self.controller_cfg.get("yaw_tolerance_rad", 0.10)),
            )
        )
        self.waypoint_tolerance_m = float(
            cfg.get("waypoint_tolerance", self.controller_cfg.get("waypoint_tolerance_m", 0.25))
        )
        self.runtime_timeout_sec = float(
            cfg.get("runtime_timeout_sec", self.local_navigation_cfg.get("runtime_timeout_sec", 180.0))
        )
        self.output_root = str(cfg.get("output_root", "output/local_navigation/skills"))
        self.scene_name = str(cfg.get("scene_name", getattr(task, "name", "local_navigation_scene")))
        self._local_done = False
        self._local_success = False
        self._plan_started = False
        self._started_time = None
        self._static_map = None
        self._controller = None
        self._driver = None
        self._plan = None
        self._approach_debug = {}
        self.manip_list = []
        self.failure_reason = ""
        self.error_message = ""

    def _resolve_goal_pose(self, task: BaseTask, cfg: DictConfig) -> tuple[float, float, float]:
        goal_name = str(cfg.get("goal", "") or "").strip()
        if goal_name:
            positions = (getattr(task, "cfg", {}) or {}).get("positions")
            if not isinstance(positions, dict):
                raise KeyError(f"navigate goal '{goal_name}' requires task.cfg['positions']")
            goal_pose = positions.get(goal_name)
            if not isinstance(goal_pose, dict):
                raise KeyError(f"navigate goal '{goal_name}' was not found in task.cfg['positions']")
            try:
                local_x, local_y, local_yaw = float(goal_pose["x"]), float(goal_pose["y"]), float(goal_pose["yaw"])
            except KeyError as exc:
                raise KeyError(f"navigate goal '{goal_name}' requires x, y, yaw") from exc
            return self._floor_center_goal_to_world(task, local_x, local_y, local_yaw)
        try:
            return float(cfg["goal_x"]), float(cfg["goal_y"]), _wrap_to_pi(float(cfg["goal_yaw"]))
        except KeyError as exc:
            raise KeyError("navigate requires goal or goal_x, goal_y, and goal_yaw") from exc

    @classmethod
    def _floor_center_goal_to_world(cls, task, local_x, local_y, local_yaw):
        floor_x, floor_y = cls._floor_world_xy(task)
        return (
            float(floor_x + local_x),
            float(floor_y + local_y),
            _wrap_to_pi(local_yaw),
        )

    @staticmethod
    def _floor_world_xy(task):
        floor = (getattr(task, "fixtures", {}) or {}).get("floor")
        if floor is None or not hasattr(floor, "get_world_pose"):
            raise KeyError("navigate positions require task.fixtures['floor']")
        translation, _ = floor.get_world_pose()
        return float(translation[0]), float(translation[1])

    def simple_generate_manip_cmds(self):
        self.manip_list = []

    def is_ready(self):
        # Navigation progresses from update()/workflow physics ticks and does
        # not populate manipulator command lists.
        return False

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        if self.workflow is not None and hasattr(self.workflow, "get_local_base_driver"):
            self._driver = self.workflow.get_local_base_driver(getattr(self.robot, "name", ""))
        if self._driver is None:
            self.failure_reason = "local_driver_unavailable"
            self.error_message = "Workflow did not initialize a local base driver"
        return self._driver

    def _get_pose(self):
        getter = getattr(self.robot, "get_nav_base_pose", None) or getattr(self.robot, "get_mobile_base_pose", None)
        if not callable(getter):
            raise ValueError("Robot must expose get_nav_base_pose or get_mobile_base_pose")
        translation, orientation = getter()
        w, x, y, z = [float(value) for value in orientation[:4]]
        return float(translation[0]), float(translation[1]), math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _target_xy(self):
        target_name = self.approach_config.target_name if self.approach_config is not None else ""
        if not target_name:
            return self.goal_x, self.goal_y
        collections = [
            getattr(self.task, "_task_objects", {}),
            getattr(self.task, "objects", {}),
            getattr(self.task, "fixtures", {}),
            getattr(self.task, "distractors", {}),
            getattr(self.task, "visuals", {}),
        ]
        for collection in collections:
            target = collection.get(target_name) if isinstance(collection, dict) else None
            if target is None or not hasattr(target, "get_world_pose"):
                continue
            translation, _ = target.get_world_pose()
            return float(translation[0]), float(translation[1])
        raise KeyError(f"approach target '{target_name}' does not expose get_world_pose")

    def _robot_cfg_for_approach(self):
        cfg = getattr(self.robot, "cfg", {})
        return cfg if isinstance(cfg, dict) else {}

    def _begin_plan(self):
        driver = self._get_driver()
        if driver is None:
            return False
        try:
            self._static_map = load_or_export_static_map(
                workflow=self.workflow,
                robot=self.robot,
                cfg=self.map_cfg,
                scene_name=self.scene_name,
            )
        except Exception as exc:
            self.failure_reason = "static_map_error"
            self.error_message = f"Static occupancy map preparation failed: {type(exc).__name__}: {exc}"
            return False
        if self._static_map is None and not bool(self.skill_cfg.get("allow_unmapped_navigation", False)):
            self.failure_reason = "static_map_unavailable"
            self.error_message = "Local navigation requires a generated or configured static occupancy map"
            return False
        start_pose = self._get_pose()
        base_cfg = getattr(self.robot, "base_cfg", {}) or {}
        footprint = resolve_footprint_points(base_cfg)
        padding = float(self.local_navigation_cfg.get("footprint_padding_m", 0.0))
        if self.approach_config is not None:
            goal, debug = select_approach_goal(
                approach_config=self.approach_config,
                target_xy=self._target_xy(),
                start_pose=start_pose,
                static_map=self._static_map,
                base_cfg=base_cfg,
                robot_cfg=self._robot_cfg_for_approach(),
                planner_cfg=self.planner_cfg,
            )
            self._approach_debug = debug
            if goal is None:
                self.failure_reason = "no_reachable_approach_goal"
                self.error_message = "No approach candidate passed footprint collision and local A* checks"
                return False
            self.goal_x, self.goal_y, self.goal_yaw = goal
        self._plan = build_navigation_plan(
            start_pose=start_pose,
            goal=(self.goal_x, self.goal_y, self.goal_yaw),
            static_map=self._static_map,
            footprint_points=footprint,
            footprint_padding_m=padding,
            planner_cfg=self.planner_cfg,
        )
        if self._plan is None:
            self.failure_reason = "local_plan_failed"
            self.error_message = "Local footprint-aware A* could not find a collision-free path"
            return False
        self._controller = WaypointController(
            max_linear_velocity=float(self.controller_cfg.get("max_linear_velocity", 0.35)),
            max_angular_velocity=float(self.controller_cfg.get("max_angular_velocity", 0.8)),
            waypoint_tolerance_m=self.waypoint_tolerance_m,
            position_tolerance_m=self.position_tolerance_m,
            yaw_tolerance_rad=self.yaw_tolerance_rad,
            rotate_first_error_rad=float(self.controller_cfg.get("rotate_first_error_rad", 0.2)),
            linear_gain=float(self.controller_cfg.get("linear_gain", 2.0)),
            angular_gain=float(self.controller_cfg.get("angular_gain", 2.0)),
        )
        self._controller.reset(self._plan.path)
        driver.prepare_for_navigation()
        self._started_time = self._now_sec()
        self._write_debug("planned")
        return True

    def _now_sec(self):
        current = getattr(self.world, "current_time", None)
        try:
            return float(current)
        except (TypeError, ValueError):
            return 0.0

    def _write_debug(self, tag):
        try:
            path = os.path.join(self.output_root, "local_navigation", self.scene_name, f"{tag}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {"goal": [self.goal_x, self.goal_y, self.goal_yaw], "failure_reason": self.failure_reason, "error_message": self.error_message, "approach": self._approach_debug}
            if self._plan is not None:
                payload["path"] = self._plan.path
                payload["collision_check"] = self._plan.collision_check
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            pass

    def update(self):
        if self._local_done:
            return
        driver = self._get_driver()
        if driver is None:
            self._local_done = True
            return
        if not self._plan_started:
            self._plan_started = True
            try:
                planned = self._begin_plan()
            except Exception as exc:
                self.failure_reason = "local_plan_error"
                self.error_message = f"Local navigation planning failed: {type(exc).__name__}: {exc}"
                try:
                    driver.finalize_after_navigation()
                except Exception:
                    pass
                planned = False
            if not planned:
                self._local_done = True
                self._local_success = False
                self._write_debug("failed")
                return
        if self._started_time is not None and self._now_sec() - self._started_time > self.runtime_timeout_sec:
            self.failure_reason = "runtime_timeout"
            self.error_message = "Local navigation exceeded runtime_timeout_sec"
            driver.finalize_after_navigation()
            self._local_done = True
            self._local_success = False
            self._write_debug("timeout")
            return
        try:
            body_vx, body_vy, body_wz, done, _ = self._controller.command(
                self._get_pose(),
                (self.goal_x, self.goal_y, self.goal_yaw),
            )
            driver.set_command(body_vx, body_vy, body_wz)
        except Exception as exc:
            self.failure_reason = "local_control_error"
            self.error_message = f"Local navigation control failed: {type(exc).__name__}: {exc}"
            try:
                driver.finalize_after_navigation()
            finally:
                self._local_done = True
                self._local_success = False
                self._write_debug("control_error")
            return
        if done:
            driver.finalize_after_navigation()
            self._local_done = True
            self._local_success = True
            self._write_debug("succeeded")

    def is_done(self):
        return bool(self._local_done)

    def is_success(self):
        return bool(self._local_success)

    def is_feasible(self):
        return not (self._local_done and not self._local_success)


SKILL_DICT["navigate"] = Navigate
