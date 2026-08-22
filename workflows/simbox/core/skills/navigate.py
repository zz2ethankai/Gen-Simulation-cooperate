"""Main-flow navigation skill backed by the local SimBox planner."""

from __future__ import annotations

import json
import importlib.util
import math
import os
from pathlib import Path
import sys

from core.skills.base_skill import BaseSkill, SKILL_DICT, register_skill
from core.mobile.navigation_settle import (
    NavigationSettleBarrier,
    NavigationSettlePort,
    NavigationSettleStatus,
)
from omegaconf import DictConfig, OmegaConf
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask

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

    def __init__(self, robot: Robot, skill_runtime, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.world = kwargs["world"]
        self.workflow = kwargs.get("workflow")
        self.skill_cfg = cfg
        cfg_container = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
        base_cfg = getattr(self.robot, "base_cfg", {}) or {}
        self.local_navigation_cfg = dict(base_cfg.get("local_navigation", {})) if isinstance(base_cfg, dict) else {}
        self.planner_cfg = dict(self.local_navigation_cfg.get("planner", {}))
        self.map_cfg = dict(self.local_navigation_cfg.get("map", {}))
        self.navigation_controller_cfg = dict(self.local_navigation_cfg.get("controller", {}))
        self.planner_cfg.update({key: value for key, value in cfg_container.items() if key in self.planner_cfg})
        self.map_cfg.update(
            {
                key: value
                for key, value in cfg_container.items()
                if key in self.map_cfg or key in {"occupancy_map_path", "map_yaml_path", "map_output_dir"}
            }
        )
        self.navigation_controller_cfg.update(
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
            self.navigation_controller_cfg.update(nested_controller_cfg)
        platform_cfg = base_cfg.get("platform", {}) if isinstance(base_cfg, dict) else {}
        platform_navigation_cfg = (
            platform_cfg.get("local_navigation", {})
            if isinstance(platform_cfg, dict)
            else {}
        )
        settle_cfg = dict(platform_navigation_cfg.get("settle", {}))
        settle_cfg.update(self.local_navigation_cfg.get("settle", {}))
        for cfg_key, settle_key in (
            ("settle_linear_speed_tolerance", "linear_speed_tolerance"),
            ("settle_angular_speed_tolerance", "angular_speed_tolerance"),
            ("settle_consecutive_steps", "consecutive_steps"),
        ):
            if cfg_key in cfg_container:
                settle_cfg[settle_key] = cfg_container[cfg_key]
        self.settle_linear_speed_tolerance = float(
            settle_cfg.get("linear_speed_tolerance", 0.005)
        )
        self.settle_angular_speed_tolerance = float(
            settle_cfg.get("angular_speed_tolerance", 0.005)
        )
        self.settle_consecutive_steps = int(settle_cfg.get("consecutive_steps", 8))
        if (
            not math.isfinite(self.settle_linear_speed_tolerance)
            or not math.isfinite(self.settle_angular_speed_tolerance)
            or self.settle_linear_speed_tolerance < 0.0
            or self.settle_angular_speed_tolerance < 0.0
            or self.settle_consecutive_steps < 1
        ):
            raise ValueError("Navigate settling tolerances must be non-negative and consecutive_steps must be positive")
        self._goal_reached = False
        self._settle_streak = 0
        self.approach_config = parse_approach_config(cfg_container)
        if self.approach_config is None:
            self.goal_x, self.goal_y, self.goal_yaw = self._resolve_goal_pose(task, cfg)
        else:
            self.goal_x = self.goal_y = self.goal_yaw = 0.0

        self.position_tolerance_m = float(
            cfg.get(
                "xy_goal_tolerance",
                cfg.get("skill_xy_goal_tolerance", self.navigation_controller_cfg.get("position_tolerance_m", 0.10)),
            )
        )
        self.yaw_tolerance_rad = float(
            cfg.get(
                "yaw_goal_tolerance",
                cfg.get("skill_yaw_goal_tolerance", self.navigation_controller_cfg.get("yaw_tolerance_rad", 0.10)),
            )
        )
        self.waypoint_tolerance_m = float(
            cfg.get("waypoint_tolerance", self.navigation_controller_cfg.get("waypoint_tolerance_m", 0.25))
        )
        self.runtime_timeout_sec = float(
            cfg.get("runtime_timeout_sec", self.local_navigation_cfg.get("runtime_timeout_sec", 180.0))
        )
        # A stopped base must not hold the DAG at the navigation boundary
        # indefinitely.  Keep this conservative code default local to the
        # barrier; existing task YAML remains unchanged.
        default_settle_timeout = (
            min(5.0, self.runtime_timeout_sec)
            if self.runtime_timeout_sec > 0.0
            else 5.0
        )
        self.settle_timeout_sec = float(
            cfg_container.get("settle_timeout_sec", default_settle_timeout)
        )
        if not math.isfinite(self.settle_timeout_sec) or self.settle_timeout_sec <= 0.0:
            raise ValueError("settle_timeout_sec must be finite and positive")
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
        self.navigation_settle_port: NavigationSettlePort | None = None
        self.settle_barrier: NavigationSettleBarrier | None = None

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
        if self.workflow is not None and hasattr(self.workflow, "get_local_base_driver"):
            workflow_driver = self.workflow.get_local_base_driver(
                getattr(self.robot, "name", "")
            )
            if workflow_driver is not None and workflow_driver is not self._driver:
                self._driver = workflow_driver
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

    def _timing_phase(self, name, category, metadata=None):
        scope = getattr(self, "_timing_scope", None)
        if scope is None:
            return None
        try:
            return scope.phase(name, category=category, metadata=metadata).start()
        except Exception:
            # Navigation must remain usable when optional telemetry is broken.
            return None

    @staticmethod
    def _finish_timing_phase(phase, success, error=None):
        if phase is None:
            return
        try:
            phase.finish(
                success=bool(success),
                reason=(str(error) if error is not None else None),
                error=error,
            )
        except Exception:
            pass

    def _start_execution_timing(self):
        if getattr(self, "_timing_execution_phase", None) is not None:
            return
        workflow_start = getattr(self.workflow, "_start_skill_execution_phase", None)
        if callable(workflow_start):
            workflow_start(self, "navigation.execution")
            return
        phase = self._timing_phase(
            "navigation.execution",
            "execution",
            metadata={"skill": "navigate"},
        )
        if phase is not None:
            self._timing_execution_phase = phase

    def _begin_plan(self):
        phase = self._timing_phase(
            "navigation.plan",
            "planner",
            metadata={"skill": "navigate"},
        )
        try:
            planned = self._begin_plan_impl()
        except Exception as exc:
            self._finish_timing_phase(phase, False, exc)
            raise
        self._finish_timing_phase(
            phase,
            planned,
            None if planned else RuntimeError(self.failure_reason or "navigation_plan_failed"),
        )
        if planned:
            self._start_execution_timing()
        return planned

    def _begin_plan_impl(self):
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
                self.error_message = "No approach candidate passed center-cell occupancy and local A* checks"
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
            self.error_message = "Local center-cell A* could not find a collision-free path"
            return False
        self._controller = WaypointController(
            max_linear_velocity=float(self.navigation_controller_cfg.get("max_linear_velocity", 0.35)),
            max_angular_velocity=float(self.navigation_controller_cfg.get("max_angular_velocity", 0.8)),
            waypoint_tolerance_m=self.waypoint_tolerance_m,
            position_tolerance_m=self.position_tolerance_m,
            yaw_tolerance_rad=self.yaw_tolerance_rad,
            rotate_first_error_rad=float(self.navigation_controller_cfg.get("rotate_first_error_rad", 0.2)),
            linear_gain=float(self.navigation_controller_cfg.get("linear_gain", 2.0)),
            angular_gain=float(self.navigation_controller_cfg.get("angular_gain", 2.0)),
        )
        self._controller.reset(self._plan.path)
        # Compose the measured-state boundary from the existing local driver;
        # this does not alter A*, waypoint control, or base actuation.
        try:
            settle_port = NavigationSettlePort.from_robot_driver(self.robot, driver)
            settle_barrier = NavigationSettleBarrier(
                settle_port,
                linear_speed_tolerance=self.settle_linear_speed_tolerance,
                angular_speed_tolerance=self.settle_angular_speed_tolerance,
                consecutive_steps=self.settle_consecutive_steps,
                timeout_sec=self.settle_timeout_sec,
            )
        except Exception as exc:
            self.failure_reason = "settle_port_unavailable"
            self.error_message = (
                "Local navigation could not compose its measured settle boundary: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        self.navigation_settle_port = settle_port
        self.settle_barrier = settle_barrier
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

    def _physics_dt(self):
        getter = getattr(self.world, "get_physics_dt", None)
        try:
            value = getter() if callable(getter) else getattr(self.world, "physics_dt", 0.0)
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        return value if math.isfinite(value) and value >= 0.0 else 0.0

    def _finalize_navigation(self):
        port = self.navigation_settle_port
        if port is not None:
            port.finalize()
            return
        driver = self._driver
        if driver is not None:
            driver.finalize_after_navigation()

    def _write_debug(self, tag):
        try:
            path = os.path.join(self.output_root, "local_navigation", self.scene_name, f"{tag}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {"goal": [self.goal_x, self.goal_y, self.goal_yaw], "failure_reason": self.failure_reason, "error_message": self.error_message, "approach": self._approach_debug}
            if self.settle_barrier is not None:
                payload["settle"] = self.settle_barrier.result.to_dict()
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
            self._finalize_navigation()
            self._local_done = True
            self._local_success = False
            self._write_debug("timeout")
            return
        if self._goal_reached:
            barrier = self.settle_barrier
            if barrier is None:
                self.failure_reason = "settle_barrier_unavailable"
                self.error_message = "Navigation reached its goal without a settle barrier"
                try:
                    driver.finalize_after_navigation()
                finally:
                    self._local_done = True
                    self._local_success = False
                    self._write_debug("settle_error")
                return
            result = barrier.step(
                now_sec=self._now_sec(),
                dt_sec=self._physics_dt(),
            )
            self._settle_streak = result.stable_steps
            if not result.complete:
                return
            if result.success:
                self._finalize_navigation()
                self._local_done = True
                self._local_success = True
                self._write_debug("succeeded")
                return
            if result.status == NavigationSettleStatus.TIMED_OUT:
                self.failure_reason = "settle_timeout"
                self.error_message = (
                    "Measured base pose/twist did not remain stable before "
                    f"settle_timeout_sec={self.settle_timeout_sec:.3f}"
                )
            else:
                self.failure_reason = "settle_measurement_failed"
                self.error_message = result.reason or "Measured base settle barrier failed"
            try:
                self._finalize_navigation()
            finally:
                self._local_done = True
                self._local_success = False
                self._write_debug("settle_failed")
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
            self._goal_reached = True
            self._settle_streak = 0
            if self.settle_barrier is not None:
                self.settle_barrier.start(self._now_sec())
            driver.set_command(0.0, 0.0, 0.0)

    def is_done(self):
        return bool(self._local_done)

    def is_success(self):
        return bool(self._local_success)

    def is_feasible(self):
        return not (self._local_done and not self._local_success)


SKILL_DICT["navigate"] = Navigate
