"""Main-flow navigation skill."""

from __future__ import annotations

import math

from core.skills.base_skill import BaseSkill, SKILL_DICT, register_skill
from omegaconf import DictConfig, OmegaConf
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask

from nav2.runtime import configure_robot_for_nav2_skill


def _wrap_to_pi(yaw: float) -> float:
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


@register_skill
class Navigate(BaseSkill):
    """Block until the mobile base reaches a floor-referenced navigation goal."""

    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.world = kwargs["world"]
        self.workflow = kwargs.get("workflow")
        self.skill_cfg = cfg

        self.goal_x, self.goal_y, self.goal_yaw = self._resolve_goal_pose(task, cfg)

        legacy_position_tolerance_m = float(cfg.get("xy_goal_tolerance", cfg.get("skill_xy_goal_tolerance", 0.10)))
        legacy_yaw_tolerance_rad = float(cfg.get("yaw_goal_tolerance", cfg.get("skill_yaw_goal_tolerance", 0.10)))
        goal_checker_cfg = dict(
            getattr(self.robot, "base_cfg", {}).get("nav2_skill", {}).get("controller_server", {}).get("goal_checker", {})
        )
        self.nav2_position_tolerance_m = float(goal_checker_cfg.get("xy_goal_tolerance", legacy_position_tolerance_m))
        self.nav2_yaw_tolerance_rad = float(goal_checker_cfg.get("yaw_goal_tolerance", legacy_yaw_tolerance_rad))
        self.position_tolerance_m = legacy_position_tolerance_m
        self.yaw_tolerance_rad = legacy_yaw_tolerance_rad
        self.startup_timeout_sec = float(cfg.get("startup_timeout_sec", 60.0))
        self.runtime_timeout_sec = float(cfg.get("runtime_timeout_sec", 240.0))
        self.output_root = str(cfg.get("output_root", "output/ros_bridge/skills"))
        self.scene_name = str(cfg.get("scene_name", getattr(task, "name", "navigate_skill_scene")))
        nav2_skill_overrides = self._nav2_skill_overrides(cfg)

        self._configured_base_cfg = configure_robot_for_nav2_skill(
            self.robot,
            map_output_dir=str(cfg.get("map_output_dir", "output/nav2_maps")),
            map_resolution=float(cfg.get("map_resolution", 0.02)),
            map_z_min=float(cfg.get("map_z_min", 0.0)),
            map_z_max=float(cfg.get("map_z_max", 0.35)),
            map_include_visual_wall_geometry=bool(cfg.get("map_include_visual_wall_geometry", True)),
            position_tolerance_m=self.nav2_position_tolerance_m,
            yaw_tolerance_rad=self.nav2_yaw_tolerance_rad,
            nav2_skill_overrides=nav2_skill_overrides,
        )
        self._manager = None
        self._goal_started = False
        self._local_done = False
        self._local_success = False
        self._hold_command = None
        self.manip_list = []
        self.failure_reason = ""
        self.error_message = ""

    def _resolve_goal_pose(self, task: BaseTask, cfg: DictConfig) -> tuple[float, float, float]:
        goal_name = str(cfg.get("goal", "") or "").strip()
        if goal_name:
            task_cfg = getattr(task, "cfg", {}) or {}
            positions = task_cfg.get("positions")
            if not isinstance(positions, dict):
                raise KeyError(f"navigate goal '{goal_name}' requires task.cfg['positions'] to be a mapping")

            goal_pose = positions.get(goal_name)
            if not isinstance(goal_pose, dict):
                raise KeyError(f"navigate goal '{goal_name}' was not found in task.cfg['positions']")

            try:
                local_x = float(goal_pose["x"])
                local_y = float(goal_pose["y"])
                local_yaw = float(goal_pose["yaw"])
            except KeyError as exc:
                raise KeyError(
                    f"navigate goal '{goal_name}' requires position fields 'x', 'y', and 'yaw'"
                ) from exc
            return self._floor_center_goal_to_world(task, local_x, local_y, local_yaw)

        try:
            return float(cfg["goal_x"]), float(cfg["goal_y"]), _wrap_to_pi(float(cfg["goal_yaw"]))
        except KeyError as exc:
            raise KeyError("navigate requires either goal or goal_x, goal_y, and goal_yaw") from exc

    @classmethod
    def _floor_center_goal_to_world(
        cls,
        task: BaseTask,
        local_x: float,
        local_y: float,
        local_yaw: float,
    ) -> tuple[float, float, float]:
        floor_x, floor_y, floor_yaw = cls._floor_world_pose(task)
        cos_yaw = math.cos(floor_yaw)
        sin_yaw = math.sin(floor_yaw)
        world_x = floor_x + local_x * cos_yaw - local_y * sin_yaw
        world_y = floor_y + local_x * sin_yaw + local_y * cos_yaw
        world_yaw = _wrap_to_pi(floor_yaw + local_yaw)
        return float(world_x), float(world_y), float(world_yaw)

    @staticmethod
    def _floor_world_pose(task: BaseTask) -> tuple[float, float, float]:
        fixtures = getattr(task, "fixtures", {}) or {}
        floor = fixtures.get("floor")
        if floor is None or not hasattr(floor, "get_world_pose"):
            raise KeyError("navigate positions require task.fixtures['floor'] as the default reference")

        translation, orientation = floor.get_world_pose()
        return (
            float(translation[0]),
            float(translation[1]),
            _wrap_to_pi(
                math.atan2(
                    2.0 * (float(orientation[0]) * float(orientation[3]) + float(orientation[1]) * float(orientation[2])),
                    1.0
                    - 2.0
                    * (float(orientation[2]) * float(orientation[2]) + float(orientation[3]) * float(orientation[3])),
                )
            ),
        )

    @staticmethod
    def _nav2_skill_overrides(cfg: DictConfig) -> dict:
        overrides = cfg.get("nav2_skill", {})
        if isinstance(overrides, DictConfig):
            overrides = OmegaConf.to_container(overrides, resolve=True)
        if not isinstance(overrides, dict):
            overrides = {}
        else:
            overrides = dict(overrides)

        if "rotate_to_heading_enabled" in cfg:
            controller_cfg = overrides.setdefault("controller_server", {})
            follow_path_cfg = controller_cfg.setdefault("follow_path", {})
            follow_path_cfg["rotate_to_heading_enabled"] = bool(cfg.get("rotate_to_heading_enabled"))
        return overrides

    def simple_generate_manip_cmds(self):
        self._hold_command = None
        self.manip_list = []

    def is_ready(self):
        return bool(self.manip_list)

    def update(self):
        if self._local_done:
            return

        if self._manager is None:
            if self.workflow is not None and hasattr(self.workflow, "get_navigation_session_manager"):
                self._manager = self.workflow.get_navigation_session_manager(getattr(self.robot, "name", ""))
            if self._manager is None:
                self.failure_reason = "manager_unavailable"
                self.error_message = "Workflow did not initialize a navigation session manager for this robot"
                self._local_done = True
                self._local_success = False
                return

        self._manager.bind(
            world=self.world,
            task=self.task,
            robot=self.robot,
            scene_name=self.scene_name,
        )
        if not self._goal_started:
            self._manager.begin_goal(
                goal_x=self.goal_x,
                goal_y=self.goal_y,
                goal_yaw=self.goal_yaw,
                nav2_position_tolerance_m=self.nav2_position_tolerance_m,
                nav2_yaw_tolerance_rad=self.nav2_yaw_tolerance_rad,
                position_tolerance_m=self.position_tolerance_m,
                yaw_tolerance_rad=self.yaw_tolerance_rad,
                startup_timeout_sec=self.startup_timeout_sec,
                runtime_timeout_sec=self.runtime_timeout_sec,
            )
            self._goal_started = True

        if self._manager.done:
            self.failure_reason = str(self._manager.result.failure_reason)
            self.error_message = str(self._manager.result.error_message)
            self._local_done = True
            self._local_success = bool(self._manager.success)

    def is_done(self):
        return bool(self._local_done)

    def is_success(self):
        return bool(self._local_success)

    def is_feasible(self):
        if self._manager is None:
            return True
        return not (self._local_done and not self._local_success)


SKILL_DICT["navigate"] = Navigate
