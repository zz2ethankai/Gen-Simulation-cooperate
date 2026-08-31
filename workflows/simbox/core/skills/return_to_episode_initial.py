"""Typed post-Place return to the arm pose captured for this episode."""

from __future__ import annotations

import logging

import numpy as np

from core.planning.motion_command import MotionPhase
from core.skills.base_skill import BaseSkill, register_skill


LOGGER = logging.getLogger("de_logger")


@register_skill
class ReturnToEpisodeInitial(BaseSkill):
    """Plan one arm back to its episode-initial joint configuration."""

    def __init__(self, robot, skill_runtime, task, cfg=None, *args, **kwargs):
        del args, kwargs
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        runtime = self._require_skill_runtime()
        self.task = task
        self.skill_cfg = cfg or {}
        self.manip_list = []
        positions = robot.get_joints_state().positions
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
        indices = np.asarray(runtime.robot_port.arm_indices, dtype=int)
        self._target = np.asarray(positions, dtype=float)[indices].copy()
        self._tolerance = float(
            self.skill_cfg.get("joint_tolerance_rad", 0.03)
        )
        if not np.isfinite(self._tolerance) or self._tolerance <= 0.0:
            raise ValueError("joint_tolerance_rad must be positive")

    def _arm_positions(self) -> np.ndarray:
        positions = self.robot.get_joints_state().positions
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
        indices = np.asarray(
            self._require_skill_runtime().robot_port.arm_indices, dtype=int
        )
        return np.asarray(positions, dtype=float)[indices]

    def generate_manip_cmds(self):
        runtime = self._require_skill_runtime()
        runtime.sync_dynamic_poses(force=True)
        self.manip_list = [
            self.joint_command(
                self._target,
                phase=MotionPhase.CARRY_HOME,
                completion_tolerance={"joint_position_rad": self._tolerance},
            )
        ]
        LOGGER.warning(
            "[ReturnInitialDebug] start robot=%s arm=%s target=%s",
            runtime.name,
            runtime.arm_name,
            np.round(self._target, 6).tolist(),
        )

    def is_feasible(self, th=5):
        return bool(
            self._require_skill_runtime().execution.state.num_plan_failed <= th
        )

    def is_subtask_done(self, *args, **kwargs):
        del args, kwargs
        if not self.manip_list:
            return True
        return self.command_complete(self.manip_list[0])

    def is_done(self):
        if self.manip_list and self.is_subtask_done():
            self.manip_list.pop(0)
        return not self.manip_list

    def is_success(self):
        runtime = self._require_skill_runtime()
        error = float(np.linalg.norm(self._arm_positions() - self._target))
        LOGGER.warning(
            "[ReturnInitialDebug] complete robot=%s arm=%s joint_error_rad=%.6f",
            runtime.name,
            runtime.arm_name,
            error,
        )
        return error <= self._tolerance
