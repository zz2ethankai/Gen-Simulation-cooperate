import logging

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill


LOGGER = logging.getLogger("de_logger")


@register_skill
class ReturnToEpisodeInitial(BaseSkill):
    """Return one arm to the pose captured at episode initialization."""

    def __init__(self, robot, controller, task, cfg=None, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg or {}
        self.manip_list = []
        self._target = np.asarray(
            getattr(controller, "episode_initial_arm_joints", None), dtype=float
        )
        if self._target.size == 0:
            self._target = np.asarray(
                robot.get_joints_state().positions[controller.arm_indices], dtype=float
            )
        self._tolerance = float(self.skill_cfg.get("joint_tolerance_rad", 0.03))
        self._steps = max(1, int(self.skill_cfg.get("return_steps", 60)))

    def simple_generate_manip_cmds(self):
        current = np.asarray(
            self.robot.get_joints_state().positions[self.controller.arm_indices],
            dtype=float,
        )
        p_ee, q_ee = self.controller.get_ee_pose()
        gripper_state = float(getattr(self.controller, "_gripper_state", 1.0))
        self.manip_list = []
        for index in range(1, self._steps + 1):
            ratio = index / float(self._steps)
            arm_action = (1.0 - ratio) * current + ratio * self._target
            self.manip_list.append(
                (
                    p_ee,
                    q_ee,
                    "dummy_forward",
                    {"arm_action": arm_action, "gripper_state": gripper_state},
                )
            )
        LOGGER.warning(
            "[ReturnInitialDebug] start robot=%s arm=%s target=%s steps=%d",
            self.controller.name,
            self.controller.lr_name,
            np.round(self._target, 6).tolist(),
            self._steps,
        )

    def is_feasible(self, th=5):
        return self.controller.num_plan_failed <= th

    def is_subtask_done(self, *args, **kwargs):
        current = np.asarray(
            self.robot.get_joints_state().positions[self.controller.arm_indices],
            dtype=float,
        )
        return float(np.linalg.norm(current - self._target)) <= self._tolerance

    def is_done(self):
        if not self.manip_list:
            return True
        if self.is_subtask_done():
            self.manip_list.clear()
        else:
            self.manip_list.pop(0)
        return not self.manip_list

    def is_success(self):
        current = np.asarray(
            self.robot.get_joints_state().positions[self.controller.arm_indices],
            dtype=float,
        )
        error = float(np.linalg.norm(current - self._target))
        LOGGER.warning(
            "[ReturnInitialDebug] complete robot=%s arm=%s joint_error_rad=%.6f",
            self.controller.name,
            self.controller.lr_name,
            error,
        )
        return error <= self._tolerance
