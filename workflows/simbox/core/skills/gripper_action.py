import numpy as np
from core.planning.motion_command import MotionPhase
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask


# pylint: disable=unused-argument
@register_skill
class Gripper_Action(BaseSkill):
    def __init__(self, robot: Robot, skill_runtime, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.skill_cfg = cfg
        self._gripper_state = self.skill_cfg["gripper_state"]

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []

    def simple_generate_manip_cmds(self):
        manip_list = []

        p_base_ee_cur, q_base_ee_cur = self.skill_runtime.ee_pose()
        if self._gripper_state == 1:  # Open
            phase = MotionPhase.GRIPPER_OPEN
            action = "open_gripper"
        elif self._gripper_state == -1:  # Close
            phase = MotionPhase.GRIPPER_CLOSE
            action = "close_gripper"
        else:
            raise NotImplementedError

        cmd = self.pose_command(
            phase,
            p_base_ee_cur,
            q_base_ee_cur,
            gripper_action=action,
            replan_allowed=False,
            dwell_steps=int(self.skill_cfg.get("wait_steps", 10)),
        )
        manip_list.append(cmd)
        self.manip_list = manip_list

    def is_feasible(self, th=5):
        return self.skill_runtime.num_plan_failed <= th

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        return self.command_complete(self.manip_list[0])

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        return True
