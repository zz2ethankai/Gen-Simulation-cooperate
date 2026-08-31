import numpy as np
from core.planning.motion_command import MotionPhase
from core.skills.base_skill import BaseSkill, register_skill
from omegaconf import DictConfig
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask


# pylint: disable=consider-using-generator,too-many-public-methods,unused-argument
@register_skill
class Wait(BaseSkill):
    def __init__(self, robot: Robot, skill_runtime, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.skill_cfg = cfg
        self.success_threshold = cfg["success_threshold"]
        self.name = cfg["name"]
        self.move_obj = task.objects[cfg["objects"][0]]
        # Wait is a non-manipulation synchronization Skill.  Its workflow
        # tick is the observable wait; no planner command is emitted.
        self._passthrough = True
        self.robot_lr = ""

        self.wait_steps = int(cfg.get("wait_steps", cfg.get("hold_steps", 1)))
        if self.wait_steps <= 0:
            raise ValueError("Wait requires wait_steps > 0")
        self.manip_list = [None] * self.wait_steps

    def simple_generate_manip_cmds(self):
        if self._passthrough:
            self.manip_list = [None] * self.wait_steps
            return
        manip_list = []
        p_base_ee_cur, q_base_ee_cur = self.skill_runtime.execution.get_ee_pose()

        self.p_base_ee_tgt = p_base_ee_cur
        self.q_base_ee_tgt = q_base_ee_cur

        action = (
            "close_gripper"
            if self.skill_cfg.get("gripper_state", -1.0) == -1.0
            else "open_gripper"
        )
        phase = MotionPhase.GRIPPER_CLOSE if action == "close_gripper" else MotionPhase.GRIPPER_OPEN
        manip_list.append(
            self.pose_command(
                phase,
                self.p_base_ee_tgt,
                self.q_base_ee_tgt,
                gripper_action=action,
                replan_allowed=False,
                dwell_steps=int(self.skill_cfg.get("wait_steps", 50)),
            )
        )

        self.manip_list = manip_list

    def is_ready(self):
        return not self._passthrough

    def is_feasible(self, th=5):
        return self._passthrough or self.skill_runtime.execution.state.num_plan_failed <= th

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        return self.command_complete(self.manip_list[0])

    def is_done(self):
        if self._passthrough:
            if self.manip_list:
                self.manip_list.pop(0)
            return not self.manip_list
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        if self._passthrough:
            return True
        p_base_ee_cur, _ = self.skill_runtime.execution.get_ee_pose()
        distance = np.linalg.norm(p_base_ee_cur - self.p_base_ee_tgt)
        flag = (distance < self.success_threshold) and (len(self.manip_list) == 0)

        return flag
