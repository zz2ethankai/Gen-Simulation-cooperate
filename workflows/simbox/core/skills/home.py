import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.planning.config_contract import DIRECT_EXECUTION_MODE
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from omegaconf import DictConfig
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask


# pylint: disable=unused-argument
@register_skill
class Home(BaseSkill):
    def __init__(self, robot: Robot, skill_runtime, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.skill_cfg = cfg
        self.move_steps = int(self.skill_cfg.get("move_steps", 50))
        if self.move_steps <= 0:
            raise ValueError("home move_steps must be positive")

        self.lr_hand = self.skill_runtime.arm_name
        if self.lr_hand == "left":
            self._joint_indices = self.skill_runtime.robot_port.arm_indices
            self._joint_home = self.robot.left_joint_home
            if self.skill_cfg.get("gripper_state", None):
                self._gripper_state = self.skill_cfg["gripper_state"]
            else:
                self._gripper_state = self.robot.left_gripper_state
        elif self.lr_hand == "right":
            self._joint_indices = self.skill_runtime.robot_port.arm_indices
            self._joint_home = self.robot.right_joint_home
            if self.skill_cfg.get("gripper_state", None):
                self._gripper_state = self.skill_cfg["gripper_state"]
            else:
                self._gripper_state = self.robot.right_gripper_state

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []
        self.execution_mode = DIRECT_EXECUTION_MODE

    def simple_generate_manip_cmds(self):
        """Build typed execution-only joint interpolation used by Home."""

        manip_list = []
        curr_ee_trans, curr_ee_ori = self.skill_runtime.execution.get_ee_pose()
        curr_joints = np.asarray(self.robot.get_joint_positions(), dtype=float)[self._joint_indices]
        home_joints = np.asarray(self._joint_home, dtype=float)

        for k in range(self.move_steps):
            alpha = float(k + 1) / float(self.move_steps)
            arm_action = home_joints * alpha + curr_joints * (1.0 - alpha)
            cmd = self.joint_command(
                arm_action,
                gripper_state=self._gripper_state,
                phase=MotionPhase.CARRY_HOME,
                direct=True,
                replan_allowed=False,
            )
            manip_list.append(cmd)

        self.manip_list = manip_list

    def is_feasible(self, th=5):
        # Home is direct joint interpolation and has no planner attempt to
        # invalidate it.
        return True

    def is_subtask_done(self, t_eps=0.088):
        assert len(self.manip_list) != 0
        command = self.manip_list[0]
        if not isinstance(command, MotionPhaseCommand) or not command.is_direct:
            raise TypeError("Home emits direct MotionPhaseCommand values only")
        curr_joints = np.asarray(self.robot.get_joint_positions(), dtype=float)[self._joint_indices]
        target_joints = np.asarray(command.direct_joint_action, dtype=float)
        return bool(np.linalg.norm(curr_joints - target_joints) < t_eps)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
        if self.is_success():
            self.manip_list.clear()
            print("Home Done")
        return len(self.manip_list) == 0

    def is_success(self, t_eps=0.088):
        curr_joints = self.robot.get_joint_positions()[self._joint_indices]
        diff_trans = np.linalg.norm(curr_joints - self._joint_home)
        # print(diff_trans)
        return diff_trans < t_eps
