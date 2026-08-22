import numpy as np
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.skills.base_skill import BaseSkill, register_skill
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

        self.lr_hand = self.skill_runtime.arm_name
        if self.lr_hand == "left":
            self._joint_indices = self.skill_runtime.arm_indices
            self._joint_home = self.robot.left_joint_home
            if self.skill_cfg.get("gripper_state", None):
                self._gripper_state = self.skill_cfg["gripper_state"]
            else:
                self._gripper_state = self.robot.left_gripper_state
        elif self.lr_hand == "right":
            self._joint_indices = self.skill_runtime.arm_indices
            self._joint_home = self.robot.right_joint_home
            if self.skill_cfg.get("gripper_state", None):
                self._gripper_state = self.skill_cfg["gripper_state"]
            else:
                self._gripper_state = self.robot.right_gripper_state

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []

    def simple_generate_manip_cmds(self):
        manip_list = []
        curr_ee_trans, curr_ee_ori = self.skill_runtime.ee_pose()
        curr_joints = self.robot.get_joint_positions()[self._joint_indices]
        home_joints = self._joint_home

        for k in range(0, 50):
            arm_action = np.array(home_joints) * ((k + 1) / 40) + np.array(curr_joints) * (1 - (k + 1) / 40)
            target_position, target_orientation = self.skill_runtime.compute_fk(
                arm_action,
                joint_names=self.skill_runtime.raw_joint_names,
            )
            cmd = MotionPhaseCommand(
                MotionPhase.CARRY_HOME,
                target_position,
                target_orientation,
                gripper_action=(
                    "open_gripper" if float(self._gripper_state) >= 0.0 else "close_gripper"
                ),
                replan_allowed=False,
                joint_target=np.asarray(arm_action, dtype=float),
            )
            manip_list.append(cmd)

        self.manip_list = manip_list

    def is_feasible(self, th=5):
        return self.skill_runtime.num_plan_failed <= th

    def is_subtask_done(self, t_eps=0.088):
        assert len(self.manip_list) != 0
        command = self.manip_list[0]
        if not isinstance(command, MotionPhaseCommand):
            raise TypeError("Home emits MotionPhaseCommand values only")
        return self.command_complete(command)

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
