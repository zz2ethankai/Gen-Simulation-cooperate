# pylint: skip-file
import numpy as np
import torch
from core.skills.base_skill import BaseSkill, register_skill
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.utils.interpolate_utils import linear_interpolation
from omegaconf import DictConfig
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask


# pylint: disable=unused-argument
@register_skill
class Joint_Ctrl(BaseSkill):
    def __init__(self, robot: Robot, skill_runtime, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.name = cfg["name"]
        self.skill_cfg = cfg
        self.robot_base_path = self.skill_runtime.robot_base_path
        if self.skill_runtime.arm_name == "left":
            self.robot_lr = "left"
        elif self.skill_runtime.arm_name == "right":
            self.robot_lr = "right"
        self.manip_list = []
        self.success_threshold_js = self.skill_cfg.get("success_threshold_js", 5e-3)

    def simple_generate_manip_cmds(self):
        manip_list = []
        curr_js, target_js = self.get_target_js()
        interp_js_list = linear_interpolation(curr_js, target_js, self.skill_cfg.get("num_steps", 10))
        for js in interp_js_list:
            gripper_state = self.skill_cfg.get("gripper_state", 1.0)
            cmd = self.joint_command(
                js,
                gripper_action=(
                    "close_gripper" if gripper_state < 0 else "open_gripper"
                ),
                phase=MotionPhase.CARRY_HOME,
                replan_allowed=False,
            )
            manip_list.append(cmd)

        self.target_js = js
        self.manip_list = manip_list

    def get_target_js(self):
        """
        Compute target joint configuration based on current joint state and
        joint control commands defined in skill configuration.

        Returns:
            curr_js (np.ndarray): Current joint positions of the controlled arm.
            target_js (np.ndarray): Target joint positions after applying commands.
        """

        # --- Get current joint positions ---
        joint_positions = self.robot.get_joints_state().positions

        if isinstance(joint_positions, torch.Tensor):
            curr_js = joint_positions.detach().cpu().numpy()[self.skill_runtime.arm_indices]
        elif isinstance(joint_positions, np.ndarray):
            curr_js = joint_positions[self.skill_runtime.arm_indices]
        else:
            raise TypeError(f"Unsupported joint state type: {type(joint_positions)}")

        target_js = curr_js.copy()

        # --- Apply joint control commands ---
        # ctrl_list: list of (joint_index, angle_in_deg, mode), mode in {"abs", "delta"}
        ctrl_list = self.skill_cfg.get("ctrl_list", [])
        for joint_idx, angle_deg, mode in ctrl_list:
            angle_rad = angle_deg * np.pi / 180.0
            if mode == "abs":
                target_js[joint_idx] = angle_rad
            elif mode == "delta":
                target_js[joint_idx] += angle_rad
            else:
                raise ValueError(f"Unknown control mode: {mode}")

        # --- Apply robot-specific joint limits / safety clamps ---
        robot_file = self.skill_runtime.robot_file.lower()

        if "piper" in robot_file:
            # Example: clamp elbow and wrist joints for Piper robot
            target_js[2] = min(target_js[2], 0.0)
            target_js[4] = np.clip(target_js[4], -1.22, 1.22)

        elif "r5a" in robot_file:
            # Reserved for R5A-specific constraints
            pass

        return curr_js, target_js

    def is_feasible(self, th=5):
        return self.skill_runtime.num_plan_failed <= th

    def is_subtask_done(self, js_eps=5e-3, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        manip_cmd = self.manip_list[0]
        if not isinstance(manip_cmd, MotionPhaseCommand):
            raise TypeError("Joint_Ctrl emits MotionPhaseCommand values only")
        return self.command_complete(manip_cmd)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done(self.success_threshold_js):
            self.manip_list.pop(0)

        return len(self.manip_list) == 0

    def is_success(self):
        joint_positions = self.robot.get_joints_state().positions
        if isinstance(joint_positions, torch.Tensor):
            curr_js = joint_positions.numpy()[self.skill_runtime.arm_indices]  # JointState
        elif isinstance(joint_positions, np.ndarray):
            curr_js = joint_positions[self.skill_runtime.arm_indices]  # JointState
        distance_js = np.linalg.norm(curr_js - self.target_js)
        flag = (distance_js < self.success_threshold_js) and (len(self.manip_list) == 0)

        return flag
