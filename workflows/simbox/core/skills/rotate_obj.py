import numpy as np
import torch
from core.skills.base_skill import BaseSkill, register_skill
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.utils.interpolate_utils import linear_interpolation
from omegaconf import DictConfig
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
from scipy.spatial.transform import Rotation as R


# pylint: disable=unused-argument
@register_skill
class Rotate_Obj(BaseSkill):
    def __init__(self, robot: Robot, skill_runtime, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self.task = task
        self.name = cfg["name"]
        self.move_obj = task.objects[cfg["objects"][0]]
        self.skill_cfg = cfg
        self.robot_base_path = self.skill_runtime.robot_base_path
        self.T_world_base = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        if self.skill_runtime.arm_name == "left":
            self.robot_lr = "left"
        elif self.skill_runtime.arm_name == "right":
            self.robot_lr = "right"
        self.manip_list = []
        self.success_threshold_move = self.skill_cfg["success_threshold_move"]
        self.success_threshold_rotate = self.skill_cfg["success_threshold_rotate"]

    def simple_generate_manip_cmds(self):
        manip_list = []
        p_base_ee_cur, q_base_ee_cur = self.skill_runtime.ee_pose()

        self.p_base_ee_tgt, self.q_base_ee_tgt = self.getTgtPose()

        direct_cfg = self.skill_cfg.get("direct_joint_motion", None)
        if direct_cfg:
            curr_js, target_js = self.get_tgt_js()
            interp_js_list = linear_interpolation(curr_js, target_js, direct_cfg.get("num_steps", 10))
            for js in interp_js_list:
                gripper_state = direct_cfg.get("gripper_state", 1.0)
                cmd = self.joint_command(
                    js,
                    phase=MotionPhase.CARRY_HOME,
                    direct=True,
                    gripper_state=gripper_state,
                    replan_allowed=False,
                )
                manip_list.append(cmd)

        first_motion = self.skill_cfg.get("first_motion", None)
        gripper_state = self.skill_cfg.get("gripper_state", -1.0)
        gripper_cmd = "close_gripper" if gripper_state == -1.0 else "open_gripper"

        if first_motion == "move":
            move_offset = self.skill_cfg.get("move_offset", [0, 0, 0])
            manip_list.append(
                self.pose_command(
                    MotionPhase.TRANSIT_PREGRASP,
                    self.p_base_ee_tgt + np.array(move_offset),
                    q_base_ee_cur,
                    gripper_action=gripper_cmd,
                    active_object=getattr(self.move_obj, "name", None),
                )
            )

        elif first_motion == "rotate":
            rotate_offset = self.skill_cfg.get("rotate_offset", [0, 0, 0])
            manip_list.append(
                self.pose_command(
                    MotionPhase.TRANSIT_PREGRASP,
                    p_base_ee_cur + np.array(rotate_offset),
                    self.q_base_ee_tgt,
                    gripper_action=gripper_cmd,
                    active_object=getattr(self.move_obj, "name", None),
                )
            )

            if self.skill_cfg.get("rotate_only", False):
                self.p_base_ee_tgt = p_base_ee_cur + np.array(rotate_offset)

        manip_list.append(
            self.pose_command(
                MotionPhase.TRANSIT_PREGRASP,
                self.p_base_ee_tgt,
                self.q_base_ee_tgt,
                gripper_action=gripper_cmd,
                active_object=getattr(self.move_obj, "name", None),
            )
        )

        self.manip_list = manip_list

    def getTgtPose(self):
        p_base_ee_cur, q_base_ee_cur = self.skill_runtime.ee_pose()
        p_world_move_obj_cur, q_world_move_obj_cur = self.move_obj.get_world_pose()  # w,x,y,z

        T_world_ee_cur = self.T_world_base @ tf_matrix_from_pose(p_base_ee_cur, q_base_ee_cur)
        T_world_move_obj_cur = tf_matrix_from_pose(p_world_move_obj_cur, q_world_move_obj_cur)

        if self.skill_cfg.get("obj_axis_offset", None):
            obj_axis_offset = self.skill_cfg.get("obj_axis_offset", None)
            axis_map = {"x": 0, "y": 1, "z": 2}
            dT = np.eye(4)
            for axis, offset in obj_axis_offset:
                if axis in axis_map:
                    dT[axis_map[axis], 3] = offset
            T_world_move_obj_cur = T_world_move_obj_cur @ dT

        q_world_move_obj_tgt = self.getTgtMoveObjOrientation()
        T_world_move_obj_tgt = T_world_move_obj_cur.copy()
        T_world_move_obj_tgt[:3, :3] = R.from_quat(q_world_move_obj_tgt, scalar_first=True).as_matrix()
        p_base_ee_tgt, q_base_ee_tgt = self.calTgtEEPose(
            T_world_ee_cur,
            T_world_move_obj_cur,
            T_world_move_obj_tgt,
        )
        trans_offset = self.skill_cfg.get("trans_offset", None)
        if trans_offset:
            p_base_ee_tgt += np.array(trans_offset)

        return p_base_ee_tgt, q_base_ee_tgt

    def calTgtEEPose(self, T_world_ee_cur=None, T_world_move_obj_cur=None, T_world_move_obj_tgt=None):
        T_base_ee_tgt = (
            np.linalg.inv(self.T_world_base)
            @ T_world_move_obj_tgt
            @ np.linalg.inv(T_world_move_obj_cur)
            @ T_world_ee_cur
        )

        return pose_from_tf_matrix(T_base_ee_tgt)

    def getTgtMoveObjOrientation(self):
        _, q_world_move_obj_cur = self.move_obj.get_world_pose()  # w,x,y,z
        rotate_obj_euler_delta = np.random.uniform(
            *self.skill_cfg.get("rotate_obj_euler_delta", [[0, 0, 0], [0, 0, 0]])
        )
        q_world_move_obj_tgt = (
            R.from_quat(q_world_move_obj_cur, scalar_first=True).as_matrix()
            @ R.from_euler("XYZ", rotate_obj_euler_delta, degrees=True).as_matrix()
        )
        q_world_move_obj_tgt = R.from_matrix(q_world_move_obj_tgt).as_quat(scalar_first=True)

        return q_world_move_obj_tgt

    def get_tgt_js(self):
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

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        command = self.manip_list[0]
        return self.command_complete(command)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        p_base_ee_cur, q_base_ee_cur = self.skill_runtime.ee_pose()
        distance = np.linalg.norm(p_base_ee_cur - self.p_base_ee_tgt)
        flag = (distance < self.success_threshold_move) and (len(self.manip_list) == 0)
        dot = np.clip(np.dot(q_base_ee_cur, self.q_base_ee_tgt), -1.0, 1.0)
        angle_diff = 2 * np.arccos(np.abs(dot))
        flag = (angle_diff < self.success_threshold_rotate) and flag
        return flag
