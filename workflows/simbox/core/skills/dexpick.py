import os
from copy import deepcopy

import numpy as np
from core.planning.motion_command import MotionPhase
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.asset_path_utils import resolve_asset_path
from omegaconf import DictConfig, OmegaConf
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.transformations import (
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)


# pylint: disable=unused-argument
@register_skill
class Dexpick(BaseSkill):
    def __init__(
        self,
        robot: Robot,
        skill_runtime,
        task: BaseTask,
        cfg: DictConfig,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self._require_skill_runtime()
        self.task = task
        if "world" in kwargs:
            self.world = kwargs["world"]
        self.skill_cfg = cfg
        object_name = self.skill_cfg["objects"][0]
        self.object = task.objects[object_name]

        # Get grasp annotation
        object_cfg = next(obj for obj in task.cfg["objects"] if obj["name"] == object_name)
        usd_path = resolve_asset_path(self.task.asset_root, object_cfg)
        dexpick_pose_path = usd_path.replace("Aligned_obj.usd", "dexpick_pose.yaml")
        self.pick_poses = []
        if os.path.exists(dexpick_pose_path):
            with open(dexpick_pose_path, "r", encoding="utf-8") as f:
                pick_data = OmegaConf.load(f)
                pick_poses = pick_data.pick_poses
                for pick_pose in pick_poses:
                    self.pick_poses.append((np.array(pick_pose[:3]), np.array(pick_pose[3:])))

            self.pick_pose_idx = cfg.get("pick_pose_idx", 0)
            self.pose_ee2o = self.pick_poses[self.pick_pose_idx]
        self.manip_list = []
        lr_arm = self.skill_runtime.arm_name
        self.pickcontact_view = task.pickcontact_views[robot.name][lr_arm][object_name]
        self.process_valid = True
        self.obj_init_trans = deepcopy(self.object.get_local_pose()[0])

    def _get_armbase_world_tf(self):
        return self.skill_runtime.arm_base_transform()

    def _get_object_world_tf(self):
        get_obj_world_pose = getattr(self.object, "get_world_pose", None)
        if callable(get_obj_world_pose):
            return tf_matrix_from_pose(*get_obj_world_pose())
        return tf_matrix_from_pose(*self.object.get_local_pose())

    def simple_generate_manip_cmds(self):
        manip_list = []

        T_world_base = self._get_armbase_world_tf()

        # Reach
        T_world_obj = self._get_object_world_tf()
        T_obj_ee_grasp = tf_matrix_from_pose(*self.pose_ee2o)
        T_world_ee_grasp = T_world_obj @ T_obj_ee_grasp
        T_base_ee_grasp = np.linalg.inv(T_world_base) @ T_world_ee_grasp
        p_base_ee_grasp, q_base_ee_grasp = pose_from_tf_matrix(T_base_ee_grasp)

        # Pre grasp
        pre_grasp_offset = self.skill_cfg.get("pre_grasp_offset", 0.1)
        if pre_grasp_offset:
            T_base_ee_pregrasp = T_base_ee_grasp.copy()
            approach_axis = getattr(self.skill_runtime, "grasp_approach_axis", 2)
            T_base_ee_pregrasp[0:3, 3] -= (
                T_base_ee_pregrasp[0:3, approach_axis] * pre_grasp_offset
            )

            p_pre, q_pre = pose_from_tf_matrix(T_base_ee_pregrasp)
            manip_list.append(
                self.pose_command(
                    MotionPhase.TRANSIT_PREGRASP,
                    p_pre,
                    q_pre,
                    gripper_action="open_gripper",
                    active_object=getattr(self.object, "name", None),
                )
            )

        # Grasp
        manip_list.append(
            self.pose_command(
                MotionPhase.TERMINAL_GRASP_APPROACH,
                p_base_ee_grasp,
                q_base_ee_grasp,
                gripper_action="open_gripper",
                active_object=getattr(self.object, "name", None),
            )
        )
        manip_list.append(
            self.pose_command(
                MotionPhase.GRIPPER_CLOSE,
                p_base_ee_grasp,
                q_base_ee_grasp,
                gripper_action="close_gripper",
                active_object=getattr(self.object, "name", None),
                replan_allowed=False,
                dwell_steps=int(self.skill_cfg.get("gripper_change_steps", 40)),
            )
        )

        # Post grasp
        post_grasp_offset = np.random.uniform(
            self.skill_cfg.get("post_grasp_offset_min", 0.05), self.skill_cfg.get("post_grasp_offset_max", 0.05)
        )
        if post_grasp_offset:
            p_base_ee_postgrasp = deepcopy(p_base_ee_grasp)
            p_base_ee_postgrasp[2] += post_grasp_offset
            manip_list.append(
                self.pose_command(
                    MotionPhase.POST_GRASP_LIFT,
                    p_base_ee_postgrasp,
                    q_base_ee_grasp,
                    gripper_action="close_gripper",
                    active_object=getattr(self.object, "name", None),
                    allow_target_finger_contact=True,
                )
            )

        self.manip_list = manip_list

    def get_contact(self, contact_threshold=0.0):
        contact = np.abs(self.pickcontact_view.get_contact_force_matrix()).squeeze()
        contact = np.sum(contact, axis=-1)
        indices = np.where(contact > contact_threshold)[0]
        return contact, indices

    def is_feasible(self, th=10):
        return int(self.skill_runtime.num_plan_failed) <= th

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        return self.skill_runtime.phase_complete(self.manip_list[0])

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done(t_eps=self.skill_cfg.get("t_eps", 1e-3), o_eps=self.skill_cfg.get("o_eps", 5e-3)):
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        _, indices = self.get_contact()
        flag = len(indices) >= 1

        if self.skill_cfg.get("process_valid", True):
            self.process_valid = np.max(np.abs(self.robot.get_joints_state().velocities)) < 5 and (
                np.max(np.abs(self.object.get_linear_velocity())) < 5
            )

        flag = flag and self.process_valid

        if self.skill_cfg.get("lift_th", 0.0) > 0.0:
            obj_curr_trans = deepcopy(self.object.get_local_pose()[0])
            flag = flag and ((obj_curr_trans[2] - self.obj_init_trans[2]) > self.skill_cfg.get("lift_th", 0.0))

        return flag
