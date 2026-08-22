# pylint: skip-file
import os
import random
from copy import deepcopy

import numpy as np
from core.planning.motion_command import MotionPhase
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.asset_path_utils import resolve_asset_path
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.transformation_utils import poses_from_tf_matrices
from omegaconf import DictConfig
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.transformations import tf_matrix_from_pose


@register_skill
class Manualpick(BaseSkill):
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
        self.skill_cfg = cfg
        object_name = self.skill_cfg["objects"][0]
        self.pick_obj = task.objects[object_name]

        # Get grasp annotation
        object_cfg = next(obj for obj in task.cfg["objects"] if obj["name"] == object_name)
        usd_path = resolve_asset_path(self.task.asset_root, object_cfg)
        grasp_pose_path = usd_path.replace(
            "Aligned_obj.usd", self.skill_cfg.get("npy_name", "Aligned_grasp_sparse.npy")
        )
        sparse_grasp_poses = np.load(grasp_pose_path)
        grasp_scale = self.skill_cfg.get("grasp_scale", 1)
        lr_arm = self.skill_runtime.lr_name
        self.T_obj_ee, self.scores = self.robot.pose_post_process_fn(
            sparse_grasp_poses, lr_arm=lr_arm, grasp_scale=grasp_scale
        )

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []
        self.pickcontact_view = task.pickcontact_views[robot.name][lr_arm][object_name]

        # Keep the picked object secured during post-grasp motion.  This
        # mirrors Pick's final-gripper-state contract; manualpick always
        # defaults to a closed gripper unless the task explicitly requests
        # the legacy open-final state.
        final_gripper_state = self.skill_cfg.get("final_gripper_state", -1)
        if final_gripper_state == 1:
            self.gripper_cmd = "open_gripper"
        elif final_gripper_state == -1:
            self.gripper_cmd = "close_gripper"
        else:
            raise ValueError(f"final_gripper_state must be 1 or -1, got {final_gripper_state}")

    def _get_armbase_world_tf(self):
        return self.skill_runtime.arm_base_transform()

    def _get_object_world_tf(self):
        get_obj_world_pose = getattr(self.pick_obj, "get_world_pose", None)
        if callable(get_obj_world_pose):
            return tf_matrix_from_pose(*get_obj_world_pose())
        return tf_matrix_from_pose(*self.pick_obj.get_local_pose())

    def simple_generate_manip_cmds(self):
        manip_list = []

        # Pre grasp
        T_base_ee_grasps = self.sample_ee_pose()  # (N, 4, 4)
        adjust_ori = self.skill_cfg.get("adjust_ori", None)
        if adjust_ori:
            pose_axis = adjust_ori[0]
            base_axis = adjust_ori[1]
            judge_flag = adjust_ori[2]
            axis_index = {"x": 0, "y": 1, "z": 2}
            rotate_axis = self.skill_cfg.get("adjust_rotate_axis", "x")

            if getattr(self.skill_runtime, "orientation_adjustment_enabled", True):
                num_poses = T_base_ee_grasps.shape[0]
                adjust_angle_list_cfg = self.skill_cfg.get("adjust_angle_list_cfg", [-15, 15, 7])
                adjust_angle_list = np.linspace(
                    adjust_angle_list_cfg[0], adjust_angle_list_cfg[1], adjust_angle_list_cfg[2]
                )  # (K,)

                # build batch rotation matrices of shape (K, 4, 4)
                thetas = np.radians(adjust_angle_list)
                rots = []
                for theta in thetas:
                    if rotate_axis == "x":
                        rot = np.array(
                            [
                                [1, 0, 0, 0],
                                [0, np.cos(theta), -np.sin(theta), 0],
                                [0, np.sin(theta), np.cos(theta), 0],
                                [0, 0, 0, 1],
                            ]
                        )
                    elif rotate_axis == "y":
                        rot = np.array(
                            [
                                [np.cos(theta), 0, np.sin(theta), 0],
                                [0, 1, 0, 0],
                                [-np.sin(theta), 0, np.cos(theta), 0],
                                [0, 0, 0, 1],
                            ]
                        )
                    elif rotate_axis == "z":
                        rot = np.array(
                            [
                                [np.cos(theta), -np.sin(theta), 0, 0],
                                [np.sin(theta), np.cos(theta), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1],
                            ]
                        )
                    else:
                        rot = np.eye(4)
                    rots.append(rot)
                rots = np.stack(rots, axis=0)  # (K, 4, 4)

                # original poses: (N, 4, 4), broadcast with rotations: (K, 4, 4)
                original_poses = T_base_ee_grasps.copy()
                rotated_poses = original_poses[:, None, :, :] @ rots[None, :, :, :]  # (N, K, 4, 4)

                # compute metric for each (pose, angle) candidate
                base_idx = axis_index[base_axis]
                pose_idx = axis_index[pose_axis]
                current_values = rotated_poses[:, :, base_idx, pose_idx]  # (N, K)

                if judge_flag == "min":
                    best_indices = np.argmin(current_values, axis=1)
                else:
                    best_indices = np.argmax(current_values, axis=1)

                # gather best poses per grasp
                idx_rows = np.arange(num_poses)
                best_poses = rotated_poses[idx_rows, best_indices]  # (N, 4, 4)
                T_base_ee_grasps = best_poses

        manual_adjust_ori = self.skill_cfg.get("manual_adjust_ori", None)
        if manual_adjust_ori:
            for adjust_ori in manual_adjust_ori:
                rotate_axis = adjust_ori[0]
                angle = adjust_ori[1]
                theta = np.radians(angle)
                if rotate_axis == "x":
                    rot = np.array(
                        [
                            [1, 0, 0, 0],
                            [0, np.cos(theta), -np.sin(theta), 0],
                            [0, np.sin(theta), np.cos(theta), 0],
                            [0, 0, 0, 1],
                        ]
                    )
                elif rotate_axis == "y":
                    rot = np.array(
                        [
                            [np.cos(theta), 0, np.sin(theta), 0],
                            [0, 1, 0, 0],
                            [-np.sin(theta), 0, np.cos(theta), 0],
                            [0, 0, 0, 1],
                        ]
                    )
                elif rotate_axis == "z":
                    rot = np.array(
                        [
                            [np.cos(theta), -np.sin(theta), 0, 0],
                            [np.sin(theta), np.cos(theta), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ]
                    )
                else:
                    rot = np.eye(4)
                # apply the same rotation to all grasps in batch
                T_base_ee_grasps = T_base_ee_grasps @ rot

        adjust_trans_offset = self.skill_cfg.get("adjust_trans_offset", [0, 0, 0])
        T_base_ee_grasps[:, :3, 3] += adjust_trans_offset
        T_base_ee_pregrasps = deepcopy(T_base_ee_grasps)
        approach_axis = getattr(self.skill_runtime, "grasp_approach_axis", 2)
        T_base_ee_pregrasps[:, :3, 3] -= (
            T_base_ee_pregrasps[:, :3, approach_axis]
            * self.skill_cfg.get("pre_grasp_offset", 0.1)
        )
        pre_grasp_offset_manual = self.skill_cfg.get("pre_grasp_offset_manual", None)
        if pre_grasp_offset_manual:
            T_base_ee_pregrasps[:, :3, 3] += np.array(pre_grasp_offset_manual)

        p_base_ee_pregrasps, q_base_ee_pregrasps = poses_from_tf_matrices(T_base_ee_pregrasps)
        p_base_ee_grasps, q_base_ee_grasps = poses_from_tf_matrices(T_base_ee_grasps)

        # The typed runtime owns candidate planning.  Keep the selected
        # candidate as a deterministic value; execution performs the actual
        # collision-aware plan from measured state.
        index = 0
        object_name = getattr(self.pick_obj, "name", None)
        manip_list.append(
            self.pose_command(
                MotionPhase.TRANSIT_PREGRASP,
                p_base_ee_pregrasps[index],
                q_base_ee_pregrasps[index],
                gripper_action="open_gripper",
                active_object=object_name,
            )
        )

        # Grasp
        manip_list.append(
            self.pose_command(
                MotionPhase.TERMINAL_GRASP_APPROACH,
                p_base_ee_grasps[index],
                q_base_ee_grasps[index],
                gripper_action="open_gripper",
                active_object=object_name,
            )
        )

        manip_list.append(
            self.pose_command(
                MotionPhase.GRIPPER_CLOSE,
                p_base_ee_grasps[index],
                q_base_ee_grasps[index],
                gripper_action="close_gripper",
                active_object=object_name,
                replan_allowed=False,
                dwell_steps=int(self.skill_cfg.get("gripper_change_steps", 40)),
            )
        )


        # Post-grasp
        post_grasp_offset = np.random.uniform(
            self.skill_cfg.get("post_grasp_offset_min", 0.05), self.skill_cfg.get("post_grasp_offset_max", 0.05)
        )
        if post_grasp_offset:
            p_base_ee_postgrasps = deepcopy(p_base_ee_grasps)
            p_base_ee_postgrasps[index][2] += post_grasp_offset
            manip_list.append(
                self.pose_command(
                    MotionPhase.POST_GRASP_LIFT,
                    p_base_ee_postgrasps[index],
                    q_base_ee_grasps[index],
                    gripper_action=self.gripper_cmd,
                    active_object=object_name,
                    allow_target_finger_contact=True,
                )
            )

        self.manip_list = manip_list

    def sample_ee_pose(self, max_length=CUROBO_BATCH_SIZE):
        T_base_ee = self.get_ee_poses("armbase")

        num_pose = T_base_ee.shape[0]
        flags = {
            "x": np.ones(num_pose, dtype=bool),
            "y": np.ones(num_pose, dtype=bool),
            "z": np.ones(num_pose, dtype=bool),
            "direction_to_obj": np.ones(num_pose, dtype=bool),
        }
        filter_conditions = {
            "x": {
                "forward": (0, 0, 1),  # (row, col, direction)
                "backward": (0, 0, -1),
                "upward": (2, 0, 1),
                "downward": (2, 0, -1),
            },
            "y": {"forward": (0, 1, 1), "backward": (0, 1, -1), "downward": (2, 1, -1), "upward": (2, 1, 1)},
            "z": {"forward": (0, 2, 1), "backward": (0, 2, -1), "downward": (2, 2, -1), "upward": (2, 2, 1)},
        }
        for axis in ["x", "y", "z"]:
            filter_list = self.skill_cfg.get(f"filter_{axis}_dir", None)
            if filter_list is not None:
                # direction, value = filter_list
                direction = filter_list[0]
                row, col, sign = filter_conditions[axis][direction]
                if len(filter_list) == 2:
                    value = filter_list[1]
                    cos_val = np.cos(np.deg2rad(value))
                    flags[axis] = T_base_ee[:, row, col] >= cos_val if sign > 0 else T_base_ee[:, row, col] <= cos_val
                elif len(filter_list) == 3:
                    value1, value2 = filter_list[1:]
                    cos_val1 = np.cos(np.deg2rad(value1))
                    cos_val2 = np.cos(np.deg2rad(value2))
                    if sign > 0:
                        flags[axis] = np.logical_and(
                            T_base_ee[:, row, col] >= cos_val1, T_base_ee[:, row, col] <= cos_val2
                        )
                    else:
                        flags[axis] = np.logical_and(
                            T_base_ee[:, row, col] <= cos_val1, T_base_ee[:, row, col] >= cos_val2
                        )
        if self.skill_cfg.get("direction_to_obj", None) is not None:
            direction_to_obj = self.skill_cfg.direction_to_obj
            T_world_obj = self._get_object_world_tf()
            T_base_world = np.linalg.inv(self._get_armbase_world_tf())
            T_base_obj = T_base_world @ T_world_obj
            if direction_to_obj == "right":
                flags["direction_to_obj"] = T_base_ee[:, 1, 3] <= T_base_obj[1, 3]
            elif direction_to_obj == "left":
                flags["direction_to_obj"] = T_base_ee[:, 1, 3] > T_base_obj[1, 3]
            else:
                raise NotImplementedError

        combined_flag = np.logical_and.reduce(list(flags.values()))
        if sum(combined_flag) == 0:
            # idx_list = [i for i in range(max_length)]
            idx_list = list(range(max_length))
        else:
            tmp_scores = self.scores[combined_flag]
            tmp_idxs = np.arange(num_pose)[combined_flag]
            combined = list(zip(tmp_scores, tmp_idxs))
            combined.sort()
            idx_list = [idx for (score, idx) in combined[:max_length]]
            score_list = self.scores[idx_list]
            weights = 1.0 / (score_list + 1e-8)
            weights = weights / weights.sum()

            sampled_idx = random.choices(idx_list, weights=weights, k=max_length)
            sampled_scores = self.scores[sampled_idx]

            # Sort indices by their scores (ascending)
            sorted_pairs = sorted(zip(sampled_scores, sampled_idx))
            idx_list = [idx for _, idx in sorted_pairs]

        print(self.scores[idx_list])
        # print((T_base_ee[idx_list])[:, 0, 1])
        return T_base_ee[idx_list]

    def get_ee_poses(self, frame: str = "world"):
        # get grasp poses at specific frame
        if frame not in ["world", "body", "armbase"]:
            raise Exception(
                "poses in {} frame is not supported: accepted values are [world, body, armbase] only".format(frame)
            )

        if frame == "body":
            return self.T_obj_ee

        T_world_obj = self._get_object_world_tf()
        T_world_ee = T_world_obj[None] @ self.T_obj_ee

        if frame == "world":
            return T_world_ee

        if frame == "armbase":  # robot base frame
            T_base_world = np.linalg.inv(self._get_armbase_world_tf())
            T_base_ee = T_base_world[None] @ T_world_ee
            return T_base_ee

    def get_contact(self, contact_threshold=0.0):
        contact = np.abs(self.pickcontact_view.get_contact_force_matrix()).squeeze()
        contact = np.sum(contact, axis=-1)
        indices = np.where(contact > contact_threshold)[0]
        return contact, indices

    def is_feasible(self, th=5):
        return int(
            getattr(
                self.skill_runtime,
                "plan_failure_count",
                getattr(self.skill_runtime, "num_plan_failed", 0),
            )
        ) <= th

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        return self.skill_runtime.phase_complete(self.manip_list[0])

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done():
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        contact, indices = self.get_contact()
        return len(indices) >= 1
