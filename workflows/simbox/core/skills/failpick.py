# pylint: skip-file
import os
from copy import deepcopy

import numpy as np
from core.planning.motion_command import MotionPhase
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.asset_path_utils import resolve_asset_path
from omegaconf import DictConfig
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.transformations import pose_from_tf_matrix, tf_matrix_from_pose


# pylint: disable=unused-argument
@register_skill
class FailPick(BaseSkill):
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
        self.object = task.objects[object_name]

        # Get grasp annotation
        object_cfg = next(obj for obj in task.cfg["objects"] if obj["name"] == object_name)
        usd_path = resolve_asset_path(self.task.asset_root, object_cfg)
        grasp_pose_path = usd_path.replace("Aligned_obj.usd", "Aligned_grasp_sparse.npy")
        sparse_grasp_poses = np.load(grasp_pose_path)
        lr_arm = self.skill_runtime.lr_name
        self.T_ee2o, self.scores = self.robot.pose_post_process_fn(sparse_grasp_poses, lr_arm=lr_arm)

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []
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
        poses = self.sample_ee_pose()
        pose = poses[0]
        grasp_trans, grasp_ori = pose_from_tf_matrix(pose)
        x_offset = np.random.choice(
            [
                np.random.uniform(
                    self.skill_cfg.get("grasp_x_offset_min", 0.05), self.skill_cfg.get("grasp_x_offset_max", 0.1)
                ),
                np.random.uniform(
                    -self.skill_cfg.get("grasp_x_offset_max", 0.1), -self.skill_cfg.get("grasp_x_offset_min", 0.05)
                ),
            ],
        )
        y_offset = np.random.choice(
            [
                np.random.uniform(
                    self.skill_cfg.get("grasp_y_offset_min", 0.05), self.skill_cfg.get("grasp_y_offset_max", 0.1)
                ),
                np.random.uniform(
                    -self.skill_cfg.get("grasp_y_offset_max", 0.1), -self.skill_cfg.get("grasp_y_offset_min", 0.05)
                ),
            ],
        )
        grasp_trans[0] += x_offset
        grasp_trans[1] += y_offset

        object_name = getattr(self.object, "name", None)
        manip_list.append(
            self.pose_command(
                MotionPhase.TERMINAL_GRASP_APPROACH,
                grasp_trans,
                grasp_ori,
                gripper_action="open_gripper",
                active_object=object_name,
            )
        )
        manip_list.append(
            self.pose_command(
                MotionPhase.GRIPPER_CLOSE,
                grasp_trans,
                grasp_ori,
                gripper_action="close_gripper",
                active_object=object_name,
                replan_allowed=False,
                dwell_steps=int(self.skill_cfg.get("gripper_change_steps", 10)),
            )
        )

        # Post grasp
        post_grasp_pose = pose.copy()
        post_grasp_pose[2, 3] += np.random.uniform(
            self.skill_cfg.get("post_grasp_offset_min", 0.05), self.skill_cfg.get("post_grasp_offset_max", 0.05)
        )
        post_position, post_orientation = pose_from_tf_matrix(post_grasp_pose)
        manip_list.append(
            self.pose_command(
                MotionPhase.POST_GRASP_LIFT,
                post_position,
                post_orientation,
                gripper_action="close_gripper",
                active_object=object_name,
                allow_target_finger_contact=True,
            )
        )

        self.manip_list = manip_list

    def sample_ee_pose(self, max_length=30):
        T_ee2r = self.get_ee_poses("armbase")

        num_pose = T_ee2r.shape[0]
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
                    flags[axis] = T_ee2r[:, row, col] >= cos_val if sign > 0 else T_ee2r[:, row, col] <= cos_val
                elif len(filter_list) == 3:
                    value1, value2 = filter_list[1:]
                    cos_val1 = np.cos(np.deg2rad(value1))
                    cos_val2 = np.cos(np.deg2rad(value2))
                    if sign > 0:
                        # flags[axis] = T_ee2r[:, row, col] >= cos_val1 and T_ee2r[:, row, col] <= cos_val2
                        flags[axis] = np.logical_and(T_ee2r[:, row, col] >= cos_val1, T_ee2r[:, row, col] <= cos_val2)
                    else:
                        # flags[axis] = T_ee2r[:, row, col] <= cos_val1 and T_ee2r[:, row, col] >= cos_val2
                        flags[axis] = np.logical_and(T_ee2r[:, row, col] <= cos_val1, T_ee2r[:, row, col] >= cos_val2)
        if self.skill_cfg.get("direction_to_obj", None) is not None:
            direction_to_obj = self.skill_cfg.direction_to_obj
            T_o2w = self._get_object_world_tf()
            T_w2r = np.linalg.inv(self._get_armbase_world_tf())
            T_o2r = T_w2r @ T_o2w
            if direction_to_obj == "right":
                flags["direction_to_obj"] = T_ee2r[:, 1, 3] <= T_o2r[1, 3]
            elif direction_to_obj == "left":
                flags["direction_to_obj"] = T_ee2r[:, 1, 3] > T_o2r[1, 3]
            else:
                raise NotImplementedError

        combined_flag = np.logical_and.reduce(list(flags.values()))
        if sum(combined_flag) == 0:
            # idx = np.random.choice(np.arange(num_pose), size=max_length, replace=True)
            idx = [0]
        else:
            tmp_scores = self.scores[combined_flag]
            tmp_idxs = np.arange(num_pose)[combined_flag]
            combined = list(zip(tmp_scores, tmp_idxs))
            combined.sort()
            idx = [idx for (score, idx) in combined[:max_length]]
            # idx = np.random.choice(np.arange(num_pose)[combined_flag], size=max_length, replace=True)
        return T_ee2r[idx]

    def get_ee_poses(self, frame: str = "world"):
        # get grasp poses at specific frame
        if frame not in ["world", "body", "armbase"]:
            raise ValueError(
                f"poses in {frame} frame is not supported: accepted values are [world, body, armbase] only"
            )

        if frame == "body":
            return self.T_ee2o

        T_o2w = self._get_object_world_tf()
        T_ee2w = T_o2w[None] @ self.T_ee2o

        if frame == "world":
            return T_ee2w

        if frame == "robot":  # robot base frame
            T_w2r = np.linalg.inv(self._get_armbase_world_tf())
            T_ee2r = T_w2r[None] @ T_ee2w
            return T_ee2r

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        return self.skill_runtime.phase_complete(self.manip_list[0])

    def is_record(self):
        return len(self.manip_list) < (1 * self.skill_cfg.get("gripper_change_steps", 10) + 2)

    def is_success(self):
        return True

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done(t_eps=self.skill_cfg.get("t_eps", 1e-3), o_eps=self.skill_cfg.get("o_eps", 5e-3)):
            # set_trace()
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_feasible(self, th=5):
        return int(
            getattr(
                self.skill_runtime,
                "plan_failure_count",
                getattr(self.skill_runtime, "num_plan_failed", 0),
            )
        ) <= th
