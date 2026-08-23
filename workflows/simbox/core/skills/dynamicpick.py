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
from omni.timeline import get_timeline_interface
from scipy.spatial.transform import Rotation as R


# pylint: disable=unused-argument
@register_skill
class Dynamicpick(BaseSkill):
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
        self.predict_pick = False
        self.meet_pose_o2w = None
        self.grasp_scale = self.skill_cfg.get("grasp_scale", 1)
        self.tcp_offset = self.skill_cfg.get("tcp_offset", 0.125)

        # Get grasp annotation
        object_cfg = next(obj for obj in task.cfg["objects"] if obj["name"] == object_name)
        usd_path = resolve_asset_path(self.task.asset_root, object_cfg)
        grasp_pose_path = usd_path.replace("Aligned_obj.usd", "Aligned_grasp_sparse.npy")
        sparse_grasp_poses = np.load(grasp_pose_path)
        lr_arm = self.skill_runtime.arm_name
        self.T_obj_ee, self.scores = self.robot.pose_post_process_fn(
            sparse_grasp_poses, lr_arm=lr_arm, grasp_scale=self.grasp_scale, tcp_offset=self.tcp_offset
        )
        self.robot_name = getattr(
            self.skill_runtime, "robot_name", getattr(self.skill_runtime, "name", robot.name)
        )
        self.object_name = object_name

        # !!! keyposes should be generated after previous skill is done
        self.manip_list = []
        self.pickcontact_view = task.pickcontact_views[robot.name][lr_arm][object_name]
        self.cmd_time = 0
        self.delta_x = np.random.uniform(self.skill_cfg["pick_range"][0], self.skill_cfg["pick_range"][1])
        self.time_bias = self.skill_cfg.get("time_bias", 0)
        self.pick_bias = self.skill_cfg.get("pick_bias", 0)
        self.process_valid = True
        self.obj_init_trans = deepcopy(self.pick_obj.get_local_pose()[0])

    def _get_armbase_world_tf(self):
        return self.skill_runtime.arm_base_transform()

    def _get_object_world_tf(self):
        if self.meet_pose_o2w is not None:
            return tf_matrix_from_pose(*self.meet_pose_o2w)
        get_obj_world_pose = getattr(self.pick_obj, "get_world_pose", None)
        if callable(get_obj_world_pose):
            return tf_matrix_from_pose(*get_obj_world_pose())
        return tf_matrix_from_pose(*self.pick_obj.get_local_pose())

    def simple_generate_manip_cmds(self):
        """Build the initial typed sequence without freezing a moving target.

        ``is_ready`` may replace this preview after it computes
        ``meet_pose_o2w`` for the conveyor intercept.  Keeping generation in
        ``predict_manip_cmds`` ensures both paths emit the same typed phase
        commands and preserve the dynamic-target callback semantics.
        """

        self.predict_manip_cmds()
        return self.manip_list

    def _preview_pose(self, position, orientation, start_arm_positions=None, *, ds_ratio=1):
        """Preview a typed pose plan without executing a reflected command."""

        success, end_positions, result = self.skill_runtime.plan_pose_from_joint_positions(
            position,
            orientation,
            start_arm_positions=start_arm_positions,
            active_target=self.object_name,
        )
        if not success:
            self.skill_runtime.num_plan_failed = 1000
            return 0.0, start_arm_positions
        path = result.trajectory
        waypoints = len(path) if path is not None else 1
        stride = max(1, int(ds_ratio))
        duration = (
            waypoints
            * self.skill_runtime.interpolation_dt
            / float(getattr(self.skill_runtime, "time_dilation_factor", 1.0))
            / stride
        )
        return duration, end_positions

    def predict_manip_cmds(self):
        # Prediction can be refreshed when the conveyor intercept is known;
        # timing must describe the latest sequence rather than accumulate
        # across the initial preview and the dynamic-target refresh.
        self.cmd_time = 0.0
        manip_list = []

        p_base_ee_cur, q_base_ee_cur = self.skill_runtime.ee_pose()

        cmd_time, expected_js = self._preview_pose(p_base_ee_cur, q_base_ee_cur)
        self.cmd_time += cmd_time

        # Pre grasp
        T_base_ee_grasps = self.sample_ee_pose()  # (N, 4, 4)
        # Batch grasp pose adjustment if needed (operate on all T_base_ee_grasps at once)
        if self.skill_cfg.get("pivot_angle_z", None) is not None:
            num_grasps = T_base_ee_grasps.shape[0]
            # sample per-grasp pivot angles
            pivot_angles_z = np.random.uniform(
                self.skill_cfg["pivot_angle_z"][0],
                self.skill_cfg["pivot_angle_z"][1],
                size=num_grasps,
            )
            # compute batch rotation matrices R_z(-pivot_angle_z)
            pivot_rotations = R.from_euler("z", -pivot_angles_z, degrees=True).as_matrix()  # (N, 3, 3)
            # apply rotations to all rotation blocks
            T_base_ee_grasps[:, :3, :3] = np.einsum("nij,njk->nik", T_base_ee_grasps[:, :3, :3], pivot_rotations)
            # sample per-grasp z translation adjustments
            pos_adjust_z = np.random.uniform(
                self.skill_cfg["pos_adjust_z"][0],
                self.skill_cfg["pos_adjust_z"][1],
                size=num_grasps,
            )
            T_base_ee_grasps[:, 2, 3] += pos_adjust_z
        T_base_ee_pregrasps = deepcopy(T_base_ee_grasps)

        approach_axis = getattr(self.skill_runtime, "grasp_approach_axis", 2)
        T_base_ee_pregrasps[:, :3, 3] -= (
            T_base_ee_pregrasps[:, :3, approach_axis]
            * self.skill_cfg.get("pre_grasp_offset", 0.1)
        )

        p_base_ee_pregrasps, q_base_ee_pregrasps = poses_from_tf_matrices(T_base_ee_pregrasps)
        p_base_ee_grasps, q_base_ee_grasps = poses_from_tf_matrices(T_base_ee_grasps)

        # Candidate ranking is a typed planner concern.  Execution consumes
        # the selected pose from measured state and owns collision validation.
        index = 0

        # Pre-grasp
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
        cmd_time, expected_js = self._preview_pose(
            p_base_ee_pregrasps[index], q_base_ee_pregrasps[index], expected_js, ds_ratio=2
        )
        self.cmd_time += cmd_time

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
        cmd_time, expected_js = self._preview_pose(
            p_base_ee_grasps[index], q_base_ee_grasps[index], expected_js, ds_ratio=2
        )
        self.cmd_time += cmd_time
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

        # Post grasp
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
                    gripper_action="close_gripper",
                    active_object=object_name,
                    allow_target_finger_contact=True,
                )
            )

        self.manip_list = manip_list
        self.cmd_time += self.time_bias

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
            direction_to_obj = self.skill_cfg["direction_to_obj"]
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
        return T_base_ee[idx_list]

    def get_ee_poses(self, frame: str = "world"):
        # get grasp poses at specific frame
        if frame not in ["world", "body", "armbase"]:
            raise ValueError(
                f"poses in {frame} frame is not supported: accepted values are [world, body, armbase] only"
            )

        if frame == "body":
            return self.T_obj_ee

        T_world_obj = self._get_object_world_tf()
        T_world_ee = T_world_obj[None] @ self.T_obj_ee

        if frame == "world":
            return T_world_ee

        if frame == "armbase":  # arm base frame
            T_base_world = np.linalg.inv(self._get_armbase_world_tf())
            T_base_ee = T_base_world[None] @ T_world_ee
            return T_base_ee

    def is_ready(self):
        get_obj_world_pose = getattr(self.pick_obj, "get_world_pose", None)
        if callable(get_obj_world_pose):
            object_position, object_orientation = get_obj_world_pose()
        else:
            object_position, object_orientation = self.pick_obj.get_local_pose()
        initial_position, initial_orientation = self.skill_runtime.initial_ee_pose()
        T_world_ee_init = self._get_armbase_world_tf() @ tf_matrix_from_pose(
            initial_position, initial_orientation
        )
        ee_init_position = deepcopy(T_world_ee_init[0:3, 3])
        x = object_position[0] - ee_init_position[0]
        self.obj_velocity = self.task.conveyor_velocity
        if (self.obj_velocity < 0 and x < 0.5) or (self.obj_velocity > 0 and x > -0.5):
            if not self.predict_pick:
                print(f"###{self.robot_name} PREDICTING {self.object_name}###")
                position = deepcopy(object_position)
                delta_x = self.delta_x
                position[0] = ee_init_position[0] + delta_x
                self.meet_pose_o2w = (position, deepcopy(object_orientation))
                self.predict_manip_cmds()
                self.epsilon = delta_x - (self.cmd_time * self.obj_velocity) + self.pick_bias
                self.predict_pick = True
            if (self.obj_velocity < 0 and x < self.epsilon) or (
                self.obj_velocity > 0 and x > self.epsilon
            ):  # start real pick
                return True
            else:
                return False
        else:
            return False

    def get_obj_velocity(self, x):
        timeline = get_timeline_interface()
        current_time = timeline.get_current_time()
        previous_time = getattr(self, "_previous_time", None)
        previous_x = getattr(self, "_previous_x", None)
        if previous_time is not None and previous_x is not None:
            time_delta = current_time - previous_time
            if time_delta > 0:
                x_velocity = (x - previous_x) / time_delta
            else:
                x_velocity = 0
        else:
            x_velocity = 0

        self._previous_time = current_time
        self._previous_x = x

        return x_velocity

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
        if not self.is_ready():
            return False
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
                np.max(np.abs(self.pick_obj.get_linear_velocity())) < 5
            )

        flag = flag and self.process_valid

        if self.skill_cfg.get("lift_th", 0.0) > 0.0:
            obj_curr_trans = deepcopy(self.pick_obj.get_local_pose()[0])
            flag = flag and ((obj_curr_trans[2] - self.obj_init_trans[2]) > self.skill_cfg.get("lift_th", 0.0))

        return flag
