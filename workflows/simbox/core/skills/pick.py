import json
import os
import random
import time
from copy import deepcopy

import numpy as np
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.plan_utils import (
    select_index_by_priority_dual,
    select_index_by_priority_single,
)
from core.utils.transformation_utils import poses_from_tf_matrices
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
from omni.isaac.core.utils.xforms import get_world_pose


# pylint: disable=unused-argument
@register_skill
class Pick(BaseSkill):
    def __init__(self, robot: Robot, controller: BaseController, task: BaseTask, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.robot = robot
        self.controller = controller
        self.task = task
        self.skill_cfg = cfg
        object_name = self.skill_cfg["objects"][0]
        self.pick_obj = task.objects[object_name]
        self.output_root = str(self.skill_cfg.get("output_root", "output/ros_bridge/skills"))
        self.debug_tag = f"{robot.name}_pick_{object_name}_{int(time.time() * 1000)}"
        self.debug_dir = os.path.join(self.output_root, self.debug_tag)
        self._sample_debug = {}
        self._plan_debug_path = None
        self._runtime_failure_debug_path = None
        self._runtime_failure_snapshot_written = False
        self._selected_candidate_debug = {}
        self._mobile_base_prim_path = getattr(self.robot, "mobile_base_prim_path", None)
        self._cached_mobile_to_armbase_tf = None
        mount_prefix = "fr" if self.controller.robot_file and "right" in self.controller.robot_file else "fl"
        self._configured_mobile_to_armbase_translation = np.array(
            self.robot.cfg.get(f"{mount_prefix}_base_mount_translation", []), dtype=np.float32
        )
        self._configured_mobile_to_armbase_orientation = np.array(
            self.robot.cfg.get(f"{mount_prefix}_base_mount_orientation", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32
        )

        # Get grasp annotation
        usd_path = [obj["path"] for obj in task.cfg["objects"] if obj["name"] == object_name][0]
        usd_path = os.path.join(self.task.asset_root, usd_path)
        grasp_pose_path = usd_path.replace(
            "Aligned_obj.usd", self.skill_cfg.get("npy_name", "Aligned_grasp_sparse.npy")
        )
        sparse_grasp_poses = np.load(grasp_pose_path)
        lr_arm = "right" if "right" in self.controller.robot_file else "left"
        self.lr_arm = lr_arm
        self.T_obj_ee, self.scores = self.robot.pose_post_process_fn(
            sparse_grasp_poses,
            lr_arm=lr_arm,
            grasp_scale=self.skill_cfg.get("grasp_scale", 1),
            tcp_offset=self.skill_cfg.get("tcp_offset", self.robot.tcp_offset),
            constraints=self.skill_cfg.get("constraints", None),
        )

        # Keyposes should be generated after previous skill is done
        self.manip_list = []
        self.pickcontact_view = task.pickcontact_views[robot.name][lr_arm][object_name]
        self.process_valid = True
        self.obj_init_trans = deepcopy(self.pick_obj.get_local_pose()[0])
        final_gripper_state = self.skill_cfg.get("final_gripper_state", -1)
        if final_gripper_state == 1:
            self.gripper_cmd = "open_gripper"
        elif final_gripper_state == -1:
            self.gripper_cmd = "close_gripper"
        else:
            raise ValueError(f"final_gripper_state must be 1 or -1, got {final_gripper_state}")
        self.fixed_orientation = self.skill_cfg.get("fixed_orientation", None)
        if self.fixed_orientation is not None:
            self.fixed_orientation = np.array(self.fixed_orientation)

    def _get_armbase_transform_in_task(self):
        reference_prim_path = str(getattr(self.controller, "reference_prim_path", "")).strip()
        if reference_prim_path:
            reference_prim = get_prim_at_path(reference_prim_path)
            if reference_prim.IsValid():
                try:
                    reference_t, reference_q = get_world_pose(reference_prim_path)
                    world_armbase = tf_matrix_from_pose(reference_t, reference_q)
                    if hasattr(self.robot, "get_mobile_base_pose"):
                        try:
                            mobile_base_t, mobile_base_q = self.robot.get_mobile_base_pose()
                            world_mobile = tf_matrix_from_pose(mobile_base_t, mobile_base_q)
                            self._cached_mobile_to_armbase_tf = np.linalg.inv(world_mobile) @ world_armbase
                        except Exception:
                            pass
                    return world_armbase
                except Exception:
                    pass

        if self._configured_mobile_to_armbase_translation.shape == (3,):
            if hasattr(self.robot, "get_mobile_base_pose"):
                mobile_base_t, mobile_base_q = self.robot.get_mobile_base_pose()
            else:
                mobile_base_t, mobile_base_q = self.robot.get_world_pose()
            world_mobile = tf_matrix_from_pose(mobile_base_t, mobile_base_q)
            mobile_to_armbase = tf_matrix_from_pose(
                self._configured_mobile_to_armbase_translation,
                self._configured_mobile_to_armbase_orientation,
            )
            self._cached_mobile_to_armbase_tf = mobile_to_armbase
            return world_mobile @ mobile_to_armbase

        reference_prim = get_prim_at_path(self.controller.reference_prim_path)
        task_prim = get_prim_at_path(self.task.root_prim_path)
        raw_task_armbase = get_relative_transform(reference_prim, task_prim)

        mobile_base_prim_path = str(self._mobile_base_prim_path or "").strip()
        if not mobile_base_prim_path:
            return raw_task_armbase

        mobile_base_prim = get_prim_at_path(mobile_base_prim_path)
        if not mobile_base_prim.IsValid():
            return raw_task_armbase

        task_mobile = get_relative_transform(mobile_base_prim, task_prim)

        if self._cached_mobile_to_armbase_tf is None:
            self._cached_mobile_to_armbase_tf = np.linalg.inv(task_mobile) @ raw_task_armbase

        return task_mobile @ self._cached_mobile_to_armbase_tf

    def _get_object_world_pose(self):
        get_world_pose = getattr(self.pick_obj, "get_world_pose", None)
        if callable(get_world_pose):
            return get_world_pose()
        return self.pick_obj.get_local_pose()

    def _json_ready(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [self._json_ready(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._json_ready(v) for k, v in value.items()}
        return value

    def _write_debug_artifact(self, filename: str, payload: dict):
        os.makedirs(self.debug_dir, exist_ok=True)
        output_path = os.path.join(self.debug_dir, filename)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self._json_ready(payload), handle, indent=2, ensure_ascii=False)
        return output_path

    def _collect_geometry_debug(self):
        mobile_base_prim_path = str(self._mobile_base_prim_path or "").strip()
        obj_world_t, obj_world_q = self._get_object_world_pose()
        ee_base_t, ee_base_q = self.controller.get_ee_pose()
        robot_world_t, robot_world_q = self.robot.get_world_pose()
        reference_world_pose = None
        reference_prim_path = str(getattr(self.controller, "reference_prim_path", "")).strip()
        if reference_prim_path:
            reference_prim = get_prim_at_path(reference_prim_path)
            if reference_prim.IsValid():
                try:
                    reference_t, reference_q = get_world_pose(reference_prim_path)
                    reference_world_pose = {"translation": reference_t, "orientation": reference_q}
                except Exception:
                    reference_world_pose = None
        T_world_obj = tf_matrix_from_pose(obj_world_t, obj_world_q)
        T_world_armbase = self._get_armbase_transform_in_task()
        T_armbase_obj = np.linalg.inv(T_world_armbase) @ T_world_obj
        obj_armbase_t, obj_armbase_q = pose_from_tf_matrix(T_armbase_obj)
        mobile_base_pose = None
        if hasattr(self.robot, "get_mobile_base_pose"):
            try:
                mobile_base_t, mobile_base_q = self.robot.get_mobile_base_pose()
                mobile_base_pose = {"translation": mobile_base_t, "orientation": mobile_base_q}
            except Exception:
                mobile_base_pose = None
        return {
            "robot_world_pose": {"translation": robot_world_t, "orientation": robot_world_q},
            "mobile_base_world_pose": mobile_base_pose,
            "object_world_pose": {"translation": obj_world_t, "orientation": obj_world_q},
            "ee_armbase_pose": {"translation": ee_base_t, "orientation": ee_base_q},
            "object_armbase_pose": {"translation": obj_armbase_t, "orientation": obj_armbase_q},
            "reference_prim_path": self.controller.reference_prim_path,
            "reference_world_pose": reference_world_pose,
            "mobile_base_prim_path": mobile_base_prim_path if mobile_base_prim_path else None,
            "cached_mobile_to_armbase_tf": self._cached_mobile_to_armbase_tf,
            "configured_mobile_to_armbase_translation": self._configured_mobile_to_armbase_translation,
            "configured_mobile_to_armbase_orientation": self._configured_mobile_to_armbase_orientation,
            "controller_lr_name": getattr(self.controller, "lr_name", None),
            "controller_robot_file": self.controller.robot_file,
        }

    def _candidate_source_debug(self, candidate_index: int):
        sampled_indices = self._sample_debug.get("sampled_indices", [])
        sampled_scores = self._sample_debug.get("sampled_scores", [])
        source_index = sampled_indices[candidate_index] if candidate_index < len(sampled_indices) else None
        source_score = sampled_scores[candidate_index] if candidate_index < len(sampled_scores) else None
        return {
            "candidate_index": candidate_index,
            "source_index": source_index,
            "source_score": source_score,
        }

    def _manip_cmd_to_debug(self, manip_cmd):
        ee_trans, ee_ori, cmd_name, params = manip_cmd
        return {
            "command": cmd_name,
            "ee_translation": ee_trans,
            "ee_orientation": ee_ori,
            "params": params,
        }

    def simple_generate_manip_cmds(self):
        manip_list = []
        self._runtime_failure_snapshot_written = False
        self._runtime_failure_debug_path = None
        self._selected_candidate_debug = {}

        # Update
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        cmd = (p_base_ee_cur, q_base_ee_cur, "update_pose_cost_metric", {"hold_vec_weight": None})
        manip_list.append(cmd)

        ignore_substring = deepcopy(self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []))
        ignore_substring.append(self.pick_obj.name)
        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)

        # Pre grasp
        T_base_ee_grasps = self.sample_ee_pose()  # (N, 4, 4)
        T_base_ee_pregrasps = deepcopy(T_base_ee_grasps)
        self.controller.update_specific(
            ignore_substring=ignore_substring, reference_prim_path=self.controller.reference_prim_path
        )

        if "r5a" in self.controller.robot_file:
            T_base_ee_pregrasps[:, :3, 3] -= T_base_ee_pregrasps[:, :3, 0] * self.skill_cfg.get("pre_grasp_offset", 0.1)
        else:
            T_base_ee_pregrasps[:, :3, 3] -= T_base_ee_pregrasps[:, :3, 2] * self.skill_cfg.get("pre_grasp_offset", 0.1)

        if T_base_ee_grasps.shape[0] == 0:
            self.manip_list = []
            self._plan_debug_path = self._write_debug_artifact(
                "pick_plan_snapshot.json",
                {
                    "robot": self.robot.name,
                    "object": self.pick_obj.name,
                    "lr_arm": self.lr_arm,
                    "reason": "no_grasp_candidates_after_sampling",
                    "sample_debug": self._sample_debug,
                    "geometry_debug": self._collect_geometry_debug(),
                },
            )
            print(f"[pick-debug] No grasp candidates after sampling. Snapshot: {self._plan_debug_path}")
            return

        p_base_ee_pregrasps, q_base_ee_pregrasps = poses_from_tf_matrices(T_base_ee_pregrasps)
        p_base_ee_grasps, q_base_ee_grasps = poses_from_tf_matrices(T_base_ee_grasps)
        candidate_results = []
        success_found = False
        index = min(T_base_ee_grasps.shape[0] - 1, 0)

        if self.controller.use_batch:
            # Check if the input arrays are exactly the same
            if np.array_equal(p_base_ee_pregrasps, p_base_ee_grasps) and np.array_equal(
                q_base_ee_pregrasps, q_base_ee_grasps
            ):
                # Inputs are identical, compute only once to avoid redundant computation
                result = self.controller.test_batch_forward(p_base_ee_grasps, q_base_ee_grasps)
                index = select_index_by_priority_single(result)
                success_mask = np.asarray(result.success.detach().cpu().numpy()).reshape(-1).astype(bool)
                success_found = bool(success_mask.any())
            else:
                # Inputs are different, compute separately
                pre_result = self.controller.test_batch_forward(p_base_ee_pregrasps, q_base_ee_pregrasps)
                result = self.controller.test_batch_forward(p_base_ee_grasps, q_base_ee_grasps)
                index = select_index_by_priority_dual(pre_result, result)
                pre_success_mask = np.asarray(pre_result.success.detach().cpu().numpy()).reshape(-1).astype(bool)
                grasp_success_mask = np.asarray(result.success.detach().cpu().numpy()).reshape(-1).astype(bool)
                success_found = bool(np.logical_and(pre_success_mask, grasp_success_mask).any())
                candidate_results = [
                    {
                        **self._candidate_source_debug(i),
                        "pregrasp_success": bool(pre_success_mask[i]),
                        "grasp_success": bool(grasp_success_mask[i]),
                    }
                    for i in range(min(len(pre_success_mask), 128))
                ]
            if not candidate_results:
                candidate_results = [
                    {
                        **self._candidate_source_debug(i),
                        "success": bool(success_mask[i]),
                    }
                    for i in range(min(len(success_mask), 128))
                ]
        else:
            for index in range(T_base_ee_grasps.shape[0]):
                p_base_ee_pregrasp, q_base_ee_pregrasp = p_base_ee_pregrasps[index], q_base_ee_pregrasps[index]
                p_base_ee_grasp, q_base_ee_grasp = p_base_ee_grasps[index], q_base_ee_grasps[index]
                test_mode = self.skill_cfg.get("test_mode", "forward")
                if test_mode == "forward":
                    result_pre = self.controller.test_single_forward(p_base_ee_pregrasp, q_base_ee_pregrasp)
                elif test_mode == "ik":
                    result_pre = self.controller.test_single_ik(p_base_ee_pregrasp, q_base_ee_pregrasp)
                else:
                    raise NotImplementedError
                if self.skill_cfg.get("pre_grasp_offset", 0.1) > 0:
                    if test_mode == "forward":
                        result = self.controller.test_single_forward(p_base_ee_grasp, q_base_ee_grasp)
                    elif test_mode == "ik":
                        result = self.controller.test_single_ik(p_base_ee_grasp, q_base_ee_grasp)
                    else:
                        raise NotImplementedError
                    candidate_results.append(
                        {
                            **self._candidate_source_debug(index),
                            "pregrasp_success": bool(result_pre),
                            "grasp_success": bool(result),
                            "pregrasp_translation": p_base_ee_pregrasp,
                            "pregrasp_orientation": q_base_ee_pregrasp,
                            "grasp_translation": p_base_ee_grasp,
                            "grasp_orientation": q_base_ee_grasp,
                        }
                    )
                    if result == 1 and result_pre == 1:
                        print("pick plan success")
                        success_found = True
                        break
                else:
                    candidate_results.append(
                        {
                            **self._candidate_source_debug(index),
                            "pregrasp_success": bool(result_pre),
                            "pregrasp_translation": p_base_ee_pregrasp,
                            "pregrasp_orientation": q_base_ee_pregrasp,
                        }
                    )
                    if result_pre == 1:
                        print("pick plan success")
                        success_found = True
                        break

        if self.fixed_orientation is not None:
            q_base_ee_pregrasps[index] = self.fixed_orientation
            q_base_ee_grasps[index] = self.fixed_orientation

        self._selected_candidate_debug = {
            **self._candidate_source_debug(index),
            "success_found": success_found,
            "selected_pregrasp_translation": p_base_ee_pregrasps[index],
            "selected_pregrasp_orientation": q_base_ee_pregrasps[index],
            "selected_grasp_translation": p_base_ee_grasps[index],
            "selected_grasp_orientation": q_base_ee_grasps[index],
        }

        # Pre-grasp
        cmd = (p_base_ee_pregrasps[index], q_base_ee_pregrasps[index], "open_gripper", {})
        manip_list.append(cmd)
        if self.skill_cfg.get("pre_grasp_hold_vec_weight", None) is not None:
            cmd = (
                p_base_ee_pregrasps[index],
                q_base_ee_pregrasps[index],
                "update_pose_cost_metric",
                {"hold_vec_weight": self.skill_cfg.get("pre_grasp_hold_vec_weight", None)},
            )
            manip_list.append(cmd)

        # Grasp
        cmd = (p_base_ee_grasps[index], q_base_ee_grasps[index], "open_gripper", {})
        manip_list.append(cmd)
        cmd = (p_base_ee_grasps[index], q_base_ee_grasps[index], self.gripper_cmd, {})
        manip_list.extend(
            [cmd] * self.skill_cfg.get("gripper_change_steps", 40)
        )  # Default we use 40 steps to make sure the gripper is fully closed
        ignore_substring = deepcopy(self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []))
        cmd = (
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "update_specific",
            {"ignore_substring": ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)
        cmd = (
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "attach_obj",
            {"obj_prim_path": self.pick_obj.mesh_prim_path},
        )
        manip_list.append(cmd)

        # Post-grasp
        post_grasp_offset = np.random.uniform(
            self.skill_cfg.get("post_grasp_offset_min", 0.05), self.skill_cfg.get("post_grasp_offset_max", 0.05)
        )
        if post_grasp_offset:
            p_base_ee_postgrasps = deepcopy(p_base_ee_grasps)
            p_base_ee_postgrasps[index][2] += post_grasp_offset
            cmd = (p_base_ee_postgrasps[index], q_base_ee_grasps[index], self.gripper_cmd, {})
            manip_list.append(cmd)

        # Whether return to pre-grasp
        if self.skill_cfg.get("return_to_pregrasp", False):
            cmd = (p_base_ee_pregrasps[index], q_base_ee_pregrasps[index], self.gripper_cmd, {})
            manip_list.append(cmd)

        self.manip_list = manip_list
        self._plan_debug_path = self._write_debug_artifact(
            "pick_plan_snapshot.json",
            {
                "robot": self.robot.name,
                "object": self.pick_obj.name,
                "lr_arm": self.lr_arm,
                "success_found": success_found,
                "selected_candidate": self._selected_candidate_debug,
                "sample_debug": self._sample_debug,
                "geometry_debug": self._collect_geometry_debug(),
                "candidate_results": candidate_results,
                "manip_command_sequence": [self._manip_cmd_to_debug(cmd) for cmd in self.manip_list],
            },
        )
        print(f"[pick-debug] Wrote pick planning snapshot: {self._plan_debug_path}")
        if not success_found:
            print("[pick-debug] No candidate passed pregrasp+grasp screening; using fallback selected candidate.")

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
            T_world_obj = tf_matrix_from_pose(*self._get_object_world_pose())
            T_world_base = self._get_armbase_transform_in_task()
            T_base_world = np.linalg.inv(T_world_base)
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
            idx_list = list(range(min(max_length, num_pose)))
            sampled_scores = self.scores[idx_list]
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
            sampled_scores = [score for score, _ in sorted_pairs]

        valid_length = min(len(idx_list), num_pose)
        idx_list = idx_list[:valid_length]
        sampled_scores = sampled_scores[:valid_length]
        self._sample_debug = {
            "candidate_count": int(num_pose),
            "filtered_candidate_count": int(np.sum(combined_flag)),
            "filter_pass_counts": {axis: int(np.sum(flag)) for axis, flag in flags.items()},
            "sampled_indices": [int(idx) for idx in idx_list],
            "sampled_scores": [float(score) for score in sampled_scores],
            "max_length": int(max_length),
        }
        print(self.scores[idx_list])
        # print((T_base_ee[idx_list])[:, 0, 1])
        return T_base_ee[idx_list]

    def get_ee_poses(self, frame: str = "world"):
        # get grasp poses at specific frame
        if frame not in ["world", "body", "armbase"]:
            raise ValueError(
                f"poses in {frame} frame is not supported: accepted values are [world, body, armbase] only"
            )

        if frame == "body":
            return self.T_obj_ee

        T_world_obj = tf_matrix_from_pose(*self._get_object_world_pose())
        T_world_ee = T_world_obj[None] @ self.T_obj_ee

        if frame == "world":
            return T_world_ee

        if frame == "armbase":  # arm base frame
            T_world_base = self._get_armbase_transform_in_task()
            T_base_world = np.linalg.inv(T_world_base)
            T_base_ee = T_base_world[None] @ T_world_ee
            return T_base_ee

    def get_contact(self, contact_threshold=0.0):
        contact = np.abs(self.pickcontact_view.get_contact_force_matrix()).squeeze()
        contact = np.sum(contact, axis=-1)
        indices = np.where(contact > contact_threshold)[0]
        return contact, indices

    def is_feasible(self, th=5):
        feasible = self.controller.num_plan_failed <= th
        if not feasible and not self._runtime_failure_snapshot_written:
            current_cmd = self.manip_list[0] if self.manip_list else None
            self._runtime_failure_debug_path = self._write_debug_artifact(
                "pick_runtime_failure_snapshot.json",
                {
                    "robot": self.robot.name,
                    "object": self.pick_obj.name,
                    "lr_arm": self.lr_arm,
                    "num_plan_failed": int(self.controller.num_plan_failed),
                    "failure_threshold": int(th),
                    "num_last_cmd": int(self.controller.num_last_cmd),
                    "selected_candidate": self._selected_candidate_debug,
                    "geometry_debug": self._collect_geometry_debug(),
                    "current_command": self._manip_cmd_to_debug(current_cmd) if current_cmd is not None else None,
                    "plan_snapshot_path": self._plan_debug_path,
                },
            )
            self._runtime_failure_snapshot_written = True
            print(f"[pick-debug] Wrote pick runtime failure snapshot: {self._runtime_failure_debug_path}")
        return feasible

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert len(self.manip_list) != 0
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        p_base_ee, q_base_ee, *_ = self.manip_list[0]
        diff_trans = np.linalg.norm(p_base_ee_cur - p_base_ee)
        diff_ori = 2 * np.arccos(min(abs(np.dot(q_base_ee_cur, q_base_ee)), 1.0))
        pose_flag = np.logical_and(
            diff_trans < t_eps,
            diff_ori < o_eps,
        )
        self.plan_flag = self.controller.num_last_cmd > 10
        return np.logical_or(pose_flag, self.plan_flag)

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done(t_eps=self.skill_cfg.get("t_eps", 1e-3), o_eps=self.skill_cfg.get("o_eps", 5e-3)):
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        flag = True

        _, indices = self.get_contact()
        if self.gripper_cmd == "close_gripper":
            flag = len(indices) >= 1

        if self.skill_cfg.get("process_valid", True):
            self.process_valid = np.max(np.abs(self.robot.get_joints_state().velocities)) < 5 and (
                np.max(np.abs(self.pick_obj.get_linear_velocity())) < 5
            )
        flag = flag and self.process_valid

        if self.skill_cfg.get("lift_th", 0.0) > 0.0:
            p_world_obj = deepcopy(self.pick_obj.get_local_pose()[0])
            flag = flag and ((p_world_obj[2] - self.obj_init_trans[2]) > self.skill_cfg.get("lift_th", 0.0))

        return flag
