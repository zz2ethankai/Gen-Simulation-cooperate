import json
import logging
import os
import random
import time
from copy import deepcopy

import numpy as np
from core.planning.grasp_plan_evaluator import GraspPlanEvaluator
from core.planning.config_contract import resolve_skill_test_mode
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.json_utils import json_ready
from core.utils.plan_utils import select_index_by_priority_dual
from core.utils.transformation_utils import poses_from_tf_matrices
from core.utils.asset_path_utils import resolve_asset_path
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

LOGGER = logging.getLogger("de_logger")


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
        self.output_root = str(self.skill_cfg.get("output_root", "output/local_navigation/skills"))
        self.debug_tag = f"{robot.name}_pick_{object_name}_{int(time.time() * 1000)}"
        self.debug_dir = os.path.join(self.output_root, self.debug_tag)
        self._sample_debug = {}
        self._plan_debug_path = None
        self._runtime_failure_debug_path = None
        self._runtime_failure_snapshot_written = False
        self._success_check_debug_path = None
        self._success_check_snapshot_written = False
        self._last_success_check_debug = {}
        self._execution_trace = []
        self._execution_trace_path = None
        self._execution_trace_write_stride = int(self.skill_cfg.get("execution_trace_write_stride", 250))
        self._execution_trace_max_steps = int(self.skill_cfg.get("execution_trace_max_steps", 500))
        self._execution_trace_total_steps = 0
        self._stalled_command_step_limit = int(self.skill_cfg.get("stalled_command_step_limit", 450))
        self._last_stall_command_signature = None
        self._last_stall_command_started_at = 0
        self._stalled_failure_written = False
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
        object_cfg = next(obj for obj in task.cfg["objects"] if obj["name"] == object_name)
        # Annotation t/depth are in the object model's native units; scale to scene
        # units using the object's declared annotation scale (previously never applied).
        ann_scale = float(object_cfg.get("grasp_annotation_scale", [1, 1, 1])[0])
        usd_path = resolve_asset_path(self.task.asset_root, object_cfg)
        grasp_pose_path = usd_path.replace(
            "Aligned_obj.usd", self.skill_cfg.get("npy_name", "Aligned_grasp_sparse.npy")
        )
        sparse_grasp_poses = np.load(grasp_pose_path)
        lr_arm = getattr(self.controller, "lr_name", None) or (
            "right" if "right" in self.controller.robot_file else "left"
        )
        self.lr_arm = lr_arm
        self.T_obj_ee, self.scores = self.robot.pose_post_process_fn(
            sparse_grasp_poses,
            lr_arm=lr_arm,
            grasp_scale=self.skill_cfg.get("grasp_scale", ann_scale),
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
        self.debug = self.skill_cfg.get("debug", False) or os.environ.get("SIMBOX_DEBUG_PICK") == "1"
        self.plan_evaluation = None
        self.sampled_scores = np.empty((0,), dtype=float)
        self.failure_reason = ""
        self._grasp_contact_verified = False

    def _debug_log(self, message: str):
        if self.debug:
            # Isaac's default logging level hides INFO.  A debug request is
            # explicit, so emit at WARNING to keep the evidence in case logs.
            LOGGER.warning("[PickDebug] %s", message)

    def _target_constraints(self):
        keys = (
            "constraints",
            "filter_x_dir",
            "filter_y_dir",
            "filter_z_dir",
            "fixed_orientation",
            "pre_grasp_offset",
            "test_mode",
        )
        return {key: self.skill_cfg[key] for key in keys if key in self.skill_cfg}

    def _get_armbase_transform_in_task(self):
        armbase_tf_getter = getattr(self.robot, "get_armbase_world_transform", None)
        if callable(armbase_tf_getter):
            return armbase_tf_getter()

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
        return json_ready(value)

    def _write_debug_artifact(self, filename: str, payload: dict):
        output_path = os.path.join(self.debug_dir, filename)
        try:
            encoded = json.dumps(
                self._json_ready(payload), indent=2, ensure_ascii=False
            )
            os.makedirs(self.debug_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(encoded)
        except Exception as exc:
            LOGGER.warning(
                "[PickDebug] failed to write non-critical artifact %s: %s",
                output_path,
                exc,
            )
            return None
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
        if isinstance(manip_cmd, MotionPhaseCommand):
            return {
                "phase": manip_cmd.phase.value,
                "ee_translation": manip_cmd.target_position,
                "ee_orientation": manip_cmd.target_orientation,
                "gripper_action": manip_cmd.gripper_action,
                "active_object": manip_cmd.active_object,
                "params": manip_cmd.params,
            }
        ee_trans, ee_ori, cmd_name, params = manip_cmd
        return {
            "command": cmd_name,
            "ee_translation": ee_trans,
            "ee_orientation": ee_ori,
            "params": params,
        }

    def _gripper_action_for_state(self, gripper_cmd: str) -> np.ndarray:
        sign = -1.0 if gripper_cmd == "close_gripper" else 1.0
        old_state = getattr(self.controller, "_gripper_state", 1.0)
        self.controller._gripper_state = sign
        try:
            return np.asarray(self.controller.get_gripper_action(), dtype=float)
        finally:
            self.controller._gripper_state = old_state

    def _append_gripper_transition(self, manip_list, ee_trans, ee_ori, gripper_cmd: str):
        steps = max(int(self.skill_cfg.get("gripper_change_steps", 40)), 1)
        start_cmd = "open_gripper" if gripper_cmd == "close_gripper" else "close_gripper"
        start = self._gripper_action_for_state(start_cmd)
        target = self._gripper_action_for_state(gripper_cmd)
        for step in range(steps):
            ratio = float(step + 1) / float(steps)
            gripper_action = start + (target - start) * ratio
            manip_list.append(
                (
                    ee_trans,
                    ee_ori,
                    gripper_cmd,
                    {"skip_plan": True, "gripper_action": gripper_action},
                )
            )

    def _append_gripper_hold(self, manip_list, ee_trans, ee_ori, gripper_cmd: str, steps=None):
        if steps is None:
            steps = self.skill_cfg.get("grasp_open_hold_steps", 8)
        steps = max(int(steps), 0)
        gripper_action = self._gripper_action_for_state(gripper_cmd)
        for _ in range(steps):
            manip_list.append(
                (
                    ee_trans,
                    ee_ori,
                    gripper_cmd,
                    {"skip_plan": True, "gripper_action": gripper_action},
                )
            )

    def _grasp_arrival_params(self):
        t_eps = min(float(self.skill_cfg.get("t_eps", 1e-3)), float(self.skill_cfg.get("grasp_t_eps", 0.008)))
        o_eps = min(float(self.skill_cfg.get("o_eps", 5e-3)), float(self.skill_cfg.get("grasp_o_eps", 0.2)))
        return {"t_eps": t_eps, "o_eps": o_eps}

    def _offset_from_grasp(self, grasp_translation, T_base_ee_grasp, offset):
        if offset <= 0.0:
            return grasp_translation
        if "r5a" in self.controller.robot_file:
            approach_axis = T_base_ee_grasp[:3, 0]
        else:
            approach_axis = T_base_ee_grasp[:3, 2]
        return grasp_translation - approach_axis * offset

    def _get_object_pose_in_armbase(self):
        T_world_obj = tf_matrix_from_pose(*self._get_object_world_pose())
        T_world_base = self._get_armbase_transform_in_task()
        T_base_obj = np.linalg.inv(T_world_base) @ T_world_obj
        return pose_from_tf_matrix(T_base_obj)

    def _get_grasp_side_projection(self, grasp_translation, obj_base_t):
        obj_xy = np.asarray(obj_base_t[:2], dtype=float)
        obj_norm = float(np.linalg.norm(obj_xy))
        if obj_norm <= 1e-6:
            return 0.0
        rel_xy = np.asarray(grasp_translation[:2], dtype=float) - obj_xy
        return float(np.dot(rel_xy, obj_xy / obj_norm))

    def _select_grasp_index(self, pre_result, result, p_base_ee_grasps, q_base_ee_grasps, T_base_ee_grasps):
        priority_index = select_index_by_priority_dual(pre_result, result)
        pre_success_mask = np.asarray(pre_result.success.detach().cpu().numpy()).reshape(-1).astype(bool)
        grasp_success_mask = np.asarray(result.success.detach().cpu().numpy()).reshape(-1).astype(bool)
        both_success = np.logical_and(pre_success_mask, grasp_success_mask)
        candidate_indices = np.where(both_success)[0]
        if len(candidate_indices) == 0:
            return priority_index

        obj_base_t, _ = self._get_object_pose_in_armbase()
        target_grasp_z = float(self.skill_cfg.get("target_grasp_z", 0.12))
        target_grasp_orientation = self.skill_cfg.get("target_grasp_orientation", None)
        if target_grasp_orientation is None and self.pick_obj.name.startswith("apple_"):
            target_grasp_orientation = [
                0.150190916268329,
                0.7899356519730897,
                -0.5611163759233482,
                0.19645041889234482,
            ]
        if target_grasp_orientation is not None:
            target_grasp_orientation = np.asarray(target_grasp_orientation, dtype=float)
            target_grasp_orientation = target_grasp_orientation / np.linalg.norm(target_grasp_orientation)
        target_grasp_orientation_weight = float(self.skill_cfg.get("target_grasp_orientation_weight", 1.0))
        side_preference = self.skill_cfg.get("grasp_side_preference", None)
        grasp_side_weight = float(self.skill_cfg.get("grasp_side_weight", 2.0))
        priority_rank = {int(idx): rank for rank, idx in enumerate(candidate_indices.tolist())}
        scored = []
        for idx in candidate_indices:
            rel = np.asarray(p_base_ee_grasps[idx], dtype=float) - np.asarray(obj_base_t, dtype=float)
            xy_norm = float(np.linalg.norm(rel[:2]))
            approach_axis = T_base_ee_grasps[idx, :3, 0 if "r5a" in self.controller.robot_file else 2]
            vertical_penalty = max(0.0, float(approach_axis[2]) + 0.75)
            height_penalty = abs(float(rel[2]) - target_grasp_z)
            orientation_penalty = 0.0
            if target_grasp_orientation is not None:
                q = np.asarray(q_base_ee_grasps[idx], dtype=float)
                q = q / np.linalg.norm(q)
                orientation_penalty = 1.0 - abs(float(np.dot(q, target_grasp_orientation)))
            side_projection = None
            side_penalty = 0.0
            if side_preference is not None:
                side_projection = self._get_grasp_side_projection(p_base_ee_grasps[idx], obj_base_t)
                if side_preference == "toward_arm":
                    preferred_side_projection = side_projection
                elif side_preference == "away_from_arm":
                    preferred_side_projection = -side_projection
                else:
                    raise NotImplementedError
                wrong_side_penalty = max(0.0, preferred_side_projection)
                preferred_side_bonus = min(0.0, preferred_side_projection)
                side_penalty = wrong_side_penalty * grasp_side_weight + preferred_side_bonus * 0.25
            score = (
                height_penalty
                + xy_norm
                + vertical_penalty
                + orientation_penalty * target_grasp_orientation_weight
                + side_penalty
            )
            scored.append(
                (
                    score,
                    priority_rank.get(int(idx), 0),
                    int(idx),
                    rel.tolist(),
                    xy_norm,
                    float(rel[2]),
                    orientation_penalty,
                    side_projection,
                    side_penalty,
                )
            )

        scored.sort()
        selected = scored[0]
        self._candidate_rank_debug = [
            {
                "score": score,
                "priority_rank": rank,
                "candidate_index": idx,
                "relative_grasp_translation": rel,
                "xy_norm": xy_norm,
                "relative_grasp_z": rel_z,
                "orientation_penalty": orientation_penalty,
                "side_projection": side_projection,
                "side_preference": side_preference,
                "side_penalty": side_penalty,
            }
            for score, rank, idx, rel, xy_norm, rel_z, orientation_penalty, side_projection, side_penalty in scored[
                : min(len(scored), 16)
            ]
        ]
        print(
            "[pick-debug] Selected grasp candidate "
            f"{selected[2]} after physical ranking; "
            f"priority_candidate={priority_index}, score={selected[0]:.6f}"
        )
        return selected[2]

    def _validate_complete_candidate_path(
        self,
        pregrasp_translation,
        pregrasp_orientation,
        grasp_translation,
        grasp_orientation,
        postgrasp_translation,
        base_ignore_substring,
        grasp_ignore_substring,
        validate_postgrasp,
    ):
        """Validate current -> pregrasp -> grasp -> postgrasp as one continuous path."""
        self.controller.update_specific(
            ignore_substring=base_ignore_substring,
            reference_prim_path=self.controller.reference_prim_path,
        )
        pregrasp_success, pregrasp_end_js, _ = self.controller.test_forward_from_joint_positions(
            pregrasp_translation,
            pregrasp_orientation,
        )
        grasp_success = False
        grasp_end_js = None
        postgrasp_success = False
        if pregrasp_success:
            self.controller.update_specific(
                ignore_substring=grasp_ignore_substring,
                reference_prim_path=self.controller.reference_prim_path,
            )
            grasp_success, grasp_end_js, _ = self.controller.test_forward_from_joint_positions(
                grasp_translation,
                grasp_orientation,
                start_arm_positions=pregrasp_end_js,
            )
        if grasp_success:
            if validate_postgrasp:
                # The grasp pass intentionally ignores the target so fingers
                # can close around it.  Restore it before validating lift,
                # then attach it at the grasp endpoint so this candidate sees
                # the same collision geometry as runtime execution.
                self.controller.update_specific(
                    ignore_substring=base_ignore_substring,
                    reference_prim_path=self.controller.reference_prim_path,
                )
                postgrasp_success, _, _ = self.controller.test_attached_forward_from_joint_positions(
                    postgrasp_translation,
                    grasp_orientation,
                    start_arm_positions=grasp_end_js,
                    obj_prim_paths=list(self.pick_obj.attach_collision_prim_paths),
                )
            else:
                postgrasp_success = True

        return {
            "pregrasp_success": bool(pregrasp_success),
            "grasp_success": bool(grasp_success),
            "postgrasp_success": bool(postgrasp_success),
        }

    def _find_complete_candidate_path(
        self,
        candidate_order,
        p_base_ee_pregrasps,
        q_base_ee_pregrasps,
        p_base_ee_grasps,
        q_base_ee_grasps,
        p_base_ee_postgrasps,
        base_ignore_substring,
        grasp_ignore_substring,
        validate_postgrasp,
        candidate_debug_by_index,
    ):
        for candidate_index in candidate_order:
            validation = self._validate_complete_candidate_path(
                p_base_ee_pregrasps[candidate_index],
                q_base_ee_pregrasps[candidate_index],
                p_base_ee_grasps[candidate_index],
                q_base_ee_grasps[candidate_index],
                p_base_ee_postgrasps[candidate_index],
                base_ignore_substring,
                grasp_ignore_substring,
                validate_postgrasp=validate_postgrasp,
            )
            candidate_debug_by_index[candidate_index].update(validation)
            if all(validation.values()):
                print(f"[pick-debug] Complete pick path succeeded for candidate {candidate_index}.")
                return candidate_index
        return None

    def simple_generate_manip_cmds(self):
        if getattr(self.controller, "collision_world_mode", "legacy_stage_scan") == "physics_schema":
            return self._physics_schema_generate_manip_cmds()
        return self._legacy_simple_generate_manip_cmds()

    @staticmethod
    def _terminal_samples(start, goal, step_m: float) -> list[np.ndarray]:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        distance = float(np.linalg.norm(goal - start))
        count = max(1, int(np.ceil(distance / float(step_m))))
        return [start + (goal - start) * (index / count) for index in range(1, count + 1)]

    def _physics_schema_generate_manip_cmds(self):
        """Generate stateful Pick phases against the exact Physics world."""

        self.failure_reason = ""
        self._grasp_contact_verified = False
        manager = self.controller.collision_scene_manager
        object_name = self.pick_obj.name
        robot, arm = self.controller.name, self.controller.lr_name
        pick_place_cfg = self.task.cfg.get("planning", {}).get("pick_place", {})
        terminal_step = float(pick_place_cfg.get("terminal_step_m", 0.005))
        max_terminal = float(pick_place_cfg.get("max_terminal_distance_m", 0.10))
        self.controller.update_pose_cost_metric(None)
        manager.sync_dynamic_poses(0, interval_steps=1, force=True)
        manager.begin_target_transit(object_name, robot, arm)
        transforms = self.sample_ee_pose()
        if os.environ.get("SIMBOX_DRAW_GRASP_AXES") == "1":
            try:
                from core.utils.debug_marker import draw_grasp_debug

                p_ee, q_ee = self.controller.get_ee_pose()
                marker_root, n = draw_grasp_debug(
                    self.controller,
                    self.task.root_prim_path,
                    p_ee,
                    q_ee,
                    transforms,
                    max_frames=int(os.environ.get("SIMBOX_DRAW_GRASP_FRAMES", "8")),
                )
                LOGGER.warning("[PickDebug] drew EE + %d grasp-axes at %s", n, marker_root)
            except Exception as exc:  # diagnostics must never break planning
                LOGGER.warning("[PickDebug] grasp-axes debug failed: %r", exc)

        evaluator = GraspPlanEvaluator(self.controller, self._debug_log)
        missing = evaluator.missing_attach_prims(self.pick_obj.attach_collision_prim_paths)
        test_mode = resolve_skill_test_mode(
            self.skill_cfg, getattr(self.controller, "collision_world_mode", "legacy_stage_scan")
        )
        self.plan_evaluation = evaluator.evaluate(
            transforms,
            self.sampled_scores,
            pregrasp_offset_m=float(self.skill_cfg.get("pre_grasp_offset", 0.1)),
            attach_prim_paths=self.pick_obj.attach_collision_prim_paths,
            fixed_orientation=self.fixed_orientation,
            test_mode=test_mode,
            attach_config_failure_code=self.pick_obj.attach_collision_failure_code,
            attach_candidate_paths=self.pick_obj.attach_collision_candidates,
            attach_missing_paths=missing,
            prepare_pregrasp_world=lambda: manager.begin_target_transit(object_name, robot, arm),
            prepare_grasp_world=lambda: manager.begin_target_approach(object_name, robot, arm),
            cartesian_ratio_limit=float(self.skill_cfg.get("cartesian_ratio_limit", 1.5)),
            cartesian_deviation_m=float(self.skill_cfg.get("cartesian_deviation_m", 0.01)),
        )
        result = self.plan_evaluation.result
        # Candidate testing leaves the owner in the terminal world.  Execution
        # always starts again from the complete transit world.
        manager.restore_world(object_name)
        world_collision_diagnostic = None
        if not result.feasible and result.pregrasp_success_count == 0:
            try:
                world_collision_diagnostic = manager.diagnose_controller_world_collision(
                    self.controller
                )
                LOGGER.warning(
                    "[PickSafety] start-state world collision diagnostic object=%s result=%s",
                    object_name,
                    world_collision_diagnostic,
                )
            except Exception as exc:
                world_collision_diagnostic = {
                    "available": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                LOGGER.exception(
                    "[PickSafety] failed to diagnose start-state world collision for %s",
                    object_name,
                )
        self._plan_debug_path = self._write_debug_artifact(
            "pick_plan_snapshot.json",
            {
                "robot": self.robot.name,
                "object": object_name,
                "lr_arm": self.lr_arm,
                "collision_world_mode": "physics_schema",
                "plan_evaluation": result.to_dict(),
                "sample_debug": self._sample_debug,
                "geometry_debug": self._collect_geometry_debug(),
                "pregrasp_positions": self.plan_evaluation.pregrasp_positions,
                "pregrasp_orientations": self.plan_evaluation.pregrasp_orientations,
                "grasp_positions": self.plan_evaluation.grasp_positions,
                "grasp_orientations": self.plan_evaluation.grasp_orientations,
                "world_collision_diagnostic": world_collision_diagnostic,
            },
        )
        if not result.feasible:
            self.failure_reason = result.failure_code or "NO_COLLISION_FREE_PLAN"
            self.publish_target_intent(
                {
                    "kind": "pick",
                    "objects": [object_name],
                    "has_target": False,
                    "failure_reason": result.failure_code or "NO_COLLISION_FREE_PLAN",
                    "candidate_count": len(transforms),
                    "constraints": self._target_constraints(),
                }
            )
            self.manip_list = []
            return

        index = int(result.selected_grasp_index)
        pre_positions = self.plan_evaluation.pregrasp_positions
        pre_orientations = self.plan_evaluation.pregrasp_orientations
        positions = self.plan_evaluation.grasp_positions
        orientations = self.plan_evaluation.grasp_orientations
        terminal_distance = float(np.linalg.norm(positions[index] - pre_positions[index]))
        if terminal_distance > max_terminal + 1e-5:
            LOGGER.warning(
                "[PickSafety] terminal grasp distance %.4fm exceeds %.4fm for %s",
                terminal_distance,
                max_terminal,
                object_name,
            )
            self.manip_list = []
            self.failure_reason = "TERMINAL_DISTANCE_EXCEEDED"
            return

        self.publish_target_intent(
            {
                "kind": "pick",
                "objects": [object_name],
                "selected_index": index,
                "selected_score": result.selected_grasp_score,
                "constraints": self._target_constraints(),
                "pregrasp_position": pre_positions[index],
                "pregrasp_orientation": pre_orientations[index],
                "grasp_position": positions[index],
                "grasp_orientation": orientations[index],
            }
        )
        tolerance = {
            "position_m": float(self.skill_cfg.get("t_eps", 0.005)),
            "orientation_rad": float(self.skill_cfg.get("o_eps", 0.05)),
        }
        commands = [
            MotionPhaseCommand(
                MotionPhase.SYNC_WORLD,
                active_object=object_name,
                replan_allowed=False,
            ),
            MotionPhaseCommand(
                MotionPhase.TRANSIT_PREGRASP,
                pre_positions[index],
                pre_orientations[index],
                gripper_action="open_gripper",
                active_object=object_name,
                completion_tolerance=tolerance,
            ),
        ]
        if self.plan_evaluation.terminal_path is not None:
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.TERMINAL_GRASP_APPROACH,
                    positions[index],
                    orientations[index],
                    gripper_action="open_gripper",
                    active_object=object_name,
                    allow_target_finger_contact=True,
                    completion_tolerance={"position_m": terminal_step, "orientation_rad": tolerance["orientation_rad"]},
                    params={
                        "preplanned_joint_path": self.plan_evaluation.terminal_path,
                        "cartesian_step_m": terminal_step,
                        "path_length_ratio": self.plan_evaluation.terminal_path_length_ratio,
                        "path_max_deviation_m": self.plan_evaluation.terminal_path_max_deviation_m,
                    },
                )
            )
        else:
            # Compatibility with legacy/mock controllers that cannot return a
            # chained pre-grasp -> grasp path.  Runtime physics controllers do.
            terminal_points = self._terminal_samples(
                pre_positions[index], positions[index], terminal_step
            )
            for point_index, point in enumerate(terminal_points):
                ratio = (point_index + 1) / len(terminal_points)
                quat = (1.0 - ratio) * pre_orientations[index] + ratio * orientations[index]
                quat = quat / np.linalg.norm(quat)
                commands.append(
                    MotionPhaseCommand(
                        MotionPhase.TERMINAL_GRASP_APPROACH,
                        point,
                        quat,
                        gripper_action="open_gripper",
                        active_object=object_name,
                        allow_target_finger_contact=True,
                        completion_tolerance={"position_m": terminal_step, "orientation_rad": tolerance["orientation_rad"]},
                    )
                )
        commands.append(
            MotionPhaseCommand(
                MotionPhase.GRIPPER_CLOSE,
                positions[index],
                orientations[index],
                gripper_action=self.gripper_cmd,
                active_object=object_name,
                allow_target_finger_contact=True,
                replan_allowed=False,
                dwell_steps=int(self.skill_cfg.get("gripper_change_steps", 40)),
                params={
                    "contact_threshold_n": float(
                        self.skill_cfg.get("grasp_contact_threshold_n", 0.0)
                    )
                },
            )
        )
        commands.append(
            MotionPhaseCommand(
                MotionPhase.ATTACH,
                active_object=object_name,
                allow_target_finger_contact=True,
                replan_allowed=False,
                params={"verify_grasp_contact": lambda: self._grasp_contact_verified},
            )
        )
        post_offset = np.random.uniform(
            self.skill_cfg.get("post_grasp_offset_min", 0.05),
            self.skill_cfg.get("post_grasp_offset_max", 0.05),
        )
        if post_offset:
            post_position = np.asarray(positions[index], dtype=float).copy()
            post_position[2] += float(post_offset)
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.POST_GRASP_LIFT,
                    post_position,
                    orientations[index],
                    gripper_action=self.gripper_cmd,
                    active_object=object_name,
                    allow_target_finger_contact=True,
                    completion_tolerance=tolerance,
                )
            )
        if self.skill_cfg.get("return_to_pregrasp", False):
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.POST_GRASP_LIFT,
                    pre_positions[index],
                    pre_orientations[index],
                    gripper_action=self.gripper_cmd,
                    active_object=object_name,
                    allow_target_finger_contact=True,
                    completion_tolerance=tolerance,
                )
            )
        self.manip_list = commands

    def _legacy_simple_generate_manip_cmds(self):
        """LEGACY_STAGE_SCAN: original tuple/substring Pick implementation."""

        # LEGACY_BEGIN: keyword-based collision world, retained for comparison
        manip_list = []
        self.process_valid = True
        self.failure_reason = ""
        self.error_message = ""
        self._runtime_failure_snapshot_written = False
        self._runtime_failure_debug_path = None
        self._selected_candidate_debug = {}
        self._candidate_rank_debug = []
        self._execution_trace = []
        self._execution_trace_path = None
        self._execution_trace_total_steps = 0
        self._last_stall_command_signature = None
        self._last_stall_command_started_at = 0
        self._stalled_failure_written = False
        object_name = self.skill_cfg["objects"][0]
        self._debug_log(
            "start object=%s arm=%s use_batch=%s test_mode=%s pre_grasp_offset=%s"
            % (
                object_name,
                getattr(self.controller, "lr_name", "unknown"),
                self.controller.use_batch,
                self.skill_cfg.get("test_mode", "forward"),
                self.skill_cfg.get("pre_grasp_offset", 0.1),
            )
        )

        # Update
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        open_gripper_action = self._gripper_action_for_state("open_gripper")
        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_pose_cost_metric",
            {"hold_vec_weight": None, "gripper_action": open_gripper_action},
        )
        manip_list.append(cmd)

        base_ignore_substring = deepcopy(self.controller.ignore_substring + self.skill_cfg.get("ignore_substring", []))
        grasp_ignore_substring = deepcopy(base_ignore_substring)
        grasp_ignore_substring.append(self.pick_obj.name)

        # Pre grasp
        T_base_ee_grasps = self.sample_ee_pose()  # (N, 4, 4)
        T_base_ee_pregrasps = deepcopy(T_base_ee_grasps)
        self.controller.update_specific(
            ignore_substring=base_ignore_substring, reference_prim_path=self.controller.reference_prim_path
        )

        if "r5a" in self.controller.robot_file:
            T_base_ee_pregrasps[:, :3, 3] -= T_base_ee_pregrasps[:, :3, 0] * self.skill_cfg.get("pre_grasp_offset", 0.1)
        else:
            T_base_ee_pregrasps[:, :3, 3] -= T_base_ee_pregrasps[:, :3, 2] * self.skill_cfg.get("pre_grasp_offset", 0.1)

        if T_base_ee_grasps.shape[0] == 0:
            self.manip_list = []
            self.process_valid = False
            self.failure_reason = "no_grasp_candidates_after_sampling"
            self.error_message = "Pick sampling produced no grasp candidates."
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
            self.publish_target_intent(
                {
                    "kind": "pick",
                    "objects": [object_name],
                    "has_target": False,
                    "failure_reason": self.failure_reason,
                    "candidate_count": 0,
                    "constraints": self._target_constraints(),
                }
            )
            print(f"[pick-debug] No grasp candidates after sampling. Snapshot: {self._plan_debug_path}")
            return

        p_base_ee_pregrasps, q_base_ee_pregrasps = poses_from_tf_matrices(T_base_ee_pregrasps)
        p_base_ee_grasps, q_base_ee_grasps = poses_from_tf_matrices(T_base_ee_grasps)
        if self.fixed_orientation is not None:
            q_base_ee_pregrasps[:] = self.fixed_orientation
            q_base_ee_grasps[:] = self.fixed_orientation

        post_grasp_offset = float(
            np.random.uniform(
                self.skill_cfg.get("post_grasp_offset_min", 0.05),
                self.skill_cfg.get("post_grasp_offset_max", 0.05),
            )
        )
        p_base_ee_postgrasps = deepcopy(p_base_ee_grasps)
        p_base_ee_postgrasps[:, 2] += post_grasp_offset
        candidate_results = []
        success_found = False
        index = min(T_base_ee_grasps.shape[0] - 1, 0)
        candidate_order = list(range(T_base_ee_grasps.shape[0]))

        if self.controller.use_batch:
            pre_result = self.controller.test_batch_forward(p_base_ee_pregrasps, q_base_ee_pregrasps)
            self.controller.update_specific(
                ignore_substring=grasp_ignore_substring, reference_prim_path=self.controller.reference_prim_path
            )
            result = self.controller.test_batch_forward(p_base_ee_grasps, q_base_ee_grasps)
            pre_success_mask = np.asarray(pre_result.success.detach().cpu().numpy()).reshape(-1).astype(bool)
            grasp_success_mask = np.asarray(result.success.detach().cpu().numpy()).reshape(-1).astype(bool)
            coarse_success_mask = np.logical_and(pre_success_mask, grasp_success_mask)
            candidate_order = np.where(coarse_success_mask)[0].tolist()
            if candidate_order:
                preferred_index = self._select_grasp_index(
                    pre_result,
                    result,
                    p_base_ee_grasps,
                    q_base_ee_grasps,
                    T_base_ee_grasps,
                )
                candidate_order.remove(preferred_index)
                candidate_order.insert(0, preferred_index)
            candidate_results = [
                {
                    **self._candidate_source_debug(i),
                    "batch_pregrasp_success": bool(pre_success_mask[i]),
                    "batch_grasp_success": bool(grasp_success_mask[i]),
                    "pregrasp_translation": p_base_ee_pregrasps[i],
                    "pregrasp_orientation": q_base_ee_pregrasps[i],
                    "grasp_translation": p_base_ee_grasps[i],
                    "grasp_orientation": q_base_ee_grasps[i],
                    "postgrasp_translation": p_base_ee_postgrasps[i],
                    "postgrasp_orientation": q_base_ee_grasps[i],
                }
                for i in range(len(pre_success_mask))
            ]
        else:
            candidate_results = [
                {
                    **self._candidate_source_debug(i),
                    "pregrasp_translation": p_base_ee_pregrasps[i],
                    "pregrasp_orientation": q_base_ee_pregrasps[i],
                    "grasp_translation": p_base_ee_grasps[i],
                    "grasp_orientation": q_base_ee_grasps[i],
                    "postgrasp_translation": p_base_ee_postgrasps[i],
                    "postgrasp_orientation": q_base_ee_grasps[i],
                }
                for i in candidate_order
            ]

        candidate_debug_by_index = {
            int(candidate_debug["candidate_index"]): candidate_debug for candidate_debug in candidate_results
        }
        complete_candidate_index = self._find_complete_candidate_path(
            candidate_order,
            p_base_ee_pregrasps,
            q_base_ee_pregrasps,
            p_base_ee_grasps,
            q_base_ee_grasps,
            p_base_ee_postgrasps,
            base_ignore_substring,
            grasp_ignore_substring,
            validate_postgrasp=bool(post_grasp_offset),
            candidate_debug_by_index=candidate_debug_by_index,
        )
        if complete_candidate_index is not None:
            index = complete_candidate_index
            success_found = True

        if not success_found:
            self._selected_candidate_debug = {
                "success_found": False,
                "attempted_candidate_indices": candidate_order,
                "post_grasp_offset": post_grasp_offset,
            }
            self.manip_list = []
            self.process_valid = False
            self.failure_reason = "no_complete_pick_path"
            self.error_message = (
                "No grasp candidate has a continuous current-to-pregrasp-to-grasp-to-postgrasp path."
            )
            self._plan_debug_path = self._write_debug_artifact(
                "pick_plan_snapshot.json",
                {
                    "robot": self.robot.name,
                    "object": self.pick_obj.name,
                    "lr_arm": self.lr_arm,
                    "reason": self.failure_reason,
                    "success_found": False,
                    "selected_candidate": self._selected_candidate_debug,
                    "sample_debug": self._sample_debug,
                    "geometry_debug": self._collect_geometry_debug(),
                    "candidate_results": candidate_results,
                    "candidate_rank_debug": self._candidate_rank_debug,
                    "manip_command_sequence": [],
                },
            )
            self.publish_target_intent(
                {
                    "kind": "pick",
                    "objects": [object_name],
                    "has_target": False,
                    "failure_reason": self.failure_reason,
                    "candidate_count": len(T_base_ee_grasps),
                    "constraints": self._target_constraints(),
                }
            )
            print(f"[pick-debug] No complete pick path found. Snapshot: {self._plan_debug_path}")
            return

        self._selected_candidate_debug = {
            **self._candidate_source_debug(index),
            "success_found": True,
            "selected_pregrasp_translation": p_base_ee_pregrasps[index],
            "selected_pregrasp_orientation": q_base_ee_pregrasps[index],
            "selected_grasp_translation": p_base_ee_grasps[index],
            "selected_grasp_orientation": q_base_ee_grasps[index],
            "selected_postgrasp_translation": p_base_ee_postgrasps[index],
            "selected_postgrasp_orientation": q_base_ee_grasps[index],
            "post_grasp_offset": post_grasp_offset,
        }
        self.publish_target_intent(
            {
                "kind": "pick",
                "objects": [object_name],
                "selected_index": index,
                "selected_score": self._candidate_source_debug(index).get("source_score"),
                "constraints": self._target_constraints(),
                "pregrasp_position": p_base_ee_pregrasps[index],
                "pregrasp_orientation": q_base_ee_pregrasps[index],
                "grasp_position": p_base_ee_grasps[index],
                "grasp_orientation": q_base_ee_grasps[index],
            }
        )

        # Pre-grasp
        cmd = (
            p_base_ee_cur,
            q_base_ee_cur,
            "update_specific",
            {
                "ignore_substring": base_ignore_substring,
                "reference_prim_path": self.controller.reference_prim_path,
                "gripper_action": open_gripper_action,
            },
        )
        manip_list.append(cmd)
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

        # Switch to grasp collision mode before guard so guard and grasp are planned
        # with pick_obj ignored, matching the candidate validation.
        cmd = (
            p_base_ee_pregrasps[index],
            q_base_ee_pregrasps[index],
            "update_specific",
            {
                "ignore_substring": grasp_ignore_substring,
                "reference_prim_path": self.controller.reference_prim_path,
                "gripper_action": open_gripper_action,
            },
        )
        manip_list.append(cmd)

        guard_offset = float(
            self.skill_cfg.get(
                "grasp_guard_offset",
                min(float(self.skill_cfg.get("pre_grasp_offset", 0.1)), 0.04),
            )
        )
        if guard_offset > 0.0:
            p_base_ee_grasp_guard = self._offset_from_grasp(
                p_base_ee_grasps[index],
                T_base_ee_grasps[index],
                guard_offset,
            )
            cmd = (
                p_base_ee_grasp_guard,
                q_base_ee_grasps[index],
                "open_gripper",
                {"gripper_action": open_gripper_action},
            )
            manip_list.append(cmd)

        # Grasp
        cmd = (p_base_ee_grasps[index], q_base_ee_grasps[index], "open_gripper", self._grasp_arrival_params())
        manip_list.append(cmd)
        self._append_gripper_hold(
            manip_list,
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "open_gripper",
        )
        self._append_gripper_transition(
            manip_list,
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            self.gripper_cmd,
        )
        self._append_gripper_hold(
            manip_list,
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            self.gripper_cmd,
            steps=self.skill_cfg.get("post_close_hold_steps", 12),
        )
        cmd = (
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "update_specific",
            {"ignore_substring": base_ignore_substring, "reference_prim_path": self.controller.reference_prim_path},
        )
        manip_list.append(cmd)
        cmd = (
            p_base_ee_grasps[index],
            q_base_ee_grasps[index],
            "attach_objects",
            {"obj_prim_paths": list(self.pick_obj.attach_collision_prim_paths), "skip_plan": True},
        )
        manip_list.append(cmd)

        # Post-grasp
        if post_grasp_offset:
            cmd = (
                p_base_ee_postgrasps[index],
                q_base_ee_grasps[index],
                self.gripper_cmd,
                {"gripper_action": self._gripper_action_for_state(self.gripper_cmd)},
            )
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
                "candidate_rank_debug": self._candidate_rank_debug,
                "manip_command_sequence": [self._manip_cmd_to_debug(cmd) for cmd in self.manip_list],
            },
        )
        print(f"[pick-debug] Wrote pick planning snapshot: {self._plan_debug_path}")

    def _record_execution_step(self):
        if not self.manip_list:
            return

        current_cmd = self.manip_list[0]
        action = getattr(self.controller, "_action", {}) or {}
        gripper_action = action.get("gripper_action", None)
        arm_action = action.get("arm_action", None)
        action_joint_positions = action.get("joint_positions", None)
        action_joint_indices = action.get("joint_indices", None)
        qpos = self.robot.get_joints_state().positions
        arm_indices = getattr(self.controller, "arm_indices", np.array([], dtype=int))
        gripper_indices = getattr(self.controller, "gripper_indices", np.array([], dtype=int))
        try:
            actual_arm_position = qpos[arm_indices]
        except Exception:
            actual_arm_position = []
        try:
            actual_gripper_position = qpos[gripper_indices]
        except Exception:
            actual_gripper_position = []

        obj_t, obj_q = self._get_object_world_pose()
        ee_t, ee_q = self.controller.get_ee_pose()
        if isinstance(current_cmd, MotionPhaseCommand):
            target_t = current_cmd.target_position if current_cmd.target_position is not None else ee_t
            target_q = current_cmd.target_orientation if current_cmd.target_orientation is not None else ee_q
            command_name = current_cmd.phase.value
        else:
            target_t = current_cmd[0]
            target_q = current_cmd[1]
            command_name = str(current_cmd[2])
        diff_trans = float(np.linalg.norm(np.asarray(ee_t) - np.asarray(target_t)))
        diff_ori = float(2 * np.arccos(min(abs(float(np.dot(ee_q, target_q))), 1.0)))
        self._execution_trace_total_steps += 1
        command_signature = (
            len(self.manip_list),
            self._json_ready(target_t),
            self._json_ready(target_q),
            command_name,
        )
        if command_signature != self._last_stall_command_signature:
            self._last_stall_command_signature = command_signature
            self._last_stall_command_started_at = self._execution_trace_total_steps
        command_age_steps = self._execution_trace_total_steps - self._last_stall_command_started_at
        self._execution_trace.append(
            {
                "step": self._execution_trace_total_steps - 1,
                "remaining_commands": len(self.manip_list),
                "current_command": self._manip_cmd_to_debug(current_cmd),
                "target_diff_trans": diff_trans,
                "target_diff_ori": diff_ori,
                "command_age_steps": int(command_age_steps),
                "controller_gripper_state": getattr(self.controller, "_gripper_state", None),
                "controller_cmd_plan_active": getattr(self.controller, "cmd_plan", None) is not None,
                "controller_cmd_idx": getattr(self.controller, "cmd_idx", None),
                "controller_num_last_cmd": getattr(self.controller, "num_last_cmd", None),
                "controller_num_plan_failed": getattr(self.controller, "num_plan_failed", None),
                "action_joint_indices": action_joint_indices,
                "action_joint_positions": action_joint_positions,
                "action_arm": arm_action,
                "action_gripper": gripper_action,
                "actual_arm_position": actual_arm_position,
                "actual_gripper_position": actual_gripper_position,
                "ee_translation": ee_t,
                "ee_orientation": ee_q,
                "object_world_translation": obj_t,
                "object_world_orientation": obj_q,
            }
        )
        max_trace_steps = max(int(self._execution_trace_max_steps), 1)
        if len(self._execution_trace) > max_trace_steps:
            self._execution_trace = self._execution_trace[-max_trace_steps:]

        write_stride = max(int(self._execution_trace_write_stride), 0)
        if write_stride > 0 and self._execution_trace_total_steps % write_stride == 0:
            self._execution_trace_path = self._write_debug_artifact(
                "pick_execution_trace.json",
                {
                    "steps": self._execution_trace,
                    "plan_snapshot_path": self._plan_debug_path,
                    "total_steps": int(self._execution_trace_total_steps),
                    "retained_steps": int(len(self._execution_trace)),
                    "max_steps": int(max_trace_steps),
                },
            )

        if (
            self._stalled_command_step_limit > 0
            and command_age_steps >= self._stalled_command_step_limit
            and diff_trans > float(self.skill_cfg.get("t_eps", 1e-3))
        ):
            self.process_valid = False
            self.failure_reason = "pick_command_stalled"
            self.error_message = (
                "Pick command did not converge before watchdog limit. "
                f"command_age_steps={command_age_steps}, "
                f"target_diff_trans={diff_trans:.6f}, target_diff_ori={diff_ori:.6f}, "
                f"remaining_commands={len(self.manip_list)}"
            )
            if not self._stalled_failure_written:
                self._runtime_failure_debug_path = self._write_debug_artifact(
                    "pick_runtime_failure_snapshot.json",
                    {
                        "robot": self.robot.name,
                        "object": self.pick_obj.name,
                        "lr_arm": self.lr_arm,
                        "reason": self.failure_reason,
                        "message": self.error_message,
                        "command_age_steps": int(command_age_steps),
                        "stalled_command_step_limit": int(self._stalled_command_step_limit),
                        "remaining_commands": int(len(self.manip_list)),
                        "target_diff_trans": diff_trans,
                        "target_diff_ori": diff_ori,
                        "num_last_cmd": int(getattr(self.controller, "num_last_cmd", 0)),
                        "current_command": self._manip_cmd_to_debug(current_cmd),
                        "recent_execution_trace": self._execution_trace[-25:],
                        "plan_snapshot_path": self._plan_debug_path,
                    },
                )
                self._runtime_failure_snapshot_written = True
                self._stalled_failure_written = True
                print(f"[pick-debug] Wrote pick stalled-command snapshot: {self._runtime_failure_debug_path}")

    def update(self):
        self._record_execution_step()

    def _flush_execution_trace(self):
        if self._execution_trace:
            self._execution_trace_path = self._write_debug_artifact(
                "pick_execution_trace.json",
                {
                    "steps": self._execution_trace,
                    "plan_snapshot_path": self._plan_debug_path,
                    "total_steps": int(self._execution_trace_total_steps),
                    "retained_steps": int(len(self._execution_trace)),
                    "max_steps": int(max(int(self._execution_trace_max_steps), 1)),
                },
            )
        # LEGACY_END

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
        filter_summaries = []
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
                filter_summaries.append(f"{axis}:{filter_list}->{int(flags[axis].sum())}/{num_pose}")
        grasp_side_projection = None
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
            filter_summaries.append(f"direction_to_obj:{direction_to_obj}->{int(flags['direction_to_obj'].sum())}/{num_pose}")

        if self.skill_cfg.get("grasp_side_preference", None) is not None:
            grasp_side_preference = self.skill_cfg["grasp_side_preference"]
            T_world_obj = tf_matrix_from_pose(*self._get_object_world_pose())
            T_world_base = self._get_armbase_transform_in_task()
            T_base_world = np.linalg.inv(T_world_base)
            T_base_obj = T_base_world @ T_world_obj
            object_xy = T_base_obj[:2, 3]
            object_norm = float(np.linalg.norm(object_xy))
            if object_norm <= 1e-6:
                raise ValueError(
                    "grasp_side_preference requires object to be offset from armbase in XY"
                )
            grasp_rel_xy = T_base_ee[:, :2, 3] - object_xy[None, :]
            grasp_side_projection = np.dot(grasp_rel_xy, object_xy / object_norm)
            if grasp_side_preference == "toward_arm":
                flags["direction_to_obj"] = np.logical_and(
                    flags["direction_to_obj"], grasp_side_projection <= 0.0
                )
            elif grasp_side_preference == "away_from_arm":
                flags["direction_to_obj"] = np.logical_and(
                    flags["direction_to_obj"], grasp_side_projection > 0.0
                )
            else:
                raise NotImplementedError

        combined_flag = np.logical_and.reduce(list(flags.values()))
        combined_count = int(combined_flag.sum())
        if sum(combined_flag) == 0:
            if self.skill_cfg.get("grasp_side_preference", None) is not None:
                idx_list = []
                sampled_scores = []
                LOGGER.warning(
                    "[PickDebug] grasp filters rejected all candidates for object=%s filters=%s; "
                    "grasp_side_preference forbids fallback",
                    self.pick_obj.name,
                    filter_summaries,
                )
            else:
                idx_list = list(range(min(max_length, num_pose)))
                sampled_scores = self.scores[idx_list]
                LOGGER.warning(
                    "[PickDebug] grasp filters rejected all candidates for object=%s filters=%s; "
                    "falling back to first %d candidates",
                    self.pick_obj.name,
                    filter_summaries,
                    max_length,
                )
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
            "grasp_side_preference": self.skill_cfg.get("grasp_side_preference", None),
            "grasp_side_projection": [float(v) for v in grasp_side_projection[idx_list]]
            if grasp_side_projection is not None and len(idx_list) > 0
            else [],
            "max_length": int(max_length),
        }
        self.sampled_scores = np.asarray(sampled_scores, dtype=float)
        print(self.scores[idx_list])
        # print((T_base_ee[idx_list])[:, 0, 1])
        if idx_list:
            selected_trans = T_base_ee[idx_list, :3, 3]
            self._debug_log(
                "filter object=%s total=%d combined=%d selected=%d filters=%s score_range=(%.4f, %.4f) "
                "selected_xyz_min=%s selected_xyz_max=%s"
                % (
                    self.pick_obj.name,
                    num_pose,
                    combined_count,
                    len(idx_list),
                    filter_summaries,
                    float(np.min(self.sampled_scores)),
                    float(np.max(self.sampled_scores)),
                    np.array2string(np.min(selected_trans, axis=0), precision=4, suppress_small=True),
                    np.array2string(np.max(selected_trans, axis=0), precision=4, suppress_small=True),
                )
            )
        else:
            self._debug_log(
                "filter object=%s total=%d combined=%d selected=0 filters=%s"
                % (self.pick_obj.name, num_pose, combined_count, filter_summaries)
            )
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
        values = np.asarray(
            self.pickcontact_view.get_contact_force_matrix(), dtype=float
        )
        if not values.size:
            return np.empty((0,), dtype=float), np.empty((0,), dtype=int)
        contact = np.atleast_1d(np.sum(np.abs(values), axis=-1).squeeze())
        indices = np.where(contact > contact_threshold)[0]
        return contact, indices
    def _debug_contact_force(self, threshold: float = 0.0) -> None:
        """Log the measured finger-to-target contact force at grasp time."""
        if os.environ.get("SIMBOX_DEBUG_CONTACT") != "1":
            return

        values = np.asarray(
            self.pickcontact_view.get_contact_force_matrix(), dtype=float
        )
        if not values.size:
            LOGGER.warning(
                "[ContactDebug] object=%s raw_shape=%s no finger-object contact",
                self.pick_obj.name,
                values.shape,
            )
            return

        # RigidContactView stores a 3-vector per filter (the last dimension).
        # Keep the same L1 reduction used by get_contact(), and also report
        # the Euclidean norm for a physically meaningful Newton value.
        l1_force = np.sum(np.abs(values), axis=-1).squeeze()
        norm_force = np.linalg.norm(values, axis=-1).squeeze()
        l1_force = np.atleast_1d(l1_force)
        norm_force = np.atleast_1d(norm_force)
        contacted = np.where(l1_force > threshold)[0].tolist()

        LOGGER.warning(
            "[ContactDebug] object=%s raw_shape=%s "
            "finger_force_n=%s max=%.6fN contacted=%s threshold=%.6fN",
            self.pick_obj.name,
            values.shape,
            np.array2string(norm_force, precision=6, suppress_small=False),
            float(np.max(norm_force)) if norm_force.size else 0.0,
            contacted,
            threshold,
        )

    def is_feasible(self, th=5):
        feasible = self.controller.num_plan_failed <= th and bool(self.process_valid)
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
                    "failure_reason": getattr(self, "failure_reason", ""),
                    "error_message": getattr(self, "error_message", ""),
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
        if isinstance(self.manip_list[0], MotionPhaseCommand):
            command = self.manip_list[0]
            done = self.controller.is_phase_command_complete(command)
            if done and command.phase == MotionPhase.GRIPPER_CLOSE:
                threshold = float(command.params.get("contact_threshold_n", 0.0))
                self._debug_contact_force(threshold=threshold)
                _, indices = self.get_contact(contact_threshold=threshold)
                self._grasp_contact_verified = len(indices) >= 1
                command.params["contact_verified"] = self._grasp_contact_verified
                if not self._grasp_contact_verified:
                    self.failure_reason = "GRASP_CONTACT_MISSING"
                    # Do not permit the next ATTACH phase. Restore the target
                    # to the complete world before ending this failed Pick.
                    self.controller.collision_scene_manager.restore_world(
                        self.pick_obj.name
                    )
                    self.manip_list[:] = [command]
            return done
        return self._legacy_is_subtask_done(t_eps=t_eps, o_eps=o_eps)

    def _legacy_is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        """LEGACY completion fallback retained only for tuple commands."""

        # LEGACY_BEGIN: pose OR wait-count completion, retained for comparison
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        p_base_ee, q_base_ee, gripper_fn, params = self.manip_list[0]
        diff_trans = np.linalg.norm(p_base_ee_cur - p_base_ee)
        diff_ori = 2 * np.arccos(min(abs(np.dot(q_base_ee_cur, q_base_ee)), 1.0))
        pose_flag = np.logical_and(
            diff_trans < t_eps,
            diff_ori < o_eps,
        )
        if bool(params.get("skip_plan", False)) or gripper_fn in {"update_pose_cost_metric", "update_specific"}:
            self.plan_flag = True
            return True
        self.plan_flag = False
        return pose_flag
        # LEGACY_END

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        current_cmd = self.manip_list[0]
        params = current_cmd.params if isinstance(current_cmd, MotionPhaseCommand) else current_cmd[3]
        if self.is_subtask_done(
            t_eps=params.get("t_eps", self.skill_cfg.get("t_eps", 1e-3)),
            o_eps=params.get("o_eps", self.skill_cfg.get("o_eps", 5e-3)),
        ):
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        self._flush_execution_trace()
        contact, indices = self.get_contact()
        contact_count = int(len(indices))
        contact_required = self.gripper_cmd == "close_gripper"
        contact_valid = (contact_count >= 1) if contact_required else True
        flag = contact_valid

        joint_velocity_max = float(np.max(np.abs(self.robot.get_joints_state().velocities)))
        object_linear_velocity_max = float(np.max(np.abs(self.pick_obj.get_linear_velocity())))

        if self.skill_cfg.get("process_valid", True):
            self.process_valid = joint_velocity_max < 5 and object_linear_velocity_max < 5
        flag = flag and self.process_valid

        lift_threshold = float(self.skill_cfg.get("lift_th", 0.0))
        object_position = deepcopy(self.pick_obj.get_local_pose()[0])
        lift_delta = float(object_position[2] - self.obj_init_trans[2])
        lift_valid = True
        if self.skill_cfg.get("lift_th", 0.0) > 0.0:
            lift_valid = lift_delta > lift_threshold
            flag = flag and lift_valid

        failure_reasons = []
        if not contact_valid:
            failure_reasons.append("no_gripper_object_contact")
        if not self.process_valid:
            failure_reasons.append("process_velocity_invalid")
        if not lift_valid:
            failure_reasons.append("lift_below_threshold")
        if (
            getattr(self.controller, "collision_world_mode", "legacy_stage_scan") == "physics_schema"
            and self.failure_reason
        ):
            flag = False
            failure_reasons.append(self.failure_reason)

        self._last_success_check_debug = {
            "robot": self.robot.name,
            "object": self.pick_obj.name,
            "lr_arm": self.lr_arm,
            "success": bool(flag),
            "failure_reasons": failure_reasons,
            "contact": {
                "required": bool(contact_required),
                "valid": bool(contact_valid),
                "count": contact_count,
                "indices": [int(idx) for idx in indices.tolist()],
                "threshold": 0.0,
                "max_force_sum": float(np.max(contact)) if np.size(contact) else 0.0,
                "force_sums": contact,
            },
            "process_valid": {
                "enabled": bool(self.skill_cfg.get("process_valid", True)),
                "valid": bool(self.process_valid),
                "joint_velocity_max": joint_velocity_max,
                "object_linear_velocity_max": object_linear_velocity_max,
                "threshold": 5.0,
            },
            "lift": {
                "enabled": bool(lift_threshold > 0.0),
                "valid": bool(lift_valid),
                "delta": lift_delta,
                "threshold": lift_threshold,
                "initial_position": self.obj_init_trans,
                "current_position": object_position,
            },
            "selected_candidate": self._selected_candidate_debug,
            "plan_snapshot_path": self._plan_debug_path,
        }

        if not flag:
            self.failure_reason = "pick_success_check_failed:" + ",".join(failure_reasons)
            self.error_message = (
                "Pick completed but success check failed. "
                f"contact_count={contact_count}, lift_delta={lift_delta:.6f}, "
                f"joint_velocity_max={joint_velocity_max:.6f}, "
                f"object_linear_velocity_max={object_linear_velocity_max:.6f}"
            )
            if not self._success_check_snapshot_written:
                self._success_check_debug_path = self._write_debug_artifact(
                    "pick_success_check_snapshot.json",
                    self._last_success_check_debug,
                )
                self._success_check_snapshot_written = True
                print(f"[pick-debug] Wrote pick success check snapshot: {self._success_check_debug_path}")
        self._debug_log(
            "success-check object=%s result=%s contact_count=%d contact_max=%.6f "
            "process_valid=%s max_joint_velocity=%.6f max_object_velocity=%.6f object_delta_z=%.6f"
            % (
                self.pick_obj.name,
                bool(flag),
                contact_count,
                float(np.max(contact)) if np.size(contact) else 0.0,
                bool(self.process_valid),
                joint_velocity_max,
                object_linear_velocity_max,
                lift_delta,
            )
        )

        return flag
