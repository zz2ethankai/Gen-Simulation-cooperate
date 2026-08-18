import json
import logging
import os
import random
import time
from copy import deepcopy

import numpy as np
from core.planning.grasp_plan_evaluator import GraspPlanEvaluator
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.json_utils import json_ready
from core.utils.plan_utils import select_index_by_priority_dual
from core.utils.transformation_utils import poses_from_tf_matrices
from core.utils.asset_path_utils import resolve_asset_path
from omegaconf import DictConfig
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import (
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
from isaacsim.core.utils.xforms import get_world_pose

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
        # Get grasp annotation
        object_cfg = next(obj for obj in task.cfg["objects"] if obj["name"] == object_name)
        usd_path = resolve_asset_path(self.task.asset_root, object_cfg)
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
        self.debug = self.skill_cfg.get("debug", False) or os.environ.get("SIMBOX_DEBUG_PICK") == "1"
        self.plan_evaluation = None
        self.sampled_scores = np.empty((0,), dtype=float)
        self.failure_reason = ""
        self._grasp_contact_verified = False
        # Physics-schema execution can spend several planner calls between
        # sampling a grasp and reaching the object.  Keep the object/base pose
        # used for those targets so a moving active target can be retargeted
        # without rebuilding the complete grasp candidate set.

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
        )
        return {key: self.skill_cfg[key] for key in keys if key in self.skill_cfg}

    def _get_armbase_transform_in_task(self):
        return self.controller.get_pick_armbase_transform()

    def _get_object_world_pose(self):
        get_world_pose = getattr(self.pick_obj, "get_world_pose", None)
        if callable(get_world_pose):
            return get_world_pose()
        return self.pick_obj.get_local_pose()

    def _capture_pick_plan_reference(self):
        self.controller.capture_pick_plan_reference(self.pick_obj.name)

    def _retarget_pick_commands_to_current_object(self, commands):
        """Shift pending Pick targets by the active object's latest rigid motion.

        Pick targets are stored in the arm-base frame, while grasp
        annotations are object-relative.  When a dynamic tabletop target
        rolls or slides during a native-v2 planning/recovery call, preserving
        the old arm-base target makes recovery chase a stale pose.  Transform
        every pending target through the object's world-frame delta and drop
        any cached terminal joint path, whose start and goal were generated
        for the old pose.

        Returns the translation and rotation magnitude of the applied object
        delta.  The method is deliberately independent of the safety monitor
        so the initial post-planning retarget and a later safety recovery use
        exactly the same geometry.
        """

        return self.controller.retarget_pick_phase_commands(self.pick_obj.name, commands)

    def replan_after_safety(self, command):
        """Retarget the remaining Pick phases after an active-object change."""

        return self.controller.replan_pick_after_safety(
            self.pick_obj.name, command, self.manip_list
        )

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
        frame_debug = self.controller.get_pick_frame_debug()
        mobile_base_prim_path = str(frame_debug.get("mobile_base_prim_path") or "").strip()
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
            "cached_mobile_to_armbase_tf": frame_debug.get("cached_mobile_to_armbase_tf"),
            "configured_mobile_to_armbase_translation": frame_debug.get(
                "configured_mobile_to_armbase_translation"
            ),
            "configured_mobile_to_armbase_orientation": frame_debug.get(
                "configured_mobile_to_armbase_orientation"
            ),
            "controller_lr_name": getattr(self.controller, "lr_name", None),
            "controller_robot_file": self.controller.robot_file,
        }

    def _manip_cmd_to_debug(self, manip_cmd):
        return {
            "phase": manip_cmd.phase.value,
            "ee_translation": manip_cmd.target_position,
            "ee_orientation": manip_cmd.target_orientation,
            "gripper_action": manip_cmd.gripper_action,
            "active_object": manip_cmd.active_object,
            "params": manip_cmd.params,
        }

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

    def _select_grasp_index(
        self,
        pre_result,
        result,
        p_base_ee_grasps,
        q_base_ee_grasps,
        T_base_ee_grasps,
        candidate_indices=None,
    ):
        priority_index = select_index_by_priority_dual(pre_result, result)
        if candidate_indices is None:
            pre_success_mask = GraspPlanEvaluator._success_mask(pre_result)
            grasp_success_mask = GraspPlanEvaluator._success_mask(result)
            both_success = np.logical_and(pre_success_mask, grasp_success_mask)
            candidate_indices = np.where(both_success)[0]
        else:
            candidate_indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
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

    def simple_generate_manip_cmds(self):
        return self._physics_schema_generate_manip_cmds()

    def _validate_native_post_grasp_candidates(self, post_grasp_offset):
        """Record that post-grasp validation is deferred until the real attach.

        The native v2 planner must receive the object pose and joint state after
        the gripper has actually closed.  A synthetic attach here runs before
        that state transition, so it can reject every candidate because the
        object is still resting on its source support (or because the
        pre-attach pose differs by a small physics step).  The authoritative
        check is the ``POST_GRASP_LIFT`` query in ``forward_phase_command``;
        it runs immediately after ``CollisionSceneManager.attach_target`` and
        therefore uses the same native attachment geometry as execution.

        Keeping this method as an explicit bookkeeping hook preserves the
        diagnostic field without doing one expensive native planner query per
        candidate.  Pre-grasp and terminal-grasp candidates have already been
        checked by ``GraspPlanEvaluator``.
        """

        evaluation = self.plan_evaluation
        result = evaluation.result
        terminal_paths = list(evaluation.terminal_paths or [])
        if not result.feasible or not terminal_paths or post_grasp_offset <= 0.0:
            return
        selected_index = result.selected_grasp_index
        evaluation.post_grasp_validation = [
            {
                "mode": "deferred_runtime_attach",
                "candidate_index": (
                    None if selected_index is None else int(selected_index)
                ),
                "success": None,
                "reason": "native_v2_uses_post_attach_pose_and_joint_state",
            }
        ]
        LOGGER.info(
            "[PickSafety] post-grasp validation deferred until runtime attach "
            "object=%s candidate=%s offset=%.4f",
            self.pick_obj.name,
            selected_index,
            float(post_grasp_offset),
        )

    def _physics_schema_generate_manip_cmds(self):
        """Generate stateful Pick phases against the exact Physics world."""

        self.failure_reason = ""
        self._grasp_contact_verified = False
        object_name = self.pick_obj.name
        pick_place_cfg = self.task.cfg.get("planning", {}).get("pick_place", {})
        max_terminal = float(pick_place_cfg.get("max_terminal_distance_m", 0.10))
        self.controller.prepare_pick_planning_world(object_name)

        self._capture_pick_plan_reference()
        transforms = self.sample_ee_pose()
        evaluator = GraspPlanEvaluator(self.controller, self._debug_log)
        missing = evaluator.missing_attach_prims(self.pick_obj.attach_collision_prim_paths)
        self.plan_evaluation = evaluator.evaluate(
            transforms,
            self.sampled_scores,
            pregrasp_offset_m=float(self.skill_cfg.get("pre_grasp_offset", 0.1)),
            attach_prim_paths=self.pick_obj.attach_collision_prim_paths,
            fixed_orientation=self.fixed_orientation,
            test_mode="forward",
            attach_config_failure_code=self.pick_obj.attach_collision_failure_code,
            attach_candidate_paths=self.pick_obj.attach_collision_candidates,
            attach_missing_paths=missing,
            prepare_pregrasp_world=lambda: self.controller.prepare_pick_pregrasp_world(object_name),
            prepare_grasp_world=lambda: self.controller.prepare_pick_grasp_world(object_name),
            candidate_selector=(
                lambda pre_result, result, candidate_indices, positions, orientations, grasp_transforms: self._select_grasp_index(
                    pre_result,
                    result,
                    positions,
                    orientations,
                    grasp_transforms,
                    candidate_indices=candidate_indices,
                )
            ),
        )
        result = self.plan_evaluation.result
        # Candidate testing leaves the owner in the terminal world.  Execution
        # always starts again from the complete transit world.
        self.controller.restore_pick_world(object_name)
        post_grasp_offset = (
            float(
                np.random.uniform(
                    self.skill_cfg.get("post_grasp_offset_min", 0.05),
                    self.skill_cfg.get("post_grasp_offset_max", 0.05),
                )
            )
            if result.feasible
            else 0.0
        )
        self._validate_native_post_grasp_candidates(post_grasp_offset)
        world_collision_diagnostic = None
        if not result.feasible and result.pregrasp_success_count == 0:
            try:
                world_collision_diagnostic = self.controller.diagnose_pick_start_world_collision()
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
                "terminal_plan_diagnostics": self.plan_evaluation.terminal_plan_diagnostics,
                "post_grasp_validation": self.plan_evaluation.post_grasp_validation,
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
        self.manip_list = self.controller.build_pick_phase_commands(
            object_name=object_name,
            pregrasp_position=pre_positions[index],
            pregrasp_orientation=pre_orientations[index],
            grasp_position=positions[index],
            grasp_orientation=orientations[index],
            gripper_action=self.gripper_cmd,
            post_grasp_offset=post_grasp_offset,
            terminal_path=self.plan_evaluation.terminal_path,
            terminal_path_length_ratio=self.plan_evaluation.terminal_path_length_ratio,
            terminal_path_max_deviation_m=self.plan_evaluation.terminal_path_max_deviation_m,
            return_to_pregrasp=bool(self.skill_cfg.get("return_to_pregrasp", False)),
            completion_tolerance={
                "position_m": float(self.skill_cfg.get("t_eps", 0.005)),
                "orientation_rad": float(self.skill_cfg.get("o_eps", 0.05)),
            },
            gripper_change_steps=int(self.skill_cfg.get("gripper_change_steps", 40)),
            contact_threshold_n=float(
                self.skill_cfg.get("grasp_contact_threshold_n", 0.0)
            ),
            verify_grasp_contact=lambda: self._grasp_contact_verified,
        )
        # Native-v2 candidate evaluation can take long enough for a dynamic
        # tabletop object to finish settling or slide to a new contact pose.
        # Retarget the freshly generated command sequence once before the
        # first execution step; later safety recoveries use the same helper.
        try:
            self._retarget_pick_commands_to_current_object(self.manip_list)
        except Exception as exc:  # pragma: no cover - simulator-only guard
            LOGGER.exception(
                "[PickSafety] failed to retarget initial object pose=%s: %s",
                object_name,
                exc,
            )
            self.failure_reason = "INITIAL_OBJECT_RETARGET_FAILED"
            self.manip_list = []


    def _execution_command_state(self, command, fallback_translation, fallback_orientation):
        if not isinstance(command, MotionPhaseCommand):
            raise TypeError("Pick execution requires MotionPhaseCommand instances")
        target_translation = (
            command.target_position
            if command.target_position is not None
            else fallback_translation
        )
        target_orientation = (
            command.target_orientation
            if command.target_orientation is not None
            else fallback_orientation
        )
        return target_translation, target_orientation, command.phase.value

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
        target_t, target_q, command_name = self._execution_command_state(
            current_cmd, ee_t, ee_q
        )
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
        command = self.manip_list[0]
        done = self.controller.is_phase_command_complete(command)
        if done and command.phase == MotionPhase.GRIPPER_CLOSE:
            threshold = float(command.params.get("contact_threshold_n", 0.0))
            _, indices = self.get_contact(contact_threshold=threshold)
            self._grasp_contact_verified = len(indices) >= 1
            command.params["contact_verified"] = self._grasp_contact_verified
            if not self._grasp_contact_verified:
                self.failure_reason = "GRASP_CONTACT_MISSING"
                # Do not permit the next ATTACH phase. Restore the target
                # to the complete world before ending this failed Pick.
                self.controller.restore_pick_world(self.pick_obj.name)
                self.manip_list[:] = [command]
        return done

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        current_cmd = self.manip_list[0]
        params = current_cmd.params
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
        if self.failure_reason:
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
