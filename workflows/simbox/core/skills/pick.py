import json
import logging
import os
import random
import time
from copy import deepcopy

import numpy as np
from core.planning.domain_types import CollisionPolicy
from core.planning.grasp_plan_evaluator import GraspPlanEvaluator
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.json_utils import json_ready
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
    def __init__(
        self,
        robot: Robot,
        skill_runtime,
        task: BaseTask,
        cfg: DictConfig,
        *args,
        pick_planning=None,
        **kwargs,
    ):
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime, pick_planning=pick_planning)
        self.planning = self._require_pick_planning()
        self._require_skill_runtime()
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
        # Candidate ranking is established by the typed grasp evaluator.  A
        # terminal safety violation must retire only the active candidate so a
        # later candidate can be planned from the measured hold state.
        self._candidate_rank_order = []
        self._candidate_failed_indices = set()
        self._candidate_failure_diagnostics = []
        self._candidate_replan_diagnostics = []
        self._terminal_replan_count = 0
        self._candidate_replan_limit = 0
        self._candidate_replan_exhausted_recorded = False
        self._candidate_replan_exhausted = False
        self._candidate_replan_exhausted_reason = ""
        # Get grasp annotation
        object_cfg = next(obj for obj in task.cfg["objects"] if obj["name"] == object_name)
        usd_path = resolve_asset_path(self.task.asset_root, object_cfg)
        grasp_pose_path = usd_path.replace(
            "Aligned_obj.usd", self.skill_cfg.get("npy_name", "Aligned_grasp_sparse.npy")
        )
        sparse_grasp_poses = np.load(grasp_pose_path)
        lr_arm = self.planning.lr_name
        self.lr_arm = lr_arm
        self.T_obj_ee, self.scores = self.robot.pose_post_process_fn(
            sparse_grasp_poses,
            lr_arm=lr_arm,
            grasp_scale=self.skill_cfg.get("grasp_scale", 1),
            tcp_offset=self.skill_cfg.get("tcp_offset", self.robot.tcp_offset),
            constraints=self.skill_cfg.get("constraints", None),
        )
        self._raw_grasp_keys = self._build_raw_grasp_keys(self.T_obj_ee)
        self._candidate_raw_indices = np.empty((0,), dtype=int)

        # Keyposes should be generated after previous skill is done
        self.manip_list = []
        self.pickcontact_view = task.pickcontact_views[robot.name][lr_arm][object_name]
        self.process_valid = True
        # Success is defined by the object's motion in the task/world frame.
        # A RigidObject can be a child of a referenced USD asset with a
        # non-unit authored scale; comparing its local z coordinate then
        # mixes the parent scale into the lift measurement.
        self.obj_init_trans = deepcopy(self._get_object_world_pose()[0])
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
        return self.planning.arm_base_transform()

    def _get_object_world_pose(self):
        get_world_pose = getattr(self.pick_obj, "get_world_pose", None)
        if callable(get_world_pose):
            return get_world_pose()
        return self.pick_obj.get_local_pose()

    def _capture_pick_plan_reference(self):
        self.planning.capture_reference(self.pick_obj.name)

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

        return self.planning.retarget_commands(
            self.pick_obj.name, commands
        )

    @staticmethod
    def _build_raw_grasp_keys(transforms):
        """Build stable pose/orientation identities for raw grasp slots."""

        values = np.asarray(transforms, dtype=float)
        if values.ndim != 3 or values.shape[1:] != (4, 4):
            raise ValueError(
                "raw grasp transforms must have shape [candidate, 4, 4]"
            )
        return [
            tuple(np.round(transform.reshape(-1), decimals=8).tolist())
            for transform in values
        ]

    def _deduplicated_raw_grasp_indices(self):
        """Return one representative raw slot per physical grasp identity.

        Annotation files occasionally contain repeated slots for one grasp.
        Duplicate slots must not consume a recovery attempt or a batch-planner
        row; the lowest-score representative preserves physical ranking.
        """

        unique = []
        representatives = {}
        for raw_index, key in enumerate(self._raw_grasp_keys):
            representative = representatives.get(key)
            if representative is None:
                representatives[key] = int(raw_index)
                continue
            # Duplicate raw slots represent one physical candidate.  Keep
            # the lower annotation score as its representative so removing
            # the duplicate cannot demote the existing physical ranking.
            if float(self.scores[raw_index]) < float(self.scores[representative]):
                representatives[key] = int(raw_index)
        unique.extend(sorted(representatives.values()))
        return unique

    def _candidate_pose_key(self, evaluation, candidate_index):
        """Return a normalized current-evaluation pose identity."""

        index = int(candidate_index)
        position = np.asarray(evaluation.grasp_positions[index], dtype=float)
        orientation = np.asarray(evaluation.grasp_orientations[index], dtype=float)
        return (
            tuple(np.round(position.reshape(-1), decimals=8).tolist()),
            tuple(np.round(orientation.reshape(-1), decimals=8).tolist()),
        )

    def _current_candidate_poses(self, candidate_index):
        """Resolve one candidate's grasp/pre-grasp poses from the current object.

        ``GraspPlanEvaluation`` stores the pose used during candidate
        generation.  Safety recovery may observe a moved object, so replacement
        planning must derive fresh arm-base poses from the raw grasp transform
        and the current object/world revision instead of copying those stale
        arrays back onto commands.
        """

        index = int(candidate_index)
        if index < 0 or index >= len(self._candidate_raw_indices):
            raise IndexError(f"candidate index has no raw grasp identity: {index}")
        raw_index = int(self._candidate_raw_indices[index])
        current_transforms = self.get_ee_poses("armbase")
        if raw_index < 0 or raw_index >= len(current_transforms):
            raise IndexError(f"raw grasp index is out of range: {raw_index}")
        grasp_transform = np.asarray(current_transforms[raw_index], dtype=float).copy()
        grasp_position, grasp_orientation = pose_from_tf_matrix(grasp_transform)
        grasp_position = np.asarray(grasp_position, dtype=float).reshape(3)
        grasp_orientation = np.asarray(grasp_orientation, dtype=float).reshape(4)

        pregrasp_transform = grasp_transform.copy()
        pregrasp_offset = float(self.skill_cfg.get("pre_grasp_offset", 0.1))
        if "r5a" in str(self.planning.robot_file).lower():
            pregrasp_transform[:3, 3] -= (
                pregrasp_transform[:3, 0] * pregrasp_offset
            )
        else:
            pregrasp_transform[:3, 3] -= (
                pregrasp_transform[:3, 2] * pregrasp_offset
            )
        pregrasp_position, pregrasp_orientation = pose_from_tf_matrix(
            pregrasp_transform
        )
        pregrasp_position = np.asarray(pregrasp_position, dtype=float).reshape(3)
        pregrasp_orientation = np.asarray(pregrasp_orientation, dtype=float).reshape(4)
        if self.fixed_orientation is not None:
            fixed_orientation = np.asarray(self.fixed_orientation, dtype=float).reshape(4)
            grasp_orientation = fixed_orientation.copy()
            pregrasp_orientation = fixed_orientation.copy()
        return (
            raw_index,
            pregrasp_position,
            pregrasp_orientation,
            grasp_position,
            grasp_orientation,
        )

    def _sync_replacement_candidate_targets(
        self,
        command,
        *,
        old_candidate_index,
        candidate_index,
        raw_index,
        pregrasp_position,
        pregrasp_orientation,
        grasp_position,
        grasp_orientation,
        world_revision,
    ):
        """Retarget every pending Pick pose to one fresh candidate.

        ``PickPlanningPort.replan_after_safety`` first applies the current
        object/world delta to the queue.  This helper then applies the
        old-candidate -> replacement-candidate rigid transform, preserving
        post-grasp offsets while never writing the stale evaluation pose back
        into the command queue.
        """

        if old_candidate_index is not None and int(old_candidate_index) >= 0:
            _, _, _, old_position, old_orientation = self._current_candidate_poses(
                int(old_candidate_index)
            )
        else:
            old_position = command.target_position
            old_orientation = command.target_orientation
        if old_position is None or old_orientation is None:
            raise ValueError("terminal replacement requires an old candidate pose")
        old_transform = tf_matrix_from_pose(old_position, old_orientation)
        new_transform = tf_matrix_from_pose(grasp_position, grasp_orientation)
        candidate_delta = new_transform @ np.linalg.inv(old_transform)

        for pending in self.manip_list:
            if not isinstance(pending, MotionPhaseCommand):
                continue
            if pending is command:
                pending.target_position = np.asarray(grasp_position, dtype=float).reshape(3).copy()
                pending.target_orientation = np.asarray(grasp_orientation, dtype=float).reshape(4).copy()
            elif pending.phase == MotionPhase.TRANSIT_PREGRASP:
                pending.target_position = np.asarray(pregrasp_position, dtype=float).reshape(3).copy()
                pending.target_orientation = np.asarray(pregrasp_orientation, dtype=float).reshape(4).copy()
            elif pending.phase == MotionPhase.GRIPPER_CLOSE:
                pending.target_position = np.asarray(grasp_position, dtype=float).reshape(3).copy()
                pending.target_orientation = np.asarray(grasp_orientation, dtype=float).reshape(4).copy()
            elif pending.target_position is not None and pending.target_orientation is not None:
                pending_transform = tf_matrix_from_pose(
                    pending.target_position, pending.target_orientation
                )
                pending_position, pending_orientation = pose_from_tf_matrix(
                    candidate_delta @ pending_transform
                )
                pending.target_position = np.asarray(pending_position, dtype=float).reshape(3)
                pending.target_orientation = np.asarray(pending_orientation, dtype=float).reshape(4)

            pending.params["candidate_index"] = int(candidate_index)
            pending.params["candidate_raw_index"] = int(raw_index)
            pending.params["candidate_world_revision"] = int(world_revision)
            pending.params["candidate_replan_limit"] = int(
                self._candidate_replan_limit
            )
            pending.metadata = {
                **dict(pending.metadata or {}),
                "candidate_index": int(candidate_index),
                "candidate_raw_index": int(raw_index),
                "candidate_world_revision": int(world_revision),
                "candidate_replan_limit": int(self._candidate_replan_limit),
            }
            if pending.phase == MotionPhase.TERMINAL_GRASP_APPROACH:
                pending.candidate_replan_limit = int(self._candidate_replan_limit)
                pending.replan_policy = "terminal_candidate_fallback"
                pending.metadata["replan_policy"] = "terminal_candidate_fallback"
        return candidate_delta

    def _record_terminal_candidate_event(
        self,
        candidate_index,
        *,
        success,
        reason,
        replan_index,
        planner_result=None,
        attempt_index=None,
        raw_index=None,
        world_revision=None,
    ):
        """Record typed terminal-candidate recovery evidence.

        Safety recovery is intentionally observable at the Pick boundary.  A
        planner result is summarized through :class:`GraspPlanEvaluator`
        helpers; no native result fields are copied into the skill snapshot.
        """

        event = {
            "candidate_index": int(candidate_index),
            "success": bool(success),
            "reason": str(reason),
            "phase": MotionPhase.TERMINAL_GRASP_APPROACH.value,
            "collision_policy": CollisionPolicy.TARGET_APPROACH.value,
            "replan_index": int(replan_index),
        }
        if attempt_index is not None:
            event["attempt_index"] = int(attempt_index)
        if raw_index is not None:
            event["raw_grasp_index"] = int(raw_index)
        if world_revision is not None:
            event["world_revision"] = int(world_revision)
        if self._candidate_replan_limit:
            event["candidate_replan_limit"] = int(self._candidate_replan_limit)
        if planner_result is not None:
            try:
                event["planner"] = GraspPlanEvaluator._result_diagnostic(
                    planner_result, int(candidate_index)
                )
            except (TypeError, AttributeError, ValueError):
                # The public Pick port is typed, but diagnostics must never
                # turn a recovery failure into an episode traceback.
                event["planner"] = {"type": type(planner_result).__name__}

        self._candidate_replan_diagnostics.append(event)
        if not success:
            self._candidate_failure_diagnostics.append(event)
        evaluation = self.plan_evaluation
        if evaluation is not None:
            # Keep the existing persisted validation stream backward
            # compatible while the dedicated Pick-side streams distinguish a
            # runtime terminal failure from deferred post-attach validation.
            evaluation.post_grasp_validation.append(dict(event))
            try:
                evaluation.terminal_plan_diagnostics.append(dict(event))
            except AttributeError:
                pass
        self._selected_candidate_debug = dict(event)
        self._write_debug_artifact(
            "pick_candidate_replan.json",
            {
                "object": self.pick_obj.name,
                "terminal_replan_count": int(self._terminal_replan_count),
                "candidate_rank_order": list(self._candidate_rank_order),
                "failed_candidate_indices": sorted(
                    int(index) for index in self._candidate_failed_indices
                ),
                "candidate_replan_limit": int(self._candidate_replan_limit),
                "candidate_replan_attempted": int(self._terminal_replan_count),
                "candidate_replan_exhausted": bool(
                    getattr(self, "_candidate_replan_exhausted", False)
                ),
                "candidate_replan_exhausted_reason": str(
                    getattr(self, "_candidate_replan_exhausted_reason", "")
                ),
                "events": self._candidate_replan_diagnostics,
            },
        )
        return event

    def _terminal_candidate_order(self, current_index):
        """Return remaining candidates in the established physical ranking."""

        evaluation = self.plan_evaluation
        if evaluation is None:
            return []
        candidate_count = len(evaluation.grasp_positions)
        ranked = []
        seen_pose_keys = set()
        for index in self._candidate_rank_order:
            index = int(index)
            if not 0 <= index < candidate_count or index in ranked:
                continue
            pose_key = self._candidate_pose_key(evaluation, index)
            if pose_key in seen_pose_keys:
                continue
            seen_pose_keys.add(pose_key)
            if 0 <= index < candidate_count:
                ranked.append(index)
        selected_index = evaluation.result.selected_grasp_index
        if selected_index is not None:
            selected_index = int(selected_index)
            if (
                0 <= selected_index < candidate_count
                and selected_index not in ranked
                and self._candidate_pose_key(evaluation, selected_index)
                not in seen_pose_keys
            ):
                seen_pose_keys.add(self._candidate_pose_key(evaluation, selected_index))
                ranked.insert(0, selected_index)
        # A planner-selected candidate can be available even when a custom
        # selector was not called (for example, zero-offset or single-query
        # evaluation).  Append the remaining geometric candidates in stable
        # index order rather than inventing a new ranking.
        for index in range(candidate_count):
            if index in ranked:
                continue
            pose_key = self._candidate_pose_key(evaluation, index)
            if pose_key in seen_pose_keys:
                continue
            seen_pose_keys.add(pose_key)
            ranked.append(index)
        current_index = None if current_index is None else int(current_index)
        return [
            index
            for index in ranked
            if index != current_index and index not in self._candidate_failed_indices
        ]

    def _terminal_candidate_replan_budget(self):
        """Return the bounded safety budget for this Pick candidate set."""

        unique_candidate_count = len(self._terminal_candidate_order(None))
        # The initial candidate is already planned.  Each remaining unique
        # candidate gets one safety replan, with a bounded cap so malformed or
        # unusually large annotation batches cannot create unbounded recovery.
        return min(max(unique_candidate_count - 1, 0), 8)

    def _replan_terminal_candidate(self, command, *, reason="terminal_safety_failure"):
        """Retire a failed terminal candidate and plan the next one.

        The safety supervisor has already cleared the execution plan before
        entering this callback.  Every replacement query therefore starts
        from the measured hold state through the typed Pick planning port and
        keeps the strict target-approach policy.  The close/attach commands
        remain later in ``manip_list`` and cannot run until this command
        completes.
        """

        evaluation = self.plan_evaluation
        if evaluation is None or not evaluation.result.feasible:
            return self.planning.replan_after_safety(
                self.pick_obj.name, command, self.manip_list
            )

        selected_index = evaluation.result.selected_grasp_index
        command_index = command.params.get("candidate_index")
        current_index = command_index if command_index is not None else selected_index
        if current_index is None:
            current_index = -1
        current_index = int(current_index)
        self._terminal_replan_count += 1
        replan_index = self._terminal_replan_count
        if current_index >= 0:
            self._candidate_failed_indices.add(current_index)
            current_raw_index = (
                int(self._candidate_raw_indices[current_index])
                if current_index < len(self._candidate_raw_indices)
                else None
            )
            self._record_terminal_candidate_event(
                current_index,
                success=False,
                reason=reason,
                replan_index=replan_index,
                raw_index=current_raw_index,
                world_revision=getattr(self.planning, "world_revision", None),
            )

        candidate_order = self._terminal_candidate_order(current_index)
        for attempt_index, candidate_index in enumerate(candidate_order, start=1):
            raw_index = None
            world_revision = None
            try:
                # Retarget the complete pending Pick queue for every
                # replacement candidate.  The object/world revision may
                # advance between candidate attempts, so one retarget at the
                # start of the safety event is insufficient.
                retargeted = self.planning.replan_after_safety(
                    self.pick_obj.name, command, self.manip_list
                )
                if not retargeted:
                    self._candidate_failed_indices.add(candidate_index)
                    self._record_terminal_candidate_event(
                        candidate_index,
                        success=False,
                        reason="terminal_retarget_failed",
                        replan_index=replan_index,
                        attempt_index=attempt_index,
                    )
                    continue
                # This transition is deliberately repeated for each recovery
                # candidate.  It preserves the exact TARGET_APPROACH scene
                # contract and does not enable robot-link target contact.
                world_revision = self.planning.transition_target(
                    self.pick_obj.name,
                    collision_policy=CollisionPolicy.TARGET_APPROACH,
                )
                if world_revision is None:
                    world_revision = self.planning.world_revision
                (
                    raw_index,
                    pregrasp_position,
                    pregrasp_orientation,
                    grasp_position,
                    grasp_orientation,
                ) = self._current_candidate_poses(candidate_index)
                planner_result = self.planning.plan_pose_result(
                    grasp_position,
                    grasp_orientation,
                    collision_policy=CollisionPolicy.TARGET_APPROACH,
                    active_target=self.pick_obj.name,
                )
                success_mask = GraspPlanEvaluator._success_mask(planner_result)
                paths = GraspPlanEvaluator._result_paths(planner_result)
                if len(success_mask) == 1:
                    candidate_success = bool(success_mask[0])
                    candidate_path = paths[0] if paths else None
                elif candidate_index < len(success_mask):
                    candidate_success = bool(success_mask[candidate_index])
                    candidate_path = (
                        paths[candidate_index]
                        if candidate_index < len(paths)
                        else None
                    )
                else:
                    candidate_success = False
                    candidate_path = None
                if not candidate_success:
                    self._candidate_failed_indices.add(candidate_index)
                    planner_reason = getattr(planner_result, "reason", None)
                    failure_reason = "terminal_plan_failure"
                    if planner_reason:
                        failure_reason = f"{failure_reason}:{planner_reason}"
                    self._record_terminal_candidate_event(
                        candidate_index,
                        success=False,
                        reason=failure_reason,
                        replan_index=replan_index,
                        planner_result=planner_result,
                        attempt_index=attempt_index,
                        raw_index=raw_index,
                        world_revision=world_revision,
                    )
                    continue
                if candidate_path is None:
                    self._candidate_failed_indices.add(candidate_index)
                    self._record_terminal_candidate_event(
                        candidate_index,
                        success=False,
                        reason="terminal_plan_missing_path",
                        replan_index=replan_index,
                        planner_result=planner_result,
                        attempt_index=attempt_index,
                        raw_index=raw_index,
                        world_revision=world_revision,
                    )
                    continue

                # Apply the same Cartesian safety bounds as initial candidate
                # evaluation to a path generated from the measured hold.
                start_position, _ = self.planning.ee_pose()
                path_ratio, path_deviation = self.planning.measure_cartesian_path(
                    candidate_path,
                    start_position,
                    grasp_position,
                )
                if path_ratio > 1.5 or path_deviation > 0.01:
                    self._candidate_failed_indices.add(candidate_index)
                    self._record_terminal_candidate_event(
                        candidate_index,
                        success=False,
                        reason="terminal_path_geometry_invalid",
                        replan_index=replan_index,
                        planner_result=planner_result,
                        attempt_index=attempt_index,
                        raw_index=raw_index,
                        world_revision=world_revision,
                    )
                    continue

                self._sync_replacement_candidate_targets(
                    command,
                    old_candidate_index=current_index,
                    candidate_index=candidate_index,
                    raw_index=raw_index,
                    pregrasp_position=pregrasp_position,
                    pregrasp_orientation=pregrasp_orientation,
                    grasp_position=grasp_position,
                    grasp_orientation=grasp_orientation,
                    world_revision=world_revision,
                )
                command.collision_policy = CollisionPolicy.TARGET_APPROACH
                command.params["preplanned_joint_path"] = candidate_path
                command.params["candidate_index"] = int(candidate_index)
                command.params["candidate_raw_index"] = int(raw_index)
                command.params["candidate_world_revision"] = int(world_revision)
                command.params["candidate_replan_index"] = int(replan_index)
                command.params["path_length_ratio"] = float(path_ratio)
                command.params["path_max_deviation_m"] = float(path_deviation)

                evaluation.result.selected_grasp_index = int(candidate_index)
                if candidate_index < len(self.sampled_scores):
                    evaluation.result.selected_grasp_score = float(
                        self.sampled_scores[candidate_index]
                    )
                if candidate_index >= len(evaluation.terminal_paths):
                    evaluation.terminal_paths.extend(
                        [None]
                        * (candidate_index + 1 - len(evaluation.terminal_paths))
                    )
                evaluation.terminal_paths[candidate_index] = candidate_path
                evaluation.terminal_path = candidate_path
                evaluation.terminal_path_length_ratio = float(path_ratio)
                evaluation.terminal_path_max_deviation_m = float(path_deviation)
                self._record_terminal_candidate_event(
                    candidate_index,
                    success=True,
                    reason="terminal_candidate_replanned",
                    replan_index=replan_index,
                    planner_result=planner_result,
                    attempt_index=attempt_index,
                    raw_index=raw_index,
                    world_revision=world_revision,
                )
                self._selected_candidate_debug = {
                    "candidate_index": int(candidate_index),
                    "replan_index": int(replan_index),
                    "reason": "terminal_candidate_replanned",
                }
                return True
            except Exception as exc:  # Candidate failure; try the next rank.
                self._candidate_failed_indices.add(candidate_index)
                self._record_terminal_candidate_event(
                    candidate_index,
                    success=False,
                    reason=f"terminal_replan_exception:{type(exc).__name__}",
                    replan_index=replan_index,
                    attempt_index=attempt_index,
                    raw_index=raw_index,
                    world_revision=world_revision,
                )

        self.failure_reason = "ALL_TERMINAL_GRASP_CANDIDATES_FAILED"
        self.error_message = (
            "All terminal grasp candidates failed after safety recovery; "
            f"replan_index={replan_index}, "
            f"failed_candidates={sorted(self._candidate_failed_indices)}"
        )
        self._candidate_replan_exhausted = True
        self._candidate_replan_exhausted_reason = "all_candidates_failed"
        command.params["candidate_replan_exhausted"] = True
        command.params["candidate_replan_exhausted_at"] = int(replan_index)
        command.params["candidate_replan_exhausted_reason"] = "all_candidates_failed"
        command.params["candidate_replan_limit"] = int(self._candidate_replan_limit)
        self._write_debug_artifact(
            "pick_candidate_replan.json",
            {
                "object": self.pick_obj.name,
                "terminal_replan_count": int(self._terminal_replan_count),
                "candidate_rank_order": list(self._candidate_rank_order),
                "failed_candidate_indices": sorted(
                    int(index) for index in self._candidate_failed_indices
                ),
                "candidate_replan_limit": int(self._candidate_replan_limit),
                "candidate_replan_attempted": int(self._terminal_replan_count),
                "candidate_replan_exhausted": True,
                "candidate_replan_exhausted_reason": "all_candidates_failed",
                "events": self._candidate_replan_diagnostics,
            },
        )
        self.process_valid = False
        try:
            self.planning.restore_world(self.pick_obj.name)
        except Exception:
            LOGGER.exception(
                "[PickSafety] failed to restore world after terminal candidates were exhausted"
            )
        self.manip_list[:] = []
        return False

    def replan_after_safety(self, command, reason=None):
        """Recover a Pick phase through the narrow planning port.

        Terminal safety failures retire the active grasp candidate.  Other
        phases retain the existing moving-object retarget behavior.
        """

        if (
            isinstance(command, MotionPhaseCommand)
            and command.phase == MotionPhase.TERMINAL_GRASP_APPROACH
            and command.active_object == self.pick_obj.name
            and self.plan_evaluation is not None
        ):
            return self._replan_terminal_candidate(
                command,
                reason=(
                    str(getattr(reason, "value", reason))
                    if reason is not None
                    else "terminal_safety_failure"
                ),
            )
        return self.planning.replan_after_safety(
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
        planning = self.planning
        frame_debug = planning.frame_debug()
        mobile_base_prim_path = str(frame_debug.get("mobile_base_prim_path") or "").strip()
        obj_world_t, obj_world_q = self._get_object_world_pose()
        ee_base_t, ee_base_q = planning.ee_pose()
        robot_world_t, robot_world_q = self.robot.get_world_pose()
        reference_world_pose = None
        reference_prim_path = str(planning.reference_prim_path or "").strip()
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
            "reference_prim_path": planning.reference_prim_path,
            "reference_world_pose": reference_world_pose,
            "mobile_base_prim_path": mobile_base_prim_path if mobile_base_prim_path else None,
            "cached_mobile_to_armbase_tf": frame_debug.get("cached_mobile_to_armbase_tf"),
            "configured_mobile_to_armbase_translation": frame_debug.get(
                "configured_mobile_to_armbase_translation"
            ),
            "configured_mobile_to_armbase_orientation": frame_debug.get(
                "configured_mobile_to_armbase_orientation"
            ),
            "controller_lr_name": planning.lr_name,
            "controller_robot_file": planning.robot_file,
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
        priority_index = GraspPlanEvaluator.choose_candidate_index(pre_result, result)
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
            approach_axis = T_base_ee_grasps[idx, :3, self.planning.grasp_approach_axis]
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
        self._candidate_rank_order = [int(item[2]) for item in scored]
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
        self._candidate_rank_order = []
        self._candidate_failed_indices.clear()
        self._candidate_failure_diagnostics.clear()
        self._candidate_replan_diagnostics.clear()
        self._terminal_replan_count = 0
        self._candidate_replan_limit = 0
        self._candidate_replan_exhausted_recorded = False
        self._candidate_replan_exhausted = False
        self._candidate_replan_exhausted_reason = ""
        self._candidate_raw_indices = np.empty((0,), dtype=int)
        object_name = self.pick_obj.name
        pick_place_cfg = self.task.cfg.get("planning", {}).get("pick_place", {})
        max_terminal = float(pick_place_cfg.get("max_terminal_distance_m", 0.10))
        planning = self.planning
        planning.prepare_world(object_name)

        self._capture_pick_plan_reference()
        transforms = self.sample_ee_pose()
        evaluator = GraspPlanEvaluator(planning, self._debug_log)
        missing = evaluator.missing_attach_prims(self.pick_obj.attach_collision_prim_paths)
        self.plan_evaluation = evaluator.evaluate(
            transforms,
            self.sampled_scores,
            pregrasp_offset_m=float(self.skill_cfg.get("pre_grasp_offset", 0.1)),
            attach_prim_paths=self.pick_obj.attach_collision_prim_paths,
            fixed_orientation=self.fixed_orientation,
            attach_config_failure_code=self.pick_obj.attach_collision_failure_code,
            attach_candidate_paths=self.pick_obj.attach_collision_candidates,
            attach_missing_paths=missing,
            prepare_pregrasp_world=lambda: planning.transition_target(
                object_name, collision_policy=CollisionPolicy.WORLD_TRANSIT
            ),
            prepare_grasp_world=lambda: planning.transition_target(
                object_name, collision_policy=CollisionPolicy.TARGET_APPROACH
            ),
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
        if not self._candidate_rank_order:
            selected_index = result.selected_grasp_index
            if selected_index is not None:
                self._candidate_rank_order.append(int(selected_index))
            self._candidate_rank_order.extend(
                index
                for index in range(len(self.plan_evaluation.grasp_positions))
                if index not in self._candidate_rank_order
            )
        if result.feasible:
            # The supervisor's ordinary phase budget is intentionally small,
            # but terminal Pick recovery is bounded by the unique physical
            # candidates available to this plan.  One initial candidate does
            # not consume a replan, so N unique candidates permit N-1
            # replacements.  Keep an explicit hard ceiling for pathological
            # annotation batches without changing other skills' budget.
            self._candidate_replan_limit = self._terminal_candidate_replan_budget()
        # Candidate testing leaves the owner in the terminal world.  Execution
        # always starts again from the complete transit world.
        planning.restore_world(object_name)
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
                world_collision_diagnostic = planning.diagnose_start_collision()
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
                "pregrasp_plan_diagnostics": self.plan_evaluation.pregrasp_plan_diagnostics,
                "terminal_plan_diagnostics": self.plan_evaluation.terminal_plan_diagnostics,
                "post_grasp_validation": self.plan_evaluation.post_grasp_validation,
                "candidate_rank_order": self._candidate_rank_order,
                "candidate_failure_diagnostics": self._candidate_failure_diagnostics,
                "candidate_replan_diagnostics": self._candidate_replan_diagnostics,
                "terminal_replan_count": int(self._terminal_replan_count),
                "candidate_replan_limit": int(self._candidate_replan_limit),
                "candidate_replan_attempted": int(self._terminal_replan_count),
                "candidate_replan_exhausted": bool(
                    self._candidate_replan_exhausted
                ),
                "candidate_replan_exhausted_reason": str(
                    self._candidate_replan_exhausted_reason
                ),
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
        self.manip_list = self.planning.build_commands(
            object_name=object_name,
            pregrasp_position=pre_positions[index],
            pregrasp_orientation=pre_orientations[index],
            grasp_position=positions[index],
            grasp_orientation=orientations[index],
            gripper_action=self.gripper_cmd,
            post_grasp_offset=post_grasp_offset,
            terminal_path=self.plan_evaluation.terminal_path,
            pregrasp_path=self.plan_evaluation.pregrasp_path,
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
        for command in self.manip_list:
            if command.phase == MotionPhase.TERMINAL_GRASP_APPROACH:
                command.params["candidate_index"] = int(index)
                if index < len(self._candidate_raw_indices):
                    command.params["candidate_raw_index"] = int(
                        self._candidate_raw_indices[index]
                    )
                command.params["candidate_world_revision"] = int(
                    planning.world_revision
                )
                command.candidate_replan_limit = int(self._candidate_replan_limit)
                command.replan_policy = "terminal_candidate_fallback"
                command.params["candidate_replan_limit"] = int(
                    self._candidate_replan_limit
                )
                command.metadata = {
                    **dict(command.metadata or {}),
                    "candidate_index": int(index),
                    "candidate_world_revision": int(planning.world_revision),
                    "candidate_replan_limit": int(self._candidate_replan_limit),
                    "replan_policy": "terminal_candidate_fallback",
                }
                break
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
        # Execution evidence is derived from the typed command itself.  The
        # controller's mutable action cache is intentionally not part of the
        # Skill boundary.
        gripper_action = current_cmd.gripper_action
        arm_action = current_cmd.joint_target
        action_joint_positions = arm_action
        action_joint_indices = self.skill_runtime.arm_indices
        qpos = self.robot.get_joints_state().positions
        arm_indices = self.skill_runtime.arm_indices
        gripper_indices = self.skill_runtime.gripper_indices
        try:
            actual_arm_position = qpos[arm_indices]
        except Exception:
            actual_arm_position = []
        try:
            actual_gripper_position = qpos[gripper_indices]
        except Exception:
            actual_gripper_position = []

        obj_t, obj_q = self._get_object_world_pose()
        ee_t, ee_q = self.planning.ee_pose()
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
        command_status = self.skill_runtime.execution_status(current_cmd)
        self._execution_trace.append(
            {
                "step": self._execution_trace_total_steps - 1,
                "remaining_commands": len(self.manip_list),
                "current_command": self._manip_cmd_to_debug(current_cmd),
                "target_diff_trans": diff_trans,
                "target_diff_ori": diff_ori,
                "command_age_steps": int(command_age_steps),
                "controller_gripper_state": gripper_action,
                "controller_command_status": (
                    self.command_status_debug(command_status)
                ),
                "controller_num_last_cmd": self.planning.last_command_count,
                "controller_num_plan_failed": self.planning.plan_failure_count,
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
                        "num_last_cmd": int(self.planning.last_command_count),
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
    def sample_ee_pose(self, max_length=CUROBO_BATCH_SIZE):
        T_base_ee = self.get_ee_poses("armbase")

        num_pose = T_base_ee.shape[0]
        unique_raw_indices = self._deduplicated_raw_grasp_indices()
        if len(self._raw_grasp_keys) != num_pose:
            raise ValueError(
                "raw grasp transform count changed without rebuilding raw identities"
            )
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
        filtered_unique_indices = [
            raw_index
            for raw_index in unique_raw_indices
            if bool(combined_flag[raw_index])
        ]
        if not filtered_unique_indices:
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
                idx_list = filtered_unique_indices[: min(max_length, len(filtered_unique_indices))]
                if not idx_list:
                    idx_list = unique_raw_indices[: min(max_length, len(unique_raw_indices))]
                sampled_scores = self.scores[idx_list]
                LOGGER.warning(
                    "[PickDebug] grasp filters rejected all candidates for object=%s filters=%s; "
                    "falling back to first %d candidates",
                    self.pick_obj.name,
                    filter_summaries,
                    max_length,
                )
        else:
            tmp_idxs = np.asarray(filtered_unique_indices, dtype=int)
            tmp_scores = self.scores[tmp_idxs]
            combined = list(zip(tmp_scores, tmp_idxs))
            combined.sort()
            ranked_indices = [int(idx) for (_score, idx) in combined[:max_length]]
            # Preserve the pre-migration planner input semantics: rank the
            # best candidates, then draw a full batch with replacement.  A
            # repeated physical candidate is intentional here.  It keeps the
            # native batch size stable when filtering leaves fewer candidates
            # than the planner batch and gives promising grasps multiple
            # native attempts.  The typed planner still receives only typed
            # poses; this does not reintroduce the legacy Controller API.
            score_list = np.asarray(self.scores[ranked_indices], dtype=float)
            weights = 1.0 / (score_list + 1e-8)
            weights = weights / weights.sum()
            sampled_idx = random.choices(
                ranked_indices,
                weights=weights,
                k=max_length,
            )
            sampled_scores = self.scores[sampled_idx]

            # Sort indices by their scores (ascending)
            sorted_pairs = sorted(zip(sampled_scores, sampled_idx))
            idx_list = [idx for _, idx in sorted_pairs]
            sampled_scores = [score for score, _ in sorted_pairs]

        # ``idx_list`` may intentionally contain repeated raw indices after
        # score-weighted sampling with replacement.  Only the source pose
        # array bounds the returned batch length.
        valid_length = min(len(idx_list), num_pose)
        idx_list = idx_list[:valid_length]
        sampled_scores = sampled_scores[:valid_length]
        self._candidate_raw_indices = np.asarray(idx_list, dtype=int)
        self._sample_debug = {
            "candidate_count": int(num_pose),
            "unique_candidate_count": int(len(unique_raw_indices)),
            "filtered_candidate_count": int(np.sum(combined_flag)),
            "filter_pass_counts": {axis: int(np.sum(flag)) for axis, flag in flags.items()},
            "sampled_indices": [int(idx) for idx in idx_list],
            "sampled_raw_indices": [int(idx) for idx in self._candidate_raw_indices],
            "deduplicated_raw_indices": [int(idx) for idx in unique_raw_indices],
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
        current_cmd = self.manip_list[0] if self.manip_list else None
        if (
            current_cmd is not None
            and current_cmd.params.get("candidate_replan_exhausted", False)
        ):
            exhausted_reason = str(
                current_cmd.params.get(
                    "candidate_replan_exhausted_reason", "replan_limit"
                )
            )
            if exhausted_reason == "all_candidates_failed":
                self.failure_reason = "ALL_TERMINAL_GRASP_CANDIDATES_FAILED"
            else:
                self.failure_reason = "TERMINAL_CANDIDATE_REPLAN_LIMIT"
            self._candidate_replan_exhausted = True
            self._candidate_replan_exhausted_reason = exhausted_reason
            if exhausted_reason == "all_candidates_failed":
                self.error_message = (
                    "All terminal grasp candidates exhausted; "
                    f"attempted={self._terminal_replan_count}, "
                    f"limit={self._candidate_replan_limit}"
                )
            else:
                self.error_message = (
                    "Terminal candidate recovery safety budget exhausted; "
                    f"attempted={self._terminal_replan_count}, "
                    f"limit={self._candidate_replan_limit}"
                )
            self.process_valid = False
            if not getattr(self, "_candidate_replan_exhausted_recorded", False):
                candidate_index = int(current_cmd.params.get("candidate_index", -1))
                if candidate_index >= 0:
                    self._candidate_failed_indices.add(candidate_index)
                    self._record_terminal_candidate_event(
                        candidate_index,
                        success=False,
                        reason=(
                            "candidate_replan_limit_exhausted"
                            if exhausted_reason != "all_candidates_failed"
                            else "all_candidates_exhausted"
                        ),
                        replan_index=int(
                            current_cmd.params.get(
                                "candidate_replan_exhausted_at",
                                self._terminal_replan_count,
                            )
                        ),
                        raw_index=current_cmd.params.get("candidate_raw_index"),
                        world_revision=current_cmd.params.get(
                            "candidate_world_revision"
                        ),
                    )
                self._candidate_replan_exhausted_recorded = True
        feasible = self.planning.plan_failure_count <= th and bool(self.process_valid)
        if not feasible and not self._runtime_failure_snapshot_written:
            self._runtime_failure_debug_path = self._write_debug_artifact(
                "pick_runtime_failure_snapshot.json",
                {
                    "robot": self.robot.name,
                    "object": self.pick_obj.name,
                    "lr_arm": self.lr_arm,
                    "num_plan_failed": int(self.planning.plan_failure_count),
                    "failure_threshold": int(th),
                    "num_last_cmd": int(self.planning.last_command_count),
                    "failure_reason": getattr(self, "failure_reason", ""),
                    "error_message": getattr(self, "error_message", ""),
                    "selected_candidate": self._selected_candidate_debug,
                    "candidate_replan_attempted": int(self._terminal_replan_count),
                    "candidate_replan_limit": int(self._candidate_replan_limit),
                    "candidate_replan_exhausted": bool(
                        self._candidate_replan_exhausted
                        or (
                            current_cmd is not None
                            and current_cmd.params.get(
                                "candidate_replan_exhausted", False
                            )
                        )
                    ),
                    "candidate_replan_exhausted_reason": str(
                        self._candidate_replan_exhausted_reason
                        or (
                            current_cmd.params.get(
                                "candidate_replan_exhausted_reason", ""
                            )
                            if current_cmd is not None
                            else ""
                        )
                    ),
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
        done = self.planning.phase_complete(command)
        if done and command.phase == MotionPhase.GRIPPER_CLOSE:
            threshold = float(command.params.get("contact_threshold_n", 0.0))
            _, indices = self.get_contact(contact_threshold=threshold)
            self._grasp_contact_verified = len(indices) >= 1
            command.params["contact_verified"] = self._grasp_contact_verified
            if not self._grasp_contact_verified:
                self.failure_reason = "GRASP_CONTACT_MISSING"
                # Do not permit the next ATTACH phase. Restore the target
                # to the complete world before ending this failed Pick.
                self.planning.restore_world(self.pick_obj.name)
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
        object_position = deepcopy(self._get_object_world_pose()[0])
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
