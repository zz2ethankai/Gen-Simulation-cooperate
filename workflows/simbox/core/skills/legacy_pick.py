"""Legacy stage-scan Pick skill.

The public ``pick`` skill uses Physics-schema MotionPhaseCommand execution.
This module keeps the historical tuple-command implementation available under
the explicit ``legacy_pick`` skill name.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from core.planning.grasp_plan_evaluator import GraspPlanEvaluator
from core.planning.motion_command import MotionPhaseCommand
from core.skills.base_skill import register_skill
from core.skills.pick import Pick
from core.utils.transformation_utils import poses_from_tf_matrices


@register_skill
class LegacyPick(Pick):
    """Tuple-command Pick for explicitly selected legacy stage-scan tasks."""

    def _candidate_source_debug(self, candidate_index: int):
        sampled_indices = self._sample_debug.get("sampled_indices", [])
        sampled_scores = self._sample_debug.get("sampled_scores", [])
        source_index = (
            sampled_indices[candidate_index]
            if candidate_index < len(sampled_indices)
            else None
        )
        source_score = (
            sampled_scores[candidate_index]
            if candidate_index < len(sampled_scores)
            else None
        )
        return {
            "candidate_index": candidate_index,
            "source_index": source_index,
            "source_score": source_score,
        }

    def _manip_cmd_to_debug(self, manip_cmd):
        if isinstance(manip_cmd, MotionPhaseCommand):
            return super()._manip_cmd_to_debug(manip_cmd)
        ee_trans, ee_ori, cmd_name, params = manip_cmd
        return {
            "command": cmd_name,
            "ee_translation": ee_trans,
            "ee_orientation": ee_ori,
            "params": params,
        }

    def _execution_command_state(self, command, fallback_translation, fallback_orientation):
        if isinstance(command, MotionPhaseCommand):
            return super()._execution_command_state(
                command, fallback_translation, fallback_orientation
            )
        target_translation, target_orientation, command_name, _ = command
        return target_translation, target_orientation, str(command_name)

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
        t_eps = min(
            float(self.skill_cfg.get("t_eps", 1e-3)),
            float(self.skill_cfg.get("grasp_t_eps", 0.008)),
        )
        o_eps = min(
            float(self.skill_cfg.get("o_eps", 5e-3)),
            float(self.skill_cfg.get("grasp_o_eps", 0.2)),
        )
        return {"t_eps": t_eps, "o_eps": o_eps}

    def _offset_from_grasp(self, grasp_translation, grasp_transform, offset):
        if offset <= 0.0:
            return grasp_translation
        axis_index = 0 if "r5a" in self.controller.robot_file else 2
        return grasp_translation - grasp_transform[:3, axis_index] * offset

    def _evaluate_legacy_candidates(
        self,
        grasp_transforms,
        grasp_orientations,
        postgrasp_positions,
        post_grasp_offset,
        base_ignore_substring,
        grasp_ignore_substring,
    ):
        evaluator = GraspPlanEvaluator(self.controller, self._debug_log)
        postgrasp_validator = None
        test_mode = str(self.skill_cfg.get("test_mode", "forward"))
        if test_mode == "forward" and post_grasp_offset > 0.0:

            def validate_postgrasp(candidate_index, terminal_path):
                command_path = self.controller._command_path(terminal_path)
                start_arm_positions = np.asarray(
                    command_path[-1].position.detach().cpu(), dtype=float
                )
                self.controller.update_specific(
                    ignore_substring=base_ignore_substring,
                    reference_prim_path=self.controller.reference_prim_path,
                )
                success, _, _ = self.controller.test_attached_forward_from_joint_positions(
                    postgrasp_positions[candidate_index],
                    grasp_orientations[candidate_index],
                    start_arm_positions=start_arm_positions,
                    obj_prim_paths=list(self.pick_obj.attach_collision_prim_paths),
                )
                return bool(success)

            postgrasp_validator = validate_postgrasp

        try:
            return evaluator.evaluate(
                grasp_transforms,
                self.sampled_scores,
                pregrasp_offset_m=float(self.skill_cfg.get("pre_grasp_offset", 0.1)),
                attach_prim_paths=self.pick_obj.attach_collision_prim_paths,
                fixed_orientation=self.fixed_orientation,
                test_mode=test_mode,
                attach_config_failure_code=self.pick_obj.attach_collision_failure_code,
                attach_candidate_paths=self.pick_obj.attach_collision_candidates,
                prepare_pregrasp_world=lambda: self.controller.update_specific(
                    ignore_substring=base_ignore_substring,
                    reference_prim_path=self.controller.reference_prim_path,
                ),
                prepare_grasp_world=lambda: self.controller.update_specific(
                    ignore_substring=grasp_ignore_substring,
                    reference_prim_path=self.controller.reference_prim_path,
                ),
                candidate_selector=(
                    lambda pre_result, result, candidate_indices, positions, orientations, transforms: self._select_grasp_index(
                        pre_result,
                        result,
                        positions,
                        orientations,
                        transforms,
                        candidate_indices=candidate_indices,
                    )
                ),
                postgrasp_validator=postgrasp_validator,
            )
        finally:
            self.controller.update_specific(
                ignore_substring=base_ignore_substring,
                reference_prim_path=self.controller.reference_prim_path,
            )

    def simple_generate_manip_cmds(self):
        return self._legacy_simple_generate_manip_cmds()

    def _legacy_simple_generate_manip_cmds(self):
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
            "start legacy object=%s arm=%s use_batch=%s test_mode=%s pre_grasp_offset=%s"
            % (
                object_name,
                getattr(self.controller, "lr_name", "unknown"),
                self.controller.use_batch,
                self.skill_cfg.get("test_mode", "forward"),
                self.skill_cfg.get("pre_grasp_offset", 0.1),
            )
        )

        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        open_gripper_action = self._gripper_action_for_state("open_gripper")
        manip_list.append(
            (
                p_base_ee_cur,
                q_base_ee_cur,
                "update_pose_cost_metric",
                {"hold_vec_weight": None, "gripper_action": open_gripper_action},
            )
        )

        base_ignore_substring = deepcopy(
            self.controller.ignore_substring
            + self.skill_cfg.get("ignore_substring", [])
        )
        grasp_ignore_substring = deepcopy(base_ignore_substring)
        grasp_ignore_substring.append(self.pick_obj.name)

        grasp_transforms = self.sample_ee_pose()
        pregrasp_transforms = deepcopy(grasp_transforms)
        self.controller.update_specific(
            ignore_substring=base_ignore_substring,
            reference_prim_path=self.controller.reference_prim_path,
        )
        axis_index = 0 if "r5a" in self.controller.robot_file else 2
        pregrasp_transforms[:, :3, 3] -= (
            pregrasp_transforms[:, :3, axis_index]
            * self.skill_cfg.get("pre_grasp_offset", 0.1)
        )

        if len(grasp_transforms) == 0:
            self._fail_without_candidates(object_name)
            return

        pre_positions, pre_orientations = poses_from_tf_matrices(pregrasp_transforms)
        positions, orientations = poses_from_tf_matrices(grasp_transforms)
        if self.fixed_orientation is not None:
            pre_orientations[:] = self.fixed_orientation
            orientations[:] = self.fixed_orientation

        post_grasp_offset = float(
            np.random.uniform(
                self.skill_cfg.get("post_grasp_offset_min", 0.05),
                self.skill_cfg.get("post_grasp_offset_max", 0.05),
            )
        )
        post_positions = deepcopy(positions)
        post_positions[:, 2] += post_grasp_offset
        self.plan_evaluation = self._evaluate_legacy_candidates(
            grasp_transforms,
            orientations,
            post_positions,
            post_grasp_offset,
            base_ignore_substring,
            grasp_ignore_substring,
        )
        result = self.plan_evaluation.result
        candidate_order = list(range(len(grasp_transforms)))
        index = result.selected_grasp_index
        candidate_results = [
            {
                **self._candidate_source_debug(candidate_index),
                "pregrasp_translation": pre_positions[candidate_index],
                "pregrasp_orientation": pre_orientations[candidate_index],
                "grasp_translation": positions[candidate_index],
                "grasp_orientation": orientations[candidate_index],
                "postgrasp_translation": post_positions[candidate_index],
                "postgrasp_orientation": orientations[candidate_index],
                "terminal_path_available": (
                    candidate_index < len(self.plan_evaluation.terminal_paths)
                    and self.plan_evaluation.terminal_paths[candidate_index] is not None
                ),
            }
            for candidate_index in candidate_order
        ]
        validation_by_index = {
            int(item["candidate_index"]): item
            for item in self.plan_evaluation.post_grasp_validation
            if item.get("candidate_index") is not None
        }
        for candidate_debug in candidate_results:
            candidate_debug["postgrasp_validation"] = validation_by_index.get(
                int(candidate_debug["candidate_index"])
            )

        if not result.feasible or index is None:
            self._fail_legacy_plan(object_name, len(grasp_transforms), candidate_results, candidate_order, post_grasp_offset)
            return

        index = int(index)
        self._selected_candidate_debug = {
            **self._candidate_source_debug(index),
            "success_found": True,
            "selected_pregrasp_translation": pre_positions[index],
            "selected_pregrasp_orientation": pre_orientations[index],
            "selected_grasp_translation": positions[index],
            "selected_grasp_orientation": orientations[index],
            "selected_postgrasp_translation": post_positions[index],
            "selected_postgrasp_orientation": orientations[index],
            "post_grasp_offset": post_grasp_offset,
        }
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

        manip_list.extend(
            [
                (
                    p_base_ee_cur,
                    q_base_ee_cur,
                    "update_specific",
                    {
                        "ignore_substring": base_ignore_substring,
                        "reference_prim_path": self.controller.reference_prim_path,
                        "gripper_action": open_gripper_action,
                    },
                ),
                (pre_positions[index], pre_orientations[index], "open_gripper", {}),
            ]
        )
        if self.skill_cfg.get("pre_grasp_hold_vec_weight", None) is not None:
            manip_list.append(
                (
                    pre_positions[index],
                    pre_orientations[index],
                    "update_pose_cost_metric",
                    {"hold_vec_weight": self.skill_cfg["pre_grasp_hold_vec_weight"]},
                )
            )

        manip_list.append(
            (
                pre_positions[index],
                pre_orientations[index],
                "update_specific",
                {
                    "ignore_substring": grasp_ignore_substring,
                    "reference_prim_path": self.controller.reference_prim_path,
                    "gripper_action": open_gripper_action,
                },
            )
        )
        guard_offset = float(
            self.skill_cfg.get(
                "grasp_guard_offset",
                min(float(self.skill_cfg.get("pre_grasp_offset", 0.1)), 0.04),
            )
        )
        if guard_offset > 0.0:
            manip_list.append(
                (
                    self._offset_from_grasp(
                        positions[index], grasp_transforms[index], guard_offset
                    ),
                    orientations[index],
                    "open_gripper",
                    {"gripper_action": open_gripper_action},
                )
            )

        manip_list.append(
            (positions[index], orientations[index], "open_gripper", self._grasp_arrival_params())
        )
        self._append_gripper_hold(
            manip_list, positions[index], orientations[index], "open_gripper"
        )
        self._append_gripper_transition(
            manip_list, positions[index], orientations[index], self.gripper_cmd
        )
        self._append_gripper_hold(
            manip_list,
            positions[index],
            orientations[index],
            self.gripper_cmd,
            steps=self.skill_cfg.get("post_close_hold_steps", 12),
        )
        manip_list.extend(
            [
                (
                    positions[index],
                    orientations[index],
                    "update_specific",
                    {
                        "ignore_substring": base_ignore_substring,
                        "reference_prim_path": self.controller.reference_prim_path,
                    },
                ),
                (
                    positions[index],
                    orientations[index],
                    "attach_objects",
                    {
                        "obj_prim_paths": list(self.pick_obj.attach_collision_prim_paths),
                        "skip_plan": True,
                    },
                ),
            ]
        )
        if post_grasp_offset:
            manip_list.append(
                (
                    post_positions[index],
                    orientations[index],
                    self.gripper_cmd,
                    {"gripper_action": self._gripper_action_for_state(self.gripper_cmd)},
                )
            )
        if self.skill_cfg.get("return_to_pregrasp", False):
            manip_list.append(
                (pre_positions[index], pre_orientations[index], self.gripper_cmd, {})
            )

        self.manip_list = manip_list
        self._write_legacy_plan_snapshot(
            object_name, True, candidate_results, post_grasp_offset
        )

    def _fail_without_candidates(self, object_name):
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
                "reason": self.failure_reason,
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

    def _fail_legacy_plan(
        self, object_name, candidate_count, candidate_results, candidate_order, post_grasp_offset
    ):
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
        self._write_legacy_plan_snapshot(
            object_name, False, candidate_results, post_grasp_offset
        )
        self.publish_target_intent(
            {
                "kind": "pick",
                "objects": [object_name],
                "has_target": False,
                "failure_reason": self.failure_reason,
                "candidate_count": candidate_count,
                "constraints": self._target_constraints(),
            }
        )

    def _write_legacy_plan_snapshot(
        self, object_name, success_found, candidate_results, post_grasp_offset
    ):
        self._plan_debug_path = self._write_debug_artifact(
            "pick_plan_snapshot.json",
            {
                "robot": self.robot.name,
                "object": object_name,
                "lr_arm": self.lr_arm,
                "collision_world_mode": "legacy_stage_scan",
                "success_found": success_found,
                "selected_candidate": self._selected_candidate_debug,
                "sample_debug": self._sample_debug,
                "geometry_debug": self._collect_geometry_debug(),
                "plan_evaluation": self.plan_evaluation.result.to_dict(),
                "terminal_plan_diagnostics": self.plan_evaluation.terminal_plan_diagnostics,
                "post_grasp_validation": self.plan_evaluation.post_grasp_validation,
                "candidate_results": candidate_results,
                "candidate_rank_debug": self._candidate_rank_debug,
                "post_grasp_offset": post_grasp_offset,
                "manip_command_sequence": [
                    self._manip_cmd_to_debug(cmd) for cmd in self.manip_list
                ],
            },
        )

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        assert self.manip_list
        if isinstance(self.manip_list[0], MotionPhaseCommand):
            return super().is_subtask_done(t_eps=t_eps, o_eps=o_eps)
        return self._legacy_is_subtask_done(t_eps=t_eps, o_eps=o_eps)

    def _legacy_is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        p_base_ee_cur, q_base_ee_cur = self.controller.get_ee_pose()
        p_base_ee, q_base_ee, gripper_fn, params = self.manip_list[0]
        diff_trans = np.linalg.norm(p_base_ee_cur - p_base_ee)
        diff_ori = 2 * np.arccos(min(abs(np.dot(q_base_ee_cur, q_base_ee)), 1.0))
        pose_flag = np.logical_and(diff_trans < t_eps, diff_ori < o_eps)
        if bool(params.get("skip_plan", False)) or gripper_fn in {
            "update_pose_cost_metric",
            "update_specific",
        }:
            self.plan_flag = True
            return True
        self.plan_flag = False
        return pose_flag

    def is_done(self):
        if not self.manip_list:
            return True
        current_cmd = self.manip_list[0]
        params = current_cmd.params if isinstance(current_cmd, MotionPhaseCommand) else current_cmd[3]
        if self.is_subtask_done(
            t_eps=params.get("t_eps", self.skill_cfg.get("t_eps", 1e-3)),
            o_eps=params.get("o_eps", self.skill_cfg.get("o_eps", 5e-3)),
        ):
            self.manip_list.pop(0)
        return not self.manip_list
