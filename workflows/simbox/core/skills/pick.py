import logging
import os
import random
from copy import deepcopy

import numpy as np
from core.planning.grasp_plan_evaluator import GraspPlanEvaluator
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.asset_path_utils import resolve_asset_path
from omegaconf import DictConfig
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    tf_matrix_from_pose,
)

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
        lr_arm = self.controller.lr_name
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
        self.plan_evaluation = evaluator.evaluate(
            transforms,
            self.sampled_scores,
            pregrasp_offset_m=float(self.skill_cfg.get("pre_grasp_offset", 0.1)),
            attach_prim_paths=self.pick_obj.attach_collision_prim_paths,
            fixed_orientation=self.fixed_orientation,
            test_mode=str(self.skill_cfg.get("test_mode", "forward")),
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

        T_base_ee_grasps = self.sample_ee_pose()
        evaluator = GraspPlanEvaluator(self.controller, self._debug_log)
        # Validate while the target is still present in the CuRobo world.  The
        # following update intentionally ignores it so grasp paths may enter
        # the object's current volume; the object is restored before attach.
        missing_attach_paths = evaluator.missing_attach_prims(
            self.pick_obj.attach_collision_prim_paths
        )
        self.controller.update_specific(
            ignore_substring=ignore_substring, reference_prim_path=self.controller.reference_prim_path
        )
        self.plan_evaluation = evaluator.evaluate(
            T_base_ee_grasps,
            self.sampled_scores,
            pregrasp_offset_m=float(self.skill_cfg.get("pre_grasp_offset", 0.1)),
            attach_prim_paths=self.pick_obj.attach_collision_prim_paths,
            fixed_orientation=self.fixed_orientation,
            test_mode=str(self.skill_cfg.get("test_mode", "forward")),
            attach_config_failure_code=self.pick_obj.attach_collision_failure_code,
            attach_candidate_paths=self.pick_obj.attach_collision_candidates,
            attach_missing_paths=missing_attach_paths,
            cartesian_ratio_limit=float(self.skill_cfg.get("cartesian_ratio_limit", 1.5)),
            cartesian_deviation_m=float(self.skill_cfg.get("cartesian_deviation_m", 0.01)),
        )
        plan_result = self.plan_evaluation.result
        if not plan_result.feasible:
            self.publish_target_intent(
                {
                    "kind": "pick",
                    "objects": [object_name],
                    "has_target": False,
                    "failure_reason": plan_result.failure_code or "no_feasible_grasp",
                    "candidate_count": len(T_base_ee_grasps),
                    "constraints": self._target_constraints(),
                }
            )
            LOGGER.warning(
                "[PickDebug] grasp planning rejected object=%s arm=%s failure=%s",
                object_name,
                plan_result.arm,
                plan_result.failure_code,
            )
            self.manip_list = []
            return
        index = int(plan_result.selected_grasp_index)
        p_base_ee_pregrasps = self.plan_evaluation.pregrasp_positions
        q_base_ee_pregrasps = self.plan_evaluation.pregrasp_orientations
        p_base_ee_grasps = self.plan_evaluation.grasp_positions
        q_base_ee_grasps = self.plan_evaluation.grasp_orientations
        self.publish_target_intent(
            {
                "kind": "pick",
                "objects": [object_name],
                "selected_index": index,
                "selected_score": plan_result.selected_grasp_score,
                "constraints": self._target_constraints(),
                "pregrasp_position": p_base_ee_pregrasps[index],
                "pregrasp_orientation": q_base_ee_pregrasps[index],
                "grasp_position": p_base_ee_grasps[index],
                "grasp_orientation": q_base_ee_grasps[index],
            }
        )

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
            "attach_objects",
            {"obj_prim_paths": self.pick_obj.attach_collision_prim_paths},
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
        if self.skill_cfg.get("direction_to_obj", None) is not None:
            direction_to_obj = self.skill_cfg["direction_to_obj"]
            T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_local_pose())
            T_base_world = get_relative_transform(
                get_prim_at_path(self.task.root_prim_path), get_prim_at_path(self.controller.reference_prim_path)
            )
            T_base_obj = T_base_world @ T_world_obj
            if direction_to_obj == "right":
                flags["direction_to_obj"] = T_base_ee[:, 1, 3] <= T_base_obj[1, 3]
            elif direction_to_obj == "left":
                flags["direction_to_obj"] = T_base_ee[:, 1, 3] > T_base_obj[1, 3]
            else:
                raise NotImplementedError
            filter_summaries.append(f"direction_to_obj:{direction_to_obj}->{int(flags['direction_to_obj'].sum())}/{num_pose}")

        combined_flag = np.logical_and.reduce(list(flags.values()))
        combined_count = int(combined_flag.sum())
        if sum(combined_flag) == 0:
            idx_list = list(range(min(max_length, num_pose)))
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

        selected_scores = self.scores[idx_list]
        self.sampled_scores = np.asarray(selected_scores, dtype=float)
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
                float(np.min(selected_scores)),
                float(np.max(selected_scores)),
                np.array2string(np.min(selected_trans, axis=0), precision=4, suppress_small=True),
                np.array2string(np.max(selected_trans, axis=0), precision=4, suppress_small=True),
            )
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

        T_world_obj = tf_matrix_from_pose(*self.pick_obj.get_local_pose())
        T_world_ee = T_world_obj[None] @ self.T_obj_ee

        if frame == "world":
            return T_world_ee

        if frame == "armbase":  # arm base frame
            T_world_base = get_relative_transform(
                get_prim_at_path(self.controller.reference_prim_path), get_prim_at_path(self.task.root_prim_path)
            )
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
        return self.controller.num_plan_failed <= th

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
        p_base_ee, q_base_ee, *_ = self.manip_list[0]
        diff_trans = np.linalg.norm(p_base_ee_cur - p_base_ee)
        diff_ori = 2 * np.arccos(min(abs(np.dot(q_base_ee_cur, q_base_ee)), 1.0))
        pose_flag = np.logical_and(
            diff_trans < t_eps,
            diff_ori < o_eps,
        )
        self.plan_flag = self.controller.num_last_cmd > 10
        return np.logical_or(pose_flag, self.plan_flag)
        # LEGACY_END

    def is_done(self):
        if len(self.manip_list) == 0:
            return True
        if self.is_subtask_done(t_eps=self.skill_cfg.get("t_eps", 1e-3), o_eps=self.skill_cfg.get("o_eps", 5e-3)):
            self.manip_list.pop(0)
        return len(self.manip_list) == 0

    def is_success(self):
        flag = True

        contact, indices = self.get_contact()
        if self.gripper_cmd == "close_gripper":
            flag = len(indices) >= 1
        if (
            getattr(self.controller, "collision_world_mode", "legacy_stage_scan")
            == "physics_schema"
            and self.failure_reason
        ):
            flag = False

        max_joint_velocity = float(np.max(np.abs(self.robot.get_joints_state().velocities)))
        max_object_velocity = float(np.max(np.abs(self.pick_obj.get_linear_velocity())))
        if self.skill_cfg.get("process_valid", True):
            self.process_valid = max_joint_velocity < 5 and max_object_velocity < 5
        flag = flag and self.process_valid

        object_translation = deepcopy(self.pick_obj.get_local_pose()[0])
        if self.skill_cfg.get("lift_th", 0.0) > 0.0:
            flag = flag and (
                (object_translation[2] - self.obj_init_trans[2]) > self.skill_cfg.get("lift_th", 0.0)
            )

        self._debug_log(
            "success-check object=%s result=%s contact_count=%d contact_max=%.6f "
            "process_valid=%s max_joint_velocity=%.6f max_object_velocity=%.6f object_delta_z=%.6f"
            % (
                self.pick_obj.name,
                bool(flag),
                len(indices),
                float(np.max(contact)) if np.size(contact) else 0.0,
                bool(self.process_valid),
                max_joint_velocity,
                max_object_velocity,
                float(object_translation[2] - self.obj_init_trans[2]),
            )
        )

        return flag
