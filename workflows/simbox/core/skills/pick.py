"""Candidate-driven Pick skill.

Pick owns grasp annotation filtering and candidate selection.  CuRobo planning
is accessed directly through the controller-owned typed runtime; execution phases are typed
commands consumed by the CuRobo controller.
"""

from copy import deepcopy
import logging
import os

import numpy as np
from core.planning.domain_types import BatchPlanResult, CollisionPolicy, PlanResult
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.utils.constants import CUROBO_BATCH_SIZE
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.asset_path_utils import resolve_asset_path
from core.utils.transformation_utils import poses_from_tf_matrices
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.transformations import pose_from_tf_matrix, tf_matrix_from_pose
from omegaconf import DictConfig


LOGGER = logging.getLogger("de_logger")


@register_skill
class Pick(BaseSkill):
    """Filter grasp candidates, query the runtime, and emit Pick phases."""

    def __init__(
        self,
        robot: Robot,
        skill_runtime,
        task: BaseTask,
        cfg: DictConfig,
        *args,
        **kwargs,
    ):
        del args, kwargs
        super().__init__()
        self.robot = robot
        self.bind_skill_runtime(skill_runtime)
        self._require_skill_runtime()
        self.task = task
        self.skill_cfg = cfg
        object_name = str(cfg["objects"][0])
        self.pick_obj = task.objects[object_name]

        object_cfg = next(obj for obj in task.cfg["objects"] if obj["name"] == object_name)
        usd_path = resolve_asset_path(task.asset_root, object_cfg)
        grasp_path = usd_path.replace(
            "Aligned_obj.usd", cfg.get("npy_name", "Aligned_grasp_sparse.npy")
        )
        sparse_grasps = np.load(grasp_path)
        self.lr_arm = self.skill_runtime.arm_name
        annotation_scale = object_cfg.get("grasp_annotation_scale", [1.0])
        if isinstance(annotation_scale, (list, tuple)):
            annotation_scale = annotation_scale[0]
        self.T_obj_ee, self.scores = robot.pose_post_process_fn(
            sparse_grasps,
            lr_arm=self.lr_arm,
            grasp_scale=cfg.get("grasp_scale", float(annotation_scale)),
            tcp_offset=cfg.get("tcp_offset", robot.tcp_offset),
            constraints=cfg.get("constraints", None),
        )
        self._candidate_raw_indices = np.empty((0,), dtype=int)
        self.sampled_scores = np.empty((0,), dtype=float)

        self.manip_list = []
        self.pickcontact_view = task.pickcontact_views[robot.name][self.lr_arm][object_name]
        self.process_valid = True
        self.failure_reason = ""
        self.obj_init_trans = deepcopy(self._object_world_pose()[0])
        final_state = cfg.get("final_gripper_state", -1)
        if final_state not in (-1, 1):
            raise ValueError("final_gripper_state must be 1 or -1")
        self.gripper_cmd = "close_gripper" if final_state == -1 else "open_gripper"
        self.fixed_orientation = cfg.get("fixed_orientation", None)
        if self.fixed_orientation is not None:
            self.fixed_orientation = np.asarray(self.fixed_orientation, dtype=float).reshape(4)
        self._grasp_contact_verified = False
        self.plan_state = None

    def _object_world_pose(self):
        getter = getattr(self.pick_obj, "get_world_pose", None)
        return getter() if callable(getter) else self.pick_obj.get_local_pose()

    def _arm_base_transform(self):
        return np.asarray(self.skill_runtime.execution.get_pick_armbase_transform(), dtype=float)

    def _object_pose_in_arm_base(self):
        return pose_from_tf_matrix(
            np.linalg.inv(self._arm_base_transform())
            @ tf_matrix_from_pose(*self._object_world_pose())
        )

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

    @staticmethod
    def _plan_mask(result):
        if isinstance(result, BatchPlanResult):
            return np.asarray(result.success, dtype=bool)
        if not isinstance(result, PlanResult):
            raise TypeError(f"unsupported plan result: {type(result).__name__}")
        return np.asarray((result.success,), dtype=bool)

    @staticmethod
    def _plan_paths(result):
        if isinstance(result, BatchPlanResult):
            return list(result.trajectories)
        if isinstance(result, PlanResult):
            return [] if result.trajectory is None else [result.trajectory]
        raise TypeError(f"unsupported plan result: {type(result).__name__}")

    @staticmethod
    def _axis_from_config(runtime):
        axis = getattr(runtime, "grasp_approach_axis", None)
        if axis is None:
            axis = {"x": 0, "y": 1, "z": 2}[
                str(runtime.robot_port.robot.cfg["ee_axis"]).lower()
            ]
        return int(axis)

    def sample_ee_pose(self, max_length=None):
        """Apply all YAML grasp filters and return every ranked candidate."""

        poses = np.asarray(self.get_ee_poses("armbase"), dtype=float)
        count = len(poses)
        if count == 0:
            self._candidate_raw_indices = np.empty((0,), dtype=int)
            self.sampled_scores = np.empty((0,), dtype=float)
            return poses

        directions = {
            "x": {
                "forward": (0, 0, 1), "backward": (0, 0, -1),
                "upward": (2, 0, 1), "downward": (2, 0, -1),
            },
            "y": {
                "forward": (0, 1, 1), "backward": (0, 1, -1),
                "upward": (2, 1, 1), "downward": (2, 1, -1),
            },
            "z": {
                "forward": (0, 2, 1), "backward": (0, 2, -1),
                "upward": (2, 2, 1), "downward": (2, 2, -1),
            },
        }
        flags = np.ones(count, dtype=bool)
        for axis, conditions in directions.items():
            spec = self.skill_cfg.get(f"filter_{axis}_dir")
            if spec is None:
                continue
            row, col, sign = conditions[spec[0]]
            values = poses[:, row, col]
            limits = [np.cos(np.deg2rad(float(value))) for value in spec[1:]]
            if len(limits) == 1:
                current = values >= limits[0] if sign > 0 else values <= limits[0]
            elif sign > 0:
                current = (values >= limits[0]) & (values <= limits[1])
            else:
                current = (values <= limits[0]) & (values >= limits[1])
            flags &= current

        side = self.skill_cfg.get("grasp_side_preference")
        if side is not None:
            object_position, _ = self._object_pose_in_arm_base()
            object_xy = np.asarray(object_position[:2], dtype=float)
            norm = float(np.linalg.norm(object_xy))
            if norm > 1e-6:
                projection = (poses[:, :2, 3] - object_xy) @ (object_xy / norm)
                flags &= projection <= 0.0 if side == "toward_arm" else projection > 0.0

        candidates = np.flatnonzero(flags)
        candidates = candidates[np.argsort(np.asarray(self.scores)[candidates])]
        if max_length is not None:
            candidates = candidates[: max(1, int(max_length))]
        self._candidate_raw_indices = np.asarray(candidates, dtype=int)
        self.sampled_scores = np.asarray(self.scores[candidates], dtype=float)
        return poses[candidates]

    def get_ee_poses(self, frame="world"):
        if frame not in {"world", "body", "armbase"}:
            raise ValueError("frame must be world, body, or armbase")
        if frame == "body":
            return np.asarray(self.T_obj_ee, dtype=float)
        object_translation, object_orientation = self._object_world_pose()
        world_object = tf_matrix_from_pose(object_translation, object_orientation)
        world_ee = world_object[None] @ np.asarray(self.T_obj_ee, dtype=float)
        if frame == "world":
            return world_ee
        arm_base = self._arm_base_transform()
        return np.linalg.inv(arm_base)[None] @ world_ee

    def _select_grasp_index(self, positions, orientations, transforms, valid_indices):
        valid = np.asarray(valid_indices, dtype=int).reshape(-1)
        if len(valid) == 0:
            return None
        object_position, _ = self._object_pose_in_arm_base()
        target_z = float(self.skill_cfg.get("target_grasp_z", 0.12))
        target_orientation = self.skill_cfg.get("target_grasp_orientation")
        if target_orientation is not None:
            target_orientation = np.asarray(target_orientation, dtype=float)
            target_orientation /= max(float(np.linalg.norm(target_orientation)), 1e-12)
        costs = []
        for index in valid:
            relative = np.asarray(positions[index]) - np.asarray(object_position)
            approach = np.asarray(transforms[index])[:3, self._axis_from_config(self.skill_runtime)]
            orientation_cost = 0.0
            if target_orientation is not None:
                orientation_cost = 1.0 - abs(float(np.dot(orientations[index], target_orientation)))
            costs.append(
                (
                    abs(float(relative[2]) - target_z)
                    + float(np.linalg.norm(relative[:2]))
                    + max(0.0, float(approach[2]) + 0.75)
                    + orientation_cost,
                    float(self.sampled_scores[index]) if index < len(self.sampled_scores) else 0.0,
                    int(index),
                )
            )
        costs.sort()
        return costs[0][2]

    def _plan_candidates(self, transforms):
        runtime = self.skill_runtime
        object_name = self.pick_obj.name
        transforms = np.asarray(transforms, dtype=float)
        count = len(transforms)
        if count == 0:
            return {"feasible": False, "failure_code": "NO_GRASP_CANDIDATE"}

        pregrasps = transforms.copy()
        pregrasps[:, :3, 3] -= (
            pregrasps[:, :3, self._axis_from_config(runtime)]
            * float(self.skill_cfg.get("pre_grasp_offset", 0.1))
        )
        pre_positions, pre_orientations = poses_from_tf_matrices(pregrasps)
        positions, orientations = poses_from_tf_matrices(transforms)
        if self.fixed_orientation is not None:
            pre_orientations[:] = self.fixed_orientation
            orientations[:] = self.fixed_orientation

        pre_paths = [None] * count
        terminal_paths = [None] * count
        pre_mask = np.zeros(count, dtype=bool)
        terminal_mask = np.zeros(count, dtype=bool)
        terminal_hold_mask = np.zeros(count, dtype=bool)
        same_pose = np.array_equal(pre_positions, positions) and np.array_equal(
            pre_orientations, orientations
        )
        runtime.transition_target(object_name, collision_policy=CollisionPolicy.WORLD_TRANSIT)
        for start in range(0, count, CUROBO_BATCH_SIZE):
            stop = min(start + CUROBO_BATCH_SIZE, count)
            result = runtime.plan_pose_batch(
                pre_positions[start:stop], pre_orientations[start:stop],
                collision_policy=CollisionPolicy.WORLD_TRANSIT,
                active_target=object_name,
                phase_id="pick_pregrasp_batch",
            )
            mask = self._plan_mask(result)
            paths = self._plan_paths(result)
            pre_mask[start:stop] = mask[: stop - start]
            pre_paths[start:stop] = paths[: stop - start]

        if same_pose:
            terminal_mask[:] = pre_mask
            terminal_hold_mask[:] = pre_mask
        else:
            valid_pre = np.flatnonzero(pre_mask & np.asarray([p is not None for p in pre_paths]))
            runtime.transition_target(object_name, collision_policy=CollisionPolicy.TARGET_APPROACH)
            for start in range(0, len(valid_pre), CUROBO_BATCH_SIZE):
                indices = valid_pre[start:start + CUROBO_BATCH_SIZE]
                result = runtime.plan_pose_batch(
                    positions[indices], orientations[indices],
                    start_paths=[pre_paths[index] for index in indices],
                    collision_policy=CollisionPolicy.TARGET_APPROACH,
                    active_target=object_name,
                    phase_id="pick_grasp_batch",
                )
                mask = self._plan_mask(result)
                paths = self._plan_paths(result)
                for local, index in enumerate(indices):
                    terminal_mask[index] = bool(local < len(mask) and mask[local])
                    terminal_paths[index] = paths[local] if local < len(paths) else None

        pre_path_mask = np.asarray(
            [path is not None for path in pre_paths], dtype=bool
        )
        terminal_path_mask = np.asarray(
            [path is not None for path in terminal_paths], dtype=bool
        )

        # A typed planner success is usable only when it also produced a
        # trajectory.  This keeps V1's candidate intersection semantics while
        # protecting the native-v2 boundary from success-without-path results.
        pre_mask &= pre_path_mask
        terminal_mask &= terminal_path_mask | terminal_hold_mask
        valid = np.flatnonzero(pre_mask & terminal_mask).astype(int)
        if os.environ.get("SIMBOX_DEBUG_PICK") == "1":
            LOGGER.warning(
                "[PickDebug] plan_masks count=%d pre=%s terminal=%s "
                "pre_paths=%s terminal_paths=%s hold=%s valid=%s valid_raw=%s",
                count,
                np.flatnonzero(pre_mask).astype(int).tolist(),
                np.flatnonzero(terminal_mask).astype(int).tolist(),
                np.flatnonzero(pre_path_mask).astype(int).tolist(),
                np.flatnonzero(terminal_path_mask).astype(int).tolist(),
                np.flatnonzero(terminal_hold_mask).astype(int).tolist(),
                valid.tolist(),
                [int(self._candidate_raw_indices[index]) for index in valid],
            )
        selected = self._select_grasp_index(
            positions, orientations, transforms, valid
        )
        if selected is None and len(valid):
            selected = int(valid[0])
        result = {
            "feasible": selected is not None,
            "arm": self.lr_arm,
            "grasp_count": count,
            "pregrasp_success_count": int(np.count_nonzero(pre_mask)),
            "grasp_success_count": int(np.count_nonzero(terminal_mask)),
            "joint_success_count": int(len(valid)),
            "selected_grasp_index": selected,
            "selected_grasp_score": None if selected is None else float(self.sampled_scores[selected]),
            "failure_code": None if selected is not None else "NO_JOINT_GRASP_PLAN",
        }
        self.plan_state = {
            "result": result,
            "pregrasp_positions": pre_positions,
            "pregrasp_orientations": pre_orientations,
            "grasp_positions": positions,
            "grasp_orientations": orientations,
            "pregrasp_path": pre_paths[selected] if selected is not None and selected < len(pre_paths) else None,
            "terminal_path": terminal_paths[selected] if selected is not None and selected < len(terminal_paths) else None,
            "terminal_hold": bool(selected is not None and terminal_hold_mask[selected]),
        }
        return self.plan_state

    def _build_commands(self, state, index, post_offset):
        object_name = self.pick_obj.name
        pre_position = state["pregrasp_positions"][index]
        pre_orientation = state["pregrasp_orientations"][index]
        grasp_position = state["grasp_positions"][index]
        grasp_orientation = state["grasp_orientations"][index]
        tolerance = {
            "position_m": float(self.skill_cfg.get("t_eps", 0.005)),
            "orientation_rad": float(self.skill_cfg.get("o_eps", 0.05)),
        }
        terminal_step = float(
            self.task.cfg.get("planning", {})
            .get("pick_place", {})
            .get("terminal_step_m", 0.005)
        )
        if not np.isfinite(terminal_step) or terminal_step <= 0.0:
            raise ValueError("planning.pick_place.terminal_step_m must be positive")
        terminal_tolerance = dict(tolerance)
        terminal_tolerance["position_m"] = terminal_step
        commands = [
            MotionPhaseCommand(
                MotionPhase.SYNC_WORLD,
                active_object=object_name,
                replan_allowed=False,
            ),
            MotionPhaseCommand(
                MotionPhase.TRANSIT_PREGRASP,
                pre_position,
                pre_orientation,
                gripper_action="open_gripper",
                active_object=object_name,
                completion_tolerance=tolerance,
                preplanned_joint_path=state["pregrasp_path"],
            ),
        ]
        terminal_params = {"cartesian_step_m": terminal_step}
        terminal_path = None
        if state.get("terminal_hold", False):
            terminal_params["hold_position"] = True
        else:
            terminal_path = state["terminal_path"]
        commands.append(MotionPhaseCommand(
                MotionPhase.TERMINAL_GRASP_APPROACH,
                grasp_position,
                grasp_orientation,
                gripper_action="open_gripper",
                active_object=object_name,
                allow_target_finger_contact=True,
                completion_tolerance=terminal_tolerance,
                params=terminal_params,
                preplanned_joint_path=terminal_path,
            ))
        commands.extend([
            MotionPhaseCommand(
                MotionPhase.GRIPPER_CLOSE,
                grasp_position,
                grasp_orientation,
                gripper_action=self.gripper_cmd,
                active_object=object_name,
                allow_target_finger_contact=True,
                replan_allowed=False,
                dwell_steps=int(self.skill_cfg.get("gripper_change_steps", 40)),
                params={"contact_threshold_n": float(self.skill_cfg.get("grasp_contact_threshold_n", 0.0))},
            ),
            MotionPhaseCommand(
                MotionPhase.ATTACH,
                active_object=object_name,
                allow_target_finger_contact=True,
                replan_allowed=False,
                params={"verify_grasp_contact": lambda: self._grasp_contact_verified},
            ),
        ])
        support = self.skill_runtime.source_support(object_name)
        if post_offset:
            lift_position = np.asarray(grasp_position, dtype=float).copy()
            lift_position[2] += float(post_offset)
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.POST_GRASP_LIFT,
                    lift_position,
                    grasp_orientation,
                    gripper_action=self.gripper_cmd,
                    active_object=object_name,
                    support_object=support,
                    allow_target_finger_contact=True,
                    allow_target_robot_contact=True,
                    allow_object_support_contact=True,
                    completion_tolerance=tolerance,
                )
            )
        if self.skill_cfg.get("return_to_pregrasp", False):
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.POST_GRASP_LIFT,
                    pre_position,
                    pre_orientation,
                    gripper_action=self.gripper_cmd,
                    active_object=object_name,
                    support_object=support,
                    allow_target_finger_contact=True,
                    allow_target_robot_contact=True,
                    allow_object_support_contact=True,
                    completion_tolerance=tolerance,
                )
            )
        self.record_selected_trajectory(
            state.get("pregrasp_path"), "transit_pregrasp"
        )
        self.record_selected_trajectory(
            terminal_path, "terminal_grasp_approach"
        )
        return commands

    def generate_manip_cmds(self):
        self.failure_reason = ""
        self._grasp_contact_verified = False
        runtime = self.skill_runtime
        transforms = self.sample_ee_pose(max_length=60)
        object_name = self.pick_obj.name
        # Keep candidate generation on the same Physics-schema world/metric
        # boundary as the original Pick implementation.  In particular, a
        # preceding Place or another skill may have left dynamic poses or
        # approach-only pose costs cached in the native planners.
        runtime.sync_dynamic_poses(force=True)
        runtime.reset_pose_cost_metric()
        runtime.transition_target(object_name, collision_policy=CollisionPolicy.WORLD_TRANSIT)
        try:
            state = self._plan_candidates(transforms)
        finally:
            runtime.restore_world(object_name)
        result = state["result"] if "result" in state else state
        if not result["feasible"]:
            self.failure_reason = result.get("failure_code", "NO_JOINT_GRASP_PLAN")
            self.manip_list = []
            self.publish_target_intent(
                {"kind": "pick", "objects": [object_name], "has_target": False, "failure_reason": self.failure_reason}
            )
            return

        index = int(result["selected_grasp_index"])
        if os.environ.get("SIMBOX_DEBUG_PICK") == "1":
            object_translation, object_orientation = self._object_world_pose()
            raw_index = (
                int(self._candidate_raw_indices[index])
                if index < len(self._candidate_raw_indices)
                else None
            )
            selected_transform = None
            approach_vector = None
            if raw_index is not None:
                all_transforms = np.asarray(self.get_ee_poses("armbase"), dtype=float)
                if 0 <= raw_index < len(all_transforms):
                    selected_transform = all_transforms[raw_index]
                    approach_vector = selected_transform[:3, self._axis_from_config(self.skill_runtime)]
            LOGGER.warning(
                "[PickDebug] object=%s usd=%s arm=%s approach_axis=%s "
                "candidate_index=%s raw_index=%s score=%s object_world=%s "
                "object_orientation=%s arm_base=%s pregrasp=%s grasp=%s "
                "grasp_orientation=%s approach_vector=%s proxy=%s",
                object_name,
                getattr(self.pick_obj, "usd_path", None),
                self.lr_arm,
                self._axis_from_config(self.skill_runtime),
                index,
                raw_index,
                result["selected_grasp_score"],
                np.asarray(object_translation, dtype=float).tolist(),
                np.asarray(object_orientation, dtype=float).tolist(),
                np.asarray(self._arm_base_transform(), dtype=float).tolist(),
                np.asarray(state["pregrasp_positions"][index], dtype=float).tolist(),
                np.asarray(state["grasp_positions"][index], dtype=float).tolist(),
                np.asarray(state["grasp_orientations"][index], dtype=float).tolist(),
                None if approach_vector is None else np.asarray(approach_vector, dtype=float).tolist(),
                getattr(self.pick_obj, "_physics_collision_proxy_path", None),
            )
        post_offset = float(
            np.random.uniform(
                self.skill_cfg.get("post_grasp_offset_min", 0.05),
                self.skill_cfg.get("post_grasp_offset_max", 0.05),
            )
        )
        self.manip_list = self._build_commands(state, index, post_offset)
        self.publish_target_intent(
            {
                "kind": "pick",
                "objects": [object_name],
                "selected_index": index,
                "selected_score": result["selected_grasp_score"],
                "constraints": self._target_constraints(),
                "pregrasp_position": state["pregrasp_positions"][index],
                "grasp_position": state["grasp_positions"][index],
            }
        )

    def replan_after_safety(self, command, reason=None):
        del reason
        if isinstance(command, MotionPhaseCommand) and command.phase in {
            MotionPhase.TRANSIT_PREGRASP,
            MotionPhase.TERMINAL_GRASP_APPROACH,
        }:
            command.preplanned_joint_path = None
        return True

    def get_contact(self, contact_threshold=0.0):
        raw_values = self.pickcontact_view.get_contact_force_matrix()
        values = np.asarray(raw_values, dtype=float)
        if not values.size:
            force = np.empty((0,), dtype=float)
        else:
            force = np.atleast_1d(np.sum(np.abs(values), axis=-1).squeeze())

        return force, np.where(force > contact_threshold)[0]

    def is_feasible(self, th=5):
        return bool(getattr(self.skill_runtime, "num_plan_failed", 0) <= th and not self.failure_reason)

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        del t_eps, o_eps
        if not self.manip_list:
            return True
        command = self.manip_list[0]
        done = bool(self.skill_runtime.execution.is_phase_command_complete(command))
        if done and command.phase == MotionPhase.GRIPPER_CLOSE:
            threshold = float(command.params.get("contact_threshold_n", 0.0))
            _, contacts = self.get_contact(threshold)
            self._grasp_contact_verified = len(contacts) > 0
            if not self._grasp_contact_verified:
                self.failure_reason = "GRASP_CONTACT_MISSING"
                self.skill_runtime.restore_world(self.pick_obj.name)
                self.manip_list.clear()
        return done

    def is_done(self):
        if not self.manip_list:
            return True
        if self.is_subtask_done(
            t_eps=self.skill_cfg.get("t_eps", 1e-3),
            o_eps=self.skill_cfg.get("o_eps", 5e-3),
        ):
            if self.manip_list:
                self.manip_list.pop(0)
        return not self.manip_list

    def update(self):
        return None

    def is_success(self):
        _, contacts = self.get_contact()
        contact_ok = self.gripper_cmd != "close_gripper" or len(contacts) > 0
        velocities = self.robot.get_joints_state().velocities
        object_velocity = self.pick_obj.get_linear_velocity()
        self.process_valid = (
            not self.skill_cfg.get("process_valid", True)
            or max(float(np.max(np.abs(velocities))), float(np.max(np.abs(object_velocity)))) < 5.0
        )
        lift_threshold = float(self.skill_cfg.get("lift_th", 0.0))
        current_z = float(self._object_world_pose()[0][2])
        lift_ok = lift_threshold <= 0.0 or current_z - float(self.obj_init_trans[2]) > lift_threshold
        return bool(contact_ok and self.process_valid and lift_ok and not self.failure_reason)
