"""Candidate-driven Place skill.

Place owns only placement-target generation and the release/settle protocol.
All collision-aware motion queries go through the controller-owned typed runtime; the
controller owns the CuRobo/native execution details.
"""

import json
import os

import numpy as np
from core.planning.domain_types import BatchPlanResult, CollisionPolicy, PlanResult
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.box import Box, get_bbox_center_and_corners
from core.utils.iou import IoU
from core.utils.plan_utils import sort_by_difference_js
from core.utils.transformation_utils import create_pose_matrices, poses_from_tf_matrices
from core.utils.usd_geom_utils import compute_bbox
from core.visualization.skill_target_math import ratio_box_corners
from core.utils.constants import CUROBO_BATCH_SIZE
from isaacsim.core.api.robots.robot import Robot
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
from isaacsim.core.utils.xforms import get_world_pose
from omegaconf import DictConfig, ListConfig, OmegaConf
from pxr import Usd
from scipy.spatial.transform import Rotation as R


class PlaceCandidateError(RuntimeError):
    """Raised when none of the generated placement targets can be planned."""


@register_skill
class Place(BaseSkill):
    """Generate placement candidates, query CuRobo, and emit Place phases."""

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

        self.name = cfg["name"]
        self.pick_obj = task._task_objects[cfg["objects"][0]]
        self.place_obj = task._task_objects[cfg["objects"][1]]
        self.place_part_prim_path = cfg.get("place_part_prim_path")
        self.place_prim_path = (
            f"{self.place_obj.prim_path}/{self.place_part_prim_path}"
            if self.place_part_prim_path
            else self.place_obj.prim_path
        )
        self.place_align_axis = cfg.get("place_align_axis")
        self.pick_align_axis = cfg.get("pick_align_axis")
        self.constraint_gripper_x = cfg.get("constraint_gripper_x", False)
        self.align_pick_obj_axis = cfg.get("align_pick_obj_axis")
        self.align_place_obj_axis = cfg.get("align_place_obj_axis")
        self.align_obj_tol = cfg.get("align_obj_tol")
        self.robot_ee_path = self.skill_runtime.setup.robot_ee_path
        self.robot_base_path = self.skill_runtime.setup.reference_prim_path

        self.manip_list = []
        self.place_ee_trans = None
        self.failure_reason = ""
        self.error_message = ""
        self._selected_plan = {}
        self._target_intent = None
        self._success_snapshot_written = False
        output_root = str(cfg.get("output_root", "output/local_navigation/skills"))
        self._debug_dir = os.path.join(
            output_root,
            f"{robot.name}_place_{self.pick_obj.name}_to_"
            f"{self.place_obj.name}",
        )

    @property
    def _pick_place_cfg(self):
        return self.task.cfg.get("planning", {}).get("pick_place", {})

    @staticmethod
    def _plan_mask(result):
        if isinstance(result, BatchPlanResult):
            return np.asarray(result.success, dtype=bool)
        if not isinstance(result, PlanResult):
            raise TypeError(f"unsupported plan result: {type(result).__name__}")
        return np.asarray([bool(result.success)], dtype=bool)

    @staticmethod
    def _plan_paths(result):
        if isinstance(result, BatchPlanResult):
            return list(result.trajectories)
        if isinstance(result, PlanResult):
            return [] if result.trajectory is None else [result.trajectory]
        raise TypeError(f"unsupported plan result: {type(result).__name__}")

    @staticmethod
    def _plan_metric(result, name):
        values = result.metrics.get(name)
        if values is None:
            return np.full(len(result.success), np.inf, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.ndim == 0:
            return np.full(len(result.success), float(values), dtype=float)
        return values.reshape(values.shape[0], -1).min(axis=1)

    @staticmethod
    def _select_priority_index(valid, position_error, rotation_error, paths):
        indices = np.flatnonzero(valid)
        path_indices = np.asarray(
            [index for index in indices if paths[index] is not None], dtype=int
        )
        if len(path_indices) == 0:
            return int(indices[0])
        position = position_error[path_indices]
        rotation = rotation_error[path_indices]
        filtered = path_indices[
            (position <= np.mean(position)) & (rotation <= np.mean(rotation))
        ]
        if len(filtered) == 0:
            filtered = path_indices
        order = sort_by_difference_js([paths[index] for index in filtered])
        return int(filtered[int(order[0])])

    @staticmethod
    def _json_ready(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, (DictConfig, ListConfig)):
            return Place._json_ready(OmegaConf.to_container(value, resolve=True))
        if type(value).__module__.startswith("pxr") and type(value).__name__.startswith("Vec"):
            return [Place._json_ready(item) for item in value]
        if isinstance(value, (list, tuple)):
            return [Place._json_ready(item) for item in value]
        if isinstance(value, dict):
            return {str(key): Place._json_ready(item) for key, item in value.items()}
        return value

    def _record_success_failure(self, reasons, details):
        self.failure_reason = "place_success_check_failed:" + ",".join(reasons)
        self.error_message = "Place completed but the success check failed."
        if self._success_snapshot_written:
            return
        try:
            os.makedirs(self._debug_dir, exist_ok=True)
            path = os.path.join(self._debug_dir, "place_success_check_snapshot.json")
            payload = {
                "robot": self.robot.name,
                "skill": self.name,
                "pick_object": self.pick_obj.name,
                "place_object": self.place_obj.name,
                "place_prim_path": self.place_prim_path,
                "success_mode": self.skill_cfg.get("success_mode", "3diou"),
                "success": False,
                "failure_reasons": reasons,
                "details": details,
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self._json_ready(payload), handle, indent=2, ensure_ascii=False)
        except Exception:  # Diagnostics must never change task behavior.
            pass
        finally:
            self._success_snapshot_written = True

    @staticmethod
    def _plan_summary(result):
        if result is None:
            return None
        if not isinstance(result, (PlanResult, BatchPlanResult)):
            raise TypeError(f"unsupported plan result: {type(result).__name__}")
        success = result.success if isinstance(result, BatchPlanResult) else (result.success,)
        return {
            "type": type(result).__name__,
            "status": str(getattr(result, "status", "")),
            "error": None if getattr(result, "error", None) is None else str(result.error),
            "success_count": int(getattr(result, "success_count", sum(success))),
            "candidate_count": len(success),
            "metrics": getattr(result, "metrics", {}),
        }

    def _write_plan_snapshot(self, payload):
        """Write candidate accounting; diagnostics must never affect planning."""

        try:
            os.makedirs(self._debug_dir, exist_ok=True)
            path = os.path.join(self._debug_dir, "place_plan_snapshot.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self._json_ready(payload), handle, indent=2, ensure_ascii=False)
        except Exception:
            pass

    @staticmethod
    def _world_pose(obj):
        getter = getattr(obj, "get_world_pose", None)
        return getter() if callable(getter) else obj.get_local_pose()

    def _object_world_tf(self, obj):
        return tf_matrix_from_pose(*self._world_pose(obj))

    def _arm_base_world_tf(self):
        try:
            pose = self.skill_runtime.execution.get_armbase_pose()
            if isinstance(pose, (tuple, list)) and len(pose) == 2:
                return tf_matrix_from_pose(*pose)
            matrix = np.asarray(pose, dtype=float)
            if matrix.shape == (4, 4):
                return matrix
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        return tf_matrix_from_pose(*get_world_pose(self.robot_base_path))

    def _candidate_geometry(self):
        self.T_world_obj = self._object_world_tf(self.pick_obj)
        self.T_world_ee = tf_matrix_from_pose(*get_world_pose(self.robot_ee_path))
        self.T_base_world = np.linalg.inv(self._arm_base_world_tf())
        self.T_obj_ee = np.linalg.inv(self.T_world_obj) @ self.T_world_ee
        self.T_world_container = self._object_world_tf(self.place_obj)

        bbox = compute_bbox(get_prim_at_path(self.place_prim_path))
        b_min = np.asarray(bbox.min, dtype=float)
        b_max = np.asarray(bbox.max, dtype=float)
        defaults = (("x_ratio_range", (0.4, 0.6)), ("y_ratio_range", (0.4, 0.6)), ("z_ratio_range", (0.4, 0.6)))
        target_rotations = self.generate_constrained_rotations()[:60]
        candidate_count = len(target_rotations)

        def ratios(key, default):
            bounds = self.skill_cfg.get(key, default)
            return np.random.uniform(float(bounds[0]), float(bounds[1]), candidate_count)

        x_ratio = ratios("x_ratio_range", (0.4, 0.6))
        y_ratio = ratios("y_ratio_range", (0.4, 0.6))
        z_ratio = ratios("z_ratio_range", (0.4, 0.6))
        direction = self.skill_cfg.get("place_direction", "vertical")
        constraint = self.skill_cfg.get("position_constraint", "gripper")
        pre_offset = float(self.skill_cfg.get("pre_place_z_offset", 0.2))
        place_offset = float(self.skill_cfg.get("place_z_offset", 0.1))

        if direction == "vertical":
            x = b_min[0] + x_ratio * (b_max[0] - b_min[0])
            y = b_min[1] + y_ratio * (b_max[1] - b_min[1])
            pre_world = np.column_stack((x, y, np.full(candidate_count, b_max[2] + pre_offset)))
            place_world = np.column_stack((x, y, np.full(candidate_count, b_max[2] + place_offset)))
            region = ratio_box_corners(
                b_min, b_max,
                tuple(tuple(float(value) for value in self.skill_cfg.get(key, default)) for key, default in defaults),
            )
            region[:, 2] = b_max[2] + place_offset
            normal = np.array([0.0, 0.0, 1.0])
            tangent = np.array([1.0, 0.0, 0.0])
        elif direction == "horizontal":
            align = self.T_world_container[:3, :3] @ np.asarray(self.skill_cfg["align_place_obj_axis"], dtype=float)
            offset = self.T_world_container[:3, :3] @ np.asarray(self.skill_cfg["offset_place_obj_axis"], dtype=float)
            points = b_min + np.column_stack((x_ratio, y_ratio, z_ratio)) * (b_max - b_min)
            if constraint == "object":
                pre_world = points + align * float(self.skill_cfg.get("pre_place_align", 0.2)) + offset * float(self.skill_cfg.get("pre_place_offset", 0.2))
                place_world = points + align * float(self.skill_cfg.get("place_align", 0.1)) + offset * float(self.skill_cfg.get("place_offset", 0.1))
                region = ratio_box_corners(b_min, b_max, tuple(tuple(float(value) for value in self.skill_cfg.get(key, default)) for key, default in defaults))
                region += align * float(self.skill_cfg.get("place_align", 0.1)) + offset * float(self.skill_cfg.get("place_offset", 0.1))
            else:
                distance = float(np.linalg.norm(self.T_obj_ee[:3, 3]))
                pre_world = points - align * (pre_offset + distance)
                place_world = points - align * (place_offset + distance)
                region = ratio_box_corners(b_min, b_max, tuple(tuple(float(value) for value in self.skill_cfg.get(key, default)) for key, default in defaults))
                region -= align * (place_offset + distance)
            normal, tangent = np.asarray(align), np.asarray(offset)
        else:
            raise NotImplementedError(f"unsupported place_direction: {direction}")

        base_rotation = self.T_base_world[:3, :3]
        pre_base = (base_rotation @ pre_world.T).T + self.T_base_world[:3, 3]
        place_base = (base_rotation @ place_world.T).T + self.T_base_world[:3, 3]
        if constraint == "object":
            ee_from_object = np.linalg.inv(self.T_obj_ee)[:3, :3]
            object_rotations = target_rotations @ ee_from_object
            pre_tf = create_pose_matrices(pre_base, object_rotations) @ self.T_obj_ee
            place_tf = create_pose_matrices(place_base, object_rotations) @ self.T_obj_ee
        elif constraint == "gripper":
            pre_tf = create_pose_matrices(pre_base, target_rotations)
            place_tf = create_pose_matrices(place_base, target_rotations)
        else:
            raise NotImplementedError(f"unsupported position_constraint: {constraint}")

        pre_positions, pre_orientations = poses_from_tf_matrices(pre_tf)
        place_positions, place_orientations = poses_from_tf_matrices(place_tf)
        return {
            "pre_positions": pre_positions,
            "pre_orientations": pre_orientations,
            "place_positions": place_positions,
            "place_orientations": place_orientations,
            "pre_world": pre_world,
            "place_world": place_world,
            "bbox_min": b_min,
            "bbox_max": b_max,
            "region": region,
            "normal": normal,
            "tangent": tangent,
            "direction": direction,
            "constraint": constraint,
        }

    def _plan_candidates(self, geometry):
        """Evaluate every target in native batches and intersect both phases."""

        runtime = self.skill_runtime
        object_name, support_name = self.pick_obj.name, self.place_obj.name
        pre = np.asarray(geometry["pre_positions"])
        place = np.asarray(geometry["place_positions"])
        pre_q = np.asarray(geometry["pre_orientations"])
        place_q = np.asarray(geometry["place_orientations"])
        count = len(place)
        pre_paths, terminal_paths = [None] * count, [None] * count
        pre_ok = np.zeros(count, dtype=bool)
        terminal_ok = np.zeros(count, dtype=bool)
        terminal_hold = np.zeros(count, dtype=bool)
        pre_position_error = np.full(count, np.inf, dtype=float)
        pre_rotation_error = np.full(count, np.inf, dtype=float)
        terminal_position_error = np.full(count, np.inf, dtype=float)
        terminal_rotation_error = np.full(count, np.inf, dtype=float)
        same_target = np.array_equal(pre, place) and np.array_equal(pre_q, place_q)
        pre_result = terminal_result = None
        runtime.transition_target(
            object_name, support_name, collision_policy=CollisionPolicy.ATTACHED_CARRY
        )
        for start in range(0, count, CUROBO_BATCH_SIZE):
            stop = min(start + CUROBO_BATCH_SIZE, count)
            pre_result = runtime.plan_pose_batch(
                pre[start:stop], pre_q[start:stop],
                collision_policy=CollisionPolicy.ATTACHED_CARRY,
                active_target=object_name,
                support=support_name,
                phase_id="place_preplace_batch",
            )
            mask = self._plan_mask(pre_result)
            paths = self._plan_paths(pre_result)
            pre_ok[start:stop] = mask[:stop - start]
            pre_paths[start:stop] = paths[:stop - start]
            pre_position_error[start:stop] = self._plan_metric(pre_result, "position_error")
            pre_rotation_error[start:stop] = self._plan_metric(pre_result, "rotation_error")

        if same_target:
            terminal_ok[:] = pre_ok
            terminal_hold[:] = pre_ok
            terminal_position_error[:] = pre_position_error
            terminal_rotation_error[:] = pre_rotation_error
        else:
            valid_pre = np.flatnonzero(
                pre_ok & np.asarray([path is not None for path in pre_paths])
            )
            runtime.transition_target(
                object_name, support_name,
                collision_policy=CollisionPolicy.PLACEMENT_DESCENT,
            )
            for start in range(0, len(valid_pre), CUROBO_BATCH_SIZE):
                indices = valid_pre[start:start + CUROBO_BATCH_SIZE]
                terminal_result = runtime.plan_pose_batch(
                    place[indices], place_q[indices],
                    start_paths=[pre_paths[index] for index in indices],
                    collision_policy=CollisionPolicy.PLACEMENT_DESCENT,
                    active_target=object_name,
                    support=support_name,
                    phase_id="place_terminal_batch",
                )
                mask = self._plan_mask(terminal_result)
                paths = self._plan_paths(terminal_result)
                position_error = self._plan_metric(terminal_result, "position_error")
                rotation_error = self._plan_metric(terminal_result, "rotation_error")
                for local, index in enumerate(indices):
                    terminal_ok[index] = bool(local < len(mask) and mask[local])
                    terminal_paths[index] = paths[local] if local < len(paths) else None
                    terminal_position_error[index] = position_error[local]
                    terminal_rotation_error[index] = rotation_error[local]

        # Candidate validation may have entered PLACEMENT_CONTACT. Leave the
        # scene attached for execution; RESTORE_WORLD is emitted only after
        # detach/settle.
        runtime.transition_target(
            object_name,
            support_name,
            collision_policy=CollisionPolicy.ATTACHED_CARRY,
        )
        # A candidate is eligible only when both batch phases produced a
        # successful result and a replayable trajectory.
        valid = pre_ok & terminal_ok & (
            np.asarray([path is not None for path in terminal_paths], dtype=bool)
            | terminal_hold
        )
        if not np.any(valid):
            self._write_plan_snapshot(
                {
                    "robot": self.robot.name,
                    "skill": self.name,
                    "pick_object": object_name,
                    "place_object": support_name,
                    "candidate_count": int(count),
                    "preplace_success_count": int(np.count_nonzero(pre_ok)),
                    "preplace_path_count": int(sum(path is not None for path in pre_paths)),
                    "terminal_success_count": int(np.count_nonzero(terminal_ok)),
                    "terminal_path_count": int(sum(path is not None for path in terminal_paths)),
                    "same_target": bool(same_target),
                    "terminal_hold_count": int(np.count_nonzero(terminal_hold)),
                    "geometry": {
                        "pre_world": geometry["pre_world"],
                        "place_world": geometry["place_world"],
                        "bbox_min": geometry["bbox_min"],
                        "bbox_max": geometry["bbox_max"],
                        "T_world_obj": self.T_world_obj,
                        "T_world_ee": self.T_world_ee,
                        "T_base_world": self.T_base_world,
                        "T_obj_ee": self.T_obj_ee,
                        "T_world_container": self.T_world_container,
                    },
                    "pre_result": self._plan_summary(pre_result),
                    "terminal_result": self._plan_summary(terminal_result),
                }
            )
            raise PlaceCandidateError("NO_COLLISION_FREE_PLACE_PLAN")

        index = self._select_priority_index(
            valid,
            terminal_position_error,
            terminal_rotation_error,
            terminal_paths,
        )
        self._selected_plan = {
            "candidate_index": index,
            "preplace_path": pre_paths[index],
            "terminal_path": terminal_paths[index],
            "terminal_hold": bool(terminal_hold[index]),
        }
        return index

    def sample_gripper_place_traj(self):
        geometry = self._candidate_geometry()
        index = self._plan_candidates(geometry)
        pre = [geometry["pre_positions"][index], geometry["pre_orientations"][index]]
        place = [geometry["place_positions"][index], geometry["place_orientations"][index]]
        self.place_ee_trans = np.asarray(place[0], dtype=float)
        self._target_intent = {
            "kind": "place",
            "objects": list(self.skill_cfg["objects"]),
            "selected_index": index,
            "place_direction": str(geometry["direction"]),
            "position_constraint": str(geometry["constraint"]),
            "bbox_world": np.concatenate((geometry["bbox_min"], geometry["bbox_max"])),
            "region_points_world": geometry["region"],
            "region_normal_world": geometry["normal"],
            "region_tangent_hint_world": geometry["tangent"],
            "selected_reference_world": geometry["place_world"][index],
            "preplace_position": pre[0],
            "preplace_orientation": pre[1],
            "place_position": place[0],
            "place_orientation": place[1],
        }
        result = [pre, place]
        if self.skill_cfg.get("post_place_vector"):
            post_tf = tf_matrix_from_pose(*place)
            post_tf[:3, 3] += post_tf[:3, :3] @ np.asarray(self.skill_cfg["post_place_vector"], dtype=float)
            result.append(list(pose_from_tf_matrix(post_tf)))
        return result

    def _build_commands(self, result):
        object_name, support_name = self.pick_obj.name, self.place_obj.name
        pre_position, pre_orientation = map(np.asarray, result[0])
        place_position, place_orientation = map(np.asarray, result[1])
        tolerance = {
            "position_m": float(self.skill_cfg.get("t_eps", 0.005)),
            "orientation_rad": float(self.skill_cfg.get("o_eps", 0.05)),
        }
        pre_path = self._selected_plan.get("preplace_path")
        terminal_params = {}
        terminal_path = None
        if self._selected_plan.get("terminal_hold", False):
            terminal_params["hold_position"] = True
        elif self._selected_plan.get("terminal_path") is not None:
            terminal_path = self._selected_plan["terminal_path"]
        commands = [
            MotionPhaseCommand(
                MotionPhase.TRANSIT_PREPLACE,
                pre_position,
                pre_orientation,
                gripper_action="close_gripper",
                active_object=object_name,
                support_object=support_name,
                allow_target_finger_contact=True,
                allow_target_robot_contact=True,
                completion_tolerance=tolerance,
                preplanned_joint_path=pre_path,
            )
        ]
        # The original Place emitted one ordinary terminal motion. The
        # Physics-schema phase enters PLACEMENT_CONTACT before planning, so
        # the same motion/release sequence remains collision-auditable without
        # the later continuous-descent segmentation and validator.
        commands.append(
            MotionPhaseCommand(
                MotionPhase.TERMINAL_PLACE_DESCENT,
                place_position,
                place_orientation,
                gripper_action="close_gripper",
                active_object=object_name,
                support_object=support_name,
                allow_target_finger_contact=True,
                allow_object_support_contact=True,
                completion_tolerance=tolerance,
                params=terminal_params,
                preplanned_joint_path=terminal_path,
            )
        )
        hesitate_steps = int(self.skill_cfg.get("hesitate_steps", 0))
        if hesitate_steps > 0:
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.GRIPPER_CLOSE,
                    gripper_action="close_gripper",
                    active_object=object_name,
                    support_object=support_name,
                    allow_target_finger_contact=True,
                    allow_object_support_contact=True,
                    replan_allowed=False,
                    dwell_steps=hesitate_steps,
                )
            )
        commands.extend(
            [
                MotionPhaseCommand(
                    MotionPhase.GRIPPER_OPEN,
                    place_position,
                    place_orientation,
                    gripper_action="open_gripper",
                    active_object=object_name,
                    support_object=support_name,
                    allow_target_finger_contact=True,
                    allow_object_support_contact=True,
                    replan_allowed=False,
                    dwell_steps=int(self.skill_cfg.get("gripper_change_steps", 10)),
                ),
                MotionPhaseCommand(
                    MotionPhase.DETACH_AND_SETTLE,
                    active_object=object_name,
                    support_object=support_name,
                    allow_target_robot_contact=True,
                    allow_object_support_contact=True,
                    replan_allowed=False,
                    dwell_steps=int(self._pick_place_cfg.get("place_settle_steps", 10)),
                ),
            ]
        )
        if len(result) > 2:
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.TERMINAL_RETREAT,
                    np.asarray(result[2][0]),
                    np.asarray(result[2][1]),
                    gripper_action="open_gripper",
                    active_object=object_name,
                    support_object=support_name,
                    completion_tolerance=tolerance,
                )
            )
        commands.append(MotionPhaseCommand(MotionPhase.RESTORE_WORLD, active_object=object_name, support_object=support_name, replan_allowed=False))
        self.record_selected_trajectory(pre_path, "transit_preplace")
        self.record_selected_trajectory(terminal_path, "terminal_place_descent")
        return commands

    def generate_manip_cmds(self):
        self.failure_reason = ""
        object_name, support_name = self.pick_obj.name, self.place_obj.name
        self.skill_runtime.assert_attached_owner(object_name)
        runtime = self.skill_runtime
        runtime.sync_dynamic_poses(force=True)
        runtime.reset_pose_cost_metric()
        self.skill_runtime.transition_target(object_name, support_name, collision_policy=CollisionPolicy.ATTACHED_CARRY)
        try:
            result = self.sample_gripper_place_traj()
            self.manip_list = self._build_commands(result)
            self.publish_target_intent(self._target_intent)
        except PlaceCandidateError as exc:
            self.failure_reason = str(exc)
            self.manip_list = []
            self.publish_target_intent({"kind": "place", "objects": list(self.skill_cfg["objects"]), "has_target": False, "failure_reason": self.failure_reason})

    def generate_constrained_rotations(self, sample_count=3000):
        if self.skill_cfg.get("preserve_attached_orientation", False):
            rotation = (self.T_base_world @ self.T_world_ee)[:3, :3]
            return rotation[None]

        conditions = {
            "x": {"forward": (0, 0, 1), "backward": (0, 0, -1), "leftward": (1, 0, 1), "rightward": (1, 0, -1), "upward": (2, 0, 1), "downward": (2, 0, -1)},
            "y": {"forward": (0, 1, 1), "backward": (0, 1, -1), "leftward": (1, 1, 1), "rightward": (1, 1, -1), "upward": (2, 1, 1), "downward": (2, 1, -1)},
            "z": {"forward": (0, 2, 1), "backward": (0, 2, -1), "leftward": (1, 2, 1), "rightward": (1, 2, -1), "upward": (2, 2, 1), "downward": (2, 2, -1)},
        }
        rotations = R.random(sample_count).as_matrix()
        valid = np.ones(sample_count, dtype=bool)
        for axis, values in conditions.items():
            spec = self.skill_cfg.get(f"filter_{axis}_dir")
            if spec is None:
                continue
            row, column, sign = values[spec[0]]
            elements = rotations[:, row, column]
            limits = [np.cos(np.deg2rad(float(value))) for value in spec[1:]]
            if len(limits) == 1:
                valid &= elements >= limits[0] if sign > 0 else elements <= limits[0]
            elif sign > 0:
                valid &= (elements >= limits[0]) & (elements <= limits[1])
            else:
                valid &= (elements <= limits[0]) & (elements >= limits[1])

        if self.align_pick_obj_axis is not None and self.align_place_obj_axis is not None and self.align_obj_tol is not None:
            pick_axis = np.asarray(self.align_pick_obj_axis, dtype=float)
            place_axis = (self.T_base_world @ self.T_world_container)[:3, :3] @ np.asarray(self.align_place_obj_axis, dtype=float)
            object_rotations = rotations @ np.linalg.inv(self.T_obj_ee)[:3, :3]
            pick_vectors = np.einsum("ijk,k->ij", object_rotations, pick_axis)
            cosine = np.sum(pick_vectors * place_axis, axis=1) / np.maximum(np.linalg.norm(pick_vectors, axis=1) * np.linalg.norm(place_axis), 1e-12)
            valid &= np.arccos(np.clip(cosine, -1.0, 1.0)) < np.deg2rad(float(self.align_obj_tol))

        choices = rotations[valid]
        return choices

    def is_feasible(self, th=5):
        return bool(self.skill_runtime.execution.state.num_plan_failed <= th and not self.failure_reason)

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        del t_eps, o_eps
        if not self.manip_list:
            return True
        return bool(self.skill_runtime.execution.is_phase_command_complete(self.manip_list[0]))

    def is_done(self):
        if self.manip_list and self.is_subtask_done(
            t_eps=self.skill_cfg.get("t_eps", 1e-3),
            o_eps=self.skill_cfg.get("o_eps", 5e-3),
        ):
            self.manip_list.pop(0)
        return not self.manip_list

    def is_success(self, th=0.0):
        mode = self.skill_cfg.get("success_mode", "3diou")
        pick_bbox = compute_bbox(self.pick_obj.prim)
        place_bbox = compute_bbox(get_prim_at_path(self.place_prim_path))
        pick_pose = self._world_pose(self.pick_obj)[0]
        reasons, details = [], {}

        if mode == "3diou":
            value = IoU(Box(get_bbox_center_and_corners(pick_bbox)), Box(get_bbox_center_and_corners(place_bbox))).iou()
            success = bool(value > th)
            details = {"iou": float(value), "threshold": float(th)}
            if not success:
                reasons.append("iou_below_threshold")
        elif mode == "height":
            relative = get_relative_transform(get_prim_at_path(self.pick_obj.prim_path), get_prim_at_path(self.robot_base_path))[:3, 3]
            threshold = float(self.place_ee_trans[2] - 0.4)
            success = bool(relative[2] < threshold)
            pick_bbox_min = np.asarray(pick_bbox.min, dtype=float)
            pick_bbox_max = np.asarray(pick_bbox.max, dtype=float)
            place_bbox_min = np.asarray(place_bbox.min, dtype=float)
            place_bbox_max = np.asarray(place_bbox.max, dtype=float)
            details = {
                "z": float(relative[2]),
                "threshold": threshold,
                "relative_xyz": np.asarray(relative, dtype=float),
                "place_ee_trans": np.asarray(self.place_ee_trans, dtype=float),
                "pick_world_pose": self._world_pose(self.pick_obj),
                "robot_base_world_pose": get_world_pose(self.robot_base_path),
                "pick_bbox_min": pick_bbox_min,
                "pick_bbox_max": pick_bbox_max,
                "place_bbox_min": place_bbox_min,
                "place_bbox_max": place_bbox_max,
                "place_world_pose": self._world_pose(self.place_obj),
                "place_meshes": [
                    {
                        "path": str(prim.GetPath()),
                        "type": str(prim.GetTypeName()),
                        "collision_enabled": (
                            prim.GetAttribute("physics:collisionEnabled").Get()
                            if prim.GetAttribute("physics:collisionEnabled")
                            else None
                        ),
                        "approximation": (
                            prim.GetAttribute("physics:approximation").Get()
                            if prim.GetAttribute("physics:approximation")
                            else None
                        ),
                        "point_count": (
                            len(prim.GetAttribute("points").Get() or [])
                            if prim.GetAttribute("points")
                            else None
                        ),
                    }
                    for prim in Usd.PrimRange(
                        get_prim_at_path(self.place_prim_path)
                    )
                    if prim.GetTypeName() == "Mesh"
                ],
                "selected_reference_world": (
                    None
                    if not self._target_intent
                    else self._target_intent.get("selected_reference_world")
                ),
            }
            if not success:
                reasons.append("height_not_below_threshold")
        elif mode == "xybbox":
            margin = float(self.skill_cfg.get("success_xy_margin", 0.015))
            minimum = np.asarray(place_bbox.min[:2], dtype=float) + margin
            maximum = np.asarray(place_bbox.max[:2], dtype=float) - margin
            xy = np.asarray(pick_pose[:2], dtype=float)
            success = bool(np.all(xy > minimum) and np.all(xy < maximum))
            details = {"pick_xy": xy, "valid_xy_min": minimum, "valid_xy_max": maximum}
            if not success:
                reasons.append("pick_center_outside_place_bbox")
        elif mode in {"left", "right"}:
            threshold = float(self.skill_cfg.get("threshold", 0.03))
            x = float(pick_pose[0])
            limit = float(place_bbox.min[0] - threshold if mode == "left" else place_bbox.max[0] + threshold)
            success = x < limit if mode == "left" else x > limit
            details = {"pick_x": x, "limit": limit}
            if not success:
                reasons.append(f"x_not_{mode}_of_place_bbox")
        elif mode == "front":
            threshold = float(self.skill_cfg.get("threshold", 0.03))
            y = float(pick_pose[1])
            limit = float(place_bbox.max[1] + threshold)
            success = y > limit
            details = {
                "pick_y": y,
                "limit": limit,
                "threshold": threshold,
                "place_bbox_min": np.asarray(place_bbox.min, dtype=float),
                "place_bbox_max": np.asarray(place_bbox.max, dtype=float),
            }
            if not success:
                reasons.append("not_beyond_place_bbox_y")
        elif mode in {"flower", "cup"}:
            iou = IoU(Box(get_bbox_center_and_corners(pick_bbox)), Box(get_bbox_center_and_corners(place_bbox))).iou()
            threshold = float(self.skill_cfg.get("success_th", 0.0))
            center = bool(place_bbox.min[0] < pick_pose[0] < place_bbox.max[0] and place_bbox.min[1] < pick_pose[1] < place_bbox.max[1])
            if mode == "flower":
                success = center and iou > threshold
            else:
                success = float(pick_bbox.min[2]) > float(place_bbox.min[2]) + 0.05 and iou > threshold
            details = {"iou": float(iou), "threshold": threshold, "center_valid": center}
            if not success:
                reasons.append("placement_geometry_invalid")
        else:
            success = False
            reasons.append("unsupported_success_mode")
            details = {"success_mode": mode}

        if not success:
            self._record_success_failure(reasons, details)
        return bool(success)
