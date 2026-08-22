"""Candidate-driven Place skill.

Place owns only placement-target generation and the release/settle protocol.
All collision-aware motion queries go through ``SkillRuntimePort``; the
controller owns the CuRobo/native execution details.
"""

from copy import deepcopy
import json
import os

import numpy as np
from core.planning.domain_types import BatchPlanResult, CollisionPolicy, PlanResult
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.skills.base_skill import BaseSkill, register_skill
from core.utils.box import Box, get_bbox_center_and_corners
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.iou import IoU
from core.utils.transformation_utils import create_pose_matrices, poses_from_tf_matrices
from core.utils.usd_geom_utils import compute_bbox
from core.visualization.skill_target_math import ratio_box_corners
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
        self.robot_ee_path = self.skill_runtime.robot_ee_path
        self.robot_base_path = self.skill_runtime.reference_prim_path

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
    def _plan_mask(result, count):
        if isinstance(result, BatchPlanResult):
            values = result.success_mask
        elif isinstance(result, PlanResult):
            values = (result.success,)
        else:
            return np.zeros(int(count), dtype=bool)
        values = np.asarray(values, dtype=bool).reshape(-1)
        return values if len(values) == int(count) else np.zeros(int(count), dtype=bool)

    @staticmethod
    def _plan_paths(result):
        if isinstance(result, BatchPlanResult):
            return list(result.trajectories)
        if isinstance(result, PlanResult):
            return [] if result.trajectory is None else [result.trajectory]
        return []

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
    def _world_pose(obj):
        getter = getattr(obj, "get_world_pose", None)
        return getter() if callable(getter) else obj.get_local_pose()

    def _object_world_tf(self, obj):
        return tf_matrix_from_pose(*self._world_pose(obj))

    def _arm_base_world_tf(self):
        try:
            pose = self.skill_runtime.arm_base_pose()
            if isinstance(pose, (tuple, list)) and len(pose) == 2:
                return tf_matrix_from_pose(*pose)
            matrix = np.asarray(pose, dtype=float)
            if matrix.shape == (4, 4):
                return matrix
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        return tf_matrix_from_pose(*get_world_pose(self.robot_base_path))

    def _terminal_options(self):
        cfg = self._pick_place_cfg
        step = float(cfg.get("place_terminal_step_m", cfg.get("terminal_step_m", 0.01)))
        tolerance = float(cfg.get("place_terminal_tolerance_m", min(0.005, step)))
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("Place terminal step must be positive and finite")
        if not np.isfinite(tolerance) or tolerance <= 0.0 or tolerance > step:
            raise ValueError("Place terminal tolerance must be in (0, step]")
        continuous = cfg.get("place_continuous_descent", True)
        if not isinstance(continuous, bool):
            raise ValueError("place_continuous_descent must be a boolean")
        return {
            "step": step,
            "tolerance": tolerance,
            "continuous": continuous,
            "max_ratio": float(cfg.get("place_terminal_max_path_length_ratio", 1.5)),
            "max_deviation": float(cfg.get("place_terminal_max_path_deviation_m", step)),
        }

    @staticmethod
    def _terminal_samples(start, goal, step):
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        count = max(1, int(np.ceil(np.linalg.norm(goal - start) / step - 1e-9)))
        return [start + (goal - start) * (index / count) for index in range(1, count + 1)]

    def _terminal_path_ok(self, path, start, goal, options):
        if path is None:
            return True
        try:
            ratio, deviation = self.skill_runtime.measure_cartesian_path(path, start, goal)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return True
        return bool(
            np.isfinite(ratio)
            and np.isfinite(deviation)
            and float(ratio) <= options["max_ratio"] + 1e-6
            and float(deviation) <= options["max_deviation"] + 1e-6
        )

    def _place_request(self, policy, phase_id):
        return {
            "phase_id": phase_id,
            "collision_policy": policy,
            "active_target": self.pick_obj.name,
            "support": self.place_obj.name,
        }

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

        def ratios(key, default):
            bounds = self.skill_cfg.get(key, default)
            return np.random.uniform(float(bounds[0]), float(bounds[1]), CUROBO_BATCH_SIZE)

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
            pre_world = np.column_stack((x, y, np.full(CUROBO_BATCH_SIZE, b_max[2] + pre_offset)))
            place_world = np.column_stack((x, y, np.full(CUROBO_BATCH_SIZE, b_max[2] + place_offset)))
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
        target_rotations = self.generate_constrained_rotation_batch()
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

    def _plan_candidates(self, geometry, options):
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

        if runtime.batch_capability:
            runtime.sync_native_batch_attachment()
            runtime.transition_target(object_name, support_name, collision_policy=CollisionPolicy.ATTACHED_CARRY)
            pre_result = runtime.plan_pose_batch(
                pre, pre_q,
                collision_policy=CollisionPolicy.ATTACHED_CARRY,
                active_target=object_name,
                support=support_name,
                phase_id="place_preplace_batch",
                request_metadata=self._place_request(CollisionPolicy.ATTACHED_CARRY, "place_preplace_batch"),
            )
            pre_ok = self._plan_mask(pre_result, count)
            values = self._plan_paths(pre_result)
            pre_paths[: len(values)] = values[:count]
            if options["continuous"]:
                valid = np.flatnonzero(pre_ok & np.asarray([path is not None for path in pre_paths]))
                if len(valid):
                    same_target = np.allclose(place[valid], pre[valid]) and np.allclose(place_q[valid], pre_q[valid])
                    if same_target:
                        terminal_ok[valid] = True
                        for index in valid:
                            terminal_paths[index] = pre_paths[index]
                    else:
                        runtime.transition_target(object_name, support_name, collision_policy=CollisionPolicy.PLACEMENT_DESCENT)
                        terminal_result = runtime.plan_pose_batch(
                            place[valid], place_q[valid],
                            start_paths=[pre_paths[index] for index in valid],
                            collision_policy=CollisionPolicy.PLACEMENT_DESCENT,
                            active_target=object_name,
                            support=support_name,
                            phase_id="place_terminal_batch",
                            request_metadata=self._place_request(CollisionPolicy.PLACEMENT_DESCENT, "place_terminal_batch"),
                        )
                        terminal_values = self._plan_paths(terminal_result)
                        terminal_flags = self._plan_mask(terminal_result, len(valid))
                        for local, index in enumerate(valid):
                            path = terminal_values[local] if local < len(terminal_values) else None
                            if terminal_flags[local] and self._terminal_path_ok(path, pre[index], place[index], options):
                                terminal_ok[index] = True
                                terminal_paths[index] = path
            else:
                terminal_ok = pre_ok.copy()
        else:
            for index in range(count):
                runtime.transition_target(object_name, support_name, collision_policy=CollisionPolicy.ATTACHED_CARRY)
                result = runtime.plan_pose_result(
                    pre[index], pre_q[index],
                    collision_policy=CollisionPolicy.ATTACHED_CARRY,
                    active_target=object_name,
                    support=support_name,
                    phase_id="place_preplace",
                    request_metadata=self._place_request(CollisionPolicy.ATTACHED_CARRY, "place_preplace"),
                )
                pre_ok[index] = bool(self._plan_mask(result, 1)[0])
                paths = self._plan_paths(result)
                pre_paths[index] = paths[0] if paths else None
                if not pre_ok[index]:
                    continue
                if not options["continuous"]:
                    terminal_ok[index] = True
                    break
                if pre_paths[index] is None:
                    continue
                if np.allclose(pre[index], place[index]) and np.allclose(pre_q[index], place_q[index]):
                    terminal_ok[index] = True
                    terminal_paths[index] = pre_paths[index]
                    break
                runtime.transition_target(object_name, support_name, collision_policy=CollisionPolicy.PLACEMENT_DESCENT)
                result = runtime.plan_pose_from_path(
                    place[index], place_q[index], pre_paths[index],
                    collision_policy=CollisionPolicy.PLACEMENT_DESCENT,
                    active_target=object_name,
                    support=support_name,
                    phase_id="place_terminal",
                    request_metadata=self._place_request(CollisionPolicy.PLACEMENT_DESCENT, "place_terminal"),
                )
                terminal_ok[index] = bool(self._plan_mask(result, 1)[0])
                paths = self._plan_paths(result)
                terminal_paths[index] = paths[0] if paths else None
                if terminal_ok[index] and not self._terminal_path_ok(terminal_paths[index], pre[index], place[index], options):
                    terminal_ok[index] = False
                if terminal_ok[index]:
                    break

        # Candidate validation may have entered PLACEMENT_CONTACT.  Leave the
        # scene in ATTACHED carry state for execution; RESTORE_WORLD is only
        # emitted after the detach/settle phase.
        runtime.transition_target(
            object_name,
            support_name,
            collision_policy=CollisionPolicy.ATTACHED_CARRY,
        )
        valid = np.flatnonzero(pre_ok & terminal_ok)
        if len(valid) == 0:
            raise PlaceCandidateError(
                "NO_COLLISION_SAFE_CONTINUOUS_PLACE_PLAN"
                if options["continuous"]
                else "NO_COLLISION_FREE_PREPLACE_PLAN"
            )
        index = int(valid[0])
        self._selected_plan = {
            "candidate_index": index,
            "preplace_path": pre_paths[index],
            "terminal_path": terminal_paths[index],
        }
        return index

    def sample_gripper_place_traj(self):
        geometry = self._candidate_geometry()
        options = self._terminal_options()
        index = self._plan_candidates(geometry, options)
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
        options = self._terminal_options()
        tolerance = {
            "position_m": float(self.skill_cfg.get("t_eps", 0.005)),
            "orientation_rad": float(self.skill_cfg.get("o_eps", 0.05)),
        }
        pre_params = {}
        if self._selected_plan.get("preplace_path") is not None:
            pre_params["preplanned_joint_path"] = self._selected_plan["preplace_path"]
        commands = [
            MotionPhaseCommand(
                MotionPhase.TRANSIT_PREPLACE,
                pre_position,
                pre_orientation,
                gripper_action="close_gripper",
                active_object=object_name,
                support_object=support_name,
                allow_target_finger_contact=True,
                completion_tolerance=tolerance,
                params=pre_params,
            )
        ]
        points = [place_position] if options["continuous"] else self._terminal_samples(pre_position, place_position, options["step"])
        for point_index, point in enumerate(points):
            ratio = (point_index + 1) / len(points)
            orientation = (1.0 - ratio) * pre_orientation + ratio * place_orientation
            orientation /= max(float(np.linalg.norm(orientation)), 1e-12)
            params = {}
            if options["continuous"]:
                params.update(
                    continuous_descent=True,
                    max_cartesian_step_m=options["step"],
                    max_path_length_ratio=options["max_ratio"],
                    max_path_deviation_m=options["max_deviation"],
                )
                if self._selected_plan.get("terminal_path") is not None:
                    params["preplanned_joint_path"] = self._selected_plan["terminal_path"]
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.TERMINAL_PLACE_DESCENT,
                    point,
                    orientation,
                    gripper_action="close_gripper",
                    active_object=object_name,
                    support_object=support_name,
                    allow_target_finger_contact=True,
                    allow_object_support_contact=True,
                    completion_tolerance={
                        "position_m": options["tolerance"],
                        "orientation_rad": tolerance["orientation_rad"],
                    },
                    params=params,
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
        return commands

    def generate_manip_cmds(self):
        self.failure_reason = ""
        object_name, support_name = self.pick_obj.name, self.place_obj.name
        self.skill_runtime.assert_attached_owner(object_name)
        self.skill_runtime.transition_target(object_name, support_name, collision_policy=CollisionPolicy.ATTACHED_CARRY)
        try:
            result = self.sample_gripper_place_traj()
            self.manip_list = self._build_commands(result)
            self.publish_target_intent(self._target_intent)
        except PlaceCandidateError as exc:
            self.failure_reason = str(exc)
            self.manip_list = []
            self.publish_target_intent({"kind": "place", "objects": list(self.skill_cfg["objects"]), "has_target": False, "failure_reason": self.failure_reason})

    def generate_constrained_rotation_batch(self, batch_size=3000):
        if self.skill_cfg.get("preserve_attached_orientation", False):
            rotation = (self.T_base_world @ self.T_world_ee)[:3, :3]
            return np.repeat(rotation[None], CUROBO_BATCH_SIZE, axis=0)

        conditions = {
            "x": {"forward": (0, 0, 1), "backward": (0, 0, -1), "leftward": (1, 0, 1), "rightward": (1, 0, -1), "upward": (2, 0, 1), "downward": (2, 0, -1)},
            "y": {"forward": (0, 1, 1), "backward": (0, 1, -1), "leftward": (1, 1, 1), "rightward": (1, 1, -1), "upward": (2, 1, 1), "downward": (2, 1, -1)},
            "z": {"forward": (0, 2, 1), "backward": (0, 2, -1), "leftward": (1, 2, 1), "rightward": (1, 2, -1), "upward": (2, 2, 1), "downward": (2, 2, -1)},
        }
        rotations = R.random(batch_size).as_matrix()
        valid = np.ones(batch_size, dtype=bool)
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
        if len(choices) == 0:
            choices = rotations
        indices = np.random.choice(len(choices), CUROBO_BATCH_SIZE, replace=len(choices) < CUROBO_BATCH_SIZE)
        return choices[indices]

    def is_feasible(self, th=5):
        return bool(self.skill_runtime.num_plan_failed <= th and not self.failure_reason)

    def is_subtask_done(self, t_eps=1e-3, o_eps=5e-3):
        del t_eps, o_eps
        if not self.manip_list:
            return True
        return bool(self.skill_runtime.phase_complete(self.manip_list[0]))

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
            details = {"z": float(relative[2]), "threshold": threshold}
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
