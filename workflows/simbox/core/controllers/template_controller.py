"""
Template Controller base class for robot motion planning.

Common functionality extracted from FR3, FrankaRobotiq85, Genie1, Lift2, SplitAloha.
Subclasses implement _get_default_ignore_substring() and _configure_joint_indices().
"""

import json
import logging
import numbers
import os
import random
import time
import warnings
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.plan_utils import (
    filter_paths_by_position_error,
    filter_paths_by_rotation_error,
    sort_by_difference_js,
)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from omni.isaac.core import World
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
)
from omni.isaac.core.utils.types import ArticulationAction

from core.utils.joint_index_resolver import JointIndexResolutionError, resolve_joint_names
from core.planning.motion_command import MotionPhase, MotionPhaseCommand

LOGGER = logging.getLogger("de_logger")


# pylint: disable=line-too-long,unused-argument
class TemplateController(BaseController):
    """Base controller for CuRobo-based motion planning. Supports single and batch planning."""

    def __init__(
        self,
        name: str,
        robot_file: str,
        task: BaseTask,
        world: World,
        constrain_grasp_approach: bool = False,
        collision_activation_distance: float = 0.03,
        ignore_substring: Optional[List[str]] = None,
        use_batch: bool = False,
        trajectory_visualizer=None,
        skill_target_visualizer=None,
        collision_scene_manager=None,
        collision_world_mode: str = "legacy_stage_scan",
        **kwargs,
    ) -> None:
        super().__init__(name=name)
        self.name = name
        self.world = world
        self.task = task
        self.robot = self.task.robots[name]
        self.trajectory_visualizer = trajectory_visualizer
        self.collision_scene_manager = collision_scene_manager
        self.collision_world_mode = str(collision_world_mode)
        self.ignore_substring = list(self._get_default_ignore_substring())
        if ignore_substring is not None:
            self.ignore_substring = list(ignore_substring)
        self.ignore_substring.append(name)
        if self.trajectory_visualizer is not None:
            self.ignore_substring.append("__debug_curobo_trajectory__")
        if skill_target_visualizer is not None:
            self.ignore_substring.append("__debug_skill_targets__")
        if self.collision_world_mode == "physics_schema" and self.ignore_substring:
            warnings.warn(
                "ignore_substring is deprecated and has no effect in physics_schema mode; "
                "CollisionSceneManager uses exact enabled CollisionAPI prims",
                DeprecationWarning,
                stacklevel=2,
            )
        self.use_batch = use_batch
        self.constrain_grasp_approach = constrain_grasp_approach
        self.collision_activation_distance = collision_activation_distance
        self.usd_help = UsdHelper()
        self.tensor_args = TensorDeviceType()
        self.init_curobo = False
        self.robot_file = robot_file
        self.num_plan_failed = 0
        self.raw_js_names = []
        self.cmd_js_names = []
        self.arm_indices = np.array([])
        self.gripper_indices = np.array([])
        self.reference_prim_path = None
        self.lr_name = None
        self._ee_trans = 0.0
        self._ee_ori = 0.0
        self._gripper_state = 1.0
        self._gripper_joint_position = np.array([1.0])
        self._last_command_name = "unknown"
        self.idx_list = None
        # Keep the command object itself, not ``id(command)``.  CPython may
        # reuse an id immediately after the previous phase command is popped;
        # that made a new Place phase inherit Pick's finished/failed flags.
        self._active_phase_command = None
        self._phase_plan_started = False
        self._phase_plan_finished = False
        self._phase_bookkeeping_done = False
        self._phase_dwell_count = 0
        self._phase_tracking_failed = False
        self._phase_plan_failed = False
        self._phase_completion_logged = False
        self._last_commanded_arm_position = None

        self._configure_joint_indices(robot_file)
        self._resolve_runtime_control_indices()
        self._load_robot(robot_file)
        self._load_kin_model()
        self._load_world()
        self._init_motion_gen()

        self.usd_help.load_stage(self.world.stage)
        if self.collision_scene_manager is not None:
            self.collision_scene_manager.bind_controller(self)
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.num_last_cmd = 0
        self._last_arm_action = None
        self._curobo_plan_debug_counter = 0
        self._curobo_plan_debug_dir = os.environ.get(
            "SIMBOX_CUROBO_PLAN_DEBUG_DIR",
            os.path.join("output", "ros_bridge", "skills", "curobo_plan_debug"),
        )
        if self.collision_world_mode == "physics_schema":
            physics_dt = float(self.world.get_physics_dt())
            interpolation_dt = float(self.motion_gen.interpolation_dt)
            requested_stride = max(1, int(round(physics_dt / interpolation_dt)))
            safety_cfg = self.task.cfg.get("planning", {}).get("execution_safety", {})
            max_stride = max(1, int(safety_cfg.get("max_waypoint_stride", 2)))
            self.ds_ratio = min(requested_stride, max_stride)
        else:
            # LEGACY_STAGE_SCAN keeps the original one-waypoint-per-step timing.
            self.ds_ratio = 1
        LOGGER.warning(
            "[ExecutionTiming] robot=%s arm=%s physics_dt=%.6f interpolation_dt=%.6f ds_ratio=%d",
            self.name,
            self.lr_name,
            float(self.world.get_physics_dt()),
            float(self.motion_gen.interpolation_dt),
            self.ds_ratio,
        )

    def _get_default_ignore_substring(self) -> List[str]:
        return ["material", "Plane", "conveyor", "scene", "table"]

    def _configure_joint_indices(self, robot_file: str) -> None:
        raise NotImplementedError

    def _resolve_runtime_control_indices(self) -> None:
        """Resolve arm actuation indices by name and fail before any command can be misrouted."""

        if self.lr_name not in {"left", "right"}:
            raise JointIndexResolutionError(f"controller {self.name} did not select a left/right arm")
        runtime_names = list(self.robot.dof_names)
        resolved_arm_indices = resolve_joint_names(
            runtime_names,
            self.cmd_js_names,
            group=f"{self.name}.{self.lr_name}_arm",
        )
        configured_arm_indices = [int(index) for index in self.arm_indices]
        if configured_arm_indices != resolved_arm_indices:
            LOGGER.warning(
                "[JointIndex] controller=%s arm=%s corrected arm indices from %s to %s using runtime dof_names",
                self.name,
                self.lr_name,
                configured_arm_indices,
                resolved_arm_indices,
            )

        arm_indices_field = f"{self.lr_name}_joint_indices"
        gripper_indices_field = f"{self.lr_name}_gripper_indices"
        self.arm_indices = np.asarray(resolved_arm_indices, dtype=np.int64)
        resolved_gripper_indices = [int(index) for index in getattr(self.robot, gripper_indices_field)]
        if set(resolved_arm_indices) & set(resolved_gripper_indices):
            raise JointIndexResolutionError(
                f"controller {self.name} {self.lr_name} arm/gripper indices overlap: "
                f"arm={resolved_arm_indices}, gripper={resolved_gripper_indices}"
            )
        self.gripper_indices = np.asarray(resolved_gripper_indices, dtype=np.int64)

        setattr(self.robot, arm_indices_field, resolved_arm_indices)
        self.robot.cfg[arm_indices_field] = resolved_arm_indices
        self.robot.cfg[gripper_indices_field] = resolved_gripper_indices
        LOGGER.warning(
            "[JointIndexAudit] controller=%s arm=%s joints=%s arm_indices=%s gripper_indices=%s",
            self.name,
            self.lr_name,
            list(self.cmd_js_names),
            resolved_arm_indices,
            resolved_gripper_indices,
        )

    def _load_robot(self, robot_file: str) -> None:
        self.robot_cfg = load_yaml(robot_file)["robot_cfg"]

    def _load_kin_model(self) -> None:
        urdf_file = self.robot_cfg["kinematics"]["urdf_path"]
        base_link = self.robot_cfg["kinematics"]["base_link"]
        ee_link = self.robot_cfg["kinematics"]["ee_link"]
        robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, self.tensor_args)
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

    def _load_world(self, use_default: bool = True) -> None:
        if self.collision_world_mode == "physics_schema":
            if self.collision_scene_manager is None:
                raise RuntimeError("physics_schema mode requires CollisionSceneManager")
            if self.ignore_substring:
                warnings.warn(
                    "ignore_substring is deprecated and ignored in physics_schema mode",
                    DeprecationWarning,
                    stacklevel=2,
                )
            self.world_cfg = self.collision_scene_manager.build_world_config(
                self.reference_prim_path
            )
            return
        if use_default:
            self.world_cfg = WorldConfig()
        else:
            world_cfg_table = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
            )
            self._world_cfg_table = world_cfg_table
            self._world_cfg_table.cuboid[0].pose[2] -= 10.5
            world_cfg1 = WorldConfig.from_dict(
                load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
            ).get_mesh_world()
            world_cfg1.mesh[0].name += "_mesh"
            world_cfg1.mesh[0].pose[2] = -10.5
            self.world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    def _get_motion_gen_collision_cache(self):
        """Override in subclasses to use different cache size (e.g. FR3 uses 1000)."""
        return {"obb": 700, "mesh": 700}

    def _get_grasp_approach_linear_axis(self) -> int:
        """Axis for grasp approach constraint (0=x, 1=y, 2=z). Override in subclasses (e.g. Lift2 uses 0)."""
        if self.robot.cfg["ee_axis"] == "x":
            return 0
        elif self.robot.cfg["ee_axis"] == "y":
            return 1
        elif self.robot.cfg["ee_axis"] == "z":
            return 2
        else:
            raise NotImplementedError

    def _get_sort_path_weights(self) -> Optional[List[float]]:
        """Optional per-joint weights for sort_by_difference_js.

        Used when selecting among batch paths. None means equal weights.
        Override in subclasses (e.g. Genie1).
        """
        return None

    @staticmethod
    def _plan_success_count(result) -> int:
        success = getattr(result, "success", None)
        if success is None:
            return -1
        return int(success.sum().item())

    @staticmethod
    def _error_range(result, name: str):
        values = getattr(result, name, None)
        if values is None or values.numel() == 0:
            return None
        return float(values.min().item()), float(values.max().item())

    def _log_plan_result(self, context: str, result, target=None):
        success_count = self._plan_success_count(result)
        pos_range = self._error_range(result, "position_error")
        rot_range = self._error_range(result, "rotation_error")
        msg = (
            f"[PlanDebug] {context} robot={self.name} arm={self.lr_name} command={self._last_command_name} "
            f"use_batch={self.use_batch} success_count={success_count}"
        )
        if target is not None:
            msg += f" target={np.array2string(np.asarray(target), precision=4, suppress_small=True)}"
        if pos_range is not None:
            msg += f" pos_error_range=({pos_range[0]:.6f}, {pos_range[1]:.6f})"
        if rot_range is not None:
            msg += f" rot_error_range=({rot_range[0]:.6f}, {rot_range[1]:.6f})"
        LOGGER.info(msg)

    def _visualize_selected_plan(self) -> None:
        if self.trajectory_visualizer is None or self.cmd_plan is None:
            return
        try:
            self.trajectory_visualizer.record_plan(self, self.cmd_plan, self._last_command_name)
        except Exception:
            # The overlay is observational and must never change controller behavior.
            LOGGER.exception(
                "[TrajectoryDebug] failed to visualize robot=%s arm=%s command=%s",
                self.name,
                self.lr_name,
                self._last_command_name,
            )

    def _init_motion_gen(self) -> None:
        pose_metric = None
        if self.constrain_grasp_approach:
            pose_metric = PoseCostMetric.create_grasp_approach_metric(
                offset_position=0.1,
                linear_axis=self._get_grasp_approach_linear_axis(),
            )
        if self.use_batch:
            self.plan_config = MotionGenPlanConfig(
                enable_graph=True,
                enable_opt=True,
                need_graph_success=True,
                enable_graph_attempt=4,
                max_attempts=4,
                enable_finetune_trajopt=True,
                parallel_finetune=True,
                time_dilation_factor=1.0,
            )
        else:
            self.plan_config = MotionGenPlanConfig(
                enable_graph=False,
                enable_graph_attempt=7,
                max_attempts=10,
                pose_cost_metric=pose_metric,
                enable_finetune_trajopt=True,
                time_dilation_factor=1.0,
            )
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            interpolation_dt=0.01,
            collision_activation_distance=self.collision_activation_distance,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            self_collision_check=True,
            collision_cache=self._get_motion_gen_collision_cache(),
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            optimize_dt=True,
            trajopt_dt=None,
            trim_steps=None,
            project_pose_to_goal_frame=False,
        )
        ik_config = IKSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": 700, "mesh": 700},
        )
        self.ik_solver = IKSolver(ik_config)
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up..")
        if self.use_batch:
            self.motion_gen.warmup(parallel_finetune=True, batch=CUROBO_BATCH_SIZE)
        else:
            self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        self.world_model = self.motion_gen.world_collision
        self.motion_gen.clear_world_cache()
        self.motion_gen.reset(reset_seed=False)
        self.motion_gen.update_world(self.world_cfg)
        self._world_update_signature = self._make_world_update_signature(self.world_cfg)
        LOGGER.info(
            "[PlanDebug] motion_gen initialized robot=%s arm=%s use_batch=%s constrain_grasp_approach=%s "
            "collision_activation_distance=%s",
            self.name,
            self.lr_name,
            self.use_batch,
            self.constrain_grasp_approach,
            self.collision_activation_distance,
        )

    def update_pose_cost_metric(self, hold_vec_weight: Optional[List[float]] = None) -> None:
        # reference: https://curobo.org/advanced_examples/3_constrained_planning.html
        # [angular-x, angular-y, angular-z, linear-x, linear-y, linear-z]
        # For example,
        # when hold_vec_weight is None, the corresponding list is [0, 0, 0, 0, 0, 0],
        # there is no cost added in any directions.
        # When hold_vec_weight = [1, 1, 1, 0, 0, 0], the tool orientation is holed.
        # assert hold_vec_weight is None or len(hold_vec_weight) == 6
        if hold_vec_weight:
            pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.motion_gen.tensor_args.to_device(hold_vec_weight),
            )
        else:
            pose_cost_metric = None
        self.plan_config.pose_cost_metric = pose_cost_metric

    @staticmethod
    def _debug_json_ready(value: Any):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, numbers.Real):
            return float(value)
        if isinstance(value, np.ndarray):
            return TemplateController._debug_json_ready(value.tolist())
        if isinstance(value, torch.Tensor):
            return TemplateController._debug_json_ready(value.detach().cpu().numpy())
        if isinstance(value, (list, tuple)):
            return [TemplateController._debug_json_ready(v) for v in value]
        if isinstance(value, dict):
            return {str(k): TemplateController._debug_json_ready(v) for k, v in value.items()}
        return str(value)

    @staticmethod
    def _debug_joint_state_to_dict(joint_state):
        if joint_state is None:
            return None
        payload = {}
        for field in ("position", "velocity", "acceleration", "jerk"):
            if hasattr(joint_state, field):
                payload[field] = TemplateController._debug_json_ready(getattr(joint_state, field))
        if hasattr(joint_state, "joint_names"):
            payload["joint_names"] = TemplateController._debug_json_ready(getattr(joint_state, "joint_names"))
        if hasattr(joint_state, "shape"):
            payload["shape"] = TemplateController._debug_json_ready(getattr(joint_state, "shape"))
        return payload

    @staticmethod
    def _debug_norm_delta(a, b):
        if a is None or b is None:
            return None
        a_arr = np.asarray(a, dtype=float).reshape(-1)
        b_arr = np.asarray(b, dtype=float).reshape(-1)
        if a_arr.shape != b_arr.shape:
            return None
        return float(np.linalg.norm(a_arr - b_arr))

    def _write_curobo_plan_debug(
        self,
        *,
        result,
        sim_js,
        js_names,
        ee_trans,
        ee_ori,
        raw_plan,
        ordered_cmd_plan,
        branch: str,
        selected_path_index=None,
        selected_path_source: str = "",
    ) -> None:
        try:
            os.makedirs(self._curobo_plan_debug_dir, exist_ok=True)
            self._curobo_plan_debug_counter += 1
            current_full = np.asarray(sim_js.positions, dtype=float).copy()
            current_arm = current_full[self.arm_indices].copy()
            first_ordered = None
            last_ordered = None
            if ordered_cmd_plan is not None and len(ordered_cmd_plan) > 0:
                first_ordered = ordered_cmd_plan[0].position.detach().cpu().numpy()
                last_ordered = ordered_cmd_plan[-1].position.detach().cpu().numpy()

            cu_js = JointState(
                position=self.tensor_args.to_device(sim_js.positions),
                velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
                acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
                jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
                joint_names=js_names,
            )
            cu_js_ordered = cu_js.get_ordered_joint_state(self.cmd_js_names)

            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = (
                f"{timestamp}_{int((time.time() % 1) * 1000000):06d}_"
                f"{self.name}_{self.lr_name}_plan_{self._curobo_plan_debug_counter:04d}.json"
            )
            output_path = os.path.join(self._curobo_plan_debug_dir, filename)

            payload = {
                "schema_version": 1,
                "controller": {
                    "name": self.name,
                    "lr_name": self.lr_name,
                    "robot_file": self.robot_file,
                    "use_batch": bool(self.use_batch),
                    "branch": branch,
                    "selected_path_index": self._debug_json_ready(selected_path_index),
                    "selected_path_source": selected_path_source,
                },
                "joint_mapping": {
                    "robot_dof_names": self._debug_json_ready(js_names),
                    "cmd_js_names": self._debug_json_ready(self.cmd_js_names),
                    "raw_js_names": self._debug_json_ready(self.raw_js_names),
                    "arm_indices": self._debug_json_ready(self.arm_indices),
                    "gripper_indices": self._debug_json_ready(self.gripper_indices),
                    "idx_list": self._debug_json_ready(self.idx_list),
                },
                "goal": {
                    "ee_translation": self._debug_json_ready(ee_trans),
                    "ee_orientation": self._debug_json_ready(ee_ori),
                },
                "input_current_state": {
                    "sim_js": self._debug_joint_state_to_dict(sim_js),
                    "current_arm_sim_order": self._debug_json_ready(current_arm),
                    "curobo_input_ordered_cmd_js_names": self._debug_joint_state_to_dict(cu_js_ordered),
                },
                "result_summary": {
                    "success": self._debug_json_ready(getattr(result, "success", None)),
                    "status": self._debug_json_ready(getattr(result, "status", None)),
                    "valid_query": self._debug_json_ready(getattr(result, "valid_query", None)),
                    "position_error": self._debug_json_ready(getattr(result, "position_error", None)),
                    "rotation_error": self._debug_json_ready(getattr(result, "rotation_error", None)),
                    "cspace_error": self._debug_json_ready(getattr(result, "cspace_error", None)),
                    "optimized_dt": self._debug_json_ready(getattr(result, "optimized_dt", None)),
                    "interpolation_dt": self._debug_json_ready(getattr(result, "interpolation_dt", None)),
                    "path_buffer_last_tstep": self._debug_json_ready(
                        getattr(result, "path_buffer_last_tstep", None)
                    ),
                    "used_graph": self._debug_json_ready(getattr(result, "used_graph", None)),
                    "attempts": self._debug_json_ready(getattr(result, "attempts", None)),
                    "trajopt_attempts": self._debug_json_ready(getattr(result, "trajopt_attempts", None)),
                },
                "continuity": {
                    "first_ordered_minus_current_arm_norm": self._debug_norm_delta(first_ordered, current_arm),
                    "last_ordered_minus_current_arm_norm": self._debug_norm_delta(last_ordered, current_arm),
                    "first_ordered_position": self._debug_json_ready(first_ordered),
                    "last_ordered_position": self._debug_json_ready(last_ordered),
                },
                "raw_plan": self._debug_joint_state_to_dict(raw_plan),
                "ordered_cmd_plan": self._debug_joint_state_to_dict(ordered_cmd_plan),
            }
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(self._debug_json_ready(payload), handle, indent=2, ensure_ascii=False)
            print(
                "[curobo-plan-debug] Wrote plan debug: "
                f"{output_path}; first_delta={payload['continuity']['first_ordered_minus_current_arm_norm']}"
            )
        except Exception as exc:  # Debug writing must never break an episode.
            print(f"[curobo-plan-debug] Failed to write plan debug: {exc}")

    @staticmethod
    def _signature_value(value: Any):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, numbers.Real):
            return round(float(value), 6)
        if isinstance(value, np.ndarray):
            return tuple(round(float(x), 6) for x in value.reshape(-1).tolist())
        if isinstance(value, torch.Tensor):
            return tuple(round(float(x), 6) for x in value.detach().cpu().reshape(-1).tolist())
        if isinstance(value, (list, tuple)):
            return tuple(TemplateController._signature_value(x) for x in value)
        return str(value)

    @classmethod
    def _make_world_update_signature(cls, world_cfg: WorldConfig):
        objects = getattr(world_cfg, "objects", None) or []
        signature = []
        for obj in objects:
            signature.append(
                (
                    type(obj).__name__,
                    getattr(obj, "name", None),
                    cls._signature_value(getattr(obj, "pose", None)),
                    cls._signature_value(getattr(obj, "dims", None)),
                    cls._signature_value(getattr(obj, "scale", None)),
                    getattr(obj, "file_path", None),
                    cls._signature_value(getattr(obj, "vertices", None)),
                    cls._signature_value(getattr(obj, "faces", None)),
                )
            )
        return tuple(signature)

    def _update_world_if_changed(self, obstacles: WorldConfig) -> None:
        signature = self._make_world_update_signature(obstacles)
        if self.motion_gen is not None and signature != getattr(self, "_world_update_signature", None):
            self.motion_gen.update_world(obstacles)
        self.world_cfg = obstacles
        self._world_update_signature = signature

    def update(self) -> None:
        if self.collision_world_mode == "physics_schema":
            self.collision_scene_manager.sync_dynamic_poses(
                self._step_idx, interval_steps=1, force=True
            )
            self.collision_scene_manager.audit_controller(self)
            return
        self._legacy_update()

    def _legacy_update(self) -> None:
        """LEGACY_STAGE_SCAN: retained for explicit legacy_stage_scan mode."""

        # LEGACY_BEGIN: keyword-based collision world, retained for comparison
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=self.ignore_substring, reference_prim_path=self.reference_prim_path
        ).get_collision_check_world()
        self._update_world_if_changed(obstacles)

    def _clear_attached_object_state(self) -> None:
        if self.motion_gen is None:
            return
        try:
            self.motion_gen.detach_object_from_robot()
            self.motion_gen.clear_world_cache()
            self._world_update_signature = None
        except Exception as exc:
            print(f"[curobo-controller] Failed to clear attached object state during reset: {exc}")
        # LEGACY_END

    def reset(self, ignore_substring: Optional[str] = None) -> None:
        if ignore_substring:
            self.ignore_substring = ignore_substring
        self._clear_attached_object_state()
        self.update()
        self.init_curobo = True
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.num_last_cmd = 0
        self._last_arm_action = None
        self.num_plan_failed = 0
        if self.lr_name == "left":
            self._gripper_state = 1.0 if self.robot.left_gripper_state == 1.0 else -1.0
        elif self.lr_name == "right":
            self._gripper_state = 1.0 if self.robot.right_gripper_state == 1.0 else -1.0
        if self.lr_name == "left":
            self.robot_ee_path = self.robot.fl_ee_path
            self.robot_base_path = self.robot.fl_base_path
        else:
            self.robot_ee_path = self.robot.fr_ee_path
            self.robot_base_path = self.robot.fr_base_path
        self.T_base_ee_init = get_relative_transform(
            get_prim_at_path(self.robot_ee_path), get_prim_at_path(self.robot_base_path)
        )
        self.T_world_base_init = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        self.T_world_ee_init = self.T_world_base_init @ self.T_base_ee_init
        self._ee_trans, self._ee_ori = self.get_ee_pose()
        self._ee_trans = self.tensor_args.to_device(self._ee_trans)
        self._ee_ori = self.tensor_args.to_device(self._ee_ori)
        self.update_pose_cost_metric()

    def plan_batch(self, ee_translation_goal_batch, ee_orientation_goal_batch, sim_js, js_names):
        t1 = time.time()
        torch.cuda.synchronize()
        sim_js_positions = (sim_js.positions)[np.newaxis, :]
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal_batch),
            quaternion=self.tensor_args.to_device(ee_orientation_goal_batch),
            batch=CUROBO_BATCH_SIZE,
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))),
            velocity=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))) * 0.0,
            acceleration=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))) * 0.0,
            jerk=self.tensor_args.to_device(np.tile(sim_js_positions, (CUROBO_BATCH_SIZE, 1))) * 0.0,
            joint_names=js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.cmd_js_names)
        result = self.motion_gen.plan_batch(cu_js, ik_goal, self.plan_config.clone())
        t2 = time.time()
        torch.cuda.synchronize()
        print("plan batch duration :", t2 - t1)
        return result

    def plan(self, ee_translation_goal, ee_orientation_goal, sim_js: JointState, js_names: list):
        if self.use_batch:
            ik_goal = Pose(
                position=self.tensor_args.to_device(ee_translation_goal.unsqueeze(0).expand(CUROBO_BATCH_SIZE, -1)),
                quaternion=self.tensor_args.to_device(ee_orientation_goal.unsqueeze(0).expand(CUROBO_BATCH_SIZE, -1)),
                batch=CUROBO_BATCH_SIZE,
            )
            cu_js = JointState(
                position=self.tensor_args.to_device(np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1))),
                velocity=self.tensor_args.to_device(np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1)))
                * 0.0,
                acceleration=self.tensor_args.to_device(
                    np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1))
                )
                * 0.0,
                jerk=self.tensor_args.to_device(np.tile((sim_js.positions)[np.newaxis, :], (CUROBO_BATCH_SIZE, 1)))
                * 0.0,
                joint_names=js_names,
            )
            cu_js = cu_js.get_ordered_joint_state(self.cmd_js_names)
            return self.motion_gen.plan_batch(cu_js, ik_goal, self.plan_config.clone())
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_goal),
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.cmd_js_names)
        return self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config.clone())

    def forward(self, manip_cmd, eps=5e-3):
        if isinstance(manip_cmd, MotionPhaseCommand):
            return self.forward_phase_command(manip_cmd)
        ee_trans, ee_ori = manip_cmd[0:2]
        gripper_fn = manip_cmd[2]
        params = dict(manip_cmd[3])
        self._last_command_name = gripper_fn
        skip_plan = bool(params.pop("skip_plan", False))
        gripper_action = params.pop("gripper_action", None)
        params.pop("t_eps", None)
        params.pop("o_eps", None)
        assert hasattr(self, gripper_fn)
        method = getattr(self, gripper_fn)
        if gripper_fn in ["in_plane_rotation", "mobile_move", "dummy_forward", "observe_hold"]:
            return method(**params)
        elif gripper_fn in ["update_pose_cost_metric", "update_specific"]:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps=eps, skip_plan=True, gripper_action=gripper_action)
        else:
            method(**params)
            return self.ee_forward(ee_trans, ee_ori, eps, skip_plan=skip_plan, gripper_action=gripper_action)

    def _begin_phase_command(self, command: MotionPhaseCommand) -> bool:
        if command is self._active_phase_command:
            return False
        self._active_phase_command = command
        self._phase_plan_started = False
        self._phase_plan_finished = False
        self._phase_bookkeeping_done = False
        self._phase_dwell_count = 0
        self._phase_tracking_failed = False
        self._phase_plan_failed = False
        self._phase_completion_logged = False
        self._last_command_name = command.phase.value
        self._phase_base_position, self._phase_base_orientation = self.get_armbase_pose()
        LOGGER.warning(
            "[PhaseDebug] start robot=%s arm=%s phase=%s object=%s support=%s target=%s",
            self.name,
            self.lr_name,
            command.phase.value,
            command.active_object,
            command.support_object,
            None
            if command.target_position is None
            else np.asarray(command.target_position, dtype=float).round(5).tolist(),
        )
        return True

    def forward_phase_command(self, command: MotionPhaseCommand):
        """Execute a structured Pick/Place phase while preserving tuple compatibility."""

        first_step = self._begin_phase_command(command)
        manager = self.collision_scene_manager
        if manager is None:
            raise RuntimeError("MotionPhaseCommand requires CollisionSceneManager")
        robot, arm = self.name, self.lr_name
        if first_step:
            if command.phase == MotionPhase.SYNC_WORLD:
                manager.sync_dynamic_poses(self._step_idx, interval_steps=1, force=True)
                if command.active_object:
                    manager.begin_target_transit(command.active_object, robot, arm)
                self._phase_bookkeeping_done = True
            elif command.phase == MotionPhase.TRANSIT_PREGRASP:
                manager.begin_target_transit(command.active_object, robot, arm)
            elif command.phase == MotionPhase.TERMINAL_GRASP_APPROACH:
                manager.begin_target_approach(command.active_object, robot, arm)
                preplanned_path = command.params.get("preplanned_joint_path")
                if preplanned_path is not None:
                    cmd_plan = self.motion_gen.get_full_js(preplanned_path)
                    self.idx_list = list(range(len(self.raw_js_names)))
                    self.cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
                    self.cmd_idx = 0
                    self._phase_plan_started = True
                    self._ee_trans = self.tensor_args.to_device(command.target_position)
                    self._ee_ori = self.tensor_args.to_device(command.target_orientation)
                    self._visualize_selected_plan()
                    LOGGER.warning(
                        "[PhaseDebug] selected-plan robot=%s arm=%s phase=%s waypoints=%d stride=%d cached=true",
                        self.name,
                        self.lr_name,
                        command.phase.value,
                        len(self.cmd_plan),
                        self.ds_ratio,
                    )
            elif command.phase == MotionPhase.ATTACH:
                verify_contact = command.params.get("verify_grasp_contact")
                if not callable(verify_contact) or not bool(verify_contact()):
                    raise RuntimeError(
                        "ATTACH requires a verified target-finger contact from GRIPPER_CLOSE"
                    )
                manager.attach_target(command.active_object, robot, arm)
                self._phase_bookkeeping_done = True
            elif command.phase == MotionPhase.TERMINAL_PLACE_DESCENT:
                manager.begin_placement_descent(
                    command.active_object, command.support_object, robot, arm
                )
            elif command.phase == MotionPhase.DETACH_AND_SETTLE:
                manager.detach_target(command.active_object, robot, arm)
                self._phase_bookkeeping_done = True
            elif command.phase == MotionPhase.TERMINAL_RETREAT:
                manager.begin_terminal_retreat(command.active_object, robot, arm)
            elif command.phase == MotionPhase.RESTORE_WORLD:
                manager.restore_world(command.active_object)
                self._phase_bookkeeping_done = True

        if command.gripper_action:
            if not hasattr(self, command.gripper_action):
                raise AttributeError(f"unknown gripper action: {command.gripper_action}")
            getattr(self, command.gripper_action)()

        if command.is_bookkeeping or command.phase in {
            MotionPhase.GRIPPER_CLOSE,
            MotionPhase.GRIPPER_OPEN,
        }:
            self._phase_dwell_count += 1
            position, orientation = self.get_ee_pose()
            return self.ee_forward(position, orientation, skip_plan=True)
        return self.ee_forward(
            command.target_position,
            command.target_orientation,
            # Planning-target identity and physical completion are different
            # contracts.  In particular, terminal Pick/Place samples are
            # spaced by exactly ``terminal_step_m`` and use that same value as
            # their completion tolerance.  Passing the tolerance here made a
            # new 5 mm target fail the strict ``delta > eps`` plan trigger,
            # while the measured EE could still be just outside completion:
            # no plan, no completion, and an infinite hold loop.  Keep target
            # change detection tight; ``is_phase_command_complete`` applies
            # the command's actual completion tolerance separately.
            eps=command.planning_epsilon,
        )

    def is_phase_command_complete(self, command: MotionPhaseCommand) -> bool:
        if command is not self._active_phase_command:
            return False
        if command.is_bookkeeping:
            if (
                command.phase == MotionPhase.DETACH_AND_SETTLE
                and self._phase_dwell_count >= max(1, command.dwell_steps)
                and not command.params.get("settle_finalized", False)
            ):
                self.collision_scene_manager.finalize_detach_target(
                    command.active_object, self.name, self.lr_name
                )
                command.params["settle_finalized"] = True
            return self._phase_bookkeeping_done and self._phase_dwell_count >= max(1, command.dwell_steps)
        if command.phase in {MotionPhase.GRIPPER_CLOSE, MotionPhase.GRIPPER_OPEN}:
            return self._phase_dwell_count >= max(1, command.dwell_steps)
        if (
            command.phase == MotionPhase.TERMINAL_PLACE_DESCENT
            and command.params.get("contact_complete", False)
        ):
            self.cmd_plan = None
            self.cmd_idx = 0
            self._phase_plan_finished = True
            return True
        position, orientation = self.get_ee_pose()
        position_error = float(np.linalg.norm(position - command.target_position))
        orientation_error = float(
            2.0
            * np.arccos(
                np.clip(abs(np.dot(orientation, command.target_orientation)), 0.0, 1.0)
            )
        )
        inside = (
            position_error <= command.translation_tolerance
            and orientation_error <= command.orientation_tolerance
        )
        if inside and self.cmd_plan is None:
            self._phase_plan_finished = True
            if not self._phase_completion_logged:
                LOGGER.warning(
                    "[PhaseDebug] complete robot=%s arm=%s phase=%s position_error=%.6f orientation_error=%.6f",
                    self.name,
                    self.lr_name,
                    command.phase.value,
                    position_error,
                    orientation_error,
                )
                self._phase_completion_logged = True
            return True
        if self._phase_plan_finished and not inside:
            self._phase_tracking_failed = True
            if not self._phase_completion_logged:
                LOGGER.warning(
                    "[PhaseDebug] tracking-failed robot=%s arm=%s phase=%s position_error=%.6f orientation_error=%.6f",
                    self.name,
                    self.lr_name,
                    command.phase.value,
                    position_error,
                    orientation_error,
                )
                self._phase_completion_logged = True
        return False

    def clear_plan_and_hold(self) -> None:
        """Stop consuming the old plan; the next action holds measured joints."""

        self.cmd_plan = None
        self.cmd_idx = 0
        self._phase_plan_started = False
        self._phase_plan_finished = False
        self._phase_tracking_failed = False
        self._phase_plan_failed = False
        self._active_phase_command = None
        position, orientation = self.get_ee_pose()
        self._ee_trans = self.tensor_args.to_device(position)
        self._ee_ori = self.tensor_args.to_device(orientation)

    def hold_action(self):
        """Return an articulation target equal to the measured current joints."""

        sim_js = self.robot.get_joints_state()
        arm_action = np.asarray(sim_js.positions[self.arm_indices], dtype=float)
        gripper_action = self.get_gripper_action()
        return {
            "joint_positions": np.concatenate([arm_action, gripper_action]),
            "joint_indices": np.concatenate([self.arm_indices, self.gripper_indices]),
            "lr_name": self.lr_name,
            "arm_action": arm_action,
            "gripper_action": gripper_action,
        }

    def observe_hold(self):
        """Hold measured arm and gripper joints without changing controller state."""

        sim_js = self.robot.get_joints_state()
        positions = sim_js.positions
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
        positions = np.asarray(positions, dtype=float)
        joint_indices = np.concatenate([self.arm_indices, self.gripper_indices])
        joint_positions = positions[joint_indices].copy()
        arm_count = len(self.arm_indices)
        return {
            "joint_positions": joint_positions,
            "joint_indices": joint_indices,
            "lr_name": self.lr_name,
            "arm_action": joint_positions[:arm_count],
            "gripper_action": joint_positions[arm_count:],
        }

    def ee_forward(
        self,
        ee_trans: torch.Tensor | np.ndarray,
        ee_ori: torch.Tensor | np.ndarray,
        eps=1e-4,
        skip_plan=False,
        gripper_action=None,
    ):
        ee_trans = self.tensor_args.to_device(ee_trans)
        ee_ori = self.tensor_args.to_device(ee_ori)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        plan_flag = torch.logical_or(
            torch.norm(self._ee_trans - ee_trans) > eps,
            torch.norm(self._ee_ori - ee_ori) > eps,
        )
        if not skip_plan:
            if plan_flag:
                self.cmd_plan = None
                self.cmd_idx = 0
                self._step_idx = 0
                self.num_last_cmd = 0
                self._last_arm_action = None
                result = self.plan(ee_trans, ee_ori, sim_js, js_names)
                self._phase_plan_started = True
                self._log_plan_result("ee_forward", result, target=ee_trans.detach().cpu().numpy())
                if self.use_batch:
                    if result.success.any():
                        self._ee_trans = ee_trans
                        self._ee_ori = ee_ori
                        paths = result.get_successful_paths()
                        position_filter_res = filter_paths_by_position_error(
                            paths, result.position_error[result.success]
                        )
                        rotation_filter_res = filter_paths_by_rotation_error(
                            paths, result.rotation_error[result.success]
                        )
                        filtered_paths = [
                            p for i, p in enumerate(paths) if position_filter_res[i] and rotation_filter_res[i]
                        ]
                        if len(filtered_paths) == 0:
                            filtered_paths = paths
                        sort_weights = self._get_sort_path_weights()  # pylint: disable=assignment-from-none
                        weights_arg = self.tensor_args.to_device(sort_weights) if sort_weights is not None else None
                        sorted_indices = sort_by_difference_js(filtered_paths, weights=weights_arg)
                        cmd_plan = self.motion_gen.get_full_js(paths[sorted_indices[0]])
                        self.idx_list = list(range(len(self.raw_js_names)))
                        self.cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
                        self._write_curobo_plan_debug(
                            result=result,
                            sim_js=sim_js,
                            js_names=js_names,
                            ee_trans=ee_trans,
                            ee_ori=ee_ori,
                            raw_plan=cmd_plan,
                            ordered_cmd_plan=self.cmd_plan,
                            branch="batch",
                            selected_path_index=sorted_indices[0],
                            selected_path_source="paths[sorted_indices[0]]",
                        )
                        self._visualize_selected_plan()
                        LOGGER.warning(
                            "[PhaseDebug] selected-plan robot=%s arm=%s phase=%s waypoints=%d stride=%d cached=false",
                            self.name,
                            self.lr_name,
                            self._last_command_name,
                            len(self.cmd_plan),
                            self.ds_ratio,
                        )
                        self.num_plan_failed = 0
                    else:
                        print("Plan did not converge to a solution.")
                        self._phase_plan_failed = True
                        self.num_plan_failed += 1
                        LOGGER.warning(
                            "[PlanDebug] plan failed robot=%s arm=%s command=%s num_plan_failed=%d",
                            self.name,
                            self.lr_name,
                            self._last_command_name,
                            self.num_plan_failed,
                        )
                else:
                    succ = result.success.item()
                    if succ:
                        self._ee_trans = ee_trans
                        self._ee_ori = ee_ori
                        cmd_plan = result.get_interpolated_plan()
                        self.idx_list = list(range(len(self.raw_js_names)))
                        self.cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
                        self._write_curobo_plan_debug(
                            result=result,
                            sim_js=sim_js,
                            js_names=js_names,
                            ee_trans=ee_trans,
                            ee_ori=ee_ori,
                            raw_plan=cmd_plan,
                            ordered_cmd_plan=self.cmd_plan,
                            branch="single",
                            selected_path_index=0,
                            selected_path_source="result.get_interpolated_plan()",
                        )
                        self._visualize_selected_plan()
                        LOGGER.warning(
                            "[PhaseDebug] selected-plan robot=%s arm=%s phase=%s waypoints=%d stride=%d cached=false",
                            self.name,
                            self.lr_name,
                            self._last_command_name,
                            len(self.cmd_plan),
                            self.ds_ratio,
                        )
                        self.num_plan_failed = 0
                    else:
                        print("Plan did not converge to a solution.")
                        self._phase_plan_failed = True
                        self.num_plan_failed += 1
                        LOGGER.warning(
                            "[PlanDebug] plan failed robot=%s arm=%s command=%s num_plan_failed=%d",
                            self.name,
                            self.lr_name,
                            self._last_command_name,
                            self.num_plan_failed,
                        )
            if self.cmd_plan and self._step_idx % 1 == 0:
                cmd_state = self.cmd_plan[self.cmd_idx]
                arm_action = cmd_state.position.cpu().numpy()
                self._last_arm_action = np.asarray(arm_action, dtype=float).copy()
                art_action = ArticulationAction(
                    arm_action,
                    cmd_state.velocity.cpu().numpy() * 0.0,
                    joint_indices=self.idx_list,
                )
                self.cmd_idx += self.ds_ratio
                if self.cmd_idx >= len(self.cmd_plan):
                    LOGGER.warning(
                        "[PhaseDebug] plan-consumed robot=%s arm=%s phase=%s waypoints=%d stride=%d",
                        self.name,
                        self.lr_name,
                        self._last_command_name,
                        len(self.cmd_plan),
                        self.ds_ratio,
                    )
                    self.cmd_idx = 0
                    self.cmd_plan = None
                    self._phase_plan_finished = True
            else:
                self.num_last_cmd += 1
                if self._last_arm_action is None:
                    arm_action = sim_js.positions[self.arm_indices]
                else:
                    arm_action = self._last_arm_action
                art_action = ArticulationAction(joint_positions=arm_action)
        else:
            arm_action = np.asarray(sim_js.positions[self.arm_indices], dtype=float).copy()
            self._last_arm_action = arm_action.copy()
            art_action = ArticulationAction(joint_positions=arm_action)
            # Bookkeeping/update commands deliberately skip motion planning,
            # but Skills still need a finite completion signal.
            self.num_last_cmd += 1
        self._step_idx += 1
        arm_action = art_action.joint_positions
        self._last_commanded_arm_position = np.asarray(arm_action, dtype=float).copy()
        if gripper_action is None:
            gripper_action = self.get_gripper_action()
        else:
            gripper_action = np.asarray(gripper_action, dtype=float)
        joint_positions = np.concatenate([arm_action, gripper_action])
        self._action = {
            "joint_positions": joint_positions,
            "joint_indices": np.concatenate([self.arm_indices, self.gripper_indices]),
            "lr_name": self.lr_name,
            "arm_action": arm_action,
            "gripper_action": gripper_action,
        }
        return self._action

    def get_gripper_action(self):
        return np.clip(self._gripper_state * self._gripper_joint_position, 0.0, 0.04)

    def get_ee_pose(self):
        sim_js = self.robot.get_joints_state()
        q_state = torch.tensor(sim_js.positions[self.arm_indices], **self.tensor_args.as_torch_dict()).reshape(1, -1)
        ee_pose = self.kin_model.get_state(q_state)
        return ee_pose.ee_position[0].cpu().numpy(), ee_pose.ee_quaternion[0].cpu().numpy()

    def get_armbase_pose(self):
        armbase_pose = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        return pose_from_tf_matrix(armbase_pose)

    def forward_kinematic(self, q_state: np.ndarray):
        q_state = q_state.reshape(1, -1)
        q_state = self.tensor_args.to_device(q_state)
        out = self.kin_model.get_state(q_state)
        return out.ee_position[0].cpu().numpy(), out.ee_quaternion[0].cpu().numpy()

    def close_gripper(self):
        self._gripper_state = -1.0

    def open_gripper(self):
        self._gripper_state = 1.0

    def _get_curobo_world_object_names(self) -> List[str]:
        if self.motion_gen is not None:
            world_model = getattr(self.motion_gen, "world_model", None)
            objects = getattr(world_model, "objects", None)
            if objects is not None:
                return [obj.name for obj in objects]

        objects = getattr(self.world_cfg, "objects", None)
        if objects is not None:
            return [obj.name for obj in objects]
        return []

    @staticmethod
    def _select_attach_descendants(object_names: List[str]) -> List[str]:
        visual_names = [name for name in object_names if "/visual" in name or name.endswith("/visual")]
        if visual_names:
            return visual_names[:1]

        collision_names = [name for name in object_names if "/collisions/" in name or name.endswith("/collisions")]
        if collision_names:
            return collision_names[:1]

        return object_names

    def _resolve_attach_object_names(self, obj_prim_path) -> Tuple[List[str], List[str]]:
        requested_names = obj_prim_path if isinstance(obj_prim_path, (list, tuple)) else [obj_prim_path]
        world_object_names = self._get_curobo_world_object_names()
        world_object_set = set(world_object_names)

        resolved_names = []
        disabled_names = []
        for requested_name in requested_names:
            if requested_name in world_object_set:
                candidates = [requested_name]
                descendants = []
            else:
                prefix = requested_name.rstrip("/") + "/"
                descendants = [name for name in world_object_names if name.startswith(prefix)]
                candidates = self._select_attach_descendants(descendants)
                if not candidates:
                    candidates = [requested_name]

            for candidate in candidates:
                if candidate not in resolved_names:
                    resolved_names.append(candidate)
            for descendant in descendants:
                if descendant not in resolved_names and descendant not in disabled_names:
                    disabled_names.append(descendant)

        return resolved_names, disabled_names

    def attach_objects(
        self,
        obj_prim_paths: List[str],
        link_name="attached_object",
        world_objects_pose_offset=None,
    ):
        paths = [str(path) for path in obj_prim_paths]
        if not paths:
            raise ValueError("attach_objects requires at least one CuRobo obstacle path")
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )
        missing = [path for path in paths if self.motion_gen.world_model.get_obstacle(path) is None]
        if missing:
            raise ValueError(f"attach collision prims are not in CuRobo world: {missing}")
        LOGGER.warning("[AttachDebug] attaching robot=%s arm=%s paths=%s", self.name, self.lr_name, paths)
        attached = self.motion_gen.attach_objects_to_robot(
            cu_js,
            paths,
            link_name=link_name,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=world_objects_pose_offset,
        )
        LOGGER.warning("[AttachDebug] attached=%s disabled_world_obstacles=%s", attached, paths)
        return attached

    def attach_obj(self, obj_prim_path: str, link_name="attached_object"):
        """Attach a legacy object path while preserving descendant selection."""

        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )
        object_names, disabled_names = self._resolve_attach_object_names(obj_prim_path)
        legacy_offset = Pose.from_list(
            [0, 0, 0.01, 1, 0, 0, 0], self.tensor_args
        )
        attached = self.motion_gen.attach_objects_to_robot(
            cu_js,
            object_names,
            link_name=link_name,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=legacy_offset,
        )
        for object_name in disabled_names:
            self.motion_gen.world_coll_checker.enable_obstacle(enable=False, name=object_name)
        return attached

    def detach_obj(self):
        self.motion_gen.detach_object_from_robot()

    def has_attached_collision_spheres(self, link_name="attached_object") -> bool:
        spheres = (
            self.motion_gen.robot_cfg.kinematics.kinematics_config.get_link_spheres(
                link_name
            )
        )
        return bool(torch.any(spheres[:, 3] > 0.0).item())

    def update_specific(self, ignore_substring, reference_prim_path):
        if self.collision_world_mode == "physics_schema":
            warnings.warn(
                "update_specific(ignore_substring=...) is ignored in physics_schema mode; "
                "use CollisionSceneManager state transitions",
                DeprecationWarning,
                stacklevel=2,
            )
            self.collision_scene_manager.sync_dynamic_poses(
                self._step_idx, interval_steps=1, force=True
            )
            return
        self._legacy_update_specific(ignore_substring, reference_prim_path)

    def _legacy_update_specific(self, ignore_substring, reference_prim_path):
        """LEGACY_STAGE_SCAN: old substring-based per-command world rebuild."""

        # LEGACY_BEGIN: keyword-based collision world, retained for comparison
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=ignore_substring, reference_prim_path=reference_prim_path
        ).get_collision_check_world()
        self._update_world_if_changed(obstacles)
        # LEGACY_END

    def test_single_ik(self, ee_trans, ee_ori):
        assert not self.use_batch
        ik_goal = Pose(position=self.tensor_args.to_device(ee_trans), quaternion=self.tensor_args.to_device(ee_ori))
        result = self.ik_solver.solve_single(ik_goal)
        succ = result.success.item()
        if succ:  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False

    def test_batch_forward(self, ee_trans_batch_np, ee_ori_batch_np):
        ee_trans_batch = self.tensor_args.to_device(ee_trans_batch_np)
        ee_ori_batch = self.tensor_args.to_device(ee_ori_batch_np)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        result = self.plan_batch(ee_trans_batch, ee_ori_batch, sim_js, js_names)
        self._log_plan_result("test_batch_forward", result)

        return result

    def test_batch_forward_from_paths(self, ee_trans_batch_np, ee_ori_batch_np, start_paths):
        """Plan each terminal target from its matching pre-grasp endpoint."""

        ee_trans_batch = self.tensor_args.to_device(ee_trans_batch_np)
        ee_ori_batch = self.tensor_args.to_device(ee_ori_batch_np)
        starts = torch.stack([path.position[-1] for path in start_paths])
        zeros = torch.zeros_like(starts)
        start_state = JointState(
            position=starts,
            velocity=zeros,
            acceleration=zeros,
            jerk=zeros,
            joint_names=self.cmd_js_names,
        )
        goal = Pose(position=ee_trans_batch, quaternion=ee_ori_batch, batch=len(start_paths))
        result = self.motion_gen.plan_batch(start_state, goal, self.plan_config.clone())
        self._log_plan_result("test_batch_forward_from_pregrasp", result)
        return result

    def measure_cartesian_path(self, path, start_position, goal_position):
        """Return path/direct length ratio and maximum straight-line deviation."""

        positions = []
        for joint_position in path.position:
            point, _ = self.forward_kinematic(joint_position.detach().cpu().numpy())
            positions.append(point)
        positions = np.asarray(positions, dtype=float)
        if not len(positions):
            return float("inf"), float("inf")
        direct_vector = np.asarray(goal_position, dtype=float) - np.asarray(start_position, dtype=float)
        direct_length = float(np.linalg.norm(direct_vector))
        path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        if direct_length <= 1e-9:
            return (1.0 if path_length <= 1e-9 else float("inf")), path_length
        direction = direct_vector / direct_length
        relative = positions - np.asarray(start_position, dtype=float)
        projection = np.clip(relative @ direction, 0.0, direct_length)
        closest = np.asarray(start_position, dtype=float) + projection[:, None] * direction
        deviation = float(np.max(np.linalg.norm(positions - closest, axis=1)))
        return path_length / direct_length, deviation

    def test_single_forward(self, ee_trans: np.ndarray, ee_ori: np.ndarray):
        result = self.test_single_forward_result(ee_trans, ee_ori)
        succ = result.success.item()
        if succ:
            print("Success")
            return 1
        print("Plan did not converge to a solution.")
        return 0

    def test_forward_from_joint_positions(
        self,
        ee_trans: np.ndarray,
        ee_ori: np.ndarray,
        start_arm_positions: Optional[np.ndarray] = None,
    ):
        """Plan without changing runtime controller state and return the path endpoint."""
        assert ee_trans is not None and ee_ori is not None
        sim_js = self.robot.get_joints_state()
        sim_js.positions = np.asarray(sim_js.positions, dtype=float).copy()
        if start_arm_positions is not None:
            start_arm_positions = np.asarray(start_arm_positions, dtype=float).reshape(-1)
            if len(start_arm_positions) != len(self.arm_indices):
                raise ValueError(
                    "start_arm_positions must match the controller arm joint count: "
                    f"got {len(start_arm_positions)}, expected {len(self.arm_indices)}"
                )
            sim_js.positions[self.arm_indices] = start_arm_positions

        result = self.plan(ee_trans, ee_ori, sim_js, self.robot.dof_names)
        success_values = result.success.detach().cpu().numpy()
        if not bool(np.asarray(success_values).any()):
            return False, None, result

        if self.use_batch:
            paths = result.get_successful_paths()
            if not paths:
                return False, None, result
            position_filter_res = filter_paths_by_position_error(paths, result.position_error[result.success])
            rotation_filter_res = filter_paths_by_rotation_error(paths, result.rotation_error[result.success])
            filtered_paths = [
                path
                for path_index, path in enumerate(paths)
                if position_filter_res[path_index] and rotation_filter_res[path_index]
            ]
            if not filtered_paths:
                filtered_paths = paths
            sort_weights = self._get_sort_path_weights()  # pylint: disable=assignment-from-none
            weights_arg = self.tensor_args.to_device(sort_weights) if sort_weights is not None else None
            sorted_indices = sort_by_difference_js(filtered_paths, weights=weights_arg)
            cmd_plan = self.motion_gen.get_full_js(filtered_paths[sorted_indices[0]])
        else:
            cmd_plan = result.get_interpolated_plan()
        cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
        end_arm_positions = np.asarray(cmd_plan[-1].position.detach().cpu(), dtype=float)
        return True, end_arm_positions, result

    def test_single_forward_result(self, ee_trans: np.ndarray, ee_ori: np.ndarray):
        """Return the single-plan result so callers can reuse its endpoint.

        The legacy boolean wrapper above remains available.  Physics-schema
        Pick needs the actual pre-grasp path because its terminal plan must
        start from that path's final joint state, not from the live initial
        articulation state.
        """

        assert ee_trans is not None and ee_ori is not None
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        result = self.plan(ee_trans, ee_ori, sim_js, js_names)
        self._log_plan_result("test_single_forward", result, target=ee_trans)
        return result

    def test_single_forward_from_path(
        self,
        ee_trans: np.ndarray,
        ee_ori: np.ndarray,
        start_path,
    ):
        """Plan one terminal target from a successful pre-grasp endpoint."""

        start_position = start_path.position[-1]
        zeros = torch.zeros_like(start_position)
        start_state = JointState(
            position=start_position,
            velocity=zeros,
            acceleration=zeros,
            jerk=zeros,
            joint_names=self.cmd_js_names,
        )
        goal = Pose(
            position=self.tensor_args.to_device(ee_trans),
            quaternion=self.tensor_args.to_device(ee_ori),
        )
        result = self.motion_gen.plan_single(
            start_state.unsqueeze(0), goal, self.plan_config.clone()
        )
        self._log_plan_result(
            "test_single_forward_from_pregrasp", result, target=ee_trans
        )
        return result

    def pre_forward(self, ee_trans: np.ndarray, ee_ori: np.ndarray, expected_js=None, ds_ratio=1):
        assert ee_trans is not None and ee_ori is not None
        ee_trans = self.tensor_args.to_device(ee_trans)
        ee_ori = self.tensor_args.to_device(ee_ori)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        if expected_js is not None:
            sim_js.positions[self.arm_indices] = expected_js
        result = self.plan(ee_trans, ee_ori, sim_js, js_names)
        if self.use_batch:
            if result.success.any():
                print("Success")
                cmd_plans = result.get_successful_paths()
                cmd_plan = random.choice(cmd_plans)
                cmd_plan = self.motion_gen.get_full_js(cmd_plan)
                cmd_plan = cmd_plan.get_ordered_joint_state(self.raw_js_names)
                N = cmd_plan.shape[0]
                dt = self.motion_gen.interpolation_dt
                self.ds_ratio = ds_ratio
                cmd_time = N * dt / self.plan_config.time_dilation_factor / self.ds_ratio
                return cmd_time, np.array(cmd_plan[-1].position.cpu())
            print("Plan did not converge to a solution.")
            self.num_plan_failed = 1000
            return 0, expected_js
        succ = result.success.item()
        if succ:
            print("Success")
            cmd_plan = result.get_interpolated_plan()
            N = cmd_plan.shape[0]
            dt = self.motion_gen.interpolation_dt
            self.ds_ratio = ds_ratio
            cmd_time = N * dt / self.plan_config.time_dilation_factor / self.ds_ratio
            return cmd_time, np.array(cmd_plan[-1].position.cpu())
        print("Plan did not converge to a solution.")
        self.num_plan_failed = 1000
        return 0, expected_js

    def in_plane_rotation(self, target_rotate: np.ndarray):
        action = deepcopy(self._action)
        last_arm = len(self.arm_indices) - 1
        action["joint_positions"][last_arm] -= target_rotate
        action["arm_action"][last_arm] -= target_rotate
        return action

    def mobile_move(self, target: np.ndarray, joint_indices: np.ndarray = None, initial_position: np.ndarray = None):
        return {
            "joint_positions": initial_position + target,
            "joint_indices": np.array(joint_indices),
            "lr_name": "whole",
        }

    def dummy_forward(self, arm_action, gripper_state, *args, **kwargs):
        arm_action = np.asarray(arm_action, dtype=float).copy()
        self.cmd_plan = None
        self.cmd_idx = 0
        self._last_arm_action = arm_action.copy()
        if gripper_state == 1.0:
            self.open_gripper()
        elif gripper_state == -1.0:
            self.close_gripper()
        else:
            raise NotImplementedError
        gripper_action = self.get_gripper_action()
        return {
            "joint_positions": np.concatenate([arm_action, gripper_action]),
            "joint_indices": np.concatenate([self.arm_indices, self.gripper_indices]),
            "lr_name": self.lr_name,
            "arm_action": arm_action,
            "gripper_action": gripper_action,
        }
