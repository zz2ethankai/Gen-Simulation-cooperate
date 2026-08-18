"""
Template Controller base class for robot motion planning.

Common functionality extracted from FR3, FrankaRobotiq85, Genie1, Lift2, SplitAloha.
Subclasses implement _get_default_ignore_substring() and _configure_joint_indices().
"""

import json
import logging
import numbers
import os
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from core.utils.constants import CUROBO_BATCH_SIZE
from core.utils.plan_utils import (
    extract_result_paths,
)
from curobo.batch_motion_planner import BatchMotionPlanner
from curobo.config_io import join_path, load_yaml
from curobo.content import get_scene_configs_path
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.sphere_fit import SphereFitType
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.types import SceneCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.types.robot import RobotCfg
from curobo._src.util.usd_scene_parser import UsdSceneParser
from isaacsim.core.api import World
from isaacsim.core.api.controllers import BaseController
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.transformations import (
    get_relative_transform,
    pose_from_tf_matrix,
    tf_matrix_from_pose,
)
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.xforms import get_world_pose

from core.utils.joint_index_resolver import JointIndexResolutionError, resolve_joint_names
from core.utils.json_utils import json_ready, joint_state_json_ready
from core.planning.motion_command import MotionPhase, MotionPhaseCommand

LOGGER = logging.getLogger("de_logger")


def _resolve_native_robot_config(robot_file: str):
    """Normalize the SimBox robot YAML into the native v2 ``RobotCfg`` input."""

    robot_path = Path(robot_file).expanduser().resolve()
    raw = load_yaml(str(robot_path))
    if not isinstance(raw, dict):
        raise TypeError(f"robot config must be a mapping, got {type(raw)!r}")
    raw = deepcopy(raw.get("robot_cfg", raw))
    kinematics = deepcopy(raw.get("kinematics", raw))
    if not isinstance(kinematics, dict):
        raise TypeError("robot_cfg.kinematics must be a mapping")

    config_dir = robot_path.parent
    asset_root = config_dir.parents[1] / "assets" if len(config_dir.parents) > 1 else config_dir
    ee_link = kinematics.get("ee_link")
    if "tool_frames" not in kinematics and ee_link:
        kinematics["tool_frames"] = [ee_link]
    for legacy_key in (
        "use_usd_kinematics",
        "isaac_usd_path",
        "usd_path",
        "usd_robot_root",
        "usd_flip_joints",
        "usd_flip_joint_limits",
        "ee_link",
    ):
        kinematics.pop(legacy_key, None)

    def resolve_path(value):
        path = Path(value)
        if path.is_absolute():
            return str(path)
        candidate = asset_root / path
        if candidate.exists():
            return str(candidate)
        candidate = config_dir / path
        return str(candidate if candidate.exists() else asset_root / path)

    if "urdf_path" in kinematics:
        kinematics["urdf_path"] = resolve_path(kinematics["urdf_path"])
    if "asset_root_path" in kinematics:
        path = Path(kinematics["asset_root_path"])
        if not path.is_absolute():
            kinematics["asset_root_path"] = str(asset_root / path)
    spheres = kinematics.get("collision_spheres")
    if isinstance(spheres, str):
        sphere_path = Path(spheres)
        if not sphere_path.is_absolute():
            local_path = config_dir / sphere_path
            sphere_path = local_path if local_path.exists() else asset_root / sphere_path
        if sphere_path.exists():
            sphere_data = load_yaml(str(sphere_path)) or {}
            kinematics["collision_spheres"] = sphere_data.get(
                "collision_spheres", sphere_data
            )
    cspace = kinematics.get("cspace")
    if isinstance(cspace, dict):
        if "default_joint_position" not in cspace and "retract_config" in cspace:
            cspace["default_joint_position"] = cspace.pop("retract_config")
        else:
            cspace.pop("retract_config", None)
    return {"kinematics": kinematics}


def _record_attachment_rollback_failure(
    primary_error: Exception, operation: str, rollback_error: Exception
) -> None:
    """Attach rollback diagnostics without replacing the triggering error."""

    failures = list(
        getattr(primary_error, "_attachment_rollback_failures", ())
    )
    failures.append((operation, rollback_error))
    try:
        primary_error._attachment_rollback_failures = tuple(failures)
    except Exception:  # pragma: no cover - built-in exceptions allow attributes
        pass

    add_note = getattr(primary_error, "add_note", None)
    if callable(add_note):
        try:
            add_note(
                "Attachment rollback failed during "
                f"{operation}: {type(rollback_error).__name__}: {rollback_error}"
            )
        except Exception:  # pragma: no cover - diagnostics must not mask root cause
            pass


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
        # Candidate grasp planning is batched by default.  Individual tasks
        # can still pass ``use_batch=False`` when they explicitly need the
        # serial fallback path.
        use_batch: bool = True,
        trajectory_visualizer=None,
        skill_target_visualizer=None,
        collision_scene_manager=None,
        collision_world_mode: str = "legacy_stage_scan",
        timing_recorder=None,
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
        # Timing is explicitly bound by the workflow to the currently running
        # skill.  Do not resolve planner ownership through the recorder's
        # process/context-global active scope: DAG skills may run together.
        self.timing_recorder = timing_recorder
        self._timing_scope = None
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
        self.usd_parser = UsdSceneParser()
        self.tensor_args = DeviceCfg(
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            dtype=torch.float32,
        )
        self.planner = None
        self.batch_planner = None
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
        self._legacy_disabled_attach_names = []
        self._native_attached_obstacle_names = []
        # Candidate batching is a Place-time concern.  Keep its attachment
        # bookkeeping separate from the execution planner so Pick can attach
        # and execute with the single native planner without a second
        # AttachmentManager mutating the live planning state.
        self._native_batch_attached_obstacle_names = []
        # A native CuRobo reset clears SceneData, while the high-level world
        # and its signature remain valid.  Keep this separate from the
        # signature so physics-schema updates can reload once before pose
        # synchronization without forcing an update on every frame.
        self._world_cache_invalidated = False
        self._world_cleanup_failed = False
        self._world_cleanup_error = None
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
        # Pick planning/execution state belongs to the controller because it
        # tracks the world-frame target and arm-base frame used by CuRobo.
        # Keep it per object so a controller can safely service successive
        # pick skills without leaking a previous target's reference.
        self._pick_plan_references = {}
        self._pick_mobile_base_prim_path = getattr(self.robot, "mobile_base_prim_path", None)
        self._pick_cached_mobile_to_armbase_tf = None
        mount_prefix = "fr" if self.robot_file and "right" in self.robot_file else "fl"
        self._pick_configured_mobile_to_armbase_translation = np.array(
            self.robot.cfg.get(f"{mount_prefix}_base_mount_translation", []), dtype=np.float32
        )
        self._pick_configured_mobile_to_armbase_orientation = np.array(
            self.robot.cfg.get(
                f"{mount_prefix}_base_mount_orientation", [1.0, 0.0, 0.0, 0.0]
            ),
            dtype=np.float32,
        )

        self._configure_joint_indices(robot_file)
        self._resolve_runtime_control_indices()
        self._load_robot(robot_file)
        self._load_kin_model()
        self._load_world()
        self._init_native_planners()

        self.usd_parser.load_stage(self.world.stage)
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
            os.path.join("output", "local_navigation", "skills", "curobo_plan_debug"),
        )
        self._configure_execution_stride()
        LOGGER.info(
            "[ExecutionTiming] robot=%s arm=%s physics_dt=%.6f interpolation_dt=%.6f ds_ratio=%d",
            self.name,
            self.lr_name,
            float(self.world.get_physics_dt()),
            float(self.interpolation_dt),
            self.ds_ratio,
        )

    def bind_timing_scope(self, scope):
        """Bind planner telemetry to one concrete skill invocation."""

        self._timing_scope = scope
        return scope

    def push_timing_scope(self, scope):
        """Temporarily select the scope owning the current controller call."""

        previous = self._timing_scope
        self._timing_scope = scope
        return previous

    def restore_timing_scope(self, previous):
        """Restore the scope that was active before a controller call."""

        self._timing_scope = previous

    def clear_timing_scope(self, scope=None):
        """Clear a timing binding without disturbing a newer skill binding."""

        if scope is None or self._timing_scope is scope:
            self._timing_scope = None

    def bind_timing_recorder(self, timing_recorder=None, scope=None):
        """Optional compatibility binding interface for timing-aware callers."""

        self.timing_recorder = timing_recorder
        return self.bind_timing_scope(scope)

    def _start_curobo_timing_phase(self, operation: str):
        scope = getattr(self, "_timing_scope", None)
        if scope is None or not bool(getattr(scope, "_is_active", True)):
            return None
        try:
            phase = scope.curobo(
                f"curobo.{operation}",
                metadata={"controller": self.name, "arm": self.lr_name, "operation": operation},
            )
            return phase.start()
        except Exception:
            # Telemetry must never change planner behavior.
            LOGGER.debug("Failed to start CuRobo timing phase", exc_info=True)
            return None

    @staticmethod
    def _finish_curobo_timing_phase(phase, success: bool, error=None):
        if phase is None:
            return
        try:
            phase.finish(
                success=bool(success),
                reason=(str(error) if error is not None else None),
                error=error,
            )
        except Exception:
            LOGGER.debug("Failed to finish CuRobo timing phase", exc_info=True)

    def _synchronize_planner_cuda(self):
        device = getattr(getattr(self, "tensor_args", None), "device", None)
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)

    def _run_timed_curobo_call(self, operation: str, call):
        """Run one native CuRobo query with complete CUDA-synchronized timing."""

        scope = getattr(self, "_timing_scope", None)
        # Do not add synchronization fences when the workflow has not bound a
        # Skill timing scope.  Initialization, probes, and non-episode callers
        # must retain their original execution path and overhead.
        if scope is None or not bool(getattr(scope, "_is_active", True)):
            return call()
        # Flush work queued while preparing the query before opening the
        # measured span.  The planner span must not absorb a preceding physics
        # or world-update fence; the post-call fence is part of the complete
        # host-visible planner duration.
        self._synchronize_planner_cuda()
        phase = self._start_curobo_timing_phase(operation)
        if phase is None:
            try:
                return call()
            finally:
                # If telemetry itself failed, still drain the planner stream
                # so a diagnostic degradation cannot leave an async query
                # running past its caller.  Never mask the business result.
                try:
                    self._synchronize_planner_cuda()
                except Exception:
                    LOGGER.debug("Failed to drain unrecorded planner call", exc_info=True)
        try:
            # The synchronization belongs inside the measured phase so CUDA
            # graph/kernel work is included on both sides of the host call.
            result = call()
            self._synchronize_planner_cuda()
        except Exception as exc:
            self._finish_curobo_timing_phase(phase, False, exc)
            raise
        self._finish_curobo_timing_phase(phase, True)
        return result

    def _configure_execution_stride(self) -> None:
        if self.collision_world_mode == "physics_schema":
            physics_dt = float(self.world.get_physics_dt())
            interpolation_dt = float(self.interpolation_dt)
            requested_stride = max(1, int(round(physics_dt / interpolation_dt)))
            safety_cfg = self.task.cfg.get("planning", {}).get("execution_safety", {})
            max_stride = max(1, int(safety_cfg.get("max_waypoint_stride", 2)))
            self.ds_ratio = min(requested_stride, max_stride)
        else:
            # LEGACY_STAGE_SCAN keeps the original one-waypoint-per-step timing.
            self.ds_ratio = 1

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
        LOGGER.info(
            "[JointIndexAudit] controller=%s arm=%s joints=%s arm_indices=%s gripper_indices=%s",
            self.name,
            self.lr_name,
            list(self.cmd_js_names),
            resolved_arm_indices,
            resolved_gripper_indices,
        )

    def _load_robot(self, robot_file: str) -> None:
        self.robot_cfg = RobotCfg.create(
            _resolve_native_robot_config(robot_file),
            device_cfg=self.tensor_args,
            num_envs=1,
        )

    def _load_kin_model(self) -> None:
        # Use the same native RobotCfg as the planner; do not create a second
        # legacy/compatibility kinematics model.
        self.kin_model = Kinematics(self.robot_cfg.kinematics)

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
            self.world_cfg = SceneCfg()
        else:
            world_cfg_table = SceneCfg.create(
                load_yaml(join_path(get_scene_configs_path(), "collision_table.yml"))
            )
            self._world_cfg_table = world_cfg_table
            self._world_cfg_table.cuboid[0].pose[2] -= 10.5
            world_cfg1 = SceneCfg.create(
                load_yaml(join_path(get_scene_configs_path(), "collision_table.yml"))
            ).get_mesh_world()
            world_cfg1.mesh[0].name += "_mesh"
            world_cfg1.mesh[0].pose[2] = -10.5
            self.world_cfg = SceneCfg(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    def _get_native_collision_cache(self):
        """Override in subclasses to use different cache size (e.g. FR3 uses 1000)."""
        return {"cuboid": 700, "mesh": 700}

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
        status = getattr(result, "status", None)
        status = getattr(status, "value", status)
        valid_query = getattr(result, "valid_query", None)
        debug_info = getattr(result, "debug_info", None)
        msg = (
            f"[PlanDebug] {context} robot={self.name} arm={self.lr_name} command={self._last_command_name} "
            f"use_batch={self.use_batch} success_count={success_count} "
            f"status={status} valid_query={valid_query}"
        )
        if debug_info is not None:
            msg += f" debug_info={debug_info}"
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

    def _init_native_planners(self) -> None:
        # Keep the native-v2 execution defaults aligned with the v1 baseline.
        # Values below 1.0 are a deliberate task-level slowdown: they increase
        # execution time and must not be introduced implicitly by the
        # controller migration.
        pick_place_cfg = self.task.cfg.get("planning", {}).get("pick_place", {})
        configured_time_dilation = pick_place_cfg.get("time_dilation_factor", 1.0)
        try:
            manipulation_time_dilation = float(configured_time_dilation)
            if not np.isfinite(manipulation_time_dilation) or manipulation_time_dilation <= 0.0:
                raise ValueError
        except (TypeError, ValueError):
            manipulation_time_dilation = 1.0
        configured_interpolation_dt = pick_place_cfg.get("interpolation_dt", 0.01)
        try:
            native_interpolation_dt = float(configured_interpolation_dt)
            if not np.isfinite(native_interpolation_dt) or native_interpolation_dt <= 0.0:
                raise ValueError
        except (TypeError, ValueError):
            native_interpolation_dt = 0.01
        configured_plan_attempts = pick_place_cfg.get(
            "max_plan_attempts",
            4 if self.collision_world_mode == "physics_schema" else 10,
        )
        try:
            max_plan_attempts = max(1, int(configured_plan_attempts))
        except (TypeError, ValueError):
            max_plan_attempts = 4 if self.collision_world_mode == "physics_schema" else 10
        self._time_dilation_factor = manipulation_time_dilation
        self._max_plan_attempts = max_plan_attempts
        # Graph planning is an opt-in escape hatch.  The old MotionGen
        # defaults did not enter the 20k-node PRM for the normal four-attempt
        # pick/place query (the configured graph attempt was equal to the
        # attempt limit).  The first native-v2 port accidentally changed this
        # to attempts 2/4, making otherwise successful candidates pay for a
        # graph search and causing the 20--26 second spikes seen in logs.
        graph_enabled = bool(pick_place_cfg.get("enable_graph", False))
        self._graph_enabled = graph_enabled
        self._single_graph_attempt = (
            max(0, min(1, max_plan_attempts - 1))
            if graph_enabled
            else max_plan_attempts
        )
        configured_batch_attempts = pick_place_cfg.get(
            "batch_max_plan_attempts", min(max_plan_attempts, 4)
        )
        try:
            self._batch_max_attempts = max(1, int(configured_batch_attempts))
        except (TypeError, ValueError):
            self._batch_max_attempts = min(max_plan_attempts, 4)
        self._batch_graph_attempt = (
            max(0, min(3, self._batch_max_attempts - 1))
            if graph_enabled
            else self._batch_max_attempts
        )
        configured_warmup_iterations = pick_place_cfg.get("warmup_iterations", 1)
        try:
            native_warmup_iterations = max(1, int(configured_warmup_iterations))
        except (TypeError, ValueError):
            native_warmup_iterations = 1

        def make_config(robot_cfg, *, batch_size, trajopt_seeds):
            config = MotionPlannerCfg.create(
                robot=robot_cfg,
                device_cfg=self.tensor_args,
                collision_cache=self._get_native_collision_cache(),
                max_goalset=1,
                max_batch_size=batch_size,
                num_ik_seeds=20,
                num_trajopt_seeds=trajopt_seeds,
                position_tolerance=0.005,
                orientation_tolerance=0.05,
                optimizer_collision_activation_distance=self.collision_activation_distance,
                self_collision_check=True,
                use_cuda_graph=True,
            )
            # MotionPlannerCfg v2 exposes interpolation_dt on the solver
            # config rather than on the factory.  Set it before constructing
            # the planner so both native planners use the same 100 Hz output
            # contract as v1 (and physics-schema stride calculation sees it).
            config.trajopt_solver_config.interpolation_dt = native_interpolation_dt
            return config

        # Ordinary transit/place/home and single-goal pick execution always
        # use one native planning problem, even when candidate batching is
        # enabled for the grasp evaluator.
        self.planner = MotionPlanner(
            make_config(self.robot_cfg, batch_size=1, trajopt_seeds=12)
        )
        self.planner.update_world(self.world_cfg)
        self._set_native_pose_criteria()
        # CuRobo v2's warmup is a complete pose solve (and, for the batch
        # planner, includes the batched IK path).  Five iterations therefore
        # repeat expensive optimization work during every controller init;
        # v1 performed one warmup on its single MotionGen.  Keep one as the
        # native default and expose extra iterations only as an explicit task
        # setting for a deployment that has measured a benefit.
        self.planner.warmup(
            enable_graph=graph_enabled,
            num_warmup_iterations=native_warmup_iterations,
        )

        self.batch_planner = None
        if self.use_batch:
            batch_robot_cfg = RobotCfg.create(
                _resolve_native_robot_config(self.robot_file),
                device_cfg=self.tensor_args,
                num_envs=1,
            )
            self.batch_planner = BatchMotionPlanner(
                make_config(
                    batch_robot_cfg,
                    batch_size=CUROBO_BATCH_SIZE,
                    trajopt_seeds=1,
                )
            )
            self.batch_planner.update_world(self.world_cfg)
            self._set_native_pose_criteria(self.batch_planner)
            self.batch_planner.warmup(
                enable_graph=graph_enabled,
                num_warmup_iterations=native_warmup_iterations,
            )

        self.ik_solver = self.planner.ik_solver
        self.interpolation_dt = float(self.planner.trajopt_solver.config.interpolation_dt)
        self._world_update_signature = self._make_world_update_signature(self.world_cfg)
        LOGGER.info(
            "[PlanDebug] native v2 planner initialized robot=%s arm=%s use_batch=%s "
            "single_trajopt_seeds=%s batch_trajopt_seeds=%s graph_enabled=%s "
            "collision_activation_distance=%s",
            self.name,
            self.lr_name,
            self.use_batch,
            self.planner.trajopt_solver.config.num_seeds,
            self.batch_planner.trajopt_solver.config.num_seeds if self.batch_planner else None,
            graph_enabled,
            self.collision_activation_distance,
        )

    def _set_native_pose_criteria(self, planner=None, criteria=None) -> None:
        planner = planner or self.planner
        if planner is None:
            return
        if criteria is None:
            non_terminal = [0.0] * 6
            if self.constrain_grasp_approach:
                # Native axes are [x, y, z, roll, pitch, yaw].  The old
                # approach metric held every axis except the approach axis.
                non_terminal = [1.0] * 6
                non_terminal[self._get_grasp_approach_linear_axis()] = 0.0
            criteria = ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1.0] * 6,
                non_terminal_pose_axes_weight_factor=non_terminal,
                device_cfg=self.tensor_args,
            )
        planner.update_tool_pose_criteria(
            {frame: criteria.clone() for frame in planner.tool_frames}
        )

    def update_pose_cost_metric(self, hold_vec_weight: Optional[List[float]] = None) -> None:
        # reference: https://curobo.org/advanced_examples/3_constrained_planning.html
        # [angular-x, angular-y, angular-z, linear-x, linear-y, linear-z]
        # For example,
        # when hold_vec_weight is None, the corresponding list is [0, 0, 0, 0, 0, 0],
        # there is no cost added in any directions.
        # When hold_vec_weight = [1, 1, 1, 0, 0, 0], the tool orientation is holed.
        # assert hold_vec_weight is None or len(hold_vec_weight) == 6
        if hold_vec_weight is None:
            criteria = ToolPoseCriteria(device_cfg=self.tensor_args)
        else:
            if len(hold_vec_weight) != 6:
                raise ValueError("hold_vec_weight must contain six legacy [r, r, r, p, p, p] weights")
            # The controller's public contract is legacy [rx, ry, rz, px,
            # py, pz]; native ToolPoseCriteria uses [px, py, pz, rx, ry, rz].
            native_weights = [hold_vec_weight[index] for index in (3, 4, 5, 0, 1, 2)]
            criteria = ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1.0] * 6,
                non_terminal_pose_axes_weight_factor=native_weights,
                device_cfg=self.tensor_args,
            )
        self._set_native_pose_criteria(self.planner, criteria)
        if self.batch_planner is not None:
            self._set_native_pose_criteria(self.batch_planner, criteria)

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

            velocities, accelerations, jerks = self._joint_state_derivatives(sim_js)
            cu_js = JointState(
                position=self.tensor_args.to_device(sim_js.positions),
                velocity=self.tensor_args.to_device(velocities),
                acceleration=self.tensor_args.to_device(accelerations),
                jerk=self.tensor_args.to_device(jerks),
                joint_names=js_names,
            )
            cu_js_ordered = cu_js.reorder(self.cmd_js_names)

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
                    "selected_path_index": json_ready(selected_path_index),
                    "selected_path_source": selected_path_source,
                },
                "joint_mapping": {
                    "robot_dof_names": json_ready(js_names),
                    "cmd_js_names": json_ready(self.cmd_js_names),
                    "raw_js_names": json_ready(self.raw_js_names),
                    "arm_indices": json_ready(self.arm_indices),
                    "gripper_indices": json_ready(self.gripper_indices),
                    "idx_list": json_ready(self.idx_list),
                },
                "goal": {
                    "ee_translation": json_ready(ee_trans),
                    "ee_orientation": json_ready(ee_ori),
                },
                "input_current_state": {
                    "sim_js": joint_state_json_ready(sim_js),
                    "current_arm_sim_order": json_ready(current_arm),
                    "curobo_input_ordered_cmd_js_names": joint_state_json_ready(cu_js_ordered),
                },
                "result_summary": {
                    "success": json_ready(getattr(result, "success", None)),
                    "status": json_ready(getattr(result, "status", None)),
                    "valid_query": json_ready(getattr(result, "valid_query", None)),
                    "position_error": json_ready(getattr(result, "position_error", None)),
                    "rotation_error": json_ready(getattr(result, "rotation_error", None)),
                    "cspace_error": json_ready(getattr(result, "cspace_error", None)),
                    "optimized_dt": json_ready(getattr(result, "optimized_dt", None)),
                    "interpolation_dt": json_ready(getattr(result, "interpolation_dt", None)),
                    "path_buffer_last_tstep": json_ready(
                        getattr(result, "path_buffer_last_tstep", None)
                    ),
                    "used_graph": json_ready(getattr(result, "used_graph", None)),
                    "attempts": json_ready(getattr(result, "attempts", None)),
                    "trajopt_attempts": json_ready(getattr(result, "trajopt_attempts", None)),
                },
                "continuity": {
                    "first_ordered_minus_current_arm_norm": self._debug_norm_delta(first_ordered, current_arm),
                    "last_ordered_minus_current_arm_norm": self._debug_norm_delta(last_ordered, current_arm),
                    "first_ordered_position": json_ready(first_ordered),
                    "last_ordered_position": json_ready(last_ordered),
                },
                "raw_plan": joint_state_json_ready(raw_plan),
                "ordered_cmd_plan": joint_state_json_ready(ordered_cmd_plan),
            }
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(json_ready(payload), handle, indent=2, ensure_ascii=False)
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
    def _make_world_update_signature(cls, world_cfg: SceneCfg):
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

    def _update_world_if_changed(self, obstacles: SceneCfg) -> None:
        signature = self._make_world_update_signature(obstacles)
        needs_update = (
            signature != getattr(self, "_world_update_signature", None)
            or getattr(self, "_world_cache_invalidated", False)
        )
        if self.planner is not None and needs_update:
            # Native v2 updates SceneData in place and preserves captured CUDA
            # graph objects.  Do not call a legacy clear/reset sequence here.
            self.planner.update_world(obstacles)
            if self.batch_planner is not None:
                self.batch_planner.update_world(obstacles)
            self._world_cache_invalidated = False
        self.world_cfg = obstacles
        self._world_update_signature = signature

    def update(self) -> None:
        if getattr(self, "_world_cleanup_failed", False):
            error = getattr(self, "_world_cleanup_error", None)
            message = (
                "CuRobo world cleanup failed; refusing world update and "
                "dynamic-pose sync until reset cleanup succeeds"
            )
            if error is None:
                raise RuntimeError(message)
            raise RuntimeError(message) from error
        if self.collision_world_mode == "physics_schema":
            self.collision_scene_manager.sync_dynamic_poses(
                self._step_idx, interval_steps=1, force=False
            )
            self.collision_scene_manager.audit_controller(self)
            return
        self._legacy_update()

    def activate_collision_world_mode(self, mode: str) -> None:
        """Switch one controller between exact Physics and legacy Stage worlds."""

        target = str(mode).strip().lower()
        if target == "passthrough":
            return
        if target not in {"physics_schema", "legacy_stage_scan"}:
            raise ValueError(f"unsupported controller collision world mode: {mode!r}")

        manager = self.collision_scene_manager
        if target == "legacy_stage_scan" and manager is not None:
            manager.prepare_controller_for_legacy(self)
        if target == self.collision_world_mode:
            # This method is called from the workflow before every control
            # step.  Refreshing all exact obstacle poses here made a moving
            # reference update the complete 230-object world on every frame.
            # Planning paths call _refresh_reference_world_for_planning()
            # immediately before the actual CuRobo query instead.
            return

        self.clear_plan_and_hold()
        if target == "legacy_stage_scan":
            self.collision_world_mode = target
            self._legacy_update()
        else:
            if manager is None:
                raise RuntimeError("physics_schema mode requires CollisionSceneManager")
            if self.has_attached_collision_spheres():
                raise RuntimeError(
                    "legacy attached collision state cannot be transferred into physics_schema"
                )
            self.collision_world_mode = target
            obstacles = manager.build_world_config(self.reference_prim_path)
            self._update_world_if_changed(obstacles)
            manager.resume_controller_physics_world(self)
        self._configure_execution_stride()
        LOGGER.info(
            "[CollisionWorld] controller=%s/%s activated_mode=%s",
            self.name,
            self.lr_name,
            self.collision_world_mode,
        )

    def _legacy_update(self) -> None:
        """LEGACY_STAGE_SCAN: retained for explicit legacy_stage_scan mode."""

        # LEGACY_BEGIN: keyword-based collision world, retained for comparison
        obstacles = self.usd_parser.get_obstacles_from_stage(
            ignore_substring=self.ignore_substring, reference_prim_path=self.reference_prim_path
        ).get_collision_check_world()
        self._update_world_if_changed(obstacles)

    def _refresh_reference_world_for_planning(self) -> None:
        """Synchronize a moved mobile reference once before a CuRobo query."""

        if self.collision_world_mode != "physics_schema":
            return
        manager = self.collision_scene_manager
        if manager is not None:
            manager.refresh_controller_reference_world(self)
            apply_exclusions = getattr(manager, "apply_controller_planning_exclusions", None)
            if callable(apply_exclusions):
                apply_exclusions(self)

    def _attached_obstacle_names(self) -> tuple[str, ...]:
        return tuple(self._native_attached_obstacle_names)

    def _native_planners(self) -> tuple[Any, ...]:
        """Return the single and optional native-v2 candidate planners once each."""

        planners = []
        for attribute in ("planner", "batch_planner"):
            planner = getattr(self, attribute, None)
            if planner is not None and not any(planner is item for item in planners):
                planners.append(planner)
        return tuple(planners)

    def _reenable_legacy_disabled_attach_objects(
        self, already_enabled_names=()
    ) -> None:
        already_enabled = set(already_enabled_names)
        pending = [
            name
            for name in dict.fromkeys(self._legacy_disabled_attach_names)
            if name not in already_enabled
        ]
        for index, object_name in enumerate(pending):
            try:
                self.planner.scene_collision_checker.enable_obstacle(object_name, True)
            except Exception:
                # Successfully restored names do not need another attempt;
                # retain the failed name and remaining descendants for a
                # fail-closed retry, then preserve the original exception.
                self._legacy_disabled_attach_names = pending[index:]
                raise
        self._legacy_disabled_attach_names = []

    def _clear_attached_object_state(self) -> None:
        if self.planner is None:
            return
        self._world_cleanup_failed = True
        try:
            attached_names = self._attached_obstacle_names()
            for planner in self._native_planners():
                planner.attachment_manager.detach()
            self._native_attached_obstacle_names = []
            self._native_batch_attached_obstacle_names = []
            self._reenable_legacy_disabled_attach_objects(attached_names)
            # Native v2 detach only changes the robot attachment spheres and
            # obstacle enable masks.  It does not require a graph reset or a
            # full world reload.
            self._world_cache_invalidated = False
        except Exception as exc:
            self._world_cleanup_error = exc
            raise
        self._world_cleanup_failed = False
        self._world_cleanup_error = None
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

    @staticmethod
    def _joint_state_derivatives(sim_js):
        """Return finite joint velocity/acceleration/jerk arrays for planning.

        Isaac articulation states expose velocity consistently, while higher
        derivatives are optional across simulator versions.  Preserving the
        measured derivatives when available lets a new Pick/Place phase start
        with the actual motion state instead of an artificial full stop.
        """

        positions = sim_js.positions
        if hasattr(positions, "detach"):
            positions = positions.detach().cpu().numpy()
        positions = np.asarray(positions, dtype=float).reshape(-1)
        size = positions.size

        def _field(name):
            value = getattr(sim_js, name, None)
            if value is None:
                return np.zeros(size, dtype=float)
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=float).reshape(-1)
            if value.size != size or not np.all(np.isfinite(value)):
                return np.zeros(size, dtype=float)
            return value.copy()

        return _field("velocities"), _field("accelerations"), _field("jerks")

    def _planner_joint_names(self) -> list[str]:
        return list(self.planner.joint_names)

    def _arm_joint_state(self, sim_js, *, repeat=1):
        """Build a native-v2 named state in the planner's active-joint order."""

        positions = np.asarray(sim_js.positions, dtype=float).reshape(-1)
        velocities, accelerations, jerks = self._joint_state_derivatives(sim_js)
        arm_names = list(self.raw_js_names)
        if len(arm_names) != len(self.arm_indices):
            raise ValueError("raw arm joint names and runtime arm indices have different lengths")
        state = JointState(
            position=self.tensor_args.to_device(positions[self.arm_indices]),
            velocity=self.tensor_args.to_device(velocities[self.arm_indices]),
            acceleration=self.tensor_args.to_device(accelerations[self.arm_indices]),
            jerk=self.tensor_args.to_device(jerks[self.arm_indices]),
            joint_names=arm_names,
        ).reorder(self._planner_joint_names())
        if repeat > 1:
            state = JointState(
                position=state.position.unsqueeze(0).expand(repeat, -1).clone(),
                velocity=state.velocity.unsqueeze(0).expand(repeat, -1).clone(),
                acceleration=state.acceleration.unsqueeze(0).expand(repeat, -1).clone(),
                jerk=state.jerk.unsqueeze(0).expand(repeat, -1).clone(),
                joint_names=state.joint_names,
            )
        return state

    def _planner_state(self, state):
        names = getattr(state, "joint_names", None)
        if names is None:
            raise ValueError("native CuRobo planning states require explicit joint_names")
        names = list(names)
        if set(self._planner_joint_names()) - set(names):
            raise ValueError(
                "planning state does not contain all native planner joints: "
                f"required={self._planner_joint_names()}, got={names}"
            )
        return state.reorder(self._planner_joint_names())

    def _goal_tool_pose(self, ee_translation, ee_orientation, batch_size=1):
        position = self.tensor_args.to_device(ee_translation)
        quaternion = self.tensor_args.to_device(ee_orientation)
        if batch_size == 1:
            position = position.reshape(1, 1, 1, 1, 3)
            quaternion = quaternion.reshape(1, 1, 1, 1, 4)
        else:
            position = position.reshape(batch_size, 1, 1, 1, 3)
            quaternion = quaternion.reshape(batch_size, 1, 1, 1, 4)
        return GoalToolPose(
            tool_frames=[self.planner.tool_frames[0]],
            position=position,
            quaternion=quaternion,
        )

    @staticmethod
    def _result_success(result) -> bool:
        success = getattr(result, "success", None)
        return success is not None and bool(torch.any(success).item())

    def _result_path(self, result, batch_index=0):
        paths = extract_result_paths(result)
        if batch_index >= len(paths) or paths[batch_index] is None:
            return None
        return paths[batch_index]

    def _command_path(self, path):
        """Convert a native result path to the controller's seven arm joints."""

        if path is None:
            return None
        names = list(getattr(path, "joint_names", ()) or ())
        if set(self.raw_js_names).issubset(names):
            return path.reorder(self.raw_js_names)
        active = self._planner_state(path)
        full = self.planner.kinematics.get_full_js(active)
        if not set(self.raw_js_names).issubset(full.joint_names):
            raise ValueError(
                "native planner result cannot be mapped to controller arm names: "
                f"result={full.joint_names}, arm={self.raw_js_names}"
            )
        return full.reorder(self.raw_js_names)

    def _install_command_plan(
        self,
        cmd_plan,
        *,
        target_position=None,
        target_orientation=None,
        phase_name: str = "unknown",
        cached: bool,
    ):
        """Install one normalized trajectory for either planning or replay."""

        if cmd_plan is None or len(cmd_plan) == 0:
            raise ValueError(f"{phase_name} received an empty native-v2 path")
        self.idx_list = list(range(len(self.raw_js_names)))
        self.cmd_plan = cmd_plan
        self.cmd_idx = 0
        self._phase_plan_started = True
        if target_position is not None:
            self._ee_trans = self.tensor_args.to_device(target_position)
        if target_orientation is not None:
            self._ee_ori = self.tensor_args.to_device(target_orientation)
        self._visualize_selected_plan()
        LOGGER.info(
            "[PhaseDebug] selected-plan robot=%s arm=%s phase=%s waypoints=%d stride=%d cached=%s",
            self.name,
            self.lr_name,
            phase_name,
            len(self.cmd_plan),
            self.ds_ratio,
            cached,
        )
        return cmd_plan

    def _plan_pose_from_state(
        self,
        ee_translation,
        ee_orientation,
        start_state,
        *,
        context: Optional[str] = None,
    ):
        goal = self._goal_tool_pose(ee_translation, ee_orientation)
        result = self._run_timed_curobo_call(
            "plan_pose",
            lambda: self.planner.plan_pose(
                goal,
                start_state.unsqueeze(0),
                use_implicit_goal=True,
                max_attempts=self._max_plan_attempts,
                enable_graph_attempt=self._single_graph_attempt,
            ),
        )
        if context:
            self._log_plan_result(context, result, target=ee_translation)
        return result

    def _plan_batch_from_state(
        self,
        ee_translation,
        ee_orientation,
        start_state,
        *,
        batch_size: Optional[int] = None,
        context: Optional[str] = None,
    ):
        if self.batch_planner is None:
            raise RuntimeError("batch planning was not enabled for this controller")
        if batch_size is None:
            batch_size = (
                1
                if getattr(start_state.position, "ndim", 1) == 1
                else int(start_state.position.shape[0])
            )
        goal = self._goal_tool_pose(
            ee_translation, ee_orientation, batch_size=batch_size
        )
        result = self._run_timed_curobo_call(
            "plan_batch",
            lambda: self.batch_planner.plan_pose(
                goal,
                start_state,
                use_implicit_goal=True,
                max_attempts=self._batch_max_attempts,
                enable_graph_attempt=self._batch_graph_attempt,
            ),
        )
        if context:
            self._log_plan_result(context, result)
        return result

    def plan_batch(self, ee_translation_goal_batch, ee_orientation_goal_batch, sim_js, js_names):
        # Batch planning is also a real CuRobo query.  Keep it consistent with
        # plan(): a mobile-base reference must be synchronized after
        # navigation and immediately before the captured-graph query.
        self._refresh_reference_world_for_planning()
        if self.batch_planner is None:
            raise RuntimeError("batch planning was not enabled for this controller")
        batch_size = int(self.batch_planner.batch_size)
        # Native v2 keeps goal tensors on the planner device.  Normalize only
        # this host-side validation boundary before checking batch length;
        # ``np.asarray(cuda_tensor)`` cannot perform the device transfer.
        if isinstance(ee_translation_goal_batch, torch.Tensor):
            ee_translation_goal_batch = ee_translation_goal_batch.detach().cpu().numpy()
        else:
            ee_translation_goal_batch = np.asarray(ee_translation_goal_batch)
        if isinstance(ee_orientation_goal_batch, torch.Tensor):
            ee_orientation_goal_batch = ee_orientation_goal_batch.detach().cpu().numpy()
        else:
            ee_orientation_goal_batch = np.asarray(ee_orientation_goal_batch)
        actual_batch_size = len(ee_translation_goal_batch)
        if actual_batch_size < 1 or actual_batch_size > batch_size or len(ee_orientation_goal_batch) != actual_batch_size:
            raise ValueError(
                f"native batch planner accepts 1..{batch_size} candidate goals"
            )
        cu_js = self._arm_joint_state(sim_js, repeat=actual_batch_size)
        return self._plan_batch_from_state(
            ee_translation_goal_batch,
            ee_orientation_goal_batch,
            cu_js,
            batch_size=actual_batch_size,
        )

    def plan(self, ee_translation_goal, ee_orientation_goal, sim_js: JointState, js_names: list):
        self._refresh_reference_world_for_planning()
        cu_js = self._arm_joint_state(sim_js)
        return self._plan_pose_from_state(
            ee_translation_goal, ee_orientation_goal, cu_js
        )

    def plan_joint_positions(self, goal_arm_positions: np.ndarray):
        """Plan to an exact arm configuration in the active collision world."""

        refresh = getattr(self, "_refresh_reference_world_for_planning", None)
        if callable(refresh):
            refresh()
        goal_arm_positions = np.asarray(goal_arm_positions, dtype=float).reshape(-1)
        if len(goal_arm_positions) != len(self.arm_indices):
            raise ValueError(
                "goal_arm_positions must match the controller arm joint count: "
                f"got {len(goal_arm_positions)}, expected {len(self.arm_indices)}"
            )
        sim_js = self.robot.get_joints_state()
        start_state = self._arm_joint_state(sim_js)
        goal_positions = goal_arm_positions
        zeros = np.zeros_like(goal_positions)
        goal_state = JointState(
            position=self.tensor_args.to_device(goal_positions),
            velocity=self.tensor_args.to_device(zeros),
            acceleration=self.tensor_args.to_device(zeros),
            jerk=self.tensor_args.to_device(zeros),
            joint_names=self.raw_js_names,
        )
        planner_names_fn = getattr(self, "_planner_joint_names", None)
        planner_names = (
            list(planner_names_fn())
            if callable(planner_names_fn)
            else list(getattr(self.planner, "joint_names", self.raw_js_names))
        )
        goal_state = goal_state.reorder(planner_names)
        result = self._run_timed_curobo_call(
            "plan_cspace",
            lambda: self.planner.plan_cspace(
                goal_state.unsqueeze(0),
                start_state.unsqueeze(0),
                max_attempts=self._max_plan_attempts,
                enable_graph_attempt=self._single_graph_attempt,
            ),
        )
        self._log_plan_result("plan_joint_positions", result, goal_arm_positions)
        return result

    def check_current_start_state(self):
        """Validate the live articulation state against the active planning world."""

        sim_js = self.robot.get_joints_state()
        velocities, accelerations, jerks = self._joint_state_derivatives(sim_js)
        start_state = self._arm_joint_state(sim_js)
        limits = self.planner.kinematics.get_joint_limits()
        position = start_state.position
        valid = bool(
            torch.isfinite(position).all().item()
            and (position >= limits.position_lower_limits).all().item()
            and (position <= limits.position_upper_limits).all().item()
        )
        return valid, "valid" if valid else "joint_limit_or_non_finite"

    def diagnose_native_start_collision(self) -> dict:
        """Report native-v2 scene collision at the live arm state.

        This is a failure-path diagnostic only.  It deliberately runs after a
        caller has exhausted its planning candidates, so the temporary
        one-step FK query cannot invalidate a subsequent CUDA-graph query.
        The query uses the same attached-object spheres and scene checker as
        the native planner; it never disables or changes a world obstacle.
        """

        try:
            from curobo._src.geom.collision import CollisionBuffer

            sim_js = self.robot.get_joints_state()
            active_state = self._arm_joint_state(sim_js)
            fk_state = self.planner.compute_kinematics(
                JointState.from_position(
                    active_state.position.unsqueeze(0).unsqueeze(0),
                    joint_names=self._planner_joint_names(),
                )
            )
            if fk_state.robot_spheres is None:
                return {"available": False, "reason": "native_fk_has_no_robot_spheres"}

            scene = self.planner.scene_collision_checker
            buffer = CollisionBuffer.from_shape(
                fk_state.robot_spheres.shape,
                self.tensor_args,
            )
            weight = self.tensor_args.to_device([1.0])
            activation_distance = self.tensor_args.to_device([0.0])
            distances = scene.get_sphere_distance(
                fk_state,
                buffer,
                weight,
                activation_distance,
            )
            distance_cpu = distances.detach().float().cpu().numpy()[0, 0]
            spheres_cpu = fk_state.robot_spheres.detach().float().cpu().numpy()[0, 0]
            collision_mask = distance_cpu > 1e-8

            kinematics_cfg = self.planner.kinematics.config.kinematics_config
            attached_indices = np.asarray([], dtype=int)
            if "attached_object" in (kinematics_cfg.link_name_to_idx_map or {}):
                attached_indices = (
                    kinematics_cfg.get_sphere_index_from_link_name("attached_object")
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(int)
                )
            attached_mask = np.zeros(len(distance_cpu), dtype=bool)
            attached_mask[attached_indices] = True

            def _sphere_rows(mask):
                indices = np.flatnonzero(mask)
                order = indices[np.argsort(distance_cpu[indices])[::-1]]
                return [
                    {
                        "sphere_index": int(index),
                        "collision_cost": float(distance_cpu[index]),
                        "center": spheres_cpu[index, :3].tolist(),
                        "radius": float(spheres_cpu[index, 3]),
                    }
                    for index in order[:8]
                ]

            summary = {
                "available": True,
                "attached_obstacle_names": list(self._native_attached_obstacle_names),
                "num_spheres": int(len(distance_cpu)),
                "collision_cost_sum": float(np.sum(distance_cpu)),
                "collision_cost_max": float(np.max(distance_cpu)) if len(distance_cpu) else 0.0,
                "collision_sphere_count": int(np.count_nonzero(collision_mask)),
                "attached_sphere_indices": attached_indices.tolist(),
                "attached_collision_cost_sum": float(np.sum(distance_cpu[attached_mask])),
                "attached_collision_cost_max": (
                    float(np.max(distance_cpu[attached_mask]))
                    if np.any(attached_mask)
                    else 0.0
                ),
                "colliding_spheres": _sphere_rows(collision_mask),
                "colliding_attached_spheres": _sphere_rows(collision_mask & attached_mask),
            }
            LOGGER.warning(
                "[NativeCollisionDebug] robot=%s arm=%s attached=%s "
                "collision_cost_sum=%.6f collision_cost_max=%.6f "
                "collision_spheres=%d attached_cost_max=%.6f",
                self.name,
                self.lr_name,
                summary["attached_obstacle_names"],
                summary["collision_cost_sum"],
                summary["collision_cost_max"],
                summary["collision_sphere_count"],
                summary["attached_collision_cost_max"],
            )
            return summary
        except Exception as exc:  # pragma: no cover - diagnostic must not mask planning failure
            LOGGER.warning(
                "[NativeCollisionDebug] unavailable robot=%s arm=%s error=%r",
                self.name,
                self.lr_name,
                exc,
            )
            return {"available": False, "reason": repr(exc)}

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
        LOGGER.info(
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

    def _install_preplanned_phase_path(self, command: MotionPhaseCommand):
        """Install a named native-v2 path for a phase without replanning it.

        Pick and Place candidate validation may run against a planner that has
        already solved the exact attached-object world.  Re-running the
        single planner at execution time can therefore reject a path that was
        just validated (and can also choose a different collision branch).
        Keep the same explicit named-joint normalization as the ordinary
        planning path, then let ``ee_forward`` consume the installed path by
        setting its phase target to the command target.
        """

        preplanned_path = command.params.get("preplanned_joint_path")
        if preplanned_path is None:
            return False
        cmd_plan = self._command_path(preplanned_path)
        self._install_command_plan(
            cmd_plan,
            target_position=command.target_position,
            target_orientation=command.target_orientation,
            phase_name=command.phase.value,
            cached=True,
        )
        return True

    @staticmethod
    def _pick_terminal_samples(start, goal, step_m: float) -> list[np.ndarray]:
        """Discretize a terminal grasp approach for controllers without a cached path."""

        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        distance = float(np.linalg.norm(goal - start))
        count = max(1, int(np.ceil(distance / float(step_m))))
        return [start + (goal - start) * (index / count) for index in range(1, count + 1)]

    def build_pick_phase_commands(
        self,
        *,
        object_name: str,
        pregrasp_position,
        pregrasp_orientation,
        grasp_position,
        grasp_orientation,
        gripper_action: str,
        post_grasp_offset: float = 0.0,
        source_support=None,
        terminal_path=None,
        terminal_path_length_ratio=None,
        terminal_path_max_deviation_m=None,
        return_to_pregrasp: bool = False,
        completion_tolerance: Optional[dict] = None,
        terminal_step_m: Optional[float] = None,
        gripper_change_steps: int = 40,
        contact_threshold_n: float = 0.0,
        verify_grasp_contact=None,
    ) -> list[MotionPhaseCommand]:
        """Build the executable Physics-schema Pick sequence.

        Pick owns grasp annotation sampling and candidate selection.  Once a
        candidate is selected, the controller owns the execution protocol:
        world synchronization, pre-grasp motion, terminal approach, gripper
        close, attachment, and post-grasp lift.  Keeping this policy here
        makes the skill independent of the low-level Physics phase details
        while retaining the structured command interface used by the
        workflow.
        """

        if verify_grasp_contact is None or not callable(verify_grasp_contact):
            raise ValueError("Physics Pick execution requires verify_grasp_contact callback")
        if terminal_step_m is None:
            terminal_step_m = float(
                self.task.cfg.get("planning", {})
                .get("pick_place", {})
                .get("terminal_step_m", 0.005)
            )
        terminal_step_m = max(float(terminal_step_m), 1e-6)
        tolerance = dict(
            completion_tolerance
            or {
                "position_m": 0.005,
                "orientation_rad": 0.05,
            }
        )
        if source_support is None and self.collision_scene_manager is not None:
            source_support = self.get_pick_source_support(object_name)

        pregrasp_position = np.asarray(pregrasp_position, dtype=float)
        pregrasp_orientation = np.asarray(pregrasp_orientation, dtype=float)
        grasp_position = np.asarray(grasp_position, dtype=float)
        grasp_orientation = np.asarray(grasp_orientation, dtype=float)
        commands = [
            MotionPhaseCommand(
                MotionPhase.SYNC_WORLD,
                active_object=object_name,
                replan_allowed=False,
            ),
            MotionPhaseCommand(
                MotionPhase.TRANSIT_PREGRASP,
                pregrasp_position,
                pregrasp_orientation,
                gripper_action="open_gripper",
                active_object=object_name,
                completion_tolerance=tolerance,
            ),
        ]

        if terminal_path is not None:
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.TERMINAL_GRASP_APPROACH,
                    grasp_position,
                    grasp_orientation,
                    gripper_action="open_gripper",
                    active_object=object_name,
                    allow_target_finger_contact=True,
                    completion_tolerance={
                        "position_m": terminal_step_m,
                        "orientation_rad": tolerance["orientation_rad"],
                    },
                    params={
                        "preplanned_joint_path": terminal_path,
                        "cartesian_step_m": terminal_step_m,
                        "path_length_ratio": terminal_path_length_ratio,
                        "path_max_deviation_m": terminal_path_max_deviation_m,
                    },
                )
            )
        else:
            # Lightweight controllers may not return a chained pre-grasp ->
            # grasp path.  Keep their fallback behavior in the controller so
            # the skill does not need to know how terminal motion is sampled.
            terminal_points = self._pick_terminal_samples(
                pregrasp_position, grasp_position, terminal_step_m
            )
            for point_index, point in enumerate(terminal_points):
                ratio = (point_index + 1) / len(terminal_points)
                quat = (1.0 - ratio) * pregrasp_orientation + ratio * grasp_orientation
                quat = quat / np.linalg.norm(quat)
                commands.append(
                    MotionPhaseCommand(
                        MotionPhase.TERMINAL_GRASP_APPROACH,
                        point,
                        quat,
                        gripper_action="open_gripper",
                        active_object=object_name,
                        allow_target_finger_contact=True,
                        completion_tolerance={
                            "position_m": terminal_step_m,
                            "orientation_rad": tolerance["orientation_rad"],
                        },
                    )
                )

        commands.extend(
            [
                MotionPhaseCommand(
                    MotionPhase.GRIPPER_CLOSE,
                    grasp_position,
                    grasp_orientation,
                    gripper_action=gripper_action,
                    active_object=object_name,
                    allow_target_finger_contact=True,
                    replan_allowed=False,
                    dwell_steps=int(gripper_change_steps),
                    params={"contact_threshold_n": float(contact_threshold_n)},
                ),
                MotionPhaseCommand(
                    MotionPhase.ATTACH,
                    active_object=object_name,
                    allow_target_finger_contact=True,
                    replan_allowed=False,
                    params={"verify_grasp_contact": verify_grasp_contact},
                ),
            ]
        )

        def append_lift(target_position, target_orientation):
            commands.append(
                MotionPhaseCommand(
                    MotionPhase.POST_GRASP_LIFT,
                    target_position,
                    target_orientation,
                    gripper_action=gripper_action,
                    active_object=object_name,
                    support_object=source_support,
                    allow_target_finger_contact=True,
                    # Native attachment leaves the physical target in the
                    # gripper; permit the attached target to touch robot links
                    # while the explicit target collision gate remains active.
                    allow_target_robot_contact=True,
                    allow_object_support_contact=source_support is not None,
                    completion_tolerance=tolerance,
                )
            )

        if post_grasp_offset:
            post_position = grasp_position.copy()
            post_position[2] += float(post_grasp_offset)
            append_lift(post_position, grasp_orientation)
        if return_to_pregrasp:
            append_lift(pregrasp_position, pregrasp_orientation)
        return commands

    def _require_pick_collision_manager(self):
        if self.collision_scene_manager is None:
            raise RuntimeError("Physics-schema Pick requires CollisionSceneManager")
        return self.collision_scene_manager

    def prepare_pick_planning_world(self, object_name: str):
        """Prepare the exact Physics world used by Pick candidate evaluation."""

        manager = self._require_pick_collision_manager()
        self.update_pose_cost_metric(None)
        manager.sync_dynamic_poses(0, interval_steps=1, force=True)
        manager.begin_target_transit(object_name, self.name, self.lr_name)

    def prepare_pick_pregrasp_world(self, object_name: str):
        """Restore the collision state used while validating pre-grasp paths."""

        self._require_pick_collision_manager().begin_target_transit(
            object_name, self.name, self.lr_name
        )

    def prepare_pick_grasp_world(self, object_name: str):
        """Switch the exact target collision state used for terminal approach."""

        self._require_pick_collision_manager().begin_target_approach(
            object_name, self.name, self.lr_name
        )

    def restore_pick_world(self, object_name: str):
        """Restore the complete world after Pick planning or a failed contact check."""

        self._require_pick_collision_manager().restore_world(object_name)

    def diagnose_pick_start_world_collision(self):
        """Return a diagnostic for an initial Physics Pick collision, if available."""

        return self._require_pick_collision_manager().diagnose_controller_world_collision(self)

    def get_pick_source_support(self, object_name: str):
        """Resolve the support entity that may remain in contact during a lift."""

        return self._require_pick_collision_manager().get_source_support_entity(object_name)

    def get_pick_armbase_transform(self):
        """Return the current world transform of this controller's arm base."""

        armbase_tf_getter = getattr(self.robot, "get_armbase_world_transform", None)
        if callable(armbase_tf_getter):
            return armbase_tf_getter()

        reference_prim_path = str(getattr(self, "reference_prim_path", "")).strip()
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
                            self._pick_cached_mobile_to_armbase_tf = (
                                np.linalg.inv(world_mobile) @ world_armbase
                            )
                        except Exception:
                            pass
                    return world_armbase
                except Exception:
                    pass

        if self._pick_configured_mobile_to_armbase_translation.shape == (3,):
            if hasattr(self.robot, "get_mobile_base_pose"):
                mobile_base_t, mobile_base_q = self.robot.get_mobile_base_pose()
            else:
                mobile_base_t, mobile_base_q = self.robot.get_world_pose()
            world_mobile = tf_matrix_from_pose(mobile_base_t, mobile_base_q)
            mobile_to_armbase = tf_matrix_from_pose(
                self._pick_configured_mobile_to_armbase_translation,
                self._pick_configured_mobile_to_armbase_orientation,
            )
            self._pick_cached_mobile_to_armbase_tf = mobile_to_armbase
            return world_mobile @ mobile_to_armbase

        reference_prim = get_prim_at_path(self.reference_prim_path)
        task_prim = get_prim_at_path(self.task.root_prim_path)
        raw_task_armbase = get_relative_transform(reference_prim, task_prim)
        mobile_base_prim_path = str(self._pick_mobile_base_prim_path or "").strip()
        if not mobile_base_prim_path:
            return raw_task_armbase

        mobile_base_prim = get_prim_at_path(mobile_base_prim_path)
        if not mobile_base_prim.IsValid():
            return raw_task_armbase
        task_mobile = get_relative_transform(mobile_base_prim, task_prim)
        if self._pick_cached_mobile_to_armbase_tf is None:
            self._pick_cached_mobile_to_armbase_tf = (
                np.linalg.inv(task_mobile) @ raw_task_armbase
            )
        return task_mobile @ self._pick_cached_mobile_to_armbase_tf

    def get_pick_frame_debug(self):
        """Expose frame-resolution details for Pick's non-critical diagnostics."""

        return {
            "mobile_base_prim_path": self._pick_mobile_base_prim_path,
            "cached_mobile_to_armbase_tf": self._pick_cached_mobile_to_armbase_tf,
            "configured_mobile_to_armbase_translation": self._pick_configured_mobile_to_armbase_translation,
            "configured_mobile_to_armbase_orientation": self._pick_configured_mobile_to_armbase_orientation,
        }

    def _get_pick_object_world_pose(self, object_name: str):
        pick_object = self.task.objects[object_name]
        get_world_pose_fn = getattr(pick_object, "get_world_pose", None)
        if callable(get_world_pose_fn):
            return get_world_pose_fn()
        return pick_object.get_local_pose()

    def capture_pick_plan_reference(self, object_name: str):
        """Capture the object and arm-base frames used by a Physics Pick plan."""

        object_translation, object_orientation = self._get_pick_object_world_pose(object_name)
        self._pick_plan_references[object_name] = {
            "object_pose": (
                np.asarray(object_translation, dtype=float).reshape(3).copy(),
                np.asarray(object_orientation, dtype=float).reshape(4).copy(),
            ),
            "world_armbase_tf": np.asarray(self.get_pick_armbase_transform(), dtype=float).copy(),
        }

    def retarget_pick_phase_commands(self, object_name: str, commands):
        """Retarget pending Physics Pick phases after rigid target motion."""

        reference = self._pick_plan_references.get(object_name)
        if reference is None:
            self.capture_pick_plan_reference(object_name)
            return 0.0, 0.0

        current_translation, current_orientation = self._get_pick_object_world_pose(object_name)
        current_object_pose = (
            np.asarray(current_translation, dtype=float).reshape(3).copy(),
            np.asarray(current_orientation, dtype=float).reshape(4).copy(),
        )
        current_world_armbase_tf = np.asarray(self.get_pick_armbase_transform(), dtype=float)
        old_object_tf = tf_matrix_from_pose(*reference["object_pose"])
        current_object_tf = tf_matrix_from_pose(*current_object_pose)
        object_delta = current_object_tf @ np.linalg.inv(old_object_tf)
        relative_rotation = object_delta[:3, :3]
        cosine = float(
            np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
        )
        rotation_delta_deg = float(np.degrees(np.arccos(cosine)))
        translation_delta = float(np.linalg.norm(object_delta[:3, 3]))

        current_base_inverse = np.linalg.inv(current_world_armbase_tf)
        for pending in commands:
            if not isinstance(pending, MotionPhaseCommand) or pending.target_position is None:
                continue
            old_base_ee_tf = tf_matrix_from_pose(
                pending.target_position, pending.target_orientation
            )
            old_world_ee_tf = reference["world_armbase_tf"] @ old_base_ee_tf
            current_world_ee_tf = object_delta @ old_world_ee_tf
            current_base_ee_tf = current_base_inverse @ current_world_ee_tf
            target_position, target_orientation = pose_from_tf_matrix(current_base_ee_tf)
            pending.target_position = np.asarray(target_position, dtype=float).reshape(3)
            pending.target_orientation = np.asarray(target_orientation, dtype=float).reshape(4)
            if pending.phase == MotionPhase.TERMINAL_GRASP_APPROACH:
                pending.params.pop("preplanned_joint_path", None)
                pending.params.pop("path_length_ratio", None)
                pending.params.pop("path_max_deviation_m", None)

        self._pick_plan_references[object_name] = {
            "object_pose": current_object_pose,
            "world_armbase_tf": current_world_armbase_tf.copy(),
        }
        return translation_delta, rotation_delta_deg

    def replan_pick_after_safety(self, object_name: str, command, commands):
        """Retarget remaining Pick phases when the active object moved."""

        if not isinstance(command, MotionPhaseCommand):
            return True
        if command.active_object != object_name:
            return True
        if command.phase not in {
            MotionPhase.TRANSIT_PREGRASP,
            MotionPhase.TERMINAL_GRASP_APPROACH,
        }:
            return True
        try:
            translation_delta, rotation_delta_deg = self.retarget_pick_phase_commands(
                object_name, commands
            )
        except Exception:
            LOGGER.exception(
                "[PickSafety] failed to retarget moving object=%s phase=%s",
                object_name,
                command.phase.value,
            )
            return False
        LOGGER.warning(
            "[PickSafety] retargeted active object=%s phase=%s "
            "translation_delta_m=%.6f rotation_delta_deg=%.3f "
            "terminal_path_invalidated=true",
            object_name,
            command.phase.value,
            translation_delta,
            rotation_delta_deg,
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
                    self._install_preplanned_phase_path(command)
            elif command.phase == MotionPhase.TRANSIT_PREPLACE:
                manager.assert_attached_owner(command.active_object, robot, arm)
                self._install_preplanned_phase_path(command)
            elif command.phase == MotionPhase.ATTACH:
                verify_contact = command.params.get("verify_grasp_contact")
                if not callable(verify_contact) or not bool(verify_contact()):
                    raise RuntimeError(
                        "ATTACH requires a verified target-finger contact from GRIPPER_CLOSE"
                    )
                manager.attach_target(command.active_object, robot, arm)
                self._phase_bookkeeping_done = True
            elif command.phase == MotionPhase.CARRY_HOME:
                manager.assert_attached_owner(command.active_object, robot, arm)
                preplanned_path = command.params.get("preplanned_joint_path")
                if preplanned_path is None:
                    raise RuntimeError("CARRY_HOME requires a preplanned joint path")
                self._install_preplanned_phase_path(command)
            elif command.phase == MotionPhase.TERMINAL_PLACE_DESCENT:
                manager.begin_placement_descent(
                    command.active_object, command.support_object, robot, arm
                )
                if command.params.get("preplanned_joint_path") is not None:
                    cached_plan = self._command_path(
                        command.params["preplanned_joint_path"]
                    )
                    if command.params.get("continuous_descent", False):
                        cached_valid = self._validate_continuous_place_plan(
                            command, cached_plan
                        )
                    else:
                        cached_valid = True
                    if cached_valid:
                        self._install_preplanned_phase_path(command)
                    else:
                        # A cached path can be invalidated by a safety hold or
                        # a small attached-object slip between evaluation and
                        # execution.  Fall back to a fresh single-plan query
                        # from the measured state instead of failing the phase
                        # on stale candidate data.
                        command.params.pop("preplanned_joint_path", None)
                        LOGGER.warning(
                            "[PhaseDebug] cached-place-plan rejected robot=%s arm=%s; "
                            "falling back to native-v2 replanning",
                            self.name,
                            self.lr_name,
                        )
            elif command.phase == MotionPhase.DETACH_AND_SETTLE:
                manager.detach_target(command.active_object, robot, arm)
                self._phase_bookkeeping_done = True
            elif command.phase == MotionPhase.TERMINAL_RETREAT:
                manager.begin_terminal_retreat(command.active_object, robot, arm)
            elif command.phase == MotionPhase.RESTORE_WORLD:
                manager.restore_world(command.active_object)
                self._phase_bookkeeping_done = True

        if (
            command.phase == MotionPhase.TERMINAL_PLACE_DESCENT
            and command.params.get("contact_complete", False)
        ):
            return self.hold_action()

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
        plan_validator = None
        if (
            command.phase == MotionPhase.TERMINAL_PLACE_DESCENT
            and command.params.get("continuous_descent", False)
        ):
            plan_validator = lambda plan: self._validate_continuous_place_plan(
                command, plan
            )
        return self.ee_forward(
            command.target_position,
            command.target_orientation,
            # Planning-target identity and physical completion are different
            # contracts. Passing completion tolerance here can suppress a
            # needed plan while the measured EE is still outside completion,
            # causing an infinite hold. Keep target-change detection tight;
            # ``is_phase_command_complete`` applies physical tolerance.
            eps=command.planning_epsilon,
            plan_validator=plan_validator,
        )

    def _validate_continuous_place_plan(self, command: MotionPhaseCommand, plan) -> bool:
        """Reject a fast descent that is indirect or advances too far per frame."""

        start_position, _ = self.get_ee_pose()
        plan_position = plan.position
        if len(plan_position):
            batch_forward_kinematic = getattr(self, "_forward_kinematic_batch", None)
            if callable(batch_forward_kinematic):
                plan_joint_names = getattr(plan, "joint_names", None)
                if plan_joint_names is None:
                    positions_from_batch = batch_forward_kinematic(plan_position)
                else:
                    positions_from_batch = batch_forward_kinematic(
                        plan_position, joint_names=list(plan_joint_names)
                    )
                positions = np.asarray(
                    positions_from_batch, dtype=float
                )
            else:
                # Keep lightweight host-side controller stubs usable without
                # importing torch/CuRobo.  The real controller always has the
                # batch helper above, so this is not the runtime path.
                positions = np.asarray(
                    [
                        self.forward_kinematic(
                            joint_position.detach().cpu().numpy()
                        )[0]
                        for joint_position in plan_position
                    ],
                    dtype=float,
                )
        else:
            positions = np.asarray([], dtype=float)
        if not len(positions) or not np.all(np.isfinite(positions)):
            LOGGER.warning(
                "[PhaseDebug] continuous-place-plan-invalid robot=%s arm=%s reason=non_finite_path",
                self.name,
                self.lr_name,
            )
            return False

        direct_vector = np.asarray(command.target_position, dtype=float) - start_position
        direct_length = float(np.linalg.norm(direct_vector))
        path_length = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
        if direct_length <= 1e-9:
            path_length_ratio = 1.0 if path_length <= 1e-9 else float("inf")
            max_deviation = path_length
        else:
            direction = direct_vector / direct_length
            relative = positions - start_position
            projection = np.clip(relative @ direction, 0.0, direct_length)
            closest = start_position + projection[:, None] * direct_vector / direct_length
            path_length_ratio = path_length / direct_length
            max_deviation = float(
                np.max(np.linalg.norm(positions - closest, axis=1))
            )

        executed = positions[:: self.ds_ratio]
        executed_with_start = np.concatenate(
            [np.asarray(start_position, dtype=float).reshape(1, 3), executed], axis=0
        )
        max_step = float(
            np.max(np.linalg.norm(np.diff(executed_with_start, axis=0), axis=1))
        )
        max_allowed_step = float(command.params["max_cartesian_step_m"])
        max_allowed_ratio = float(command.params["max_path_length_ratio"])
        max_allowed_deviation = float(command.params["max_path_deviation_m"])
        valid_limits = (
            np.isfinite(max_allowed_step)
            and max_allowed_step > 0.0
            and np.isfinite(max_allowed_ratio)
            and max_allowed_ratio >= 1.0
            and np.isfinite(max_allowed_deviation)
            and max_allowed_deviation >= 0.0
        )
        valid = bool(
            valid_limits
            and max_step <= max_allowed_step + 1e-6
            and path_length_ratio <= max_allowed_ratio + 1e-6
            and max_deviation <= max_allowed_deviation + 1e-6
        )
        log = LOGGER.info if valid else LOGGER.warning
        log(
            "[PhaseDebug] continuous-place-plan robot=%s arm=%s valid=%s waypoints=%d "
            "stride=%d max_step=%.6f/%.6f path_ratio=%.4f/%.4f "
            "max_deviation=%.6f/%.6f",
            self.name,
            self.lr_name,
            valid,
            len(positions),
            self.ds_ratio,
            max_step,
            max_allowed_step,
            path_length_ratio,
            max_allowed_ratio,
            max_deviation,
            max_allowed_deviation,
        )
        return valid

    def complete_terminal_place_on_contact(self, command: MotionPhaseCommand) -> None:
        """Cancel the remaining descent without resetting phase completion state."""

        if command is not self._active_phase_command:
            return
        self.cmd_plan = None
        self.cmd_idx = 0
        self._phase_plan_finished = True
        self._last_arm_action = None
        if not command.params.get("contact_stop_logged", False):
            LOGGER.info(
                "[PhaseDebug] contact-stop robot=%s arm=%s phase=%s",
                self.name,
                self.lr_name,
                command.phase.value,
            )
            command.params["contact_stop_logged"] = True

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
                LOGGER.info(
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

    def _make_action(self, arm_action, gripper_action):
        """Build the common simulator action payload for every hold path."""

        arm_action = np.asarray(arm_action, dtype=float).copy()
        gripper_action = np.asarray(gripper_action, dtype=float).copy()
        joint_indices = np.concatenate([self.arm_indices, self.gripper_indices])
        return {
            "joint_positions": np.concatenate([arm_action, gripper_action]),
            "joint_indices": joint_indices,
            "lr_name": self.lr_name,
            "arm_action": arm_action,
            "gripper_action": gripper_action,
        }

    def hold_action(self):
        """Return an articulation target equal to the measured current joints."""

        sim_js = self.robot.get_joints_state()
        arm_action = np.asarray(sim_js.positions[self.arm_indices], dtype=float)
        return self._make_action(arm_action, self.get_gripper_action())

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
        return self._make_action(
            joint_positions[:arm_count], joint_positions[arm_count:]
        )

    def ee_forward(
        self,
        ee_trans: torch.Tensor | np.ndarray,
        ee_ori: torch.Tensor | np.ndarray,
        eps=1e-4,
        skip_plan=False,
        gripper_action=None,
        plan_validator=None,
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
            new_plan_created = False
            if plan_flag:
                self.cmd_plan = None
                self.cmd_idx = 0
                self._step_idx = 0
                self.num_last_cmd = 0
                self._last_arm_action = None
                self._phase_plan_started = True
                result = self.plan(ee_trans, ee_ori, sim_js, js_names)
                self._log_plan_result("ee_forward", result, target=ee_trans.detach().cpu().numpy())
                if self._result_success(result):
                    raw_plan = self._result_path(result)
                    cmd_plan = self._command_path(raw_plan)
                    if cmd_plan is not None:
                        self._install_command_plan(
                            cmd_plan,
                            target_position=ee_trans,
                            target_orientation=ee_ori,
                            phase_name=self._last_command_name,
                            cached=False,
                        )
                        self._write_curobo_plan_debug(
                            result=result,
                            sim_js=sim_js,
                            js_names=js_names,
                            ee_trans=ee_trans,
                            ee_ori=ee_ori,
                            raw_plan=raw_plan,
                            ordered_cmd_plan=self.cmd_plan,
                            branch="single",
                            selected_path_index=0,
                            selected_path_source="native result interpolated_trajectory",
                        )
                        self.num_plan_failed = 0
                        new_plan_created = True
                if not new_plan_created:
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
            if (
                new_plan_created
                and self.cmd_plan is not None
                and plan_validator is not None
                and not bool(plan_validator(self.cmd_plan))
            ):
                self.cmd_plan = None
                self.cmd_idx = 0
                self._last_arm_action = None
                self._phase_plan_failed = True
                self.num_plan_failed += 1
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
                    LOGGER.info(
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
        self._action = self._make_action(arm_action, gripper_action)
        return self._action

    def get_gripper_action(self):
        return np.clip(self._gripper_state * self._gripper_joint_position, 0.0, 0.04)

    def get_ee_pose(self):
        sim_js = self.robot.get_joints_state()
        q_state = self._arm_joint_state(sim_js)
        state = self.kin_model.compute_kinematics(q_state.unsqueeze(1))
        ee_pose = state.tool_poses.get_link_pose(self.kin_model.tool_frames[0])
        return ee_pose.position[0].cpu().numpy(), ee_pose.quaternion[0].cpu().numpy()

    def get_armbase_pose(self):
        armbase_pose = get_relative_transform(
            get_prim_at_path(self.robot_base_path), get_prim_at_path(self.task.root_prim_path)
        )
        return pose_from_tf_matrix(armbase_pose)

    def forward_kinematic(self, q_state: np.ndarray):
        q_state = self.tensor_args.to_device(q_state.reshape(1, -1))
        state = JointState.from_position(q_state, joint_names=self._planner_joint_names())
        out = self.kin_model.compute_kinematics(state.unsqueeze(1))
        ee_pose = out.tool_poses.get_link_pose(self.kin_model.tool_frames[0])
        return ee_pose.position[0].cpu().numpy(), ee_pose.quaternion[0].cpu().numpy()

    def _forward_kinematic_batch(self, joint_positions, joint_names=None):
        """Compute tool positions for a trajectory in one native v2 FK call."""

        joint_positions = self.tensor_args.to_device(joint_positions)
        if joint_positions.ndim != 2:
            raise ValueError(
                "batched Cartesian FK requires a [time, dof] position tensor, "
                f"got shape {tuple(joint_positions.shape)}"
            )
        planner_names = self._planner_joint_names()
        source_names = planner_names if joint_names is None else list(joint_names)
        if len(source_names) != joint_positions.shape[-1]:
            raise ValueError(
                "batched Cartesian FK joint_names do not match position DOF: "
                f"position_shape={tuple(joint_positions.shape)}, "
                f"joint_names={source_names!r}"
            )
        if len(set(source_names)) != len(source_names):
            raise ValueError(
                f"batched Cartesian FK joint_names must be unique: {source_names!r}"
            )
        if set(source_names) != set(planner_names):
            raise ValueError(
                "batched Cartesian FK joint contract does not match the native "
                f"planner: source={source_names!r}, planner={planner_names!r}"
            )
        if source_names != planner_names:
            reorder = [source_names.index(name) for name in planner_names]
            joint_positions = joint_positions[..., reorder].contiguous()
        state = JointState.from_position(
            joint_positions.contiguous(),
            joint_names=planner_names,
        )
        # Cartesian validation is read-only.  Avoid retaining an autograd graph
        # for every candidate path while keeping the native v2 FK kernel and
        # its single device-to-host transfer intact.
        with torch.inference_mode():
            out = self.kin_model.compute_kinematics(state)
        ee_pose = out.tool_poses.get_link_pose(self.kin_model.tool_frames[0])
        # Transfer the complete batch only after FK has finished.  The
        # Cartesian checks use position only, matching forward_kinematic's
        # existing callers while avoiding one device synchronization per
        # waypoint.
        return ee_pose.position.detach().cpu().numpy()

    def close_gripper(self):
        self._gripper_state = -1.0

    def open_gripper(self):
        self._gripper_state = 1.0

    def _get_curobo_world_object_names(self) -> List[str]:
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
        if any(not isinstance(name, str) or not name.strip() for name in requested_names):
            raise ValueError("attach_obj requires non-empty CuRobo obstacle path strings")
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

        # A later requested path can resolve an exact name that an earlier
        # parent path tentatively classified for disabling.  Finalize the
        # roles globally so attachment is the semantic winner, while keeping
        # first-seen order within each exact-name list.
        resolved_names = list(dict.fromkeys(resolved_names))
        resolved_name_set = set(resolved_names)
        disabled_names = list(
            dict.fromkeys(
                name for name in disabled_names if name not in resolved_name_set
            )
        )

        return resolved_names, disabled_names

    def _native_attachment_geometry(self, object_names: List[str]):
        """Build a native-v2 local mesh and current reference-frame offset.

        Native ``AttachmentManager.attach`` expects fitted sphere centers in an
        object-local frame and receives that object's current reference-frame
        pose separately.  Scene obstacles are stored in the planner reference
        frame, so passing them directly would make the attached spheres appear
        at the wrong pose (and is especially wrong after a dynamic-object
        update).  Consolidate all requested collider meshes in the first
        collider's current frame, then let native v2 compute the link-local
        transform from the explicit ``world_objects_pose_offset``.
        """

        from curobo._src.geom.types import Mesh

        scene_collision = self.planner.scene_collision_checker
        scene_model = scene_collision.scene_model
        if isinstance(scene_model, list):
            scene_model = scene_model[0]
        if scene_model is None:
            scene_model = self.world_cfg

        obstacles = []
        current_poses = []
        for object_name in object_names:
            obstacle = scene_model.get_obstacle(object_name)
            if obstacle is None:
                raise ValueError(
                    f"attach collision prim is not in native v2 scene model: {object_name}"
                )
            obstacles.append(obstacle)
            if self.collision_scene_manager is not None and self.collision_world_mode == "physics_schema":
                current_poses.append(
                    self.collision_scene_manager._controller_obstacle_pose(  # pylint: disable=protected-access
                        self, object_name
                    )
                )
            else:
                pose = getattr(obstacle, "pose", None)
                if pose is None:
                    raise ValueError(f"native v2 obstacle has no pose: {object_name}")
                current_poses.append(Pose.from_list(list(pose), device_cfg=self.tensor_args))

        anchor_pose = current_poses[0]
        anchor_inverse = np.linalg.inv(anchor_pose.get_numpy_matrix()[0])
        vertices = []
        faces = []
        vertex_offset = 0
        for obstacle, current_pose in zip(obstacles, current_poses):
            mesh = obstacle.get_trimesh_mesh(transform_with_pose=False)
            local_vertices = np.asarray(mesh.vertices, dtype=np.float32)
            if local_vertices.ndim != 2 or local_vertices.shape[1] != 3:
                raise ValueError(
                    f"native v2 attachment mesh has invalid vertices: {obstacle.name}"
                )
            object_matrix = current_pose.get_numpy_matrix()[0]
            world_vertices = (object_matrix[:3, :3] @ local_vertices.T).T + object_matrix[:3, 3]
            anchor_vertices = (
                (anchor_inverse[:3, :3] @ world_vertices.T).T + anchor_inverse[:3, 3]
            )
            obstacle_faces = np.asarray(mesh.faces, dtype=np.int64)
            if obstacle_faces.ndim != 2 or obstacle_faces.shape[1] != 3:
                raise ValueError(
                    f"native v2 attachment mesh has invalid faces: {obstacle.name}"
                )
            vertices.append(anchor_vertices)
            faces.append(obstacle_faces + vertex_offset)
            vertex_offset += anchor_vertices.shape[0]

        if not vertices:
            raise ValueError("native v2 attachment requires at least one mesh")
        combined = Mesh(
            name="__native_attached_object__",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vertices=np.concatenate(vertices, axis=0),
            faces=np.concatenate(faces, axis=0),
        )
        return [combined], anchor_pose

    def _attach_native_planner(
        self,
        planner,
        object_names: List[str],
        *,
        link_name="attached_object",
        joint_state=None,
        world_objects_pose_offset=None,
    ):
        """Attach one resolved object set to a native planner.

        Pick execution, Place candidate evaluation, and transient post-grasp
        validation all use the same CuRobo attachment contract.  Keeping the
        validation, mesh fitting, sphere count, and named-joint conversion in
        one helper prevents those paths from drifting apart.
        """

        if not object_names or any(
            not isinstance(path, str) or not path.strip() for path in object_names
        ):
            raise ValueError("native attachment requires non-empty obstacle path strings")
        paths = [path.strip() for path in object_names]
        if planner is None:
            raise RuntimeError("native attachment requires an initialized planner")
        missing = [path for path in paths if self.world_cfg.get_obstacle(path) is None]
        if missing:
            raise ValueError(f"attach collision prims are not in CuRobo world: {missing}")
        if joint_state is None:
            joint_state = self._arm_joint_state(self.robot.get_joints_state())
        attachment_meshes, attachment_offset = self._native_attachment_geometry(paths)
        if world_objects_pose_offset is not None:
            attachment_offset = world_objects_pose_offset
        planner.attachment_manager.attach(
            joint_state,
            attachment_meshes,
            link_name=link_name,
            num_spheres=max(
                1,
                self._attached_sphere_count(link_name, 1, planner=planner),
            ),
            surface_radius=0.001,
            sphere_fit_type=SphereFitType.VOXEL,
            world_objects_pose_offset=attachment_offset,
            disable_obstacle_names=paths,
        )
        return paths

    def attach_objects(
        self,
        obj_prim_paths: List[str],
        link_name="attached_object",
        world_objects_pose_offset=None,
    ):
        try:
            # Pick execution is owned by the single native planner.  The
            # batch planner is synchronized lazily by Place immediately
            # before candidate evaluation; attaching both here makes a
            # candidate-only planner participate in the Pick attach
            # transition and can invalidate the execution planner's next
            # start-state query on Warp/graph-backed scenes.
            paths = self._attach_native_planner(
                self.planner,
                obj_prim_paths,
                link_name=link_name,
                world_objects_pose_offset=world_objects_pose_offset,
            )
        except Exception as primary_error:
            # The single planner is the only manager mutated by this method.
            # Keep the failure atomic for callers that retry the attach.
            try:
                self.planner.attachment_manager.detach()
            except Exception as rollback_error:
                _record_attachment_rollback_failure(
                    primary_error,
                    "detach partially attached native-v2 planner",
                    rollback_error,
                )
            raise
        self._native_attached_obstacle_names = list(paths)
        self._native_batch_attached_obstacle_names = []
        LOGGER.warning("[AttachDebug] attached=native disabled_world_obstacles=%s", paths)
        # Native v2 mutates the attachment manager in-place and intentionally
        # returns None. Keep this controller API boolean for its callers.
        return True

    def sync_native_batch_attachment(
        self,
        link_name="attached_object",
        world_objects_pose_offset=None,
    ):
        """Attach the currently held object to the native batch planner.

        The batch planner is used only to rank Place candidates.  It must see
        the same attached geometry as the execution planner, but it must not
        be attached during Pick: two CUDA-backed AttachmentManagers share no
        solver state contract and the second attach can invalidate the live
        single-planner start-state query.  This method is therefore an
        explicit Place-side synchronization point.
        """

        if self.batch_planner is None:
            return False
        paths = list(self._native_attached_obstacle_names)
        if not paths:
            raise RuntimeError(
                "cannot synchronize native batch attachment without an attached object"
            )
        if self._native_batch_attached_obstacle_names == paths:
            return True

        # A failed Place retry must not retain a stale candidate-world
        # attachment.  Detach only the candidate planner; the execution
        # planner remains authoritative for the held object.
        try:
            self.batch_planner.attachment_manager.detach()
            self._attach_native_planner(
                self.batch_planner,
                paths,
                link_name=link_name,
                world_objects_pose_offset=world_objects_pose_offset,
            )
        except Exception as primary_error:
            try:
                self.batch_planner.attachment_manager.detach()
            except Exception as rollback_error:
                _record_attachment_rollback_failure(
                    primary_error,
                    "detach failed native-v2 batch attachment",
                    rollback_error,
                )
            self._native_batch_attached_obstacle_names = []
            raise

        self._native_batch_attached_obstacle_names = paths
        LOGGER.info(
            "[AttachDebug] synchronized=native_batch robot=%s arm=%s paths=%s",
            self.name,
            self.lr_name,
            paths,
        )
        return True

    def test_attached_forward_from_joint_positions(
        self,
        ee_trans: np.ndarray,
        ee_ori: np.ndarray,
        start_arm_positions: np.ndarray,
        obj_prim_paths: List[str],
    ):
        """Plan a post-grasp target with the object attached at the grasp endpoint.

        Candidate validation must use the same attached collision geometry as
        execution.  The attachment is deliberately transient and is always
        removed before returning to the caller.
        """
        if not obj_prim_paths:
            raise ValueError("post-grasp validation requires attach collision prim paths")
        start_arm_positions = np.asarray(start_arm_positions, dtype=float).reshape(-1)
        if len(start_arm_positions) != len(self.arm_indices):
            raise ValueError(
                "start_arm_positions must match the controller arm joint count: "
                f"got {len(start_arm_positions)}, expected {len(self.arm_indices)}"
            )

        sim_js = self.robot.get_joints_state()
        sim_js.positions = np.asarray(sim_js.positions, dtype=float).copy()
        sim_js.positions[self.arm_indices] = start_arm_positions
        cu_js = self._arm_joint_state(sim_js)

        try:
            paths = self._attach_native_planner(
                self.planner,
                obj_prim_paths,
                link_name="attached_object",
                joint_state=cu_js,
            )
            self._native_attached_obstacle_names = list(paths)
            success, end_positions, result = self.test_forward_from_joint_positions(
                ee_trans,
                ee_ori,
                start_arm_positions=start_arm_positions,
            )
            self._log_plan_result("test_attached_forward", result, target=ee_trans)
            return success, end_positions, result
        finally:
            self.planner.attachment_manager.detach()
            self._native_attached_obstacle_names = []

    def attach_obj(self, obj_prim_path: str, link_name="attached_object"):
        """Attach a legacy object path while preserving descendant selection."""

        sim_js = self.robot.get_joints_state()
        cu_js = self._arm_joint_state(sim_js)
        object_names, disabled_names = self._resolve_attach_object_names(obj_prim_path)
        self._attach_native_planner(
            self.planner,
            object_names,
            link_name=link_name,
            joint_state=cu_js,
        )
        self._native_attached_obstacle_names = list(object_names)
        disabled_now = []
        try:
            for object_name in disabled_names:
                self.planner.scene_collision_checker.enable_obstacle(object_name, False)
                # Record each completed mutation before attempting the next
                # descendant so a mid-loop failure is recoverable.
                disabled_now.append(object_name)
                if object_name not in self._legacy_disabled_attach_names:
                    self._legacy_disabled_attach_names.append(object_name)
        except Exception as primary_error:
            # Restore every descendant that this call disabled.  One restore
            # failure must not prevent the remaining restores or native
            # detach; failed names stay tracked for reset-time cleanup.
            for object_name in reversed(disabled_now):
                try:
                    self.planner.scene_collision_checker.enable_obstacle(object_name, True)
                except Exception as rollback_error:
                    _record_attachment_rollback_failure(
                        primary_error,
                        f"re-enable legacy obstacle {object_name!r}",
                        rollback_error,
                    )
                else:
                    self._legacy_disabled_attach_names = [
                        name
                        for name in self._legacy_disabled_attach_names
                        if name != object_name
                    ]
            try:
                self.planner.attachment_manager.detach()
                self._native_attached_obstacle_names = []
            except Exception as rollback_error:
                _record_attachment_rollback_failure(
                    primary_error, "detach native attachment", rollback_error
                )
            raise
        return True

    def detach_obj(self):
        attached_names = self._attached_obstacle_names()
        for planner in self._native_planners():
            planner.attachment_manager.detach()
        self._native_attached_obstacle_names = []
        self._native_batch_attached_obstacle_names = []
        self._reenable_legacy_disabled_attach_objects(attached_names)

    def has_attached_collision_spheres(self, link_name="attached_object") -> bool:
        spheres = (
            self.planner.kinematics.config.kinematics_config.get_link_spheres(link_name)
        )
        return bool(torch.any(spheres[:, 3] > 0.0).item())

    def _attached_sphere_count(
        self, link_name: str, object_count: int, *, planner=None
    ) -> int:
        planner = planner or self.planner
        total = planner.kinematics.config.kinematics_config.get_number_of_spheres(
            link_name
        )
        return max(1, int(total) // max(1, int(object_count)))

    def update_specific(self, ignore_substring, reference_prim_path):
        if self.collision_world_mode == "physics_schema":
            warnings.warn(
                "update_specific(ignore_substring=...) is ignored in physics_schema mode; "
                "use CollisionSceneManager state transitions",
                DeprecationWarning,
                stacklevel=2,
            )
            self.collision_scene_manager.sync_dynamic_poses(
                self._step_idx, interval_steps=1, force=False
            )
            return
        self._legacy_update_specific(ignore_substring, reference_prim_path)

    def _legacy_update_specific(self, ignore_substring, reference_prim_path):
        """LEGACY_STAGE_SCAN: old substring-based per-command world rebuild."""

        # LEGACY_BEGIN: keyword-based collision world, retained for comparison
        obstacles = self.usd_parser.get_obstacles_from_stage(
            ignore_substring=ignore_substring, reference_prim_path=reference_prim_path
        ).get_collision_check_world()
        self._update_world_if_changed(obstacles)
        # LEGACY_END

    def test_single_ik(self, ee_trans, ee_ori):
        refresh = getattr(self, "_refresh_reference_world_for_planning", None)
        if callable(refresh):
            refresh()
        sim_js = self.robot.get_joints_state()
        ik_goal = self._goal_tool_pose(ee_trans, ee_ori)
        result = self._run_timed_curobo_call(
            "ik.solve_pose",
            lambda: self.ik_solver.solve_pose(
                ik_goal,
                current_state=self._arm_joint_state(sim_js).unsqueeze(0),
                return_seeds=1,
            ),
        )
        return self._result_success(result)

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

        refresh = getattr(self, "_refresh_reference_world_for_planning", None)
        if callable(refresh):
            refresh()
        ee_trans_batch = self.tensor_args.to_device(ee_trans_batch_np)
        ee_ori_batch = self.tensor_args.to_device(ee_ori_batch_np)
        if not start_paths:
            raise ValueError(
                "batch terminal planning requires at least one named pre-grasp path"
            )
        if len(ee_trans_batch_np) != len(ee_ori_batch_np) or len(ee_trans_batch_np) != len(start_paths):
            raise ValueError(
                "batch terminal planning requires equal goal-position, goal-orientation, "
                f"and pre-grasp-path counts: positions={len(ee_trans_batch_np)}, "
                f"orientations={len(ee_ori_batch_np)}, paths={len(start_paths)}"
            )

        terminal_positions = []
        expected_joint_name_set = None
        for path_index, path in enumerate(start_paths):
            if path is None:
                # Native batch results intentionally keep failed items as
                # ``None``.  The batch solver still needs a complete start
                # tensor; these fallback rows are masked by the pre-grasp
                # success mask in the evaluator and cannot become valid joint
                # candidates on their own.
                terminal_positions.append(self._arm_joint_state(self.robot.get_joints_state()))
                continue
            names = getattr(path, "joint_names", None)
            if names is None or isinstance(names, (str, bytes)):
                raise ValueError(
                    "batch pre-grasp endpoint must provide explicit joint_names: "
                    f"path_index={path_index}"
                )
            names = list(names)
            if not names or len(set(names)) != len(names):
                raise ValueError(
                    "batch pre-grasp endpoint joint_names must be non-empty and "
                    f"unique: path_index={path_index}, joint_names={names!r}"
                )
            name_set = set(names)
            if expected_joint_name_set is None:
                expected_joint_name_set = name_set
            elif name_set != expected_joint_name_set:
                raise ValueError(
                    "batch pre-grasp endpoints must use the same named joint contract: "
                    f"path_index={path_index}, expected={sorted(expected_joint_name_set)!r}, "
                    f"got={sorted(name_set)!r}"
                )

            position = getattr(path, "position", None)
            if position is None:
                raise ValueError(
                    "batch pre-grasp endpoint must provide position: "
                    f"path_index={path_index}"
                )
            if not isinstance(position, torch.Tensor):
                position = self.tensor_args.to_device(position)
            if position.ndim < 2 or position.shape[0] < 1:
                raise ValueError(
                    "batch pre-grasp endpoint position must be a non-empty "
                    "trajectory with shape [time, dof]: "
                    f"path_index={path_index}, position_shape={tuple(position.shape)}"
                )
            if position.shape[-1] != len(names):
                raise ValueError(
                    "batch pre-grasp endpoint position DOF count does not match "
                    f"its joint_names: path_index={path_index}, "
                    f"position_shape={tuple(position.shape)}, joint_names={names!r}"
                )

            endpoint = JointState.from_position(
                position[-1], joint_names=names
            )
            terminal_positions.append(self._planner_state(endpoint))

        starts = torch.stack([state.position for state in terminal_positions])
        zeros = torch.zeros_like(starts)
        start_state = JointState(
            position=starts,
            velocity=zeros,
            acceleration=zeros,
            jerk=zeros,
            joint_names=self._planner_joint_names(),
        )
        if self.batch_planner is None:
            raise RuntimeError("batch planning was not enabled for this controller")
        if len(start_paths) < 1 or len(start_paths) > self.batch_planner.batch_size:
            raise ValueError(
                f"native batch planner accepts 1..{self.batch_planner.batch_size} paths"
            )
        return self._plan_batch_from_state(
            ee_trans_batch,
            ee_ori_batch,
            start_state,
            batch_size=len(start_paths),
            context="test_batch_forward_from_pregrasp",
        )

    def measure_cartesian_path(self, path, start_position, goal_position):
        """Return path/direct length ratio and maximum straight-line deviation."""

        path_names = getattr(path, "joint_names", None)
        if path_names is None or isinstance(path_names, (str, bytes)):
            raise ValueError(
                "Cartesian path measurement requires explicit joint_names; "
                "positional trajectory mapping is unsupported"
            )
        path_names = list(path_names)
        path_position = getattr(path, "position", None)
        if path_position is None or path_position.ndim < 2:
            raise ValueError(
                "Cartesian path measurement requires a [time, dof] "
                f"position tensor, got {getattr(path_position, 'shape', None)}"
            )
        if path_position.shape[-1] != len(path_names):
            raise ValueError(
                "Cartesian path position DOF count does not match its joint_names: "
                f"position_shape={tuple(path_position.shape)}, joint_names={path_names!r}"
            )
        if not path_names or len(set(path_names)) != len(path_names):
            raise ValueError(
                "Cartesian path joint_names must be non-empty and unique: "
                f"{path_names!r}"
            )

        # CuRobo's trajectory/result contract may be full (7 arm + 2 locked
        # fingers), while the kinematics model is deliberately active-arm
        # only.  Reduce by the explicit active names before FK; never slice a
        # nine-dimensional tensor positionally.
        active_names = list(self.raw_js_names)
        if set(path_names) != set(active_names) or path_names != active_names:
            reorder = getattr(path, "reorder", None)
            if not callable(reorder):
                raise ValueError(
                    "Cartesian path joint order/contract differs from active arm "
                    "names but the path cannot reorder by explicit names: "
                    f"path_names={path_names!r}, active_names={active_names!r}"
                )
            path = reorder(active_names)
            path_position = getattr(path, "position", None)
            if path_position is None or path_position.ndim < 2:
                raise ValueError(
                    "reordered Cartesian path has an invalid position tensor"
                )
            if path_position.shape[-1] != len(active_names):
                raise ValueError(
                    "reordered Cartesian path position DOF count does not match "
                    f"active arm names: position_shape={tuple(path_position.shape)}, "
                    f"active_names={active_names!r}"
                )

        if path_position.shape[0] == 0:
            return float("inf"), float("inf")

        batch_forward_kinematic = getattr(self, "_forward_kinematic_batch", None)
        if callable(batch_forward_kinematic):
            positions = np.asarray(
                batch_forward_kinematic(path_position, joint_names=active_names),
                dtype=float,
            )
        else:
            # Keep AST-loaded/unit-test controller stubs compatible when they
            # intentionally omit the torch/CuRobo batch implementation.
            planner_names_fn = getattr(self, "_planner_joint_names", None)
            planner_names = (
                list(planner_names_fn())
                if callable(planner_names_fn)
                else list(active_names)
            )
            if set(active_names) != set(planner_names):
                raise ValueError(
                    "Cartesian path active joints do not match the native planner: "
                    f"active={active_names!r}, planner={planner_names!r}"
                )
            reorder = [active_names.index(name) for name in planner_names]
            positions = np.asarray(
                [
                    self.forward_kinematic(
                        joint_position[..., reorder].detach().cpu().numpy()
                    )[0]
                    for joint_position in path_position
                ],
                dtype=float,
            )
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
        succ = self._result_success(result)
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
        if not self._result_success(result):
            return False, None, result

        cmd_plan = self._command_path(self._result_path(result))
        if cmd_plan is None:
            return False, None, result
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

        refresh = getattr(self, "_refresh_reference_world_for_planning", None)
        if callable(refresh):
            refresh()
        start_position = start_path.position[-1]
        start_joint_names = getattr(start_path, "joint_names", None)
        if start_joint_names is None:
            raise ValueError(
                "pre-grasp endpoint must provide explicit joint_names; "
                "positional trajectory mapping is unsupported"
            )
        start_joint_names = list(start_joint_names)
        if start_position.shape[-1] != len(start_joint_names):
            raise ValueError(
                "pre-grasp endpoint position DOF count does not match its "
                f"joint_names: position_shape={tuple(start_position.shape)}, "
                f"joint_names={start_joint_names!r}"
            )
        start_state = self._planner_state(
            JointState.from_position(start_position, joint_names=start_joint_names)
        )
        return self._plan_pose_from_state(
            ee_trans,
            ee_ori,
            start_state,
            context="test_single_forward_from_pregrasp",
        )

    def pre_forward(self, ee_trans: np.ndarray, ee_ori: np.ndarray, expected_js=None, ds_ratio=1):
        assert ee_trans is not None and ee_ori is not None
        ee_trans = self.tensor_args.to_device(ee_trans)
        ee_ori = self.tensor_args.to_device(ee_ori)
        sim_js = self.robot.get_joints_state()
        js_names = self.robot.dof_names
        if expected_js is not None:
            sim_js.positions[self.arm_indices] = expected_js
        result = self.plan(ee_trans, ee_ori, sim_js, js_names)
        if self._result_success(result):
            print("Success")
            cmd_plan = self._command_path(self._result_path(result))
            N = cmd_plan.shape[0]
            dt = self.interpolation_dt
            self.ds_ratio = ds_ratio
            cmd_time = N * dt / self._time_dilation_factor / self.ds_ratio
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
        return self._make_action(arm_action, gripper_action)
