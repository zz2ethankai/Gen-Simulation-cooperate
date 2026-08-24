"""Controller-owned runtime composition.

The controller façade owns no CuRobo planner.  ``PlannerRuntime`` owns the
native single/batch instances and this class supplies the simulator-specific
factories, state conversion and attachment/scene adapters around it.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from core.planning.attachment_runtime import AttachmentRuntime
from core.controllers.curobo.phase_execution import PhaseExecutor
from core.planning.domain_types import (
    AttachmentSpec,
    BatchPlanResult,
    BatchPosePlanRequest,
    CollisionOptions,
    CollisionPolicy,
    CspacePlanRequest,
    PlanResult,
    PosePlanRequest,
    PlanningProfile,
    PlannerRuntimeProfile,
    JointTrajectory,
)
from core.planning.native_scene_adapter import NativeSceneAdapter
from core.planning.planner_runtime import PlannerRuntime
from core.planning.scene_runtime import SceneRuntime
from core.planning.motion_command import MotionPhase
from core.utils.constants import CUROBO_BATCH_SIZE
from core.controllers.curobo.trajectory import normalize_named_trajectory

LOGGER = logging.getLogger("de_logger")

if TYPE_CHECKING:
    from core.planning.native_planner_factory import PlannerBuildConfig
else:
    # Keep annotations resolvable without importing CuRobo/Warp at module load.
    PlannerBuildConfig = Any


def resolve_native_robot_config(robot_file: str) -> dict[str, Any]:
    """Normalize a SimBox robot YAML for native planner construction."""

    from curobo.config_io import load_yaml

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
    for key in (
        "use_usd_kinematics", "isaac_usd_path", "usd_path", "usd_robot_root",
        "usd_flip_joints", "usd_flip_joint_limits", "ee_link",
    ):
        kinematics.pop(key, None)

    def resolve_path(value: str) -> str:
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
            local = config_dir / sphere_path
            sphere_path = local if local.exists() else asset_root / sphere_path
        if sphere_path.exists():
            sphere_data = load_yaml(str(sphere_path)) or {}
            kinematics["collision_spheres"] = sphere_data.get("collision_spheres", sphere_data)
    cspace = kinematics.get("cspace")
    if isinstance(cspace, dict):
        if "default_joint_position" not in cspace and "retract_config" in cspace:
            cspace["default_joint_position"] = cspace.pop("retract_config")
        else:
            cspace.pop("retract_config", None)
    return {"kinematics": kinematics}


class _NativeAttachmentAdapter:
    """Adapt one PlannerRuntime-owned native planner to AttachmentRuntime."""

    def __init__(self, runtime: "MotionPlannerRuntime", *, batch: bool = False) -> None:
        self.runtime = runtime
        self.batch = bool(batch)

    def _planner(self):
        if self.batch:
            return self.runtime.ensure_batch_planner()
        return self.runtime.native_planner

    def attach(
        self,
        state,
        meshes,
        *,
        link_name="attached_object",
        world_objects_pose_offset=None,
        disable_obstacle_names=(),
        **_kwargs,
    ):
        return self.runtime.attach_native(
            list(disable_obstacle_names),
            state,
            meshes,
            link_name=link_name,
            world_objects_pose_offset=world_objects_pose_offset,
            planner=self._planner(),
        )

    def detach(self):
        return self._planner().attachment_manager.detach()


@dataclass
class RobotPort:
    """Narrow robot/state inputs consumed by :class:`MotionPlannerRuntime`."""

    name: str
    lr_name: str
    task_cfg: Mapping[str, Any]
    robot: Any
    tensor_args: Any
    arm_spec: Any
    arm_indices: Any
    raw_js_names: list[str]
    batch_capability: bool
    constrain_grasp_approach: bool
    collision_scene_manager: Any = None
    collision_world_mode: str = "physics_schema"
    obstacle_pose: Callable[[str], Any] | None = None
    native_start_collision_diagnostic: Callable[[], Any] | None = None
    interpolation_dt: float = 0.01
    ik_solver: Any = None
    kin_model: Any = None


class MotionPlannerRuntime:
    """Own planner construction and simulator-independent native operations."""

    batch_attachment_runtime: AttachmentRuntime | None = None

    def __init__(
        self,
        planner_build_config: PlannerBuildConfig,
        robot_port: RobotPort,
        *,
        world: Any = None,
        phase_executor: PhaseExecutor | None = None,
        execution_state: Any = None,
        setup: Any = None,
    ) -> None:
        self.planner_build_config = planner_build_config
        self.robot_port = robot_port
        self.execution_state = execution_state
        self.setup = setup
        self.scene_runtime = SceneRuntime(world)
        self.phase_executor = phase_executor or PhaseExecutor()
        self._configure_native_runtime()
        self.attachment_runtime = AttachmentRuntime(
            manager=_NativeAttachmentAdapter(self), strict=False
        )
        self.batch_attachment_runtime: AttachmentRuntime | None = None
        self._require_batch_scene_adapter: Callable[[], Any] | None = None
        # Keep the controller-local geometry adapter on the same revision
        # fanout as every collision-scene adapter.  It is used by attachment
        # fitting and diagnostics, so leaving it outside PlannerRuntime would
        # leave stale world provenance after a scene update (and after lazy
        # batch materialization).
        self.native_scene_adapter = NativeSceneAdapter(
            self.native_planner,
            strict=True,
            world=self.scene_runtime.world,
            world_revision=self.scene_revision,
        )
        self.planner_runtime.register_scene_adapter(self.native_scene_adapter)

    @property
    def scene_revision(self) -> int:
        """Revision shared by the controller scene and native planners."""

        return int(self.planner_runtime.scene_revision)

    @property
    def world_revision(self) -> int:
        """Alias used by scene ports and collision audits."""

        return self.scene_revision

    @property
    def world(self):
        return self.scene_runtime.world

    def adopt_scene_revision(self, revision: int) -> int:
        return self.planner_runtime.adopt_scene_revision(revision)

    @property
    def native_planner(self):
        return self.planner_runtime.ensure_planner()

    @property
    def batch_planner(self):
        return self.planner_runtime.batch_planner

    @property
    def planner_names(self) -> list[str]:
        return list(self.native_planner.joint_names)

    def _configure_native_runtime(self) -> None:
        # CuRobo's native factory imports Warp.  Keep that simulator-only
        # dependency behind runtime construction so the typed request façade
        # remains importable for host-side contract tests.
        from core.planning.native_planner_factory import NativePlannerFactory

        port = self.robot_port
        pick_cfg = dict(port.task_cfg).get("planning", {}).get("pick_place", {})
        graph_enabled = bool(pick_cfg.get("enable_graph", False))
        # Candidate batches need one viable path, not a proof that every
        # sampled pose is solvable.  CuRobo v2's batch graph seed checks all
        # start/goal nodes as one graph; one colliding IK goal can therefore
        # poison every candidate.  Keep it as an explicit opt-in for tasks
        # that need it, while the normal batch path stays on independent
        # IK/TrajOpt attempts.
        batch_graph_enabled = bool(pick_cfg.get("batch_enable_graph", False))
        try:
            # Single-object phases are sensitive to the sampled IK seed after
            # attachment.  Keep the historical retry budget close to 10
            # without making every batch candidate pay for it; batch attempts
            # remain capped independently below.
            max_attempts = max(1, int(pick_cfg.get("max_plan_attempts", 8)))
        except (TypeError, ValueError):
            max_attempts = 8
        try:
            batch_attempts = max(1, int(pick_cfg.get("batch_max_plan_attempts", min(max_attempts, 4))))
        except (TypeError, ValueError):
            batch_attempts = min(max_attempts, 4)
        try:
            batch_single_fallback = max(
                0,
                min(
                    CUROBO_BATCH_SIZE,
                    int(pick_cfg.get("batch_single_fallback_candidates", 4)),
                ),
            )
        except (TypeError, ValueError):
            batch_single_fallback = 4
        try:
            interpolation_dt = float(pick_cfg.get("interpolation_dt", 0.01))
            if not np.isfinite(interpolation_dt) or interpolation_dt <= 0:
                raise ValueError
        except (TypeError, ValueError):
            interpolation_dt = 0.01
        try:
            warmup_iterations = max(1, int(pick_cfg.get("warmup_iterations", 1)))
        except (TypeError, ValueError):
            warmup_iterations = 1
        self.max_plan_attempts = max_attempts
        self.batch_max_attempts = batch_attempts
        self.batch_single_fallback_candidates = batch_single_fallback
        self.graph_enabled = graph_enabled
        self.single_graph_attempt = max(0, min(1, max_attempts - 1)) if graph_enabled else max_attempts
        self.batch_graph_enabled = batch_graph_enabled
        self.batch_graph_attempt = max(0, min(3, batch_attempts - 1)) if batch_graph_enabled else batch_attempts

        factory = NativePlannerFactory(
            self.planner_build_config,
            robot_config=resolve_native_robot_config,
            pose_criteria=self._set_pose_criteria,
            collision_cache=self._collision_cache(),
            graph_enabled=graph_enabled,
            warmup_iterations=warmup_iterations,
            interpolation_dt=interpolation_dt,
        )

        self.planner_runtime = PlannerRuntime(
            name=f"{port.name}.{port.lr_name}",
            profile=PlannerRuntimeProfile(
                name=f"{port.name}.{port.lr_name}",
                batch_enabled=bool(port.batch_capability),
                max_batch_size=CUROBO_BATCH_SIZE,
                lazy_batch=True,
                warmup_config={
                    "enable_graph": graph_enabled or batch_graph_enabled,
                    "num_warmup_iterations": warmup_iterations,
                },
            ),
            planner_factory=factory.build_single,
            batch_planner_factory=factory.build_batch,
            scene_revision=self.scene_runtime.revision,
            # Bind before the first native planner is materialized.  The
            # replay installs the initial world, then PlannerRuntime's
            # factory/update/warmup transaction builds the solver against it.
            scene=self.scene_runtime,
        )
        if self.scene_runtime.world is None:
            raise RuntimeError("MotionPlannerRuntime requires an explicit planning world")
        # SceneRuntime owns revision increments.  Binding here makes its
        # replay/update fanout the one path into PlannerRuntime, including the
        # revision on every world update and on lazy batch materialization.
        self.planner_runtime.bind_scene(self.scene_runtime)
        port.ik_solver = self.native_planner.ik_solver
        port.kin_model = self.native_planner.kinematics
        port.interpolation_dt = float(self.native_planner.trajopt_solver.config.interpolation_dt)
    def _collision_cache(self):
        spec = getattr(self.robot_port, "arm_spec", None)
        cache = getattr(spec, "collision_cache", None)
        return dict(cache) if cache is not None else {"cuboid": 700, "mesh": 700}

    def _approach_axis(self) -> int:
        spec = getattr(self.robot_port, "arm_spec", None)
        axis = getattr(spec, "grasp_approach_axis", None)
        if axis is not None:
            return int(axis)
        return {"x": 0, "y": 1, "z": 2}[self.robot_port.robot.cfg["ee_axis"]]

    def _make_tool_goal(self, ee_translation, ee_orientation, batch_size=1):
        from curobo.types import GoalToolPose

        position = self.robot_port.tensor_args.to_device(ee_translation)
        quaternion = self.robot_port.tensor_args.to_device(ee_orientation)
        return GoalToolPose(
            tool_frames=[self.native_planner.tool_frames[0]],
            position=position.reshape(batch_size, 1, 1, 1, 3),
            quaternion=quaternion.reshape(batch_size, 1, 1, 1, 4),
        )

    def _set_pose_criteria(self, planner, criteria=None):
        from core.planning.native_bridge import ToolPoseCriteria

        port = self.robot_port
        if criteria is None:
            non_terminal = [0.0] * 6
            if port.constrain_grasp_approach:
                non_terminal = [1.0] * 6
                non_terminal[self._approach_axis()] = 0.0
            criteria = ToolPoseCriteria(
                terminal_pose_axes_weight_factor=[1.0] * 6,
                non_terminal_pose_axes_weight_factor=non_terminal,
                device_cfg=port.tensor_args,
            )
        planner.update_tool_pose_criteria({frame: criteria.clone() for frame in planner.tool_frames})

    def ensure_batch_planner(self):
        batch = self.planner_runtime.ensure_batch_planner()
        if self.batch_attachment_runtime is None:
            self.batch_attachment_runtime = AttachmentRuntime(
                manager=_NativeAttachmentAdapter(self, batch=True), strict=False
            )
        return batch

    def update_world(self, world):
        return self.scene_runtime.update_world(world)

    def update_obstacle_poses(self, poses, *, force: bool = False):
        """Publish dynamic collider poses with the shared scene revision."""

        update = self.scene_runtime.update_poses(poses, force=force)
        return update

    def check_current_start_state(self):
        """Validate the live articulation arm state against native joint limits."""

        import torch

        start_state = self.arm_joint_state(self.robot_port.robot.get_joints_state())
        limits = self.native_planner.kinematics.get_joint_limits()
        position = start_state.position
        valid = bool(
            torch.isfinite(position).all().item()
            and (position >= limits.position_lower_limits).all().item()
            and (position <= limits.position_upper_limits).all().item()
        )
        return valid, "valid" if valid else "joint_limit_or_non_finite"

    # ------------------------------------------------------------------
    # Pick/place scene transitions
    # ------------------------------------------------------------------
    def transition_target(
        self,
        object_name: str,
        support_name: str | None = None,
        *,
        collision_policy: CollisionPolicy | str | None = None,
    ):
        """Apply one typed collision transition to the shared scene manager."""

        manager = self.robot_port.collision_scene_manager
        policy = collision_policy or CollisionPolicy.WORLD_TRANSIT
        if not isinstance(policy, CollisionPolicy):
            policy = CollisionPolicy(str(policy).lower())
        object_name = str(object_name)
        support_name = None if support_name is None else str(support_name)
        if manager is None:
            if policy is CollisionPolicy.PASSTHROUGH:
                return self.scene_revision
            raise RuntimeError("MotionPlannerRuntime scene transition is unavailable")

        robot, arm = self.name, self.arm_name
        if policy is CollisionPolicy.WORLD_TRANSIT:
            manager.begin_target_transit(object_name, robot, arm)
        elif policy is CollisionPolicy.TARGET_APPROACH:
            manager.begin_target_approach(object_name, robot, arm)
        elif policy is CollisionPolicy.ATTACHED_CARRY:
            record = getattr(manager, "records", {}).get(object_name)
            state = getattr(getattr(record, "state", None), "value", None)
            if state == "placement_contact" and support_name:
                cleanup = getattr(manager, "restore_placement_support", None)
                if not callable(cleanup):
                    raise RuntimeError(
                        "placement query cleanup is unavailable for attached carry"
                    )
                cleanup(object_name, support_name, robot, arm)
            manager.assert_attached_owner(object_name, robot, arm)
        elif policy is CollisionPolicy.PLACEMENT_DESCENT:
            if not support_name:
                raise ValueError("PLACEMENT_DESCENT requires a support entity")
            manager.begin_placement_descent(object_name, support_name, robot, arm)
        elif policy is CollisionPolicy.RETREAT:
            manager.begin_terminal_retreat(object_name, robot, arm)
        elif policy is not CollisionPolicy.PASSTHROUGH:
            raise ValueError(f"unsupported collision policy: {policy!r}")
        return self.scene_revision

    def restore_world(self, object_name: str):
        manager = self.robot_port.collision_scene_manager
        if manager is None:
            raise RuntimeError("MotionPlannerRuntime world restore is unavailable")
        manager.restore_world(str(object_name))
        return self.scene_revision

    def source_support(self, object_name: str):
        manager = self.robot_port.collision_scene_manager
        if manager is None:
            raise RuntimeError("MotionPlannerRuntime source-support lookup is unavailable")
        return manager.get_source_support_entity(str(object_name))

    def assert_attached_owner(self, entity_name: str):
        manager = self.robot_port.collision_scene_manager
        if manager is None:
            raise RuntimeError("MotionPlannerRuntime attachment ownership is unavailable")
        return manager.assert_attached_owner(str(entity_name), self.name, self.arm_name)

    def finalize_detach_target(self, entity_name: str):
        manager = self.robot_port.collision_scene_manager
        if manager is None:
            raise RuntimeError("MotionPlannerRuntime detach finalization is unavailable")
        return manager.finalize_detach_target(str(entity_name), self.name, self.arm_name)

    def prepare_phase(self, command: Any):
        """Apply phase-owned scene bookkeeping before execution consumes a path.

        Pick and Place still construct typed phase commands, but execution no
        longer contains a second copy of the collision-scene state machine.
        This method returns a cached path, if the phase supplied one, for the
        execution component to install.
        """

        manager = self.robot_port.collision_scene_manager
        if manager is None:
            raise RuntimeError("MotionPhaseCommand requires CollisionSceneManager")
        object_name = command.active_object
        support_name = command.support_object
        robot, arm = self.name, self.arm_name
        phase = command.phase
        if phase is MotionPhase.SYNC_WORLD:
            manager.sync_dynamic_poses(
                getattr(self.execution_state, "step_idx", 0),
                interval_steps=1,
                force=True,
            )
            if object_name:
                self.transition_target(object_name, collision_policy=CollisionPolicy.WORLD_TRANSIT)
            return None
        if phase is MotionPhase.TRANSIT_PREGRASP:
            self.transition_target(object_name, collision_policy=CollisionPolicy.WORLD_TRANSIT)
        elif phase is MotionPhase.TERMINAL_GRASP_APPROACH:
            self.transition_target(object_name, collision_policy=CollisionPolicy.TARGET_APPROACH)
        elif phase is MotionPhase.TRANSIT_PREPLACE:
            self.assert_attached_owner(object_name)
        elif phase is MotionPhase.ATTACH:
            verify_contact = command.params.get("verify_grasp_contact")
            if not callable(verify_contact) or not bool(verify_contact()):
                raise RuntimeError(
                    "ATTACH requires a verified target-finger contact from GRIPPER_CLOSE"
                )
            manager.attach_target(object_name, robot, arm)
        elif phase is MotionPhase.CARRY_HOME:
            self.assert_attached_owner(object_name)
        elif phase is MotionPhase.TERMINAL_PLACE_DESCENT:
            manager.begin_placement_descent(object_name, support_name, robot, arm)
        elif phase is MotionPhase.DETACH_AND_SETTLE:
            manager.detach_target(object_name, robot, arm)
        elif phase is MotionPhase.TERMINAL_RETREAT:
            self.transition_target(object_name, collision_policy=CollisionPolicy.RETREAT)
        elif phase is MotionPhase.RESTORE_WORLD:
            self.restore_world(object_name)
        return command.params.get("preplanned_joint_path")

    # ------------------------------------------------------------------
    # State conversion and candidate/query operations
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.robot_port.name

    @property
    def arm_name(self) -> str:
        return self.robot_port.lr_name

    @property
    def raw_joint_names(self) -> tuple[str, ...]:
        return tuple(self.robot_port.raw_js_names)

    @property
    def arm_indices(self):
        return self.robot_port.arm_indices

    @property
    def tensor_args(self):
        return self.robot_port.tensor_args

    def _planner_joint_names(self) -> list[str]:
        return list(self.native_planner.joint_names)

    def _planner_state(self, state):
        names = getattr(state, "joint_names", None)
        if names is None:
            raise ValueError("native CuRobo planning states require explicit joint_names")
        names = list(names)
        planner_names = self._planner_joint_names()
        if set(planner_names) - set(names):
            raise ValueError(
                "planning state does not contain all native planner joints: "
                f"required={planner_names}, got={names}"
            )
        return state.reorder(planner_names)

    @staticmethod
    def _result_success(result) -> bool:
        if isinstance(result, BatchPlanResult):
            return result.is_success
        if isinstance(result, PlanResult):
            return result.success
        raise TypeError(
            "native planning requires a normalized PlanResult or "
            "BatchPlanResult"
        )

    @staticmethod
    def _result_path(result, batch_index=0):
        if isinstance(result, BatchPlanResult):
            paths = result.trajectories
        elif isinstance(result, PlanResult):
            paths = (result.trajectory,)
        else:
            raise TypeError(
            "native planning requires a normalized PlanResult or "
                "BatchPlanResult"
            )
        if batch_index >= len(paths) or paths[batch_index] is None:
            return None
        return paths[batch_index]

    def _command_path(self, path):
        """Normalize one native/typed path for phase execution and Skills."""

        if path is None:
            return None
        names = list(getattr(path, "joint_names", ()) or ())
        trajectory = (
            path
            if isinstance(path, JointTrajectory)
            else JointTrajectory.from_native(path, joint_names=names)
        )
        if set(self.raw_joint_names).issubset(trajectory.joint_names):
            return trajectory.reorder(self.raw_joint_names)
        active = self.joint_state_from_trajectory(trajectory)
        full_native = self.native_planner.kinematics.get_full_js(active)
        full_names = list(getattr(full_native, "joint_names", ()) or ())
        full = JointTrajectory.from_native(
            full_native,
            joint_names=full_names or self._planner_joint_names(),
        )
        if not set(self.raw_joint_names).issubset(full.joint_names):
            raise ValueError(
                "native planner result cannot be mapped to controller arm names: "
                f"result={full.joint_names}, arm={self.raw_joint_names}"
            )
        return full.reorder(self.raw_joint_names)

    def _install_command_plan(
        self,
        trajectory,
        *,
        target_position=None,
        target_orientation=None,
        phase_name: str = "unknown",
        cached: bool,
    ):
        """Install one normalized trajectory for ControllerExecution."""

        if trajectory is None or len(trajectory) == 0:
            raise ValueError(f"{phase_name} received an empty native-v2 path")
        if self.execution_state is None:
            raise RuntimeError("runtime execution state is not bound")
        self.execution_state.idx_list = list(range(len(self.raw_joint_names)))
        self.phase_executor.install(trajectory)
        self.execution_state.phase_plan_started = True
        if target_position is not None:
            self.execution_state.ee_trans = self.tensor_args.to_device(target_position)
        if target_orientation is not None:
            self.execution_state.ee_ori = self.tensor_args.to_device(target_orientation)
        setup = self.setup
        if setup is not None:
            visualize = getattr(setup, "_visualize_selected_plan", None)
            if callable(visualize):
                visualize()
        LOGGER.info(
            "[PhaseDebug] selected-plan robot=%s arm=%s phase=%s waypoints=%d stride=%d cached=%s",
            self.name,
            self.arm_name,
            phase_name,
            len(self.phase_executor.current),
            int(getattr(setup, "ds_ratio", 1) or 1),
            cached,
        )
        return trajectory

    def _refresh_reference_world_for_planning(self):
        setup = self.setup
        refresh = getattr(setup, "_refresh_reference_world_for_planning", None)
        if callable(refresh):
            return refresh()
        return None

    def _log_plan_result(self, context: str, result, target=None):
        setup = self.setup
        logger = getattr(setup, "_log_plan_result", None)
        if callable(logger):
            return logger(context, result, target=target)
        return None

    def plan_pose_result(
        self,
        ee_translation,
        ee_orientation,
        *,
        request_metadata=None,
    ):
        result = self.plan_pose(ee_translation, ee_orientation, request_metadata=request_metadata)
        self._log_plan_result("plan_pose", result, target=ee_translation)
        return result

    def plan_pose_from_path(
        self,
        ee_translation,
        ee_orientation,
        start_path,
        *,
        request_metadata=None,
    ):
        start_state = self.joint_state_from_path_endpoint(start_path)
        return self.plan_pose(
            ee_translation,
            ee_orientation,
            start_state=start_state,
            request_metadata=request_metadata,
        )

    def plan_pose_from_joint_positions(
        self,
        ee_translation,
        ee_orientation,
        *,
        start_arm_positions=None,
        request_metadata=None,
    ):
        positions = np.asarray(
            self.robot_port.robot.get_joints_state().positions,
            dtype=float,
        ).copy()
        if start_arm_positions is not None:
            start_arm_positions = np.asarray(start_arm_positions, dtype=float).reshape(-1)
            if len(start_arm_positions) != len(self.arm_indices):
                raise ValueError(
                    "start_arm_positions must match the controller arm joint count: "
                    f"got {len(start_arm_positions)}, expected {len(self.arm_indices)}"
                )
            positions[self.arm_indices] = start_arm_positions
        sim_state = type("JointStateSnapshot", (), {"positions": positions})()
        start_state = self.arm_joint_state(sim_state)
        result = self.plan_pose(
            ee_translation,
            ee_orientation,
            start_state=start_state,
            request_metadata=request_metadata,
        )
        if not self._result_success(result):
            return False, None, result
        trajectory = self._command_path(self._result_path(result))
        if trajectory is None or len(trajectory) == 0:
            return False, None, result
        endpoint = np.asarray(trajectory.positions[-1], dtype=float)
        return True, endpoint, result

    def measure_cartesian_path(self, path, start_position, goal_position):
        """Return path/direct length ratio and maximum straight-line deviation."""

        positions, path_names = normalize_named_trajectory(
            getattr(path, "position", getattr(path, "positions", None)),
            getattr(path, "joint_names", None),
            self.tensor_args,
        )
        active_names = list(self.raw_joint_names)
        if tuple(path_names) != tuple(active_names):
            reorder = getattr(path, "reorder", None)
            if not callable(reorder):
                raise ValueError(
                    "Cartesian path joint order differs from active arm names "
                    "but the path cannot reorder by explicit names"
                )
            path = reorder(active_names)
            positions, path_names = normalize_named_trajectory(
                getattr(path, "position", getattr(path, "positions", None)),
                getattr(path, "joint_names", None),
                self.tensor_args,
            )
        if len(positions) == 0:
            return float("inf"), float("inf")
        fk_positions = np.asarray(
            self._compute_cartesian_fk_batch(positions, joint_names=active_names),
            dtype=float,
        )
        direct_vector = np.asarray(goal_position, dtype=float) - np.asarray(start_position, dtype=float)
        direct_length = float(np.linalg.norm(direct_vector))
        path_length = float(np.sum(np.linalg.norm(np.diff(fk_positions, axis=0), axis=1)))
        if direct_length <= 1e-9:
            return (1.0 if path_length <= 1e-9 else float("inf")), path_length
        direction = direct_vector / direct_length
        relative = fk_positions - np.asarray(start_position, dtype=float)
        projection = np.clip(relative @ direction, 0.0, direct_length)
        closest = np.asarray(start_position, dtype=float) + projection[:, None] * direction
        deviation = float(np.max(np.linalg.norm(fk_positions - closest, axis=1)))
        return path_length / direct_length, deviation

    def _request_common_kwargs(
        self,
        request_metadata: Mapping[str, Any] | None,
        *,
        default_profile: PlanningProfile,
        attachment_runtime: AttachmentRuntime | None = None,
    ) -> dict[str, Any]:
        """Normalize command metadata before constructing a typed request.

        MotionPhaseCommand is the execution boundary, while the public planner
        request types are the native boundary.  Keep the conversion explicit
        here so phase identity, collision policy, target entities and
        completion/replan declarations cannot disappear between those layers.
        """

        supplied = dict(request_metadata or {})
        metadata = dict(supplied.get("metadata", {}))
        metadata.setdefault("world_revision", self.scene_revision)
        profile = supplied.get("profile", default_profile)
        if profile is None:
            profile = default_profile
        collision_policy = supplied.get(
            "collision_policy", CollisionPolicy.WORLD_TRANSIT
        )
        if collision_policy is None:
            collision_policy = CollisionPolicy.WORLD_TRANSIT
        if not isinstance(collision_policy, CollisionPolicy):
            collision_policy = CollisionPolicy(str(collision_policy).lower())

        # Native CuRobo only addresses exact world collider paths.  Phase
        # metadata carries logical entity names, so resolve those names once
        # at this simulator-owned request builder and keep the native adapter
        # free of scene-manager lookups or substring/path heuristics.
        collision_options = CollisionOptions.from_mapping(
            supplied.get("collision_options"),
            default_policy=collision_policy,
        )
        collision_scene = getattr(self.robot_port, "collision_scene_manager", None)
        target_entity = supplied.get("active_target", supplied.get("active_object"))
        support_entity = supplied.get("support", supplied.get("support_object"))
        target_paths = self._collision_entity_paths(collision_scene, target_entity)
        support_paths = self._collision_entity_paths(collision_scene, support_entity)
        attachment_runtime = attachment_runtime or getattr(self, "attachment_runtime", None)
        attached_paths = tuple(
            str(path)
            for path in (getattr(attachment_runtime, "attached_obstacle_names", ()) or ())
        )
        option_mapping = collision_options.to_dict()
        if not option_mapping["target_obstacles"] and target_paths:
            option_mapping["target_obstacles"] = list(target_paths)
        if not option_mapping["support_obstacles"] and support_paths:
            option_mapping["support_obstacles"] = list(support_paths)
        if not option_mapping["attached_obstacles"] and attached_paths:
            option_mapping["attached_obstacles"] = list(attached_paths)
        collision_options = CollisionOptions.from_mapping(
            option_mapping,
            default_policy=collision_policy,
        )
        return {
            "phase_id": str(supplied.get("phase_id", "phase")),
            "completion_policy": supplied.get("completion_policy", "default"),
            "replan_policy": supplied.get("replan_policy", "allowed"),
            "collision_policy": collision_policy,
            "collision_options": collision_options,
            "active_target": target_entity,
            "support": support_entity,
            "profile": profile,
            "preplanned_joint_path": supplied.get(
                "preplanned_joint_path", supplied.get("preplanned_trajectory")
            ),
            "metadata": metadata,
            "world_revision": self.scene_revision,
        }

    @staticmethod
    def _collision_entity_paths(scene_manager: Any, entity_name: Any) -> tuple[str, ...]:
        """Resolve one logical scene entity to its exact native collider paths."""

        if scene_manager is None or entity_name is None:
            return ()
        records = getattr(scene_manager, "records", None)
        if not isinstance(records, Mapping):
            return ()
        record = records.get(str(entity_name))
        if record is None:
            return ()
        paths = (
            record.get("collision_prim_paths", ())
            if isinstance(record, Mapping)
            else getattr(record, "collision_prim_paths", ())
        )
        if paths is None or isinstance(paths, (str, bytes)):
            paths = (paths,) if paths else ()
        return tuple(dict.fromkeys(str(path) for path in paths if str(path)))

    def _single_pose_native_kwargs(self) -> dict[str, Any]:
        """Map the typed single-pose profile to CuRobo V2 parameters."""

        return {
            # Keep the native-v2 pose query equivalent to the pre-migration
            # controller call.  CuRobo's implicit-goal mode is required for
            # the GoalToolPose contract used by Pick terminal queries; relying
            # on a backend default makes typed single fallbacks diverge from
            # the old ``plan_pose(..., use_implicit_goal=True)`` call.
            "use_implicit_goal": True,
            "max_attempts": self.max_plan_attempts,
            "enable_graph_attempt": self.single_graph_attempt,
        }

    def _batch_pose_native_kwargs(self) -> dict[str, Any]:
        """Map the typed batch-pose profile to CuRobo V2 parameters."""

        return {
            "use_implicit_goal": True,
            "max_attempts": self.batch_max_attempts,
            # Pick/Place batches are candidate searches.  Requiring all
            # sampled candidates to converge makes one bad sample invalidate
            # otherwise usable paths and needlessly burns every retry.
            "success_ratio": 1.0 / float(CUROBO_BATCH_SIZE),
            "enable_graph_attempt": self.batch_graph_attempt,
        }

    def _single_cspace_native_kwargs(self) -> dict[str, Any]:
        """Map the typed single-cspace profile to CuRobo V2 parameters."""

        return {
            "max_attempts": self.max_plan_attempts,
            "enable_graph_attempt": self.single_graph_attempt,
        }

    def plan_pose(
        self,
        position,
        orientation,
        *,
        start_state=None,
        context=None,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> PlanResult:
        del context
        self._refresh_reference_world_for_planning()
        if start_state is None:
            start_state = self.arm_joint_state(self.robot_port.robot.get_joints_state())
        if getattr(getattr(start_state, "position", None), "ndim", 1) == 1:
            start_state = start_state.unsqueeze(0)
        goal = self._make_tool_goal(position, orientation)
        common = self._request_common_kwargs(
            request_metadata,
            default_profile=PlanningProfile.TRANSIT,
            attachment_runtime=getattr(self, "attachment_runtime", None),
        )
        result = self.planner_runtime.plan_pose(
            PosePlanRequest(
                goal=goal,
                start_state=start_state,
                kwargs=self._single_pose_native_kwargs(),
                **common,
            )
        )
        return result

    def plan_pose_batch(
        self,
        positions,
        orientations,
        *,
        start_state=None,
        batch_size=None,
        start_paths=None,
        context=None,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> PlanResult:
        del context
        self._refresh_reference_world_for_planning()
        self.ensure_batch_planner()
        if start_paths is not None:
            start_state = self.batch_start_state_from_paths(
                start_paths, batch_size=len(positions)
            )
        elif start_state is None:
            start_state = self.arm_joint_state(self.robot_port.robot.get_joints_state(), repeat=len(positions))
        elif getattr(getattr(start_state, "position", None), "ndim", 1) == 1:
            start_state = start_state.unsqueeze(0)
            if len(positions) > 1:
                start_state = start_state.repeat((len(positions), 1))
        goals = self._make_tool_goal(positions, orientations, batch_size=len(positions))
        common = self._request_common_kwargs(
            request_metadata,
            default_profile=PlanningProfile.TRANSIT,
            attachment_runtime=getattr(self, "batch_attachment_runtime", None),
        )
        result = self.planner_runtime.plan_pose_batch(
            BatchPosePlanRequest(
                goals=goals,
                start_state=start_state,
                batch_size=batch_size or len(positions),
                kwargs=self._batch_pose_native_kwargs(),
                **common,
            )
        )
        if not result.is_success and self.batch_single_fallback_candidates:
            fallback_indices = None
            if start_paths is not None:
                fallback_indices = tuple(
                    index for index, path in enumerate(start_paths) if path is not None
                )
            result = self._single_fallback_for_batch(
                result,
                positions,
                orientations,
                start_state,
                common,
                candidate_indices=fallback_indices,
            )
        if not result.is_success:
            self._log_batch_failure(result, phase_id=common.get("phase_id"))
        return result

    def _single_fallback_for_batch(
        self,
        batch_result: BatchPlanResult,
        positions,
        orientations,
        start_state,
        common: Mapping[str, Any],
        candidate_indices=None,
    ) -> BatchPlanResult:
        """Retry a few failed candidates through the shared single planner.

        Batch CuRobo is useful for candidate parallelism, but its result is
        not a proof that every candidate can be solved by the normal single
        planner.  Keep the fallback here, beside the native planning calls,
        so skills only receive the same candidate mask/path contract.
        """

        count = len(positions)
        limit = min(int(self.batch_single_fallback_candidates), count)
        if candidate_indices is None:
            indices = tuple(range(limit))
        else:
            indices = tuple(
                index
                for index in candidate_indices
                if 0 <= int(index) < count
            )[:limit]
        LOGGER.warning(
            "[CuRoboBatchFallback] robot=%s arm=%s phase=%s batch=0/%d "
            "trying_single_candidates=%s",
            self.robot_port.name,
            self.robot_port.lr_name,
            common.get("phase_id"),
            count,
            indices,
        )
        trajectories = [None] * count
        success = [False] * count
        batch_position = getattr(start_state, "position", None)
        if batch_position is None:
            return batch_result
        if len(getattr(batch_position, "shape", ())) == 1:
            batch_position = batch_position.unsqueeze(0)
        from curobo.types import JointState

        selected = None
        for index in indices:
            try:
                single_state = JointState.from_position(
                    batch_position[index : index + 1],
                    joint_names=self.planner_names,
                )
                metadata = dict(common.get("metadata", {}))
                metadata["batch_fallback_candidate"] = int(index)
                single_common = dict(common)
                single_common["phase_id"] = f"{common.get('phase_id', 'phase')}.single_fallback"
                single_common["metadata"] = metadata
                goal = self._make_tool_goal(positions[index], orientations[index])
                single_result = self.planner_runtime.plan_pose(
                    PosePlanRequest(
                        goal=goal,
                        start_state=single_state,
                        kwargs=self._single_pose_native_kwargs(),
                        **single_common,
                    )
                )
                if single_result.is_success and single_result.trajectory is not None:
                    success[index] = True
                    trajectories[index] = single_result.trajectory
                    selected = index
                    LOGGER.info(
                        "[CuRoboBatchFallback] robot=%s arm=%s phase=%s "
                        "candidate=%d success total_time=%s",
                        self.robot_port.name,
                        self.robot_port.lr_name,
                        common.get("phase_id"),
                        index,
                        single_result.metrics.get("total_time"),
                    )
                    break
            except Exception as exc:
                LOGGER.warning(
                    "[CuRoboBatchFallback] robot=%s arm=%s phase=%s "
                    "candidate=%d error=%r",
                    self.robot_port.name,
                    self.robot_port.lr_name,
                    common.get("phase_id"),
                    index,
                    exc,
                )

        if selected is None:
            return batch_result
        metrics = dict(batch_result.metrics)
        metrics["single_fallback_candidate"] = int(selected)
        return BatchPlanResult(
            success=success,
            trajectories=trajectories,
            status="ok",
            source="single_fallback",
            selected_candidate_index=selected,
            metrics=metrics,
            phase_id=batch_result.phase_id,
            profile=batch_result.profile,
            collision_policy=batch_result.collision_policy,
            world_revision=batch_result.world_revision,
            candidate_indices=tuple(range(count)),
        )

    def _log_batch_failure(self, result: PlanResult, *, phase_id: str | None) -> None:
        """Emit compact native diagnostics only after a batch is empty.

        Place/Pick candidate code should only consume the typed success mask.
        Keeping the failure audit here avoids duplicating native-result
        inspection in each skill and makes a zero-candidate batch actionable
        without changing the Physics Schema collision policy.
        """

        metrics = getattr(result, "metrics", {}) or {}
        feasible = metrics.get("feasible")
        converged = metrics.get("converged")
        debug = metrics.get("debug_info", metrics.get("debug_info_keys"))

        def count_true(value):
            if isinstance(value, (list, tuple)):
                return sum(count_true(item) for item in value)
            return int(bool(value))

        LOGGER.warning(
            "[CuRoboBatchDebug] robot=%s arm=%s phase=%s status=%s "
            "success=%d/%d feasible=%s converged=%s debug=%s",
            self.robot_port.name,
            self.robot_port.lr_name,
            phase_id,
            getattr(result, "status", None),
            int(getattr(result, "success_count", 0)),
            len(getattr(result, "success_mask", ()) or ()),
            count_true(feasible) if feasible is not None else None,
            count_true(converged) if converged is not None else None,
            debug,
        )
        diagnostic = self.robot_port.native_start_collision_diagnostic
        if callable(diagnostic):
            try:
                diagnostic()
            except Exception as exc:  # diagnostics must not mask planning failure
                LOGGER.warning(
                    "[CuRoboBatchDebug] native start collision audit unavailable "
                    "robot=%s arm=%s error=%r",
                    self.robot_port.name,
                    self.robot_port.lr_name,
                    exc,
                )

    def plan_cspace(
        self,
        goal_positions,
        *,
        start_state=None,
        context=None,
        request_metadata: Mapping[str, Any] | None = None,
    ) -> PlanResult:
        from curobo.types import JointState

        del context
        self._refresh_reference_world_for_planning()
        goal = JointState.from_position(
            self.robot_port.tensor_args.to_device(goal_positions),
            joint_names=self.planner_names,
        ).unsqueeze(0)
        if start_state is None:
            start_state = self.arm_joint_state(
                self.robot_port.robot.get_joints_state()
            ).unsqueeze(0)
        common = self._request_common_kwargs(
            request_metadata,
            default_profile=PlanningProfile.CSPACE,
        )
        return self.planner_runtime.plan_cspace(
            CspacePlanRequest(
                goal_positions=goal,
                start_state=start_state,
                kwargs=self._single_cspace_native_kwargs(),
                **common,
            )
        )

    def compute_fk(self, joint_positions, *, joint_names=None):
        from curobo.types import JointState

        names = list(self.planner_names if joint_names is None else joint_names)
        state = JointState.from_position(self.robot_port.tensor_args.to_device(joint_positions), joint_names=names)
        state = state.reorder(self.planner_names)
        out = self.native_planner.compute_kinematics(state)
        pose = out.tool_poses.get_link_pose(self.native_planner.tool_frames[0])
        return pose.position.detach().cpu().numpy(), pose.quaternion.detach().cpu().numpy()

    def _compute_cartesian_fk_batch(self, joint_positions, joint_names=None):
        """Compute tool positions for a named trajectory in one FK call."""

        import torch
        from curobo.types import JointState

        values = self.tensor_args.to_device(joint_positions)
        if values.ndim != 2:
            raise ValueError(
                "batched Cartesian FK requires a [time, dof] position tensor, "
                f"got shape {tuple(values.shape)}"
            )
        planner_names = self._planner_joint_names()
        source_names = planner_names if joint_names is None else list(joint_names)
        if len(source_names) != values.shape[-1] or len(set(source_names)) != len(source_names):
            raise ValueError("batched Cartesian FK joint_names do not match position DOF")
        if set(source_names) != set(planner_names):
            raise ValueError(
                "batched Cartesian FK joint contract does not match the native planner"
            )
        if source_names != planner_names:
            reorder = [source_names.index(name) for name in planner_names]
            values = values[..., reorder].contiguous()
        state = JointState.from_position(values.contiguous(), joint_names=planner_names)
        with torch.inference_mode():
            out = self.native_planner.compute_kinematics(state)
        pose = out.tool_poses.get_link_pose(self.native_planner.tool_frames[0])
        return pose.position.detach().cpu().numpy()

    def arm_joint_state(self, sim_state, *, repeat=1):
        from curobo.types import JointState

        positions = np.asarray(sim_state.positions, dtype=float)
        names = list(self.robot_port.robot.dof_names)
        values = positions[self.robot_port.arm_indices]
        state = JointState.from_position(
            self.robot_port.tensor_args.to_device(values), joint_names=self.robot_port.raw_js_names
        )
        state = state.reorder(self.planner_names)
        if repeat > 1:
            state = state.unsqueeze(0).repeat((repeat, 1))
        return state

    def joint_state_from_trajectory(self, path):
        """Convert one normalized trajectory to a planner JointState.

        ``PlannerRuntime`` deliberately publishes native-independent
        ``JointTrajectory`` values.  The conversion back to CuRobo belongs at
        this runtime boundary: native list, NumPy, and tensor positions all
        pass through the controller's ``DeviceCfg`` before the public
        ``JointState.from_position`` constructor is used.
        """

        from curobo.types import JointState

        positions = getattr(path, "positions", None)
        if positions is None:
            positions = getattr(path, "position", None)
        position, names = normalize_named_trajectory(
            positions,
            getattr(path, "joint_names", None),
            self.robot_port.tensor_args,
        )
        state = JointState.from_position(position, joint_names=names)
        return state.reorder(self.planner_names)

    def joint_state_from_path_endpoint(self, path):
        """Convert one normalized trajectory endpoint to a planner JointState."""

        from curobo.types import JointState

        trajectory = self.joint_state_from_trajectory(path)
        endpoint = trajectory.position[-1]
        return JointState.from_position(
            endpoint,
            joint_names=list(trajectory.joint_names),
        )

    def batch_start_state_from_paths(self, paths, *, batch_size=None):
        from curobo.types import JointState

        expected_count = len(paths) if batch_size is None else int(batch_size)
        if expected_count <= 0:
            raise ValueError("batch start state requires at least one candidate")
        normalized_paths = list(paths)
        normalized_paths = normalized_paths[:expected_count]
        normalized_paths.extend([None] * (expected_count - len(normalized_paths)))

        states = []
        for path in normalized_paths:
            if path is None:
                states.append(self.arm_joint_state(self.robot_port.robot.get_joints_state()))
                continue
            states.append(self.joint_state_from_path_endpoint(path))
        positions = self.robot_port.tensor_args.to_device(
            np.stack([state.position.detach().cpu().numpy() for state in states])
        )
        if getattr(positions, "ndim", 0) != 2 or positions.shape[0] != expected_count:
            raise RuntimeError(
                "batch start state must contain one joint row per candidate: "
                f"expected={expected_count}, shape={getattr(positions, 'shape', None)}"
            )
        return JointState.from_position(positions, joint_names=self.planner_names)

    def attach_native(
        self,
        paths,
        joint_state,
        meshes,
        *,
        link_name="attached_object",
        world_objects_pose_offset=None,
        planner=None,
    ):
        from curobo.sphere_fit import SphereFitType

        planner = planner or self.native_planner
        if not paths:
            raise ValueError("native attachment requires non-empty obstacle paths")
        if joint_state is None:
            joint_state = self.arm_joint_state(self.robot_port.robot.get_joints_state())
        if meshes is None:
            meshes, world_objects_pose_offset = self._attachment_geometry(paths)
        planner.attachment_manager.attach(
            joint_state,
            meshes,
            link_name=link_name,
            num_spheres=max(1, self._attached_sphere_count(link_name, 1, planner=planner)),
            surface_radius=0.001,
            sphere_fit_type=SphereFitType.VOXEL,
            world_objects_pose_offset=world_objects_pose_offset,
            disable_obstacle_names=paths,
        )
        return paths

    def attach_collision_object(
        self,
        obj_prim_paths,
        link_name="attached_object",
        world_objects_pose_offset=None,
    ):
        """Attach exact scene colliders to the execution planner."""

        paths = [str(path).strip() for path in obj_prim_paths]
        if not paths or any(not path for path in paths):
            raise ValueError("attachment requires non-empty exact collider paths")
        self.native_scene_adapter.require_obstacles(paths)
        meshes, attachment_offset = self._attachment_geometry(paths)
        if world_objects_pose_offset is not None:
            attachment_offset = world_objects_pose_offset
        joint_state = self.arm_joint_state(self.robot_port.robot.get_joints_state())
        self.attach_object(
            AttachmentSpec(
                name="|".join(paths),
                state=joint_state,
                meshes=meshes,
                link_name=link_name,
                pose_offset=attachment_offset,
                disable_obstacle_names=tuple(paths),
            )
        )
        if self.batch_attachment_runtime is not None:
            self.batch_attachment_runtime.detach()
        return True

    def sync_native_batch_attachment(
        self,
        link_name="attached_object",
        world_objects_pose_offset=None,
    ) -> bool:
        """Synchronize the held object with the candidate batch planner."""

        if not self.robot_port.batch_capability:
            return False
        self.ensure_batch_planner()
        require_adapter = self._require_batch_scene_adapter
        if not callable(require_adapter):
            raise RuntimeError(
                "native batch attachment requires the collision-scene "
                "manager's strict target-batch adapter"
            )
        batch_adapter = require_adapter()
        if batch_adapter is None or not callable(
            getattr(batch_adapter, "require_obstacles", None)
        ):
            raise RuntimeError(
                "collision-scene manager did not provide a strict target-batch "
                "scene adapter"
            )
        paths = list(self.attachment_runtime.attached_obstacle_names)
        if not paths:
            raise RuntimeError(
                "cannot synchronize native batch attachment without an attached object"
            )
        batch_adapter.require_obstacles(paths)
        batch_attachment_runtime = self.batch_attachment_runtime
        if batch_attachment_runtime is None:
            raise RuntimeError("batch attachment runtime was not initialized")
        if list(batch_attachment_runtime.attached_obstacle_names) == paths:
            return True

        meshes, attachment_offset = self._attachment_geometry(paths)
        if world_objects_pose_offset is not None:
            attachment_offset = world_objects_pose_offset
        batch_attachment_runtime.attach(
            AttachmentSpec(
                name="|".join(paths),
                state=self.arm_joint_state(self.robot_port.robot.get_joints_state()),
                meshes=meshes,
                link_name=link_name,
                pose_offset=attachment_offset,
                disable_obstacle_names=tuple(paths),
            )
        )
        return True

    def detach_attachment(self):
        """Detach the active object from execution and candidate planners."""

        self.detach_object()
        if self.batch_attachment_runtime is not None:
            self.batch_attachment_runtime.detach()

    def has_attached_collision_spheres(self, link_name="attached_object") -> bool:
        spheres = self.native_planner.kinematics.config.kinematics_config.get_link_spheres(
            link_name
        )
        return bool(np.any(spheres[:, 3].detach().cpu().numpy() > 0.0))

    def _attached_sphere_count(self, link_name, object_count, *, planner=None):
        planner = planner or self.native_planner
        total = planner.kinematics.config.kinematics_config.get_number_of_spheres(
            link_name
        )
        return max(1, int(total) // max(1, int(object_count)))

    def _attachment_geometry(self, object_names):
        """Build local meshes and the current pose for native attachment."""

        from curobo.types import Pose
        from core.planning.native_bridge import Mesh

        obstacles = []
        current_poses = []
        for object_name in object_names:
            try:
                obstacle = self.native_scene_adapter.get_obstacle_geometry(object_name)
            except Exception as exc:
                raise ValueError(
                    f"attach collision prim is not in native scene model: {object_name}"
                ) from exc
            obstacles.append(obstacle)
            if (
                self.robot_port.obstacle_pose is not None
                and self.robot_port.collision_world_mode == "physics_schema"
            ):
                current_poses.append(self.robot_port.obstacle_pose(object_name))
            else:
                pose = getattr(obstacle, "pose", None)
                if pose is None:
                    raise ValueError(f"native obstacle has no pose: {object_name}")
                current_poses.append(
                    Pose.from_list(list(pose), device_cfg=self.robot_port.tensor_args)
                )

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
                    f"native attachment mesh has invalid vertices: {obstacle.name}"
                )
            object_matrix = current_pose.get_numpy_matrix()[0]
            world_vertices = (
                object_matrix[:3, :3] @ local_vertices.T
            ).T + object_matrix[:3, 3]
            anchor_vertices = (
                anchor_inverse[:3, :3] @ world_vertices.T
            ).T + anchor_inverse[:3, 3]
            obstacle_faces = np.asarray(mesh.faces, dtype=np.int64)
            if obstacle_faces.ndim != 2 or obstacle_faces.shape[1] != 3:
                raise ValueError(
                    f"native attachment mesh has invalid faces: {obstacle.name}"
                )
            vertices.append(anchor_vertices)
            faces.append(obstacle_faces + vertex_offset)
            vertex_offset += anchor_vertices.shape[0]

        if not vertices:
            raise ValueError("native attachment requires at least one mesh")
        return [
            Mesh(
                name="__native_attached_object__",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                vertices=np.concatenate(vertices, axis=0),
                faces=np.concatenate(faces, axis=0),
            )
        ], anchor_pose

    def attach_object(self, spec: AttachmentSpec):
        return self.attachment_runtime.attach(spec)

    def detach_object(self):
        return self.attachment_runtime.detach()

    def reset_attachments(self) -> None:
        """Detach execution and candidate attachments through their owners."""

        if self.batch_attachment_runtime is not None:
            self.batch_attachment_runtime.detach()
        self.attachment_runtime.detach()

    def destroy(self) -> None:
        """Release planner resources and detach all scene subscriptions."""

        if self.batch_attachment_runtime is not None:
            self.batch_attachment_runtime.detach()
        self.attachment_runtime.detach()
        self.planner_runtime.unregister_scene_adapter(self.native_scene_adapter)
        # PlannerRuntime owns the native single/batch instances and its scene
        # subscription.  Clear the lightweight scene registry as well so a
        # runtime that is retained by a test/worker cannot fan out updates
        # after native teardown.
        try:
            self.planner_runtime.destroy()
        finally:
            self.scene_runtime.clear()

    close = destroy

    def __enter__(self) -> "MotionPlannerRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()


__all__ = [
    "MotionPlannerRuntime",
    "RobotPort",
]
