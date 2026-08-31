from __future__ import annotations
from dataclasses import dataclass
import logging
import time
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
from core.planning.planner_runtime import PlannerRuntime
from core.planning.motion_command import MotionPhase, MotionPhaseCommand
from core.controllers.curobo.trajectory import normalize_named_trajectory
from core.utils.constants import CUROBO_BATCH_SIZE
LOGGER = logging.getLogger("de_logger")
if TYPE_CHECKING:
    from core.planning.native_planner_factory import PlannerBuildConfig
else:
    PlannerBuildConfig = Any
@dataclass
class RobotPort:
    name: str
    lr_name: str
    task_cfg: Mapping[str, Any]
    robot: Any
    tensor_args: Any
    arm_spec: Any
    arm_indices: Any
    raw_js_names: list[str]
    constrain_grasp_approach: bool
    collision_scene_manager: Any = None
    collision_world_mode: str = "physics_schema"
    obstacle_pose: Callable[[str], Any] | None = None
    native_start_collision_diagnostic: Callable[[], Any] | None = None
    interpolation_dt: float = 0.01
    ik_solver: Any = None
    kin_model: Any = None
class MotionPlannerRuntime(PlannerRuntime):
    def __init__(
        self,
        planner_build_config: PlannerBuildConfig,
        robot_port: RobotPort,
        *,
        world: Any = None,
        phase_executor: PhaseExecutor | None = None,
        execution_state: Any = None,
        setup: Any = None,
        execution: Any = None,
    ) -> None:
        self.planner_build_config = planner_build_config
        self.robot_port = robot_port
        self.execution_state = execution_state
        self.setup = setup
        self.execution = execution
        self._timing_scope = None
        if self.execution is not None:
            self.execution.runtime = self
        if world is None:
            raise RuntimeError("MotionPlannerRuntime requires an explicit planning world")
        self.planning_world = world
        self.phase_executor = phase_executor or PhaseExecutor()
        self._configure_native_runtime()
        self.attachment_runtime = AttachmentRuntime(
            manager=self.native_planner.attachment_manager,
            scene=self,
        )
        self.batch_attachment_runtime = None
        self._attachment_spec = None
    @property
    def native_planner(self):
        return self.ensure_planner()

    def ensure_batch_planner(self):
        batch = super().ensure_batch_planner()
        if self.batch_attachment_runtime is None:
            self.batch_attachment_runtime = AttachmentRuntime(
                manager=batch.attachment_manager,
                scene=self,
            )
            if self._attachment_spec is not None:
                self.batch_attachment_runtime.attach(
                    self._batch_attachment_spec(self._attachment_spec, batch)
                )
        return batch

    @staticmethod
    def _batch_attachment_spec(spec: AttachmentSpec, planner) -> AttachmentSpec:
        state = spec.state
        if getattr(getattr(state, "position", None), "ndim", 1) == 1:
            state = state.unsqueeze(0)
        return AttachmentSpec(
            name=spec.name,
            state=state,
            meshes=spec.meshes,
            link_name=spec.link_name,
            pose_offset=spec.pose_offset,
            disable_obstacle_names=spec.disable_obstacle_names,
            num_spheres=spec.num_spheres,
            surface_radius=spec.surface_radius,
            sphere_fit_type=spec.sphere_fit_type,
        )
    @property
    def planner_names(self) -> list[str]:
        return list(self.native_planner.joint_names)
    def _configure_native_runtime(self) -> None:
        from core.planning.native_planner_factory import (
            NativePlannerFactory,
            resolve_native_robot_config,
        )
        port = self.robot_port
        pick_cfg = dict(port.task_cfg).get("planning", {}).get("pick_place", {})
        graph_enabled = bool(pick_cfg.get("enable_graph", False))
        try:
            max_attempts = max(1, int(pick_cfg.get("max_plan_attempts", 8)))
        except (TypeError, ValueError):
            max_attempts = 8
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
        self.graph_enabled = graph_enabled
        self.single_graph_attempt = max(0, min(1, max_attempts - 1)) if graph_enabled else max_attempts
        factory = NativePlannerFactory(
            self.planner_build_config,
            robot_config=resolve_native_robot_config,
            pose_criteria=self._set_pose_criteria,
            collision_cache=self._collision_cache(),
            interpolation_dt=interpolation_dt,
        )
        super().__init__(
            name=port.name,
            profile=PlannerRuntimeProfile(
                name=port.name,
                device=port.tensor_args,
                max_batch_size=CUROBO_BATCH_SIZE,
                batch_enabled=True,
                warmup_config={
                    "enable_graph": graph_enabled,
                    "num_warmup_iterations": warmup_iterations,
                },
            ),
            planner_factory=factory.build_single,
            batch_planner_factory=factory.build_batch,
            world=self.planning_world,
            scene_revision=0,
        )
        port.ik_solver = self.native_planner.ik_solver
        port.kin_model = self.native_planner.kinematics
        port.interpolation_dt = float(self.native_planner.trajopt_solver.config.interpolation_dt)
    def _collision_cache(self):
        world = self.planning_world
        if world is None:
            raise RuntimeError("planning world must be available before planner construction")
        cuboid_count = len(getattr(world, "cuboid", None) or [])
        return {"cuboid": cuboid_count} if cuboid_count else {}
    def _approach_axis(self) -> int:
        spec = getattr(self.robot_port, "arm_spec", None)
        axis = getattr(spec, "grasp_approach_axis", None)
        if axis is not None:
            return int(axis)
        return {"x": 0, "y": 1, "z": 2}[self.robot_port.robot.cfg["ee_axis"]]
    def _goal_tool_pose(self, ee_translation, ee_orientation, *, batch_size=1):
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
    def sync_dynamic_poses(self, *, force: bool = True):
        manager = self.robot_port.collision_scene_manager
        if manager is None:
            raise RuntimeError("MotionPlannerRuntime dynamic-pose synchronization is unavailable")
        step_id = int(getattr(self.execution_state, "step_idx", 0) or 0)
        return manager.sync_dynamic_poses(step_id, interval_steps=1, force=bool(force))
    def reset_pose_cost_metric(self):
        from core.planning.native_bridge import ToolPoseCriteria
        criteria = ToolPoseCriteria(device_cfg=self.robot_port.tensor_args)
        self._set_pose_criteria(self.native_planner, criteria)
        if self.batch_planner is not None:
            self._set_pose_criteria(self.batch_planner, criteria)
        return True
    def ee_pose(self):
        return self.execution.get_ee_pose()
    def arm_base_pose(self):
        return self.execution.get_armbase_pose()
    def execution_status(self, command=None):
        return self.execution.execution_status(command)
    def hold(self, reason=None):
        del reason
        return self.execution.hold_action()
    def complete_contact_phase(self, command):
        return self.execution.complete_terminal_place_on_contact(command)
    def push_timing_scope(self, scope):
        previous = self._timing_scope
        self._timing_scope = scope
        return previous
    def restore_timing_scope(self, previous):
        self._timing_scope = previous
    def clear_timing_scope(self, scope=None):
        if scope is None or self._timing_scope is scope:
            self._timing_scope = None
    def update_world(
        self,
        world,
        *,
        revision=None,
        force=False,
    ):
        self.planning_world = world
        return PlannerRuntime.update_world(self, world, revision=revision, force=force)
    def update_obstacle_poses(
        self,
        poses,
        *,
        force: bool = False,
        revision=None,
    ):
        del force
        return PlannerRuntime.update_obstacle_poses(self, poses, revision=revision)
    def check_current_start_state(self):
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
    def transition_target(
        self,
        object_name: str,
        support_name: str | None = None,
        *,
        collision_policy: CollisionPolicy | str | None = None,
    ):
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
        elif phase is MotionPhase.CARRY_HOME and object_name:
            self.assert_attached_owner(object_name)
        elif phase is MotionPhase.TERMINAL_PLACE_DESCENT:
            manager.begin_placement_descent(object_name, support_name, robot, arm)
        elif phase is MotionPhase.DETACH_AND_SETTLE:
            manager.detach_target(object_name, robot, arm)
        elif phase is MotionPhase.TERMINAL_RETREAT:
            self.transition_target(object_name, collision_policy=CollisionPolicy.RETREAT)
        elif phase is MotionPhase.RESTORE_WORLD:
            self.restore_world(object_name)
        return command.preplanned_joint_path
    @property
    def arm_name(self) -> str:
        return self.robot_port.lr_name
    @property
    def robot(self):
        return self.robot_port.robot
    @property
    def grasp_approach_axis(self) -> int:
        return self._approach_axis()
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
    def _command_path(self, path):
        if path is None:
            return None
        if not isinstance(path, JointTrajectory):
            raise TypeError(
                "controller command paths require named JointTrajectory, "
                f"got {type(path).__name__}"
            )
        trajectory = path
        if set(self.raw_joint_names) == set(trajectory.joint_names):
            return trajectory.reorder(self.raw_joint_names)
        active = self.joint_state_from_trajectory(trajectory)
        full_native = self.native_planner.kinematics.get_full_js(active)
        if not set(self.raw_joint_names).issubset(full_native.joint_names):
            raise ValueError(
                "native planner result cannot be mapped to controller arm names: "
                f"result={full_native.joint_names}, arm={self.raw_joint_names}"
            )
        arm_native = full_native.reorder(list(self.raw_joint_names))
        return JointTrajectory(
            arm_native.position,
            joint_names=self.raw_joint_names,
        )
    def _install_command_plan(
        self,
        trajectory,
        *,
        target_position=None,
        target_orientation=None,
        phase_name: str = "unknown",
        cached: bool,
    ):
        if trajectory is None or len(trajectory) == 0:
            raise ValueError(f"{phase_name} received an empty native-v2 path")
        if self.execution_state is None:
            raise RuntimeError("runtime execution state is not bound")
        self.phase_executor.install(trajectory)
        if target_position is not None:
            self.execution_state.ee_trans = self.tensor_args.to_device(target_position)
        if target_orientation is not None:
            self.execution_state.ee_ori = self.tensor_args.to_device(target_orientation)
        LOGGER.info(
            "[PhaseDebug] selected-plan robot=%s arm=%s phase=%s waypoints=%d stride=%d cached=%s",
            self.name,
            self.arm_name,
            phase_name,
            len(self.phase_executor.current),
            int(getattr(self.setup, "ds_ratio", 1) or 1),
            cached,
        )
        return trajectory
    def _refresh_reference_world_for_planning(self):
        manager = self.robot_port.collision_scene_manager
        if manager is None:
            return None
        manager.refresh_controller_reference_world(self.scene_port)
        manager.apply_controller_planning_exclusions(self.scene_port)
    def plan_pose_batch(
        self,
        positions,
        orientations,
        *,
        start_state=None,
        start_paths=None,
        phase_id="plan_pose_batch",
        collision_policy=None,
        active_target=None,
        support=None,
        collision_options=None,
        metadata=None,
    ) -> BatchPlanResult:
        positions = np.asarray(positions, dtype=float)
        orientations = np.asarray(orientations, dtype=float)
        count = len(positions)
        if count == 0 or count > CUROBO_BATCH_SIZE:
            raise ValueError(f"native batch requires 1..{CUROBO_BATCH_SIZE} candidates")
        wall_start = time.perf_counter()
        refresh_start = wall_start
        self._refresh_reference_world_for_planning()
        refresh_time = time.perf_counter() - refresh_start
        planner_start = time.perf_counter()
        self.ensure_batch_planner()
        planner_time = time.perf_counter() - planner_start
        state_start = time.perf_counter()
        if start_paths is not None:
            start_state = self.batch_start_state_from_paths(start_paths)
        elif start_state is None:
            start_state = self.arm_joint_state(
                self.robot_port.robot.get_joints_state(), repeat=count
            )
        state_time = time.perf_counter() - state_start
        common = self._request_common(
            phase_id=phase_id,
            default_profile=PlanningProfile.TRANSIT,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            collision_options=collision_options,
            metadata=metadata,
            attachment_runtime=self.batch_attachment_runtime or self.attachment_runtime,
        )
        request_start = time.perf_counter()
        try:
            result = super().plan_pose_batch(
                BatchPosePlanRequest(
                    goals=self._goal_tool_pose(positions, orientations, batch_size=count),
                    start_state=start_state,
                    batch_size=count,
                    use_implicit_goal=True,
                    max_attempts=1,
                    success_ratio=1.0,
                    enable_graph_attempt=self.single_graph_attempt,
                    **common,
                )
            )
        finally:
            request_time = time.perf_counter() - request_start
            LOGGER.info(
                "[CuroboBatchTiming] robot=%s arm=%s phase=%s candidates=%d "
                "world_refresh=%.3fs planner_init_warmup=%.3fs state_prepare=%.3fs "
                "native_plan=%.3fs total=%.3fs",
                self.name,
                self.arm_name,
                phase_id,
                count,
                refresh_time,
                planner_time,
                state_time,
                request_time,
                time.perf_counter() - wall_start,
            )
        return result
    def _request_common(
        self,
        *,
        phase_id: str,
        default_profile: PlanningProfile,
        collision_policy: CollisionPolicy | str | None = None,
        active_target: str | None = None,
        support: str | None = None,
        collision_options: CollisionOptions | None = None,
        profile: PlanningProfile | str | None = None,
        completion_policy: Any = "default",
        replan_policy: Any = "allowed",
        metadata: Mapping[str, Any] | None = None,
        attachment_runtime: AttachmentRuntime | None = None,
    ) -> dict[str, Any]:
        metadata = dict(metadata or {})
        metadata.setdefault("world_revision", self.scene_revision)
        if profile is None:
            profile = default_profile
        if collision_policy is None:
            collision_policy = CollisionPolicy.WORLD_TRANSIT
        if not isinstance(collision_policy, CollisionPolicy):
            collision_policy = CollisionPolicy(str(collision_policy).lower())
        attachment_runtime = attachment_runtime or self.attachment_runtime
        collision_options = self._resolve_collision_options(
            collision_options,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            attachment_runtime=attachment_runtime,
        )
        return {
            "phase_id": str(phase_id),
            "completion_policy": completion_policy,
            "replan_policy": replan_policy,
            "collision_policy": collision_policy,
            "collision_options": collision_options,
            "active_target": active_target,
            "support": support,
            "profile": profile,
            "world_revision": self.scene_revision,
        }
    def _resolve_collision_options(
        self,
        collision_options: CollisionOptions | None,
        *,
        collision_policy: CollisionPolicy,
        active_target: str | None,
        support: str | None,
        attachment_runtime: AttachmentRuntime | None = None,
    ) -> CollisionOptions:
        options = collision_options or CollisionOptions(policy=collision_policy)
        collision_scene = getattr(self.robot_port, "collision_scene_manager", None)
        target_paths = self._collision_entity_paths(collision_scene, active_target)
        support_paths = self._support_collision_paths(collision_scene, support)
        attachment_runtime = attachment_runtime or self.attachment_runtime
        attached_paths = tuple(
            str(path)
            for path in (getattr(attachment_runtime, "attached_obstacle_names", ()) or ())
        )
        if not options.target_obstacles and target_paths:
            options.target_obstacles = target_paths
        if not options.support_obstacles and support_paths:
            options.support_obstacles = support_paths
        if not options.attached_obstacles and attached_paths:
            options.attached_obstacles = attached_paths
        return options
    @staticmethod
    def _collision_entity_paths(scene_manager: Any, entity_name: Any) -> tuple[str, ...]:
        if scene_manager is None or entity_name is None:
            return ()
        records = getattr(scene_manager, "records", None)
        if not isinstance(records, Mapping):
            return ()
        record = records.get(str(entity_name))
        if record is None:
            return ()
        paths = record.collision_prim_paths
        if paths is None or isinstance(paths, (str, bytes)):
            paths = (paths,) if paths else ()
        return tuple(dict.fromkeys(str(path) for path in paths if str(path)))
    @staticmethod
    def _support_collision_paths(scene_manager: Any, entity_name: Any) -> tuple[str, ...]:
        if scene_manager is None or entity_name is None:
            return ()
        paths = scene_manager.support_collision_paths(str(entity_name))
        return tuple(sorted(str(path) for path in paths))
    def plan_pose(
        self,
        position,
        orientation,
        *,
        start_state=None,
        context=None,
        command: MotionPhaseCommand | None = None,
        phase_id="plan_pose",
        collision_policy=None,
        active_target=None,
        support=None,
        collision_options=None,
        profile=None,
        completion_policy="default",
        replan_policy="allowed",
        metadata=None,
    ) -> PlanResult:
        del context
        self._refresh_reference_world_for_planning()
        if start_state is None:
            start_state = self.arm_joint_state(self.robot_port.robot.get_joints_state())
        if getattr(getattr(start_state, "position", None), "ndim", 1) == 1:
            start_state = start_state.unsqueeze(0)
        goal = self._goal_tool_pose(position, orientation)
        if command is not None:
            phase_id = command.phase_id
            collision_policy = command.collision_policy
            active_target = command.active_object
            support = command.support_object
            collision_options = command.collision_options
            profile = command.profile
            completion_policy = command.completion_policy
            replan_policy = command.replan_policy
            metadata = command.metadata
        common = self._request_common(
            phase_id=phase_id,
            default_profile=PlanningProfile.TRANSIT,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            collision_options=collision_options,
            profile=profile,
            completion_policy=completion_policy,
            replan_policy=replan_policy,
            metadata=metadata,
            attachment_runtime=self.attachment_runtime,
        )
        request = PosePlanRequest(
            goal=goal,
            start_state=start_state,
            use_implicit_goal=True,
            max_attempts=self.max_plan_attempts,
            enable_graph_attempt=self.single_graph_attempt,
            **common,
        )
        result = super().plan_pose(
            request
        )
        return result
    def plan_cspace(
        self,
        goal_positions,
        *,
        start_state=None,
        context=None,
        phase_id="plan_cspace",
        collision_policy=None,
        active_target=None,
        support=None,
        collision_options=None,
        profile=None,
        completion_policy="default",
        replan_policy="allowed",
        metadata=None,
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
        common = self._request_common(
            phase_id=phase_id,
            default_profile=PlanningProfile.CSPACE,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            collision_options=collision_options,
            profile=profile,
            completion_policy=completion_policy,
            replan_policy=replan_policy,
            metadata=metadata,
            attachment_runtime=self.attachment_runtime,
        )
        return super().plan_cspace(
            CspacePlanRequest(
                goal_positions=goal,
                start_state=start_state,
                max_attempts=self.max_plan_attempts,
                enable_graph_attempt=self.single_graph_attempt,
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
        position = pose.position.detach().cpu().numpy()
        quaternion = pose.quaternion.detach().cpu().numpy()
        if position.size != 3 or quaternion.size != 4:
            raise ValueError(
                "single FK must return one pose: "
                f"position_shape={position.shape}, quaternion_shape={quaternion.shape}"
            )
        return position.reshape(3), quaternion.reshape(4)
    def _compute_cartesian_fk_batch(self, joint_positions, joint_names=None):
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
    def batch_start_state_from_paths(self, paths):
        states = [
            self.arm_joint_state(self.robot_port.robot.get_joints_state())
            if path is None else self.joint_state_from_path_endpoint(path)
            for path in paths
        ]
        positions = self.robot_port.tensor_args.to_device(
            np.stack([state.position.detach().cpu().numpy() for state in states])
        )
        from curobo.types import JointState
        return JointState.from_position(positions, joint_names=self.planner_names)
    def joint_state_from_trajectory(self, path):
        from curobo.types import JointState
        positions = path.positions
        position, names = normalize_named_trajectory(
            positions,
            getattr(path, "joint_names", None),
            self.robot_port.tensor_args,
        )
        state = JointState.from_position(position, joint_names=names)
        return state.reorder(self.planner_names)
    def joint_state_from_path_endpoint(self, path):
        from curobo.types import JointState
        trajectory = self.joint_state_from_trajectory(path)
        endpoint = trajectory.position[-1]
        return JointState.from_position(
            endpoint,
            joint_names=list(trajectory.joint_names),
        )
    def attach_collision_object(
        self,
        obj_prim_paths,
        link_name="attached_object",
        world_objects_pose_offset=None,
    ):
        from curobo.sphere_fit import SphereFitType
        paths = [str(path).strip() for path in obj_prim_paths]
        if not paths or any(not path for path in paths):
            raise ValueError("attachment requires non-empty exact collider paths")
        self.require_obstacles(paths)
        meshes, attachment_offset = self._build_attachment_geometry(paths)
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
                num_spheres=self._attached_sphere_count(link_name, 1),
                surface_radius=0.001,
                sphere_fit_type=SphereFitType.VOXEL,
            )
        )
        return True
    def detach_attachment(self):
        self.detach_object()
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
        count = int(total) // int(object_count)
        if count <= 0:
            raise ValueError(f"native attachment link has no collision spheres: {link_name}")
        return count
    def _build_attachment_geometry(self, object_names):
        pose_resolver = None
        if (
            self.robot_port.obstacle_pose is not None
            and self.robot_port.collision_world_mode == "physics_schema"
        ):
            pose_resolver = self.robot_port.obstacle_pose
        return self.build_attachment_geometry(
            object_names,
            pose_resolver=pose_resolver,
            device_cfg=self.robot_port.tensor_args,
        )
    def attach_object(self, spec: AttachmentSpec):
        result = self.attachment_runtime.attach(spec)
        self._attachment_spec = spec
        if self.batch_attachment_runtime is not None:
            self.batch_attachment_runtime.attach(
                self._batch_attachment_spec(spec, self.batch_planner)
            )
        return result
    def detach_object(self):
        result = self.attachment_runtime.detach()
        if self.batch_attachment_runtime is not None:
            self.batch_attachment_runtime.detach()
        self._attachment_spec = None
        return result
    def reset_attachments(self) -> None:
        self.detach_object()
    def destroy(self) -> None:
        self.detach_attachment()
        super().destroy()
    close = destroy
__all__ = [
    "MotionPlannerRuntime",
    "RobotPort",
]
