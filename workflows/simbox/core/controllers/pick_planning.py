"""Typed Pick planning port composed from the controller operation ports."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import numpy as np

from core.planning.collision_scene_manager import PlannerScenePort
from core.planning.domain_types import CollisionOptions, CollisionPolicy


class PickPlanningQueryPort(Protocol):
    """Typed query surface consumed by :class:`GraspPlanEvaluator`."""

    lr_name: str
    robot_file: str
    batch_capability: bool
    reference_prim_path: str
    robot_base_path: str
    robot_ee_path: str
    robot_name: str
    robot_family: str
    interpolation_dt: float
    time_dilation_factor: float
    grasp_approach_axis: int
    orientation_adjustment_enabled: bool
    robot_metadata: Mapping[str, Any]
    plan_failure_count: int
    last_command_count: int

    def has_native_obstacle(self, path: str) -> bool: ...

    def transition_target(
        self,
        object_name: str,
        *,
        collision_policy: CollisionPolicy,
    ) -> int: ...

    def plan_pose_batch(
        self,
        positions: Any,
        orientations: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
        start_paths: Any = None,
    ) -> Any: ...

    def plan_pose_result(
        self,
        position: Any,
        orientation: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
    ) -> Any: ...

    def plan_pose_from_path(
        self,
        position: Any,
        orientation: Any,
        start_path: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
    ) -> Any: ...

    def plan_pose_from_joint_positions(
        self,
        position: Any,
        orientation: Any,
        *,
        start_arm_positions: Any = None,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
    ) -> Any: ...

    def measure_cartesian_path(self, path: Any, start: Any, goal: Any) -> Any: ...

    def ee_pose(self) -> Any: ...

    def initial_ee_pose(self) -> Any: ...

    def execution_status(self, command: Any = None) -> Any: ...

    def set_plan_failure_count(self, value: int) -> None: ...

    def phase_complete(self, command: Any) -> bool: ...


class PickPlanningPort(PickPlanningQueryPort):
    """Narrow Pick-facing port for scene transitions and phase helpers.

    Pick does not own a controller component and must not reach through the
    controller façade to ``_phases`` or another implementation detail.  The
    façade composes this port once, supplying the already-wired operation
    callbacks and the formal :class:`PlannerScenePort` used by the collision
    manager.
    """

    def __init__(
        self,
        *,
        scene_port: PlannerScenePort,
        collision_scene_manager: Any,
        update_pose_cost_metric: Callable[[Any], None],
        build_commands: Callable[..., list[Any]],
        arm_base_transform: Callable[[], Any],
        frame_debug: Callable[[], Mapping[str, Any]],
        capture_reference: Callable[[str], None],
        retarget_commands: Callable[[str, Any], Any],
        replan_after_safety: Callable[[str, Any, Any], bool],
        execution_ee_pose: Callable[[], Any],
        phase_complete: Callable[[Any], bool],
        plan_failure_count: Callable[[], int] | None = None,
        set_plan_failure_count: Callable[[int], None] | None = None,
        last_command_count: Callable[[], int] | None = None,
        execution_status: Callable[[Any], Any] | None = None,
        initial_ee_pose: Callable[[], Any] | None = None,
        robot_metadata: Mapping[str, Any] | None = None,
        plan_pose_from_joint_positions: Callable[..., Any] | None = None,
        robot_file: str = "",
        batch_capability: bool = False,
        plan_pose_batch: Callable[..., Any] | None = None,
        plan_pose_result: Callable[..., Any] | None = None,
        plan_pose_from_path: Callable[..., Any] | None = None,
        measure_cartesian_path: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(scene_port, PlannerScenePort):
            raise TypeError("PickPlanningPort requires a formal PlannerScenePort")
        if collision_scene_manager is None:
            raise ValueError("PickPlanningPort requires CollisionSceneManager")
        self.scene_port = scene_port
        self._collision_scene_manager = collision_scene_manager
        self._update_pose_cost_metric = update_pose_cost_metric
        self._build_commands = build_commands
        self._arm_base_transform = arm_base_transform
        self._frame_debug = frame_debug
        self._capture_reference = capture_reference
        self._retarget_commands = retarget_commands
        self._replan_after_safety = replan_after_safety
        self._execution_ee_pose = execution_ee_pose
        self._phase_complete = phase_complete
        self._plan_failure_count = plan_failure_count
        self._set_plan_failure_count = set_plan_failure_count
        self._last_command_count = last_command_count
        self._execution_status = execution_status
        self._plan_pose_from_joint_positions = plan_pose_from_joint_positions
        self.robot_file = str(robot_file)
        self.batch_capability = bool(batch_capability)
        self._plan_pose_batch = plan_pose_batch
        self._plan_pose_result = plan_pose_result
        self._plan_pose_from_path = plan_pose_from_path
        self._measure_cartesian_path = measure_cartesian_path
        self._collision_policy = CollisionPolicy.WORLD_TRANSIT
        self._active_target: str | None = None
        # ``get_ee_pose`` is an arm-base-frame query.  Capture its initial
        # value once, after the composed runtime is fully wired, so moving
        # targets can compare against a stable reference without reaching
        # through the controller façade.
        initial_pose_getter = initial_ee_pose or execution_ee_pose
        self._initial_ee_pose = deepcopy(initial_pose_getter())
        family = Path(self.robot_file).stem.lower()
        metadata = dict(robot_metadata or {})
        metadata.setdefault("robot_file", self.robot_file)
        metadata.setdefault("robot_name", family)
        metadata.setdefault("robot_family", family)
        metadata.setdefault("arm", self.lr_name)
        metadata.setdefault("reference_prim_path", self.reference_prim_path)
        metadata.setdefault("robot_ee_path", self.robot_ee_path)
        self.robot_metadata = MappingProxyType(metadata)

    @property
    def lr_name(self) -> str:
        """Arm identity exposed to the typed grasp-query protocol."""

        return str(self.scene_port.lr_name)

    @property
    def reference_prim_path(self) -> str:
        """The collision/reference frame path owned by the scene port."""

        return str(self.scene_port.reference_prim_path)

    @property
    def robot_base_path(self) -> str:
        """Selected arm-base prim path from the formal scene port."""

        return self.reference_prim_path

    @property
    def robot_ee_path(self) -> str:
        """Selected end-effector prim path from the formal scene port."""

        return str(self.scene_port.robot_ee_path)

    @property
    def robot_name(self) -> str:
        return str(self.robot_metadata["robot_name"])

    @property
    def robot_family(self) -> str:
        return str(self.robot_metadata["robot_family"])

    @property
    def interpolation_dt(self) -> float:
        """Native trajectory interpolation period exposed to Pick timing."""

        return float(self.runtime.robot_port.interpolation_dt)

    @property
    def time_dilation_factor(self) -> float:
        """Execution time dilation for dynamic-pick prediction.

        The typed runtime has no mutable controller-side dilation knob.  Keep
        this explicit and finite so prediction does not need a reflected
        private-field lookup.
        """

        return 1.0

    @property
    def grasp_approach_axis(self) -> int:
        """Return the configured EE approach axis (0/1/2)."""

        configured = self.robot_metadata.get("grasp_approach_axis")
        if configured is not None:
            axis = int(configured)
        else:
            # R5A configs historically used the EE x-axis; Piper configs use
            # the z-axis despite sharing the manual-orientation adjustment
            # path.  Keep this mapping in the typed config surface; Skills
            # never inspect robot-file spelling themselves.
            axis = 0 if "r5a" in self.robot_family else 2
        if axis not in (0, 1, 2):
            raise ValueError(f"grasp_approach_axis must be 0, 1, or 2, got {axis}")
        return axis

    @property
    def orientation_adjustment_enabled(self) -> bool:
        """Whether the robot family supports manual grasp-orientation sweeps."""

        configured = self.robot_metadata.get("orientation_adjustment_enabled")
        if configured is not None:
            return bool(configured)
        return "piper" in self.robot_family or "r5a" in self.robot_family

    @property
    def plan_failure_count(self) -> int:
        if self._plan_failure_count is None:
            raise RuntimeError("PickPlanningPort has no plan-failure counter")
        return int(self._plan_failure_count())

    @plan_failure_count.setter
    def plan_failure_count(self, value: int) -> None:
        self.set_plan_failure_count(value)

    @property
    def last_command_count(self) -> int:
        if self._last_command_count is None:
            raise RuntimeError("PickPlanningPort has no command counter")
        return int(self._last_command_count())

    @property
    def runtime(self) -> Any:
        """Return the planner-owned scene runtime behind the formal scene port."""

        return self.scene_port.runtime

    @property
    def world_revision(self) -> int:
        """Return the revision used by the next typed Pick query."""

        return int(self.runtime.scene_revision)

    def prepare_world(self, object_name: str) -> int:
        """Synchronize the exact Physics world before grasp candidates are planned."""

        self._update_pose_cost_metric(None)
        manager = self._collision_scene_manager
        manager.refresh_controller_reference_world(self.scene_port, force=True)
        manager.sync_dynamic_poses(0, interval_steps=1, force=True)
        return self.transition_target(
            object_name,
            collision_policy=CollisionPolicy.WORLD_TRANSIT,
        )

    def transition_target(
        self,
        object_name: str,
        *,
        collision_policy: CollisionPolicy,
    ) -> int:
        """Apply one typed target collision policy through the scene port."""

        if not isinstance(collision_policy, CollisionPolicy):
            raise TypeError("Pick target transitions require CollisionPolicy")
        self._collision_policy = collision_policy
        self._active_target = str(object_name)
        manager = self._collision_scene_manager
        if collision_policy is CollisionPolicy.WORLD_TRANSIT:
            manager.begin_target_transit(
                object_name,
                self.scene_port.name,
                self.scene_port.lr_name,
            )
        elif collision_policy is CollisionPolicy.TARGET_APPROACH:
            manager.begin_target_approach(
                object_name,
                self.scene_port.name,
                self.scene_port.lr_name,
            )
        else:
            raise ValueError(
                "Pick target transitions support only WORLD_TRANSIT or "
                "TARGET_APPROACH policies"
            )
        return self.world_revision

    def restore_world(self, object_name: str) -> int:
        """Restore the target and all temporary scene exclusions."""

        self._collision_scene_manager.restore_world(object_name)
        self._collision_policy = CollisionPolicy.WORLD_TRANSIT
        self._active_target = str(object_name)
        return self.world_revision

    def diagnose_start_collision(self) -> Mapping[str, Any]:
        """Return the formal scene-port start-state collision diagnostic."""

        return self._collision_scene_manager.diagnose_controller_world_collision(
            self.scene_port
        )

    def source_support(self, object_name: str) -> str | None:
        """Resolve the source support entity for the target lift."""

        return self._collision_scene_manager.get_source_support_entity(object_name)

    def has_native_obstacle(self, path: str) -> bool:
        """Check exact collider presence through the manager's scene port."""

        return bool(
            self._collision_scene_manager.has_native_obstacle(self.scene_port, str(path))
        )

    @staticmethod
    def _request_metadata(
        collision_policy: CollisionPolicy | None,
        *,
        phase_id: str,
        active_target: str | None = None,
        default_policy: CollisionPolicy = CollisionPolicy.WORLD_TRANSIT,
    ) -> dict[str, Any]:
        if collision_policy is None:
            collision_policy = default_policy
        if not isinstance(collision_policy, CollisionPolicy):
            raise TypeError("Pick planner queries require CollisionPolicy")
        metadata = {
            "phase_id": phase_id,
            "collision_policy": collision_policy,
            "active_target": active_target,
        }
        if collision_policy is CollisionPolicy.TARGET_APPROACH:
            # A terminal grasp intentionally enters the target volume with the
            # fingers.  Keep this contact contract on the typed request rather
            # than relying on a prior scene-manager side effect: the native
            # adapter can then apply the exact target-collider exclusion around
            # every single or batch query.  Runtime request construction fills
            # the target's exact collider paths from ``active_target``.
            metadata["collision_options"] = CollisionOptions(
                policy=collision_policy,
                allow_target_contact=True,
                allow_target_finger_contact=True,
            )
        return metadata

    def plan_pose_batch(
        self,
        positions: Any,
        orientations: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
        start_paths: Any = None,
    ) -> Any:
        """Issue a batch grasp query with typed collision metadata."""

        if self._plan_pose_batch is None:
            raise RuntimeError("PickPlanningPort has no batch planning callback")
        policy = collision_policy or self._collision_policy
        phase_id = (
            "pick_terminal_grasp_batch"
            if policy is CollisionPolicy.TARGET_APPROACH
            else "pick_pregrasp_batch"
        )
        return self._plan_pose_batch(
            positions,
            orientations,
            start_paths=start_paths,
            request_metadata=self._request_metadata(
                collision_policy,
                phase_id=phase_id,
                active_target=active_target or self._active_target,
                default_policy=policy,
            ),
        )

    def plan_pose_result(
        self,
        position: Any,
        orientation: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
    ) -> Any:
        """Issue one typed Pick query with the requested collision policy."""

        if self._plan_pose_result is None:
            raise RuntimeError("PickPlanningPort has no single planning callback")
        policy = collision_policy or self._collision_policy
        phase_id = (
            "pick_terminal_grasp"
            if policy is CollisionPolicy.TARGET_APPROACH
            else "pick_pregrasp"
        )
        return self._plan_pose_result(
            position,
            orientation,
            request_metadata=self._request_metadata(
                collision_policy,
                phase_id=phase_id,
                active_target=active_target or self._active_target,
                default_policy=policy,
            ),
        )

    def plan_pose_from_path(
        self,
        position: Any,
        orientation: Any,
        start_path: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
    ) -> Any:
        """Issue one typed terminal-grasp query from a named pre-grasp path."""

        if self._plan_pose_from_path is None:
            raise RuntimeError("PickPlanningPort has no path planning callback")
        policy = collision_policy or self._collision_policy
        return self._plan_pose_from_path(
            position,
            orientation,
            start_path,
            request_metadata=self._request_metadata(
                collision_policy,
                phase_id="pick_terminal_grasp",
                active_target=active_target or self._active_target,
                default_policy=policy,
            ),
        )

    def plan_pose_from_joint_positions(
        self,
        position: Any,
        orientation: Any,
        *,
        start_arm_positions: Any = None,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
    ) -> Any:
        """Plan one pose from an explicit typed arm-joint start state.

        Dynamic Pick uses this query only for timing previews.  The callback
        hook is supplied by a controller operation component when available;
        otherwise the composed runtime path below builds a named CuRobo
        ``JointState`` directly.  Neither path mutates the controller façade
        or accepts untyped collision metadata.
        """

        policy = collision_policy or self._collision_policy
        request_metadata = self._request_metadata(
            collision_policy,
            phase_id="pick_preview",
            active_target=active_target or self._active_target,
            default_policy=policy,
        )
        if self._plan_pose_from_joint_positions is not None:
            return self._plan_pose_from_joint_positions(
                position,
                orientation,
                start_arm_positions=start_arm_positions,
                request_metadata=request_metadata,
            )

        runtime = self.runtime
        if start_arm_positions is None:
            robot = runtime.robot_port.robot
            start_state = runtime.arm_joint_state(robot.get_joints_state())
        else:
            values = np.asarray(start_arm_positions, dtype=float).reshape(-1)
            planner_names = list(runtime.planner_names)
            if values.size != len(planner_names):
                raise ValueError(
                    "start_arm_positions must match the typed planner joint count: "
                    f"got {values.size}, expected {len(planner_names)}"
                )
            from curobo.types import JointState

            start_state = JointState.from_position(
                runtime.robot_port.tensor_args.to_device(values),
                joint_names=planner_names,
            )
        result = runtime.plan_pose(
            position,
            orientation,
            start_state=start_state,
            context="pick_preview",
            request_metadata=request_metadata,
        )
        if not bool(result.success):
            return False, None, result
        path = result.trajectory
        if path is None:
            return False, None, result
        trajectory_state = runtime.joint_state_from_trajectory(path)
        end_positions = np.asarray(
            trajectory_state.position[-1].detach().cpu(), dtype=float
        )
        return True, end_positions, result

    def measure_cartesian_path(self, path: Any, start: Any, goal: Any) -> Any:
        if self._measure_cartesian_path is None:
            raise RuntimeError("PickPlanningPort has no Cartesian path callback")
        return self._measure_cartesian_path(path, start, goal)

    def build_commands(self, **kwargs: Any) -> list[Any]:
        """Build typed Pick execution commands through the composed phase port."""

        return self._build_commands(**kwargs)

    def arm_base_transform(self) -> Any:
        return self._arm_base_transform()

    def frame_debug(self) -> Mapping[str, Any]:
        return self._frame_debug()

    def capture_reference(self, object_name: str) -> None:
        self._capture_reference(object_name)

    def retarget_commands(self, object_name: str, commands: Any) -> Any:
        return self._retarget_commands(object_name, commands)

    def replan_after_safety(self, object_name: str, command: Any, commands: Any) -> bool:
        return self._replan_after_safety(object_name, command, commands)

    def ee_pose(self) -> Any:
        return self._execution_ee_pose()

    def initial_ee_pose(self) -> Any:
        """Return the captured initial EE pose in the arm-base frame."""

        return deepcopy(self._initial_ee_pose)

    def set_plan_failure_count(self, value: int) -> None:
        if self._set_plan_failure_count is None:
            raise RuntimeError("PickPlanningPort has no plan-failure counter")
        value = int(value)
        if value < 0:
            raise ValueError("plan-failure count must be non-negative")
        self._set_plan_failure_count(value)

    def execution_status(self, command: Any = None) -> Any:
        if self._execution_status is None:
            raise RuntimeError("PickPlanningPort has no execution-status callback")
        return self._execution_status(command)

    def phase_complete(self, command: Any) -> bool:
        return bool(self._phase_complete(command))


def compose_pick_planning_port(
    scene_port: PlannerScenePort,
    collision_scene_manager: Any,
    setup: Any,
    phases: Any,
    execution: Any,
    robot_file: str,
    batch_capability: bool,
    plan_pose_batch: Callable[..., Any],
    plan_pose_result: Callable[..., Any],
    plan_pose_from_path: Callable[..., Any],
    measure_cartesian_path: Callable[..., Any],
) -> PickPlanningPort:
    """Compose the public Pick port from already-wired operation components."""

    return PickPlanningPort(
        scene_port=scene_port,
        collision_scene_manager=collision_scene_manager,
        update_pose_cost_metric=setup.update_pose_cost_metric,
        build_commands=phases.build_pick_phase_commands,
        arm_base_transform=phases.get_pick_armbase_transform,
        frame_debug=phases.get_pick_frame_debug,
        capture_reference=phases.capture_pick_plan_reference,
        retarget_commands=phases.retarget_pick_phase_commands,
        replan_after_safety=phases.replan_pick_after_safety,
        execution_ee_pose=execution.get_ee_pose,
        phase_complete=execution.is_phase_command_complete,
        plan_failure_count=lambda: int(execution.num_plan_failed),
        set_plan_failure_count=lambda value: setattr(execution, "num_plan_failed", int(value)),
        last_command_count=lambda: int(execution.num_last_cmd),
        execution_status=execution.execution_status,
        robot_file=robot_file,
        batch_capability=batch_capability,
        plan_pose_batch=plan_pose_batch,
        plan_pose_result=plan_pose_result,
        plan_pose_from_path=plan_pose_from_path,
        measure_cartesian_path=measure_cartesian_path,
    )


__all__ = [
    "PickPlanningPort",
    "PickPlanningQueryPort",
    "compose_pick_planning_port",
]
