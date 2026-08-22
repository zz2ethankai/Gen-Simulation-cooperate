"""Single generic runtime contract exposed to manipulation Skills.

Skills are deliberately not given the :class:`TemplateController` façade.
The façade is an Isaac lifecycle object and also owns the component assembly;
passing it into a Skill makes it very easy to accidentally depend on a
private implementation detail (or on a stale compatibility alias).  This
module contains the narrow, simulator-facing values and callbacks that a
Skill may need while querying a plan, executing a typed command, or observing
the collision scene.  Candidate evaluation, candidate recovery, and
Pick/Place command construction deliberately remain outside this port.

The port is immutable by construction.  Mutable execution counters are not
stored on the port: they are properties backed by the one
``MutableExecutionState`` owned by the controller components.  Callback
fields are intentionally private so consumers can only invoke the typed
operations and cannot replace the controller's wiring at runtime.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from core.controllers.curobo.components import MutableExecutionState
from core.planning.domain_types import CollisionOptions, CollisionPolicy


class SkillRuntimePort:
    """Immutable Skill-facing view of one arm's runtime.

    ``SkillRuntimePort`` is intentionally a small callable object instead of
    a façade proxy.  It has no ``__getattr__`` fallback and does not retain a
    controller reference.  A controller composes one instance after its
    operation components and scene port have been wired.

    The constructor accepts callbacks rather than component objects for
    generic pose queries, execution, collision-world transitions, and scene
    ownership operations.  This keeps the dependency direction explicit and
    makes host-side tests able to construct a port without Isaac Sim or
    CuRobo.
    """

    __slots__ = (
        "_sealed",
        "_robot",
        "_runtime",
        "_execution_state",
        "_arm_spec",
        "_arm_indices",
        "_gripper_indices",
        "_raw_joint_names",
        "_control_joint_names",
        "_robot_file",
        "_robot_config",
        "_robot_base_path",
        "_robot_ee_path",
        "_reference_prim_path",
        "_name",
        "_arm_name",
        "_batch_capability",
        "_interpolation_dt",
        "_ee_pose_fn",
        "_arm_base_pose_fn",
        "_compute_fk_fn",
        "_initial_ee_pose_fn",
        "_plan_pose_fn",
        "_plan_pose_batch_fn",
        "_plan_pose_result_fn",
        "_plan_pose_from_path_fn",
        "_plan_pose_from_joint_positions_fn",
        "_measure_cartesian_path_fn",
        "_plan_cspace_fn",
        "_forward_kinematic_fn",
        "_arm_base_transform_fn",
        "_phase_complete_fn",
        "_sync_native_batch_attachment_fn",
        "_update_pose_cost_metric_fn",
        "_complete_contact_phase_fn",
        "_execution_status_fn",
        "_command_status_fn",
        "_execute_fn",
        "_dummy_forward_fn",
        "_hold_fn",
        "_clear_plan_and_hold_fn",
        "_push_timing_scope_fn",
        "_restore_timing_scope_fn",
        "_clear_timing_scope_fn",
        "_prepare_world_fn",
        "_transition_target_fn",
        "_restore_world_fn",
        "_diagnose_start_collision_fn",
        "_source_support_fn",
        "_collision_record_fn",
        "_refresh_world_fn",
        "_sync_dynamic_poses_fn",
        "_collision_policy_fn",
        "_active_target_fn",
        "_support_fn",
        "_scene_callbacks",
    )

    @staticmethod
    def _as_getter(value: Callable[[], Any] | Any | None):
        if value is None:
            return None
        if callable(value):
            return value
        return lambda: value

    def __init__(
        self,
        *,
        robot: Any,
        runtime: Any,
        execution_state: MutableExecutionState,
        arm_spec: Any,
        arm_indices: Sequence[int],
        gripper_indices: Sequence[int],
        raw_joint_names: Sequence[str] = (),
        control_joint_names: Sequence[str] = (),
        robot_file: str = "",
        robot_config: Mapping[str, Any] | None = None,
        robot_base_path: str = "",
        robot_ee_path: str = "",
        reference_prim_path: str = "",
        name: str = "",
        arm_name: str = "",
        batch_capability: bool = False,
        interpolation_dt: float = 0.01,
        ee_pose: Callable[[], Any],
        arm_base_pose: Callable[[], Any],
        compute_fk: Callable[..., Any],
        initial_ee_pose: Callable[[], Any] | None = None,
        plan_pose: Callable[..., Any] | None = None,
        plan_pose_batch: Callable[..., Any] | None = None,
        plan_pose_result: Callable[..., Any] | None = None,
        plan_pose_from_path: Callable[..., Any] | None = None,
        plan_pose_from_joint_positions: Callable[..., Any] | None = None,
        measure_cartesian_path: Callable[..., Any] | None = None,
        plan_cspace: Callable[..., Any] | None = None,
        forward_kinematic: Callable[..., Any] | None = None,
        arm_base_transform: Callable[[], Any] | None = None,
        phase_complete: Callable[[Any], bool] | None = None,
        sync_native_batch_attachment: Callable[..., Any] | None = None,
        update_pose_cost_metric: Callable[..., Any] | None = None,
        complete_contact_phase: Callable[[Any], Any] | None = None,
        execution_status: Callable[..., Any] | None = None,
        command_status: Callable[..., Any] | None = None,
        execute: Callable[..., Any] | None = None,
        dummy_forward: Callable[..., Any] | None = None,
        hold: Callable[..., Any] | None = None,
        clear_plan_and_hold: Callable[[], Any] | None = None,
        push_timing_scope: Callable[[Any], Any] | None = None,
        restore_timing_scope: Callable[[Any], Any] | None = None,
        clear_timing_scope: Callable[[Any], Any] | None = None,
        prepare_world: Callable[..., Any] | None = None,
        transition_target: Callable[..., Any] | None = None,
        restore_world: Callable[..., Any] | None = None,
        diagnose_start_collision: Callable[..., Any] | None = None,
        source_support: Callable[..., Any] | None = None,
        collision_record: Callable[..., Any] | None = None,
        refresh_world: Callable[..., Any] | None = None,
        sync_dynamic_poses: Callable[..., Any] | None = None,
        collision_policy: Callable[[], Any] | Any | None = None,
        active_target: Callable[[], Any] | Any | None = None,
        support: Callable[[], Any] | Any | None = None,
        scene_callbacks: Mapping[str, Callable[..., Any]] | None = None,
    ) -> None:
        if not isinstance(execution_state, MutableExecutionState):
            raise TypeError("SkillRuntimePort requires MutableExecutionState")
        for callback_name, callback in {
            "ee_pose": ee_pose,
            "arm_base_pose": arm_base_pose,
            "compute_fk": compute_fk,
        }.items():
            if not callable(callback):
                raise TypeError(f"SkillRuntimePort {callback_name} callback must be callable")

        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_robot", robot)
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_execution_state", execution_state)
        object.__setattr__(self, "_arm_spec", arm_spec)
        object.__setattr__(self, "_arm_indices", tuple(int(index) for index in arm_indices))
        object.__setattr__(self, "_gripper_indices", tuple(int(index) for index in gripper_indices))
        object.__setattr__(self, "_raw_joint_names", tuple(str(name) for name in raw_joint_names))
        object.__setattr__(self, "_control_joint_names", tuple(str(name) for name in control_joint_names))
        object.__setattr__(self, "_robot_file", str(robot_file))
        config = dict(robot_config or {})
        object.__setattr__(self, "_robot_config", MappingProxyType(config))
        object.__setattr__(self, "_robot_base_path", str(robot_base_path or ""))
        object.__setattr__(self, "_robot_ee_path", str(robot_ee_path or ""))
        object.__setattr__(self, "_reference_prim_path", str(reference_prim_path or ""))
        object.__setattr__(self, "_name", str(name))
        object.__setattr__(self, "_arm_name", str(arm_name))
        object.__setattr__(self, "_batch_capability", bool(batch_capability))
        object.__setattr__(self, "_interpolation_dt", float(interpolation_dt))
        object.__setattr__(self, "_ee_pose_fn", ee_pose)
        object.__setattr__(self, "_arm_base_pose_fn", arm_base_pose)
        object.__setattr__(self, "_compute_fk_fn", compute_fk)
        object.__setattr__(self, "_initial_ee_pose_fn", initial_ee_pose)
        object.__setattr__(self, "_plan_pose_fn", plan_pose)
        object.__setattr__(self, "_plan_pose_batch_fn", plan_pose_batch)
        object.__setattr__(self, "_plan_pose_result_fn", plan_pose_result)
        object.__setattr__(self, "_plan_pose_from_path_fn", plan_pose_from_path)
        object.__setattr__(
            self,
            "_plan_pose_from_joint_positions_fn",
            plan_pose_from_joint_positions,
        )
        object.__setattr__(self, "_measure_cartesian_path_fn", measure_cartesian_path)
        object.__setattr__(self, "_plan_cspace_fn", plan_cspace)
        object.__setattr__(self, "_forward_kinematic_fn", forward_kinematic)
        object.__setattr__(self, "_arm_base_transform_fn", arm_base_transform)
        object.__setattr__(self, "_phase_complete_fn", phase_complete)
        object.__setattr__(
            self,
            "_sync_native_batch_attachment_fn",
            sync_native_batch_attachment,
        )
        object.__setattr__(self, "_update_pose_cost_metric_fn", update_pose_cost_metric)
        object.__setattr__(self, "_complete_contact_phase_fn", complete_contact_phase)
        object.__setattr__(self, "_execution_status_fn", execution_status)
        object.__setattr__(self, "_command_status_fn", command_status)
        object.__setattr__(self, "_execute_fn", execute)
        object.__setattr__(self, "_dummy_forward_fn", dummy_forward)
        object.__setattr__(self, "_hold_fn", hold)
        object.__setattr__(self, "_clear_plan_and_hold_fn", clear_plan_and_hold)
        object.__setattr__(self, "_push_timing_scope_fn", push_timing_scope)
        object.__setattr__(self, "_restore_timing_scope_fn", restore_timing_scope)
        object.__setattr__(self, "_clear_timing_scope_fn", clear_timing_scope)
        object.__setattr__(self, "_prepare_world_fn", prepare_world)
        object.__setattr__(self, "_transition_target_fn", transition_target)
        object.__setattr__(self, "_restore_world_fn", restore_world)
        object.__setattr__(self, "_diagnose_start_collision_fn", diagnose_start_collision)
        object.__setattr__(self, "_source_support_fn", source_support)
        object.__setattr__(self, "_collision_record_fn", collision_record)
        object.__setattr__(self, "_refresh_world_fn", refresh_world)
        object.__setattr__(self, "_sync_dynamic_poses_fn", sync_dynamic_poses)
        object.__setattr__(self, "_collision_policy_fn", self._as_getter(collision_policy))
        object.__setattr__(self, "_active_target_fn", self._as_getter(active_target))
        object.__setattr__(self, "_support_fn", self._as_getter(support))
        object.__setattr__(self, "_scene_callbacks", MappingProxyType(dict(scene_callbacks or {})))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "num_plan_failed" and getattr(self, "_sealed", False):
            # The port remains immutable; this one state operation is
            # deliberately routed to the authoritative execution owner.
            type(self).num_plan_failed.fset(self, value)
            return
        if getattr(self, "_sealed", False):
            raise AttributeError("SkillRuntimePort is read-only")
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Immutable robot/configuration values
    # ------------------------------------------------------------------
    @property
    def robot(self) -> Any:
        return self._robot

    @property
    def runtime(self) -> Any:
        return self._runtime

    @property
    def execution_state(self) -> MutableExecutionState:
        return self._execution_state

    @property
    def arm_spec(self) -> Any:
        return self._arm_spec

    @property
    def arm_indices(self) -> np.ndarray:
        return np.asarray(self._arm_indices, dtype=np.int64).copy()

    @property
    def gripper_indices(self) -> np.ndarray:
        return np.asarray(self._gripper_indices, dtype=np.int64).copy()

    @property
    def raw_joint_names(self) -> tuple[str, ...]:
        return self._raw_joint_names

    @property
    def control_joint_names(self) -> tuple[str, ...]:
        return self._control_joint_names

    @property
    def robot_file(self) -> str:
        return self._robot_file

    @property
    def robot_config(self) -> Mapping[str, Any]:
        return self._robot_config

    @property
    def robot_cfg(self) -> Mapping[str, Any]:
        """Typed robot configuration spelling used by runtime integrations."""

        return self._robot_config

    @property
    def robot_base_path(self) -> str:
        return self._robot_base_path

    @property
    def robot_ee_path(self) -> str:
        return self._robot_ee_path

    @property
    def reference_prim_path(self) -> str:
        return self._reference_prim_path

    @property
    def name(self) -> str:
        return self._name

    @property
    def arm_name(self) -> str:
        return self._arm_name

    @property
    def lr_name(self) -> str:
        """Explicit arm-name spelling for scene/diagnostic contracts."""

        return self._arm_name

    @property
    def batch_capability(self) -> bool:
        return self._batch_capability

    @property
    def interpolation_dt(self) -> float:
        return self._interpolation_dt

    @property
    def scene_revision(self) -> int:
        """Current generic planning-scene revision."""

        value = getattr(self._runtime, "scene_revision", None)
        if value is None:
            value = getattr(self._runtime, "world_revision", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @property
    def world_revision(self) -> int:
        """Alias for callers recording planner-world provenance."""

        return self.scene_revision

    @property
    def grasp_approach_axis(self) -> int:
        """Configured end-effector approach axis for generic grasp geometry."""

        value = getattr(self._arm_spec, "grasp_approach_axis", None)
        if value is None:
            ee_axis = self._robot_config.get("ee_axis", "z")
            value = {"x": 0, "y": 1, "z": 2}.get(str(ee_axis).lower(), 2)
        value = int(value)
        if value not in (0, 1, 2):
            raise ValueError(f"grasp_approach_axis must be 0, 1, or 2, got {value}")
        return value

    # ------------------------------------------------------------------
    # Authoritative execution state
    # ------------------------------------------------------------------
    @property
    def num_last_cmd(self) -> int:
        return int(self._execution_state.num_last_cmd)

    @property
    def num_plan_failed(self) -> int:
        return int(self._execution_state.num_plan_failed)

    @num_plan_failed.setter
    def num_plan_failed(self, value: int) -> None:
        self._execution_state.num_plan_failed = int(value)

    @property
    def step_idx(self) -> int:
        return int(self._execution_state.step_idx)

    @property
    def active_phase_command(self) -> Any:
        """Current typed command owned by the execution state."""

        return self._execution_state.active_phase_command

    @property
    def last_commanded_arm_position(self) -> np.ndarray | None:
        """Copy of the latest arm target recorded by execution."""

        value = self._execution_state.last_commanded_arm_position
        return None if value is None else np.asarray(value, dtype=float).copy()

    def phase_base_pose(self):
        """Base pose captured when the active typed phase began."""

        position = self._execution_state.phase_base_position
        orientation = self._execution_state.phase_base_orientation
        if position is None or orientation is None:
            return None
        return np.asarray(position).copy(), np.asarray(orientation).copy()

    def record_plan_failure(self) -> int:
        self._execution_state.num_plan_failed += 1
        return int(self._execution_state.num_plan_failed)

    def reset_plan_failures(self) -> None:
        self._execution_state.num_plan_failed = 0

    @property
    def last_command_count(self) -> int:
        """Number of commands currently owned by the execution state."""

        return self.num_last_cmd

    @property
    def plan_failure_count(self) -> int:
        """Compatibility spelling for the generic plan-failure counter."""

        return self.num_plan_failed

    @plan_failure_count.setter
    def plan_failure_count(self, value: int) -> None:
        self.num_plan_failed = value

    @staticmethod
    def _coerce_collision_policy(value: Any) -> CollisionPolicy | None:
        if value is None:
            return None
        if isinstance(value, CollisionPolicy):
            return value
        return CollisionPolicy(str(value).lower())

    @staticmethod
    def _callback_accepts(callback: Callable[..., Any], keyword: str) -> bool:
        """Check optional query keywords without constraining host-side fakes."""

        try:
            parameters = inspect.signature(callback).parameters.values()
        except (TypeError, ValueError):
            return True
        return any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            or parameter.name == keyword
            for parameter in parameters
        )

    @classmethod
    def _invoke_query(
        cls,
        callback: Callable[..., Any],
        *args: Any,
        request_metadata: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if request_metadata is not None and cls._callback_accepts(
            callback, "request_metadata"
        ):
            kwargs["request_metadata"] = dict(request_metadata)
        return callback(*args, **kwargs)

    def _query_context(
        self,
        *,
        phase_id: str,
        collision_policy: CollisionPolicy | str | None = None,
        active_target: str | None = None,
        support: str | None = None,
        request_metadata: Mapping[str, Any] | None = None,
        collision_options: CollisionOptions | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the typed metadata shared by all generic planner queries."""

        metadata = dict(request_metadata or {})
        policy = self._coerce_collision_policy(collision_policy)
        if policy is None:
            policy = self.collision_policy
        if policy is None:
            policy = CollisionPolicy.WORLD_TRANSIT
        metadata.setdefault("phase_id", str(phase_id))
        metadata["collision_policy"] = policy

        target = active_target if active_target is not None else self.active_target
        support_value = support if support is not None else self.support
        if target is not None:
            metadata["active_target"] = str(target)
        if support_value is not None:
            metadata["support"] = str(support_value)

        supplied_options = collision_options
        if supplied_options is None:
            supplied_options = metadata.get("collision_options")
        if supplied_options is not None or policy in {
            CollisionPolicy.TARGET_APPROACH,
            CollisionPolicy.PLACEMENT_DESCENT,
        }:
            options = CollisionOptions.from_mapping(
                supplied_options,
                default_policy=policy,
            )
            option_values = options.to_dict()
            if policy is CollisionPolicy.TARGET_APPROACH:
                option_values["allow_target_contact"] = True
                option_values["allow_target_finger_contact"] = True
            if policy is CollisionPolicy.PLACEMENT_DESCENT:
                option_values["allow_support_contact"] = True
                option_values["allow_object_support_contact"] = True
            metadata["collision_options"] = CollisionOptions.from_mapping(
                option_values,
                default_policy=policy,
            )
        return metadata

    # ------------------------------------------------------------------
    # Typed runtime/execution callbacks
    # ------------------------------------------------------------------
    @staticmethod
    def _pose_copy(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, tuple):
            return tuple(np.asarray(item).copy() for item in value)
        if isinstance(value, list):
            return [np.asarray(item).copy() for item in value]
        return np.asarray(value).copy()

    def ee_pose(self):
        return self._pose_copy(self._ee_pose_fn())

    def arm_base_pose(self):
        return self._pose_copy(self._arm_base_pose_fn())

    def compute_fk(self, joint_positions: Any, *, joint_names: Sequence[str] | None = None):
        """Compute FK through the typed runtime callback and named joints."""

        values = np.asarray(joint_positions, dtype=float).copy()
        names = None if joint_names is None else tuple(str(name) for name in joint_names)
        if names is None:
            # Keep the callback contract friendly to small host-side fakes
            # that only accept the position vector.  The production runtime
            # accepts the optional named-joint keyword as well.
            return self._compute_fk_fn(values)
        return self._compute_fk_fn(values, joint_names=names)

    def initial_ee_pose(self):
        if self._initial_ee_pose_fn is None:
            return None
        return self._pose_copy(self._initial_ee_pose_fn())

    def plan_pose(
        self,
        position: Any,
        orientation: Any,
        *args: Any,
        collision_policy: CollisionPolicy | str | None = None,
        active_target: str | None = None,
        support: str | None = None,
        phase_id: str = "plan_pose",
        request_metadata: Mapping[str, Any] | None = None,
        collision_options: CollisionOptions | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        if self._plan_pose_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose pose planning")
        metadata = self._query_context(
            phase_id=phase_id,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            request_metadata=request_metadata,
            collision_options=collision_options,
        )
        return self._invoke_query(
            self._plan_pose_fn,
            position,
            orientation,
            *args,
            request_metadata=metadata,
            **kwargs,
        )

    def plan_pose_batch(
        self,
        positions: Any,
        orientations: Any,
        *args: Any,
        collision_policy: CollisionPolicy | str | None = None,
        active_target: str | None = None,
        support: str | None = None,
        start_paths: Any = None,
        batch_size: int | None = None,
        phase_id: str = "plan_pose_batch",
        request_metadata: Mapping[str, Any] | None = None,
        collision_options: CollisionOptions | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        if self._plan_pose_batch_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose batch pose planning")
        metadata = self._query_context(
            phase_id=phase_id,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            request_metadata=request_metadata,
            collision_options=collision_options,
        )
        if start_paths is not None:
            kwargs["start_paths"] = start_paths
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        return self._invoke_query(
            self._plan_pose_batch_fn,
            positions,
            orientations,
            *args,
            request_metadata=metadata,
            **kwargs,
        )

    def plan_pose_result(
        self,
        position: Any,
        orientation: Any,
        *args: Any,
        collision_policy: CollisionPolicy | str | None = None,
        active_target: str | None = None,
        support: str | None = None,
        phase_id: str = "plan_pose_result",
        request_metadata: Mapping[str, Any] | None = None,
        collision_options: CollisionOptions | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        if self._plan_pose_result_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose pose-result planning")
        metadata = self._query_context(
            phase_id=phase_id,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            request_metadata=request_metadata,
            collision_options=collision_options,
        )
        return self._invoke_query(
            self._plan_pose_result_fn,
            position,
            orientation,
            *args,
            request_metadata=metadata,
            **kwargs,
        )

    def plan_pose_from_path(
        self,
        position: Any,
        orientation: Any,
        start_path: Any,
        *args: Any,
        collision_policy: CollisionPolicy | str | None = None,
        active_target: str | None = None,
        support: str | None = None,
        phase_id: str = "plan_pose_from_path",
        request_metadata: Mapping[str, Any] | None = None,
        collision_options: CollisionOptions | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        if self._plan_pose_from_path_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose path pose planning")
        metadata = self._query_context(
            phase_id=phase_id,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            request_metadata=request_metadata,
            collision_options=collision_options,
        )
        return self._invoke_query(
            self._plan_pose_from_path_fn,
            position,
            orientation,
            start_path,
            *args,
            request_metadata=metadata,
            **kwargs,
        )

    def plan_pose_from_joint_positions(
        self,
        position: Any,
        orientation: Any,
        *args: Any,
        start_arm_positions: Any = None,
        collision_policy: CollisionPolicy | str | None = None,
        active_target: str | None = None,
        support: str | None = None,
        phase_id: str = "plan_pose_from_joint_positions",
        request_metadata: Mapping[str, Any] | None = None,
        collision_options: CollisionOptions | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ):
        if self._plan_pose_from_joint_positions_fn is None:
            raise RuntimeError(
                "SkillRuntimePort does not expose joint-position pose planning"
            )
        metadata = self._query_context(
            phase_id=phase_id,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            request_metadata=request_metadata,
            collision_options=collision_options,
        )
        kwargs["start_arm_positions"] = start_arm_positions
        return self._invoke_query(
            self._plan_pose_from_joint_positions_fn,
            position,
            orientation,
            *args,
            request_metadata=metadata,
            **kwargs,
        )

    def measure_cartesian_path(self, path: Any, start: Any, goal: Any):
        if self._measure_cartesian_path_fn is None:
            raise RuntimeError(
                "SkillRuntimePort does not expose Cartesian path measurement"
            )
        return self._measure_cartesian_path_fn(path, start, goal)

    def forward_kinematic(self, joint_positions: Any, *, joint_names=None):
        callback = self._forward_kinematic_fn or self._compute_fk_fn
        values = np.asarray(joint_positions, dtype=float).copy()
        if joint_names is None:
            return callback(values)
        names = tuple(str(name) for name in joint_names)
        try:
            return callback(values, joint_names=names)
        except TypeError:
            return callback(values)

    def arm_base_transform(self):
        if self._arm_base_transform_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose an arm-base transform")
        return self._arm_base_transform_fn()

    def phase_complete(self, command: Any) -> bool:
        if self._phase_complete_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose phase completion")
        return bool(self._phase_complete_fn(command))

    def sync_native_batch_attachment(self, *args: Any, **kwargs: Any) -> bool:
        if self._sync_native_batch_attachment_fn is None:
            raise RuntimeError(
                "SkillRuntimePort does not expose native batch attachment sync"
            )
        return bool(self._sync_native_batch_attachment_fn(*args, **kwargs))

    def update_pose_cost_metric(self, hold_vec_weight=None):
        if self._update_pose_cost_metric_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose pose cost updates")
        return self._update_pose_cost_metric_fn(hold_vec_weight)

    def complete_contact_phase(self, command: Any):
        """Complete a contact-gated phase through the execution owner."""

        if self._complete_contact_phase_fn is None:
            raise RuntimeError(
                "SkillRuntimePort does not expose contact-phase completion"
            )
        return self._complete_contact_phase_fn(command)

    def plan_cspace(self, *args, **kwargs):
        if self._plan_cspace_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose c-space planning")
        return self._plan_cspace_fn(*args, **kwargs)

    def execution_status(self, command=None):
        if self._execution_status_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose execution status")
        return self._execution_status_fn(command)

    def command_status(self, command=None):
        if self._command_status_fn is not None:
            return self._command_status_fn(command)
        status = self.execution_status(command)
        return getattr(status, "status", status)

    def execute(self, command):
        if self._execute_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose typed execution")
        return self._execute_fn(command)

    def dummy_forward(self, arm_action, gripper_state, *args, **kwargs):
        """Send one direct joint action through the controller owner.

        Unlike ``execute(MotionPhaseCommand)``, this path does not create or
        consume a Physics-schema planner trajectory.  A Skill that uses it is
        responsible for generating interpolation points and deciding when the
        current point is complete.
        """

        if self._dummy_forward_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose dummy_forward")
        return self._dummy_forward_fn(arm_action, gripper_state, *args, **kwargs)

    def hold(self, reason: str | None = None):
        """Emit a measured hold through the typed runtime boundary."""

        if self._hold_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose typed hold")
        # Runtime hold callbacks currently take no payload; keep ``reason``
        # for the diagnostic boundary without reintroducing a façade alias.
        del reason
        return self._hold_fn()

    def clear_plan_and_hold(self):
        if self._clear_plan_and_hold_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose plan reset")
        return self._clear_plan_and_hold_fn()

    def push_timing_scope(self, scope):
        if self._push_timing_scope_fn is None:
            return None
        return self._push_timing_scope_fn(scope)

    def restore_timing_scope(self, previous):
        if self._restore_timing_scope_fn is not None:
            return self._restore_timing_scope_fn(previous)
        return None

    def clear_timing_scope(self, scope=None):
        if self._clear_timing_scope_fn is not None:
            return self._clear_timing_scope_fn(scope)
        return None

    # ------------------------------------------------------------------
    # Generic collision-policy and world-transition callbacks
    # ------------------------------------------------------------------
    @property
    def collision_policy(self) -> CollisionPolicy:
        value = self._collision_policy_fn() if self._collision_policy_fn else None
        return self._coerce_collision_policy(value) or CollisionPolicy.WORLD_TRANSIT

    @property
    def active_target(self) -> str | None:
        value = self._active_target_fn() if self._active_target_fn else None
        return None if value is None else str(value)

    @property
    def support(self) -> str | None:
        value = self._support_fn() if self._support_fn else None
        return None if value is None else str(value)

    def prepare_world(
        self,
        object_name: str,
        support_name: str | None = None,
        *,
        collision_policy: CollisionPolicy | str | None = None,
    ):
        if self._prepare_world_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose world preparation")
        policy = self._coerce_collision_policy(collision_policy)
        return self._prepare_world_fn(
            str(object_name),
            None if support_name is None else str(support_name),
            collision_policy=policy,
        )

    def transition_target(
        self,
        object_name: str,
        support_name: str | None = None,
        *,
        collision_policy: CollisionPolicy | str | None = None,
        support: str | None = None,
    ):
        if self._transition_target_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose world transitions")
        policy = self._coerce_collision_policy(collision_policy)
        if support_name is None:
            support_name = support
        if policy is None:
            policy = self.collision_policy
        return self._transition_target_fn(
            str(object_name),
            None if support_name is None else str(support_name),
            collision_policy=policy,
        )

    def restore_world(self, object_name: str):
        if self._restore_world_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose world restore")
        return self._restore_world_fn(str(object_name))

    def diagnose_start_collision(self):
        if self._diagnose_start_collision_fn is None:
            raise RuntimeError(
                "SkillRuntimePort does not expose start-collision diagnostics"
            )
        return self._diagnose_start_collision_fn()

    def source_support(self, object_name: str):
        if self._source_support_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose source-support lookup")
        return self._source_support_fn(str(object_name))

    def collision_record(self, object_name: str):
        if self._collision_record_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose collision records")
        return self._collision_record_fn(str(object_name))

    def refresh_world(self):
        if self._refresh_world_fn is None:
            return None
        return self._refresh_world_fn()

    def sync_dynamic_poses(
        self,
        step_id: int = 0,
        *,
        interval_steps: int = 1,
        force: bool = False,
    ):
        if self._sync_dynamic_poses_fn is None:
            return None
        return self._sync_dynamic_poses_fn(
            int(step_id), interval_steps=int(interval_steps), force=bool(force)
        )

    # ------------------------------------------------------------------
    # Typed scene ownership callbacks
    # ------------------------------------------------------------------
    def _scene_call(self, name: str, *args, **kwargs):
        callback = self._scene_callbacks.get(name)
        if not callable(callback):
            raise RuntimeError(f"SkillRuntimePort scene callback is unavailable: {name}")
        return callback(*args, **kwargs)

    def assert_attached_owner(self, entity_name: str):
        return self._scene_call("assert_attached_owner", str(entity_name))

    def get_source_support_entity(self, entity_name: str):
        return self._scene_call("get_source_support_entity", str(entity_name))

    def get_attached_entity(self):
        return self._scene_call("get_attached_entity")

    def has_native_obstacle(self, path: str) -> bool:
        return bool(self._scene_call("has_native_obstacle", str(path)))

    def attached_object_slip(self, entity_name: str):
        return self._scene_call("attached_object_slip", str(entity_name))


def compose_skill_runtime_port(
    *,
    robot: Any,
    runtime: Any,
    execution_state: MutableExecutionState,
    arm_spec: Any,
    arm_indices: Sequence[int],
    gripper_indices: Sequence[int],
    raw_joint_names: Sequence[str],
    control_joint_names: Sequence[str],
    robot_file: str,
    robot_config: Mapping[str, Any] | None,
    robot_base_path: str,
    robot_ee_path: str,
    reference_prim_path: str,
    name: str,
    arm_name: str,
    batch_capability: bool,
    interpolation_dt: float,
    ee_pose: Callable[[], Any],
    arm_base_pose: Callable[[], Any],
    compute_fk: Callable[..., Any],
    initial_ee_pose: Callable[[], Any] | None = None,
    plan_pose: Callable[..., Any] | None = None,
    plan_pose_batch: Callable[..., Any] | None = None,
    plan_pose_result: Callable[..., Any] | None = None,
    plan_pose_from_path: Callable[..., Any] | None = None,
    plan_pose_from_joint_positions: Callable[..., Any] | None = None,
    measure_cartesian_path: Callable[..., Any] | None = None,
    forward_kinematic: Callable[..., Any] | None = None,
    arm_base_transform: Callable[[], Any] | None = None,
    phase_complete: Callable[[Any], bool] | None = None,
    sync_native_batch_attachment: Callable[..., Any] | None = None,
    update_pose_cost_metric: Callable[..., Any] | None = None,
    complete_contact_phase: Callable[[Any], Any] | None = None,
    execution_status: Callable[..., Any] | None = None,
    command_status: Callable[..., Any] | None = None,
    execute: Callable[..., Any] | None = None,
    dummy_forward: Callable[..., Any] | None = None,
    hold: Callable[..., Any] | None = None,
    clear_plan_and_hold: Callable[[], Any] | None = None,
    push_timing_scope: Callable[[Any], Any] | None = None,
    restore_timing_scope: Callable[[Any], Any] | None = None,
    clear_timing_scope: Callable[[Any], Any] | None = None,
    collision_scene_manager: Any = None,
    scene_port: Any = None,
    refresh_reference_world: Callable[[], Any] | None = None,
    sync_dynamic_poses: Callable[..., Any] | None = None,
) -> SkillRuntimePort:
    """Compose the Skill-facing runtime view from explicit operation ports."""

    if not callable(compute_fk):
        def compute_fk(*_args, **_kwargs):
            raise RuntimeError("SkillRuntimePort FK callback is unavailable")

    manager = collision_scene_manager
    world_state = {
        "collision_policy": CollisionPolicy.WORLD_TRANSIT,
        "active_target": None,
        "support": None,
    }

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("SkillRuntimePort scene callback is unavailable")

    def scene_revision() -> int:
        return int(
            getattr(runtime, "scene_revision", getattr(runtime, "world_revision", 0))
        )

    def transition_target(
        object_name: str,
        support_name: str | None = None,
        *,
        collision_policy: CollisionPolicy | str | None = None,
    ) -> int:
        """Apply one generic collision policy through the scene manager."""

        policy = SkillRuntimePort._coerce_collision_policy(collision_policy)
        if policy is None:
            policy = world_state["collision_policy"]
        if policy is None:
            policy = CollisionPolicy.WORLD_TRANSIT
        if manager is None or scene_port is None:
            if policy is CollisionPolicy.PASSTHROUGH:
                world_state.update(
                    collision_policy=policy,
                    active_target=str(object_name),
                    support=(
                        None if support_name is None else str(support_name)
                    ),
                )
                return scene_revision()
            raise RuntimeError("SkillRuntimePort world transition is unavailable")

        object_name = str(object_name)
        support_name = None if support_name is None else str(support_name)
        if policy is CollisionPolicy.WORLD_TRANSIT:
            manager.begin_target_transit(object_name, name, arm_name)
        elif policy is CollisionPolicy.TARGET_APPROACH:
            manager.begin_target_approach(object_name, name, arm_name)
        elif policy is CollisionPolicy.ATTACHED_CARRY:
            # A placement candidate query temporarily enters PLACEMENT_CONTACT
            # to disable support colliders.  Returning to carry is a planner
            # transaction cleanup, not a world restore; the latter is only
            # legal after DETACH_AND_SETTLE.
            record = getattr(manager, "records", {}).get(object_name)
            state = getattr(getattr(record, "state", None), "value", None)
            if state == "placement_contact" and support_name:
                cleanup = getattr(manager, "restore_placement_support", None)
                if not callable(cleanup):
                    raise RuntimeError(
                        "placement query cleanup is unavailable for attached carry"
                    )
                cleanup(object_name, support_name, name, arm_name)
            assert_owner = getattr(manager, "assert_attached_owner", None)
            if callable(assert_owner):
                assert_owner(object_name, name, arm_name)
        elif policy is CollisionPolicy.PLACEMENT_DESCENT:
            if not support_name:
                raise ValueError("PLACEMENT_DESCENT requires a support entity")
            manager.begin_placement_descent(
                object_name, support_name, name, arm_name
            )
        elif policy is CollisionPolicy.RETREAT:
            manager.begin_terminal_retreat(object_name, name, arm_name)
        elif policy is CollisionPolicy.PASSTHROUGH:
            pass
        else:  # pragma: no cover - CollisionPolicy is exhaustive
            raise ValueError(f"unsupported collision policy: {policy!r}")
        world_state.update(
            collision_policy=policy,
            active_target=object_name,
            support=support_name,
        )
        return scene_revision()

    def prepare_world(
        object_name: str,
        support_name: str | None = None,
        *,
        collision_policy: CollisionPolicy | str | None = None,
    ) -> int:
        """Refresh dynamic scene state before one generic world transition."""

        if refresh_reference_world is not None:
            refresh_reference_world()
        sync = sync_dynamic_poses
        if sync is None and manager is not None:
            sync = getattr(manager, "sync_dynamic_poses", None)
        if callable(sync):
            sync(0, interval_steps=1, force=True)
        policy = SkillRuntimePort._coerce_collision_policy(collision_policy)
        if policy is None:
            policy = (
                CollisionPolicy.ATTACHED_CARRY
                if support_name is not None
                else CollisionPolicy.WORLD_TRANSIT
            )
        return transition_target(
            object_name,
            support_name,
            collision_policy=policy,
        )

    def restore_world(object_name: str) -> int:
        if manager is None:
            raise RuntimeError("SkillRuntimePort world restore is unavailable")
        manager.restore_world(str(object_name))
        world_state.update(
            collision_policy=CollisionPolicy.WORLD_TRANSIT,
            active_target=str(object_name),
            support=None,
        )
        return scene_revision()

    def diagnose_start_collision():
        if manager is None or scene_port is None:
            raise RuntimeError(
                "SkillRuntimePort start-collision diagnostics are unavailable"
            )
        return manager.diagnose_controller_world_collision(scene_port)

    def source_support(object_name: str):
        if manager is None:
            raise RuntimeError("SkillRuntimePort source-support lookup is unavailable")
        return manager.get_source_support_entity(str(object_name))

    def collision_record(object_name: str):
        if manager is None:
            raise RuntimeError("SkillRuntimePort collision records are unavailable")
        return getattr(manager, "records", {}).get(str(object_name))

    def sync_dynamic_poses_callback(
        step_id: int = 0,
        *,
        interval_steps: int = 1,
        force: bool = False,
    ):
        sync = sync_dynamic_poses
        if sync is None and manager is not None:
            sync = getattr(manager, "sync_dynamic_poses", None)
        if not callable(sync):
            return None
        return sync(
            int(step_id), interval_steps=int(interval_steps), force=bool(force)
        )

    scene_callbacks = {
        "assert_attached_owner": (
            (lambda entity: manager.assert_attached_owner(entity, name, arm_name))
            if manager is not None
            else unavailable
        ),
        "get_source_support_entity": (
            (lambda entity: manager.get_source_support_entity(entity))
            if manager is not None
            else unavailable
        ),
        "get_attached_entity": (
            (lambda: manager.get_attached_entity(name, arm_name))
            if manager is not None
            else unavailable
        ),
        "has_native_obstacle": (
            (lambda path: manager.has_native_obstacle(scene_port, path))
            if manager is not None and scene_port is not None
            else unavailable
        ),
        "attached_object_slip": (
            (lambda entity: manager.get_attached_object_slip(entity))
            if manager is not None
            else unavailable
        ),
    }

    scene_callbacks["collision_record"] = collision_record

    native_plan_pose = plan_pose
    if native_plan_pose is None:
        native_plan_pose = getattr(runtime, "plan_pose", None)
    native_plan_pose_batch = plan_pose_batch
    if native_plan_pose_batch is None:
        native_plan_pose_batch = getattr(runtime, "plan_pose_batch", None)

    def wrapped_plan_pose(position, orientation, *args, **kwargs):
        if refresh_reference_world is not None:
            refresh_reference_world()
        if not callable(native_plan_pose):
            raise RuntimeError("SkillRuntimePort pose planning callback is unavailable")
        return native_plan_pose(position, orientation, *args, **kwargs)

    def plan_pose_batch(positions, orientations, *args, **kwargs):
        if refresh_reference_world is not None:
            refresh_reference_world()
        if not callable(native_plan_pose_batch):
            raise RuntimeError("SkillRuntimePort batch pose planning callback is unavailable")
        return native_plan_pose_batch(positions, orientations, *args, **kwargs)

    def plan_cspace(goal_positions, **kwargs):
        if refresh_reference_world is not None:
            refresh_reference_world()
        return runtime.plan_cspace(goal_positions, **kwargs)

    return SkillRuntimePort(
        robot=robot,
        runtime=runtime,
        execution_state=execution_state,
        arm_spec=arm_spec,
        arm_indices=arm_indices,
        gripper_indices=gripper_indices,
        raw_joint_names=raw_joint_names,
        control_joint_names=control_joint_names,
        robot_file=robot_file,
        robot_config=robot_config,
        robot_base_path=robot_base_path,
        robot_ee_path=robot_ee_path,
        reference_prim_path=reference_prim_path,
        name=name,
        arm_name=arm_name,
        batch_capability=batch_capability,
        interpolation_dt=interpolation_dt,
        ee_pose=ee_pose,
        arm_base_pose=arm_base_pose,
        compute_fk=compute_fk,
        initial_ee_pose=initial_ee_pose,
        plan_pose=wrapped_plan_pose,
        plan_pose_batch=plan_pose_batch,
        plan_pose_result=plan_pose_result,
        plan_pose_from_path=plan_pose_from_path,
        plan_pose_from_joint_positions=plan_pose_from_joint_positions,
        measure_cartesian_path=measure_cartesian_path,
        forward_kinematic=forward_kinematic,
        arm_base_transform=arm_base_transform,
        phase_complete=phase_complete,
        sync_native_batch_attachment=sync_native_batch_attachment,
        update_pose_cost_metric=update_pose_cost_metric,
        complete_contact_phase=complete_contact_phase,
        plan_cspace=plan_cspace,
        execution_status=execution_status,
        command_status=command_status,
        execute=execute,
        dummy_forward=dummy_forward,
        hold=hold,
        clear_plan_and_hold=clear_plan_and_hold,
        push_timing_scope=push_timing_scope,
        restore_timing_scope=restore_timing_scope,
        clear_timing_scope=clear_timing_scope,
        prepare_world=prepare_world,
        transition_target=transition_target,
        restore_world=restore_world,
        diagnose_start_collision=diagnose_start_collision,
        source_support=source_support,
        collision_record=collision_record,
        refresh_world=refresh_reference_world,
        sync_dynamic_poses=sync_dynamic_poses_callback,
        collision_policy=lambda: world_state["collision_policy"],
        active_target=lambda: world_state["active_target"],
        support=lambda: world_state["support"],
        scene_callbacks=scene_callbacks,
    )


__all__ = ["SkillRuntimePort", "compose_skill_runtime_port"]
