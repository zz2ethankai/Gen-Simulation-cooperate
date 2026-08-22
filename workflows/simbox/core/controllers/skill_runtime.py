"""Small, read-only runtime contract exposed to manipulation Skills.

Skills are deliberately not given the :class:`TemplateController` façade.
The façade is an Isaac lifecycle object and also owns the component assembly;
passing it into a Skill makes it very easy to accidentally depend on a
private implementation detail (or on a stale compatibility alias).  This
module contains the narrow, simulator-facing values and callbacks that a
Skill may need while constructing or checking a typed motion command.

The port is immutable by construction.  Mutable execution counters are not
stored on the port: they are properties backed by the one
``MutableExecutionState`` owned by the controller components.  Callback
fields are intentionally private so consumers can only invoke the typed
operations and cannot replace the controller's wiring at runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from core.controllers.controller_component import MutableExecutionState


class SkillRuntimePort:
    """Immutable Skill-facing view of one arm's runtime.

    ``SkillRuntimePort`` is intentionally a small callable object instead of
    a façade proxy.  It has no ``__getattr__`` fallback and does not retain a
    controller reference.  A controller composes one instance after its
    operation components and scene port have been wired.

    The constructor accepts callbacks rather than component objects for
    pose, FK, planning, execution, and scene ownership operations.  This
    keeps the dependency direction explicit and makes host-side tests able to
    construct a port without Isaac Sim or CuRobo.
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
        "_plan_cspace_fn",
        "_execution_status_fn",
        "_command_status_fn",
        "_execute_fn",
        "_hold_fn",
        "_clear_plan_and_hold_fn",
        "_push_timing_scope_fn",
        "_restore_timing_scope_fn",
        "_clear_timing_scope_fn",
        "_scene_callbacks",
    )

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
        plan_cspace: Callable[..., Any] | None = None,
        execution_status: Callable[..., Any] | None = None,
        command_status: Callable[..., Any] | None = None,
        execute: Callable[..., Any] | None = None,
        hold: Callable[..., Any] | None = None,
        clear_plan_and_hold: Callable[[], Any] | None = None,
        push_timing_scope: Callable[[Any], Any] | None = None,
        restore_timing_scope: Callable[[Any], Any] | None = None,
        clear_timing_scope: Callable[[Any], Any] | None = None,
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
        object.__setattr__(self, "_plan_cspace_fn", plan_cspace)
        object.__setattr__(self, "_execution_status_fn", execution_status)
        object.__setattr__(self, "_command_status_fn", command_status)
        object.__setattr__(self, "_execute_fn", execute)
        object.__setattr__(self, "_hold_fn", hold)
        object.__setattr__(self, "_clear_plan_and_hold_fn", clear_plan_and_hold)
        object.__setattr__(self, "_push_timing_scope_fn", push_timing_scope)
        object.__setattr__(self, "_restore_timing_scope_fn", restore_timing_scope)
        object.__setattr__(self, "_clear_timing_scope_fn", clear_timing_scope)
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

    def plan_pose(self, *args, **kwargs):
        if self._plan_pose_fn is None:
            raise RuntimeError("SkillRuntimePort does not expose pose planning")
        return self._plan_pose_fn(*args, **kwargs)

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
    initial_ee_pose: Callable[[], Any] | None,
    execution_status: Callable[..., Any],
    command_status: Callable[..., Any],
    execute: Callable[..., Any] | None = None,
    hold: Callable[..., Any] | None = None,
    clear_plan_and_hold: Callable[[], Any] | None = None,
    push_timing_scope: Callable[[Any], Any] | None = None,
    restore_timing_scope: Callable[[Any], Any] | None = None,
    clear_timing_scope: Callable[[Any], Any] | None = None,
    collision_scene_manager: Any = None,
    scene_port: Any = None,
    refresh_reference_world: Callable[[], Any] | None = None,
) -> SkillRuntimePort:
    """Compose the Skill-facing runtime view from explicit operation ports."""

    if not callable(compute_fk):
        def compute_fk(*_args, **_kwargs):
            raise RuntimeError("SkillRuntimePort FK callback is unavailable")

    manager = collision_scene_manager

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("SkillRuntimePort scene ownership callback is unavailable")

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

    def plan_pose(position, orientation, **kwargs):
        if refresh_reference_world is not None:
            refresh_reference_world()
        return runtime.plan_pose(position, orientation, **kwargs)

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
        plan_pose=plan_pose,
        plan_cspace=plan_cspace,
        execution_status=execution_status,
        command_status=command_status,
        execute=execute,
        hold=hold,
        clear_plan_and_hold=clear_plan_and_hold,
        push_timing_scope=push_timing_scope,
        restore_timing_scope=restore_timing_scope,
        clear_timing_scope=clear_timing_scope,
        scene_callbacks=scene_callbacks,
    )


__all__ = ["SkillRuntimePort", "compose_skill_runtime_port"]
