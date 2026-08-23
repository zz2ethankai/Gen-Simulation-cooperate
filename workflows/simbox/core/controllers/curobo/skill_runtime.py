"""The only CuRobo runtime boundary visible to manipulation Skills.

The port binds the two controller components that implement the contract:
``MotionPlannerRuntime`` owns planning, scene state, and attachments, while
``ControllerExecution`` owns trajectory consumption and articulation actions.
It intentionally does not expose the TemplateController, native CuRobo
objects, or compatibility callback aliases.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

import numpy as np

from core.controllers.curobo.components import MutableExecutionState
from core.planning.domain_types import CollisionOptions, CollisionPolicy


class SkillRuntimePort:
    """Immutable, component-backed runtime contract for one manipulation Skill."""

    __slots__ = (
        "_sealed",
        "_robot",
        "_runtime",
        "_execution",
        "_execution_state",
        "_arm_indices",
        "_gripper_indices",
        "_robot_file",
        "_robot_config",
        "_robot_base_path",
        "_robot_ee_path",
        "_reference_prim_path",
        "_name",
        "_arm_name",
        "_batch_capability",
        "_interpolation_dt",
        "_initial_ee_pose",
        "_collision_scene_manager",
        "_timing_owner",
    )

    def __init__(
        self,
        *,
        runtime: Any,
        execution: Any,
        execution_state: MutableExecutionState,
        robot: Any,
        arm_indices: Sequence[int],
        gripper_indices: Sequence[int],
        robot_file: str,
        robot_config: Mapping[str, Any] | None,
        robot_base_path: str,
        robot_ee_path: str,
        reference_prim_path: str,
        name: str,
        arm_name: str,
        batch_capability: bool,
        interpolation_dt: float,
        initial_ee_pose: Any = None,
        collision_scene_manager: Any = None,
        timing_owner: Any = None,
    ) -> None:
        if runtime is None:
            raise TypeError("SkillRuntimePort requires MotionPlannerRuntime")
        if execution is None:
            raise TypeError("SkillRuntimePort requires ControllerExecution")
        if not isinstance(execution_state, MutableExecutionState):
            raise TypeError("SkillRuntimePort requires MutableExecutionState")

        object.__setattr__(self, "_sealed", False)
        object.__setattr__(self, "_robot", robot)
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_execution", execution)
        object.__setattr__(self, "_execution_state", execution_state)
        object.__setattr__(self, "_arm_indices", tuple(int(index) for index in arm_indices))
        object.__setattr__(self, "_gripper_indices", tuple(int(index) for index in gripper_indices))
        object.__setattr__(self, "_robot_file", str(robot_file))
        object.__setattr__(
            self,
            "_robot_config",
            MappingProxyType(dict(robot_config or {})),
        )
        object.__setattr__(self, "_robot_base_path", str(robot_base_path or ""))
        object.__setattr__(self, "_robot_ee_path", str(robot_ee_path or ""))
        object.__setattr__(self, "_reference_prim_path", str(reference_prim_path or ""))
        object.__setattr__(self, "_name", str(name))
        object.__setattr__(self, "_arm_name", str(arm_name))
        object.__setattr__(self, "_batch_capability", bool(batch_capability))
        object.__setattr__(self, "_interpolation_dt", float(interpolation_dt))
        object.__setattr__(self, "_initial_ee_pose", initial_ee_pose)
        object.__setattr__(self, "_collision_scene_manager", collision_scene_manager)
        object.__setattr__(self, "_timing_owner", timing_owner)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "num_plan_failed" and getattr(self, "_sealed", False):
            type(self).num_plan_failed.fset(self, value)
            return
        if getattr(self, "_sealed", False):
            raise AttributeError("SkillRuntimePort is read-only")
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # State and configuration
    # ------------------------------------------------------------------
    @property
    def robot(self) -> Any:
        return self._robot

    @property
    def name(self) -> str:
        return self._name

    @property
    def arm_name(self) -> str:
        return self._arm_name

    @property
    def arm_indices(self) -> np.ndarray:
        return np.asarray(self._arm_indices, dtype=np.int64).copy()

    @property
    def gripper_indices(self) -> np.ndarray:
        return np.asarray(self._gripper_indices, dtype=np.int64).copy()

    @property
    def robot_file(self) -> str:
        return self._robot_file

    @property
    def robot_config(self) -> Mapping[str, Any]:
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
    def batch_capability(self) -> bool:
        return self._batch_capability

    @property
    def interpolation_dt(self) -> float:
        return self._interpolation_dt

    @property
    def num_plan_failed(self) -> int:
        return int(self._execution_state.num_plan_failed)

    @num_plan_failed.setter
    def num_plan_failed(self, value: int) -> None:
        self._execution_state.num_plan_failed = int(value)

    # ------------------------------------------------------------------
    # Kinematics and planning
    # ------------------------------------------------------------------
    @staticmethod
    def _copy_pose(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, tuple):
            return tuple(np.asarray(item).copy() for item in value)
        if isinstance(value, list):
            return [np.asarray(item).copy() for item in value]
        return np.asarray(value).copy()

    def ee_pose(self):
        return self._copy_pose(self._execution.get_ee_pose())

    def arm_base_pose(self):
        return self._copy_pose(self._execution.get_armbase_pose())

    def initial_ee_pose(self):
        value = self._initial_ee_pose
        if value is None:
            setup = getattr(self._execution, "setup", None)
            value = getattr(setup, "T_world_ee_init", None)
        return self._copy_pose(value)

    def compute_fk(self, joint_positions: Any, *, joint_names: Sequence[str] | None = None):
        values = np.asarray(joint_positions, dtype=float).copy()
        if joint_names is None:
            return self._runtime.compute_fk(values)
        return self._runtime.compute_fk(
            values,
            joint_names=tuple(str(name) for name in joint_names),
        )

    def arm_base_transform(self):
        return self._execution.get_pick_armbase_transform()

    @staticmethod
    def _coerce_policy(value: CollisionPolicy | str | None) -> CollisionPolicy | None:
        if value is None:
            return None
        if isinstance(value, CollisionPolicy):
            return value
        return CollisionPolicy(str(value).lower())

    def _request_metadata(
        self,
        *,
        phase_id: str,
        collision_policy: CollisionPolicy | str | None,
        active_target: str | None,
        support: str | None,
        request_metadata: Mapping[str, Any] | None,
        collision_options: CollisionOptions | Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        metadata = dict(request_metadata or {})
        policy = self._coerce_policy(collision_policy)
        if policy is None:
            policy = self._coerce_policy(metadata.get("collision_policy"))
        if policy is None:
            policy = CollisionPolicy.WORLD_TRANSIT
        metadata.setdefault("phase_id", str(phase_id))
        metadata["collision_policy"] = policy
        if active_target is not None:
            metadata["active_target"] = str(active_target)
        if support is not None:
            metadata["support"] = str(support)

        supplied = collision_options
        if supplied is None:
            supplied = metadata.get("collision_options")
        if supplied is not None or policy in {
            CollisionPolicy.TARGET_APPROACH,
            CollisionPolicy.PLACEMENT_DESCENT,
        }:
            options = CollisionOptions.from_mapping(supplied, default_policy=policy)
            option_values = options.to_dict()
            if policy is CollisionPolicy.TARGET_APPROACH:
                option_values["allow_target_contact"] = True
                option_values["allow_target_finger_contact"] = True
            elif policy is CollisionPolicy.PLACEMENT_DESCENT:
                option_values["allow_support_contact"] = True
                option_values["allow_object_support_contact"] = True
            metadata["collision_options"] = CollisionOptions.from_mapping(
                option_values,
                default_policy=policy,
            )
        return metadata

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
        metadata = self._request_metadata(
            phase_id=phase_id,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            request_metadata=request_metadata,
            collision_options=collision_options,
        )
        return self._runtime.plan_pose(
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
        metadata = self._request_metadata(
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
        return self._runtime.plan_pose_batch(
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
        metadata = self._request_metadata(
            phase_id=phase_id,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            request_metadata=request_metadata,
            collision_options=collision_options,
        )
        return self._runtime.plan_pose_result(
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
        metadata = self._request_metadata(
            phase_id=phase_id,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            request_metadata=request_metadata,
            collision_options=collision_options,
        )
        return self._runtime.plan_pose_from_path(
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
        metadata = self._request_metadata(
            phase_id=phase_id,
            collision_policy=collision_policy,
            active_target=active_target,
            support=support,
            request_metadata=request_metadata,
            collision_options=collision_options,
        )
        kwargs["start_arm_positions"] = start_arm_positions
        return self._runtime.plan_pose_from_joint_positions(
            position,
            orientation,
            *args,
            request_metadata=metadata,
            **kwargs,
        )

    def measure_cartesian_path(self, path: Any, start: Any, goal: Any):
        return self._runtime.measure_cartesian_path(path, start, goal)

    # ------------------------------------------------------------------
    # Execution and workflow timing
    # ------------------------------------------------------------------
    def execute(self, command: Any):
        return self._execution.forward_phase_command(command)

    def dummy_forward(self, arm_action: Any, gripper_state: float):
        return self._execution.dummy_forward(arm_action, gripper_state)

    def phase_complete(self, command: Any) -> bool:
        return bool(self._execution.is_phase_command_complete(command))

    def execution_status(self, command: Any = None):
        return self._execution.execution_status(command)

    def hold(self, reason: str | None = None):
        del reason
        return self._execution.hold_action()

    def clear_plan_and_hold(self):
        return self._execution.clear_plan_and_hold()

    def push_timing_scope(self, scope):
        owner = self._timing_owner
        if owner is None:
            return None
        return owner.push_timing_scope(scope)

    def restore_timing_scope(self, previous):
        owner = self._timing_owner
        if owner is not None:
            return owner.restore_timing_scope(previous)
        return None

    def clear_timing_scope(self, scope=None):
        owner = self._timing_owner
        if owner is not None:
            return owner.clear_timing_scope(scope)
        return None

    # ------------------------------------------------------------------
    # Scene and attachment operations
    # ------------------------------------------------------------------
    def transition_target(
        self,
        object_name: str,
        support_name: str | None = None,
        *,
        collision_policy: CollisionPolicy | str | None = None,
    ):
        manager = self._collision_scene_manager
        policy = self._coerce_policy(collision_policy) or CollisionPolicy.WORLD_TRANSIT
        object_name = str(object_name)
        support_name = None if support_name is None else str(support_name)
        if manager is None:
            if policy is CollisionPolicy.PASSTHROUGH:
                return self._runtime.scene_revision
            raise RuntimeError("SkillRuntimePort world transition is unavailable")

        if policy is CollisionPolicy.WORLD_TRANSIT:
            manager.begin_target_transit(object_name, self.name, self.arm_name)
        elif policy is CollisionPolicy.TARGET_APPROACH:
            manager.begin_target_approach(object_name, self.name, self.arm_name)
        elif policy is CollisionPolicy.ATTACHED_CARRY:
            record = getattr(manager, "records", {}).get(object_name)
            state = getattr(getattr(record, "state", None), "value", None)
            if state == "placement_contact" and support_name:
                cleanup = getattr(manager, "restore_placement_support", None)
                if not callable(cleanup):
                    raise RuntimeError(
                        "placement query cleanup is unavailable for attached carry"
                    )
                cleanup(object_name, support_name, self.name, self.arm_name)
            manager.assert_attached_owner(object_name, self.name, self.arm_name)
        elif policy is CollisionPolicy.PLACEMENT_DESCENT:
            if not support_name:
                raise ValueError("PLACEMENT_DESCENT requires a support entity")
            manager.begin_placement_descent(
                object_name, support_name, self.name, self.arm_name
            )
        elif policy is CollisionPolicy.RETREAT:
            manager.begin_terminal_retreat(object_name, self.name, self.arm_name)
        elif policy is not CollisionPolicy.PASSTHROUGH:
            raise ValueError(f"unsupported collision policy: {policy!r}")
        return self._runtime.scene_revision

    def restore_world(self, object_name: str):
        manager = self._collision_scene_manager
        if manager is None:
            raise RuntimeError("SkillRuntimePort world restore is unavailable")
        manager.restore_world(str(object_name))
        return self._runtime.scene_revision

    def source_support(self, object_name: str):
        manager = self._collision_scene_manager
        if manager is None:
            raise RuntimeError("SkillRuntimePort source-support lookup is unavailable")
        return manager.get_source_support_entity(str(object_name))

    def assert_attached_owner(self, entity_name: str):
        manager = self._collision_scene_manager
        if manager is None:
            raise RuntimeError("SkillRuntimePort attachment ownership is unavailable")
        return manager.assert_attached_owner(str(entity_name), self.name, self.arm_name)

    def sync_native_batch_attachment(self, *args: Any, **kwargs: Any) -> bool:
        return bool(self._runtime.sync_native_batch_attachment(*args, **kwargs))

    def complete_contact_phase(self, command: Any):
        return self._execution.complete_terminal_place_on_contact(command)


__all__ = ["SkillRuntimePort"]
