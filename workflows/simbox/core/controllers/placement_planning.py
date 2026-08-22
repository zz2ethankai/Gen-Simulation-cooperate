"""Typed Placement planning port composed from controller operation ports.

Placement candidate generation needs a little more context than a generic pose
query: the carried object and its support have different collision identities,
and a batched candidate planner must carry the same attachment geometry as the
execution planner.  This module keeps that policy at the placement boundary.

The port deliberately stores callbacks and a :class:`PlannerScenePort`, never
the controller façade.  That makes it usable by host-side contract tests and
prevents a Skill from reaching through ``TemplateController`` to native
planner/scene implementation details.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from core.planning.collision_scene_manager import PlannerScenePort
from core.planning.domain_types import CollisionPolicy


class PlacementPlanningQueryPort(Protocol):
    """Narrow query and scene-transition surface consumed by Place."""

    lr_name: str
    robot_file: str
    batch_capability: bool

    def ee_pose(self) -> Any: ...

    def forward_kinematic(self, joint_positions: Any, *, joint_names: Any = None) -> Any: ...

    def plan_pose(
        self,
        position: Any,
        orientation: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
        support: str | None = None,
    ) -> Any: ...

    def plan_pose_batch(
        self,
        positions: Any,
        orientations: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
        support: str | None = None,
        start_paths: Any = None,
    ) -> Any: ...

    def plan_pose_result(
        self,
        position: Any,
        orientation: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
        support: str | None = None,
    ) -> Any: ...

    def plan_pose_from_path(
        self,
        position: Any,
        orientation: Any,
        start_path: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
        support: str | None = None,
    ) -> Any: ...

    def measure_cartesian_path(self, path: Any, start: Any, goal: Any) -> Any: ...

    def sync_native_batch_attachment(
        self,
        link_name: str = "attached_object",
        world_objects_pose_offset: Any = None,
    ) -> bool: ...

    def complete_terminal_place_on_contact(self, command: Any) -> None: ...


class PlacementPlanningPort(PlacementPlanningQueryPort):
    """Typed placement planner boundary.

    ``Place`` candidate evaluation calls only this object.  Every native
    request is stamped with a finite :class:`CollisionPolicy`, active target,
    support, and current scene revision through the callback supplied by
    composition.  The scene manager is addressed through the formal scene
    port, so lazy batch materialization and attachment updates remain part of
    the same scene transaction.
    """

    def __init__(
        self,
        *,
        scene_port: PlannerScenePort,
        collision_scene_manager: Any,
        execution_ee_pose: Callable[[], Any],
        execution_forward_kinematic: Callable[..., Any],
        sync_native_batch_attachment: Callable[..., bool] | None = None,
        update_pose_cost_metric: Callable[[Any], None] | None = None,
        arm_base_transform: Callable[[], Any] | None = None,
        plan_pose: Callable[..., Any] | None = None,
        plan_pose_batch: Callable[..., Any] | None = None,
        plan_pose_result: Callable[..., Any] | None = None,
        plan_pose_from_path: Callable[..., Any] | None = None,
        measure_cartesian_path: Callable[..., Any] | None = None,
        phase_complete: Callable[[Any], bool] | None = None,
        execution_status: Callable[[Any], Any] | None = None,
        complete_terminal_place_on_contact: Callable[[Any], None] | None = None,
        robot_file: str = "",
        batch_capability: bool = False,
    ) -> None:
        if not isinstance(scene_port, PlannerScenePort):
            raise TypeError("PlacementPlanningPort requires a formal PlannerScenePort")
        if collision_scene_manager is None:
            raise ValueError("PlacementPlanningPort requires CollisionSceneManager")
        if not callable(execution_ee_pose):
            raise TypeError("PlacementPlanningPort requires an EE-pose callback")
        if not callable(execution_forward_kinematic):
            raise TypeError("PlacementPlanningPort requires an FK callback")

        self.scene_port = scene_port
        self._collision_scene_manager = collision_scene_manager
        self._execution_ee_pose = execution_ee_pose
        self._execution_forward_kinematic = execution_forward_kinematic
        self._sync_native_batch_attachment = sync_native_batch_attachment
        self._update_pose_cost_metric = update_pose_cost_metric
        self._arm_base_transform = arm_base_transform
        self._plan_pose = plan_pose
        self._plan_pose_batch = plan_pose_batch
        self._plan_pose_result = plan_pose_result or plan_pose
        self._plan_pose_from_path = plan_pose_from_path
        self._measure_cartesian_path = measure_cartesian_path
        self._phase_complete = phase_complete
        self._execution_status = execution_status
        self._complete_terminal_place_on_contact = complete_terminal_place_on_contact
        self.robot_file = str(robot_file)
        self.batch_capability = bool(batch_capability)
        self.batch_enabled = self.batch_capability
        self._collision_policy = CollisionPolicy.ATTACHED_CARRY
        self._active_target: str | None = None
        self._support: str | None = None

    @property
    def lr_name(self) -> str:
        return str(self.scene_port.lr_name)

    @property
    def runtime(self) -> Any:
        """Planner-owned runtime behind the formal scene port."""

        return self.scene_port.runtime

    @property
    def world_revision(self) -> int:
        return int(self.runtime.scene_revision)

    @property
    def collision_policy(self) -> CollisionPolicy:
        return self._collision_policy

    @property
    def active_target(self) -> str | None:
        return self._active_target

    @property
    def support(self) -> str | None:
        return self._support

    @property
    def records(self) -> Mapping[str, Any]:
        """Read-only view of collision records for diagnostics.

        The manager remains the owner of records; this property is only a
        narrow diagnostic endpoint used by Place snapshots and state checks.
        """

        return self._collision_scene_manager.records

    def collision_record(self, entity_name: str) -> Any:
        return self.records.get(str(entity_name))

    def require_attached(self, object_name: str) -> Any:
        """Validate that the carried object belongs to this controller."""

        manager = self._collision_scene_manager
        assert_owner = getattr(manager, "assert_attached_owner", None)
        if callable(assert_owner):
            # The manager is the authoritative owner check.  Calling it first
            # also keeps this port easy to exercise with a narrow manager
            # double that does not expose its internal records mapping.
            return assert_owner(object_name, self.scene_port.name, self.scene_port.lr_name)
        record = self.collision_record(object_name)
        if record is None:
            raise RuntimeError(f"unknown placement object: {object_name}")
        state = getattr(record, "state", None)
        state_value = getattr(state, "value", state)
        if state_value != "attached":
            raise RuntimeError(
                "Place requires ATTACHED object state, "
                f"got {state_value!r} for {object_name}"
            )
        return record

    def prepare_world(self, object_name: str, support_name: str | None = None) -> int:
        """Synchronize the attached-object/support world before candidates."""

        self._active_target = str(object_name)
        self._support = None if support_name is None else str(support_name)
        if self._update_pose_cost_metric is not None:
            self._update_pose_cost_metric(None)
        manager = self._collision_scene_manager
        refresh = getattr(manager, "refresh_controller_reference_world", None)
        if callable(refresh):
            refresh(self.scene_port, force=True)
        sync = getattr(manager, "sync_dynamic_poses", None)
        if callable(sync):
            sync(0, interval_steps=1, force=True)
        self.transition_target(
            object_name,
            support_name,
            collision_policy=CollisionPolicy.ATTACHED_CARRY,
        )
        return self.world_revision

    def transition_target(
        self,
        object_name: str,
        support_name: str | None = None,
        *,
        collision_policy: CollisionPolicy,
        support: str | None = None,
    ) -> int:
        """Apply a placement target/support transition through the manager."""

        if not isinstance(collision_policy, CollisionPolicy):
            raise TypeError("Placement target transitions require CollisionPolicy")
        if support_name is None:
            support_name = support
        self._active_target = str(object_name)
        self._support = None if support_name is None else str(support_name)
        self._collision_policy = collision_policy
        manager = self._collision_scene_manager
        robot, arm = self.scene_port.name, self.scene_port.lr_name

        if collision_policy in {
            CollisionPolicy.WORLD_TRANSIT,
            CollisionPolicy.ATTACHED_CARRY,
        }:
            self.require_attached(object_name)
        elif collision_policy is CollisionPolicy.PLACEMENT_DESCENT:
            if not self._support:
                raise ValueError("PLACEMENT_DESCENT requires a support entity")
            manager.begin_placement_descent(
                str(object_name), self._support, robot, arm
            )
        elif collision_policy is CollisionPolicy.RETREAT:
            manager.begin_terminal_retreat(str(object_name), robot, arm)
        else:
            raise ValueError(
                "Placement target transitions support only WORLD_TRANSIT, "
                "ATTACHED_CARRY, PLACEMENT_DESCENT, or RETREAT policies"
            )
        return self.world_revision

    def restore_world(self, object_name: str) -> int:
        self._collision_scene_manager.restore_world(str(object_name))
        self._collision_policy = CollisionPolicy.WORLD_TRANSIT
        self._active_target = str(object_name)
        return self.world_revision

    def diagnose_start_collision(self) -> Mapping[str, Any]:
        return self._collision_scene_manager.diagnose_controller_world_collision(
            self.scene_port
        )

    def source_support(self, object_name: str) -> str | None:
        return self._collision_scene_manager.get_source_support_entity(str(object_name))

    def has_native_obstacle(self, path: str) -> bool:
        return bool(self._collision_scene_manager.has_native_obstacle(self.scene_port, str(path)))

    def sync_dynamic_poses(
        self, step_id: int = 0, *, interval_steps: int = 1, force: bool = False
    ) -> Any:
        return self._collision_scene_manager.sync_dynamic_poses(
            int(step_id), interval_steps=int(interval_steps), force=bool(force)
        )

    def attached_object_slip(self, object_name: str) -> Any:
        getter = getattr(self._collision_scene_manager, "get_attached_object_slip", None)
        return getter(str(object_name)) if callable(getter) else None

    @staticmethod
    def _request_metadata(
        collision_policy: CollisionPolicy | None,
        *,
        phase_id: str,
        active_target: str | None,
        support: str | None,
        default_policy: CollisionPolicy,
    ) -> dict[str, Any]:
        policy = default_policy if collision_policy is None else collision_policy
        if not isinstance(policy, CollisionPolicy):
            raise TypeError("Placement planner queries require CollisionPolicy")
        return {
            "phase_id": phase_id,
            "collision_policy": policy,
            "active_target": active_target,
            "support": support,
        }

    def _query_context(
        self,
        collision_policy: CollisionPolicy | None,
        active_target: str | None,
        support: str | None,
    ) -> tuple[CollisionPolicy, str | None, str | None]:
        policy = self._collision_policy if collision_policy is None else collision_policy
        target = self._active_target if active_target is None else str(active_target)
        support_value = self._support if support is None else str(support)
        if not isinstance(policy, CollisionPolicy):
            raise TypeError("Placement planner queries require CollisionPolicy")
        return policy, target, support_value

    def plan_pose(
        self,
        position: Any,
        orientation: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
        support: str | None = None,
    ) -> Any:
        """Issue one typed placement prepose query."""

        if self._plan_pose is None and self._plan_pose_result is None:
            raise RuntimeError("PlacementPlanningPort has no single planning callback")
        policy, target, support_value = self._query_context(
            collision_policy, active_target, support
        )
        callback = self._plan_pose or self._plan_pose_result
        return callback(
            position,
            orientation,
            request_metadata=self._request_metadata(
                policy,
                phase_id="place_preplace",
                active_target=target,
                support=support_value,
                default_policy=CollisionPolicy.ATTACHED_CARRY,
            ),
        )

    def plan_pose_batch(
        self,
        positions: Any,
        orientations: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
        support: str | None = None,
        start_paths: Any = None,
    ) -> Any:
        """Issue a typed batch placement query, optionally chained from paths."""

        if self._plan_pose_batch is None:
            raise RuntimeError("PlacementPlanningPort has no batch planning callback")
        policy, target, support_value = self._query_context(
            collision_policy, active_target, support
        )
        phase_id = (
            "place_terminal_batch"
            if policy is CollisionPolicy.PLACEMENT_DESCENT
            else "place_preplace_batch"
        )
        return self._plan_pose_batch(
            positions,
            orientations,
            start_paths=start_paths,
            request_metadata=self._request_metadata(
                policy,
                phase_id=phase_id,
                active_target=target,
                support=support_value,
                default_policy=CollisionPolicy.ATTACHED_CARRY,
            ),
        )

    def plan_pose_result(
        self,
        position: Any,
        orientation: Any,
        *,
        collision_policy: CollisionPolicy | None = None,
        active_target: str | None = None,
        support: str | None = None,
    ) -> Any:
        """Issue one typed placement query and retain its result envelope."""

        if self._plan_pose_result is None:
            raise RuntimeError("PlacementPlanningPort has no single result callback")
        policy, target, support_value = self._query_context(
            collision_policy, active_target, support
        )
        phase_id = (
            "place_terminal"
            if policy is CollisionPolicy.PLACEMENT_DESCENT
            else "place_preplace"
        )
        return self._plan_pose_result(
            position,
            orientation,
            request_metadata=self._request_metadata(
                policy,
                phase_id=phase_id,
                active_target=target,
                support=support_value,
                default_policy=CollisionPolicy.ATTACHED_CARRY,
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
        support: str | None = None,
    ) -> Any:
        """Plan a terminal placement pose from a named preplace path."""

        if self._plan_pose_from_path is None:
            raise RuntimeError("PlacementPlanningPort has no path planning callback")
        policy, target, support_value = self._query_context(
            collision_policy, active_target, support
        )
        return self._plan_pose_from_path(
            position,
            orientation,
            start_path,
            request_metadata=self._request_metadata(
                policy,
                phase_id="place_terminal",
                active_target=target,
                support=support_value,
                default_policy=CollisionPolicy.PLACEMENT_DESCENT,
            ),
        )

    def measure_cartesian_path(self, path: Any, start: Any, goal: Any) -> Any:
        if self._measure_cartesian_path is None:
            raise RuntimeError("PlacementPlanningPort has no Cartesian path callback")
        return self._measure_cartesian_path(path, start, goal)

    def sync_native_batch_attachment(
        self,
        link_name: str = "attached_object",
        world_objects_pose_offset: Any = None,
    ) -> bool:
        if self._sync_native_batch_attachment is None:
            raise RuntimeError("PlacementPlanningPort has no batch attachment callback")
        return bool(
            self._sync_native_batch_attachment(
                link_name=link_name,
                world_objects_pose_offset=world_objects_pose_offset,
            )
        )

    def ee_pose(self) -> Any:
        return self._execution_ee_pose()

    def forward_kinematic(self, joint_positions: Any, *, joint_names: Any = None) -> Any:
        if joint_names is None:
            return self._execution_forward_kinematic(joint_positions)
        try:
            return self._execution_forward_kinematic(
                joint_positions, joint_names=joint_names
            )
        except TypeError:
            return self._execution_forward_kinematic(joint_positions)

    def arm_base_transform(self) -> Any:
        if self._arm_base_transform is None:
            raise RuntimeError("PlacementPlanningPort has no arm-base transform callback")
        return self._arm_base_transform()

    def phase_complete(self, command: Any) -> bool:
        if self._phase_complete is None:
            raise RuntimeError("PlacementPlanningPort has no phase completion callback")
        return bool(self._phase_complete(command))

    def execution_status(self, command: Any = None) -> Any:
        if self._execution_status is None:
            raise RuntimeError("PlacementPlanningPort has no execution-status callback")
        return self._execution_status(command)

    def complete_terminal_place_on_contact(self, command: Any) -> None:
        if self._complete_terminal_place_on_contact is None:
            raise RuntimeError(
                "PlacementPlanningPort has no terminal-contact completion callback"
            )
        self._complete_terminal_place_on_contact(command)


def compose_placement_planning_port(
    scene_port: PlannerScenePort,
    collision_scene_manager: Any,
    *,
    execution: Any,
    attachment: Any,
    runtime: Any,
    planning_queries: Any | None = None,
    arm_base_transform: Callable[[], Any] | None = None,
    setup: Any | None = None,
    robot_file: str = "",
    batch_capability: bool = False,
) -> PlacementPlanningPort:
    """Compose Placement from already-wired controller operation components."""

    # Runtime owns generic pose entrypoints; the planning-query component owns
    # named-path conversion and Cartesian validation.  Keeping those callbacks
    # explicit avoids reintroducing TemplateController planner aliases.
    queries = planning_queries
    if queries is None:
        queries = getattr(runtime, "planning_queries", None)
    if queries is None:
        raise ValueError("placement composition requires planning-query callbacks")
    if arm_base_transform is None and setup is not None:
        arm_base_transform = getattr(setup, "get_armbase_pose", None)
    if arm_base_transform is None:
        arm_base_transform = getattr(execution, "get_armbase_pose", None)
    return PlacementPlanningPort(
        scene_port=scene_port,
        collision_scene_manager=collision_scene_manager,
        execution_ee_pose=execution.get_ee_pose,
        execution_forward_kinematic=execution.forward_kinematic,
        sync_native_batch_attachment=attachment.sync_native_batch_attachment,
        update_pose_cost_metric=(
            getattr(setup, "update_pose_cost_metric", None)
            if setup is not None
            else None
        ),
        arm_base_transform=arm_base_transform,
        plan_pose=runtime.plan_pose,
        plan_pose_batch=runtime.plan_pose_batch,
        plan_pose_result=queries.plan_pose_result,
        plan_pose_from_path=queries.plan_pose_from_path,
        measure_cartesian_path=queries.measure_cartesian_path,
        phase_complete=getattr(execution, "is_phase_command_complete", None),
        execution_status=getattr(execution, "execution_status", None),
        complete_terminal_place_on_contact=execution.complete_terminal_place_on_contact,
        robot_file=robot_file,
        batch_capability=batch_capability,
    )


__all__ = [
    "PlacementPlanningPort",
    "PlacementPlanningQueryPort",
    "compose_placement_planning_port",
]
