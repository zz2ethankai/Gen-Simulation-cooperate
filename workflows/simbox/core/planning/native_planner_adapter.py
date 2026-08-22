"""Private typed adapter for native single and batch planners.

The runtime owns the public request/result contract.  This module is the only
place where that contract is translated into the two fixed CuRobo entrypoints
(``plan_pose`` and ``plan_cspace``).  It intentionally performs no string
query dispatch and never returns a native value to callers other than the
owning :class:`PlannerRuntime`.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping

from .domain_types import (
    BatchPosePlanRequest,
    CollisionMode,
    CollisionOptions,
    CollisionPolicy,
    CspacePlanRequest,
    PosePlanRequest,
)
from .native_scene_adapter import NativeSceneAdapter


class NativePlannerAdapterError(RuntimeError):
    """Raised when a native planner lacks a required fixed entrypoint."""


class NativeCollisionPolicyError(NativePlannerAdapterError):
    """Raised when a typed collision policy cannot be expressed natively."""


def _unique_names(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        values = (values,)
    result: list[str] = []
    for value in values:
        name = str(value)
        if name and name not in result:
            result.append(name)
    return tuple(result)


def _exact_names(value: Any) -> tuple[str, ...]:
    """Normalize an exact exclusion value without accepting path globs."""

    if isinstance(value, Mapping):
        names: list[str] = []
        for item in value.values():
            names.extend(_exact_names(item))
        return _unique_names(names)
    return _unique_names(value)


def _role_names(
    options: CollisionOptions,
    role: str,
    identity: str | None,
) -> tuple[str, ...]:
    """Resolve exact native collider names for one logical role.

    Entity names normally become exact collider paths in the controller
    request builder.  Direct callers may provide a native path as the entity
    identity; accepting that one unambiguous form keeps this boundary useful
    for small host-side fakes without reintroducing substring/path scans.
    """

    names = _unique_names(getattr(options, f"{role}_obstacles", ()))
    exact = options.exact_exclusions
    if not names and isinstance(exact, Mapping):
        keys = (role, f"{role}_object", f"{role}_target")
        if role == "target":
            keys = (*keys, "active_target", "active_object")
        elif role == "support":
            keys = (*keys, "support_object")
        for key in keys:
            if key in exact:
                names = _exact_names(exact[key])
                if names:
                    break
        if not names and identity is not None and identity in exact:
            names = _exact_names(exact[identity])
    if not names and identity and str(identity).startswith("/"):
        names = (str(identity),)
    return names


@dataclass(frozen=True)
class NativeCollisionOptions:
    """Concrete operations supported by CuRobo v2's public collision API.

    CuRobo v2.0.8 has no per-call ``CollisionOptions`` argument on
    ``MotionPlanner.plan_pose``/``plan_cspace``.  The only public operations
    available at this boundary are exact world-obstacle enable/disable and
    the already-attached robot collision spheres.  Keeping this value typed
    makes that limitation visible instead of leaking unsupported kwargs into
    the native call.
    """

    policy: CollisionPolicy
    target_obstacles: tuple[str, ...] = ()
    support_obstacles: tuple[str, ...] = ()
    attached_obstacles: tuple[str, ...] = ()
    excluded_obstacles: tuple[str, ...] = ()
    disable_obstacles: tuple[str, ...] = ()
    enable_obstacles: tuple[str, ...] = ()
    require_attached_spheres: bool = False
    allow_target_contact: bool = False
    allow_target_robot_contact: bool = False
    allow_support_contact: bool = False
    persist_target_disable: bool = False
    persist_support_disable: bool = False
    native_expressible: bool = True
    unsupported_reason: str | None = None

    @property
    def temporary_support_disable(self) -> tuple[str, ...]:
        if self.persist_support_disable:
            return ()
        return tuple(
            name for name in self.support_obstacles if name in self.disable_obstacles
        )


def map_collision_policy(request: Any) -> NativeCollisionOptions:
    """Map one typed planning request to deterministic native operations.

    This function is dependency-free and intentionally does not inspect a
    native planner.  Capability checks (scene checker methods and attached
    sphere state) happen immediately before a native call in
    :class:`NativePlannerAdapter`.
    """

    policy = getattr(request, "collision_policy", CollisionPolicy.WORLD_TRANSIT)
    if not isinstance(policy, CollisionPolicy):
        policy = CollisionPolicy(str(policy).lower())
    options = CollisionOptions.from_mapping(
        getattr(request, "collision_options", None),
        default_policy=policy,
    )
    if options.policy is not policy:
        raise NativeCollisionPolicyError(
            "request collision_policy and collision_options.policy disagree: "
            f"{policy.value!r} != {options.policy.value!r}"
        )

    active_target = getattr(request, "active_target", None)
    support = getattr(request, "support", None)
    target = _role_names(options, "target", active_target)
    support_paths = _role_names(options, "support", support)
    attached = _unique_names(options.attached_obstacles)
    if not attached and target and policy in {
        CollisionPolicy.ATTACHED_CARRY,
        CollisionPolicy.PLACEMENT_DESCENT,
    }:
        # The attached world identity is normally the target's exact collider
        # list.  Keep it explicit in the mapped value for diagnostics and
        # sphere/target consistency checks.
        attached = target

    unsupported_reason: str | None = None
    if options.mode is CollisionMode.DISABLED:
        unsupported_reason = (
            "CuRobo v2 public planners do not expose per-call collision disable"
        )
    elif policy is CollisionPolicy.PASSTHROUGH:
        unsupported_reason = (
            "PASSTHROUGH is an execution-only policy; it cannot be sent to "
            "a CuRobo v2 pose/cspace planner"
        )
    elif options.allow_self_collision:
        unsupported_reason = (
            "CuRobo v2 public planners do not expose per-call self-collision "
            "configuration"
        )

    require_attached = bool(
        options.require_attached_spheres
        or policy
        in {CollisionPolicy.ATTACHED_CARRY, CollisionPolicy.PLACEMENT_DESCENT}
    )
    if require_attached and not active_target:
        raise NativeCollisionPolicyError(
            f"{policy.value} requires an active attached target"
        )
    if require_attached and not attached:
        raise NativeCollisionPolicyError(
            f"{policy.value} requires exact attached collider names"
        )

    # The target world proxy must be disabled while the attached spheres are
    # active.  TARGET_APPROACH only disables it when target contact was
    # explicitly requested; ordinary approach remains a collision-checked
    # transit to the target.
    disable_target = policy in {
        CollisionPolicy.ATTACHED_CARRY,
        CollisionPolicy.PLACEMENT_DESCENT,
        CollisionPolicy.RETREAT,
    } or (policy is CollisionPolicy.TARGET_APPROACH and options.allow_target_contact)
    if disable_target and not target:
        raise NativeCollisionPolicyError(
            f"{policy.value} requests target collision semantics but no exact "
            "target collider names were provided"
        )

    disable_support = bool(options.allow_support_contact)
    if disable_support and not support_paths:
        raise NativeCollisionPolicyError(
            f"{policy.value} requests object-support contact but no exact "
            "support collider names were provided"
        )

    disabled: list[str] = list(options.excluded_obstacles)
    if disable_target:
        disabled.extend(target)
    if disable_support:
        disabled.extend(support_paths)
    enabled = list(options.included_obstacles)
    disabled_tuple = _unique_names(disabled)
    enabled_tuple = _unique_names(enabled)
    overlap = set(disabled_tuple) & set(enabled_tuple)
    if overlap:
        raise NativeCollisionPolicyError(
            "native collision policy both enables and disables exact obstacles: "
            f"{sorted(overlap)}"
        )

    # PLACEMENT_DESCENT is already entered through the scene manager's
    # persistent support exclusion.  ATTACHED_CARRY (notably post-grasp lift)
    # uses a temporary support exclusion for the planning call and restores it
    # before execution, preserving the scene manager's state machine.
    persist_support = policy is CollisionPolicy.PLACEMENT_DESCENT
    return NativeCollisionOptions(
        policy=policy,
        target_obstacles=target,
        support_obstacles=support_paths,
        attached_obstacles=attached,
        excluded_obstacles=tuple(options.excluded_obstacles),
        disable_obstacles=disabled_tuple,
        enable_obstacles=enabled_tuple,
        require_attached_spheres=require_attached,
        allow_target_contact=bool(options.allow_target_contact),
        allow_target_robot_contact=bool(options.allow_target_robot_contact),
        allow_support_contact=bool(options.allow_support_contact),
        persist_target_disable=bool(disable_target),
        persist_support_disable=persist_support,
        native_expressible=unsupported_reason is None,
        unsupported_reason=unsupported_reason,
    )


def _plain_values(value: Any) -> Any:
    """Copy a scalar/tensor-like sphere value without importing torch."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            value = detach()
        except Exception:
            pass
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        try:
            value = cpu()
        except Exception:
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _plain_values(tolist())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_plain_values(item) for item in value]
    return value


def _has_positive_attached_spheres(planner: Any, link_name: str = "attached_object") -> bool:
    """Check the public CuRobo kinematics sphere state for an attachment."""

    for owner in (planner, getattr(planner, "attachment_manager", None)):
        marker = getattr(owner, "has_attached_collision_spheres", None)
        if callable(marker):
            try:
                return bool(marker(link_name))
            except TypeError:
                return bool(marker())
    kinematics = getattr(planner, "kinematics", None)
    config = getattr(getattr(kinematics, "config", None), "kinematics_config", None)
    getter = getattr(config, "get_link_spheres", None)
    if not callable(getter):
        return False
    try:
        values = _plain_values(getter(link_name))
    except Exception:
        return False
    if not isinstance(values, (list, tuple)):
        return False
    # Sphere rows are [x, y, z, radius].  A fitted attachment has at least
    # one positive radius; reset/empty slots use zero or a negative sentinel.
    for row in values:
        if isinstance(row, (list, tuple)) and len(row) >= 4:
            try:
                if float(row[3]) > 0.0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


class _NativeCollisionScope(AbstractContextManager):
    """Apply exact native obstacle state around one planner invocation."""

    def __init__(self, adapter: "NativePlannerAdapter", options: NativeCollisionOptions):
        self.adapter = adapter
        self.options = options
        self.scene: NativeSceneAdapter | None = None
        self._temporary_support: tuple[str, ...] = ()

    def __enter__(self) -> NativeCollisionOptions:
        if not self.options.native_expressible:
            raise NativeCollisionPolicyError(self.options.unsupported_reason or "unsupported collision policy")
        if self.options.require_attached_spheres:
            if getattr(self.adapter._planner, "attachment_manager", None) is None:
                raise NativeCollisionPolicyError(
                    f"{self.options.policy.value} requires a native attachment_manager"
                )
            if not _has_positive_attached_spheres(self.adapter._planner):
                raise NativeCollisionPolicyError(
                    f"{self.options.policy.value} requires active attached_object collision spheres"
                )

        actions = self.options.disable_obstacles or self.options.enable_obstacles
        if actions:
            try:
                self.scene = NativeSceneAdapter(self.adapter._planner, strict=True)
            except Exception as exc:
                raise NativeCollisionPolicyError(
                    f"{self.options.policy.value} requires a native scene collision checker"
                ) from exc

        # Explicit enables happen first, then policy exclusions win on any
        # accidental duplicate (overlap is rejected by map_collision_policy).
        if self.scene is not None:
            try:
                for name in self.options.enable_obstacles:
                    self.scene.set_obstacle_enabled(name, True)
                for name in self.options.disable_obstacles:
                    self.scene.set_obstacle_enabled(name, False)
            except Exception as exc:
                raise NativeCollisionPolicyError(
                    f"{self.options.policy.value} could not apply exact native "
                    f"collision obstacles: {exc}"
                ) from exc
            self._temporary_support = tuple(
                name
                for name in self.options.temporary_support_disable
                if name not in self.options.enable_obstacles
                and name not in self.options.excluded_obstacles
            )
        return self.options

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.scene is None or not self._temporary_support:
            return False
        try:
            for name in self._temporary_support:
                self.scene.set_obstacle_enabled(name, True)
        except Exception as restore_error:
            if exc is not None:
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(
                        "native collision policy support restore failed: "
                        f"{type(restore_error).__name__}: {restore_error}"
                    )
                return False
            raise NativeCollisionPolicyError(
                "native collision policy support restore failed"
            ) from restore_error
        return False


class NativePlannerAdapter:
    """Call one injected native planner through explicit typed operations."""

    def __init__(self, planner: Any) -> None:
        self._planner = planner

    @staticmethod
    def map_collision_policy(request: Any) -> NativeCollisionOptions:
        return map_collision_policy(request)

    def collision_policy_scope(self, request: Any) -> _NativeCollisionScope:
        return _NativeCollisionScope(self, map_collision_policy(request))

    @staticmethod
    def _call(method: Any, goal: Any, start_state: Any, kwargs: Any) -> Any:
        if not callable(method):
            raise NativePlannerAdapterError("native planner entrypoint is not callable")
        options = dict(kwargs or {})
        # Both CuRobo MotionPlanner and BatchMotionPlanner expose positional
        # (goal, current_state) arguments.  Keeping this call positional also
        # makes narrow fakes deterministic without signature reflection.
        return method(goal, start_state, **options)

    def plan_pose(self, request: PosePlanRequest | BatchPosePlanRequest) -> Any:
        method = getattr(self._planner, "plan_pose", None)
        with self.collision_policy_scope(request):
            return self._call(method, request.goal, request.start_state, request.kwargs)

    def plan_cspace(self, request: CspacePlanRequest) -> Any:
        method = getattr(self._planner, "plan_cspace", None)
        with self.collision_policy_scope(request):
            return self._call(method, request.goal_positions, request.start_state, request.kwargs)


__all__ = [
    "NativeCollisionOptions",
    "NativeCollisionPolicyError",
    "NativePlannerAdapter",
    "NativePlannerAdapterError",
    "map_collision_policy",
]
