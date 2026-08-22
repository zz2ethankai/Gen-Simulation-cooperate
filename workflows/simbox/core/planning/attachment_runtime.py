"""Transactional native attachment lifecycle.

Attaching an object changes both the native planner's attached geometry and
the collision-world enable mask.  Partial updates are particularly dangerous
when single and candidate planners are both alive, so this module treats an
attach/detach as a small transaction and restores the prior state on any
failure.  The implementation only relies on duck-typed manager/checker
methods; no CuRobo import is needed.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from .domain_types import (
    AttachmentResult,
    AttachmentSpec,
    AttachmentState,
    CollisionOptions,
    CollisionPolicy,
)

LOGGER = logging.getLogger(__name__)


class AttachmentRuntimeError(RuntimeError):
    """Base class for attachment transaction failures."""


class AttachmentRollbackError(AttachmentRuntimeError):
    """Rollback itself failed after a primary operation error."""


@dataclass(frozen=True)
class AttachmentSnapshot:
    """Value snapshot of the runtime's logical attachment state."""

    specs: tuple[AttachmentSpec, ...] = ()
    enabled_overrides: tuple[tuple[str, bool], ...] = ()

    @property
    def attached_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)


def _manager_from(value: Any) -> Any:
    manager = getattr(value, "attachment_manager", None)
    return manager if manager is not None else value


def _iter_managers(manager: Any = None, managers: Iterable[Any] | None = None) -> tuple[Any, ...]:
    values = list(managers or ())
    if manager is not None:
        values.insert(0, manager)
    result: list[Any] = []
    for value in values:
        if value is None:
            continue
        value = _manager_from(value)
        if not any(value is existing for existing in result):
            result.append(value)
    return tuple(result)


def _call_attach(manager: Any, spec: AttachmentSpec) -> Any:
    method = getattr(manager, "attach", None)
    if not callable(method):
        raise AttachmentRuntimeError(
            f"attachment manager {type(manager).__name__} does not expose attach"
        )
    kwargs = {
        "link_name": spec.link_name,
        "world_objects_pose_offset": spec.pose_offset,
        "pose_offset": spec.pose_offset,
        "disable_obstacle_names": list(spec.disable_obstacle_names),
    }
    # Remove None values; native APIs often distinguish omitted from null.
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    signature = None
    try:
        signature = inspect.signature(method)
        parameters = signature.parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    except (TypeError, ValueError):
        pass

    # Native v2 attachment takes joint state and mesh sequence positionally.
    # A narrow fake may accept one AttachmentSpec, so use signature shape to
    # select that form before invoking and never hide a body TypeError.
    try:
        signature = inspect.signature(method)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        positional = []
    if len(positional) <= 1 and not any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in getattr(signature, "parameters", {}).values()
    ):
        if positional and positional[0].name in {"spec", "attachment", "request"}:
            return method(spec, **kwargs)
    try:
        return method(spec.state, spec.meshes, **kwargs)
    except TypeError as first_error:
        # Keyword-only fakes or wrappers often call the fields ``state`` and
        # ``meshes``.  Retry only when the signature explicitly supports it.
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            raise first_error
        if "spec" in parameters or "attachment" in parameters:
            key = "spec" if "spec" in parameters else "attachment"
            return method(**{key: spec, **kwargs})
        named = {}
        if "state" in parameters:
            named["state"] = spec.state
        elif "joint_state" in parameters:
            named["joint_state"] = spec.state
        if "meshes" in parameters:
            named["meshes"] = spec.meshes
        elif "geometry" in parameters:
            named["geometry"] = spec.meshes
        if named:
            named.update(kwargs)
            return method(**named)
        raise first_error


def _call_detach(manager: Any) -> Any:
    method = getattr(manager, "detach", None)
    if not callable(method):
        raise AttachmentRuntimeError(
            f"attachment manager {type(manager).__name__} does not expose detach"
        )
    return method()


def _annotate_rollback(primary: BaseException, failures: list[tuple[str, Exception]]) -> None:
    if not failures:
        return
    previous = list(getattr(primary, "_attachment_rollback_failures", ()))
    previous.extend(failures)
    try:
        setattr(primary, "_attachment_rollback_failures", tuple(previous))
    except Exception:
        pass
    add_note = getattr(primary, "add_note", None)
    if callable(add_note):
        for operation, failure in failures:
            try:
                add_note(
                    f"attachment rollback failed during {operation}: "
                    f"{type(failure).__name__}: {failure}"
                )
            except Exception:
                pass


class AttachmentRuntime:
    """Keep attached geometry synchronized across one or more managers."""

    def __init__(
        self,
        manager: Any = None,
        *,
        managers: Iterable[Any] | None = None,
        scene: Any = None,
        policy: CollisionOptions | CollisionPolicy | Mapping[str, Any] | None = None,
        strict: bool = True,
    ) -> None:
        self.managers = _iter_managers(manager, managers)
        self.scene = scene
        self.policy = CollisionOptions.from_mapping(policy)
        self.strict = bool(strict)
        self._specs: dict[str, AttachmentSpec] = {}
        self._enabled_overrides: dict[str, bool] = {}
        self._state = AttachmentState.DETACHED
        self._last_error: Exception | None = None

    @property
    def state(self) -> AttachmentState:
        return self._state

    @property
    def attached_names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    @property
    def attached_obstacle_names(self) -> tuple[str, ...]:
        """Return exact collider names owned by the active attachment spec."""

        names: list[str] = []
        for spec in self._specs.values():
            names.extend(str(name) for name in spec.disable_obstacle_names)
        return tuple(names)

    @property
    def attached(self) -> bool:
        return bool(self._specs)

    @property
    def current(self) -> AttachmentSpec | None:
        return next(iter(self._specs.values()), None)

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def _snapshot(self) -> AttachmentSnapshot:
        return AttachmentSnapshot(
            specs=tuple(self._specs.values()),
            enabled_overrides=tuple(self._enabled_overrides.items()),
        )

    def snapshot(self) -> AttachmentSnapshot:
        return self._snapshot()

    def _scene_enabled(self, name: str, enabled: bool) -> None:
        if self.scene is None:
            return
        setter = getattr(self.scene, "set_obstacle_enabled", None)
        if callable(setter):
            setter(name, enabled)
            self._enabled_overrides[str(name)] = bool(enabled)
            return
        adapter = getattr(self.scene, "native_scene_adapter", None)
        setter = getattr(adapter, "set_obstacle_enabled", None)
        if callable(setter):
            setter(name, enabled)
            self._enabled_overrides[str(name)] = bool(enabled)
            return
        if self.strict and self.policy.strict:
            raise AttachmentRuntimeError("scene does not expose set_obstacle_enabled")

    def _restore_scene_enabled(self, snapshot: AttachmentSnapshot) -> list[tuple[str, Exception]]:
        failures: list[tuple[str, Exception]] = []
        # First restore explicit previous values, then restore default enabled
        # state for names introduced by the attempted attachment.
        previous = dict(snapshot.enabled_overrides)
        names = set(self._enabled_overrides) | set(previous)
        for name in names:
            enabled = previous.get(name, True)
            try:
                self._scene_enabled(name, enabled)
            except Exception as exc:
                failures.append((f"scene_enable:{name}", exc))
        return failures

    def _restore_snapshot(self, snapshot: AttachmentSnapshot) -> list[tuple[str, Exception]]:
        failures: list[tuple[str, Exception]] = []
        for manager in self.managers:
            try:
                _call_detach(manager)
            except Exception as exc:
                failures.append((f"detach:{type(manager).__name__}", exc))
        # Only attempt reattach if all manager detach calls succeeded.  A
        # manager in an unknown state cannot be safely composed with a second
        # attach and should remain a visible rollback diagnostic.
        if not failures:
            for spec in snapshot.specs:
                for manager in self.managers:
                    try:
                        _call_attach(manager, spec)
                    except Exception as exc:
                        failures.append((f"reattach:{spec.name}:{type(manager).__name__}", exc))
                        break
        failures.extend(self._restore_scene_enabled(snapshot))
        self._specs = {spec.name: spec for spec in snapshot.specs}
        self._enabled_overrides = dict(snapshot.enabled_overrides)
        self._state = AttachmentState.ATTACHED if snapshot.specs else AttachmentState.DETACHED
        return failures

    def _rollback_or_annotate(self, snapshot: AttachmentSnapshot, primary: Exception) -> None:
        self._state = AttachmentState.ROLLING_BACK
        failures = self._restore_snapshot(snapshot)
        _annotate_rollback(primary, failures)
        self._last_error = primary
        if failures:
            # Keep the original exception as the raised value, matching the
            # safety contract used by the controller integration.
            LOGGER.error("attachment rollback had %d failure(s)", len(failures))

    def attach(
        self,
        spec: AttachmentSpec | Mapping[str, Any] | None = None,
        *,
        name: str | None = None,
        state: Any = None,
        meshes: Any = None,
        link_name: str = "attached_object",
        pose_offset: Any = None,
        disable_obstacle_names: Iterable[str] = (),
        policy: CollisionOptions | CollisionPolicy | Mapping[str, Any] | None = None,
        **metadata: Any,
    ) -> AttachmentResult:
        if spec is None:
            if name is None:
                raise ValueError("attach requires an AttachmentSpec or name")
            spec = AttachmentSpec(
                name=name,
                state=state,
                meshes=meshes,
                link_name=link_name,
                pose_offset=pose_offset,
                disable_obstacle_names=tuple(disable_obstacle_names),
                metadata=metadata,
            )
        else:
            spec = AttachmentSpec.from_mapping(spec, **metadata)
        policy = CollisionOptions.from_mapping(policy or self.policy)
        if not self.managers:
            raise AttachmentRuntimeError("attach requires at least one attachment manager")
        snapshot = self._snapshot()
        self._state = AttachmentState.ROLLING_BACK
        try:
            # Native managers generally expose one attached object.  Replace
            # an existing logical attachment transactionally.
            if self._specs:
                for manager in self.managers:
                    _call_detach(manager)
            for manager in self.managers:
                _call_attach(manager, spec)
            for obstacle in spec.disable_obstacle_names:
                self._scene_enabled(obstacle, False)
            self._specs = {spec.name: spec}
            self._state = AttachmentState.ATTACHED
            self._last_error = None
            return AttachmentResult(spec=spec, state=AttachmentState.ATTACHED)
        except Exception as exc:
            self._rollback_or_annotate(snapshot, exc)
            raise

    def detach(self, name: str | None = None) -> AttachmentResult:
        if name is not None and name not in self._specs:
            return AttachmentResult(state=AttachmentState.DETACHED)
        if not self._specs:
            return AttachmentResult(state=AttachmentState.DETACHED)
        snapshot = self._snapshot()
        self._state = AttachmentState.ROLLING_BACK
        try:
            for manager in self.managers:
                _call_detach(manager)
            for spec in snapshot.specs:
                for obstacle in spec.disable_obstacle_names:
                    self._scene_enabled(obstacle, True)
            self._specs.clear()
            self._enabled_overrides.clear()
            self._state = AttachmentState.DETACHED
            self._last_error = None
            return AttachmentResult(state=AttachmentState.DETACHED, spec=snapshot.specs[0])
        except Exception as exc:
            self._rollback_or_annotate(snapshot, exc)
            raise

    detach_all = detach

    def restore(self, snapshot: AttachmentSnapshot) -> None:
        """Restore a previously captured snapshot, raising on failure."""

        if not isinstance(snapshot, AttachmentSnapshot):
            raise TypeError("restore expects AttachmentSnapshot")
        failures = self._restore_snapshot(snapshot)
        if failures:
            error = AttachmentRollbackError(
                f"failed to restore attachment snapshot ({len(failures)} failure(s))"
            )
            _annotate_rollback(error, failures)
            self._last_error = error
            raise error

    def begin_transaction(self) -> "AttachmentTransaction":
        return AttachmentTransaction(self)

    transaction = begin_transaction

    def __enter__(self) -> "AttachmentRuntime":
        self._check_context = self.begin_transaction()
        self._check_context.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._check_context.__exit__(exc_type, exc, tb)


class AttachmentTransaction(AbstractContextManager["AttachmentTransaction"]):
    """Explicit transaction object for multi-step attachment operations."""

    def __init__(self, runtime: AttachmentRuntime) -> None:
        self.runtime = runtime
        self.snapshot = runtime.snapshot()
        self._closed = False
        self._committed = False

    @property
    def committed(self) -> bool:
        return self._committed

    def __enter__(self) -> "AttachmentTransaction":
        if self._closed:
            raise AttachmentRuntimeError("attachment transaction is already closed")
        return self

    def attach(self, *args: Any, **kwargs: Any) -> AttachmentResult:
        if self._closed:
            raise AttachmentRuntimeError("attachment transaction is already closed")
        return self.runtime.attach(*args, **kwargs)

    def detach(self, *args: Any, **kwargs: Any) -> AttachmentResult:
        if self._closed:
            raise AttachmentRuntimeError("attachment transaction is already closed")
        return self.runtime.detach(*args, **kwargs)

    def commit(self) -> None:
        if self._closed:
            raise AttachmentRuntimeError("attachment transaction is already closed")
        self._committed = True
        self._closed = True

    def rollback(self) -> None:
        if self._closed and self._committed:
            raise AttachmentRuntimeError("cannot rollback a committed transaction")
        if self._closed:
            return
        self.runtime.restore(self.snapshot)
        self._closed = True

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._closed:
            return False
        if exc_type is not None:
            try:
                self.runtime.restore(self.snapshot)
            finally:
                self._closed = True
            return False
        self.commit()
        return False


__all__ = [
    "AttachmentRollbackError",
    "AttachmentRuntime",
    "AttachmentRuntimeError",
    "AttachmentSnapshot",
    "AttachmentTransaction",
]
