"""Scene revision ownership and planner fanout.

``SceneRuntime`` is intentionally independent from USD and CuRobo.  It stores
the latest world object, assigns monotonic revisions, and broadcasts immutable
``SceneUpdate`` values to registered sinks.  A sink may be a
``PlannerRuntime``, ``NativeSceneAdapter`` or a tiny callable fake.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import inspect
from typing import Any

from .domain_types import SceneRevision, SceneUpdate


class SceneRuntimeError(RuntimeError):
    """Base class for scene runtime failures."""


class SceneFanoutError(SceneRuntimeError):
    """One or more registered sinks rejected a scene update."""

    def __init__(self, message: str, errors: Mapping[Any, Exception]):
        super().__init__(message)
        self.errors = dict(errors)


@dataclass(frozen=True)
class SceneSubscription:
    """Opaque registration token returned by ``SceneRuntime.subscribe``."""

    value: int

    def __int__(self) -> int:
        return self.value


def _canonical(value: Any) -> Any:
    """Best-effort stable equality key without touching native objects."""

    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_canonical(item) for item in value))
    # Native scene configs may be mutable dataclasses or objects with no useful
    # equality.  Prefer an explicit signature hook, then repr as a safe final
    # fallback.  Identity is retained for opaque native values whose repr is
    # intentionally non-deterministic.
    try:
        hash(value)
    except Exception:
        try:
            return repr(value)
        except Exception:
            return id(value)
    return value


class SceneRuntime:
    """Own a scene world and fan revisions to all planner consumers."""

    def __init__(
        self,
        world: Any = None,
        *,
        initial_revision: int = 0,
        signature: Any = None,
        signature_fn: Callable[[Any], Any] | None = None,
        adapters: Any = (),
        fanout_errors: str = "raise",
    ) -> None:
        self._world = world
        self._signature_fn = signature_fn
        self._signature = _canonical(world) if signature is None else signature
        self._revision = SceneRevision(initial_revision)
        self._next_token = 1
        self._subscribers: dict[SceneSubscription, Any] = {}
        self._fanout_errors = str(fanout_errors).lower()
        if self._fanout_errors not in {"raise", "collect", "ignore"}:
            raise ValueError("fanout_errors must be 'raise', 'collect' or 'ignore'")
        self.last_fanout_errors: dict[Any, Exception] = {}
        for adapter in adapters or ():
            self.subscribe(adapter, replay=False)

    @property
    def world(self) -> Any:
        return self._world

    @property
    def revision(self) -> int:
        return int(self._revision)

    @property
    def scene_revision(self) -> SceneRevision:
        return self._revision

    @property
    def subscription_count(self) -> int:
        return len(self._subscribers)

    def _make_signature(self, world: Any) -> Any:
        if self._signature_fn is not None:
            return self._signature_fn(world)
        explicit = getattr(world, "signature", None)
        if callable(explicit):
            return explicit()
        if explicit is not None:
            return explicit
        return _canonical(world)

    def _event(
        self,
        *,
        world: Any = None,
        changed: bool = True,
        force: bool = False,
        dynamic_poses: Mapping[str, Any] | None = None,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SceneUpdate:
        return SceneUpdate(
            revision=self._revision,
            world=world,
            changed=changed,
            force=force,
            dynamic_poses=dynamic_poses or {},
            reason=reason,
            metadata=metadata or {},
        )

    @staticmethod
    def _deliver(sink: Any, update: SceneUpdate) -> None:
        callback = getattr(sink, "on_scene_update", None)
        if callable(callback):
            callback(update)
            return
        callback = getattr(sink, "apply_scene_update", None)
        if callable(callback):
            callback(update)
            return
        if callable(sink):
            sink(update)
            return
        # Adapters that expose only update_world/update_obstacle_poses remain
        # useful without importing SceneUpdate.
        world = getattr(sink, "update_world", None)
        if callable(world) and (update.world is not None or update.force):
            try:
                parameters = inspect.signature(world).parameters
                accepts_kwargs = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                kwargs = {
                    key: value
                    for key, value in {
                        "revision": int(update.revision),
                        "force": update.force,
                    }.items()
                    if accepts_kwargs or key in parameters
                }
            except (TypeError, ValueError):
                kwargs = {"revision": int(update.revision), "force": update.force}
            world(update.world, **kwargs)
        poses = getattr(sink, "update_obstacle_poses", None)
        if callable(poses) and update.dynamic_poses:
            try:
                parameters = inspect.signature(poses).parameters
                accepts_kwargs = any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                kwargs = {
                    "revision": int(update.revision)
                    if accepts_kwargs or "revision" in parameters
                    else None
                }
                kwargs = {key: value for key, value in kwargs.items() if value is not None}
            except (TypeError, ValueError):
                kwargs = {"revision": int(update.revision)}
            poses(update.dynamic_poses, **kwargs)
        elif update.dynamic_poses:
            pose = getattr(sink, "update_obstacle_pose", None)
            if callable(pose):
                for name, value in update.dynamic_poses.items():
                    try:
                        parameters = inspect.signature(pose).parameters
                        accepts_kwargs = any(
                            parameter.kind == inspect.Parameter.VAR_KEYWORD
                            for parameter in parameters.values()
                        )
                        kwargs = {
                            "revision": int(update.revision)
                            if accepts_kwargs or "revision" in parameters
                            else None
                        }
                        kwargs = {key: value for key, value in kwargs.items() if value is not None}
                    except (TypeError, ValueError):
                        kwargs = {"revision": int(update.revision)}
                    pose(name, value, **kwargs)

    def _fanout(self, update: SceneUpdate) -> dict[Any, Exception]:
        errors: dict[Any, Exception] = {}
        for token, sink in tuple(self._subscribers.items()):
            try:
                self._deliver(sink, update)
            except Exception as exc:
                errors[token] = exc
        self.last_fanout_errors = errors
        if errors and self._fanout_errors == "raise":
            raise SceneFanoutError(
                f"{len(errors)} scene subscriber(s) rejected revision {int(update.revision)}",
                errors,
            ) from next(iter(errors.values()))
        return errors

    # ------------------------------------------------------------------
    # Subscription/fanout
    def subscribe(self, sink: Any, *, replay: bool = True) -> SceneSubscription:
        if sink is None:
            raise ValueError("scene subscriber must not be None")
        token = SceneSubscription(self._next_token)
        self._next_token += 1
        self._subscribers[token] = sink
        if replay and (self._world is not None):
            update = self._event(world=self._world, changed=False, reason="replay")
            try:
                self._deliver(sink, update)
            except Exception as exc:
                self.last_fanout_errors = {token: exc}
                if self._fanout_errors == "raise":
                    raise SceneFanoutError(
                        f"scene subscriber rejected replay revision {self.revision}",
                        {token: exc},
                    ) from exc
        return token

    register = subscribe
    add = subscribe

    def unsubscribe(self, token_or_sink: SceneSubscription | int | Any) -> None:
        if isinstance(token_or_sink, SceneSubscription):
            self._subscribers.pop(token_or_sink, None)
            return
        if isinstance(token_or_sink, int):
            for token in tuple(self._subscribers):
                if token.value == token_or_sink:
                    self._subscribers.pop(token, None)
                    return
            return
        for token, sink in tuple(self._subscribers.items()):
            if sink is token_or_sink:
                self._subscribers.pop(token, None)

    unregister = unsubscribe
    remove = unsubscribe

    def subscribers(self) -> tuple[Any, ...]:
        return tuple(self._subscribers.values())

    # ------------------------------------------------------------------
    # Revisions
    def update_world(
        self,
        world: Any,
        *,
        force: bool = False,
        reason: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SceneUpdate:
        new_signature = self._make_signature(world)
        changed = bool(force or new_signature != self._signature)
        if not changed:
            return self._event(
                world=self._world,
                changed=False,
                force=False,
                reason=reason or "unchanged",
                metadata=metadata,
            )
        self._world = world
        self._signature = new_signature
        self._revision = SceneRevision(self.revision + 1, reason=reason)
        update = self._event(
            world=world,
            changed=True,
            force=force,
            reason=reason,
            metadata=metadata,
        )
        self._fanout(update)
        return update

    publish = update_world
    set_world = update_world
    update = update_world

    def update_poses(
        self,
        poses: Mapping[str, Any],
        *,
        force: bool = False,
        reason: str | None = "dynamic_poses",
        metadata: Mapping[str, Any] | None = None,
    ) -> SceneUpdate:
        if not isinstance(poses, Mapping):
            raise TypeError("poses must be a mapping of exact obstacle names to poses")
        if not poses and not force:
            return self._event(
                world=None,
                changed=False,
                reason=reason or "empty_dynamic_poses",
                metadata=metadata,
            )
        self._revision = SceneRevision(self.revision + 1, reason=reason)
        update = self._event(
            # A pose-only event must not make every planner rebuild/update its
            # full world.  Consumers can use SceneRuntime.world when they
            # genuinely need the cached object.
            world=None,
            changed=True,
            force=force,
            dynamic_poses=poses,
            reason=reason,
            metadata=metadata,
        )
        self._fanout(update)
        return update

    update_dynamic_poses = update_poses
    sync_dynamic_poses = update_poses

    def apply(self, update: SceneUpdate) -> SceneUpdate:
        """Apply an externally created update, preserving monotonic revisions."""

        if not isinstance(update, SceneUpdate):
            raise TypeError("apply expects a SceneUpdate")
        incoming = int(update.revision)
        if incoming < self.revision:
            raise SceneRuntimeError(
                f"cannot apply scene revision {incoming} behind current {self.revision}"
            )
        self._revision = update.revision
        if update.world is not None:
            self._world = update.world
            self._signature = self._make_signature(update.world)
        self._fanout(update)
        return update

    def __enter__(self) -> "SceneRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.clear()

    def clear(self) -> None:
        self._subscribers.clear()


__all__ = [
    "SceneFanoutError",
    "SceneRuntime",
    "SceneRuntimeError",
    "SceneRevision",
    "SceneSubscription",
    "SceneUpdate",
]
