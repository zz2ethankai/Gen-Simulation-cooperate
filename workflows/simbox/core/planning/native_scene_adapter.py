"""Thin, dependency-free adapter around a native planner's scene checker.

Native CuRobo releases have used slightly different names for collision-world
operations.  ``NativeSceneAdapter`` centralizes those spellings while keeping
the rest of SimBox unaware of native objects.  It intentionally accepts any
object with the relevant methods, which makes scene fanout tests possible
without importing CuRobo or USD.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


class NativeSceneAdapterError(RuntimeError):
    """Raised when a native planner cannot service a scene operation."""


def _checker(planner: Any) -> Any:
    for name in ("scene_collision_checker", "collision_checker", "scene_checker"):
        value = getattr(planner, name, None)
        if value is not None:
            return value
    return None


def _method(obj: Any, names: tuple[str, ...]) -> Callable[..., Any] | None:
    for name in names:
        value = getattr(obj, name, None)
        if callable(value):
            return value
    return None


def _presence_bool(value: Any) -> bool:
    """Normalize scalar/one-environment native presence results."""

    if isinstance(value, Mapping):
        return bool(value) and all(_presence_bool(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return bool(value) and all(_presence_bool(item) for item in value)
    # Torch/NumPy scalar wrappers expose ``item``; tensors with one element
    # do as well.  For a multi-environment result, ``all`` is the strict
    # interpretation: the exact collider must be present everywhere.
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
            value = tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_presence_bool(item) for item in value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return bool(item())
        except Exception:
            pass
    return bool(value)


def _name_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (str(value),)
    if isinstance(value, Mapping):
        value = value.keys()
    try:
        return tuple(str(name) for name in value)
    except TypeError:
        return (str(value),)


class NativeSceneAdapter:
    """Adapt a native single planner or collision checker to scene events."""

    def __init__(
        self,
        planner: Any,
        *,
        world_updater: Callable[[Any, Any], Any] | None = None,
        pose_updater: Callable[[Any, str, Any], Any] | None = None,
        enabled_updater: Callable[[Any, str, bool], Any] | None = None,
        strict: bool = True,
        name: str | None = None,
        world: Any = None,
        world_revision: int = 0,
    ) -> None:
        if planner is None:
            raise ValueError("NativeSceneAdapter requires a planner or checker")
        self.planner = planner
        self.world_updater = world_updater
        self.pose_updater = pose_updater
        self.enabled_updater = enabled_updater
        self.strict = bool(strict)
        self.name = str(name or getattr(planner, "name", type(planner).__name__))
        # ``world`` is cached metadata, not a second native scene.  A late
        # batch planner has already received the current world from
        # PlannerRuntime before its adapter is registered; retaining that
        # provenance here makes the adapter's revision explicit without
        # issuing a duplicate native ``update_world`` call.
        self.world = world
        self.world_revision = int(world_revision)

    @property
    def checker(self) -> Any:
        return _checker(self.planner) or self.planner

    def update_world(self, world: Any, *, revision: int | None = None, force: bool = False) -> Any:
        """Replace/update the native world in place."""

        if self.world_updater is not None:
            result = self.world_updater(self.planner, world)
        else:
            updater = _method(self.planner, ("update_world", "set_world"))
            if updater is None:
                raise NativeSceneAdapterError(
                    f"{self.name} does not expose update_world/set_world"
                )
            result = updater(world)
        self.world = world
        if revision is not None:
            self.world_revision = int(revision)
        return result

    def set_cached_world(self, world: Any, *, revision: int | None = None) -> None:
        """Record planner world provenance without touching the native scene."""

        self.world = world
        if revision is not None:
            self.world_revision = int(revision)

    def set_scene_revision(self, revision: int) -> None:
        """Record a scene revision for metadata-only fanout updates."""

        self.world_revision = int(revision)

    def update_obstacle_pose(self, name: str, pose: Any, *, revision: int | None = None) -> Any:
        name = str(name)
        if self.pose_updater is not None:
            result = self.pose_updater(self.planner, name, pose)
        else:
            target = self.checker
            updater = _method(target, ("update_obstacle_pose", "update_pose", "set_obstacle_pose"))
            if updater is None:
                if self.strict:
                    raise NativeSceneAdapterError(
                        f"{self.name} does not expose an obstacle pose update method"
                    )
                return None
            try:
                result = updater(name, pose)
            except TypeError:
                result = updater(name=name, pose=pose)
        if revision is not None:
            self.world_revision = int(revision)
        return result

    def update_obstacle_poses(
        self,
        poses: Mapping[str, Any],
        *,
        revision: int | None = None,
    ) -> None:
        for name, pose in poses.items():
            self.update_obstacle_pose(name, pose, revision=revision)

    def set_obstacle_enabled(
        self,
        name: str,
        enabled: bool,
        *,
        revision: int | None = None,
    ) -> Any:
        name = str(name)
        if self.enabled_updater is not None:
            result = self.enabled_updater(self.planner, name, bool(enabled))
        else:
            target = self.checker
            updater = _method(target, ("enable_obstacle", "set_obstacle_enabled", "set_enabled"))
            if updater is None:
                if self.strict:
                    raise NativeSceneAdapterError(
                        f"{self.name} does not expose obstacle enable/disable"
                    )
                return None
            try:
                result = updater(name, bool(enabled))
            except TypeError:
                result = updater(name=name, enabled=bool(enabled))
        if revision is not None:
            self.world_revision = int(revision)
        return result

    def disable_obstacle(self, name: str, *, revision: int | None = None) -> Any:
        return self.set_obstacle_enabled(name, False, revision=revision)

    def enable_obstacle(self, name: str, *, revision: int | None = None) -> Any:
        return self.set_obstacle_enabled(name, True, revision=revision)

    def obstacle_names(self) -> tuple[str, ...]:
        """Return the native world's exact addressable obstacle names.

        CuRobo's v2 ``SceneCollision`` deliberately exposes presence through
        ``get_obstacle_names``/``check_obstacle_exists`` rather than a generic
        object getter.  Keeping this lookup here makes every scene consumer
        use the same native addressability contract and avoids treating a
        ``SceneCfg.objects`` diagnostic snapshot as an independently loaded
        world.
        """

        target = self.checker
        getter = _method(target, ("get_obstacle_names",))
        if getter is not None:
            try:
                names = getter()
            except TypeError:
                try:
                    names = getter(env_idx=0)
                except TypeError:
                    names = getter(0)
            return _name_tuple(names)

        if self.strict:
            raise NativeSceneAdapterError(
                f"{self.name} does not expose native obstacle presence"
            )
        return ()

    # Keep the native v2 method spellings available at this boundary.  The
    # aliases are intentional: callers should use presence checks, never the
    # removed legacy generic obstacle getter.
    def get_obstacle_names(self) -> tuple[str, ...]:
        return self.obstacle_names()

    def has_obstacle(self, name: str) -> bool:
        name = str(name)
        target = self.checker
        checker = _method(target, ("check_obstacle_exists",))
        if checker is not None:
            try:
                return _presence_bool(checker(name))
            except TypeError:
                try:
                    return _presence_bool(checker(name=name, env_idx=0))
                except TypeError:
                    return _presence_bool(checker(name, 0))
        return name in self.obstacle_names()

    def check_obstacle_exists(self, name: str) -> bool:
        return self.has_obstacle(name)

    def get_obstacle_geometry(self, name: str) -> Any:
        """Return one exact obstacle geometry through the scene boundary.

        Attachment fitting needs the native obstacle's mesh and source pose,
        but callers must not reach through ``checker.scene_model`` themselves.
        The generic ``get_obstacle`` adapter surface was intentionally removed;
        this method is the narrow geometry-only query used by attachment
        construction.  Presence is checked first so a partially loaded world
        fails explicitly instead of producing a silent fallback mesh.
        """

        name = str(name)
        if not self.has_obstacle(name):
            raise NativeSceneAdapterError(
                f"{self.name} native obstacle is not addressable: {name}"
            )
        target = self.checker
        scene_model = getattr(target, "scene_model", None)
        if isinstance(scene_model, list):
            scene_model = scene_model[0] if scene_model else None
        getter = _method(scene_model, ("get_obstacle",))
        if getter is None:
            # A few dependency-injected scene fakes expose the logical world
            # as the geometry source.  This remains inside the adapter and is
            # never a controller/runtime direct native access.
            getter = _method(self.world, ("get_obstacle",))
        if getter is None:
            raise NativeSceneAdapterError(
                f"{self.name} does not expose obstacle geometry for {name}"
            )
        try:
            obstacle = getter(name)
        except TypeError:
            obstacle = getter(name=name)
        if obstacle is None:
            raise NativeSceneAdapterError(
                f"{self.name} native obstacle geometry is missing: {name}"
            )
        return obstacle

    def get_sphere_distance(self, *args: Any, **kwargs: Any) -> Any:
        """Run the native sphere-distance diagnostic through this adapter."""

        method = _method(self.checker, ("get_sphere_distance",))
        if method is None:
            if self.strict:
                raise NativeSceneAdapterError(
                    f"{self.name} does not expose get_sphere_distance"
                )
            return None
        return method(*args, **kwargs)

    def require_obstacles(
        self,
        names: Iterable[str],
        *,
        exact: bool = False,
    ) -> tuple[str, ...]:
        """Require exact native-v2 obstacle addressability.

        ``SceneCollision`` exposes obstacle presence through
        ``check_obstacle_exists`` and ``get_obstacle_names``.  This helper is
        deliberately based on those APIs and raises in strict mode instead
        of silently accepting a partially loaded world.
        """

        expected = tuple(dict.fromkeys(str(name) for name in names))
        missing = tuple(name for name in expected if not self.has_obstacle(name))
        unexpected: tuple[str, ...] = ()
        if exact:
            actual = set(self.obstacle_names())
            unexpected = tuple(sorted(actual.difference(expected)))
        if missing or unexpected:
            raise NativeSceneAdapterError(
                f"{self.name} native obstacle presence mismatch: "
                f"missing={list(missing)} unexpected={list(unexpected)}"
            )
        return expected

    def signature(self) -> Any:
        for obj in (self.planner, self.checker):
            value = getattr(obj, "world_signature", None)
            if callable(value):
                return value()
            if value is not None:
                return value
        return None

    # SceneRuntime accepts either an ``on_scene_update`` sink or this adapter
    # directly.  Keep the method tiny and avoid importing SceneUpdate here to
    # prevent an import cycle.
    def on_scene_update(self, update: Any) -> None:
        revision = int(getattr(getattr(update, "revision", 0), "value", getattr(update, "revision", 0)))
        world = getattr(update, "world", None)
        force = bool(getattr(update, "force", False))
        if world is not None or force:
            self.update_world(world, revision=revision, force=force)
        poses = getattr(update, "dynamic_poses", {}) or {}
        if poses:
            self.update_obstacle_poses(poses, revision=revision)


class SceneFanoutAdapter:
    """Broadcast a scene operation to a collection of native adapters."""

    def __init__(self, adapters: Any = ()) -> None:
        self.adapters: list[NativeSceneAdapter] = list(adapters)

    def add(self, adapter: NativeSceneAdapter) -> NativeSceneAdapter:
        if adapter not in self.adapters:
            self.adapters.append(adapter)
        return adapter

    def remove(self, adapter: NativeSceneAdapter) -> None:
        if adapter in self.adapters:
            self.adapters.remove(adapter)

    def update_world(self, world: Any, *, revision: int | None = None, force: bool = False) -> None:
        for adapter in tuple(self.adapters):
            adapter.update_world(world, revision=revision, force=force)

    def update_obstacle_poses(self, poses: Mapping[str, Any], *, revision: int | None = None) -> None:
        for adapter in tuple(self.adapters):
            adapter.update_obstacle_poses(poses, revision=revision)


__all__ = ["NativeSceneAdapter", "NativeSceneAdapterError", "SceneFanoutAdapter"]
