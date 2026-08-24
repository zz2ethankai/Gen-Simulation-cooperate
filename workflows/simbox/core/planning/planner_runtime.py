"""Dependency-injected runtime facade for native CuRobo planners.

``PlannerRuntime`` owns the lifecycle and shape semantics of one native
single planner plus an optional candidate batch planner.  It deliberately does
not import CuRobo: production code supplies factories, while unit tests can
provide tiny fakes.  The facade is also the seam where scene revisions and
typed collision policies are checked before a native call is made.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any

from .domain_types import (
    BatchPlanResult,
    BatchPosePlanRequest,
    CollisionOptions,
    CollisionPolicy,
    CspacePlanRequest,
    PlanResult,
    PosePlanRequest,
    PlannerKind,
    PlannerRuntimeProfile,
    JointTrajectory,
    _plain_mapping,
    _plain_success_mask,
    _plain_value,
    _normalize_trajectory,
    PlannerStatus,
    PlannerStatusSnapshot,
    SceneUpdate,
)
from .native_planner_adapter import NativePlannerAdapter
from .native_scene_adapter import NativeSceneAdapter

LOGGER = logging.getLogger(__name__)


class PlannerRuntimeError(RuntimeError):
    """Base class for runtime boundary errors."""


class PlannerDestroyedError(PlannerRuntimeError):
    """Raised when an operation is attempted after ``destroy``."""


class PlannerFactoryError(PlannerRuntimeError):
    """Raised when a configured planner factory cannot construct a planner."""


class StaleSceneError(PlannerRuntimeError):
    """A request was created for a scene revision older than the live scene."""


class PlannerCallError(PlannerRuntimeError):
    """Native planner call failed; the original exception is chained."""


def _factory_call(factory: Any, profile: PlannerRuntimeProfile, kind: PlannerKind) -> Any:
    """Call a dependency-injection factory without swallowing its own errors.

    Factories in tests commonly have one of three forms: ``factory()``,
    ``factory(profile)`` or ``factory(profile, kind=...)``.  Signature
    inspection chooses a form before invocation, so a ``TypeError`` raised by
    the factory body is not misinterpreted as a signature mismatch.
    """

    if factory is None:
        return None
    if not callable(factory):
        return factory
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    accepts_positional = any(
        parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters.values()
    )
    kwargs = {}
    if accepts_kwargs or "kind" in parameters:
        kwargs["kind"] = kind
    if accepts_kwargs or "planner_kind" in parameters:
        kwargs["planner_kind"] = kind
    if accepts_kwargs or "profile" in parameters:
        kwargs["profile"] = profile
    # Named-only factories should receive their names; ordinary one-argument
    # callables are treated as profile factories unless they explicitly use a
    # different parameter name.
    if accepts_positional:
        positional = [
            parameter
            for parameter in parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        first = positional[0] if positional else None
        if first is not None and first.name not in {"kind", "planner_kind", "profile"}:
            return factory(profile, **kwargs)
        if first is not None and first.name == "profile":
            return factory(profile, **{key: value for key, value in kwargs.items() if key != "profile"})
        if first is not None and first.name in {"kind", "planner_kind"}:
            return factory(
                kind,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key not in {"kind", "planner_kind"}
                },
            )
    return factory(**kwargs)


def _native_result_field(raw: Any, *names: str, default: Any = None) -> Any:
    """Read a native result field without returning the native object itself."""

    for name in names:
        value = getattr(raw, name, None)
        if value is not None:
            return value
        if isinstance(raw, Mapping) and name in raw:
            value = raw[name]
            if value is not None:
                return value
    return default


def _native_result_success(raw: Any) -> Any:
    """Extract a native success field without importing torch/numpy."""

    if raw is None:
        return False
    value = _native_result_field(raw, "success")
    if value is None:
        # Some narrow fakes return a path directly.  A non-None path is a
        # successful result, while ``None`` is a failed result.
        return raw is not None
    return value


def _native_result_path(raw: Any) -> Any:
    """Extract the path-like native field for private normalization."""

    if raw is None:
        return None
    # Native v2 puts the execution path in interpolated_trajectory.  The
    # remaining names cover test doubles and older CuRobo result wrappers.
    value = _native_result_field(
        raw,
        "interpolated_trajectory",
        "trajectory",
        "path",
        "js_solution",
        "joint_trajectory",
    )
    if value is not None:
        return value
    return raw if _native_result_field(raw, "success") is None else None


def _nested_shape(value: Any, *, limit: int = 5) -> tuple[int, ...]:
    """Infer a small nested-list shape after native values are copied."""

    shape: list[int] = []
    current = value
    while len(shape) < limit and isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(shape)


def _native_joint_names(path: Any, request: Any = None) -> tuple[str, ...]:
    names = _native_result_field(path, "joint_names", "names")
    if names is None:
        names = _native_result_field(request, "joint_names", "planner_joint_names")
    if names is None and request is not None:
        metadata = getattr(request, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            names = metadata.get("joint_names", metadata.get("planner_joint_names"))
    if names is None or isinstance(names, (str, bytes)):
        return () if names is None else (str(names),)
    try:
        return tuple(str(name) for name in names)
    except TypeError:
        return ()


def _native_success_seed(raw: Any, batch_index: int) -> int:
    """Return the first successful seed for one native batch item."""

    success = _plain_value(_native_result_success(raw))
    if not isinstance(success, (list, tuple)):
        return 0
    if batch_index >= len(success):
        return 0
    seeds = success[batch_index]
    if not isinstance(seeds, (list, tuple)):
        return 0
    for seed_index, seed_ok in enumerate(seeds):
        if bool(seed_ok):
            return int(seed_index)
    return 0


def _native_last_tstep(
    raw: Any,
    batch_index: int,
    seed_index: int,
    *,
    batch_count: int | None = None,
) -> int | None:
    """Read one native interpolation horizon without exposing native values."""

    value = _plain_value(
        _native_result_field(
            raw,
            "interpolated_last_tstep",
            "path_buffer_last_tstep",
        )
    )
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # CuRobo normally reports [batch, seed].  Accept the flat [batch] and
    # [seed] forms used by smaller native wrappers/fakes as well.
    selected = value
    nested_batch = bool(selected and isinstance(selected[0], (list, tuple)))
    if nested_batch:
        if batch_index >= len(selected):
            return None
        selected = selected[batch_index]
        if isinstance(selected, (list, tuple)):
            if not selected:
                return None
            index = seed_index if seed_index < len(selected) else 0
            selected = selected[index]
    elif batch_count is not None and batch_count > 1:
        if batch_index >= len(selected):
            return None
        selected = selected[batch_index]
    elif isinstance(selected, (list, tuple)):
        if not selected:
            return None
        index = seed_index if seed_index < len(selected) else 0
        selected = selected[index]
    try:
        return int(selected)
    except (TypeError, ValueError):
        return None


def _trim_native_positions(value: Any, last_tstep: int | None) -> Any:
    """Trim CuRobo's padded interpolation tail from copied positions.

    ``interpolated_last_tstep`` is an exclusive endpoint in CuRobo's
    ``trim_joint_state_trajectory`` contract.  Keep a missing, zero, negative,
    or out-of-range endpoint as the complete path; zero is CuRobo's sentinel
    for an untrimmed trajectory.
    """

    if last_tstep is None:
        return value
    try:
        end = int(last_tstep)
    except (TypeError, ValueError):
        return value
    if end <= 0:
        return value

    shape = _nested_shape(value, limit=6)
    if len(shape) < 2 or end >= shape[-2]:
        return value
    leading = shape[:-2]
    if any(size != 1 for size in leading):
        # Non-singleton batch/seed axes must be selected before this helper is
        # called.  Refuse to guess rather than mixing candidate trajectories.
        return value

    def trim(node: Any, depth: int) -> Any:
        if depth == len(leading):
            return list(node[:end])
        return [trim(item, depth + 1) for item in node]

    return trim(value, 0)


def _native_batch_paths(
    raw: Any,
    *,
    success_mask: tuple[bool, ...],
    request: Any = None,
) -> tuple[JointTrajectory | None, ...]:
    """Copy one candidate trajectory per native batch item.

    The native shape is usually ``[B, S, T, D]`` (batch and IK seed), but
    small fakes often return ``[[T, D], ...]`` or a single path.  Shape
    inspection is performed only on copied Python values, so no torch import
    or native object escapes this helper.
    """

    native_path = _native_result_path(raw)
    count = len(success_mask)
    if native_path is None:
        return tuple(None for _ in range(count))
    names = _native_joint_names(native_path, request)
    positions = _native_result_field(native_path, "positions", "position", "values")
    if positions is None and isinstance(native_path, Mapping):
        positions = native_path
    if positions is None:
        positions = native_path
    copied = _plain_value(positions)
    shape = _nested_shape(copied)
    if not success_mask and shape:
        success_mask = (True,)

    # A plain list of paths from a fake is unambiguous when its top-level
    # length equals the candidate count and each item is itself path-like.
    if (
        count > 1
        and len(shape) <= 3
        and isinstance(copied, (list, tuple))
        and len(copied) == count
    ):
        paths = []
        for batch_index, (item, ok) in enumerate(zip(copied, success_mask)):
            if not ok:
                paths.append(None)
                continue
            seed_index = _native_success_seed(raw, batch_index)
            item = _trim_native_positions(
                item,
                _native_last_tstep(
                    raw,
                    batch_index,
                    seed_index,
                    batch_count=count,
                ),
            )
            paths.append(_normalize_trajectory(item, joint_names=names))
        return tuple(paths)

    # Convert native [B,S,T,D] or [B,T,D] into one copied path per batch item.
    # A seed dimension is selected from the first successful seed when the
    # original success field exposes it; failed candidates remain None.
    paths: list[JointTrajectory | None] = []
    if count == 0:
        count = 1 if shape else 0
    for batch_index in range(count):
        if (not success_mask[batch_index]) if batch_index < len(success_mask) else True:
            paths.append(None)
            continue
        candidate = copied
        if shape and len(shape) >= 3 and count > 1:
            try:
                candidate = copied[batch_index]
            except (IndexError, TypeError):
                candidate = copied
        seed_index = _native_success_seed(raw, batch_index)
        if shape and len(shape) >= 4:
            try:
                candidate = copied[batch_index][seed_index]
            except (IndexError, TypeError):
                candidate = copied
        elif count == 1 and shape and len(shape) >= 3:
            # A single batch item can still contain a seed dimension.
            try:
                candidate = copied[seed_index]
            except (IndexError, TypeError):
                candidate = copied
        candidate = _trim_native_positions(
            candidate,
            _native_last_tstep(
                raw,
                batch_index,
                seed_index,
                batch_count=count,
            ),
        )
        paths.append(_normalize_trajectory(candidate, joint_names=names))
    return tuple(paths)


def _native_result_metrics(raw: Any, success_mask: tuple[bool, ...]) -> Mapping[str, Any]:
    """Copy ranking/error diagnostics needed by downstream candidate logic."""

    metrics = _native_result_field(raw, "metrics", "diagnostics", "metadata", default={})
    values = dict(metrics) if isinstance(metrics, Mapping) else {}
    for key in (
        "position_error",
        "position_errors",
        "goal_position_error",
        "orientation_error",
        "orientation_errors",
        "rotation_error",
        "rotation_errors",
        "pose_error",
        "pose_errors",
        "path_cost",
        "path_costs",
        "path_length",
        "path_lengths",
        "joint_distance",
        "joint_distances",
        "trajectory_cost",
        "trajectory_costs",
        "candidate_score",
        "candidate_scores",
        "rank",
        "ranks",
        "cost",
        "costs",
        "score",
        "scores",
        "interpolated_last_tstep",
        # Keep the native feasibility split visible when a candidate batch
        # fails.  Pose error alone is not enough to tell an IK/TrajOpt
        # collision failure from a goal-convergence failure.
        "feasible",
        "converged",
        "goalset_index",
        "solve_time",
        "total_time",
        "valid_query",
    ):
        value = _native_result_field(raw, key)
        if value is not None and key not in values:
            values[key] = value
    debug_info = _native_result_field(raw, "debug_info")
    if isinstance(debug_info, str) and debug_info:
        values.setdefault("debug_info", debug_info)
    elif isinstance(debug_info, Mapping):
        # Solver traces can contain CUDA tensors and very large histories.
        # Preserve only their top-level names at this boundary.
        values.setdefault("debug_info_keys", tuple(str(key) for key in debug_info))
    values["success_count"] = int(sum(success_mask))
    values["candidate_count"] = len(success_mask)
    return _plain_mapping(values)


class PlannerRuntime:
    """Own native single/batch planners with lazy construction and teardown.

    Parameters are intentionally keyword-friendly.  Existing native planner
    instances can be injected through ``planner``/``batch_planner``; factories
    are only invoked when the corresponding planner is first needed.  A batch
    factory is never called by ordinary ``plan`` or ``update_world`` calls.
    """

    def __init__(
        self,
        profile: PlannerRuntimeProfile | Mapping[str, Any] | None = None,
        *,
        planner: Any = None,
        batch_planner: Any = None,
        planner_factory: Callable[..., Any] | Any | None = None,
        batch_planner_factory: Callable[..., Any] | Any | None = None,
        scene: Any = None,
        scene_revision: int = 0,
        name: str | None = None,
    ) -> None:
        self.profile = PlannerRuntimeProfile.from_mapping(profile)
        self.name = str(name or self.profile.name)
        self._planner = planner
        self._batch_planner = batch_planner
        self._planner_factory = (
            planner_factory
            if planner_factory is not None
            else self.profile.planner_factory
        )
        self._batch_planner_factory = (
            batch_planner_factory
            if batch_planner_factory is not None
            else self.profile.batch_planner_factory
        )
        self._scene = None
        self._scene_token = None
        self._world = None
        self._world_set = False
        self._obstacle_poses: dict[str, Any] = {}
        self._scene_revision = int(scene_revision)
        self._status = PlannerStatus.NEW
        self._last_error: Exception | None = None
        self._planning_count = 0
        self._world_update_count = 0
        self._destroyed = False
        self._warmup_done = False
        self._warmup_kinds: set[PlannerKind] = set()
        self._batch_world_synced = False
        # Scene adapters are lifecycle-owned by this runtime.  Keep an
        # explicit collection from construction onward so teardown and
        # introspection are deterministic even before a scene manager binds.
        self._scene_adapters: list[NativeSceneAdapter] = []
        # Scene consumers (notably CollisionSceneManager) can subscribe to
        # planner materialization without reaching through a controller
        # façade.  Listeners are called only after a newly-created planner has
        # received the cached world, so a late batch planner is never exposed
        # as an unsynchronized scene participant.
        self._planner_listeners: list[Callable[..., Any]] = []
        if planner is not None:
            self._status = PlannerStatus.READY
        if batch_planner is not None and self._world_set:
            self._batch_world_synced = True
        if scene is not None:
            self.bind_scene(scene)

    # ------------------------------------------------------------------
    # Lifecycle and introspection
    @property
    def status(self) -> PlannerStatus:
        return self._status

    @property
    def planner(self) -> Any:
        return self._planner

    @property
    def batch_planner(self) -> Any:
        return self._batch_planner

    @property
    def scene_revision(self) -> int:
        return self._scene_revision

    @property
    def world_revision(self) -> int:
        """Alias for consumers that name the planner scene revision as world."""

        return self._scene_revision

    @property
    def world(self) -> Any:
        return self._world

    @property
    def world_set(self) -> bool:
        """Whether the runtime has received a complete planning world."""

        return bool(self._world_set)

    @property
    def is_destroyed(self) -> bool:
        return self._destroyed

    @property
    def attachment_manager(self) -> Any:
        planner = self.ensure_planner()
        return getattr(planner, "attachment_manager", None)

    def snapshot(self) -> PlannerStatusSnapshot:
        return PlannerStatusSnapshot(
            status=self._status,
            planner_ready=self._planner is not None,
            batch_ready=self._batch_planner is not None,
            scene_revision=self._scene_revision,
            planning_count=self._planning_count,
            world_update_count=self._world_update_count,
            last_error=None if self._last_error is None else str(self._last_error),
        )

    def _check_alive(self) -> None:
        if self._destroyed or self._status == PlannerStatus.DESTROYED:
            raise PlannerDestroyedError(f"planner runtime {self.name!r} has been destroyed")

    def _construct(self, factory: Any, kind: PlannerKind) -> Any:
        if factory is None:
            raise PlannerFactoryError(
                f"no {kind.value} planner or factory configured for runtime {self.name!r}"
            )
        try:
            value = _factory_call(factory, self.profile, kind)
        except Exception as exc:  # preserve a stable boundary exception
            raise PlannerFactoryError(
                f"failed to construct {kind.value} planner for {self.name!r}: {exc}"
            ) from exc
        if value is None:
            raise PlannerFactoryError(
                f"{kind.value} planner factory returned None for {self.name!r}"
            )
        return value

    def register_planner_listener(
        self,
        listener: Callable[..., Any],
        *,
        replay: bool = True,
    ) -> Callable[..., Any]:
        """Subscribe to native planner materialization events.

        The callback receives ``(planner, kind, world, scene_revision)``.  A
        replay is useful for composition code that binds after the single
        planner has already been constructed; collision-scene binding uses
        ``replay=False`` after auditing the currently materialized planners.
        """

        self._check_alive()
        if not callable(listener):
            raise TypeError("planner listener must be callable")
        if listener not in self._planner_listeners:
            self._planner_listeners.append(listener)
        if replay:
            if self._planner is not None:
                listener(
                    self._planner,
                    PlannerKind.SINGLE,
                    self._world,
                    self._scene_revision,
                )
            if self._batch_planner is not None:
                listener(
                    self._batch_planner,
                    PlannerKind.BATCH,
                    self._world,
                    self._scene_revision,
                )
        return listener

    def unregister_planner_listener(self, listener: Callable[..., Any]) -> None:
        try:
            self._planner_listeners.remove(listener)
        except ValueError:
            pass

    # Naming aliases make the seam discoverable to lightweight runtime ports
    # while keeping one implementation and no controller dependency.
    subscribe_planner = register_planner_listener
    register_planner_materialization_listener = register_planner_listener

    def _notify_planner_materialized(self, planner: Any, kind: PlannerKind) -> None:
        for listener in tuple(self._planner_listeners):
            listener(planner, kind, self._world, self._scene_revision)

    def register_scene_adapter(self, adapter: NativeSceneAdapter) -> NativeSceneAdapter:
        """Track an adapter's world/revision metadata with this runtime.

        Native world updates are still performed exactly once by the runtime;
        this registration only keeps the adapter's cached provenance current.
        """

        if not isinstance(adapter, NativeSceneAdapter):
            raise TypeError("PlannerRuntime requires a NativeSceneAdapter")
        adapters = self._scene_adapters
        if not any(existing is adapter for existing in adapters):
            adapters.append(adapter)
        if self._world_set:
            adapter.set_cached_world(self._world, revision=self._scene_revision)
        else:
            adapter.set_scene_revision(self._scene_revision)
        return adapter

    def unregister_scene_adapter(self, adapter: NativeSceneAdapter) -> None:
        adapters = getattr(self, "_scene_adapters", ())
        try:
            adapters.remove(adapter)
        except ValueError:
            pass

    def _sync_scene_adapter_metadata(self, *, revision: int | None = None) -> None:
        adapters = getattr(self, "_scene_adapters", ())
        current_revision = self._scene_revision if revision is None else int(revision)
        for adapter in tuple(adapters):
            adapter.set_cached_world(self._world, revision=current_revision)

    def _replay_cached_obstacle_poses(self, planner: Any) -> None:
        """Replay dynamic poses that predate native planner materialization."""

        if not self._obstacle_poses:
            return
        NativeSceneAdapter(planner, strict=True).update_obstacle_poses(
            self._obstacle_poses,
            revision=self._scene_revision,
        )

    def _warmup_native(self, planner: Any, kind: PlannerKind) -> None:
        """Warm one materialized native planner after its world is installed.

        Native planner factories intentionally return an un-warmed planner.
        CuRobo warmup captures solver/graph state, so the current world must be
        injected first.  Empty profile configuration preserves the generic
        runtime's historical behavior for injected/fake planners.
        """

        if kind in self._warmup_kinds:
            return
        method = getattr(planner, "warmup", None)
        if not callable(method) or not self.profile.warmup_config:
            return
        method(**dict(self.profile.warmup_config))
        self._warmup_kinds.add(kind)
        if kind is PlannerKind.SINGLE:
            self._warmup_done = True

    def ensure_planner(self) -> Any:
        self._check_alive()
        if self._planner is None:
            self._planner = self._construct(self._planner_factory, PlannerKind.SINGLE)
            self._status = PlannerStatus.READY
            if self._world_set:
                self._update_native_world(self._planner, self._world)
                self._replay_cached_obstacle_poses(self._planner)
                self._warmup_native(self._planner, PlannerKind.SINGLE)
            try:
                self._notify_planner_materialized(self._planner, PlannerKind.SINGLE)
            except Exception as exc:
                self._last_error = exc
                self._status = PlannerStatus.FAILED
                raise
        return self._planner

    def ensure_batch_planner(self) -> Any:
        self._check_alive()
        if self._batch_planner is None:
            if not self.profile.batch_enabled:
                raise PlannerRuntimeError("batch planning is disabled by the planner profile")
            batch = self._construct(
                self._batch_planner_factory, PlannerKind.BATCH
            )
            # Keep the construction private until the complete materialization
            # transaction (world, cached poses, and scene listeners) succeeds.
            # A listener normally belongs to CollisionSceneManager and may
            # reject a partial/missing native world.  Leaving that failed
            # planner in ``_batch_planner`` would let a later query bypass the
            # audit entirely.
            self._batch_planner = batch
            # A late-created batch planner must inherit the exact current
            # world but must not cause a revision or another single update.
            self._batch_world_synced = False
            try:
                if self._world_set:
                    self._update_native_world(batch, self._world)
                    self._replay_cached_obstacle_poses(batch)
                    self._batch_world_synced = True
                    self._warmup_native(batch, PlannerKind.BATCH)
                self._notify_planner_materialized(self._batch_planner, PlannerKind.BATCH)
            except Exception as exc:
                self._batch_planner = None
                self._batch_world_synced = False
                try:
                    self._destroy_native(batch)
                except Exception as destroy_error:
                    add_note = getattr(exc, "add_note", None)
                    if callable(add_note):
                        add_note(
                            "failed batch planner cleanup after materialization error: "
                            f"{type(destroy_error).__name__}: {destroy_error}"
                        )
                self._last_error = exc
                self._status = PlannerStatus.FAILED
                raise
        return self._batch_planner

    # Friendly aliases used by callers that prefer ``get_*`` wording.
    get_planner = ensure_planner
    get_batch_planner = ensure_batch_planner
    ensure_batch = ensure_batch_planner

    # ------------------------------------------------------------------
    # Scene binding and fanout sink
    def bind_scene(self, scene: Any) -> Any:
        self._check_alive()
        if self._scene is scene:
            return self._scene_token
        self.unbind_scene()
        self._scene = scene
        subscribe = getattr(scene, "subscribe", None)
        if callable(subscribe):
            self._scene_token = subscribe(self)
        elif callable(getattr(scene, "register", None)):
            self._scene_token = scene.register(self)
        return self._scene_token

    def unbind_scene(self) -> None:
        if self._scene is None:
            return
        unsubscribe = getattr(self._scene, "unsubscribe", None)
        if not callable(unsubscribe):
            unsubscribe = getattr(self._scene, "unregister", None)
        if callable(unsubscribe):
            try:
                unsubscribe(self._scene_token if self._scene_token is not None else self)
            except (KeyError, ValueError):
                pass
        self._scene = None
        self._scene_token = None

    def on_scene_update(self, update: SceneUpdate | Any) -> None:
        """Receive a :class:`SceneUpdate` from ``SceneRuntime``."""

        self._check_alive()
        if not isinstance(update, SceneUpdate):
            # Keep the adapter useful for a minimal fake scene.
            update = SceneUpdate(
                revision=getattr(update, "revision", self._scene_revision + 1),
                world=getattr(update, "world", update),
                dynamic_poses=getattr(update, "dynamic_poses", {}),
            )
        self._scene_revision = int(update.revision)
        if update.world is not None or not self._world_set:
            self.update_world(
                update.world,
                revision=self._scene_revision,
                force=update.force,
                _from_scene=True,
            )
        if update.dynamic_poses:
            self.update_obstacle_poses(
                update.dynamic_poses,
                revision=self._scene_revision,
                _from_scene=True,
            )

    def adopt_scene_revision(self, revision: int) -> int:
        """Adopt an externally-owned scene revision without rebuilding world.

        Simulator composition may own a ``SceneRuntime`` wrapper while this
        object owns the native planners.  Once the wrapper has already
        updated the native world, this metadata-only seam keeps lazy planner
        materialization and typed result provenance on the same revision.
        """

        self._check_alive()
        revision = int(revision)
        if revision < self._scene_revision:
            raise StaleSceneError(
                f"world revision moved backwards for {self.name!r}: "
                f"current={self._scene_revision}, received={revision}"
            )
        self._scene_revision = revision
        self._sync_scene_adapter_metadata(revision=revision)
        return revision

    # ------------------------------------------------------------------
    # World update and native dispatch
    @staticmethod
    def _update_native_world(planner: Any, world: Any = None) -> Any:
        updater = getattr(planner, "update_world", None)
        if not callable(updater):
            updater = getattr(planner, "set_world", None)
        if not callable(updater):
            raise PlannerRuntimeError(
                f"native planner {type(planner).__name__} does not expose update_world"
            )
        return updater(world)

    def update_world(
        self,
        world: Any,
        *,
        revision: int | None = None,
        force: bool = False,
        policy: CollisionOptions | CollisionPolicy | Mapping[str, Any] | None = None,
        _from_scene: bool = False,
    ) -> int:
        """Update all *currently instantiated* planners in place.

        The batch planner remains lazy: ordinary world updates never
        construct it.  Its first construction receives the cached world once.
        """

        self._check_alive()
        policy = CollisionOptions.from_mapping(policy)
        if revision is not None:
            revision = int(revision)
            if revision < self._scene_revision and not policy.allow_stale_scene:
                raise StaleSceneError(
                    f"world revision moved backwards for {self.name!r}: "
                    f"current={self._scene_revision}, received={revision}"
                )
            self._scene_revision = revision
        self._world = world
        self._world_set = True
        single_missing = self._planner is None
        planners = [self.ensure_planner()]
        if self._batch_planner is not None:
            planners.append(self._batch_planner)
        try:
            for native in planners:
                if native is self._planner and single_missing:
                    # ensure_planner() synchronized a just-constructed native
                    # planner with the cached world already.
                    continue
                self._update_native_world(native, world)
                self._replay_cached_obstacle_poses(native)
                self._warmup_native(
                    native,
                    PlannerKind.SINGLE if native is self._planner else PlannerKind.BATCH,
                )
            self._batch_world_synced = self._batch_planner is not None
            self._sync_scene_adapter_metadata(revision=self._scene_revision)
            self._world_update_count += 1
            self._status = PlannerStatus.READY
        except Exception as exc:
            self._last_error = exc
            self._status = PlannerStatus.FAILED
            raise PlannerCallError(f"failed to update planner world for {self.name!r}") from exc
        return self._scene_revision

    def update_world_if_changed(
        self,
        world: Any,
        *,
        revision: int | None = None,
        force: bool = False,
        policy: CollisionOptions | CollisionPolicy | Mapping[str, Any] | None = None,
    ) -> bool:
        """Update only when the cached world object/signature changed.

        Native world configs are often mutable, so identity alone is not a
        safe check.  An explicit ``signature`` hook on the config is preferred
        and a conservative equality fallback is used for plain mappings.
        """

        if not force:
            old_signature = getattr(self._world, "signature", None)
            if callable(old_signature):
                old_signature = old_signature()
            new_signature = getattr(world, "signature", None)
            if callable(new_signature):
                new_signature = new_signature()
            if old_signature is None and new_signature is None:
                try:
                    unchanged = self._world_set and self._world == world
                except Exception:
                    unchanged = self._world is world
            else:
                unchanged = self._world_set and old_signature == new_signature
            if unchanged:
                if revision is not None:
                    revision = int(revision)
                    if revision < self._scene_revision:
                        raise StaleSceneError(
                            f"world revision moved backwards for {self.name!r}: "
                            f"current={self._scene_revision}, received={revision}"
                        )
                    self._scene_revision = revision
                    self._sync_scene_adapter_metadata(revision=revision)
                return False
        self.update_world(world, revision=revision, force=force, policy=policy)
        return True

    def update_obstacle_pose(
        self,
        name: str,
        pose: Any,
        *,
        revision: int | None = None,
        policy: CollisionOptions | CollisionPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        self.update_obstacle_poses({str(name): pose}, revision=revision, policy=policy)

    def update_obstacle_poses(
        self,
        poses: Mapping[str, Any],
        *,
        revision: int | None = None,
        policy: CollisionOptions | CollisionPolicy | Mapping[str, Any] | None = None,
        _from_scene: bool = False,
    ) -> None:
        self._check_alive()
        policy = CollisionOptions.from_mapping(policy)
        if revision is not None:
            revision = int(revision)
            if revision < self._scene_revision and not policy.allow_stale_scene:
                raise StaleSceneError(
                    f"pose revision moved backwards for {self.name!r}: "
                    f"current={self._scene_revision}, received={revision}"
                )
            self._scene_revision = revision
        self._obstacle_poses.update({str(name): pose for name, pose in poses.items()})
        single_missing = self._planner is None
        planners = [self.ensure_planner()]
        if self._batch_planner is not None:
            planners.append(self._batch_planner)
        try:
            for native in planners:
                if native is self._planner and single_missing:
                    # ensure_planner() replayed the cached poses while
                    # materializing the single planner.
                    continue
                adapter = NativeSceneAdapter(native, strict=policy.strict)
                adapter.update_obstacle_poses(poses, revision=revision)
            for adapter in tuple(getattr(self, "_scene_adapters", ())):
                adapter.set_scene_revision(
                    self._scene_revision if revision is None else revision
                )
        except Exception as exc:
            self._last_error = exc
            self._status = PlannerStatus.FAILED
            raise PlannerCallError(f"failed to update obstacle poses for {self.name!r}") from exc

    # ------------------------------------------------------------------
    # Planning
    def _check_request_revision(self, request: Any) -> None:
        if request.world_revision is None:
            return
        options = getattr(request, "collision_options", None)
        allow_stale = bool(getattr(options, "allow_stale_scene", False))
        if request.world_revision != self._scene_revision and not allow_stale:
            raise StaleSceneError(
                f"planner request revision {request.world_revision} does not match "
                f"live revision {self._scene_revision} for {self.name!r}"
            )

    def _wrap_result(
        self,
        raw: Any,
        *,
        revision: int,
        request: Any = None,
        batch: bool = False,
    ) -> PlanResult:
        """Normalize a native result without exposing it to callers.

        This is the only method in the facade allowed to inspect a native
        planner result.  The returned object contains copied Python values
        only; in particular it never stores ``raw`` or a native trajectory.
        """

        result_type = BatchPlanResult if batch else PlanResult
        if isinstance(raw, (PlanResult, BatchPlanResult)):
            # Typed results are already on the safe side of the boundary.  A
            # shallow field completion keeps request/revision provenance when
            # a narrow fake returns one directly.
            if batch and not isinstance(raw, BatchPlanResult):
                return BatchPlanResult(
                    success=(raw.success,),
                    trajectories=(raw.trajectory,),
                    status=raw.status,
                    error=raw.error,
                    source=raw.source or "typed",
                    selected_candidate_index=raw.selected_candidate_index,
                    metrics=raw.metrics,
                    request_id=raw.request_id or getattr(request, "request_id", None),
                    phase_id=raw.phase_id or getattr(request, "phase_id", None),
                    profile=raw.profile or getattr(request, "profile", None),
                    collision_policy=raw.collision_policy or getattr(request, "collision_policy", None),
                    world_revision=revision,
                    candidate_indices=raw.candidate_indices,
                )
            if not batch and isinstance(raw, BatchPlanResult):
                index = raw.selected_candidate_index
                if index is None:
                    index = next((i for i, ok in enumerate(raw.success_mask) if ok), None)
                path = raw.trajectories[index] if index is not None and index < len(raw.trajectories) else None
                success = bool(index is not None and index < len(raw.success_mask) and raw.success_mask[index])
                return PlanResult(
                    success=success,
                    trajectory=path,
                    status=raw.status,
                    error=raw.error,
                    source=raw.source or "typed",
                    selected_candidate_index=index,
                    metrics=raw.metrics,
                    request_id=raw.request_id or getattr(request, "request_id", None),
                    phase_id=raw.phase_id or getattr(request, "phase_id", None),
                    profile=raw.profile or getattr(request, "profile", None),
                    collision_policy=raw.collision_policy or getattr(request, "collision_policy", None),
                    world_revision=revision,
                    candidate_indices=raw.candidate_indices,
                )
            raw.world_revision = revision
            if raw.request_id is None:
                raw.request_id = getattr(request, "request_id", None)
            if raw.source is None:
                raw.source = "typed"
            return raw

        success_value = _native_result_success(raw)
        # A native batch planner is allowed to return ``None`` when IK finds
        # no solution.  Preserve the request cardinality in that case (and
        # for scalar test doubles) so the public result still has one entry
        # per sampled candidate.  Without this, ``None`` becomes ``(False,)``
        # and the controller reports ``0/1`` for a 20-candidate query,
        # dropping the candidate/path alignment before fallback handling.
        requested_batch_size = (
            getattr(request, "batch_size", None) if batch else None
        )
        success_mask = _plain_success_mask(
            success_value,
            count=requested_batch_size,
        )
        native_path = _native_result_path(raw)
        joint_names = _native_joint_names(native_path, request)
        request_metadata = getattr(request, "metadata", {}) or {}
        native_metrics = dict(_native_result_metrics(raw, success_mask))
        if isinstance(request_metadata, Mapping):
            # Request metadata can carry ranking context (for example the
            # expected pose-error labels); copy only values into the metrics
            # map and let native diagnostics win on key collisions.
            for key, value in request_metadata.items():
                native_metrics.setdefault(str(key), value)
        status_value = _native_result_field(raw, "status", "state")
        if status_value is None:
            status_value = "ok" if any(success_mask) else "plan_failed"
        error_value = _native_result_field(raw, "error", "reason", "message")
        source_value = _native_result_field(raw, "source", "backend") or "native"
        selected = _native_result_field(
            raw,
            "selected_candidate_index",
            "selected_index",
            "candidate_index",
        )
        if selected is None and isinstance(request_metadata, Mapping):
            selected = request_metadata.get("selected_candidate_index")
        request_id = getattr(request, "request_id", None)
        phase_id = getattr(request, "phase_id", None)
        profile = getattr(request, "profile", None)
        collision_policy = getattr(request, "collision_policy", None)

        if batch:
            paths = _native_batch_paths(raw, success_mask=success_mask, request=request)
            if not success_mask and paths:
                success_mask = tuple(path is not None for path in paths)
            candidate_indices = _native_result_field(raw, "candidate_indices", "indices")
            if candidate_indices is None:
                candidate_indices = tuple(range(len(success_mask)))
            return result_type(
                success=success_mask,
                trajectories=paths,
                status=status_value,
                error=error_value,
                source=source_value,
                selected_candidate_index=selected,
                metrics=native_metrics,
                request_id=request_id,
                phase_id=phase_id,
                profile=profile,
                collision_policy=collision_policy,
                world_revision=revision,
                candidate_indices=_plain_value(candidate_indices),
            )

        # Use the same candidate/seed selection and interpolation-tail trim as
        # batch queries.  Native single results still carry singleton batch and
        # seed axes, and CuRobo pads those buffers to a fixed horizon.
        single_paths = _native_batch_paths(
            raw,
            success_mask=(bool(any(success_mask)),),
            request=request,
        )
        trajectory = single_paths[0] if single_paths else None
        return result_type(
            success=success_value,
            trajectory=trajectory,
            status=status_value,
            error=error_value,
            source=source_value,
            selected_candidate_index=selected,
            metrics=native_metrics,
            request_id=request_id,
            phase_id=phase_id,
            profile=profile,
            collision_policy=collision_policy,
            world_revision=revision,
        )

    def _execute_typed(
        self,
        request: PosePlanRequest | BatchPosePlanRequest | CspacePlanRequest,
        *,
        native_call: Callable[[NativePlannerAdapter], Any],
        batch: bool,
        operation: str,
    ) -> PlanResult:
        self._check_alive()
        self._check_request_revision(request)
        if batch:
            if request.batch_size is not None and request.batch_size > self.profile.max_batch_size:
                raise ValueError(
                    f"batch size {request.batch_size} exceeds profile maximum {self.profile.max_batch_size}"
                )
            native = self.ensure_batch_planner()
        else:
            native = self.ensure_planner()
        self._status = PlannerStatus.PLANNING
        try:
            raw = native_call(NativePlannerAdapter(native))
            result = self._wrap_result(
                raw,
                revision=self._scene_revision,
                request=request,
                batch=batch,
            )
            self._planning_count += 1
            self._status = PlannerStatus.READY
            self._last_error = None
            return result
        except Exception as exc:
            self._last_error = exc
            self._status = PlannerStatus.FAILED
            if isinstance(exc, PlannerRuntimeError):
                raise
            raise PlannerCallError(f"{operation} failed for {self.name!r}") from exc

    @staticmethod
    def _require_request(request: Any, expected: type[Any]) -> None:
        if not isinstance(request, expected):
            raise TypeError(
                f"PlannerRuntime expects {expected.__name__}; got {type(request).__name__}"
            )

    def plan_pose(self, request: PosePlanRequest) -> PlanResult:
        """Plan one typed tool-pose request."""

        self._require_request(request, PosePlanRequest)
        return self._execute_typed(
            request,
            native_call=lambda adapter: adapter.plan_pose(request),
            batch=False,
            operation="pose planning",
        )

    def plan_pose_batch(self, request: BatchPosePlanRequest) -> BatchPlanResult:
        """Plan one typed batch tool-pose request."""

        self._require_request(request, BatchPosePlanRequest)
        return self._execute_typed(
            request,
            native_call=lambda adapter: adapter.plan_pose(request),
            batch=True,
            operation="batch pose planning",
        )

    def plan_cspace(self, request: CspacePlanRequest) -> PlanResult:
        """Plan one typed joint-space request."""

        self._require_request(request, CspacePlanRequest)
        return self._execute_typed(
            request,
            native_call=lambda adapter: adapter.plan_cspace(request),
            batch=False,
            operation="cspace planning",
        )

    def warmup(self, *args: Any, **kwargs: Any) -> Any:
        self._check_alive()
        native = self.ensure_planner()
        if PlannerKind.SINGLE in self._warmup_kinds and not args and not kwargs:
            return None
        method = getattr(native, "warmup", None)
        if not callable(method):
            return None
        if not args and not kwargs:
            kwargs = dict(self.profile.warmup_config)
        result = method(*args, **kwargs)
        self._warmup_done = True
        self._warmup_kinds.add(PlannerKind.SINGLE)
        return result

    @property
    def warmup_done(self) -> bool:
        return self._warmup_done

    # ------------------------------------------------------------------
    # Teardown
    @staticmethod
    def _destroy_native(native: Any) -> None:
        if native is None:
            return
        for method_name in ("destroy", "close", "shutdown"):
            method = getattr(native, method_name, None)
            if callable(method):
                method()
                return

    def destroy(self) -> None:
        if self._destroyed:
            return
        self.unbind_scene()
        errors: list[Exception] = []
        # Destroy both even if one fails; native resources are independent.
        for native in (self._batch_planner, self._planner):
            try:
                self._destroy_native(native)
            except Exception as exc:
                errors.append(exc)
        self._batch_planner = None
        self._planner = None
        self._obstacle_poses.clear()
        self._planner_listeners.clear()
        getattr(self, "_scene_adapters", []).clear()
        self._destroyed = True
        self._status = PlannerStatus.DESTROYED
        if errors:
            self._last_error = errors[0]
            raise PlannerRuntimeError(
                f"one or more native planners failed to destroy for {self.name!r}"
            ) from errors[0]

    close = destroy

    def __enter__(self) -> "PlannerRuntime":
        self._check_alive()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()


__all__ = [
    "PlannerCallError",
    "PlannerDestroyedError",
    "PlannerFactoryError",
    "PlannerRuntime",
    "PlannerRuntimeError",
    "StaleSceneError",
]
