"""The single typed Physics Schema to CuRobo planning boundary."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import logging
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .domain_types import (
    BatchPlanResult, BatchPosePlanRequest, CollisionMode, CollisionOptions,
    CollisionPolicy, CspacePlanRequest, JointTrajectory, PlanResult,
    PosePlanRequest, PlannerKind, PlannerRuntimeProfile, PlannerStatus,
    PlannerStatusSnapshot,
)
LOGGER = logging.getLogger("de_logger")


def _plain_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _plain_value(tolist())
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _plain_bool(value: Any) -> bool:
    value = _plain_value(value)
    if isinstance(value, (list, tuple)):
        return any(_plain_bool(item) for item in value)
    return bool(value)


def _plain_success_mask(value: Any, *, count: int | None = None) -> tuple[bool, ...]:
    value = _plain_value(value)
    if isinstance(value, (list, tuple)):
        mask = tuple(
            any(_plain_bool(seed) for seed in item)
            if isinstance(item, (list, tuple)) else _plain_bool(item)
            for item in value
        )
    else:
        mask = (_plain_bool(value),)
    if count is not None and len(mask) != count:
        raise PlannerRuntimeError(
            f"native batch returned {len(mask)} success values for {count} candidates"
        )
    return mask


class PlannerRuntimeError(RuntimeError):
    """Base class for planning boundary errors."""


class PlannerDestroyedError(PlannerRuntimeError):
    pass


class PlannerFactoryError(PlannerRuntimeError):
    pass


class StaleSceneError(PlannerRuntimeError):
    pass


class PlannerCallError(PlannerRuntimeError):
    pass


def _names(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        return (str(values),)
    return tuple(dict.fromkeys(str(value) for value in values))


@dataclass(frozen=True)
class NativeCollisionOptions:
    """Exact native obstacle operations for one typed request."""

    policy: CollisionPolicy
    target_obstacles: tuple[str, ...] = ()
    support_obstacles: tuple[str, ...] = ()
    attached_obstacles: tuple[str, ...] = ()
    excluded_obstacles: tuple[str, ...] = ()
    disable_obstacles: tuple[str, ...] = ()
    enable_obstacles: tuple[str, ...] = ()
    require_attached_spheres: bool = False
    native_expressible: bool = True
    unsupported_reason: str | None = None

    @property
    def temporary_support_disable(self) -> tuple[str, ...]:
        if self.policy is CollisionPolicy.PLACEMENT_DESCENT:
            return ()
        return tuple(name for name in self.support_obstacles if name in self.disable_obstacles)


def map_collision_policy(request: Any) -> NativeCollisionOptions:
    """Map typed contact semantics to exact native collider paths."""

    policy = request.collision_policy
    options = request.collision_options
    if not isinstance(policy, CollisionPolicy) or not isinstance(options, CollisionOptions):
        raise TypeError("planning requests must carry typed collision policy/options")
    target = _names(options.target_obstacles)
    support = _names(options.support_obstacles)
    attached = _names(options.attached_obstacles)
    if not attached and policy in (CollisionPolicy.ATTACHED_CARRY, CollisionPolicy.PLACEMENT_DESCENT):
        attached = target
    if policy in (CollisionPolicy.ATTACHED_CARRY, CollisionPolicy.PLACEMENT_DESCENT) and not attached:
        raise PlannerRuntimeError(f"{policy.value} requires exact attached collider names")
    if options.allow_support_contact and not support:
        raise PlannerRuntimeError(f"{policy.value} requires exact support collider names")
    disabled = list(options.excluded_obstacles)
    if policy in (CollisionPolicy.ATTACHED_CARRY, CollisionPolicy.PLACEMENT_DESCENT, CollisionPolicy.RETREAT):
        disabled.extend(target)
    elif policy is CollisionPolicy.TARGET_APPROACH and options.allow_target_contact:
        disabled.extend(target)
    if options.allow_support_contact:
        disabled.extend(support)
    disabled, enabled = _names(disabled), _names(options.included_obstacles)
    overlap = set(disabled) & set(enabled)
    if overlap:
        raise PlannerRuntimeError(f"obstacles both enabled and disabled: {sorted(overlap)}")
    reason = None
    if options.mode is CollisionMode.DISABLED:
        reason = "disabled collision mode is not a native planner operation"
    elif policy is CollisionPolicy.PASSTHROUGH:
        reason = "passthrough is an execution-only policy"
    elif options.allow_self_collision:
        reason = "per-request self collision is not a native planner operation"
    return NativeCollisionOptions(
        policy=policy, target_obstacles=target, support_obstacles=support,
        attached_obstacles=attached, excluded_obstacles=_names(options.excluded_obstacles),
        disable_obstacles=disabled, enable_obstacles=enabled,
        require_attached_spheres=bool(
            options.require_attached_spheres
            or policy in (CollisionPolicy.ATTACHED_CARRY, CollisionPolicy.PLACEMENT_DESCENT)
        ),
        native_expressible=reason is None, unsupported_reason=reason,
    )


def _positive_spheres(planner: Any, link_name: str = "attached_object") -> bool:
    values = _plain_value(
        planner.kinematics.config.kinematics_config.get_link_spheres(link_name)
    )
    return any(
        isinstance(row, (list, tuple)) and len(row) >= 4 and float(row[3]) > 0
        for row in values
    )


class _CollisionScope(AbstractContextManager):
    def __init__(self, runtime: "PlannerRuntime", planner: Any, options: NativeCollisionOptions) -> None:
        self.runtime, self.planner, self.options = runtime, planner, options
        self.previous_enabled: dict[str, bool] = {}

    def __enter__(self) -> NativeCollisionOptions:
        if not self.options.native_expressible:
            raise PlannerRuntimeError(self.options.unsupported_reason or "unsupported collision policy")
        if self.options.require_attached_spheres and not _positive_spheres(self.planner):
            raise PlannerRuntimeError(f"{self.options.policy.value} requires attached collision spheres")
        names = _names((*self.options.enable_obstacles, *self.options.disable_obstacles))
        self.previous_enabled = {
            name: self.runtime._obstacle_enabled.get(name, True)
            for name in names
        }
        for name in self.options.enable_obstacles:
            self.runtime.set_obstacle_enabled(name, True, planner=self.planner)
        for name in self.options.disable_obstacles:
            self.runtime.set_obstacle_enabled(name, False, planner=self.planner)
        return self.options

    def __exit__(self, exc_type, exc, tb) -> bool:
        for name, enabled in self.previous_enabled.items():
            self.runtime.set_obstacle_enabled(name, enabled, planner=self.planner)
        return False


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(size) for size in value.shape)
    except AttributeError:
        pass
    shape: list[int] = []
    while len(shape) < 5 and isinstance(value, (list, tuple)):
        shape.append(len(value))
        value = value[0] if value else ()
    return tuple(shape)


def _last_tstep(raw: Any, seed: int, batch: int | None = None) -> int | None:
    value = _plain_value(raw.interpolated_last_tstep)
    if value is None:
        return None
    if batch is not None and isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            value = value[batch]
    while isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[min(seed, len(value) - 1)]
    return int(value)


def _trim(path: Any, end: int | None) -> Any:
    if end is None or end <= 0:
        return path
    shape = _shape(path)
    if len(shape) < 2 or end >= shape[-2] or any(size != 1 for size in shape[:-2]):
        return path
    for _ in shape[:-2]:
        path = path[0]
    trimmed = path[:end]
    return trimmed if hasattr(trimmed, "shape") else list(trimmed)


def _joint_names(path: Any) -> tuple[str, ...]:
    return tuple(str(name) for name in path.joint_names)


def _seed_index(success: Any, batch: int | None = None) -> int:
    success = _plain_value(success)
    if batch is not None and isinstance(success, (list, tuple)):
        success = success[batch]
    if isinstance(success, (list, tuple)):
        return next((i for i, ok in enumerate(success) if _plain_bool(ok)), 0)
    return 0


def _trajectory(raw: Any, batch: int | None = None) -> JointTrajectory | None:
    path = raw.interpolated_trajectory
    if path is None:
        return None
    positions = path.position
    seed = _seed_index(raw.success, batch)
    shape = _shape(positions)
    if batch is not None and len(shape) >= 3:
        positions = positions[batch]
    if len(_shape(positions)) >= 3:
        positions = positions[seed]
    if hasattr(positions, "detach"):
        positions = positions.detach()
    else:
        positions = _plain_value(positions)
    return JointTrajectory(
        _trim(positions, _last_tstep(raw, seed, batch)),
        joint_names=_joint_names(path),
    )


def _metrics(raw: Any, success: tuple[bool, ...]) -> dict[str, Any]:
    native_metrics = getattr(raw, "metrics", None)
    if isinstance(native_metrics, Mapping):
        values = {
            str(key): _plain_value(value)
            for key, value in native_metrics.items()
        }
    else:
        values = {}
        for name in (
            "position_error",
            "rotation_error",
            "cspace_error",
            "goalset_index",
            "solve_time",
            "total_time",
            "seed_rank",
            "seed_cost",
            "feasible",
        ):
            value = getattr(raw, name, None)
            if value is not None:
                values[name] = _plain_value(value)
    values.update(success_count=sum(success), candidate_count=len(success))
    return values


class PlannerRuntime:
    """Own CuRobo planners and their complete mutable scene state."""

    def __init__(
        self,
        profile: PlannerRuntimeProfile | None = None,
        *, planner: Any = None, batch_planner: Any = None,
        planner_factory: Callable[[PlannerRuntimeProfile, PlannerKind], Any] | Any = None,
        batch_planner_factory: Callable[[PlannerRuntimeProfile, PlannerKind], Any] | Any = None,
        world: Any = None, scene_revision: int = 0, name: str | None = None,
    ) -> None:
        if profile is not None and not isinstance(profile, PlannerRuntimeProfile):
            raise TypeError("PlannerRuntime requires PlannerRuntimeProfile")
        self.profile = profile or PlannerRuntimeProfile()
        self.name = str(name or self.profile.name)
        self._planner, self._batch_planner = planner, batch_planner
        self._planner_factory = planner_factory or self.profile.planner_factory
        self._batch_planner_factory = batch_planner_factory or self.profile.batch_planner_factory
        self._world, self._world_set = world, world is not None
        self._obstacle_poses: dict[str, Any] = {}
        self._obstacle_enabled: dict[str, bool] = {}
        self._scene_revision = int(scene_revision)
        self._status = PlannerStatus.READY if planner is not None else PlannerStatus.NEW
        self._last_error: Exception | None = None
        self._planning_count = self._world_update_count = 0
        self._destroyed = False
        self._warmup_kinds: set[PlannerKind] = set()

    @property
    def status(self) -> PlannerStatus: return self._status
    @property
    def planner(self) -> Any: return self._planner
    @property
    def batch_planner(self) -> Any: return self._batch_planner
    @property
    def native_planner(self) -> Any: return self.ensure_planner()
    @property
    def scene_revision(self) -> int: return self._scene_revision
    @property
    def world_revision(self) -> int: return self._scene_revision
    @property
    def world(self) -> Any: return self._world
    @property
    def world_set(self) -> bool: return self._world_set
    @property
    def is_destroyed(self) -> bool: return self._destroyed
    def snapshot(self) -> PlannerStatusSnapshot:
        return PlannerStatusSnapshot(
            status=self._status, planner_ready=self._planner is not None,
            batch_ready=self._batch_planner is not None, scene_revision=self._scene_revision,
            planning_count=self._planning_count, world_update_count=self._world_update_count,
            last_error=None if self._last_error is None else str(self._last_error),
        )

    def _check_alive(self) -> None:
        if self._destroyed:
            raise PlannerDestroyedError(f"planner runtime {self.name!r} has been destroyed")

    def _construct(self, factory: Any, kind: PlannerKind) -> Any:
        if factory is None:
            raise PlannerFactoryError(f"no {kind.value} planner configured for {self.name!r}")
        value = factory(self.profile, kind) if callable(factory) else factory
        if value is None:
            raise PlannerFactoryError(f"{kind.value} planner factory returned None")
        return value

    def _update_native_world(self, planner: Any) -> None:
        updater = getattr(planner, "update_world", None)
        if not callable(updater):
            raise PlannerRuntimeError(f"{type(planner).__name__} does not expose update_world")
        updater(self._world)

    def _replay_poses(self, planner: Any) -> None:
        if not self._obstacle_poses and not self._obstacle_enabled:
            return
        checker = planner.scene_collision_checker
        for name, pose in self._obstacle_poses.items():
            checker.update_obstacle_pose(name, pose)
        for name, enabled in self._obstacle_enabled.items():
            checker.enable_obstacle(name, enabled)

    def _warmup(self, planner: Any, kind: PlannerKind) -> None:
        if kind in self._warmup_kinds:
            return
        warmup = getattr(planner, "warmup", None)
        if callable(warmup) and self.profile.warmup_config:
            warmup(**dict(self.profile.warmup_config))
        self._warmup_kinds.add(kind)

    def ensure_planner(self) -> Any:
        self._check_alive()
        if self._planner is None:
            self._planner = self._construct(self._planner_factory, PlannerKind.SINGLE)
            if self._world_set:
                self._update_native_world(self._planner)
                self._replay_poses(self._planner)
                self._warmup(self._planner, PlannerKind.SINGLE)
            self._status = PlannerStatus.READY
        return self._planner

    def ensure_batch_planner(self) -> Any:
        self._check_alive()
        if not self.profile.batch_enabled:
            raise PlannerRuntimeError("batch planning is disabled")
        if self._batch_planner is None:
            self._batch_planner = self._construct(self._batch_planner_factory, PlannerKind.BATCH)
            if self._world_set:
                self._update_native_world(self._batch_planner)
                self._replay_poses(self._batch_planner)
                self._warmup(self._batch_planner, PlannerKind.BATCH)
        return self._batch_planner

    def adopt_scene_revision(self, revision: int) -> int:
        self._check_alive()
        revision = int(revision)
        if revision < self._scene_revision:
            raise StaleSceneError(f"scene revision moved backwards: {revision} < {self._scene_revision}")
        self._scene_revision = revision
        return revision

    def update_world(self, world: Any, *, revision: int | None = None, force: bool = False) -> int:
        del force
        self._check_alive()
        if revision is not None and int(revision) < self._scene_revision:
            raise StaleSceneError(f"world revision moved backwards for {self.name!r}")
        self._world, self._world_set = world, True
        if revision is not None:
            self._scene_revision = int(revision)
        try:
            was_ready = self._planner is not None
            planner = self.ensure_planner()
            if was_ready:
                self._update_native_world(planner)
                self._replay_poses(planner)
            self._warmup(planner, PlannerKind.SINGLE)
            if self._batch_planner is not None:
                self._update_native_world(self._batch_planner)
                self._replay_poses(self._batch_planner)
                self._warmup(self._batch_planner, PlannerKind.BATCH)
            self._world_update_count += 1
            self._status = PlannerStatus.READY
        except PlannerRuntimeError:
            raise
        except Exception as exc:
            self._status, self._last_error = PlannerStatus.FAILED, exc
            raise PlannerCallError(f"world update failed for {self.name!r}") from exc
        return self._scene_revision

    def update_obstacle_pose(self, name: str, pose: Any, *, revision: int | None = None) -> None:
        self.update_obstacle_poses({str(name): pose}, revision=revision)

    def update_obstacle_poses(self, poses: Mapping[str, Any], *, revision: int | None = None) -> None:
        self._check_alive()
        if revision is not None and int(revision) < self._scene_revision:
            raise StaleSceneError(f"pose revision moved backwards for {self.name!r}")
        self._obstacle_poses.update(poses)
        if revision is not None:
            self._scene_revision = int(revision)
        for planner in (self._planner, self._batch_planner):
            if planner is not None:
                checker = planner.scene_collision_checker
                for name, pose in poses.items():
                    checker.update_obstacle_pose(str(name), pose)

    @staticmethod
    def _checker(planner: Any) -> Any:
        return planner.scene_collision_checker

    def obstacle_names(self, planner: Any | None = None) -> tuple[str, ...]:
        checker = self._checker(planner or self.ensure_planner())
        return tuple(str(name) for name in checker.get_obstacle_names())

    def has_obstacle(self, name: str, planner: Any | None = None) -> bool:
        checker = self._checker(planner or self.ensure_planner())
        return bool(checker.check_obstacle_exists(str(name)))

    def require_obstacles(self, names: Iterable[str], *, planner: Any | None = None) -> tuple[str, ...]:
        expected = tuple(dict.fromkeys(str(name) for name in names))
        missing = tuple(name for name in expected if not self.has_obstacle(name, planner))
        if missing:
            raise PlannerRuntimeError(f"native CuRobo world is missing exact obstacles: {list(missing)}")
        return expected

    def set_obstacle_enabled(self, name: str, enabled: bool, *, planner: Any | None = None) -> None:
        planners = (planner,) if planner is not None else (
            self.ensure_planner(),
            *(() if self._batch_planner is None else (self._batch_planner,)),
        )
        for native in planners:
            if not self.has_obstacle(name, native):
                raise PlannerRuntimeError(f"native CuRobo world is missing exact obstacle: {name}")
            self._checker(native).enable_obstacle(str(name), bool(enabled))
        self._obstacle_enabled[str(name)] = bool(enabled)

    def get_obstacle_geometry(self, name: str, *, planner: Any | None = None) -> Any:
        planner = planner or self.ensure_planner()
        self.require_obstacles((name,), planner=planner)
        checker = self._checker(planner)
        model = checker.scene_model
        if isinstance(model, list):
            model = model[0]
        obstacle = model.get_obstacle(str(name))
        if obstacle is None:
            raise PlannerRuntimeError(f"native CuRobo obstacle geometry is missing: {name}")
        return obstacle

    def build_attachment_geometry(self, names: Iterable[str], *, pose_resolver=None, device_cfg=None) -> tuple[list[Any], Any]:
        from curobo.types import Pose
        from .native_bridge import Mesh
        import numpy as np

        names = tuple(str(name) for name in names)
        obstacles = [self.get_obstacle_geometry(name) for name in names]
        poses = []
        for name, obstacle in zip(names, obstacles):
            pose = pose_resolver(name) if pose_resolver is not None else obstacle.pose
            if pose is None:
                raise PlannerRuntimeError(f"native obstacle has no pose: {name}")
            poses.append(pose if pose_resolver is not None else Pose.from_list(list(pose), device_cfg=device_cfg or self.profile.device))
        anchor = poses[0]
        inverse = np.linalg.inv(anchor.get_numpy_matrix()[0])
        vertices, faces, offset = [], [], 0
        for obstacle, pose in zip(obstacles, poses):
            mesh = obstacle.get_trimesh_mesh(transform_with_pose=False)
            local = np.asarray(mesh.vertices, dtype=np.float32)
            triangles = np.asarray(mesh.faces, dtype=np.int64)
            matrix = pose.get_numpy_matrix()[0]
            world = (matrix[:3, :3] @ local.T).T + matrix[:3, 3]
            vertices.append((inverse[:3, :3] @ world.T).T + inverse[:3, 3])
            faces.append(triangles + offset)
            offset += len(local)
        return [Mesh(name="__attached_object__", pose=[0, 0, 0, 1, 0, 0, 0], vertices=np.concatenate(vertices), faces=np.concatenate(faces))], anchor

    def _check_request(self, request: Any) -> None:
        if request.world_revision is not None and request.world_revision != self._scene_revision and not request.collision_options.allow_stale_scene:
            raise StaleSceneError(f"request revision {request.world_revision} != live {self._scene_revision}")

    def _normalize(self, raw: Any, request: Any, batch: bool) -> PlanResult | BatchPlanResult:
        if raw is None:
            if batch:
                count = request.candidate_count
                return BatchPlanResult(
                    success=(False,) * count,
                    trajectories=(None,) * count,
                    status="failed",
                    error="native planner returned no result",
                    source="native",
                    metrics={"success_count": 0, "candidate_count": count},
                    request_id=request.request_id,
                    phase_id=request.phase_id,
                    profile=request.profile,
                    collision_policy=request.collision_policy,
                    world_revision=self._scene_revision,
                    candidate_indices=tuple(range(count)),
                )
            return PlanResult(
                success=False,
                status="failed",
                error="native planner returned no result",
                source="native",
                metrics={"success_count": 0, "candidate_count": 1},
                request_id=request.request_id,
                phase_id=request.phase_id,
                profile=request.profile,
                collision_policy=request.collision_policy,
                world_revision=self._scene_revision,
            )
        if batch:
            mask = _plain_success_mask(raw.success, count=request.candidate_count)
            paths = tuple(_trajectory(raw, i) if ok else None for i, ok in enumerate(mask))
            return BatchPlanResult(
                success=mask, trajectories=paths, status=getattr(raw, "status", "ok"),
                error=getattr(raw, "error", None), source="native", metrics=_metrics(raw, mask),
                request_id=request.request_id, phase_id=request.phase_id, profile=request.profile,
                collision_policy=request.collision_policy, world_revision=self._scene_revision,
                candidate_indices=tuple(range(len(mask))),
            )
        success = _plain_bool(raw.success)
        return PlanResult(
            success=success, trajectory=_trajectory(raw),
            status=getattr(raw, "status", "ok"), error=getattr(raw, "error", None), source="native",
            selected_candidate_index=getattr(raw, "selected_candidate_index", None),
            metrics=_metrics(raw, (success,)), request_id=request.request_id,
            phase_id=request.phase_id, profile=request.profile, collision_policy=request.collision_policy,
            world_revision=self._scene_revision,
        )

    def _execute(self, request: Any, planner: Any, operation: str, batch: bool) -> Any:
        self._check_alive()
        self._check_request(request)
        self._status = PlannerStatus.PLANNING
        try:
            native_start = time.perf_counter()
            with _CollisionScope(self, planner, map_collision_policy(request)):
                if isinstance(request, CspacePlanRequest):
                    raw = planner.plan_cspace(
                        request.goal_positions,
                        request.start_state,
                        max_attempts=request.max_attempts,
                        enable_graph_attempt=request.enable_graph_attempt,
                    )
                else:
                    raw = planner.plan_pose(
                        request.goals if batch else request.goal,
                        request.start_state,
                        use_implicit_goal=request.use_implicit_goal,
                        max_attempts=request.max_attempts,
                        **({"success_ratio": request.success_ratio} if batch else {}),
                        enable_graph_attempt=request.enable_graph_attempt,
                    )
            native_time = time.perf_counter() - native_start
            normalize_start = time.perf_counter()
            result = self._normalize(raw, request, batch)
            normalize_time = time.perf_counter() - normalize_start
            if batch:
                LOGGER.info(
                    "[PlannerBatchResultTiming] operation=%s candidates=%d "
                    "native_call=%.3fs result_normalize=%.3fs total=%.3fs",
                    operation,
                    request.candidate_count,
                    native_time,
                    normalize_time,
                    time.perf_counter() - native_start,
                )
            self._planning_count += 1
            self._status, self._last_error = PlannerStatus.READY, None
            return result
        except PlannerRuntimeError:
            self._status = PlannerStatus.FAILED
            raise
        except Exception as exc:
            self._status, self._last_error = PlannerStatus.FAILED, exc
            raise PlannerCallError(f"{operation} failed for {self.name!r}") from exc

    def plan_pose(self, request: PosePlanRequest) -> PlanResult:
        if not isinstance(request, PosePlanRequest):
            raise TypeError("plan_pose requires PosePlanRequest")
        return self._execute(request, self.ensure_planner(), "pose planning", False)

    def plan_pose_batch(self, request: BatchPosePlanRequest) -> BatchPlanResult:
        if not isinstance(request, BatchPosePlanRequest):
            raise TypeError("plan_pose_batch requires BatchPosePlanRequest")
        count = request.candidate_count
        if count is None or count <= 0 or count > self.profile.max_batch_size:
            raise ValueError("batch request size is outside native capacity")
        return self._execute(request, self.ensure_batch_planner(), "pose batch planning", True)

    def plan_cspace(self, request: CspacePlanRequest) -> PlanResult:
        if not isinstance(request, CspacePlanRequest):
            raise TypeError("plan_cspace requires CspacePlanRequest")
        return self._execute(request, self.ensure_planner(), "cspace planning", False)

    def warmup(self, **kwargs: Any) -> Any:
        planner = self.ensure_planner()
        method = getattr(planner, "warmup", None)
        if not callable(method):
            return None
        result = method(**(kwargs or dict(self.profile.warmup_config)))
        self._warmup_kinds.add(PlannerKind.SINGLE)
        return result

    @property
    def warmup_done(self) -> bool:
        return PlannerKind.SINGLE in self._warmup_kinds

    def destroy(self) -> None:
        if self._destroyed:
            return
        for planner in (self._planner, self._batch_planner):
            if planner is not None:
                destroy = getattr(planner, "destroy", None)
                if callable(destroy):
                    destroy()
        self._planner = self._batch_planner = None
        self._destroyed, self._status = True, PlannerStatus.DESTROYED

    def __enter__(self) -> "PlannerRuntime":
        self._check_alive()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.destroy()


__all__ = [
    "NativeCollisionOptions", "PlannerCallError", "PlannerDestroyedError",
    "PlannerFactoryError", "PlannerRuntime", "PlannerRuntimeError",
    "StaleSceneError", "map_collision_policy",
]
