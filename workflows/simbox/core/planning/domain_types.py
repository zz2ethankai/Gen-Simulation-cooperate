"""Typed planning values at the CuRobo v2 boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence


def _names(values: Sequence[Any]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("obstacle names must be a sequence")
    return tuple(dict.fromkeys(str(value) for value in values))


class PlanningProfile(str, Enum):
    TRANSIT = "transit"
    TERMINAL_LINEAR = "terminal_linear"
    ATTACHED_CARRY = "attached_carry"
    CSPACE = "cspace"
    DYNAMIC_REPLAN = "dynamic_replan"


class CollisionPolicy(str, Enum):
    WORLD_TRANSIT = "world_transit"
    TARGET_APPROACH = "target_approach"
    ATTACHED_CARRY = "attached_carry"
    PLACEMENT_DESCENT = "placement_descent"
    RETREAT = "retreat"
    PASSTHROUGH = "passthrough"


class CollisionMode(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"


class CommandStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    COMPLETED = "completed"
    PLAN_FAILED = "plan_failed"
    TRACKING_FAILED = "tracking_failed"
    SCENE_FAILED = "scene_failed"
    CANCELLED = "cancelled"


class PlannerKind(str, Enum):
    SINGLE = "single"
    BATCH = "batch"


class PlannerStatus(str, Enum):
    NEW = "new"
    READY = "ready"
    PLANNING = "planning"
    FAILED = "failed"
    DESTROYED = "destroyed"


@dataclass
class CollisionOptions:
    policy: CollisionPolicy
    mode: CollisionMode = CollisionMode.ENABLED
    excluded_obstacles: tuple[str, ...] = ()
    included_obstacles: tuple[str, ...] = ()
    allow_self_collision: bool = False
    allow_target_contact: bool = False
    allow_support_contact: bool = False
    require_attached_spheres: bool = False
    target_obstacles: tuple[str, ...] = ()
    support_obstacles: tuple[str, ...] = ()
    attached_obstacles: tuple[str, ...] = ()
    allow_stale_scene: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.policy, CollisionPolicy):
            raise TypeError("CollisionOptions.policy must be CollisionPolicy")
        if not isinstance(self.mode, CollisionMode):
            raise TypeError("CollisionOptions.mode must be CollisionMode")
        self.excluded_obstacles = _names(self.excluded_obstacles)
        self.included_obstacles = _names(self.included_obstacles)
        self.target_obstacles = _names(self.target_obstacles)
        self.support_obstacles = _names(self.support_obstacles)
        self.attached_obstacles = _names(self.attached_obstacles)
        overlap = set(self.excluded_obstacles) & set(self.included_obstacles)
        if overlap:
            raise ValueError(f"obstacles cannot be both included and excluded: {sorted(overlap)}")


@dataclass
class PlannerRuntimeProfile:
    name: str = "simbox"
    robot_config: Any = None
    device: Any = None
    max_batch_size: int = 20
    batch_enabled: bool = True
    lazy_batch: bool = True
    planner_factory: Callable[..., Any] | Any | None = None
    batch_planner_factory: Callable[..., Any] | Any | None = None
    warmup_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.max_batch_size = int(self.max_batch_size)
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")


@dataclass
class PosePlanRequest:
    goal: Any
    start_state: Any
    phase_id: str = "phase"
    completion_policy: Any = "default"
    replan_policy: Any = "allowed"
    collision_policy: CollisionPolicy = CollisionPolicy.WORLD_TRANSIT
    collision_options: CollisionOptions | None = None
    active_target: str | None = None
    support: str | None = None
    profile: PlanningProfile = PlanningProfile.TRANSIT
    world_revision: int | None = None
    request_id: str | None = None
    use_implicit_goal: bool = True
    max_attempts: int = 1
    enable_graph_attempt: int = 0

    def __post_init__(self) -> None:
        if self.collision_options is None:
            self.collision_options = CollisionOptions(self.collision_policy)
        if not isinstance(self.collision_options, CollisionOptions):
            raise TypeError("PosePlanRequest.collision_options must be CollisionOptions")


@dataclass
class BatchPosePlanRequest:
    goals: Any
    start_state: Any
    batch_size: int
    phase_id: str = "phase"
    completion_policy: Any = "default"
    replan_policy: Any = "allowed"
    collision_policy: CollisionPolicy = CollisionPolicy.WORLD_TRANSIT
    collision_options: CollisionOptions | None = None
    active_target: str | None = None
    support: str | None = None
    profile: PlanningProfile = PlanningProfile.TRANSIT
    world_revision: int | None = None
    request_id: str | None = None
    use_implicit_goal: bool = True
    max_attempts: int = 1
    success_ratio: float = 0.0
    enable_graph_attempt: int = 0
    start_paths: Any = None

    @property
    def candidate_count(self) -> int:
        return self.batch_size

    def __post_init__(self) -> None:
        self.batch_size = int(self.batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.collision_options is None:
            self.collision_options = CollisionOptions(self.collision_policy)
        if not isinstance(self.collision_options, CollisionOptions):
            raise TypeError("BatchPosePlanRequest.collision_options must be CollisionOptions")


@dataclass
class CspacePlanRequest:
    goal_positions: Any
    start_state: Any
    phase_id: str = "phase"
    completion_policy: Any = "default"
    replan_policy: Any = "allowed"
    collision_policy: CollisionPolicy = CollisionPolicy.WORLD_TRANSIT
    collision_options: CollisionOptions | None = None
    active_target: str | None = None
    support: str | None = None
    profile: PlanningProfile = PlanningProfile.CSPACE
    world_revision: int | None = None
    request_id: str | None = None
    max_attempts: int = 1
    enable_graph_attempt: int = 0

    def __post_init__(self) -> None:
        if self.collision_options is None:
            self.collision_options = CollisionOptions(self.collision_policy)
        if not isinstance(self.collision_options, CollisionOptions):
            raise TypeError("CspacePlanRequest.collision_options must be CollisionOptions")


def _canonical_positions(value: Any) -> Any:
    shape = []
    current = value
    while len(shape) < 5:
        try:
            current_shape = tuple(int(size) for size in current.shape)
        except AttributeError:
            current_shape = ()
        if current_shape:
            shape = list(current_shape)
            break
        if not isinstance(current, (list, tuple)):
            break
        shape.append(len(current))
        current = current[0] if current else []
    if len(shape) <= 2:
        return value
    if len(shape) > 4 or any(size != 1 for size in shape[:-2]):
        raise ValueError(f"trajectory must be [time,dof], got shape={tuple(shape)}")
    for _ in shape[:-2]:
        value = value[0]
    return value


@dataclass(frozen=True)
class JointTrajectory:
    positions: Any
    joint_names: tuple[str, ...] = ()
    velocities: Any = None
    accelerations: Any = None
    jerks: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", _canonical_positions(self.positions))
        object.__setattr__(self, "joint_names", tuple(str(name) for name in self.joint_names))
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("JointTrajectory requires unique joint_names")
        if self.velocities is not None:
            object.__setattr__(self, "velocities", _canonical_positions(self.velocities))
        if self.accelerations is not None:
            object.__setattr__(self, "accelerations", _canonical_positions(self.accelerations))
        if self.jerks is not None:
            object.__setattr__(self, "jerks", _canonical_positions(self.jerks))

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: Any) -> Any:
        return self.positions[index]

    def reorder(self, joint_names: Sequence[Any]) -> "JointTrajectory":
        target = tuple(str(name) for name in joint_names)
        if target == self.joint_names:
            return self
        if not self.joint_names:
            raise ValueError("cannot reorder an unnamed trajectory")
        if set(target) != set(self.joint_names):
            raise ValueError("trajectory joint names do not match controller joints")
        indices = [self.joint_names.index(name) for name in target]

        def reorder_axis(value: Any) -> Any:
            if hasattr(value, "shape") and len(value.shape) >= 1:
                return value[..., indices]
            if not isinstance(value, list):
                return value
            if value and not isinstance(value[0], list):
                return [value[index] for index in indices]
            return [reorder_axis(item) for item in value]

        return JointTrajectory(
            reorder_axis(self.positions), target,
            None if self.velocities is None else reorder_axis(self.velocities),
            None if self.accelerations is None else reorder_axis(self.accelerations),
            None if self.jerks is None else reorder_axis(self.jerks),
        )


@dataclass
class PlanResult:
    success: bool
    trajectory: JointTrajectory | None = None
    status: str = "ok"
    error: str | None = None
    source: str | None = None
    selected_candidate_index: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    phase_id: str | None = None
    profile: PlanningProfile | None = None
    collision_policy: CollisionPolicy | None = None
    world_revision: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("PlanResult.success must be bool")
        if self.trajectory is not None and not isinstance(self.trajectory, JointTrajectory):
            raise TypeError("PlanResult.trajectory must be JointTrajectory")

    def __bool__(self) -> bool:
        return self.success


@dataclass
class BatchPlanResult:
    success: tuple[bool, ...]
    trajectories: tuple[JointTrajectory | None, ...]
    status: str = "ok"
    error: str | None = None
    source: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    phase_id: str | None = None
    profile: PlanningProfile | None = None
    collision_policy: CollisionPolicy | None = None
    world_revision: int | None = None
    candidate_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.success, tuple) or not all(isinstance(item, bool) for item in self.success):
            raise TypeError("BatchPlanResult.success must be tuple[bool, ...]")
        if not isinstance(self.trajectories, tuple) or not all(
            item is None or isinstance(item, JointTrajectory) for item in self.trajectories
        ):
            raise TypeError("BatchPlanResult.trajectories must contain JointTrajectory values")

    def __bool__(self) -> bool:
        return any(self.success)


class AttachmentState(str, Enum):
    DETACHED = "detached"
    ATTACHED = "attached"


@dataclass(frozen=True)
class AttachmentSpec:
    name: str
    state: Any
    meshes: Any
    link_name: str = "attached_object"
    pose_offset: Any = None
    disable_obstacle_names: tuple[str, ...] = ()
    num_spheres: int = 1
    surface_radius: float = 0.001
    sphere_fit_type: Any = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("attachment name must be non-empty")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "link_name", str(self.link_name))
        object.__setattr__(self, "disable_obstacle_names", _names(self.disable_obstacle_names))
        if self.num_spheres <= 0 or self.surface_radius <= 0:
            raise ValueError("attachment sphere settings must be positive")


@dataclass(frozen=True)
class AttachmentResult:
    state: AttachmentState
    spec: AttachmentSpec | None = None

    @property
    def success(self) -> bool:
        return self.state is AttachmentState.ATTACHED


@dataclass(frozen=True)
class PlannerStatusSnapshot:
    status: PlannerStatus
    planner_ready: bool
    batch_ready: bool
    scene_revision: int
    planning_count: int = 0
    world_update_count: int = 0
    last_error: str | None = None


__all__ = [
    "AttachmentResult", "AttachmentSpec", "AttachmentState", "BatchPlanResult",
    "BatchPosePlanRequest", "CollisionMode", "CollisionOptions", "CollisionPolicy",
    "CommandStatus", "CspacePlanRequest", "JointTrajectory", "PlanResult",
    "PlannerKind", "PlannerRuntimeProfile", "PlannerStatus", "PlannerStatusSnapshot",
    "PlanningProfile", "PosePlanRequest",
]
