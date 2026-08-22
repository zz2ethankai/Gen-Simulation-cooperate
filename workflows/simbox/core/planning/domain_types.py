"""Dependency-free SimBox planning domain types.

The public planning boundary is intentionally small and typed.  Native
CuRobo/Isaac objects are converted to plain Python values at the
``PlannerRuntime`` boundary; importing this module does not import Isaac Sim,
CuRobo, Torch, USD or NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping, Sequence


def _tuple_strings(values: Sequence[Any] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise TypeError("expected a sequence of names, not a string")
    return tuple(str(value) for value in values)


def _mapping_copy(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping, got {type(value).__name__}")
    return MappingProxyType(dict(value))


def _plain_value(value: Any) -> Any:
    """Return a native-independent copy of a scalar/container value.

    CuRobo result fields are commonly torch tensors, but importing torch just
    to inspect a result would defeat the dependency-free API.  The small
    protocol below handles tensors and NumPy-like values through their public
    ``detach``/``cpu``/``tolist`` methods and recursively copies ordinary
    mappings and sequences.  Unknown objects are represented by a stable
    string instead of leaking a native object through a result.
    """

    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
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
            return _plain_value(tolist())
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    # A few array-like implementations only expose iteration.  Avoid using
    # it for strings and mappings (handled above), and do not retain the
    # implementation object if conversion fails.
    try:
        if hasattr(value, "__iter__"):
            return [_plain_value(item) for item in value]
    except Exception:
        pass
    return repr(value)


def _canonical_trajectory_value(value: Any) -> Any:
    """Collapse only leading singleton batch/seed axes of a trajectory.

    Native CuRobo paths can arrive as ``[1, 1, T, D]`` even for a single
    request.  The public trajectory contract is ``[T, D]``; retaining those
    axes makes a phase executor count the batch dimension as one waypoint.
    Non-singleton leading axes are deliberately rejected because selecting or
    flattening them would silently mix candidates.
    """

    plain = _plain_value(value)
    shape: list[int] = []
    current = plain
    while len(shape) < 5 and isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    if len(shape) <= 2:
        return plain
    if len(shape) > 4 or any(size != 1 for size in shape[:-2]):
        raise ValueError(
            "trajectory positions contain non-singleton leading dimensions; "
            f"expected [T, D] or leading singleton batch/seed axes, got shape={tuple(shape)}"
        )
    for _ in shape[:-2]:
        plain = plain[0]
    return plain


def _plain_bool(value: Any) -> bool:
    """Convert a scalar or tensor-like success value to one Python bool."""

    value = _plain_value(value)
    if isinstance(value, Mapping):
        return any(_plain_bool(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_plain_bool(item) for item in value)
    try:
        return bool(value)
    except Exception:
        return False


def _plain_success_mask(value: Any, *, count: int | None = None) -> tuple[bool, ...]:
    """Normalize native success arrays to one flag per candidate."""

    value = _plain_value(value)
    if value is None:
        values: tuple[bool, ...] = ()
    elif isinstance(value, Mapping):
        values = tuple(_plain_bool(item) for item in value.values())
    elif isinstance(value, (list, tuple)):
        # Native success is often [batch, seeds].  Collapse each candidate's
        # seed dimension, while preserving a flat [batch] mask.
        values = tuple(
            any(_plain_bool(seed) for seed in item)
            if isinstance(item, (list, tuple))
            else _plain_bool(item)
            for item in value
        )
    else:
        values = (_plain_bool(value),)
    if count is not None and count > 0:
        if not values:
            values = tuple(False for _ in range(count))
        elif len(values) == 1 and count > 1:
            values = values * count
    return tuple(bool(item) for item in values)


class PlanningProfile(str, Enum):
    """Finite planning profile used by requests and execution commands."""

    TRANSIT = "transit"
    TERMINAL_LINEAR = "terminal_linear"
    ATTACHED_CARRY = "attached_carry"
    CSPACE = "cspace"
    DYNAMIC_REPLAN = "dynamic_replan"


class CollisionPolicy(str, Enum):
    """Finite collision policy for a planning phase."""

    WORLD_TRANSIT = "world_transit"
    TARGET_APPROACH = "target_approach"
    ATTACHED_CARRY = "attached_carry"
    PLACEMENT_DESCENT = "placement_descent"
    RETREAT = "retreat"
    PASSTHROUGH = "passthrough"


class CommandStatus(str, Enum):
    """Canonical status shared by all typed execution commands."""

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


class PlannerOperation(str, Enum):
    PLAN = "plan"
    PLAN_BATCH = "plan_batch"
    UPDATE_WORLD = "update_world"
    UPDATE_POSES = "update_poses"
    WARMUP = "warmup"
    ATTACH = "attach"
    DETACH = "detach"
    DESTROY = "destroy"


class CommandType(str, Enum):
    POSE = "pose"
    JOINT = "joint"
    GRIPPER = "gripper"
    SCENE = "scene"
    HOLD = "hold"


class CollisionMode(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass
class CollisionOptions:
    """Private/internal collision options carried alongside a policy enum.

    The public ``CollisionPolicy`` is deliberately finite.  Integrations may
    still need exact obstacle exclusions and stale-scene behavior; those
    implementation details belong here rather than in the policy enum.
    """

    policy: CollisionPolicy = CollisionPolicy.WORLD_TRANSIT
    mode: CollisionMode = CollisionMode.ENABLED
    excluded_obstacles: tuple[str, ...] = ()
    included_obstacles: tuple[str, ...] = ()
    exact_exclusions: Mapping[str, Any] = field(default_factory=dict)
    allow_self_collision: bool = False
    allow_target_contact: bool = False
    allow_support_contact: bool = False
    # The workflow-facing names are retained as typed aliases.  CuRobo v2
    # does not accept arbitrary contact flags on ``plan_pose``; the native
    # adapter consumes these values to enable/disable exact scene obstacles
    # and to validate the attached-object sphere contract.
    allow_target_finger_contact: bool = False
    allow_target_robot_contact: bool = False
    allow_object_support_contact: bool = False
    require_attached_spheres: bool = False
    target_obstacles: tuple[str, ...] = ()
    support_obstacles: tuple[str, ...] = ()
    attached_obstacles: tuple[str, ...] = ()
    allow_stale_scene: bool = False
    strict: bool = True

    @property
    def enabled(self) -> bool:
        return self.mode == CollisionMode.ENABLED

    @property
    def disabled_obstacles(self) -> tuple[str, ...]:
        return self.excluded_obstacles

    @classmethod
    def from_mapping(
        cls,
        value: "CollisionOptions | CollisionPolicy | Mapping[str, Any] | None",
        *,
        default_policy: CollisionPolicy = CollisionPolicy.WORLD_TRANSIT,
    ) -> "CollisionOptions":
        if value is None:
            return cls(policy=default_policy)
        if isinstance(value, cls):
            return value
        if isinstance(value, CollisionPolicy):
            return cls(policy=value)
        raw = dict(value)
        policy = raw.pop("policy", raw.pop("collision_policy", default_policy))
        if not isinstance(policy, CollisionPolicy):
            policy = CollisionPolicy(str(policy).lower())
        mode = raw.pop("mode", raw.pop("collision_mode", CollisionMode.ENABLED))
        if isinstance(mode, bool):
            mode = CollisionMode.ENABLED if mode else CollisionMode.DISABLED
        elif not isinstance(mode, CollisionMode):
            mode = CollisionMode(str(mode).lower())
        excluded = raw.pop(
            "excluded_obstacles",
            raw.pop("exclude", raw.pop("disabled_obstacles", ())),
        )
        included = raw.pop("included_obstacles", raw.pop("include", ()))
        exact = raw.pop("exact_exclusions", raw.pop("exact_exclude", {}))
        target_obstacles = raw.pop(
            "target_obstacles",
            raw.pop("target_collision_names", raw.pop("target_paths", ())),
        )
        support_obstacles = raw.pop(
            "support_obstacles",
            raw.pop("support_collision_names", raw.pop("support_paths", ())),
        )
        attached_obstacles = raw.pop(
            "attached_obstacles",
            raw.pop("attached_collision_names", raw.pop("attached_paths", ())),
        )
        allow_support_contact = bool(raw.pop("allow_support_contact", False))
        allow_object_support_contact = bool(
            raw.pop("allow_object_support_contact", allow_support_contact)
        )
        # ``allow_support_contact`` is the canonical native-boundary spelling;
        # accepting the workflow spelling above keeps MotionPhaseCommand
        # metadata lossless without introducing an untyped params flag.
        allow_support_contact = allow_support_contact or allow_object_support_contact
        return cls(
            policy=policy,
            mode=mode,
            excluded_obstacles=_tuple_strings(excluded),
            included_obstacles=_tuple_strings(included),
            exact_exclusions=dict(exact or {}),
            allow_self_collision=bool(raw.pop("allow_self_collision", False)),
            allow_target_contact=bool(raw.pop("allow_target_contact", False)),
            allow_support_contact=allow_support_contact,
            allow_target_finger_contact=bool(
                raw.pop("allow_target_finger_contact", False)
            ),
            allow_target_robot_contact=bool(
                raw.pop("allow_target_robot_contact", False)
            ),
            allow_object_support_contact=allow_object_support_contact,
            require_attached_spheres=bool(
                raw.pop("require_attached_spheres", False)
            ),
            target_obstacles=_tuple_strings(target_obstacles),
            support_obstacles=_tuple_strings(support_obstacles),
            attached_obstacles=_tuple_strings(attached_obstacles),
            allow_stale_scene=bool(raw.pop("allow_stale_scene", False)),
            strict=bool(raw.pop("strict", True)),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.policy, CollisionPolicy):
            self.policy = CollisionPolicy(str(self.policy).lower())
        if not isinstance(self.mode, CollisionMode):
            self.mode = CollisionMode(str(self.mode).lower())
        self.excluded_obstacles = _tuple_strings(self.excluded_obstacles)
        self.included_obstacles = _tuple_strings(self.included_obstacles)
        self.target_obstacles = _tuple_strings(self.target_obstacles)
        self.support_obstacles = _tuple_strings(self.support_obstacles)
        self.attached_obstacles = _tuple_strings(self.attached_obstacles)
        self.allow_target_contact = bool(
            self.allow_target_contact
            or self.allow_target_finger_contact
            or self.allow_target_robot_contact
        )
        self.allow_support_contact = bool(
            self.allow_support_contact or self.allow_object_support_contact
        )
        self.allow_object_support_contact = bool(
            self.allow_object_support_contact or self.allow_support_contact
        )
        self.exact_exclusions = _mapping_copy(self.exact_exclusions)
        overlap = set(self.excluded_obstacles) & set(self.included_obstacles)
        if overlap:
            raise ValueError(f"obstacles cannot be both included and excluded: {sorted(overlap)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "mode": self.mode.value,
            "excluded_obstacles": list(self.excluded_obstacles),
            "included_obstacles": list(self.included_obstacles),
            "exact_exclusions": dict(self.exact_exclusions),
            "allow_self_collision": self.allow_self_collision,
            "allow_target_contact": self.allow_target_contact,
            "allow_support_contact": self.allow_support_contact,
            "allow_target_finger_contact": self.allow_target_finger_contact,
            "allow_target_robot_contact": self.allow_target_robot_contact,
            "allow_object_support_contact": self.allow_object_support_contact,
            "require_attached_spheres": self.require_attached_spheres,
            "target_obstacles": list(self.target_obstacles),
            "support_obstacles": list(self.support_obstacles),
            "attached_obstacles": list(self.attached_obstacles),
            "allow_stale_scene": self.allow_stale_scene,
            "strict": self.strict,
        }


@dataclass
class PlannerRuntimeProfile:
    """Construction and shape constraints for a PlannerRuntime."""

    name: str = "simbox"
    robot_config: Any = None
    device: Any = None
    max_batch_size: int = 20
    batch_enabled: bool = True
    lazy_batch: bool = True
    use_cuda_graph: bool | None = None
    planner_factory: Callable[..., Any] | Any | None = None
    batch_planner_factory: Callable[..., Any] | Any | None = None
    planner_config: Any = None
    warmup_config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.max_batch_size = int(self.max_batch_size)
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.warmup_config = _mapping_copy(self.warmup_config)
        self.metadata = _mapping_copy(self.metadata)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "PlannerRuntimeProfile":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, PlanningProfile):
            return cls(metadata={"planning_profile": value.value})
        raw = dict(value)
        if "batch_factory" in raw and "batch_planner_factory" not in raw:
            raw["batch_planner_factory"] = raw.pop("batch_factory")
        return cls(**raw)


@dataclass(init=False)
class _PlanRequestBase:
    """Common request metadata; concrete request types are public below."""

    phase_id: str
    completion_policy: Any
    replan_policy: Any
    collision_policy: CollisionPolicy
    collision_options: CollisionOptions
    active_target: str | None
    support: str | None
    profile: PlanningProfile
    preplanned_trajectory: Any
    world_revision: int | None
    request_id: str | None
    kwargs: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def _init_common(
        self,
        *,
        phase_id: str = "phase",
        completion_policy: Any = "default",
        replan_policy: Any = "allowed",
        collision_policy: CollisionPolicy | str = CollisionPolicy.WORLD_TRANSIT,
        collision_options: CollisionOptions | Mapping[str, Any] | None = None,
        active_target: str | None = None,
        active_object: str | None = None,
        support: str | None = None,
        support_object: str | None = None,
        profile: PlanningProfile | str = PlanningProfile.TRANSIT,
        preplanned_trajectory: Any = None,
        preplanned_joint_path: Any = None,
        world_revision: int | None = None,
        request_id: str | None = None,
        kwargs: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self.phase_id = str(phase_id)
        self.completion_policy = completion_policy
        self.replan_policy = replan_policy
        if collision_policy is None:
            collision_policy = CollisionPolicy.WORLD_TRANSIT
        self.collision_policy = (
            collision_policy
            if isinstance(collision_policy, CollisionPolicy)
            else CollisionPolicy(str(collision_policy).lower())
        )
        self.collision_options = CollisionOptions.from_mapping(
            collision_options, default_policy=self.collision_policy
        )
        self.active_target = active_target if active_target is not None else active_object
        self.support = support if support is not None else support_object
        self.profile = profile if isinstance(profile, PlanningProfile) else PlanningProfile(str(profile).lower())
        self.preplanned_trajectory = (
            preplanned_trajectory
            if preplanned_trajectory is not None
            else preplanned_joint_path
        )
        self.world_revision = None if world_revision is None else int(world_revision)
        self.request_id = None if request_id is None else str(request_id)
        merged = dict(kwargs or {})
        merged.update(extra)
        self.kwargs = _mapping_copy(merged)
        self.metadata = _mapping_copy(metadata)

    @property
    def active_object(self) -> str | None:
        return self.active_target

    @property
    def support_object(self) -> str | None:
        return self.support

    @property
    def preplanned_joint_path(self) -> Any:
        return self.preplanned_trajectory

    @property
    def collision_config(self) -> CollisionOptions:
        return self.collision_options


@dataclass(init=False)
class PosePlanRequest(_PlanRequestBase):
    """One native tool-pose planning request."""

    goal: Any
    start_state: Any
    position: Any
    orientation: Any

    def __init__(
        self,
        goal: Any = None,
        start_state: Any = None,
        *,
        target_pose: Any = None,
        pose: Any = None,
        position: Any = None,
        orientation: Any = None,
        state: Any = None,
        start: Any = None,
        **common: Any,
    ) -> None:
        if target_pose is not None:
            goal = target_pose if goal is None else goal
        if pose is not None:
            goal = pose if goal is None else goal
        if state is not None:
            start_state = state if start_state is None else start_state
        if start is not None:
            start_state = start if start_state is None else start_state
        self.goal = goal
        self.start_state = start_state
        self.position = position
        self.orientation = orientation
        self._init_common(**common)

    @property
    def query(self) -> str:
        return "pose"

    @property
    def kind(self) -> PlannerKind:
        return PlannerKind.SINGLE


@dataclass(init=False)
class BatchPosePlanRequest(_PlanRequestBase):
    """Native batch tool-pose request preserving actual candidate count."""

    goals: Any
    start_state: Any
    batch_size: int | None
    start_paths: Any

    def __init__(
        self,
        goals: Any = None,
        start_state: Any = None,
        *,
        goal: Any = None,
        positions: Any = None,
        orientations: Any = None,
        start_states: Any = None,
        start_paths: Any = None,
        batch_size: int | None = None,
        candidate_count: int | None = None,
        state: Any = None,
        **common: Any,
    ) -> None:
        if goals is None:
            goals = goal
        if goals is None and positions is not None:
            goals = (positions, orientations)
        if start_state is None:
            start_state = start_states if start_states is not None else state
        if candidate_count is not None:
            if batch_size is not None and int(batch_size) != int(candidate_count):
                raise ValueError("batch_size and candidate_count disagree")
            batch_size = int(candidate_count)
        self.goals = goals
        self.start_state = start_state
        self.start_paths = start_paths
        self.batch_size = batch_size if batch_size is None else int(batch_size)
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._init_common(**common)

    @property
    def goal(self) -> Any:
        return self.goals

    @property
    def query(self) -> str:
        return "pose"

    @property
    def kind(self) -> PlannerKind:
        return PlannerKind.BATCH

    @property
    def candidate_count(self) -> int | None:
        return self.batch_size


@dataclass(init=False)
class CspacePlanRequest(_PlanRequestBase):
    """One native joint-space planning request."""

    goal_positions: Any
    start_state: Any

    def __init__(
        self,
        goal_positions: Any = None,
        start_state: Any = None,
        *,
        goal: Any = None,
        target_positions: Any = None,
        state: Any = None,
        start: Any = None,
        **common: Any,
    ) -> None:
        if goal_positions is None:
            goal_positions = goal if goal is not None else target_positions
        if start_state is None:
            start_state = state if state is not None else start
        self.goal_positions = goal_positions
        self.start_state = start_state
        self._init_common(profile=common.pop("profile", PlanningProfile.CSPACE), **common)

    @property
    def goal(self) -> Any:
        return self.goal_positions

    @property
    def query(self) -> str:
        return "cspace"

    @property
    def kind(self) -> PlannerKind:
        return PlannerKind.SINGLE


@dataclass(frozen=True, eq=False)
class JointTrajectory:
    """Native-independent joint trajectory.

    ``positions`` is a Python scalar/container (normally ``[T, D]``), never
    a torch tensor or CuRobo ``JointState``.  Keeping ``position`` as a
    read-only alias preserves the small surface used by existing execution
    adapters while making the ownership boundary explicit.
    """

    positions: Any
    joint_names: tuple[str, ...] = ()
    velocities: Any = None
    accelerations: Any = None
    jerks: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "positions",
            _canonical_trajectory_value(self.positions),
        )
        object.__setattr__(self, "joint_names", _tuple_strings(self.joint_names))
        object.__setattr__(self, "velocities", _canonical_trajectory_value(self.velocities))
        object.__setattr__(self, "accelerations", _canonical_trajectory_value(self.accelerations))
        object.__setattr__(self, "jerks", _canonical_trajectory_value(self.jerks))

    @property
    def position(self) -> Any:
        return self.positions

    @classmethod
    def from_native(
        cls,
        value: Any,
        *,
        joint_names: Sequence[Any] | None = None,
    ) -> "JointTrajectory":
        """Copy a native JointState/trajectory or plain path into this type."""

        if isinstance(value, cls):
            if joint_names and not value.joint_names:
                return cls(value.positions, joint_names=joint_names, velocities=value.velocities,
                           accelerations=value.accelerations, jerks=value.jerks)
            return value
        source = value
        if isinstance(source, Mapping):
            positions = source.get("positions", source.get("position", source.get("values")))
            names = source.get("joint_names", joint_names)
            velocities = source.get("velocities", source.get("velocity"))
            accelerations = source.get("accelerations", source.get("acceleration"))
            jerks = source.get("jerks", source.get("jerk"))
        else:
            positions = getattr(source, "positions", None)
            if positions is None:
                positions = getattr(source, "position", None)
            if positions is None:
                positions = getattr(source, "values", None)
            if positions is None:
                positions = source
            names = getattr(source, "joint_names", None) or joint_names
            velocities = getattr(source, "velocities", None)
            if velocities is None:
                velocities = getattr(source, "velocity", None)
            accelerations = getattr(source, "accelerations", None)
            if accelerations is None:
                accelerations = getattr(source, "acceleration", None)
            jerks = getattr(source, "jerks", None)
            if jerks is None:
                jerks = getattr(source, "jerk", None)
        return cls(
            positions=positions,
            joint_names=names or (),
            velocities=velocities,
            accelerations=accelerations,
            jerks=jerks,
        )

    def __len__(self) -> int:
        try:
            return len(self.positions)
        except TypeError:
            return 0

    def __getitem__(self, index: Any) -> Any:
        return self.positions[index]

    def reorder(self, joint_names: Sequence[Any]) -> "JointTrajectory":
        """Return a copy with the final joint axis in ``joint_names`` order."""

        target = _tuple_strings(joint_names)
        if target == self.joint_names:
            return self
        if not self.joint_names:
            raise ValueError("cannot reorder an unnamed trajectory")
        if set(target) - set(self.joint_names):
            missing = sorted(set(target) - set(self.joint_names))
            raise ValueError(f"trajectory is missing joints: {missing}")
        indices = [self.joint_names.index(name) for name in target]

        def reorder_axis(value: Any) -> Any:
            if not isinstance(value, (list, tuple)):
                return value
            if value and not isinstance(value[0], (list, tuple)):
                return [value[index] for index in indices]
            return [reorder_axis(item) for item in value]

        return JointTrajectory(
            positions=reorder_axis(self.positions),
            joint_names=target,
            velocities=None if self.velocities is None else reorder_axis(self.velocities),
            accelerations=None if self.accelerations is None else reorder_axis(self.accelerations),
            jerks=None if self.jerks is None else reorder_axis(self.jerks),
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, JointTrajectory):
            return (
                self.positions == other.positions
                and self.joint_names == other.joint_names
                and self.velocities == other.velocities
                and self.accelerations == other.accelerations
                and self.jerks == other.jerks
            )
        # Keep comparisons useful for tests/adapters that compare a normalized
        # trajectory directly with its copied positions.
        return self.positions == other


def _normalize_trajectory(value: Any, *, joint_names: Sequence[Any] | None = None) -> JointTrajectory | None:
    if value is None:
        return None
    return JointTrajectory.from_native(value, joint_names=joint_names)


@dataclass(init=False)
class PlanResult:
    """Fully normalized result for one pose or cspace query.

    There is deliberately no ``raw`` field.  Native planner results are
    consumed only inside ``PlannerRuntime._wrap_result`` and are represented
    here by scalar status/error/source fields, metrics, and a copied
    :class:`JointTrajectory`.
    """

    success: bool
    trajectory: JointTrajectory | None
    status: str
    error: str | None
    source: str | None
    selected_candidate_index: int | None
    metrics: Mapping[str, Any]
    request_id: str | None
    phase_id: str | None
    profile: PlanningProfile | None
    collision_policy: CollisionPolicy | None
    world_revision: int | None
    candidate_indices: tuple[int, ...]

    def __init__(
        self,
        success: Any = False,
        trajectory: Any = None,
        *,
        status: Any = "ok",
        error: Any = None,
        source: Any = None,
        selected_candidate_index: Any = None,
        metrics: Mapping[str, Any] | None = None,
        request_id: Any = None,
        phase_id: Any = None,
        profile: PlanningProfile | str | None = None,
        collision_policy: CollisionPolicy | str | None = None,
        world_revision: int | None = None,
        candidate_indices: Sequence[int] = (),
        joint_names: Sequence[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **unsupported: Any,
    ) -> None:
        if "raw" in unsupported:
            raise TypeError("PlanResult does not accept native/raw result values")
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"unexpected PlanResult fields: {names}")
        if metrics is None:
            metrics = metadata
        self.success = _plain_bool(success)
        self.trajectory = _normalize_trajectory(trajectory, joint_names=joint_names)
        self.status = _result_text(status, default="ok")
        self.error = _result_error(error)
        self.source = _result_text(source, default=None)
        self.selected_candidate_index = _candidate_index(selected_candidate_index)
        self.metrics = _plain_mapping(metrics)
        self.request_id = _result_text(request_id, default=None)
        self.phase_id = _result_text(phase_id, default=None)
        self.profile = _profile_or_none(profile)
        self.collision_policy = _collision_policy_or_none(collision_policy)
        self.world_revision = None if world_revision is None else int(world_revision)
        self.candidate_indices = tuple(int(index) for index in (candidate_indices or ()))

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Compatibility read alias for normalized ``metrics``."""

        return self.metrics

    @property
    def reason(self) -> str | None:
        """Compatibility read alias for normalized ``error``."""

        return self.error

    @property
    def is_success(self) -> bool:
        return self.success

    @property
    def success_count(self) -> int:
        return int(self.success)

    def __bool__(self) -> bool:
        return self.success


def _result_text(value: Any, *, default: str | None) -> str | None:
    if value is None:
        return default
    value = _plain_value(value)
    if isinstance(value, (list, tuple, dict)):
        return repr(value)
    return str(value)


def _result_error(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, BaseException):
        return str(value) or type(value).__name__
    return _result_text(value, default=None)


def _candidate_index(value: Any) -> int | None:
    if value is None:
        return None
    try:
        value = _plain_value(value)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        return None if value is None else int(value)
    except (TypeError, ValueError, IndexError):
        return None


def _profile_or_none(value: Any) -> PlanningProfile | None:
    if value is None:
        return None
    if isinstance(value, PlanningProfile):
        return value
    return PlanningProfile(str(value).lower())


def _collision_policy_or_none(value: Any) -> CollisionPolicy | None:
    if value is None:
        return None
    if isinstance(value, CollisionPolicy):
        return value
    return CollisionPolicy(str(value).lower())


def _plain_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        return {"value": _plain_value(value)}
    return {str(key): _plain_value(item) for key, item in value.items()}


@dataclass(init=False)
class BatchPlanResult(PlanResult):
    """Fully normalized candidate-batched result.

    ``success`` is always a tuple of Python bools, one per candidate, and
    ``trajectories`` contains copied trajectories (or ``None`` for a failed
    candidate).  ``success_mask`` is an explicit, immutable alias used by
    candidate ranking code.
    """

    success: tuple[bool, ...]
    trajectory: tuple[JointTrajectory | None, ...]
    trajectories: tuple[JointTrajectory | None, ...]

    def __init__(
        self,
        success: Any = (),
        trajectory: Any = None,
        *,
        trajectories: Any = None,
        status: Any = "ok",
        error: Any = None,
        source: Any = None,
        selected_candidate_index: Any = None,
        metrics: Mapping[str, Any] | None = None,
        request_id: Any = None,
        phase_id: Any = None,
        profile: PlanningProfile | str | None = None,
        collision_policy: CollisionPolicy | str | None = None,
        world_revision: int | None = None,
        candidate_indices: Sequence[int] = (),
        joint_names: Sequence[Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        **unsupported: Any,
    ) -> None:
        if "raw" in unsupported:
            raise TypeError("BatchPlanResult does not accept native/raw result values")
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise TypeError(f"unexpected BatchPlanResult fields: {names}")
        paths = trajectories if trajectories is not None else trajectory
        if paths is None:
            normalized_paths: tuple[JointTrajectory | None, ...] = ()
        elif isinstance(paths, (list, tuple)):
            normalized_paths = tuple(
                _normalize_trajectory(path, joint_names=joint_names) for path in paths
            )
        else:
            normalized_paths = (_normalize_trajectory(paths, joint_names=joint_names),)
        mask = _plain_success_mask(success, count=len(normalized_paths) or None)
        if not mask and normalized_paths:
            mask = tuple(path is not None for path in normalized_paths)
        if normalized_paths and len(mask) < len(normalized_paths):
            mask = mask + tuple(False for _ in range(len(normalized_paths) - len(mask)))
        if normalized_paths and len(mask) > len(normalized_paths):
            normalized_paths = normalized_paths + tuple(
                None for _ in range(len(mask) - len(normalized_paths))
            )
        # Populate common fields without allowing PlanResult to coerce the
        # candidate mask into a scalar.
        self.success = tuple(bool(item) for item in mask)
        self.trajectory = normalized_paths
        self.trajectories = normalized_paths
        self.status = _result_text(status, default="ok")
        self.error = _result_error(error)
        self.source = _result_text(source, default=None)
        self.selected_candidate_index = _candidate_index(selected_candidate_index)
        self.metrics = _plain_mapping(metrics if metrics is not None else metadata)
        self.request_id = _result_text(request_id, default=None)
        self.phase_id = _result_text(phase_id, default=None)
        self.profile = _profile_or_none(profile)
        self.collision_policy = _collision_policy_or_none(collision_policy)
        self.world_revision = None if world_revision is None else int(world_revision)
        self.candidate_indices = tuple(int(index) for index in (candidate_indices or ()))

    @property
    def success_mask(self) -> tuple[bool, ...]:
        return self.success

    @property
    def is_success(self) -> bool:
        return any(self.success)

    @property
    def success_count(self) -> int:
        return sum(self.success)

    def __bool__(self) -> bool:
        return self.is_success


@dataclass(init=False)
class _CommandBase:
    """Common command metadata and discriminant."""

    phase_id: str
    completion_policy: Any
    replan_policy: Any
    collision_policy: CollisionPolicy
    active_target: str | None
    support: str | None
    profile: PlanningProfile
    preplanned_trajectory: Any
    metadata: Mapping[str, Any]

    command_type: ClassVar[CommandType] = CommandType.HOLD

    def _init_command(
        self,
        *,
        phase_id: str = "phase",
        completion_policy: Any = "default",
        replan_policy: Any = "allowed",
        collision_policy: CollisionPolicy | str = CollisionPolicy.PASSTHROUGH,
        active_target: str | None = None,
        active_object: str | None = None,
        support: str | None = None,
        support_object: str | None = None,
        profile: PlanningProfile | str = PlanningProfile.TRANSIT,
        preplanned_trajectory: Any = None,
        preplanned_joint_path: Any = None,
        metadata: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        self.phase_id = str(phase_id)
        self.completion_policy = completion_policy
        self.replan_policy = replan_policy
        self.collision_policy = (
            collision_policy
            if isinstance(collision_policy, CollisionPolicy)
            else CollisionPolicy(str(collision_policy).lower())
        )
        self.active_target = active_target if active_target is not None else active_object
        self.support = support if support is not None else support_object
        self.profile = profile if isinstance(profile, PlanningProfile) else PlanningProfile(str(profile).lower())
        self.preplanned_trajectory = (
            preplanned_trajectory
            if preplanned_trajectory is not None
            else preplanned_joint_path
        )
        merged = dict(metadata or {})
        merged.update(extra)
        self.metadata = _mapping_copy(merged)

    @property
    def kind(self) -> CommandType:
        return self.command_type

    @property
    def active_object(self) -> str | None:
        return self.active_target

    @property
    def support_object(self) -> str | None:
        return self.support

    @property
    def preplanned_joint_path(self) -> Any:
        return self.preplanned_trajectory


@dataclass(init=False)
class PoseCommand(_CommandBase):
    position: Any
    orientation: Any
    request: PosePlanRequest | None
    command_type: ClassVar[CommandType] = CommandType.POSE

    def __init__(
        self,
        position: Any = None,
        orientation: Any = None,
        *,
        target_position: Any = None,
        target_orientation: Any = None,
        request: PosePlanRequest | None = None,
        **common: Any,
    ) -> None:
        self.position = position if position is not None else target_position
        self.orientation = orientation if orientation is not None else target_orientation
        self.request = request
        self._init_command(**common)

    @property
    def target_position(self) -> Any:
        return self.position

    @property
    def target_orientation(self) -> Any:
        return self.orientation


@dataclass(init=False)
class JointCommand(_CommandBase):
    joint_positions: Any
    request: CspacePlanRequest | None
    command_type: ClassVar[CommandType] = CommandType.JOINT

    def __init__(
        self,
        joint_positions: Any = None,
        *,
        target_positions: Any = None,
        request: CspacePlanRequest | None = None,
        **common: Any,
    ) -> None:
        self.joint_positions = joint_positions if joint_positions is not None else target_positions
        self.request = request
        self._init_command(profile=common.pop("profile", PlanningProfile.CSPACE), **common)

    @property
    def target_positions(self) -> Any:
        return self.joint_positions


@dataclass(init=False)
class GripperCommand(_CommandBase):
    action: Any
    value: Any
    command_type: ClassVar[CommandType] = CommandType.GRIPPER

    def __init__(self, action: Any = None, value: Any = None, *, gripper_action: Any = None, **common: Any) -> None:
        self.action = action if action is not None else gripper_action
        self.value = value
        self._init_command(**common)

    @property
    def gripper_action(self) -> Any:
        return self.action


@dataclass(init=False)
class SceneCommand(_CommandBase):
    operation: Any
    world: Any
    poses: Mapping[str, Any]
    command_type: ClassVar[CommandType] = CommandType.SCENE

    def __init__(
        self,
        operation: Any = "update_world",
        world: Any = None,
        *,
        dynamic_poses: Mapping[str, Any] | None = None,
        poses: Mapping[str, Any] | None = None,
        **common: Any,
    ) -> None:
        self.operation = operation
        self.world = world
        self.poses = dict(poses or dynamic_poses or {})
        self._init_command(**common)

    @property
    def dynamic_poses(self) -> Mapping[str, Any]:
        return self.poses


@dataclass(init=False)
class HoldCommand(_CommandBase):
    duration_steps: int | None
    command_type: ClassVar[CommandType] = CommandType.HOLD

    def __init__(self, duration_steps: int | None = None, **common: Any) -> None:
        self.duration_steps = None if duration_steps is None else int(duration_steps)
        self._init_command(**common)


@dataclass
class PlannerCommand:
    """Compatibility envelope for non-phase planner operations."""

    operation: PlannerOperation | str
    request: Any = None
    payload: Any = None
    scene_revision: int | None = None
    collision_policy: CollisionPolicy = CollisionPolicy.PASSTHROUGH
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, PlannerOperation):
            self.operation = PlannerOperation(str(self.operation).lower())
        if not isinstance(self.collision_policy, CollisionPolicy):
            self.collision_policy = CollisionPolicy(str(self.collision_policy).lower())
        self.metadata = _mapping_copy(self.metadata)


@dataclass(frozen=True, order=True)
class SceneRevision:
    value: int = 0
    reason: str | None = None

    def __post_init__(self) -> None:
        if int(self.value) < 0:
            raise ValueError("scene revision must be non-negative")
        object.__setattr__(self, "value", int(self.value))

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SceneRevision):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        return NotImplemented


@dataclass(frozen=True)
class SceneUpdate:
    revision: SceneRevision
    world: Any = None
    changed: bool = True
    force: bool = False
    dynamic_poses: Mapping[str, Any] = field(default_factory=dict)
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        revision = self.revision if isinstance(self.revision, SceneRevision) else SceneRevision(self.revision)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "dynamic_poses", _mapping_copy(self.dynamic_poses))
        object.__setattr__(self, "metadata", _mapping_copy(self.metadata))


class AttachmentState(str, Enum):
    DETACHED = "detached"
    ATTACHED = "attached"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"


@dataclass(frozen=True)
class AttachmentSpec:
    name: str
    state: Any = None
    meshes: Any = None
    link_name: str = "attached_object"
    pose_offset: Any = None
    disable_obstacle_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("attachment name must be non-empty")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "link_name", str(self.link_name))
        object.__setattr__(self, "disable_obstacle_names", _tuple_strings(self.disable_obstacle_names))
        object.__setattr__(self, "metadata", _mapping_copy(self.metadata))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "AttachmentSpec", **overrides: Any) -> "AttachmentSpec":
        if isinstance(value, cls):
            data = {
                "name": value.name,
                "state": value.state,
                "meshes": value.meshes,
                "link_name": value.link_name,
                "pose_offset": value.pose_offset,
                "disable_obstacle_names": value.disable_obstacle_names,
                "metadata": dict(value.metadata),
            }
        else:
            data = dict(value)
            if "object_name" in data and "name" not in data:
                data["name"] = data.pop("object_name")
            if "collision_names" in data and "disable_obstacle_names" not in data:
                data["disable_obstacle_names"] = data.pop("collision_names")
            if "world_objects_pose_offset" in data and "pose_offset" not in data:
                data["pose_offset"] = data.pop("world_objects_pose_offset")
        data.update(overrides)
        return cls(**data)


@dataclass
class AttachmentResult:
    state: AttachmentState = AttachmentState.ATTACHED
    spec: AttachmentSpec | None = None
    error: Exception | None = None
    rolled_back: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.state == AttachmentState.ATTACHED and self.error is None


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
    "AttachmentResult",
    "AttachmentSpec",
    "AttachmentState",
    "BatchPlanResult",
    "BatchPosePlanRequest",
    "CollisionMode",
    "CollisionOptions",
    "CollisionPolicy",
    "CommandStatus",
    "CommandType",
    "CspacePlanRequest",
    "GripperCommand",
    "HoldCommand",
    "JointCommand",
    "JointTrajectory",
    "PlanResult",
    "PlannerCommand",
    "PlannerKind",
    "PlannerOperation",
    "PlannerStatus",
    "PlannerStatusSnapshot",
    "PlanningProfile",
    "PoseCommand",
    "PosePlanRequest",
    "SceneCommand",
    "SceneRevision",
    "SceneUpdate",
]
