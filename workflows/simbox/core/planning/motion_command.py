"""Structured motion phases shared by Pick, Place and execution safety."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np

from core.planning.domain_types import (
    CollisionOptions,
    CollisionPolicy,
    PlanningProfile,
)


class MotionPhase(str, Enum):
    SYNC_WORLD = "sync_world"
    TRANSIT_PREGRASP = "transit_pregrasp"
    TERMINAL_GRASP_APPROACH = "terminal_grasp_approach"
    GRIPPER_CLOSE = "gripper_close"
    ATTACH = "attach"
    POST_GRASP_LIFT = "post_grasp_lift"
    CARRY_HOME = "carry_home"
    TRANSIT_PREPLACE = "transit_preplace"
    TERMINAL_PLACE_DESCENT = "terminal_place_descent"
    GRIPPER_OPEN = "gripper_open"
    DETACH_AND_SETTLE = "detach_and_settle"
    TERMINAL_RETREAT = "terminal_retreat"
    RESTORE_WORLD = "restore_world"


BOOKKEEPING_PHASES = {
    MotionPhase.SYNC_WORLD,
    MotionPhase.ATTACH,
    MotionPhase.DETACH_AND_SETTLE,
    MotionPhase.RESTORE_WORLD,
}


_PHASE_COLLISION_POLICIES = {
    MotionPhase.SYNC_WORLD: CollisionPolicy.WORLD_TRANSIT,
    MotionPhase.TRANSIT_PREGRASP: CollisionPolicy.WORLD_TRANSIT,
    MotionPhase.TERMINAL_GRASP_APPROACH: CollisionPolicy.TARGET_APPROACH,
    MotionPhase.GRIPPER_CLOSE: CollisionPolicy.TARGET_APPROACH,
    MotionPhase.ATTACH: CollisionPolicy.TARGET_APPROACH,
    MotionPhase.POST_GRASP_LIFT: CollisionPolicy.ATTACHED_CARRY,
    MotionPhase.CARRY_HOME: CollisionPolicy.ATTACHED_CARRY,
    MotionPhase.TRANSIT_PREPLACE: CollisionPolicy.ATTACHED_CARRY,
    MotionPhase.TERMINAL_PLACE_DESCENT: CollisionPolicy.PLACEMENT_DESCENT,
    MotionPhase.GRIPPER_OPEN: CollisionPolicy.PLACEMENT_DESCENT,
    MotionPhase.DETACH_AND_SETTLE: CollisionPolicy.PLACEMENT_DESCENT,
    MotionPhase.TERMINAL_RETREAT: CollisionPolicy.RETREAT,
    MotionPhase.RESTORE_WORLD: CollisionPolicy.WORLD_TRANSIT,
}

_PHASE_PROFILES = {
    MotionPhase.TERMINAL_GRASP_APPROACH: PlanningProfile.TERMINAL_LINEAR,
    MotionPhase.TERMINAL_PLACE_DESCENT: PlanningProfile.TERMINAL_LINEAR,
    MotionPhase.POST_GRASP_LIFT: PlanningProfile.ATTACHED_CARRY,
    MotionPhase.CARRY_HOME: PlanningProfile.CSPACE,
    MotionPhase.TRANSIT_PREPLACE: PlanningProfile.ATTACHED_CARRY,
    MotionPhase.TERMINAL_RETREAT: PlanningProfile.TERMINAL_LINEAR,
}


@dataclass
class MotionPhaseCommand:
    """One semantically named phase of a manipulation Skill.

    Positions and orientations are expressed in the owning controller's arm
    base frame, matching the existing CuRobo controller contract.
    """

    phase: MotionPhase
    target_position: np.ndarray | None = None
    target_orientation: np.ndarray | None = None
    gripper_action: str | None = None
    active_object: str | None = None
    support_object: str | None = None
    allow_target_finger_contact: bool = False
    # An attached or just-detached target may retain a short-lived contact
    # with a non-finger robot link while it is being lifted or settled.  This
    # flag is kept separate from finger contact so other environment
    # collisions remain safety violations.
    allow_target_robot_contact: bool = False
    allow_object_support_contact: bool = False
    # Pick may opt into a larger, candidate-scoped safety budget.  ``None``
    # keeps the ordinary ExecutionSupervisor budget unchanged.
    candidate_replan_limit: int | None = None
    completion_tolerance: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    dwell_steps: int = 0
    plan_id: str | None = None
    # Canonical planner/execution metadata.  ``active_object`` and
    # ``support_object`` remain the workflow-facing spellings; the canonical
    phase_id: str | None = None
    completion_policy: Any = "default"
    replan_policy: Any = None
    collision_policy: CollisionPolicy | str | None = None
    collision_options: CollisionOptions | None = None
    profile: PlanningProfile | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    replan_allowed: InitVar[bool] = True
    # A candidate planner may already have produced a named trajectory.  Keep
    # it on the typed command instead of hiding a planner input in ``params``.
    preplanned_joint_path: Any = None
    # A joint target is a planner request, not an articulation action.  The
    # latter is reserved for explicit measured-state hold commands.
    joint_target: np.ndarray | None = None
    # An explicit articulation action is execution-only. It must never be
    # interpreted as a planner goal or smuggled through ``params``.
    direct_joint_action: np.ndarray | None = None
    # Direct callers may preserve the numeric gripper contract instead of
    # translating it to an open/close verb.
    gripper_state: float | None = None

    def __post_init__(self, replan_allowed: bool) -> None:
        if not isinstance(self.phase, MotionPhase):
            self.phase = MotionPhase(self.phase)
        if self.phase_id is None:
            self.phase_id = self.phase.value
        else:
            self.phase_id = str(self.phase_id)
        if self.replan_policy is None:
            self.replan_policy = "allowed" if replan_allowed else "forbidden"
        if self.candidate_replan_limit is not None:
            self.candidate_replan_limit = int(self.candidate_replan_limit)
            if self.candidate_replan_limit < 0:
                raise ValueError("candidate_replan_limit must be non-negative")

        if self.active_object is not None:
            self.active_object = str(self.active_object)
        if self.support_object is not None:
            self.support_object = str(self.support_object)

        requested_collision_policy = self.collision_policy
        if self.collision_policy is None:
            default_policy = _PHASE_COLLISION_POLICIES.get(
                self.phase, CollisionPolicy.WORLD_TRANSIT
            )
            if (
                self.active_object is None
                and default_policy == CollisionPolicy.ATTACHED_CARRY
            ):
                default_policy = CollisionPolicy.WORLD_TRANSIT
            self.collision_policy = default_policy
        elif not isinstance(self.collision_policy, CollisionPolicy):
            self.collision_policy = CollisionPolicy(str(self.collision_policy).lower())
        if self.profile is None:
            self.profile = _PHASE_PROFILES.get(self.phase, PlanningProfile.TRANSIT)
        elif not isinstance(self.profile, PlanningProfile):
            self.profile = PlanningProfile(str(self.profile).lower())
        # Contact declarations are part of the typed collision contract.  A
        # CuRobo v2 planner does not understand arbitrary command kwargs, so
        # preserve them in ``CollisionOptions`` for the PlannerRuntime/native
        # adapter to map to exact scene/attachment operations.
        if self.collision_options is not None and not isinstance(self.collision_options, CollisionOptions):
            raise TypeError("MotionPhaseCommand.collision_options must be CollisionOptions")
        base = self.collision_options or CollisionOptions(self.collision_policy)
        self.collision_options = CollisionOptions(
            policy=self.collision_policy,
            mode=base.mode,
            excluded_obstacles=base.excluded_obstacles,
            included_obstacles=base.included_obstacles,
            allow_self_collision=base.allow_self_collision,
            allow_target_contact=base.allow_target_contact or self.allow_target_finger_contact or self.allow_target_robot_contact,
            allow_support_contact=base.allow_support_contact or self.allow_object_support_contact,
            require_attached_spheres=base.require_attached_spheres or self.collision_policy in {
                CollisionPolicy.ATTACHED_CARRY,
                CollisionPolicy.PLACEMENT_DESCENT,
            },
            target_obstacles=base.target_obstacles,
            support_obstacles=base.support_obstacles,
            attached_obstacles=base.attached_obstacles,
            allow_stale_scene=base.allow_stale_scene,
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("MotionPhaseCommand metadata must be a mapping")
        self.metadata = dict(self.metadata)

        if self.gripper_state is not None:
            self.gripper_state = float(self.gripper_state)
            if not np.isfinite(self.gripper_state) or self.gripper_state not in {
                -1.0,
                1.0,
            }:
                raise ValueError("gripper_state must be exactly -1.0 or 1.0")
            if self.gripper_action is not None:
                raise ValueError("use gripper_action or gripper_state, not both")

        if self.joint_target is not None:
            self.joint_target = np.asarray(self.joint_target, dtype=float).reshape(-1)
            if self.joint_target.size == 0 or not np.all(np.isfinite(self.joint_target)):
                raise ValueError("joint_target must contain finite joint positions")
        if self.direct_joint_action is not None:
            if requested_collision_policy is not None and self.collision_policy is not CollisionPolicy.PASSTHROUGH:
                raise ValueError(
                    "direct_joint_action requires CollisionPolicy.PASSTHROUGH"
                )
            if self.joint_target is not None:
                raise ValueError("direct_joint_action cannot be combined with joint_target")
            if self.target_position is not None or self.target_orientation is not None:
                raise ValueError(
                    "direct_joint_action cannot be combined with an EE target"
                )
            self.direct_joint_action = np.asarray(
                self.direct_joint_action, dtype=float
            ).reshape(-1)
            if self.direct_joint_action.size == 0 or not np.all(
                np.isfinite(self.direct_joint_action)
            ):
                raise ValueError("direct_joint_action must contain finite joint positions")
            # Direct actions are never planner requests. This assignment is
            # intentionally automatic so every producer gets the same policy.
            self.collision_policy = CollisionPolicy.PASSTHROUGH
            self.collision_options = CollisionOptions(CollisionPolicy.PASSTHROUGH)
        elif self.gripper_state is not None:
            raise ValueError("gripper_state is only valid with direct_joint_action")
        if self.target_position is not None:
            self.target_position = np.asarray(self.target_position, dtype=float)
            if self.target_position.shape != (3,):
                raise ValueError("target_position must have shape (3,)")
        if self.target_orientation is not None:
            self.target_orientation = np.asarray(self.target_orientation, dtype=float)
            if self.target_orientation.shape != (4,):
                raise ValueError("target_orientation must have shape (4,)")
        if (self.target_position is None) != (self.target_orientation is None):
            raise ValueError("target_position and target_orientation must be provided together")
        if self.dwell_steps < 0:
            raise ValueError("dwell_steps must be non-negative")

    @property
    def target_joint_positions(self) -> np.ndarray | None:
        """Typed c-space target consumed by the controller runtime."""

        return self.joint_target

    @property
    def is_direct(self) -> bool:
        """Whether this command is an execution-only articulation action."""

        return self.direct_joint_action is not None

    @property
    def is_bookkeeping(self) -> bool:
        return self.phase in BOOKKEEPING_PHASES

    @property
    def is_terminal(self) -> bool:
        return self.phase in {
            MotionPhase.TERMINAL_GRASP_APPROACH,
            MotionPhase.TERMINAL_PLACE_DESCENT,
            MotionPhase.TERMINAL_RETREAT,
        }

    @property
    def translation_tolerance(self) -> float:
        return float(self.completion_tolerance.get("position_m", 0.01))

    @property
    def orientation_tolerance(self) -> float:
        return float(self.completion_tolerance.get("orientation_rad", 0.05))

    @property
    def planning_epsilon(self) -> float:
        """Target-change epsilon, deliberately independent of completion tolerance."""

        return 1e-4

    @property
    def replan_allowed(self) -> bool:
        return str(self.replan_policy).lower() != "forbidden"
