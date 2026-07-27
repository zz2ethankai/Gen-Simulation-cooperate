"""Structured motion phases shared by Pick, Place and execution safety."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MotionPhase(str, Enum):
    SYNC_WORLD = "sync_world"
    TRANSIT_PREGRASP = "transit_pregrasp"
    TERMINAL_GRASP_APPROACH = "terminal_grasp_approach"
    GRIPPER_CLOSE = "gripper_close"
    ATTACH = "attach"
    POST_GRASP_LIFT = "post_grasp_lift"
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
    allow_object_support_contact: bool = False
    replan_allowed: bool = True
    completion_tolerance: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    dwell_steps: int = 0
    plan_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, MotionPhase):
            self.phase = MotionPhase(self.phase)
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
