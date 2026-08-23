"""Small shared state records used by the CuRobo controller.

The controller used to assemble setup and execution objects through a generic
port/state pair.  That indirection made ownership hard to follow and
encouraged controller-shaped aliases on the components.  Only the two
concrete owners need shared mutable state, so this module now contains the
records themselves and one private property helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MutableExecutionState:
    """Single mutable owner for phase lifecycle and command bookkeeping."""

    active_phase_command: Any = None
    last_command_name: str = "unknown"
    phase_base_position: Any = None
    phase_base_orientation: Any = None
    phase_bookkeeping_done: bool = False
    phase_dwell_count: int = 0
    phase_plan_started: bool = False
    phase_plan_finished: bool = False
    phase_tracking_failed: bool = False
    phase_plan_failed: bool = False
    phase_completion_logged: bool = False
    step_idx: int = 0
    num_last_cmd: int = 0
    num_plan_failed: int = 0
    last_arm_action: Any = None
    last_commanded_arm_position: Any = None
    idx_list: Any = None
    ee_trans: Any = 0.0
    ee_ori: Any = 0.0
    gripper_state: float = 1.0
    gripper_joint_position: Any = None

    def reset(self) -> None:
        """Reset phase/trajectory state while preserving robot configuration."""

        self.active_phase_command = None
        self.last_command_name = "unknown"
        self.phase_base_position = None
        self.phase_base_orientation = None
        self.phase_bookkeeping_done = False
        self.phase_dwell_count = 0
        self.phase_plan_started = False
        self.phase_plan_finished = False
        self.phase_tracking_failed = False
        self.phase_plan_failed = False
        self.phase_completion_logged = False
        self.step_idx = 0
        self.num_last_cmd = 0
        self.num_plan_failed = 0
        self.last_arm_action = None
        self.last_commanded_arm_position = None
        self.idx_list = None


@dataclass
class PlanningConfig:
    """Setup-time planning configuration shared by setup and execution."""

    ds_ratio: int = 1


def _state_property(field: str):
    """Build a private owner property backed by shared execution state."""

    def getter(owner):
        return getattr(owner.execution_state, field)

    def setter(owner, value):
        setattr(owner.execution_state, field, value)

    return property(getter, setter)


__all__ = ["MutableExecutionState", "PlanningConfig"]
