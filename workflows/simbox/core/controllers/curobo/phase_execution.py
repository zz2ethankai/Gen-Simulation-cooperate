from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from core.planning.domain_types import CommandStatus, JointTrajectory
@dataclass(frozen=True)
class ExecutionStatus:
    status: CommandStatus = CommandStatus.IDLE
    phase: str | None = None
    complete: bool = False
    plan_active: bool = False
    plan_failed: bool = False
    tracking_failed: bool = False
    plan_steps_remaining: int = 0
    reason: str | None = None
    plan_id: str | None = None
    replan_allowed: bool = True
    active: bool = False
    last_commanded_arm_position: Any = None
    phase_base_position: Any = None
    phase_base_orientation: Any = None
    gripper_state: float = 1.0
class PhaseExecutor:
    def __init__(self) -> None:
        self._current: JointTrajectory | None = None
        self._index = 0
    @property
    def current(self) -> Any | None:
        return self._current
    @property
    def index(self) -> int:
        return int(self._index)
    @property
    def active(self) -> bool:
        return self._current is not None
    @property
    def remaining(self) -> int:
        if self._current is None:
            return 0
        return max(0, len(self._current) - int(self._index))
    def install(self, path: JointTrajectory) -> JointTrajectory:
        if not isinstance(path, JointTrajectory):
            raise TypeError(
                "phase executor requires JointTrajectory, "
                f"got {type(path).__name__}"
            )
        if len(path) == 0:
            raise ValueError("phase executor cannot install an empty path")
        self._current = path
        self._index = 0
        return path
    def advance(self, step: int = 1) -> Any | None:
        if step < 0:
            raise ValueError("phase executor step must be non-negative")
        if self._current is None:
            return None
        self._index = min(len(self._current), self._index + int(step))
        if self._index >= len(self._current):
            return None
        return self._current[self._index]
    def status(self) -> ExecutionStatus:
        active = self.active
        return ExecutionStatus(
            status=CommandStatus.ACTIVE if active else CommandStatus.IDLE,
            plan_active=active,
            plan_steps_remaining=self.remaining,
        )
    def clear(self) -> None:
        self._current = None
        self._index = 0
__all__ = ["ExecutionStatus", "PhaseExecutor"]
