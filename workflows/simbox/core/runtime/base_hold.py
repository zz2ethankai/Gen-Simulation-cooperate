"""Explicit, simulator-independent mobile-base hold strategy.

The manipulation controller owns arm/gripper joint indices only.  A mobile
base hold is a separate runtime concern: it resolves its own named joints,
captures their measured positions, and reapplies position/zero-velocity
targets while manipulation is active.  Keeping the strategy behind a small
port makes the policy testable without importing Isaac Sim and prevents base
DOFs from leaking into an arm action.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

import numpy as np


class BaseHoldPort(Protocol):
    """Operations needed by :class:`BaseHoldStrategy` at the Isaac boundary."""

    def resolve_joint_indices(self, joint_names: Sequence[str]) -> Sequence[int]:
        """Resolve the configured base joint names in runtime DOF order."""

    def read_joint_positions(self, indices: Sequence[int]) -> Sequence[float]:
        """Read the measured positions for ``indices``."""

    def get_drive_state(
        self, indices: Sequence[int]
    ) -> tuple[Sequence[float], Sequence[float], Sequence[float]]:
        """Read ``(kp, kd, max_effort)`` for ``indices``."""

    def set_drive_state(
        self,
        indices: Sequence[int],
        kps: Sequence[float],
        kds: Sequence[float],
        max_efforts: Sequence[float],
    ) -> None:
        """Set ``(kp, kd, max_effort)`` for ``indices``."""

    def set_position_targets(
        self, indices: Sequence[int], positions: Sequence[float]
    ) -> None:
        """Set position targets for ``indices``."""

    def set_velocity_targets(
        self, indices: Sequence[int], velocities: Sequence[float]
    ) -> None:
        """Set velocity targets for ``indices``."""


def _finite_positive(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"base hold {field_name} must be finite and positive") from exc
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"base hold {field_name} must be finite and positive")
    return result


def _joint_names(values: Sequence[Any] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("base hold joint_names must be a sequence of names")
    names = tuple(str(value) for value in values)
    if any(not value for value in names):
        raise ValueError("base hold joint_names must not contain empty names")
    if len(set(names)) != len(names):
        raise ValueError("base hold joint_names must not contain duplicates")
    return names


@dataclass(frozen=True)
class BaseHoldConfig:
    """Validated configuration for one explicitly named base hold group."""

    enabled: bool = False
    joint_names: tuple[str, ...] = ()
    stiffness: float = 100000.0
    damping: float = 3000.0
    max_effort: float = 10000.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "BaseHoldConfig":
        if values is None:
            return cls()
        if isinstance(values, cls):
            return values
        if not isinstance(values, Mapping):
            raise ValueError("base hold configuration must be a mapping")
        enabled = bool(values.get("enabled", False))
        names = _joint_names(values.get("joint_names", ()))
        if enabled and not names:
            raise ValueError("enabled base hold requires joint_names")
        return cls(
            enabled=enabled,
            joint_names=names,
            stiffness=_finite_positive(values.get("stiffness", cls.stiffness), "stiffness"),
            damping=_finite_positive(values.get("damping", cls.damping), "damping"),
            max_effort=_finite_positive(values.get("max_effort", cls.max_effort), "max_effort"),
        )


class BaseHoldStrategy:
    """Capture and continuously reapply an explicitly configured base hold.

    ``BaseHoldStrategy`` never constructs or edits an arm action.  The port
    owns all simulator-specific articulation APIs, while this class owns only
    the hold lifecycle and measured target state:

    * :meth:`enable` captures the current base joint positions and saves the
      navigation drive state;
    * :meth:`suspend` restores the saved navigation drives for navigation;
    * :meth:`resume` captures the post-navigation positions before restoring
      the hold drives; and
    * :meth:`reapply` writes position and zero-velocity targets.  It is safe
      to call before every manipulation action and every physics step.
    """

    def __init__(
        self,
        config: BaseHoldConfig | Mapping[str, Any] | None = None,
        port: BaseHoldPort | None = None,
    ) -> None:
        if port is None:
            raise TypeError("BaseHoldStrategy requires an explicit BaseHoldPort")
        self.config = BaseHoldConfig.from_mapping(config)
        self.port = port
        self._active = False
        self._indices = np.asarray([], dtype=np.int64)
        self._target_positions = np.asarray([], dtype=float)
        self._saved_drive_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def active(self) -> bool:
        return self._active

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self._indices.tolist())

    @property
    def target_positions(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self._target_positions.tolist())

    @property
    def saved_drive_state(self):
        """Return a copy of the original ``(kp, kd, max_effort)`` state."""

        if self._saved_drive_state is None:
            return None
        return tuple(values.copy() for values in self._saved_drive_state)

    @staticmethod
    def _vector(values: Sequence[float], *, field_name: str, size: int) -> np.ndarray:
        result = np.asarray(values, dtype=float).reshape(-1)
        if result.size != size:
            raise ValueError(
                f"base hold {field_name} count {result.size} does not match joint count {size}"
            )
        if not np.all(np.isfinite(result)):
            raise ValueError(f"base hold {field_name} must be finite")
        return result.copy()

    def _resolve_indices(self) -> np.ndarray:
        raw = self.port.resolve_joint_indices(self.config.joint_names)
        indices = np.asarray(raw, dtype=np.int64).reshape(-1)
        if indices.size != len(self.config.joint_names):
            raise ValueError(
                "base hold resolved joint count does not match configured joint_names"
            )
        if len(set(int(index) for index in indices.tolist())) != indices.size:
            raise ValueError("base hold resolved joint indices must be unique")
        if np.any(indices < 0):
            raise ValueError("base hold resolved joint indices must be non-negative")
        return indices

    def _save_drive_state(self, indices: np.ndarray) -> None:
        kps, kds, max_efforts = self.port.get_drive_state(indices)
        size = int(indices.size)
        self._saved_drive_state = (
            self._vector(kps, field_name="saved stiffness", size=size),
            self._vector(kds, field_name="saved damping", size=size),
            self._vector(max_efforts, field_name="saved max_effort", size=size),
        )

    def _apply_hold_drive(self, indices: np.ndarray) -> None:
        size = int(indices.size)
        self.port.set_drive_state(
            indices,
            np.full(size, self.config.stiffness, dtype=float),
            np.full(size, self.config.damping, dtype=float),
            np.full(size, self.config.max_effort, dtype=float),
        )

    def enable(self) -> bool:
        """Capture the measured base pose and activate the hold."""

        if not self.enabled:
            return False
        indices = self._resolve_indices()
        if self._saved_drive_state is None or not np.array_equal(indices, self._indices):
            self._save_drive_state(indices)
        positions = self._vector(
            self.port.read_joint_positions(indices),
            field_name="target positions",
            size=int(indices.size),
        )
        self._indices = indices
        self._target_positions = positions
        self._active = True
        self.reapply()
        return True

    def suspend(self) -> bool:
        """Restore the drive state saved at the beginning of manipulation."""

        if not self._active:
            return False
        if self._saved_drive_state is None:
            raise RuntimeError("base hold is active without a saved drive state")
        self.port.set_drive_state(self._indices, *self._saved_drive_state)
        self._active = False
        return True

    def resume(self) -> bool:
        """Capture the current post-navigation pose and reactivate the hold."""

        if not self.enabled:
            return False
        return self.enable()

    def recapture(self) -> bool:
        """Refresh an active target after an explicit robot/reset pose write."""

        if not self._active:
            return False
        self._target_positions = self._vector(
            self.port.read_joint_positions(self._indices),
            field_name="target positions",
            size=int(self._indices.size),
        )
        self.reapply()
        return True

    def reapply(self) -> bool:
        """Reapply hold gains plus position and zero-velocity targets."""

        if not self._active:
            return False
        self._apply_hold_drive(self._indices)
        self.port.set_position_targets(self._indices, self._target_positions)
        self.port.set_velocity_targets(
            self._indices,
            np.zeros_like(self._target_positions, dtype=float),
        )
        return True

    def reset(self) -> None:
        """Forget episode-local state after a world/articulation reset."""

        self._active = False
        self._indices = np.asarray([], dtype=np.int64)
        self._target_positions = np.asarray([], dtype=float)
        self._saved_drive_state = None


__all__ = ["BaseHoldConfig", "BaseHoldPort", "BaseHoldStrategy"]
