"""Typed measured-state barrier for the navigation/manipulation boundary.

The local navigation controller owns path planning and command generation.  A
separate, small barrier owns only the transition after the goal is reached:
the measured base pose and twist must remain stable for a consecutive number
of physics ticks before a dependent manipulation skill may run.

This module is intentionally independent of Isaac Sim.  The production
adapter is built from a robot and :class:`LocalBaseDriver`, while unit tests
can provide a three-callback ``NavigationSettlePort`` without constructing a
simulator world.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Protocol

import numpy as np


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _yaw_from_wxyz(orientation: Any) -> float:
    values = np.asarray(orientation, dtype=float).reshape(-1)
    if values.size < 4:
        raise ValueError("base orientation must contain at least four values")
    w, x, y, z = (float(value) for value in values[:4])
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


@dataclass(frozen=True, init=False)
class NavigationBaseState:
    """One measured base pose/twist sample.

    ``position`` is a world-frame XYZ translation, ``yaw`` is world-frame
    heading, and ``twist_body`` is ``(vx, vy, wz)`` in the base frame.  The
    values are copied into immutable tuples so a simulator tensor cannot be
    mutated underneath a dwell calculation.
    """

    position: tuple[float, float, float]
    yaw: float
    twist_body: tuple[float, float, float]

    def __init__(
        self,
        position: Any,
        yaw: float | None = None,
        twist_body: Any = None,
        *,
        orientation: Any | None = None,
    ) -> None:
        translation = np.asarray(position, dtype=float).reshape(-1)
        if translation.size < 2:
            raise ValueError("base position must contain at least x and y")
        if twist_body is None:
            raise ValueError("base twist_body is required")
        twist = np.asarray(twist_body, dtype=float).reshape(-1)
        if twist.size < 3:
            raise ValueError("base twist_body must contain vx, vy, and wz")
        if yaw is None:
            if orientation is None:
                raise ValueError("base yaw or orientation is required")
            yaw = _yaw_from_wxyz(orientation)
        yaw_value = float(yaw)
        position_values = (
            translation[:3]
            if translation.size >= 3
            else np.pad(translation[:2], (0, 1))
        )
        values = np.concatenate([position_values, twist[:3], [yaw_value]])
        if not np.all(np.isfinite(values)):
            raise ValueError("base pose/twist must contain finite values")
        object.__setattr__(self, "position", tuple(float(value) for value in values[:3]))
        object.__setattr__(self, "yaw", _wrap_angle(yaw_value))
        object.__setattr__(self, "twist_body", tuple(float(value) for value in values[3:6]))

    @classmethod
    def from_pose(
        cls,
        translation: Any,
        orientation: Any,
        twist_body: Any,
    ) -> "NavigationBaseState":
        """Construct a sample from the robot's measured pose quaternion."""

        return cls(
            translation,
            _yaw_from_wxyz(orientation),
            twist_body,
        )

    @property
    def xy(self) -> tuple[float, float]:
        return self.position[0], self.position[1]

    @property
    def linear_speed(self) -> float:
        return math.hypot(self.twist_body[0], self.twist_body[1])

    @property
    def angular_speed(self) -> float:
        return abs(self.twist_body[2])


class NavigationSettleQueryPort(Protocol):
    """Measured-state and lifecycle surface consumed by the barrier."""

    def measure_base_state(self) -> NavigationBaseState: ...

    def stop(self) -> None: ...

    def finalize(self) -> None: ...


class NavigationSettlePort:
    """Typed callback port for one navigation settle boundary.

    The port deliberately has no controller or workflow reference.  It is
    composed from the robot/driver at the navigation boundary and is equally
    usable with a deterministic fake in host-side tests.
    """

    def __init__(
        self,
        measure_base_state: Callable[[], NavigationBaseState],
        stop: Callable[[], None] | None = None,
        finalize: Callable[[], None] | None = None,
    ) -> None:
        if not callable(measure_base_state):
            raise TypeError("NavigationSettlePort requires a measured-state callback")
        if stop is not None and not callable(stop):
            raise TypeError("NavigationSettlePort stop callback must be callable")
        if finalize is not None and not callable(finalize):
            raise TypeError("NavigationSettlePort finalize callback must be callable")
        self._measure_base_state = measure_base_state
        self._stop = stop or (lambda: None)
        self._finalize = finalize or (lambda: None)

    @classmethod
    def from_robot_driver(cls, robot: Any, driver: Any) -> "NavigationSettlePort":
        """Compose a port from the existing measured robot/driver surface."""

        pose_getter = getattr(robot, "get_nav_base_pose", None) or getattr(
            robot, "get_mobile_base_pose", None
        )
        if not callable(pose_getter):
            raise TypeError("robot must expose get_nav_base_pose or get_mobile_base_pose")
        twist_getter = getattr(driver, "get_actual_twist_body", None)
        if not callable(twist_getter):
            raise TypeError("local base driver must expose get_actual_twist_body")
        command_setter = getattr(driver, "set_command", None)
        if not callable(command_setter):
            raise TypeError("local base driver must expose set_command")
        finalizer = getattr(driver, "finalize_after_navigation", None)
        if not callable(finalizer):
            raise TypeError("local base driver must expose finalize_after_navigation")

        def measure() -> NavigationBaseState:
            translation, orientation = pose_getter()
            return NavigationBaseState.from_pose(
                translation,
                orientation,
                twist_getter(),
            )

        def stop() -> None:
            command_setter(0.0, 0.0, 0.0)

        return cls(measure, stop, finalizer)

    def measure_base_state(self) -> NavigationBaseState:
        sample = self._measure_base_state()
        if isinstance(sample, NavigationBaseState):
            return sample
        # Keep the adapter friendly to small test doubles while retaining a
        # typed value at the barrier itself.
        if isinstance(sample, Mapping):
            return NavigationBaseState(
                sample["position"],
                sample.get("yaw"),
                sample["twist_body"],
                orientation=sample.get("orientation"),
            )
        raise TypeError(
            "NavigationSettlePort measure callback must return NavigationBaseState"
        )

    def measure(self) -> NavigationBaseState:
        """Compatibility spelling for a measured-state query."""

        return self.measure_base_state()

    def stop(self) -> None:
        self._stop()

    def finalize(self) -> None:
        self._finalize()


class NavigationSettleStatus(str, Enum):
    """Lifecycle result of a measured settle barrier."""

    IDLE = "idle"
    WAITING = "waiting"
    SETTLED = "settled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    # Readable aliases for callers that use result-oriented names.
    SUCCESS = "settled"
    TIMEOUT = "timed_out"
    FAILURE = "failed"


@dataclass(frozen=True)
class NavigationSettleResult:
    """Inspectable status returned after each barrier physics tick."""

    status: NavigationSettleStatus
    steps: int
    stable_steps: int
    elapsed_sec: float
    reason: str = ""
    pose_delta_m: float | None = None
    yaw_delta_rad: float | None = None
    linear_speed_mps: float | None = None
    angular_speed_rad_s: float | None = None

    @property
    def complete(self) -> bool:
        return self.status in {
            NavigationSettleStatus.SETTLED,
            NavigationSettleStatus.TIMED_OUT,
            NavigationSettleStatus.FAILED,
        }

    @property
    def success(self) -> bool:
        return self.status == NavigationSettleStatus.SETTLED

    @property
    def done(self) -> bool:
        return self.complete

    @property
    def timed_out(self) -> bool:
        return self.status == NavigationSettleStatus.TIMED_OUT

    @property
    def failed(self) -> bool:
        return self.status == NavigationSettleStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "complete": self.complete,
            "success": self.success,
            "steps": int(self.steps),
            "stable_steps": int(self.stable_steps),
            "elapsed_sec": float(self.elapsed_sec),
            "reason": self.reason,
            "pose_delta_m": self.pose_delta_m,
            "yaw_delta_rad": self.yaw_delta_rad,
            "linear_speed_mps": self.linear_speed_mps,
            "angular_speed_rad_s": self.angular_speed_rad_s,
        }


class NavigationSettleBarrier:
    """Dwell on measured pose/twist before releasing manipulation."""

    def __init__(
        self,
        port: NavigationSettlePort,
        *,
        linear_speed_tolerance: float = 0.005,
        angular_speed_tolerance: float = 0.005,
        consecutive_steps: int = 8,
        timeout_sec: float = 5.0,
        timeout_steps: int | None = None,
        pose_drift_tolerance_m: float = 0.002,
        yaw_drift_tolerance_rad: float = 0.002,
    ) -> None:
        if not isinstance(port, NavigationSettlePort):
            raise TypeError("NavigationSettleBarrier requires NavigationSettlePort")
        self.port = port
        self.linear_speed_tolerance = self._nonnegative_finite(
            linear_speed_tolerance, "linear_speed_tolerance"
        )
        self.angular_speed_tolerance = self._nonnegative_finite(
            angular_speed_tolerance, "angular_speed_tolerance"
        )
        self.pose_drift_tolerance_m = self._nonnegative_finite(
            pose_drift_tolerance_m, "pose_drift_tolerance_m"
        )
        self.yaw_drift_tolerance_rad = self._nonnegative_finite(
            yaw_drift_tolerance_rad, "yaw_drift_tolerance_rad"
        )
        self.consecutive_steps = int(consecutive_steps)
        if self.consecutive_steps < 1:
            raise ValueError("consecutive_steps must be positive")
        self.timeout_sec = self._positive_finite(timeout_sec, "timeout_sec")
        if timeout_steps is not None:
            self.timeout_steps = int(timeout_steps)
            if self.timeout_steps < 1:
                raise ValueError("timeout_steps must be positive")
        else:
            self.timeout_steps = None
        self.reset()

    @staticmethod
    def _nonnegative_finite(value: Any, name: str) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return value

    @classmethod
    def _positive_finite(cls, value: Any, name: str) -> float:
        value = cls._nonnegative_finite(value, name)
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")
        return value

    def reset(self) -> None:
        self._status = NavigationSettleStatus.IDLE
        self._steps = 0
        self._stable_steps = 0
        self._elapsed_sec = 0.0
        self._start_time_sec: float | None = None
        self._last_time_sec: float | None = None
        self._previous_sample: NavigationBaseState | None = None
        self._last_measurement: dict[str, float | None] = {
            "pose_delta_m": None,
            "yaw_delta_rad": None,
            "linear_speed_mps": None,
            "angular_speed_rad_s": None,
        }
        self._reason = ""

    @property
    def status(self) -> NavigationSettleStatus:
        return self._status

    @property
    def stable_steps(self) -> int:
        return self._stable_steps

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def result(self) -> NavigationSettleResult:
        return NavigationSettleResult(
            status=self._status,
            steps=self._steps,
            stable_steps=self._stable_steps,
            elapsed_sec=self._elapsed_sec,
            reason=self._reason,
            **self._last_measurement,
        )

    def start(self, now_sec: float | None = None) -> NavigationSettleResult:
        self.reset()
        self._status = NavigationSettleStatus.WAITING
        if now_sec is not None:
            now_value = float(now_sec)
            if not math.isfinite(now_value):
                raise ValueError("settle start time must be finite")
            self._start_time_sec = now_value
            self._last_time_sec = now_value
        return self.result

    def _advance_time(self, now_sec: float | None, dt_sec: float | None) -> None:
        dt_value = 0.0 if dt_sec is None else float(dt_sec)
        if not math.isfinite(dt_value) or dt_value < 0.0:
            raise ValueError("settle dt_sec must be finite and non-negative")
        if now_sec is not None:
            now_value = float(now_sec)
            if not math.isfinite(now_value):
                raise ValueError("settle now_sec must be finite")
            if self._start_time_sec is None:
                self._start_time_sec = now_value
            if self._last_time_sec is not None and now_value > self._last_time_sec:
                self._elapsed_sec += now_value - self._last_time_sec
            elif dt_value > 0.0:
                # Small fakes and paused simulator clocks still progress by
                # physics ticks when an explicit dt is supplied.
                self._elapsed_sec += dt_value
            self._last_time_sec = now_value
            return
        self._elapsed_sec += dt_value

    def _finish(self, status: NavigationSettleStatus, reason: str) -> NavigationSettleResult:
        self._status = status
        self._reason = str(reason)
        return self.result

    def step(
        self,
        now_sec: float | None = None,
        *,
        dt_sec: float | None = None,
    ) -> NavigationSettleResult:
        """Consume one post-goal physics sample.

        ``Navigate.update`` calls this once after each physics step.  The
        command is stopped before measuring, but the measurement itself is
        read-only and therefore reflects the previous physical tick.
        """

        if self._status == NavigationSettleStatus.IDLE:
            self.start(now_sec)
        if self.result.complete:
            return self.result
        try:
            self._advance_time(now_sec, dt_sec)
            self.port.stop()
            sample = self.port.measure_base_state()
            if not isinstance(sample, NavigationBaseState):
                raise TypeError("settle measurement must be NavigationBaseState")
        except Exception as exc:
            self._steps += 1
            return self._finish(
                NavigationSettleStatus.FAILED,
                f"settle_measurement_failed:{type(exc).__name__}:{exc}",
            )

        self._steps += 1
        pose_delta = None
        yaw_delta = None
        if self._previous_sample is not None:
            pose_delta = float(
                np.linalg.norm(
                    np.asarray(sample.xy, dtype=float)
                    - np.asarray(self._previous_sample.xy, dtype=float)
                )
            )
            yaw_delta = abs(_wrap_angle(sample.yaw - self._previous_sample.yaw))
        else:
            pose_delta = 0.0
            yaw_delta = 0.0
        linear_speed = float(sample.linear_speed)
        angular_speed = float(sample.angular_speed)
        self._last_measurement = {
            "pose_delta_m": pose_delta,
            "yaw_delta_rad": yaw_delta,
            "linear_speed_mps": linear_speed,
            "angular_speed_rad_s": angular_speed,
        }
        stable = bool(
            linear_speed <= self.linear_speed_tolerance
            and angular_speed <= self.angular_speed_tolerance
            and pose_delta <= self.pose_drift_tolerance_m
            and yaw_delta <= self.yaw_drift_tolerance_rad
        )
        self._stable_steps = self._stable_steps + 1 if stable else 0
        self._previous_sample = sample

        if self._stable_steps >= self.consecutive_steps:
            return self._finish(NavigationSettleStatus.SETTLED, "settled")
        if (
            (self.timeout_steps is not None and self._steps >= self.timeout_steps)
            or self._elapsed_sec >= self.timeout_sec
        ):
            return self._finish(NavigationSettleStatus.TIMED_OUT, "settle_timeout")
        return self.result

    def abort(self, reason: str = "aborted") -> NavigationSettleResult:
        """Mark the barrier failed without changing navigation ownership."""

        if self.result.complete:
            return self.result
        return self._finish(NavigationSettleStatus.FAILED, reason)


__all__ = [
    "NavigationBaseState",
    "NavigationSettleBarrier",
    "NavigationSettlePort",
    "NavigationSettleQueryPort",
    "NavigationSettleResult",
    "NavigationSettleStatus",
]
