"""ROS-free mobile-base command driver used by the local navigation skill.

The driver deliberately keeps the chassis-specific command mapping that was
previously hidden behind the ROS ``/cmd_vel`` bridge.  It accepts one body-frame
twist directly from a skill and applies the resulting joint targets to Isaac.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class BaseCommand:
    """Body-frame twist command with the local receive timestamp."""

    vx_body: float
    vy_body: float
    wz_body: float
    received_time_sec: float

    @classmethod
    def zero(cls, *, received_time_sec: float) -> "BaseCommand":
        return cls(0.0, 0.0, 0.0, float(received_time_sec))


class LocalBaseDriver(ABC):
    """Translate a body twist into direct Isaac base joint targets."""

    def __init__(self, robot, *, world=None):
        self.robot = robot
        self.world = world
        self.base_interface = robot.get_base_interface()
        self.base_cfg = self.base_interface["base_cfg"]

        self._command_timeout = float(self.base_cfg.get("command_timeout", 0.25))
        self._steering_limit = float(self.base_cfg.get("steering_limit", math.pi))
        self._steering_rate_limit = float(self.base_cfg.get("steering_rate_limit", 100.0))
        self._wheel_velocity_limit = float(self.base_cfg.get("wheel_velocity_limit", 100.0))
        self._wheel_base = float(self.base_cfg.get("wheel_base", 0.0))
        self._track_width = float(self.base_cfg.get("track_width", 0.0))
        self._wheel_radius = float(self.base_cfg.get("wheel_radius", 0.1))
        self._steering_command_sign = float(self.base_cfg.get("steering_command_sign", 1.0))
        (
            self._min_body_velocity,
            self._max_body_velocity,
            self._max_body_acceleration,
            self._max_body_deceleration,
        ) = self._load_body_motion_limits()
        if abs(self._steering_command_sign) <= 1.0e-6:
            raise ValueError("steering_command_sign must be non-zero")
        if self._wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")

        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        self._validate_configuration(steering_count=steering_count, wheel_count=wheel_count)

        now = self._now_sec()
        self._command = BaseCommand.zero(received_time_sec=now)
        self._last_step_command = self._command
        self._last_applied_steering = np.zeros(steering_count, dtype=np.float32)
        self._last_requested_steering = np.zeros(steering_count, dtype=np.float32)
        self._last_requested_wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        self._last_applied_wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        self._last_wheel_shaping_debug = {}
        self._last_step_time_sec = now
        self._last_step_dt = 1.0e-3
        self._navigation_active = False
        self._has_command = False
        self._base_hold_suspended = False
        self._last_shaped_body_velocity = np.zeros(3, dtype=np.float32)
        history_size = max(int(self.base_cfg.get("debug_history_size", 64)), 1)
        self._driver_command_message_count = 0
        self._applied_driver_command_count = 0
        self._debug_command_history = deque(maxlen=history_size)
        self._last_actual_translation = np.zeros(3, dtype=np.float32)
        self._last_actual_yaw = 0.0
        self._last_published_pose_debug = {}
        self._non_finite_state_detected = False
        self._non_finite_state_reason = ""
        translation, orientation = self._get_robot_base_pose()
        self._last_actual_translation = np.asarray(translation, dtype=np.float32).copy()
        self._last_actual_yaw = float(self._yaw_from_wxyz(orientation))

    @abstractmethod
    def _validate_configuration(self, *, steering_count: int, wheel_count: int):
        """Validate the selected chassis profile."""

    @abstractmethod
    def _map_command(self, command: BaseCommand) -> tuple[np.ndarray, np.ndarray]:
        """Map body-frame velocity to steering and wheel targets."""

    def _now_sec(self) -> float:
        if self.world is not None:
            current_time = getattr(self.world, "current_time", None)
            if current_time is not None:
                try:
                    return float(current_time)
                except (TypeError, ValueError):
                    pass
            getter = getattr(self.world, "get_current_time", None)
            if callable(getter):
                try:
                    return float(getter())
                except (TypeError, ValueError):
                    pass
        return time.monotonic()

    def set_command(self, vx_body: float, vy_body: float, wz_body: float) -> None:
        """Accept exactly one body-frame twist per navigation control cycle."""
        now = self._now_sec()
        command = BaseCommand(float(vx_body), float(vy_body), float(wz_body), now)
        if not self._is_finite_command(command):
            raise ValueError("Local base command must be finite")
        self._command = self._clamp_command(command)
        self._has_command = True
        self._driver_command_message_count += 1
        self._applied_driver_command_count += int(self._navigation_active)

    def prepare_for_navigation(self) -> None:
        suspend_hold = getattr(self.robot, "suspend_manipulation_base_hold", None)
        if callable(suspend_hold):
            self._base_hold_suspended = self._base_hold_suspended or bool(suspend_hold())
        self._navigation_active = True
        self._has_command = False
        self._last_shaped_body_velocity.fill(0.0)
        self.set_command(0.0, 0.0, 0.0)

    def finalize_after_navigation(self) -> None:
        self._navigation_active = False
        self._has_command = False
        self._command = BaseCommand.zero(received_time_sec=self._now_sec())
        self._last_shaped_body_velocity.fill(0.0)
        try:
            self._apply_command(self._command, max(self._last_step_dt, 1.0e-3))
        finally:
            if self._base_hold_suspended:
                resume_hold = getattr(self.robot, "resume_manipulation_base_hold", None)
                if callable(resume_hold):
                    resume_hold()
            self._base_hold_suspended = False

    def reset(self, *, clear_debug_history: bool = False) -> None:
        now = self._now_sec()
        self._command = BaseCommand.zero(received_time_sec=now)
        self._last_step_command = self._command
        self._last_step_time_sec = now
        self._last_step_dt = 1.0e-3
        self._navigation_active = False
        self._has_command = False
        if self._base_hold_suspended:
            resume_hold = getattr(self.robot, "resume_manipulation_base_hold", None)
            if callable(resume_hold):
                resume_hold()
        self._base_hold_suspended = False
        self._last_shaped_body_velocity.fill(0.0)
        self._last_wheel_shaping_debug = {}
        current_steering = self._get_current_steering_positions()
        self._last_applied_steering = current_steering.copy()
        self._last_requested_steering = current_steering.copy()
        self._last_requested_wheel_velocities.fill(0.0)
        self._last_applied_wheel_velocities.fill(0.0)
        self._non_finite_state_detected = False
        self._non_finite_state_reason = ""
        if clear_debug_history:
            self._driver_command_message_count = 0
            self._applied_driver_command_count = 0
            self._debug_command_history.clear()
        translation, orientation = self._get_robot_base_pose()
        self._last_actual_translation = np.asarray(translation, dtype=np.float32).copy()
        self._last_actual_yaw = float(self._yaw_from_wxyz(orientation))

    def step(self, step_dt: float | None = None) -> None:
        now = self._now_sec()
        dt = max(float(step_dt) if step_dt is not None else now - self._last_step_time_sec, 1.0e-3)
        self._last_step_time_sec = now
        self._last_step_dt = dt
        if not self._navigation_active:
            return
        target = self._resolve_active_command(now) if self._has_command else BaseCommand.zero(received_time_sec=now)
        command = self._apply_body_acceleration_limits(target, dt)
        self._apply_command(command, dt)

    def _apply_command(self, command: BaseCommand, dt: float) -> None:
        self._last_step_command = command
        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        requested_steering, requested_wheels = self._map_command(command)
        requested_steering = self._require_finite_vector(requested_steering, steering_count, "requested steering")
        requested_wheels = self._require_finite_vector(requested_wheels, wheel_count, "requested wheel velocities")
        self._last_requested_steering = requested_steering.copy()
        self._last_requested_wheel_velocities = requested_wheels.copy()
        steering = self._apply_steering_limits(requested_steering, dt)
        actual_steering = self._get_current_steering_positions()
        wheels = self._shape_wheel_velocities_for_applied_steering(
            command=command,
            requested_steering=requested_steering,
            applied_steering=actual_steering,
            requested_wheel_velocities=requested_wheels,
        )
        wheels = self._require_finite_vector(wheels, wheel_count, "applied wheel velocities")
        self._last_applied_wheel_velocities = wheels.copy()
        try:
            self.robot.apply_base_command(steering_positions=steering, wheel_velocities=wheels, step_dt=dt)
        except TypeError:
            self.robot.apply_base_command(steering_positions=steering, wheel_velocities=wheels)
        self._record_debug_history(command, steering, wheels, now=self._now_sec(), dt=dt)

    def _resolve_active_command(self, now: float) -> BaseCommand:
        if now - self._command.received_time_sec <= self._command_timeout:
            return self._command
        return BaseCommand.zero(received_time_sec=self._command.received_time_sec)

    def _load_body_motion_limits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        platform_cfg = self.base_cfg.get("platform", {})
        local_cfg = platform_cfg.get("local_navigation", {}) if isinstance(platform_cfg, dict) else {}
        limits = local_cfg.get("controller_hard_limits") if isinstance(local_cfg, dict) else None
        if not isinstance(limits, dict):
            raise KeyError("Missing required mapping config: platform.local_navigation.controller_hard_limits")
        minimum = np.asarray(limits.get("min_velocity", [-1.0, -1.0, -1.0]), dtype=np.float32).reshape(-1)
        maximum = np.asarray(limits.get("max_velocity", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(-1)
        acceleration = np.asarray(limits.get("max_accel", [np.inf, np.inf, np.inf]), dtype=np.float32).reshape(-1)
        deceleration = np.abs(np.asarray(limits.get("max_decel", [np.inf, np.inf, np.inf]), dtype=np.float32).reshape(-1))
        if minimum.size != 3 or maximum.size != 3 or not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("local_navigation controller velocity limits must be 3-element finite lists")
        if acceleration.size != 3 or deceleration.size != 3 or np.any(np.isnan(acceleration)) or np.any(np.isnan(deceleration)) or np.any(acceleration < 0.0) or np.any(deceleration < 0.0):
            raise ValueError("local_navigation acceleration limits must be three non-negative values")
        return minimum, maximum, acceleration, deceleration

    def _apply_body_acceleration_limits(self, command: BaseCommand, dt: float) -> BaseCommand:
        target = np.asarray([command.vx_body, command.vy_body, command.wz_body], dtype=np.float32)
        previous = self._last_shaped_body_velocity.copy()
        shaped = previous.copy()
        for index in range(3):
            target_value = float(target[index])
            previous_value = float(previous[index])
            if previous_value * target_value < 0.0:
                max_change = float(self._max_body_deceleration[index]) * dt
                shaped[index] = 0.0 if max_change >= abs(previous_value) else previous_value - math.copysign(max_change, previous_value)
                continue
            speeding_up = abs(target_value) > abs(previous_value)
            rate = float(self._max_body_acceleration[index] if speeding_up else self._max_body_deceleration[index])
            max_change = rate * dt
            delta = target_value - previous_value
            shaped[index] = target_value if abs(delta) <= max_change + 1.0e-7 else previous_value + math.copysign(max_change, delta)
        self._last_shaped_body_velocity = shaped
        return BaseCommand(float(shaped[0]), float(shaped[1]), float(shaped[2]), command.received_time_sec)

    def _clamp_command(self, command: BaseCommand) -> BaseCommand:
        values = np.clip(
            np.asarray([command.vx_body, command.vy_body, command.wz_body], dtype=np.float32),
            self._min_body_velocity,
            self._max_body_velocity,
        )
        return BaseCommand(float(values[0]), float(values[1]), float(values[2]), command.received_time_sec)

    def _apply_steering_limits(self, requested: np.ndarray, dt: float) -> np.ndarray:
        if requested.size == 0:
            return requested.copy()
        requested = np.clip(requested, -self._steering_limit, self._steering_limit)
        max_delta = max(self._steering_rate_limit, 0.0) * dt
        limited = self._last_applied_steering + np.clip(requested - self._last_applied_steering, -max_delta, max_delta)
        self._last_applied_steering = limited.astype(np.float32)
        return self._last_applied_steering.copy()

    def _get_current_steering_positions(self) -> np.ndarray:
        joint_state = self.robot.get_base_joint_state()
        expected = len(self.base_interface["steering_joint_names"])
        current = np.asarray(joint_state["steering_positions"], dtype=np.float32).reshape(-1)
        if current.size != expected or not np.all(np.isfinite(current)):
            raise ValueError("Current steering positions must match steering joints and be finite")
        return np.asarray([self._wrap_angle(float(value)) for value in current], dtype=np.float32)

    def _shape_wheel_velocities_for_applied_steering(self, *, command, requested_steering, applied_steering, requested_wheel_velocities):
        del command, requested_steering, applied_steering
        return np.asarray(requested_wheel_velocities, dtype=np.float32).copy()

    def get_logging_action_snapshot(self) -> dict:
        command = self._last_step_command
        return {
            "vx_body": float(command.vx_body),
            "vy_body": float(command.vy_body),
            "wz_body": float(command.wz_body),
            "navigation_active": bool(self._navigation_active),
            "has_local_command": bool(self._has_command),
            "requested_steering": [float(v) for v in self._last_requested_steering.tolist()],
            "requested_wheel_velocities": [float(v) for v in self._last_requested_wheel_velocities.tolist()],
            "applied_wheel_velocities": [float(v) for v in self._last_applied_wheel_velocities.tolist()],
            "wheel_shaping": dict(self._last_wheel_shaping_debug),
        }

    def get_logging_state_snapshot(self) -> dict:
        translation, orientation = self._get_robot_base_pose()
        translation = np.asarray(translation, dtype=np.float32).reshape(-1)[:3]
        yaw = float(self._yaw_from_wxyz(orientation))
        dt = max(float(self._last_step_dt), 1.0e-3)
        linear_world = (translation - self._last_actual_translation) / dt
        angular_world_z = self._wrap_angle(yaw - self._last_actual_yaw) / dt
        self._last_actual_translation = translation.copy()
        self._last_actual_yaw = yaw
        linear_body = self._world_linear_velocity_to_body(linear_world, orientation)
        joint_state = self.robot.get_base_joint_state()
        return {
            "pose": [float(translation[0]), float(translation[1]), float(translation[2]), yaw],
            "twist_body": [float(linear_body[0]), float(linear_body[1]), float(angular_world_z)],
            "steering_positions": [float(v) for v in np.asarray(joint_state["steering_positions"]).reshape(-1)],
            "wheel_positions": [float(v) for v in np.asarray(joint_state["wheel_positions"]).reshape(-1)],
            "steering_velocities": [float(v) for v in np.asarray(joint_state["steering_velocities"]).reshape(-1)],
            "wheel_velocities": [float(v) for v in np.asarray(joint_state["wheel_velocities"]).reshape(-1)],
        }

    def has_non_finite_state(self) -> bool:
        return bool(self._non_finite_state_detected)

    def non_finite_state_reason(self) -> str:
        return str(self._non_finite_state_reason)

    def _record_debug_history(self, command, steering, wheels, *, now, dt):
        self._debug_command_history.append({
            "time_sec": float(now),
            "dt": float(dt),
            "command": {"vx_body": float(command.vx_body), "vy_body": float(command.vy_body), "wz_body": float(command.wz_body)},
            "applied_steering": [float(v) for v in steering.tolist()],
            "wheel_velocities": [float(v) for v in wheels.tolist()],
        })

    @staticmethod
    def _require_finite_vector(values, expected_size: int, name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.size != expected_size or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must have size {expected_size} and contain only finite values")
        return vector.copy()

    @staticmethod
    def _is_finite_command(command: BaseCommand) -> bool:
        return all(math.isfinite(float(v)) for v in (command.vx_body, command.vy_body, command.wz_body, command.received_time_sec))

    def _get_robot_base_pose(self):
        getter = getattr(self.robot, "get_nav_base_pose", None)
        if callable(getter):
            return getter()
        getter = getattr(self.robot, "get_mobile_base_pose", None)
        if not callable(getter):
            raise ValueError("Robot must provide get_mobile_base_pose")
        return getter()

    @staticmethod
    def _yaw_from_wxyz(q_wxyz):
        w, x, y, z = [float(v) for v in q_wxyz[:4]]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _world_linear_velocity_to_body(linear_world, orientation):
        yaw = LocalBaseDriver._yaw_from_wxyz(orientation)
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        vx, vy, vz = [float(v) for v in np.asarray(linear_world).reshape(-1)[:3]]
        return np.asarray([cos_yaw * vx + sin_yaw * vy, -sin_yaw * vx + cos_yaw * vy, vz], dtype=np.float32)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


class LocalVirtualBaseDriver(LocalBaseDriver):
    def __init__(self, robot, *, world=None):
        self._body_twist_deadband = 0.0
        self._base_velocity_signs = np.ones(3, dtype=np.float32)
        super().__init__(robot, world=world)

    def _validate_configuration(self, *, steering_count, wheel_count):
        if steering_count != 0 or wheel_count != 3:
            raise ValueError("LocalVirtualBaseDriver expects 0 steering and 3 velocity joints")
        names = list(self.base_interface["wheel_joint_names"])
        if list(self.base_cfg.get("base_velocity_joint_names", names)) != names:
            raise ValueError("base_velocity_joint_names must match wheel_joint_names")
        self._body_twist_deadband = max(float(self.base_cfg.get("body_twist_deadband", 0.0)), 0.0)
        signs = np.asarray(self.base_cfg.get("base_velocity_command_signs", [1.0, 1.0, 1.0]), dtype=np.float32)
        if signs.size != 3 or not np.all(np.isfinite(signs)):
            raise ValueError("base_velocity_command_signs must be three finite values")
        self._base_velocity_signs = signs

    def _map_command(self, command):
        values = np.asarray([command.vx_body, command.vy_body, command.wz_body], dtype=np.float32)
        values[np.abs(values) <= self._body_twist_deadband] = 0.0
        values = np.clip(values, self._min_body_velocity, self._max_body_velocity)
        self._last_wheel_shaping_debug = {"mode": "virtual_base_velocity_target", "body_velocity": values.tolist()}
        return np.zeros(0, dtype=np.float32), values * self._base_velocity_signs


class LocalRangerMiniV3Driver(LocalBaseDriver):
    def __init__(self, robot, *, world=None):
        self._module_positions = np.zeros((4, 2), dtype=np.float32)
        self._body_twist_deadband = 0.0
        self._module_speed_deadband = 0.0
        self._wheel_velocity_lowpass_alpha = 0.8
        super().__init__(robot, world=world)

    def _validate_configuration(self, *, steering_count, wheel_count):
        if steering_count != 4 or wheel_count != 4:
            raise ValueError("LocalRangerMiniV3Driver expects 4 steering and 4 wheel joints")
        self._module_positions = np.asarray([
            [0.5 * self._wheel_base, 0.5 * self._track_width],
            [0.5 * self._wheel_base, -0.5 * self._track_width],
            [-0.5 * self._wheel_base, 0.5 * self._track_width],
            [-0.5 * self._wheel_base, -0.5 * self._track_width],
        ], dtype=np.float32)
        self._body_twist_deadband = max(float(self.base_cfg.get("body_twist_deadband", 0.0)), 0.0)
        self._module_speed_deadband = max(float(self.base_cfg.get("module_speed_deadband", 0.0)), 0.0)
        self._wheel_velocity_lowpass_alpha = float(np.clip(self.base_cfg.get("wheel_velocity_lowpass_alpha", 0.8), 0.0, 1.0))

    def _map_command(self, command):
        current = self._last_applied_steering.astype(np.float32).copy()
        requested = current.copy()
        linear = np.zeros(4, dtype=np.float32)
        if max(abs(command.vx_body), abs(command.vy_body), abs(command.wz_body)) <= self._body_twist_deadband:
            return requested, linear
        for index, (module_x, module_y) in enumerate(self._module_positions):
            vx = float(command.vx_body) - float(command.wz_body) * float(module_y)
            vy = float(command.vy_body) + float(command.wz_body) * float(module_x)
            speed = math.hypot(vx, vy)
            if speed <= self._module_speed_deadband:
                continue
            desired = self._steering_command_sign * math.atan2(vy, vx)
            desired, speed = self._minimize_rotation(desired, float(current[index]), speed)
            requested[index] = float(np.clip(desired, -self._steering_limit, self._steering_limit))
            linear[index] = float(speed)
        wheels = linear / self._wheel_radius
        peak = float(np.max(np.abs(wheels))) if wheels.size else 0.0
        if peak > self._wheel_velocity_limit > 0.0:
            wheels *= self._wheel_velocity_limit / peak
        return requested, wheels.astype(np.float32)

    def _shape_wheel_velocities_for_applied_steering(self, *, command, requested_steering, applied_steering, requested_wheel_velocities):
        applied = np.asarray(applied_steering, dtype=np.float32).reshape(-1)
        shaped = np.zeros(4, dtype=np.float32)
        if max(abs(command.vx_body), abs(command.vy_body), abs(command.wz_body)) <= self._body_twist_deadband:
            self._last_wheel_shaping_debug = {
                "mode": "ranger_immediate_stop",
                "applied_steering": applied.tolist(),
            }
            return shaped
        for index, (module_x, module_y) in enumerate(self._module_positions):
            theta = float(applied[index]) * self._steering_command_sign
            vx = float(command.vx_body) - float(command.wz_body) * float(module_y)
            vy = float(command.vy_body) + float(command.wz_body) * float(module_x)
            shaped[index] = (vx * math.cos(theta) + vy * math.sin(theta)) / self._wheel_radius
        alpha = self._wheel_velocity_lowpass_alpha
        if 0.0 < alpha < 1.0 and self._last_applied_wheel_velocities.size == 4:
            shaped = alpha * shaped + (1.0 - alpha) * self._last_applied_wheel_velocities
        peak = float(np.max(np.abs(shaped))) if shaped.size else 0.0
        if peak > self._wheel_velocity_limit > 0.0:
            shaped *= self._wheel_velocity_limit / peak
        self._last_wheel_shaping_debug = {"mode": "ranger_actual_steering_ik", "applied_steering": applied.tolist(), "requested_wheel_velocities": np.asarray(requested_wheel_velocities).tolist()}
        return shaped.astype(np.float32)

    @staticmethod
    def _minimize_rotation(desired, current, speed):
        desired = math.atan2(math.sin(desired), math.cos(desired))
        current = math.atan2(math.sin(current), math.cos(current))
        error = math.atan2(math.sin(desired - current), math.cos(desired - current))
        if abs(error) > 0.5 * math.pi:
            desired = math.atan2(math.sin(desired + math.pi), math.cos(desired + math.pi))
            speed = -speed
        return desired, speed


class LocalDifferentialDriveDriver(LocalBaseDriver):
    def __init__(self, robot, *, world=None):
        self._left_wheel_indices = []
        self._right_wheel_indices = []
        self._body_twist_deadband = 0.0
        self._wheel_velocity_signs = np.ones(2, dtype=np.float32)
        super().__init__(robot, world=world)

    def _validate_configuration(self, *, steering_count, wheel_count):
        if steering_count != 0 or wheel_count != 2:
            raise ValueError("LocalDifferentialDriveDriver expects 0 steering and 2 wheel joints")
        names = list(self.base_interface["wheel_joint_names"])
        self._left_wheel_indices = [names.index(name) for name in self.base_cfg.get("left_wheel_joint_names", [names[0]])]
        self._right_wheel_indices = [names.index(name) for name in self.base_cfg.get("right_wheel_joint_names", [names[1]])]
        self._body_twist_deadband = max(float(self.base_cfg.get("body_twist_deadband", 0.0)), 0.0)
        signs = np.asarray(self.base_cfg.get("wheel_velocity_command_signs", [1.0, 1.0]), dtype=np.float32)
        if signs.size != 2:
            raise ValueError("wheel_velocity_command_signs must have two values")
        self._wheel_velocity_signs = signs

    def _map_command(self, command):
        vx = 0.0 if abs(command.vx_body) <= self._body_twist_deadband else float(command.vx_body)
        wz = 0.0 if abs(command.wz_body) <= self._body_twist_deadband else float(command.wz_body)
        if abs(command.vy_body) > self._body_twist_deadband:
            raise ValueError("Differential drive cannot execute lateral body velocity")
        wheels = np.zeros(2, dtype=np.float32)
        left = (vx - 0.5 * wz * self._track_width) / self._wheel_radius
        right = (vx + 0.5 * wz * self._track_width) / self._wheel_radius
        for index in self._left_wheel_indices:
            wheels[index] = left
        for index in self._right_wheel_indices:
            wheels[index] = right
        wheels *= self._wheel_velocity_signs
        peak = float(np.max(np.abs(wheels))) if wheels.size else 0.0
        if peak > self._wheel_velocity_limit > 0.0:
            wheels *= self._wheel_velocity_limit / peak
        return np.zeros(0, dtype=np.float32), wheels


def build_local_base_driver(robot, *, world=None):
    """Build the pure local driver from ``platform.profile``."""
    base_cfg = robot.get_base_interface().get("base_cfg", {})
    profile = str(dict(base_cfg.get("platform", {})).get("profile", "")).strip().lower().replace("-", "_")
    if profile in {"virtual_base", "omni_virtual_base", "panda_omron_virtual", "panda_omron_virtual_base"}:
        return LocalVirtualBaseDriver(robot, world=world)
    if profile in {"ranger_mini_v3", "ranger_mini_v3_4wis", "ranger_mini", "split_aloha_base"}:
        return LocalRangerMiniV3Driver(robot, world=world)
    if profile in {"differential_drive", "diff_drive", "omron_diff_drive", "panda_omron_base"}:
        return LocalDifferentialDriveDriver(robot, world=world)
    raise KeyError(f"Unsupported local mobile base platform profile: {profile or '<missing>'}")
