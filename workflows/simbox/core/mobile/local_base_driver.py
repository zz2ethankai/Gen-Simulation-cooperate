"""Body-twist executor used by the ROS-free local navigation skill.

Virtual mobile bases consume the twist through their X/Y/yaw articulation
joints. Other profiles retain the direct-pose fallback used by local tests and
non-virtual assets. No physical wheel or steering mapping is performed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class BaseCommand:
    vx_body: float
    vy_body: float
    wz_body: float
    received_time_sec: float

    @classmethod
    def zero(cls, *, received_time_sec: float) -> "BaseCommand":
        return cls(0.0, 0.0, 0.0, float(received_time_sec))


class LocalBaseDriver:
    """Execute a local-navigation body twist without chassis kinematics."""

    def __init__(self, robot, *, world=None):
        self.robot = robot
        self.world = world
        self.base_cfg = robot.get_base_interface()["base_cfg"]
        platform_cfg = self.base_cfg.get("platform", {})
        profile = str(platform_cfg.get("profile", "") if isinstance(platform_cfg, dict) else "").strip().lower().replace("-", "_")
        self._uses_virtual_base_joints = profile in {
            "virtual_base",
            "omni_virtual_base",
            "panda_omron_virtual",
            "panda_omron_virtual_base",
        }
        signs = np.asarray(self.base_cfg.get("base_velocity_command_signs", [1.0, 1.0, 1.0]), dtype=np.float32).reshape(-1)
        if self._uses_virtual_base_joints and (signs.size != 3 or not np.all(np.isfinite(signs))):
            raise ValueError("Virtual base velocity command signs must contain three finite values")
        self._virtual_base_velocity_signs = signs if self._uses_virtual_base_joints else np.ones(3, dtype=np.float32)
        self._command_timeout = float(self.base_cfg.get("command_timeout", 0.25))
        now = self._now_sec()
        self._command = BaseCommand.zero(received_time_sec=now)
        self._last_step_command = self._command
        self._last_step_time_sec = now
        self._last_step_dt = 1.0e-3
        self._navigation_active = False
        self._has_command = False
        self._base_hold_suspended = False
        self._driver_command_message_count = 0
        self._applied_driver_command_count = 0
        self._debug_command_history = deque(maxlen=max(int(self.base_cfg.get("debug_history_size", 64)), 1))
        self._last_actual_translation, orientation = self._get_robot_base_pose()
        self._last_actual_translation = np.asarray(self._last_actual_translation, dtype=np.float32).copy()
        self._last_actual_yaw = self._yaw_from_wxyz(orientation)
        self._non_finite_state_detected = False
        self._non_finite_state_reason = ""

    def _now_sec(self) -> float:
        if self.world is not None:
            for name in ("current_time", "get_current_time"):
                value = getattr(self.world, name, None)
                try:
                    return float(value() if callable(value) else value)
                except (TypeError, ValueError):
                    continue
        return time.monotonic()

    def set_command(self, vx_body: float, vy_body: float, wz_body: float) -> None:
        values = np.asarray([vx_body, vy_body, wz_body], dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("Local base command must be finite")
        self._command = BaseCommand(float(values[0]), float(values[1]), float(values[2]), self._now_sec())
        self._has_command = True
        self._driver_command_message_count += 1
        self._applied_driver_command_count += int(self._navigation_active)

    def prepare_for_navigation(self) -> None:
        suspend_hold = getattr(self.robot, "suspend_manipulation_base_hold", None)
        if callable(suspend_hold):
            self._base_hold_suspended = self._base_hold_suspended or bool(suspend_hold())
        self._navigation_active = True
        self._has_command = False
        self.set_command(0.0, 0.0, 0.0)

    def finalize_after_navigation(self) -> None:
        if self._navigation_active and self._uses_virtual_base_joints:
            self._apply_virtual_base_command(BaseCommand.zero(received_time_sec=self._now_sec()), self._last_step_dt)
        self._navigation_active = False
        self._has_command = False
        self._command = BaseCommand.zero(received_time_sec=self._now_sec())
        if self._base_hold_suspended:
            resume_hold = getattr(self.robot, "resume_manipulation_base_hold", None)
            if callable(resume_hold):
                resume_hold()
        self._base_hold_suspended = False

    def reset(self, *, clear_debug_history: bool = False) -> None:
        self.finalize_after_navigation()
        now = self._now_sec()
        self._last_step_time_sec = now
        self._last_step_dt = 1.0e-3
        self._last_step_command = BaseCommand.zero(received_time_sec=now)
        if clear_debug_history:
            self._driver_command_message_count = 0
            self._applied_driver_command_count = 0
            self._debug_command_history.clear()
        translation, orientation = self._get_robot_base_pose()
        self._last_actual_translation = np.asarray(translation, dtype=np.float32).copy()
        self._last_actual_yaw = self._yaw_from_wxyz(orientation)

    def step(self, step_dt: float | None = None) -> None:
        now = self._now_sec()
        dt = max(float(step_dt) if step_dt is not None else now - self._last_step_time_sec, 1.0e-3)
        self._last_step_time_sec = now
        self._last_step_dt = dt
        if not self._navigation_active:
            return
        command = self._command if self._has_command and now - self._command.received_time_sec <= self._command_timeout else BaseCommand.zero(received_time_sec=now)
        self._apply_command(command, dt)

    def _apply_command(self, command: BaseCommand, dt: float) -> None:
        if self._uses_virtual_base_joints:
            self._apply_virtual_base_command(command, dt)
            return
        nav_translation, nav_orientation = self._get_robot_base_pose()
        nav_translation = np.asarray(nav_translation, dtype=np.float32).reshape(3)
        nav_yaw = self._yaw_from_wxyz(nav_orientation)
        cos_yaw, sin_yaw = math.cos(nav_yaw), math.sin(nav_yaw)
        target_nav_translation = nav_translation.copy()
        target_nav_translation[0] += dt * (cos_yaw * command.vx_body - sin_yaw * command.vy_body)
        target_nav_translation[1] += dt * (sin_yaw * command.vx_body + cos_yaw * command.vy_body)
        target_nav_yaw = self._wrap_angle(nav_yaw + dt * command.wz_body)
        target_translation, target_orientation = self._nav_pose_to_mobile_pose(
            target_nav_translation, target_nav_yaw, nav_translation, nav_orientation
        )
        self.robot.set_mobile_base_world_pose(target_translation, target_orientation)
        self._last_step_command = command
        self._debug_command_history.append({
            "time_sec": float(self._now_sec()), "dt": float(dt),
            "command": {"vx_body": command.vx_body, "vy_body": command.vy_body, "wz_body": command.wz_body},
        })

    def _apply_virtual_base_command(self, command: BaseCommand, dt: float) -> None:
        velocities = np.asarray(
            [command.vx_body, command.vy_body, command.wz_body],
            dtype=np.float32,
        ) * self._virtual_base_velocity_signs
        apply = getattr(self.robot, "apply_base_command", None)
        if not callable(apply):
            raise ValueError("Virtual base robot must provide apply_base_command")
        try:
            apply(
                steering_positions=np.zeros(0, dtype=np.float32),
                wheel_velocities=velocities,
                step_dt=float(dt),
            )
        except TypeError:
            apply(
                steering_positions=np.zeros(0, dtype=np.float32),
                wheel_velocities=velocities,
            )
        self._last_step_command = command
        self._debug_command_history.append({
            "time_sec": float(self._now_sec()),
            "dt": float(dt),
            "command": {"vx_body": command.vx_body, "vy_body": command.vy_body, "wz_body": command.wz_body},
            "execution_mode": "virtual_base_joint_velocity_target",
        })

    def _nav_pose_to_mobile_pose(self, target_nav_translation, target_nav_yaw, nav_translation, nav_orientation):
        mobile_translation, mobile_orientation = self.robot.get_mobile_base_pose()
        mobile_translation = np.asarray(mobile_translation, dtype=np.float32).reshape(3)
        mobile_yaw = self._yaw_from_wxyz(mobile_orientation)
        nav_yaw = self._yaw_from_wxyz(nav_orientation)
        relative_yaw = self._wrap_angle(nav_yaw - mobile_yaw)
        dx, dy = np.asarray(nav_translation, dtype=np.float32).reshape(3)[:2] - mobile_translation[:2]
        c, s = math.cos(mobile_yaw), math.sin(mobile_yaw)
        relative_xy = np.asarray([c * dx + s * dy, -s * dx + c * dy], dtype=np.float32)
        target_mobile_yaw = self._wrap_angle(target_nav_yaw - relative_yaw)
        c, s = math.cos(target_mobile_yaw), math.sin(target_mobile_yaw)
        target_mobile = np.asarray(target_nav_translation, dtype=np.float32).copy()
        target_mobile[:2] -= np.asarray([c * relative_xy[0] - s * relative_xy[1], s * relative_xy[0] + c * relative_xy[1]])
        return target_mobile, self._wxyz_from_yaw(target_mobile_yaw)

    def get_logging_action_snapshot(self) -> dict:
        command = self._last_step_command
        return {"vx_body": command.vx_body, "vy_body": command.vy_body, "wz_body": command.wz_body,
                "navigation_active": self._navigation_active, "has_local_command": self._has_command,
                "execution_mode": "virtual_base_joint_velocity_target" if self._uses_virtual_base_joints else "direct_body_twist",
                # Legacy logger keys are intentionally empty: no wheel command
                # is generated by this executor.
                "requested_steering": [], "requested_wheel_velocities": [],
                "applied_wheel_velocities": []}

    def get_logging_state_snapshot(self) -> dict:
        translation, orientation = self._get_robot_base_pose()
        translation = np.asarray(translation, dtype=np.float32).reshape(3)
        yaw = self._yaw_from_wxyz(orientation)
        dt = max(self._last_step_dt, 1.0e-3)
        linear_world = (translation - self._last_actual_translation) / dt
        linear_body = self._world_linear_velocity_to_body(linear_world, orientation)
        yaw_rate = self._wrap_angle(yaw - self._last_actual_yaw) / dt
        self._last_actual_translation, self._last_actual_yaw = translation.copy(), yaw
        joint_state = self.robot.get_base_joint_state()
        # These arrays remain as observation-schema compatibility fields only;
        # the direct executor never reads them or writes wheel targets.
        return {"pose": [float(translation[0]), float(translation[1]), float(translation[2]), yaw],
                "twist_body": [float(linear_body[0]), float(linear_body[1]), float(yaw_rate)],
                "steering_positions": [float(v) for v in np.asarray(joint_state.get("steering_positions", ())).reshape(-1)],
                "wheel_positions": [float(v) for v in np.asarray(joint_state.get("wheel_positions", ())).reshape(-1)],
                "steering_velocities": [float(v) for v in np.asarray(joint_state.get("steering_velocities", ())).reshape(-1)],
                "wheel_velocities": [float(v) for v in np.asarray(joint_state.get("wheel_velocities", ())).reshape(-1)]}

    def get_actual_twist_body(self) -> np.ndarray:
        """Return the current measured body twist without advancing log state.

        Skill scheduling runs after a physics step, whereas observation logging
        runs before it.  Keeping this query read-only lets Navigate wait for
        the latest physical motion to settle without corrupting the next
        observation's pose-delta measurement.
        """
        translation, orientation = self._get_robot_base_pose()
        translation = np.asarray(translation, dtype=np.float32).reshape(3)
        dt = max(self._last_step_dt, 1.0e-3)
        linear_world = (translation - self._last_actual_translation) / dt
        linear_body = self._world_linear_velocity_to_body(linear_world, orientation)
        yaw = self._yaw_from_wxyz(orientation)
        yaw_rate = self._wrap_angle(yaw - self._last_actual_yaw) / dt
        return np.asarray([linear_body[0], linear_body[1], yaw_rate], dtype=np.float32)

    def has_non_finite_state(self) -> bool:
        return self._non_finite_state_detected

    def non_finite_state_reason(self) -> str:
        return self._non_finite_state_reason

    def _get_robot_base_pose(self):
        getter = getattr(self.robot, "get_nav_base_pose", None) or getattr(self.robot, "get_mobile_base_pose", None)
        if not callable(getter):
            raise ValueError("Robot must provide get_nav_base_pose or get_mobile_base_pose")
        return getter()

    @staticmethod
    def _yaw_from_wxyz(q_wxyz):
        w, x, y, z = [float(value) for value in q_wxyz[:4]]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wxyz_from_yaw(yaw: float) -> np.ndarray:
        return np.asarray([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=np.float32)

    @staticmethod
    def _world_linear_velocity_to_body(linear_world, orientation):
        yaw = LocalBaseDriver._yaw_from_wxyz(orientation)
        c, s = math.cos(yaw), math.sin(yaw)
        vx, vy, vz = np.asarray(linear_world, dtype=np.float32).reshape(3)
        return np.asarray([c * vx + s * vy, -s * vx + c * vy, vz], dtype=np.float32)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))


def build_local_base_driver(robot, *, world=None):
    """Build the body-twist executor selected by the robot base profile."""
    base_cfg = robot.get_base_interface()["base_cfg"]
    platform_cfg = base_cfg.get("platform", {})
    profile = str(platform_cfg.get("profile", "")).strip().lower().replace("-", "_")
    if profile == "unitree_g1_decoupled_wbc":
        from .g1_locomotion_driver import G1LocomotionDriver

        return G1LocomotionDriver(robot, world=world)
    return LocalBaseDriver(robot, world=world)
