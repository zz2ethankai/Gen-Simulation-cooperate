"""Unitree G1 decoupled-WBC executor for the local navigation skill."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import time

import numpy as np

from .g1_decoupled_wbc import G1DecoupledWbcPolicy, MotorCommand


@dataclass(frozen=True)
class G1NavigationCommand:
    """One bounded body command and its decoupled-WBC yaw target."""

    vx_body: float
    vy_body: float
    turn_flag: float
    target_yaw: float

    @property
    def navigate_cmd(self) -> np.ndarray:
        return np.asarray(
            [self.vx_body, self.vy_body, self.turn_flag, self.target_yaw],
            dtype=np.float64,
        )


class G1NavigationCommandAdapter:
    """Translate a waypoint twist into one bounded G1 motion primitive."""

    def __init__(
        self,
        *,
        hard_limits: dict,
        yaw_target_horizon_sec: float = 0.5,
        turn_deadband_rad_per_sec: float = 0.01,
        walk_heading_tolerance_rad: float = 0.2,
        final_heading_linear_speed_threshold: float = 0.24,
    ):
        self._min_velocity = self._vector(hard_limits, "min_velocity")
        self._max_velocity = self._vector(hard_limits, "max_velocity")
        if np.any(self._min_velocity > self._max_velocity):
            raise ValueError("G1 min_velocity must not exceed max_velocity")
        for name in ("max_accel", "max_decel"):
            if name in hard_limits:
                self._vector(hard_limits, name)
        self._yaw_target_horizon_sec = float(yaw_target_horizon_sec)
        self._turn_deadband = float(turn_deadband_rad_per_sec)
        self._walk_heading_tolerance = float(walk_heading_tolerance_rad)
        self._final_heading_linear_speed_threshold = float(
            final_heading_linear_speed_threshold
        )
        values = (
            self._yaw_target_horizon_sec,
            self._turn_deadband,
            self._walk_heading_tolerance,
            self._final_heading_linear_speed_threshold,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("G1 command-adapter thresholds must be finite and non-negative")
        if self._yaw_target_horizon_sec == 0.0:
            raise ValueError("yaw_target_horizon_sec must be positive")
        self._target_yaw: float | None = None

    @staticmethod
    def _vector(config: dict, name: str) -> np.ndarray:
        values = np.asarray(config.get(name), dtype=np.float64).reshape(-1)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"G1 controller_hard_limits.{name} must contain three finite values"
            )
        return values

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def reset(self, *, current_yaw: float | None = None) -> None:
        if current_yaw is None:
            self._target_yaw = None
            return
        if not math.isfinite(float(current_yaw)):
            raise ValueError("current_yaw must be finite")
        self._target_yaw = self._wrap_angle(float(current_yaw))

    def translate(
        self,
        *,
        vx_body: float,
        vy_body: float,
        wz_body: float,
        current_yaw: float,
    ) -> G1NavigationCommand:
        requested = np.asarray([vx_body, vy_body, wz_body], dtype=np.float64)
        if not np.all(np.isfinite(requested)) or not math.isfinite(float(current_yaw)):
            raise ValueError("G1 navigation twist and current_yaw must be finite")
        if self._target_yaw is None:
            self._target_yaw = self._wrap_angle(float(current_yaw))

        translation_speed = float(np.linalg.norm(requested[:2]))
        bounded_wz = float(
            np.clip(requested[2], self._min_velocity[2], self._max_velocity[2])
        )
        final_heading_phase = (
            translation_speed <= self._final_heading_linear_speed_threshold
            and abs(bounded_wz) > self._turn_deadband
        )
        pure_lateral = abs(requested[0]) <= 1.0e-6 and abs(requested[1]) > 1.0e-6
        if pure_lateral and not final_heading_phase:
            return G1NavigationCommand(
                0.0,
                float(np.clip(requested[1], self._min_velocity[1], self._max_velocity[1])),
                0.0,
                self._target_yaw,
            )
        if translation_speed > 1.0e-6 and not final_heading_phase:
            heading_error = math.atan2(float(requested[1]), float(requested[0]))
            turning = abs(heading_error) > self._walk_heading_tolerance
            if turning:
                heading_rate = float(
                    np.clip(
                        heading_error / self._yaw_target_horizon_sec,
                        self._min_velocity[2],
                        self._max_velocity[2],
                    )
                )
                self._target_yaw = self._wrap_angle(
                    float(current_yaw) + heading_rate * self._yaw_target_horizon_sec
                )
                forward_speed = 0.0
            else:
                forward_speed = float(
                    np.clip(translation_speed, 0.0, self._max_velocity[0])
                )
            return G1NavigationCommand(
                forward_speed,
                0.0,
                float(turning),
                self._target_yaw,
            )

        turning = abs(bounded_wz) > self._turn_deadband
        if turning:
            self._target_yaw = self._wrap_angle(
                float(current_yaw) + bounded_wz * self._yaw_target_horizon_sec
            )
        return G1NavigationCommand(0.0, 0.0, float(turning), self._target_yaw)


@dataclass(frozen=True)
class G1BaseCommand:
    vx_body: float
    vy_body: float
    wz_body: float
    received_time_sec: float

    @classmethod
    def zero(cls, *, received_time_sec: float) -> "G1BaseCommand":
        return cls(0.0, 0.0, 0.0, float(received_time_sec))


class G1LocomotionDriver:
    """Translate Navigate body twists into HUMANO-style GEAR-WBC targets."""

    preserve_reset_warmup_state = True

    def __init__(self, robot, *, world=None, policy=None, command_adapter=None):
        self.robot = robot
        self.world = world
        self.base_cfg = robot.get_base_interface()["base_cfg"]
        self._command_timeout = float(self.base_cfg.get("command_timeout", 0.25))
        platform_cfg = self.base_cfg.get("platform", {})
        navigation_cfg = platform_cfg.get("local_navigation", {})
        wbc_cfg = platform_cfg.get("decoupled_wbc", {})
        self.required_reset_warmup_steps = int(wbc_cfg.get("bootstrap_steps", 600))
        if self.required_reset_warmup_steps < 1:
            raise ValueError("decoupled_wbc.bootstrap_steps must be positive")
        self.required_navigation_warmup_steps = int(
            wbc_cfg.get("navigation_warmup_steps", 50)
        )
        if self.required_navigation_warmup_steps < 1:
            raise ValueError("decoupled_wbc.navigation_warmup_steps must be positive")
        if policy is None:
            resource_root = Path(str(wbc_cfg.get("resource_root", ""))).expanduser()
            if not resource_root.is_absolute():
                resource_root = Path.cwd() / resource_root
            policy = G1DecoupledWbcPolicy(
                resource_root / str(wbc_cfg.get("balance_model", "GR00T-WholeBodyControl-Balance.onnx")),
                resource_root / str(wbc_cfg.get("walk_model", "GR00T-WholeBodyControl-Walk.onnx")),
                control_dt=float(wbc_cfg.get("control_dt", 0.02)),
            )
        if command_adapter is None:
            command_adapter = G1NavigationCommandAdapter(
                hard_limits=navigation_cfg.get("controller_hard_limits", {}),
                yaw_target_horizon_sec=float(wbc_cfg.get("yaw_target_horizon_sec", 0.5)),
                walk_heading_tolerance_rad=float(
                    wbc_cfg.get("walk_heading_tolerance_rad", 0.2)
                ),
                final_heading_linear_speed_threshold=float(
                    wbc_cfg.get("final_heading_linear_speed_threshold", 0.24)
                ),
            )
        self._policy = policy
        self._command_adapter = command_adapter
        now = self._now_sec()
        self._command = G1BaseCommand.zero(received_time_sec=now)
        self._navigation_active = False
        self._has_command = False
        self._driver_command_message_count = 0
        self._applied_driver_command_count = 0
        self._debug_command_history = deque(
            maxlen=max(int(self.base_cfg.get("debug_history_size", 64)), 1)
        )
        self._last_motor_command = MotorCommand()
        self._last_joint_efforts = np.zeros(29, dtype=np.float64)
        self._last_navigate_cmd = np.zeros(4, dtype=np.float64)
        self._last_step_time_sec = now
        self._last_physics_step_index = self._world_physics_step_index()
        self._last_step_dt = 1.0e-3
        self._last_pose = self._get_robot_base_pose()
        self._actual_twist = np.zeros(3, dtype=np.float64)

    def _world_time_sec(self) -> float | None:
        if self.world is None:
            return None
        for name in ("current_time", "get_current_time"):
            value = getattr(self.world, name, None)
            try:
                return float(value() if callable(value) else value)
            except (TypeError, ValueError):
                continue
        return None

    def _now_sec(self) -> float:
        world_time = self._world_time_sec()
        return time.monotonic() if world_time is None else world_time

    def _world_physics_step_index(self) -> int | None:
        if self.world is None:
            return None
        value = getattr(self.world, "current_time_step_index", None)
        try:
            return int(value() if callable(value) else value)
        except (TypeError, ValueError):
            return None

    def _world_physics_dt(self) -> float | None:
        if self.world is None:
            return None
        value = getattr(self.world, "get_physics_dt", None)
        try:
            physics_dt = float(value() if callable(value) else value)
        except (TypeError, ValueError):
            return None
        return physics_dt if physics_dt > 0.0 else None

    def _get_robot_base_pose(self):
        getter = getattr(self.robot, "get_nav_base_pose", None)
        if not callable(getter):
            raise ValueError("Unitree G1 robot must provide get_nav_base_pose")
        position, orientation = getter()
        return (
            np.asarray(position, dtype=np.float64).reshape(3).copy(),
            np.asarray(orientation, dtype=np.float64).reshape(4).copy(),
        )

    @staticmethod
    def _yaw_from_wxyz(quaternion) -> float:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def set_command(self, vx_body: float, vy_body: float, wz_body: float) -> None:
        values = np.asarray([vx_body, vy_body, wz_body], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Humanoid navigation command must be finite")
        self._command = G1BaseCommand(*map(float, values), received_time_sec=self._now_sec())
        self._has_command = True
        self._driver_command_message_count += 1

    def prepare_for_navigation(self) -> None:
        self._reset_command_adapter()
        self._navigation_active = True
        self._has_command = False
        self.set_command(0.0, 0.0, 0.0)

    def finalize_after_navigation(self) -> None:
        self._navigation_active = False
        self._has_command = False
        self._command = G1BaseCommand.zero(received_time_sec=self._now_sec())

    def reset(self, *, clear_debug_history: bool = False) -> None:
        self.finalize_after_navigation()
        self._policy.reset()
        self._reset_command_adapter()
        self._last_physics_step_index = self._world_physics_step_index()
        self._last_pose = self._get_robot_base_pose()
        self._last_navigate_cmd.fill(0.0)
        self._actual_twist.fill(0.0)
        if clear_debug_history:
            self._driver_command_message_count = 0
            self._applied_driver_command_count = 0
            self._debug_command_history.clear()

    def finish_reset_warmup(self, *, clear_debug_history: bool = False) -> None:
        """Keep stabilized WBC state while closing the reset warmup phase."""

        self.finalize_after_navigation()
        now = self._now_sec()
        self._last_step_time_sec = now
        self._last_physics_step_index = self._world_physics_step_index()
        self._last_step_dt = 1.0e-3
        self._last_pose = self._get_robot_base_pose()
        self._last_navigate_cmd.fill(0.0)
        self._actual_twist.fill(0.0)
        if clear_debug_history:
            self._driver_command_message_count = 0
            self._applied_driver_command_count = 0
            self._debug_command_history.clear()

    def _reset_command_adapter(self) -> None:
        reset = getattr(self._command_adapter, "reset", None)
        if not callable(reset):
            return
        _position, orientation = self._get_robot_base_pose()
        reset(current_yaw=self._yaw_from_wxyz(orientation))

    def _active_command(self, now: float) -> G1BaseCommand:
        if (
            self._navigation_active
            and self._has_command
            and now - self._command.received_time_sec <= self._command_timeout
        ):
            return self._command
        return G1BaseCommand.zero(received_time_sec=now)

    def step(self, step_dt: float | None = None) -> None:
        now = self._now_sec()
        physics_step_index = self._world_physics_step_index()
        elapsed_physics_steps = (
            physics_step_index - self._last_physics_step_index
            if physics_step_index is not None
            and self._last_physics_step_index is not None
            else None
        )
        physics_dt = self._world_physics_dt()
        if (
            elapsed_physics_steps is not None
            and elapsed_physics_steps > 0
            and physics_dt
        ):
            # The workflow calls this driver immediately before world.step(). A
            # render=True step advances one 50 Hz render interval by running four
            # internal 200 Hz physics ticks, holding this WBC target across them.
            # On the next driver call, the physics-step delta is therefore four;
            # render=False advances one tick. Convert that exact tick count back
            # to seconds because the WBC rate gate is configured with control_dt.
            dt = elapsed_physics_steps * physics_dt
        elif step_dt is not None:
            # First call after initialization/reset has no positive step delta.
            # Keep the original workflow contract and use its physics_dt once.
            dt = float(step_dt)
        else:
            dt = now - self._last_step_time_sec
        dt = max(dt, 1.0e-3)
        self._last_step_time_sec = now
        self._last_physics_step_index = physics_step_index
        self._last_step_dt = dt
        command = self._active_command(now)
        _position, orientation = self._get_robot_base_pose()
        translated = self._command_adapter.translate(
            vx_body=command.vx_body,
            vy_body=command.vy_body,
            wz_body=command.wz_body,
            current_yaw=self._yaw_from_wxyz(orientation),
        )
        state = self.robot.get_locomotion_state()
        motor_command = self._policy.step(state, translated.navigate_cmd, env_step_dt=dt)
        applied_efforts = self.robot.apply_locomotion_command(motor_command)
        self._last_motor_command = motor_command
        self._last_joint_efforts = (
            np.asarray(applied_efforts, dtype=np.float64).reshape(29)
            if applied_efforts is not None
            else np.asarray(motor_command.tau_ff, dtype=np.float64).reshape(29)
        )
        self._last_navigate_cmd = translated.navigate_cmd
        self._applied_driver_command_count += int(self._navigation_active)
        self._update_actual_twist(dt)
        self._debug_command_history.append(
            {
                "time_sec": now,
                "dt": dt,
                "command": {
                    "vx_body": command.vx_body,
                    "vy_body": command.vy_body,
                    "wz_body": command.wz_body,
                    "wbc_mode": self._policy.last_mode,
                    "wbc_navigate_cmd": translated.navigate_cmd.tolist(),
                },
            }
        )

    def _update_actual_twist(self, dt: float) -> None:
        position, orientation = self._get_robot_base_pose()
        previous_position, previous_orientation = self._last_pose
        previous_yaw = self._yaw_from_wxyz(previous_orientation)
        current_yaw = self._yaw_from_wxyz(orientation)
        delta_world = (position[:2] - previous_position[:2]) / dt
        cos_yaw, sin_yaw = math.cos(previous_yaw), math.sin(previous_yaw)
        self._actual_twist = np.asarray(
            [
                cos_yaw * delta_world[0] + sin_yaw * delta_world[1],
                -sin_yaw * delta_world[0] + cos_yaw * delta_world[1],
                self._wrap_angle(current_yaw - previous_yaw) / dt,
            ],
            dtype=np.float64,
        )
        self._last_pose = (position, orientation)

    def get_actual_twist_body(self):
        return tuple(map(float, self._actual_twist))

    def get_logging_action_snapshot(self):
        command = self._command
        return {
            "vx_body": float(command.vx_body),
            "vy_body": float(command.vy_body),
            "wz_body": float(command.wz_body),
            "locomotion_mode": 1 if self._policy.last_mode == "walk" else 0,
            "wbc_navigate_cmd": self._last_navigate_cmd.tolist(),
            "joint_position_targets": self._last_motor_command.q_target.tolist(),
            "joint_velocity_targets": self._last_motor_command.dq_target.tolist(),
            "joint_efforts": self._last_joint_efforts.tolist(),
            "driver_command_message_count": self._driver_command_message_count,
            "applied_driver_command_count": self._applied_driver_command_count,
        }

    def get_logging_state_snapshot(self):
        position, orientation = self._get_robot_base_pose()
        return {
            "actual_vx_body": float(self._actual_twist[0]),
            "actual_vy_body": float(self._actual_twist[1]),
            "actual_wz_body": float(self._actual_twist[2]),
            "base_position": position.tolist(),
            "base_orientation": orientation.tolist(),
            "pelvis_z": float(position[2]),
            "navigation_active": bool(self._navigation_active),
            "wbc_inference_count": int(self._policy.inference_count),
        }
