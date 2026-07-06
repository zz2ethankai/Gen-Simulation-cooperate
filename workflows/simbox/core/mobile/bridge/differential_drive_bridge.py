"""Differential-drive mobile base bridge for direct wheel-speed control."""

from __future__ import annotations

import numpy as np

from .base_bridge import BaseBridge
from .types import BaseCommand


class DifferentialDriveBridge(BaseBridge):
    """Map body-frame /cmd_vel to left/right wheel angular velocities."""

    def __init__(self, robot, node_name: str = "differential_drive_bridge", driver=None):
        self._left_wheel_indices = []
        self._right_wheel_indices = []
        self._body_twist_deadband = 0.0
        self._wheel_velocity_signs = np.ones(0, dtype=np.float32)
        super().__init__(robot=robot, node_name=node_name, driver=driver)

    def _validate_bridge_configuration(self, *, steering_count: int, wheel_count: int):
        if steering_count != 0:
            raise ValueError(f"DifferentialDriveBridge expects 0 steering joints, got {steering_count}")
        if wheel_count != 2:
            raise ValueError(f"DifferentialDriveBridge expects exactly 2 wheel joints, got {wheel_count}")

        wheel_names = list(self.base_interface["wheel_joint_names"])
        left_names = list(self.base_cfg["left_wheel_joint_names"])
        right_names = list(self.base_cfg["right_wheel_joint_names"])
        if len(left_names) != 1 or len(right_names) != 1:
            raise KeyError(
                "DifferentialDriveBridge requires exactly one left and one right wheel joint"
            )

        self._left_wheel_indices = [wheel_names.index(name) for name in left_names]
        self._right_wheel_indices = [wheel_names.index(name) for name in right_names]
        if sorted(self._left_wheel_indices + self._right_wheel_indices) != list(range(wheel_count)):
            raise ValueError("Left/right wheel joint names must partition wheel_joint_names")

        if self._track_width <= 0.0:
            raise ValueError("track_width must be positive for differential drive")
        self._body_twist_deadband = max(float(self.base_cfg["body_twist_deadband"]), 0.0)
        signs = np.asarray(self.base_cfg["wheel_velocity_command_signs"], dtype=np.float32)
        if signs.size != wheel_count:
            raise ValueError("wheel_velocity_command_signs must match wheel_joint_names length")
        if not np.all(np.isfinite(signs)) or np.any(np.abs(signs) <= 1.0e-6):
            raise ValueError("wheel_velocity_command_signs must be finite non-zero values")
        self._wheel_velocity_signs = signs.astype(np.float32)

    def _map_command(self, command: BaseCommand) -> tuple[np.ndarray, np.ndarray]:
        vx_body = float(command.vx_body)
        vy_body = float(command.vy_body)
        wz_body = float(command.wz_body)
        if abs(vx_body) <= self._body_twist_deadband:
            vx_body = 0.0
        if abs(vy_body) > self._body_twist_deadband:
            raise ValueError(f"DifferentialDriveBridge cannot execute lateral body velocity vy={vy_body}")
        if abs(wz_body) <= self._body_twist_deadband:
            wz_body = 0.0
        if vx_body == 0.0 and wz_body == 0.0:
            return np.zeros(0, dtype=np.float32), np.zeros(len(self.base_interface["wheel_joint_names"]), dtype=np.float32)

        left_linear = vx_body - 0.5 * wz_body * self._track_width
        right_linear = vx_body + 0.5 * wz_body * self._track_width
        wheel_velocities = np.zeros(len(self.base_interface["wheel_joint_names"]), dtype=np.float32)
        for index in self._left_wheel_indices:
            wheel_velocities[index] = left_linear / self._wheel_radius
        for index in self._right_wheel_indices:
            wheel_velocities[index] = right_linear / self._wheel_radius

        wheel_velocities *= self._wheel_velocity_signs
        peak_wheel_velocity = float(np.max(np.abs(wheel_velocities))) if wheel_velocities.size else 0.0
        if peak_wheel_velocity > self._wheel_velocity_limit > 0.0:
            wheel_velocities *= float(self._wheel_velocity_limit / peak_wheel_velocity)
        return np.zeros(0, dtype=np.float32), wheel_velocities.astype(np.float32)
