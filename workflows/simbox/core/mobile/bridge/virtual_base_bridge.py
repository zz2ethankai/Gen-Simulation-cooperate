"""Direct /cmd_vel bridge for velocity-target virtual X/Y/yaw base joints."""

from __future__ import annotations

import numpy as np

from .base_bridge import BaseBridge
from .types import BaseCommand


class VirtualBaseBridge(BaseBridge):
    """Map body-frame twist directly to virtual base joint velocity targets."""

    def __init__(self, robot, node_name: str = "virtual_base_bridge", driver=None):
        self._body_twist_deadband = 0.0
        self._base_velocity_signs = np.ones(3, dtype=np.float32)
        super().__init__(robot=robot, node_name=node_name, driver=driver)

    def _validate_bridge_configuration(self, *, steering_count: int, wheel_count: int):
        if steering_count != 0:
            raise ValueError(f"VirtualBaseBridge expects 0 steering joints, got {steering_count}")
        if wheel_count != 3:
            raise ValueError(f"VirtualBaseBridge expects exactly 3 base velocity joints, got {wheel_count}")

        wheel_names = list(self.base_interface["wheel_joint_names"])
        velocity_names = list(self.base_cfg["base_velocity_joint_names"])
        if velocity_names != wheel_names:
            raise ValueError("base_velocity_joint_names must exactly match wheel_joint_names order")

        self._body_twist_deadband = max(float(self.base_cfg["body_twist_deadband"]), 0.0)
        signs = np.asarray(self.base_cfg["base_velocity_command_signs"], dtype=np.float32).reshape(-1)
        if signs.size != wheel_count:
            raise ValueError("base_velocity_command_signs must match base_velocity_joint_names length")
        if not np.all(np.isfinite(signs)) or np.any(np.abs(signs) <= 1.0e-6):
            raise ValueError("base_velocity_command_signs must be finite non-zero values")
        self._base_velocity_signs = signs.astype(np.float32)

    def _map_command(self, command: BaseCommand) -> tuple[np.ndarray, np.ndarray]:
        body_velocity = np.asarray(
            [command.vx_body, command.vy_body, command.wz_body],
            dtype=np.float32,
        )
        body_velocity[np.abs(body_velocity) <= self._body_twist_deadband] = 0.0
        body_velocity = np.clip(body_velocity, self._min_body_velocity, self._max_body_velocity)
        base_joint_velocities = body_velocity * self._base_velocity_signs
        self._last_wheel_shaping_debug = {
            "mode": "virtual_base_velocity_target",
            "body_velocity": [float(v) for v in body_velocity.tolist()],
            "base_velocity_command_signs": [float(v) for v in self._base_velocity_signs.tolist()],
        }
        return np.zeros(0, dtype=np.float32), base_joint_velocities.astype(np.float32)

    def _apply_robot_base_command(
        self,
        *,
        steering_positions: np.ndarray,
        wheel_velocities: np.ndarray,
        step_dt: float,
    ) -> None:
        self.robot.apply_base_command(
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
            step_dt=step_dt,
        )
