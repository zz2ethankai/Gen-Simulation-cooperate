"""Debug helpers and runtime result models for the split Nav2 runtime."""

from __future__ import annotations

from dataclasses import dataclass


def runtime_control_debug_snapshot(robot) -> dict:
    snapshot = {}

    bridge = getattr(robot, "_simbox_ros_base_bridge", None)
    if bridge is not None:
        requested_steering = getattr(bridge, "_last_requested_steering", None)
        requested_wheel_velocities = getattr(bridge, "_last_requested_wheel_velocities", None)
        bridge_info = {
            "received_cmd_vel_count": int(getattr(bridge, "_received_cmd_vel_count", 0)),
            "driver_command_message_count": int(getattr(bridge, "_driver_command_message_count", 0)),
            "pending_driver_command_count": int(getattr(bridge, "_pending_driver_command_count", 0)),
            "applied_driver_command_count": int(getattr(bridge, "_applied_driver_command_count", 0)),
            "last_received_cmd_vel": dict(getattr(bridge, "_last_received_cmd_vel", {}) or {}),
            "recent_cmd_vel_history": list(getattr(bridge, "_debug_cmd_vel_history", []))[-20:],
            "steering_command_sign": float(getattr(bridge, "_steering_command_sign", 1.0)),
            "virtual_odom_enabled": False,
            "last_requested_steering": [float(value) for value in list(requested_steering)]
            if requested_steering is not None
            else [],
            "last_requested_wheel_velocities": [float(value) for value in list(requested_wheel_velocities)]
            if requested_wheel_velocities is not None
            else [],
            "last_published_pose": dict(getattr(bridge, "_last_published_pose_debug", {}) or {}),
            "recent_command_history": list(getattr(bridge, "_debug_command_history", []))[-20:],
        }
        active_command = getattr(bridge, "_command", None)
        if active_command is not None:
            bridge_info["active_command"] = {
                "vx_body": float(getattr(active_command, "vx_body", 0.0)),
                "vy_body": float(getattr(active_command, "vy_body", 0.0)),
                "wz_body": float(getattr(active_command, "wz_body", 0.0)),
                "received_time_sec": float(getattr(active_command, "received_time_sec", 0.0)),
            }
        snapshot["bridge"] = bridge_info

    return snapshot


class TaskShim:
    def __init__(self, task):
        self.task = task


@dataclass
class Nav2SkillResult:
    done: bool = False
    success: bool = False
    failure_reason: str = ""
    error_message: str = ""
    final_world_xy: tuple[float, float] = (0.0, 0.0)
    final_world_yaw: float = 0.0
    final_nav_xy: tuple[float, float] = (0.0, 0.0)
    final_nav_yaw: float = 0.0
    final_distance_to_goal: float = float("inf")
    final_nav_distance_to_goal: float = float("inf")
    final_yaw_error_rad: float = float("inf")
