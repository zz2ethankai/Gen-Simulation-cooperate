"""Debug helpers and runtime result models for the split Nav2 runtime."""

from __future__ import annotations

from dataclasses import dataclass


def runtime_control_debug_snapshot(robot) -> dict:
    snapshot = {}

    bridge = getattr(robot, "_simbox_ros_base_bridge", None)
    if bridge is not None:
        requested_steering = getattr(bridge, "_last_requested_steering", None)
        requested_wheel_velocities = getattr(bridge, "_last_requested_wheel_velocities", None)
        applied_steering = getattr(bridge, "_last_applied_steering", None)
        applied_wheel_velocities = getattr(bridge, "_last_applied_wheel_velocities", None)
        restore_target_steering = getattr(bridge, "_restore_target_steering", None)
        bridge_info = {
            "navigation_active": bool(getattr(bridge, "_navigation_active", False)),
            "hold_after_navigation": bool(getattr(bridge, "_hold_after_navigation", False)),
            "non_finite_state_detected": bool(getattr(bridge, "_non_finite_state_detected", False)),
            "non_finite_state_reason": str(getattr(bridge, "_non_finite_state_reason", "")),
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
            "last_applied_steering": [float(value) for value in list(applied_steering)]
            if applied_steering is not None
            else [],
            "last_applied_wheel_velocities": [float(value) for value in list(applied_wheel_velocities)]
            if applied_wheel_velocities is not None
            else [],
            "restore_after_navigation": {
                "active": bool(getattr(bridge, "_restore_after_navigation", False)),
                "target_steering": [float(value) for value in list(restore_target_steering)]
                if restore_target_steering is not None
                else [],
                "steering_rate_limit": float(getattr(bridge, "_steering_rate_limit", 0.0)),
                "wheel_velocity_limit": float(getattr(bridge, "_wheel_velocity_limit", 0.0)),
            },
            "last_published_pose": dict(getattr(bridge, "_last_published_pose_debug", {}) or {}),
            "recent_command_history": list(getattr(bridge, "_debug_command_history", []))[-20:],
        }
        action_snapshot_fn = getattr(bridge, "get_logging_action_snapshot", None)
        if callable(action_snapshot_fn):
            try:
                action_snapshot = action_snapshot_fn()
                bridge_info["action_snapshot"] = action_snapshot
                if isinstance(action_snapshot, dict) and "joint_mapping" in action_snapshot:
                    bridge_info["joint_mapping"] = action_snapshot["joint_mapping"]
            except Exception as exc:  # pylint: disable=broad-except
                bridge_info["action_snapshot_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
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

    view = getattr(robot, "_articulation_view", None)
    dof_names = list(getattr(view, "dof_names", []) or [])
    base_indices = list(getattr(robot, "base_steering_joint_indices", []) or []) + list(
        getattr(robot, "base_wheel_joint_indices", []) or []
    )
    arm_indices = list(getattr(robot, "left_joint_indices", []) or []) + list(
        getattr(robot, "right_joint_indices", []) or []
    )
    gripper_indices = list(getattr(robot, "left_gripper_indices", []) or []) + list(
        getattr(robot, "right_gripper_indices", []) or []
    )
    if dof_names or base_indices or arm_indices or gripper_indices:
        snapshot["robot_joint_mapping"] = {
            "articulation_dof_names": dof_names,
            "base_steering_joint_names": list(getattr(robot, "base_steering_joint_names", []) or []),
            "base_wheel_joint_names": list(getattr(robot, "base_wheel_joint_names", []) or []),
            "base_steering_joint_indices": [
                int(v) for v in list(getattr(robot, "base_steering_joint_indices", []) or [])
            ],
            "base_wheel_joint_indices": [
                int(v) for v in list(getattr(robot, "base_wheel_joint_indices", []) or [])
            ],
            "left_joint_indices": [int(v) for v in list(getattr(robot, "left_joint_indices", []) or [])],
            "right_joint_indices": [int(v) for v in list(getattr(robot, "right_joint_indices", []) or [])],
            "left_gripper_indices": [int(v) for v in list(getattr(robot, "left_gripper_indices", []) or [])],
            "right_gripper_indices": [int(v) for v in list(getattr(robot, "right_gripper_indices", []) or [])],
            "base_manipulator_index_overlap": sorted(
                {int(v) for v in base_indices} & {int(v) for v in arm_indices + gripper_indices}
            ),
        }

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
