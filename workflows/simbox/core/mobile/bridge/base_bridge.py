"""移动底盘 Isaac Bridge 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import math

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from .types import BaseCommand


class BaseBridge(ABC):
    """将标准 /cmd_vel 直接桥接为 Isaac articulation 控制。"""

    def __init__(self, robot, node_name: str, driver=None):
        if driver is not None:
            raise ValueError("Internal driver translation is disabled; publish /cmd_vel directly to Isaac bridge.")

        self.robot = robot
        self.base_interface = robot.get_base_interface()
        self.base_cfg = self.base_interface["base_cfg"]
        self.ros_cfg = self.base_cfg["ros"]

        required_ros_fields = ["cmd_vel_topic", "joint_state_topic", "odom_topic", "base_frame", "odom_frame"]
        missing_fields = [field for field in required_ros_fields if field not in self.ros_cfg]
        if missing_fields:
            raise KeyError(f"Missing ROS base bridge config fields: {missing_fields}")

        virtual_odom_cfg = self.ros_cfg.get("virtual_odom", {})
        if not isinstance(virtual_odom_cfg, dict):
            raise TypeError("ros.virtual_odom must be a dict when present")
        if bool(virtual_odom_cfg.get("enabled", False)):
            raise ValueError("virtual_odom is not supported for the direct /cmd_vel 4WIS bridge")

        self._command_timeout = float(self.base_cfg["command_timeout"])
        self._steering_limit = float(self.base_cfg["steering_limit"])
        self._steering_rate_limit = float(self.base_cfg["steering_rate_limit"])
        self._wheel_velocity_limit = float(self.base_cfg["wheel_velocity_limit"])
        self._wheel_base = float(self.base_cfg["wheel_base"])
        self._track_width = float(self.base_cfg["track_width"])
        self._wheel_radius = float(self.base_cfg["wheel_radius"])
        self._steering_command_sign = float(
            self.base_cfg.get("steering_command_sign", self.ros_cfg.get("steering_command_sign", 1.0))
        )
        self._min_body_velocity, self._max_body_velocity = self._load_body_velocity_limits()
        if abs(self._steering_command_sign) <= 1.0e-6:
            raise ValueError("steering_command_sign must be non-zero")
        if self._wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")

        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        self._validate_bridge_configuration(steering_count=steering_count, wheel_count=wheel_count)

        self._owns_rclpy_context = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        self.node = Node(node_name)
        self._tf_pub = self.node.create_publisher(TFMessage, "/tf", 10)
        self._joint_state_pub = self.node.create_publisher(JointState, self.ros_cfg["joint_state_topic"], 10)
        self._odom_pub = self.node.create_publisher(Odometry, self.ros_cfg["odom_topic"], 10)
        self._cmd_vel_sub = self.node.create_subscription(
            Twist,
            self.ros_cfg["cmd_vel_topic"],
            self._on_cmd_vel,
            10,
        )

        now_sec = self._now_sec()
        self._command = BaseCommand.zero(received_time_sec=now_sec)
        self._last_step_command = BaseCommand.zero(received_time_sec=now_sec)
        self._last_applied_steering = np.zeros(steering_count, dtype=np.float32)
        self._last_requested_steering = np.zeros(steering_count, dtype=np.float32)
        self._last_requested_wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        self._last_applied_wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        self._last_wheel_shaping_debug = {}
        self._last_step_time_sec = now_sec
        self._last_step_dt = 1e-3
        self._navigation_active = False
        self._hold_after_navigation = True
        self._restore_after_navigation = False
        self._restore_waiting_for_wheel_stop = False
        self._restore_target_steering = np.zeros(steering_count, dtype=np.float32)
        self._restore_done_tolerance = float(self.base_cfg.get("restore_done_tolerance", 5.0e-3))
        self._restore_done_velocity_tolerance = float(
            self.base_cfg.get("restore_done_velocity_tolerance", 0.5)
        )
        self._heading_gate_enabled = False
        self._heading_gate_target_yaw = 0.0
        self._heading_gate_tolerance_rad = 0.0
        self._heading_gate_rotate_vel = 0.0

        history_size = max(int(self.base_cfg.get("debug_history_size", 256)), 1)
        self._received_cmd_vel_count = 0
        self._driver_command_message_count = 0
        self._pending_driver_command_count = 0
        self._applied_driver_command_count = 0
        self._motion_mode_message_count = 0
        self._has_motion_mode = False
        self._latest_motion_mode = 0
        self._last_received_cmd_vel = {
            "linear_x": 0.0,
            "linear_y": 0.0,
            "angular_z": 0.0,
            "received_time_sec": float(now_sec),
        }
        self._debug_cmd_vel_history = deque(maxlen=history_size)
        self._debug_command_history = deque(maxlen=history_size)
        self._last_published_pose_debug = {}
        self._non_finite_state_detected = False
        self._non_finite_state_reason = ""

        translation, orientation = self._get_robot_base_pose()
        self._last_actual_translation = np.array(translation, dtype=np.float32)
        self._last_actual_yaw = float(self._yaw_from_wxyz(orientation))
        self._last_actual_linear_velocity_world = np.zeros(3, dtype=np.float32)
        self._last_actual_angular_velocity_world = np.zeros(3, dtype=np.float32)

    @abstractmethod
    def _validate_bridge_configuration(self, *, steering_count: int, wheel_count: int):
        """校验桥接器与目标底盘的配置是否匹配。"""

    @abstractmethod
    def _map_command(self, command: BaseCommand) -> tuple[np.ndarray, np.ndarray]:
        """将车体级命令映射为关节转向角和轮角速度。"""

    def destroy(self):
        self.node.destroy_node()
        if self._owns_rclpy_context and rclpy.ok():
            rclpy.shutdown()

    def reset(self, *, clear_debug_history: bool = False):
        self._spin_available_callbacks()
        now_sec = self._now_sec()
        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        zero_command = BaseCommand.zero(received_time_sec=now_sec)
        zero_steering = np.zeros(steering_count, dtype=np.float32)
        zero_wheel = np.zeros(wheel_count, dtype=np.float32)

        self._command = zero_command
        self._last_step_command = zero_command
        current_steering = self._get_current_steering_positions(default=zero_steering)
        self._last_applied_steering = current_steering.copy()
        self._last_requested_steering = current_steering.copy()
        self._last_requested_wheel_velocities = zero_wheel.copy()
        self._last_applied_wheel_velocities = zero_wheel.copy()
        self._last_wheel_shaping_debug = {}
        self._last_step_time_sec = now_sec
        self._last_step_dt = 1e-3
        self._navigation_active = False
        self._hold_after_navigation = True
        self._restore_after_navigation = False
        self._restore_waiting_for_wheel_stop = False
        self._last_received_cmd_vel = {
            "linear_x": 0.0,
            "linear_y": 0.0,
            "angular_z": 0.0,
            "received_time_sec": float(now_sec),
        }
        self._heading_gate_enabled = False
        self._non_finite_state_detected = False
        self._non_finite_state_reason = ""

        if clear_debug_history:
            self._received_cmd_vel_count = 0
            self._driver_command_message_count = 0
            self._pending_driver_command_count = 0
            self._applied_driver_command_count = 0
            self._motion_mode_message_count = 0
            self._debug_cmd_vel_history.clear()
            self._debug_command_history.clear()

        self.robot.apply_base_command(
            steering_positions=current_steering,
            wheel_velocities=zero_wheel,
        )
        translation, orientation = self._get_robot_base_pose()
        self._last_actual_translation = np.array(translation, dtype=np.float32)
        self._last_actual_yaw = float(self._yaw_from_wxyz(orientation))
        self._last_actual_linear_velocity_world = np.zeros(3, dtype=np.float32)
        self._last_actual_angular_velocity_world = np.zeros(3, dtype=np.float32)
        self._publish_joint_state()
        self._publish_odometry()
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def prepare_for_navigation(self):
        self._navigation_active = True
        self._hold_after_navigation = False
        self._restore_after_navigation = False
        self._heading_gate_enabled = False

    def finalize_after_navigation(self):
        self._spin_available_callbacks()
        now_sec = self._now_sec()
        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        current_steering = self._get_current_steering_positions(
            default=np.zeros(steering_count, dtype=np.float32)
        )
        self._command = BaseCommand.zero(received_time_sec=now_sec)
        self._last_step_command = self._command
        self._navigation_active = False
        self._hold_after_navigation = True
        self._heading_gate_enabled = False
        self._last_applied_steering = current_steering.copy()
        self._last_requested_wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        self._last_applied_wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        self._last_wheel_shaping_debug = {}
        self._restore_target_steering = self._get_restore_target_steering(steering_count)
        self._restore_after_navigation = True
        self._restore_waiting_for_wheel_stop = True
        self._last_step_time_sec = now_sec
        self._last_step_dt = 1e-3
        self._publish_joint_state()
        self._publish_odometry()
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def step_restore_after_navigation(self, step_dt: float | None = None) -> bool:
        if not self._restore_after_navigation:
            return True
        self._spin_available_callbacks()
        now_sec = self._now_sec()
        if step_dt is None:
            dt = max(now_sec - self._last_step_time_sec, 1e-3)
        else:
            dt = max(float(step_dt), 1e-3)
        self._last_step_time_sec = now_sec
        self._last_step_dt = dt

        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        requested_wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        current_wheel_velocities = self._get_current_wheel_velocities(default=requested_wheel_velocities)
        wheel_stop_reached = bool(
            current_wheel_velocities.size == wheel_count
            and np.all(np.isfinite(current_wheel_velocities))
            and float(np.max(np.abs(current_wheel_velocities))) <= self._restore_done_velocity_tolerance
        )

        if self._restore_waiting_for_wheel_stop and not wheel_stop_reached:
            requested_steering = self._get_current_steering_positions(
                default=self._get_restore_target_steering(steering_count)
            )
            steering_positions = self._apply_steering_limits(requested_steering, dt)
        else:
            self._restore_waiting_for_wheel_stop = False
            requested_steering = self._restore_target_steering.copy()
            steering_positions = self._apply_steering_limits(requested_steering, dt)

        wheel_velocities = requested_wheel_velocities.copy()
        self._last_step_command = BaseCommand.zero(received_time_sec=now_sec)
        self._last_requested_steering = requested_steering.astype(np.float32).copy()
        self._last_requested_wheel_velocities = requested_wheel_velocities.copy()
        self._last_applied_wheel_velocities = wheel_velocities.astype(np.float32).copy()
        self._last_wheel_shaping_debug = {}
        self.robot.apply_base_command(
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
        )
        self._publish_joint_state()
        self._publish_odometry()
        self._record_debug_history(
            command=self._last_step_command,
            requested_steering=requested_steering,
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
            now_sec=now_sec,
            dt=dt,
            mode="restore_after_navigation",
        )
        done = self.restore_after_navigation_done()
        if done:
            self._restore_after_navigation = False
            self._restore_waiting_for_wheel_stop = False
            self._hold_after_navigation = True
        rclpy.spin_once(self.node, timeout_sec=0.0)
        return done

    def restore_after_navigation_done(self) -> bool:
        if not self._restore_after_navigation:
            return True
        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        try:
            joint_state = self.robot.get_base_joint_state()
            current_steering = np.asarray(joint_state["steering_positions"], dtype=np.float32).reshape(-1)
            steering_velocities = np.asarray(joint_state["steering_velocities"], dtype=np.float32).reshape(-1)
            wheel_velocities = np.asarray(joint_state["wheel_velocities"], dtype=np.float32).reshape(-1)
        except Exception:
            current_steering = self._get_current_steering_positions(
                default=self._get_restore_target_steering(steering_count),
            )
            steering_velocities = np.zeros(steering_count, dtype=np.float32)
            wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        target_steering = self._get_restore_target_steering(steering_count)
        if (
            current_steering.size != target_steering.size
            or steering_velocities.size != target_steering.size
            or wheel_velocities.size != wheel_count
        ):
            return False
        if (
            not np.all(np.isfinite(current_steering))
            or not np.all(np.isfinite(steering_velocities))
            or not np.all(np.isfinite(wheel_velocities))
        ):
            return False
        if self._restore_waiting_for_wheel_stop:
            return bool(float(np.max(np.abs(wheel_velocities))) <= self._restore_done_velocity_tolerance)
        return bool(
            np.allclose(current_steering, target_steering, atol=self._restore_done_tolerance)
            and float(np.max(np.abs(steering_velocities))) <= self._restore_done_velocity_tolerance
            and float(np.max(np.abs(wheel_velocities))) <= self._restore_done_velocity_tolerance
        )

    def hold_after_navigation(self, step_dt: float | None = None):
        self._spin_available_callbacks()
        now_sec = self._now_sec()
        if step_dt is None:
            dt = max(now_sec - self._last_step_time_sec, 1e-3)
        else:
            dt = max(float(step_dt), 1e-3)
        self._last_step_time_sec = now_sec
        self._last_step_dt = dt

        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        target_steering = self._get_restore_target_steering(steering_count)
        wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        command = BaseCommand.zero(received_time_sec=now_sec)

        self._command = command
        self._last_step_command = command
        self._last_requested_steering = target_steering.astype(np.float32).copy()
        self._last_requested_wheel_velocities = wheel_velocities.copy()
        self._last_applied_wheel_velocities = wheel_velocities.copy()
        self._last_wheel_shaping_debug = {}
        self.robot.apply_base_command(
            steering_positions=target_steering,
            wheel_velocities=wheel_velocities,
        )
        self._publish_joint_state()
        self._publish_odometry()
        self._record_debug_history(
            command=command,
            requested_steering=target_steering,
            steering_positions=target_steering,
            wheel_velocities=wheel_velocities,
            now_sec=now_sec,
            dt=dt,
            mode="hold_after_navigation",
        )
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def _get_current_wheel_velocities(self, *, default: np.ndarray):
        try:
            joint_state = self.robot.get_base_joint_state()
            current = np.asarray(joint_state["wheel_velocities"], dtype=np.float32).reshape(-1)
        except Exception:
            current = np.asarray(default, dtype=np.float32).reshape(-1)
        expected = len(self.base_interface["wheel_joint_names"])
        if current.size != expected or not np.all(np.isfinite(current)):
            return np.asarray(default, dtype=np.float32).reshape(-1).copy()
        return current.astype(np.float32).copy()

    def start_heading_alignment(
        self,
        *,
        target_x: float,
        target_y: float,
        tolerance_rad: float = 0.12,
        rotate_vel: float = 0.3,
    ):
        if not self._navigation_active:
            self._heading_gate_enabled = False
            return
        translation, _orientation = self._get_robot_base_pose()
        dx = float(target_x) - float(translation[0])
        dy = float(target_y) - float(translation[1])
        if math.hypot(dx, dy) <= 1.0e-6:
            self._heading_gate_enabled = False
            return
        self._heading_gate_target_yaw = math.atan2(dy, dx)
        self._heading_gate_tolerance_rad = max(float(tolerance_rad), 0.0)
        self._heading_gate_rotate_vel = abs(float(rotate_vel))
        self._heading_gate_enabled = self._heading_gate_rotate_vel > 0.0

    def step(self, step_dt: float | None = None):
        if self._restore_after_navigation:
            self.step_restore_after_navigation(step_dt=step_dt)
            return
        if self._hold_after_navigation and not self._navigation_active:
            self.hold_after_navigation(step_dt=step_dt)
            return

        self._spin_available_callbacks()
        now_sec = self._now_sec()
        if step_dt is None:
            dt = max(now_sec - self._last_step_time_sec, 1e-3)
        else:
            dt = max(float(step_dt), 1e-3)
        self._last_step_time_sec = now_sec
        self._last_step_dt = dt
        self._restore_after_navigation = False

        command = self._apply_heading_gate(self._resolve_active_command(now_sec), now_sec)
        self._last_step_command = command
        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        current_steering = self._get_current_steering_positions(
            default=self._get_restore_target_steering(steering_count)
        )
        self._last_applied_steering = current_steering.copy()
        requested_steering, requested_wheel_velocities = self._map_command(command)
        requested_steering = self._as_finite_vector(
            requested_steering,
            expected_size=steering_count,
            fallback=self._last_applied_steering,
        )
        requested_wheel_velocities = self._as_finite_vector(
            requested_wheel_velocities,
            expected_size=wheel_count,
            fallback=np.zeros(wheel_count, dtype=np.float32),
        )
        self._last_requested_steering = requested_steering.astype(np.float32).copy()
        self._last_requested_wheel_velocities = requested_wheel_velocities.astype(np.float32).copy()
        steering_positions = self._apply_steering_limits(requested_steering, dt)
        actual_steering = self._get_current_steering_positions(default=steering_positions)
        wheel_velocities = self._shape_wheel_velocities_for_applied_steering(
            command=command,
            requested_steering=requested_steering,
            applied_steering=actual_steering,
            requested_wheel_velocities=requested_wheel_velocities,
        )
        wheel_velocities = self._as_finite_vector(
            wheel_velocities,
            expected_size=wheel_count,
            fallback=np.zeros(wheel_count, dtype=np.float32),
        )
        self._last_applied_wheel_velocities = wheel_velocities.astype(np.float32).copy()
        self.robot.apply_base_command(
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
        )
        self._publish_joint_state()
        self._publish_odometry()
        self._record_debug_history(
            command=command,
            requested_steering=requested_steering,
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
            now_sec=now_sec,
            dt=dt,
        )
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def _apply_heading_gate(self, command: BaseCommand, now_sec: float) -> BaseCommand:
        if not self._heading_gate_enabled:
            return command
        _translation, orientation = self._get_robot_base_pose()
        yaw = float(self._yaw_from_wxyz(orientation))
        yaw_error = math.atan2(
            math.sin(self._heading_gate_target_yaw - yaw),
            math.cos(self._heading_gate_target_yaw - yaw),
        )
        if abs(yaw_error) <= self._heading_gate_tolerance_rad:
            self._heading_gate_enabled = False
            return command
        rotate_vel = math.copysign(self._heading_gate_rotate_vel, yaw_error)
        return BaseCommand(vx_body=0.0, vy_body=0.0, wz_body=rotate_vel, received_time_sec=float(now_sec))

    def _on_cmd_vel(self, msg: Twist):
        received_time_sec = self._now_sec()
        self._received_cmd_vel_count += 1
        self._driver_command_message_count += 1
        self._last_received_cmd_vel = {
            "linear_x": float(msg.linear.x),
            "linear_y": float(msg.linear.y),
            "angular_z": float(msg.angular.z),
            "received_time_sec": float(received_time_sec),
        }
        raw_command = BaseCommand.from_twist_message(msg, received_time_sec=received_time_sec)
        finite_command = self._is_finite_command(raw_command)
        if finite_command:
            command = self._clamp_command(raw_command)
        else:
            command = BaseCommand.zero(received_time_sec=received_time_sec)
            self._command = command
        accepted = bool(self._navigation_active and finite_command)
        if accepted:
            self._command = command
            self._applied_driver_command_count += 1
        self._debug_cmd_vel_history.append(
            {
                "received_time_sec": float(received_time_sec),
                "accepted": accepted,
                "rejected_reason": None if finite_command else "non_finite_cmd_vel",
                "cmd_vel": {
                    "linear_x": float(msg.linear.x),
                    "linear_y": float(msg.linear.y),
                    "angular_z": float(msg.angular.z),
                },
                "resolved_command": {
                    "vx_body": float(command.vx_body),
                    "vy_body": float(command.vy_body),
                    "wz_body": float(command.wz_body),
                },
            }
        )

    def _resolve_active_command(self, now_sec: float) -> BaseCommand:
        if self._is_finite_command(self._command) and now_sec - self._command.received_time_sec <= self._command_timeout:
            return self._command
        return BaseCommand.zero(received_time_sec=self._command.received_time_sec)

    @staticmethod
    def _is_finite_command(command: BaseCommand) -> bool:
        return all(
            math.isfinite(float(value))
            for value in (command.vx_body, command.vy_body, command.wz_body, command.received_time_sec)
        )

    @staticmethod
    def _as_finite_vector(values, *, expected_size: int, fallback) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        fallback = np.asarray(fallback, dtype=np.float32).reshape(-1)
        if vector.size != expected_size or not np.all(np.isfinite(vector)):
            if fallback.size == expected_size and np.all(np.isfinite(fallback)):
                return fallback.astype(np.float32).copy()
            return np.zeros(expected_size, dtype=np.float32)
        return vector.astype(np.float32).copy()

    def _load_body_velocity_limits(self) -> tuple[np.ndarray, np.ndarray]:
        hard_limits = (
            self.base_cfg.get("platform", {})
            .get("nav2", {})
            .get("controller_hard_limits", {})
        )
        min_velocity = hard_limits.get("min_velocity", [-float("inf"), -float("inf"), -float("inf")])
        max_velocity = hard_limits.get("max_velocity", [float("inf"), float("inf"), float("inf")])
        min_velocity = np.asarray(min_velocity, dtype=np.float32).reshape(-1)
        max_velocity = np.asarray(max_velocity, dtype=np.float32).reshape(-1)
        if min_velocity.size != 3 or max_velocity.size != 3:
            raise ValueError("platform.nav2.controller_hard_limits velocity limits must be 3-element lists")
        return min_velocity, max_velocity

    def _clamp_command(self, command: BaseCommand) -> BaseCommand:
        if not self._is_finite_command(command):
            return BaseCommand.zero(received_time_sec=self._now_sec())
        clamped = np.clip(
            np.asarray([command.vx_body, command.vy_body, command.wz_body], dtype=np.float32),
            self._min_body_velocity,
            self._max_body_velocity,
        )
        return BaseCommand(
            vx_body=float(clamped[0]),
            vy_body=float(clamped[1]),
            wz_body=float(clamped[2]),
            received_time_sec=command.received_time_sec,
        )

    def get_logging_action_snapshot(self) -> dict:
        command = self._last_step_command
        return {
            "vx_body": float(command.vx_body),
            "vy_body": float(command.vy_body),
            "wz_body": float(command.wz_body),
            "navigation_active": bool(self._navigation_active),
            "hold_after_navigation": bool(self._hold_after_navigation),
            "restore_after_navigation": bool(self._restore_after_navigation),
            "restore_after_navigation_done": bool(self.restore_after_navigation_done()),
            "restore_done_tolerance": float(self._restore_done_tolerance),
            "restore_done_velocity_tolerance": float(self._restore_done_velocity_tolerance),
            "requested_steering": [float(v) for v in self._last_requested_steering.tolist()],
            "requested_wheel_velocities": [float(v) for v in self._last_requested_wheel_velocities.tolist()],
            "applied_wheel_velocities": [float(v) for v in self._last_applied_wheel_velocities.tolist()],
            "wheel_shaping": dict(self._last_wheel_shaping_debug),
        }

    def get_logging_state_snapshot(self) -> dict:
        translation, orientation = self._get_robot_base_pose()
        translation = np.asarray(translation, dtype=np.float32)
        yaw = float(self._yaw_from_wxyz(orientation))
        dt = max(float(self._last_step_dt), 1e-3)
        linear_velocity_world = (translation - self._last_actual_translation) / dt
        yaw_delta = self._wrap_angle(yaw - self._last_actual_yaw)
        angular_velocity_world = np.array([0.0, 0.0, yaw_delta / dt], dtype=np.float32)
        linear_velocity_body = self._world_linear_velocity_to_body(linear_velocity_world, orientation)
        joint_state = self.robot.get_base_joint_state()

        return {
            "pose": [
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
                float(yaw),
            ],
            "twist_body": [
                float(linear_velocity_body[0]),
                float(linear_velocity_body[1]),
                float(angular_velocity_world[2]),
            ],
            "steering_positions": [float(v) for v in np.asarray(joint_state["steering_positions"]).reshape(-1).tolist()],
            "wheel_positions": [float(v) for v in np.asarray(joint_state["wheel_positions"]).reshape(-1).tolist()],
            "steering_velocities": [
                float(v) for v in np.asarray(joint_state["steering_velocities"]).reshape(-1).tolist()
            ],
            "wheel_velocities": [float(v) for v in np.asarray(joint_state["wheel_velocities"]).reshape(-1).tolist()],
        }

    def _apply_steering_limits(self, requested_positions: np.ndarray, dt: float):
        steering_count = len(self.base_interface["steering_joint_names"])
        fallback = self._get_current_steering_positions(default=self._get_restore_target_steering(steering_count))
        self._last_applied_steering = self._as_finite_vector(
            self._last_applied_steering,
            expected_size=steering_count,
            fallback=fallback,
        )
        requested_positions = self._as_finite_vector(
            requested_positions,
            expected_size=steering_count,
            fallback=fallback,
        )
        requested_positions = np.clip(requested_positions, -self._steering_limit, self._steering_limit)
        max_delta = self._steering_rate_limit * dt
        delta = requested_positions - self._last_applied_steering
        delta = np.clip(delta, -max_delta, max_delta)
        limited = self._last_applied_steering + delta
        self._last_applied_steering = limited.astype(np.float32)
        return self._last_applied_steering.copy()

    def _get_current_steering_positions(self, *, default: np.ndarray):
        try:
            joint_state = self.robot.get_base_joint_state()
            current = np.asarray(joint_state["steering_positions"], dtype=np.float32).reshape(-1)
        except Exception:
            current = np.asarray(default, dtype=np.float32).reshape(-1)
        expected = len(self.base_interface["steering_joint_names"])
        if current.size != expected or not np.all(np.isfinite(current)):
            return np.asarray(default, dtype=np.float32).reshape(-1).copy()
        current = np.asarray([self._wrap_angle(float(value)) for value in current], dtype=np.float32)
        return np.clip(current, -self._steering_limit, self._steering_limit).astype(np.float32)

    def _get_restore_target_steering(self, steering_count: int):
        getter = getattr(self.robot, "get_base_initial_steering_positions", None)
        if callable(getter):
            target = np.asarray(getter(), dtype=np.float32).reshape(-1)
            if target.size == steering_count and np.all(np.isfinite(target)):
                return np.clip(target, -self._steering_limit, self._steering_limit).astype(np.float32)
        return np.zeros(steering_count, dtype=np.float32)

    def has_non_finite_state(self) -> bool:
        return bool(self._non_finite_state_detected)

    def non_finite_state_reason(self) -> str:
        return str(self._non_finite_state_reason)

    def _shape_wheel_velocities_for_applied_steering(
        self,
        *,
        command: BaseCommand,
        requested_steering: np.ndarray,
        applied_steering: np.ndarray,
        requested_wheel_velocities: np.ndarray,
    ) -> np.ndarray:
        del command, requested_steering, applied_steering
        return np.asarray(requested_wheel_velocities, dtype=np.float32).copy()

    def _publish_joint_state(self):
        joint_state = self.robot.get_base_joint_state()
        if not self._joint_state_is_finite(joint_state):
            self._mark_non_finite_state("non_finite_joint_state")
            return
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.name = self.base_interface["steering_joint_names"] + self.base_interface["wheel_joint_names"]
        msg.position = (
            list(joint_state["steering_positions"].astype(float)) + list(joint_state["wheel_positions"].astype(float))
        )
        msg.velocity = (
            list(joint_state["steering_velocities"].astype(float)) + list(joint_state["wheel_velocities"].astype(float))
        )
        self._joint_state_pub.publish(msg)

    def _publish_odometry(self):
        translation, orientation = self._get_robot_base_pose()
        if not self._pose_is_finite(translation, orientation):
            self._mark_non_finite_state("non_finite_base_pose")
            return

        # Path-B fix: keep side effects of _get_actual_base_twist (updates internal tracking),
        # but use the *nominal* command twist for odometry feedback to preserve semantic
        # consistency between Nav2's motion model and the bridge's control intent.
        linear_velocity, angular_velocity = self._get_actual_base_twist(translation, orientation)
        linear_velocity_body = self._world_linear_velocity_to_body(linear_velocity, orientation)
        if not np.all(np.isfinite(linear_velocity_body)) or not np.all(np.isfinite(angular_velocity)):
            self._mark_non_finite_state("non_finite_base_twist")
            return

        # Nominal twist from the last commanded step (semantic odometry)
        command = self._last_step_command
        nominal_vx = float(command.vx_body) if math.isfinite(command.vx_body) else 0.0
        nominal_vy = float(command.vy_body) if math.isfinite(command.vy_body) else 0.0
        nominal_wz = float(command.wz_body) if math.isfinite(command.wz_body) else 0.0

        odom_msg = Odometry()
        odom_msg.header.stamp = self.node.get_clock().now().to_msg()
        odom_msg.header.frame_id = self.ros_cfg["odom_frame"]
        odom_msg.child_frame_id = self.ros_cfg["base_frame"]
        odom_msg.pose.pose.position.x = float(translation[0])
        odom_msg.pose.pose.position.y = float(translation[1])
        odom_msg.pose.pose.position.z = float(translation[2])
        odom_msg.pose.pose.orientation.x = float(orientation[1])
        odom_msg.pose.pose.orientation.y = float(orientation[2])
        odom_msg.pose.pose.orientation.z = float(orientation[3])
        odom_msg.pose.pose.orientation.w = float(orientation[0])
        odom_msg.twist.twist.linear.x = nominal_vx
        odom_msg.twist.twist.linear.y = nominal_vy
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = nominal_wz

        self._odom_pub.publish(odom_msg)
        self._last_published_pose_debug = {
            "x": float(translation[0]),
            "y": float(translation[1]),
            "z": float(translation[2]),
            "yaw": float(self._yaw_from_wxyz(orientation)),
            "linear_velocity_body": [nominal_vx, nominal_vy, 0.0],
            "angular_velocity_world": [0.0, 0.0, nominal_wz],
            "actual_linear_velocity_body": [float(v) for v in list(linear_velocity_body)],
            "actual_angular_velocity_world": [float(v) for v in list(angular_velocity)],
        }

        if self.ros_cfg["tf_enabled"]:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = odom_msg.header.stamp
            tf_msg.header.frame_id = self.ros_cfg["odom_frame"]
            tf_msg.child_frame_id = self.ros_cfg["base_frame"]
            tf_msg.transform.translation.x = float(translation[0])
            tf_msg.transform.translation.y = float(translation[1])
            tf_msg.transform.translation.z = float(translation[2])
            tf_msg.transform.rotation.x = float(orientation[1])
            tf_msg.transform.rotation.y = float(orientation[2])
            tf_msg.transform.rotation.z = float(orientation[3])
            tf_msg.transform.rotation.w = float(orientation[0])
            self._tf_pub.publish(TFMessage(transforms=[tf_msg]))

    def _get_robot_base_pose(self):
        getter = getattr(self.robot, "get_mobile_base_pose", None)
        if callable(getter):
            return getter()
        return self.robot.get_world_pose()

    def _get_actual_base_twist(self, translation, orientation):
        translation = np.asarray(translation, dtype=np.float32)
        yaw = float(self._yaw_from_wxyz(orientation))
        if not np.all(np.isfinite(translation)) or not math.isfinite(yaw):
            self._mark_non_finite_state("non_finite_base_pose")
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        dt = max(float(self._last_step_dt), 1e-3)

        linear_velocity = (translation - self._last_actual_translation) / dt
        yaw_delta = self._wrap_angle(yaw - self._last_actual_yaw)
        angular_velocity = np.array([0.0, 0.0, yaw_delta / dt], dtype=np.float32)

        self._last_actual_translation = translation.copy()
        self._last_actual_yaw = yaw
        self._last_actual_linear_velocity_world = linear_velocity.astype(np.float32)
        self._last_actual_angular_velocity_world = angular_velocity
        return self._last_actual_linear_velocity_world, self._last_actual_angular_velocity_world

    @staticmethod
    def _world_linear_velocity_to_body(linear_velocity_world, orientation_wxyz):
        linear_velocity_world = np.asarray(linear_velocity_world, dtype=np.float32)
        yaw = float(BaseBridge._yaw_from_wxyz(orientation_wxyz))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        vx_world = float(linear_velocity_world[0])
        vy_world = float(linear_velocity_world[1])
        vz_world = float(linear_velocity_world[2])
        return np.array(
            [
                cos_yaw * vx_world + sin_yaw * vy_world,
                -sin_yaw * vx_world + cos_yaw * vy_world,
                vz_world,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _yaw_from_wxyz(q_wxyz):
        w = float(q_wxyz[0])
        x = float(q_wxyz[1])
        y = float(q_wxyz[2])
        z = float(q_wxyz[3])
        if not all(math.isfinite(value) for value in (w, x, y, z)):
            return float("nan")
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _mark_non_finite_state(self, reason: str):
        self._non_finite_state_detected = True
        self._non_finite_state_reason = str(reason)
        self._command = BaseCommand.zero(received_time_sec=self._now_sec())
        wheel_count = len(self.base_interface["wheel_joint_names"])
        self._last_requested_wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        self._last_applied_wheel_velocities = np.zeros(wheel_count, dtype=np.float32)
        self._last_wheel_shaping_debug = {}

    @staticmethod
    def _pose_is_finite(translation, orientation) -> bool:
        translation = np.asarray(translation, dtype=np.float32).reshape(-1)
        orientation = np.asarray(orientation, dtype=np.float32).reshape(-1)
        return (
            translation.size >= 3
            and orientation.size >= 4
            and np.all(np.isfinite(translation[:3]))
            and np.all(np.isfinite(orientation[:4]))
            and float(np.linalg.norm(orientation[:4])) > 1.0e-8
        )

    @staticmethod
    def _joint_state_is_finite(joint_state: dict) -> bool:
        for key in ("steering_positions", "wheel_positions", "steering_velocities", "wheel_velocities"):
            value = np.asarray(joint_state.get(key, []), dtype=np.float32).reshape(-1)
            if value.size and not np.all(np.isfinite(value)):
                return False
        return True

    @staticmethod
    def _wrap_angle(angle: float):
        return math.atan2(math.sin(angle), math.cos(angle))

    def _record_debug_history(
        self,
        *,
        command: BaseCommand,
        requested_steering: np.ndarray,
        steering_positions: np.ndarray,
        wheel_velocities: np.ndarray,
        now_sec: float,
        dt: float,
        mode: str = "cmd_vel",
    ):
        joint_state = self.robot.get_base_joint_state()
        history_item = {
            "time_sec": float(now_sec),
            "dt": float(dt),
            "mode": str(mode),
            "command": {
                "vx_body": float(command.vx_body),
                "vy_body": float(command.vy_body),
                "wz_body": float(command.wz_body),
            },
            "predicted_body_twist": {
                "vx": float(command.vx_body),
                "vy": float(command.vy_body),
                "wz": float(command.wz_body),
            },
            "requested_steering": [float(v) for v in list(np.asarray(requested_steering).reshape(-1))],
            "applied_steering": [float(v) for v in list(np.asarray(steering_positions).reshape(-1))],
            "requested_wheel_velocities": [
                float(v) for v in list(np.asarray(self._last_requested_wheel_velocities).reshape(-1))
            ],
            "wheel_velocities": [float(v) for v in list(np.asarray(wheel_velocities).reshape(-1))],
            "wheel_shaping": dict(self._last_wheel_shaping_debug),
            "actual_joint_state": {
                "steering_positions": [
                    float(v) for v in np.asarray(joint_state["steering_positions"]).reshape(-1).tolist()
                ],
                "wheel_positions": [
                    float(v) for v in np.asarray(joint_state["wheel_positions"]).reshape(-1).tolist()
                ],
                "steering_velocities": [
                    float(v) for v in np.asarray(joint_state["steering_velocities"]).reshape(-1).tolist()
                ],
                "wheel_velocities": [
                    float(v) for v in np.asarray(joint_state["wheel_velocities"]).reshape(-1).tolist()
                ],
            },
            "pose": dict(self._last_published_pose_debug),
        }
        self._debug_command_history.append(history_item)

    def _now_sec(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _spin_available_callbacks(self, max_callbacks: int = 8):
        callback_count = 0
        while callback_count < max(int(max_callbacks), 1):
            timeout_sec = 0.001 if callback_count == 0 else 0.0
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)
            callback_count += 1
