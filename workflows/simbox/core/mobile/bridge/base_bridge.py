"""移动底盘 Isaac Bridge 抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import importlib
import math

import numpy as np

from .types import BaseCommand


def _load_ros_message_types():
    """延迟加载 ROS message 类型，避免模块导入阶段强依赖 Isaac ROS bridge。"""
    geometry_msgs = importlib.import_module("geometry_msgs.msg")
    nav_msgs = importlib.import_module("nav_msgs.msg")
    sensor_msgs = importlib.import_module("sensor_msgs.msg")
    tf2_msgs = importlib.import_module("tf2_msgs.msg")
    return {
        "TransformStamped": geometry_msgs.TransformStamped,
        "Twist": geometry_msgs.Twist,
        "Odometry": nav_msgs.Odometry,
        "JointState": sensor_msgs.JointState,
        "TFMessage": tf2_msgs.TFMessage,
    }


class BaseBridge(ABC):
    """将标准 /cmd_vel 直接桥接为 Isaac articulation 控制。"""

    def __init__(self, robot, node_name: str, driver=None):
        """初始化 ROS 节点、/cmd_vel 订阅、状态发布器和底盘限幅参数。"""
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

        virtual_odom_cfg = self.ros_cfg["virtual_odom"]
        if not isinstance(virtual_odom_cfg, dict):
            raise TypeError("ros.virtual_odom must be a dict")
        if bool(virtual_odom_cfg["enabled"]):
            raise ValueError("virtual_odom is not supported for the direct /cmd_vel bridge")

        self._command_timeout = float(self.base_cfg["command_timeout"])
        self._steering_limit = float(self.base_cfg["steering_limit"])
        self._steering_rate_limit = float(self.base_cfg["steering_rate_limit"])
        self._wheel_velocity_limit = float(self.base_cfg["wheel_velocity_limit"])
        self._wheel_base = float(self.base_cfg["wheel_base"])
        self._track_width = float(self.base_cfg["track_width"])
        self._wheel_radius = float(self.base_cfg["wheel_radius"])
        self._steering_command_sign = float(self.base_cfg["steering_command_sign"])
        self._min_body_velocity, self._max_body_velocity = self._load_body_velocity_limits()
        if abs(self._steering_command_sign) <= 1.0e-6:
            raise ValueError("steering_command_sign must be non-zero")
        if self._wheel_radius <= 0.0:
            raise ValueError("wheel_radius must be positive")

        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        self._validate_bridge_configuration(steering_count=steering_count, wheel_count=wheel_count)

        from nav2.bridge.clock import ensure_isaac_ros2_bridge_ready

        ros2_imports = ensure_isaac_ros2_bridge_ready(
            max_wait_sec=float(self.ros_cfg.get("bridge_ready_timeout_sec", 90.0)),
        )
        ros_message_types = _load_ros_message_types()
        self._rclpy = ros2_imports["rclpy"]
        self._Node = ros2_imports["Node"]
        self._TransformStamped = ros_message_types["TransformStamped"]
        self._Twist = ros_message_types["Twist"]
        self._Odometry = ros_message_types["Odometry"]
        self._JointState = ros_message_types["JointState"]
        self._TFMessage = ros_message_types["TFMessage"]

        self._owns_rclpy_context = False
        if not self._rclpy.ok():
            self._rclpy.init(args=None)
            self._owns_rclpy_context = True
        self.node = self._Node(node_name)
        self.node.set_parameters([ros2_imports["Parameter"]("use_sim_time", value=True)])
        self._tf_pub = self.node.create_publisher(self._TFMessage, "/tf", 10)
        self._joint_state_pub = self.node.create_publisher(self._JointState, self.ros_cfg["joint_state_topic"], 10)
        self._odom_pub = self.node.create_publisher(self._Odometry, self.ros_cfg["odom_topic"], 10)
        self._cmd_vel_sub = self.node.create_subscription(
            self._Twist,
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
        self._has_nav2_command = False
        history_size = max(int(self.base_cfg["debug_history_size"]), 1)
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
        """销毁 bridge ROS 节点，并在本对象初始化 rclpy 时负责关闭上下文。"""
        self.node.destroy_node()
        if self._owns_rclpy_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def reset(self, *, clear_debug_history: bool = False):
        """重置 bridge 缓存和调试状态，不向底盘写入控制目标。"""
        self._spin_available_callbacks()
        now_sec = self._now_sec()
        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        zero_command = BaseCommand.zero(received_time_sec=now_sec)
        zero_wheel = np.zeros(wheel_count, dtype=np.float32)

        self._command = zero_command
        self._last_step_command = zero_command
        current_steering = self._get_current_steering_positions()
        self._last_applied_steering = current_steering.copy()
        self._last_requested_steering = current_steering.copy()
        self._last_requested_wheel_velocities = zero_wheel.copy()
        self._last_applied_wheel_velocities = zero_wheel.copy()
        self._last_wheel_shaping_debug = {}
        self._last_step_time_sec = now_sec
        self._last_step_dt = 1e-3
        self._navigation_active = False
        self._has_nav2_command = False
        self._last_received_cmd_vel = {
            "linear_x": 0.0,
            "linear_y": 0.0,
            "angular_z": 0.0,
            "received_time_sec": float(now_sec),
        }
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

        translation, orientation = self._get_robot_base_pose()
        self._last_actual_translation = np.array(translation, dtype=np.float32)
        self._last_actual_yaw = float(self._yaw_from_wxyz(orientation))
        self._last_actual_linear_velocity_world = np.zeros(3, dtype=np.float32)
        self._last_actual_angular_velocity_world = np.zeros(3, dtype=np.float32)
        self._publish_joint_state()
        self._publish_odometry()
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def prepare_for_navigation(self):
        """进入导航态，使 bridge 开始接受 Nav2 发来的 /cmd_vel。"""
        self._navigation_active = True
        self._has_nav2_command = False

    def finalize_after_navigation(self):
        """退出导航态，立即写入零速度并停止接受后续 /cmd_vel。"""
        self._spin_available_callbacks()
        now_sec = self._now_sec()
        zero_command = BaseCommand.zero(received_time_sec=now_sec)
        steering_positions, wheel_velocities = self._map_command(zero_command)
        self._apply_robot_base_command(
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
            step_dt=1e-3,
        )
        self._command = zero_command
        self._last_step_command = zero_command
        self._last_requested_steering = steering_positions.copy()
        self._last_requested_wheel_velocities = wheel_velocities.copy()
        self._last_applied_steering = steering_positions.copy()
        self._last_applied_wheel_velocities = wheel_velocities.copy()
        self._navigation_active = False
        self._has_nav2_command = False
        self._last_wheel_shaping_debug = {}
        self._last_step_time_sec = now_sec
        self._last_step_dt = 1e-3
        self._publish_joint_state()
        self._publish_odometry()
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def step(self, step_dt: float | None = None):
        """推进 bridge 一帧：接收回调、解析有效 /cmd_vel、映射并应用到底盘关节。"""
        self._spin_available_callbacks()
        now_sec = self._now_sec()
        if step_dt is None:
            dt = max(now_sec - self._last_step_time_sec, 1e-3)
        else:
            dt = max(float(step_dt), 1e-3)
        self._last_step_time_sec = now_sec
        self._last_step_dt = dt

        if not self._navigation_active or not self._has_nav2_command:
            self._publish_joint_state()
            self._publish_odometry()
            self._rclpy.spin_once(self.node, timeout_sec=0.0)
            return

        command = self._resolve_active_command(now_sec)
        self._last_step_command = command
        steering_count = len(self.base_interface["steering_joint_names"])
        wheel_count = len(self.base_interface["wheel_joint_names"])
        current_steering = self._get_current_steering_positions()
        self._last_applied_steering = current_steering.copy()
        requested_steering, requested_wheel_velocities = self._map_command(command)
        requested_steering = self._require_finite_vector(
            requested_steering,
            expected_size=steering_count,
            name="requested steering",
        )
        requested_wheel_velocities = self._require_finite_vector(
            requested_wheel_velocities,
            expected_size=wheel_count,
            name="requested wheel velocities",
        )
        self._last_requested_steering = requested_steering.astype(np.float32).copy()
        self._last_requested_wheel_velocities = requested_wheel_velocities.astype(np.float32).copy()
        steering_positions = self._apply_steering_limits(requested_steering, dt)
        actual_steering = self._get_current_steering_positions()
        wheel_velocities = self._shape_wheel_velocities_for_applied_steering(
            command=command,
            requested_steering=requested_steering,
            applied_steering=actual_steering,
            requested_wheel_velocities=requested_wheel_velocities,
        )
        wheel_velocities = self._require_finite_vector(
            wheel_velocities,
            expected_size=wheel_count,
            name="applied wheel velocities",
        )
        self._last_applied_wheel_velocities = wheel_velocities.astype(np.float32).copy()
        self._apply_robot_base_command(
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
            step_dt=dt,
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
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def _apply_robot_base_command(
        self,
        *,
        steering_positions: np.ndarray,
        wheel_velocities: np.ndarray,
        step_dt: float,
    ) -> None:
        """把计算出的转向角和轮速发送给 robot，默认忽略 step_dt。"""
        del step_dt
        self.robot.apply_base_command(
            steering_positions=steering_positions,
            wheel_velocities=wheel_velocities,
        )

    def _on_cmd_vel(self, msg):
        """处理 Nav2 /cmd_vel：校验有限性、做硬限幅，并仅在导航态接收。"""
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
        if not finite_command:
            raise ValueError("Received non-finite /cmd_vel")
        command = self._clamp_command(raw_command)
        accepted = bool(self._navigation_active and finite_command)
        if accepted:
            self._command = command
            self._has_nav2_command = True
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
        """返回未超时的当前命令；超时后返回零命令，避免继续执行旧 /cmd_vel。"""
        if not self._is_finite_command(self._command):
            raise ValueError("Active base command is non-finite")
        if now_sec - self._command.received_time_sec <= self._command_timeout:
            return self._command
        return BaseCommand.zero(received_time_sec=self._command.received_time_sec)

    @staticmethod
    def _is_finite_command(command: BaseCommand) -> bool:
        """检查 BaseCommand 的速度和时间戳是否都是有限数。"""
        return all(
            math.isfinite(float(value))
            for value in (command.vx_body, command.vy_body, command.wz_body, command.received_time_sec)
        )

    @staticmethod
    def _require_finite_vector(values, *, expected_size: int, name: str) -> np.ndarray:
        """把输入转换为固定长度 float32 向量，并拒绝 NaN/Inf。"""
        vector = np.asarray(values, dtype=np.float32).reshape(-1)
        if vector.size != expected_size or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must have size {expected_size} and contain only finite values")
        return vector.astype(np.float32).copy()

    def _load_body_velocity_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """从配置读取 Nav2 controller hard limits，作为 bridge 侧唯一 body twist 限幅。"""
        hard_limits = self.base_cfg["platform"]["nav2"]["controller_hard_limits"]
        min_velocity = hard_limits["min_velocity"]
        max_velocity = hard_limits["max_velocity"]
        min_velocity = np.asarray(min_velocity, dtype=np.float32).reshape(-1)
        max_velocity = np.asarray(max_velocity, dtype=np.float32).reshape(-1)
        if (
            min_velocity.size != 3
            or max_velocity.size != 3
            or not np.all(np.isfinite(min_velocity))
            or not np.all(np.isfinite(max_velocity))
        ):
            raise ValueError("platform.nav2.controller_hard_limits velocity limits must be 3-element lists")
        return min_velocity, max_velocity

    def _clamp_command(self, command: BaseCommand) -> BaseCommand:
        """按配置的 min/max body velocity 对 /cmd_vel 做逐轴硬限幅。"""
        if not self._is_finite_command(command):
            raise ValueError("Cannot clamp non-finite base command")
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
        """导出最近一次 bridge 命令、关节目标和映射信息，供 episode/action 日志使用。"""
        command = self._last_step_command
        dof_names = list(getattr(getattr(self.robot, "_articulation_view", None), "dof_names", []))
        return {
            "vx_body": float(command.vx_body),
            "vy_body": float(command.vy_body),
            "wz_body": float(command.wz_body),
            "navigation_active": bool(self._navigation_active),
            "has_nav2_command": bool(self._has_nav2_command),
            "requested_steering": [float(v) for v in self._last_requested_steering.tolist()],
            "requested_wheel_velocities": [float(v) for v in self._last_requested_wheel_velocities.tolist()],
            "applied_wheel_velocities": [float(v) for v in self._last_applied_wheel_velocities.tolist()],
            "wheel_shaping": dict(self._last_wheel_shaping_debug),
            "joint_mapping": {
                "steering_joint_names": list(self.base_interface["steering_joint_names"]),
                "wheel_joint_names": list(self.base_interface["wheel_joint_names"]),
                "steering_joint_indices": [int(v) for v in self.base_interface["steering_joint_indices"]],
                "wheel_joint_indices": [int(v) for v in self.base_interface["wheel_joint_indices"]],
                "articulation_dof_names": dof_names,
            },
        }

    def get_logging_state_snapshot(self) -> dict:
        """导出当前底盘 pose/twist/joint state，供 episode/state 日志使用。"""
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
        """对转向关节目标施加角度限幅和转向速率限幅。"""
        steering_count = len(self.base_interface["steering_joint_names"])
        self._last_applied_steering = self._require_finite_vector(
            self._last_applied_steering,
            expected_size=steering_count,
            name="last applied steering",
        )
        requested_positions = self._require_finite_vector(
            requested_positions,
            expected_size=steering_count,
            name="requested steering positions",
        )
        requested_positions = np.clip(requested_positions, -self._steering_limit, self._steering_limit)
        max_delta = self._steering_rate_limit * dt
        delta = requested_positions - self._last_applied_steering
        delta = np.clip(delta, -max_delta, max_delta)
        limited = self._last_applied_steering + delta
        self._last_applied_steering = limited.astype(np.float32)
        return self._last_applied_steering.copy()

    def _get_current_steering_positions(self):
        """读取当前转向关节角，wrap 到 [-pi, pi] 后再按 steering_limit 截断。"""
        joint_state = self.robot.get_base_joint_state()
        current = np.asarray(joint_state["steering_positions"], dtype=np.float32).reshape(-1)
        expected = len(self.base_interface["steering_joint_names"])
        if current.size != expected or not np.all(np.isfinite(current)):
            raise ValueError("Current steering positions must match steering joints and be finite")
        current = np.asarray([self._wrap_angle(float(value)) for value in current], dtype=np.float32)
        return np.clip(current, -self._steering_limit, self._steering_limit).astype(np.float32)

    def has_non_finite_state(self) -> bool:
        """返回 bridge 是否曾检测到底盘状态中存在非有限数。"""
        return bool(self._non_finite_state_detected)

    def non_finite_state_reason(self) -> str:
        """返回最近一次非有限底盘状态的诊断原因。"""
        return str(self._non_finite_state_reason)

    def _shape_wheel_velocities_for_applied_steering(
        self,
        *,
        command: BaseCommand,
        requested_steering: np.ndarray,
        applied_steering: np.ndarray,
        requested_wheel_velocities: np.ndarray,
    ) -> np.ndarray:
        """给子类按实际转向角修正轮速的 hook；默认直接执行映射出的轮速。"""
        del command, requested_steering, applied_steering
        return np.asarray(requested_wheel_velocities, dtype=np.float32).copy()

    def _publish_joint_state(self):
        """发布当前底盘 steering/wheel joint state 给 ROS/Nav2。"""
        joint_state = self.robot.get_base_joint_state()
        if not self._joint_state_is_finite(joint_state):
            raise ValueError("Base joint state is missing required finite vectors")
        msg = self._JointState()
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
        """从 Isaac 当前底盘位姿计算真实速度，并发布 odom 和可选 tf。"""
        translation, orientation = self._get_robot_base_pose()
        if not self._pose_is_finite(translation, orientation):
            raise ValueError("Base pose must be finite before publishing odometry")

        linear_velocity, angular_velocity = self._get_actual_base_twist(translation, orientation)
        linear_velocity_body = self._world_linear_velocity_to_body(linear_velocity, orientation)
        if not np.all(np.isfinite(linear_velocity_body)) or not np.all(np.isfinite(angular_velocity)):
            raise ValueError("Base twist must be finite before publishing odometry")

        odom_msg = self._Odometry()
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
        odom_msg.twist.twist.linear.x = float(linear_velocity_body[0])
        odom_msg.twist.twist.linear.y = float(linear_velocity_body[1])
        odom_msg.twist.twist.linear.z = 0.0
        odom_msg.twist.twist.angular.x = 0.0
        odom_msg.twist.twist.angular.y = 0.0
        odom_msg.twist.twist.angular.z = float(angular_velocity[2])

        self._odom_pub.publish(odom_msg)
        self._last_published_pose_debug = {
            "x": float(translation[0]),
            "y": float(translation[1]),
            "z": float(translation[2]),
            "yaw": float(self._yaw_from_wxyz(orientation)),
            "linear_velocity_body": [float(v) for v in list(linear_velocity_body)],
            "angular_velocity_world": [float(v) for v in list(angular_velocity)],
            "actual_linear_velocity_body": [float(v) for v in list(linear_velocity_body)],
            "actual_angular_velocity_world": [float(v) for v in list(angular_velocity)],
        }

        if self.ros_cfg["tf_enabled"]:
            tf_msg = self._TransformStamped()
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
            self._tf_pub.publish(self._TFMessage(transforms=[tf_msg]))

    def _get_robot_base_pose(self):
        """优先读取导航基座 pose，缺省退回移动底盘 pose。"""
        getter = getattr(self.robot, "get_nav_base_pose", None)
        if callable(getter):
            return getter()
        getter = getattr(self.robot, "get_mobile_base_pose", None)
        if not callable(getter):
            raise ValueError("Robot must provide get_mobile_base_pose for mobile base bridge")
        return getter()

    def _get_actual_base_twist(self, translation, orientation):
        """用连续两帧 pose 差分估计实际 world-frame 线速度和角速度。"""
        translation = np.asarray(translation, dtype=np.float32)
        yaw = float(self._yaw_from_wxyz(orientation))
        if not np.all(np.isfinite(translation)) or not math.isfinite(yaw):
            raise ValueError("Base pose must be finite before computing actual twist")
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
        """按当前 yaw 把 world-frame 线速度旋转到底盘 body-frame。"""
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
        """从 wxyz 四元数提取 yaw；输入非有限时返回 NaN。"""
        w = float(q_wxyz[0])
        x = float(q_wxyz[1])
        y = float(q_wxyz[2])
        z = float(q_wxyz[3])
        if not all(math.isfinite(value) for value in (w, x, y, z)):
            return float("nan")
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _pose_is_finite(translation, orientation) -> bool:
        """检查 pose 是否至少包含有限 xyz 和有效四元数。"""
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
        """检查底盘 joint_state 是否包含所有必需字段且数值有限。"""
        for key in ("steering_positions", "wheel_positions", "steering_velocities", "wheel_velocities"):
            if key not in joint_state:
                return False
            value = np.asarray(joint_state[key], dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(value)):
                return False
        return True

    @staticmethod
    def _wrap_angle(angle: float):
        """把角度 wrap 到 [-pi, pi]。"""
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
        """记录 bridge 最近命令、关节目标、实际关节状态和 pose 调试历史。"""
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
        """返回当前 ROS 节点时钟秒数；在 sim time 模式下跟随 /clock。"""
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _spin_available_callbacks(self, max_callbacks: int = 8):
        """短时间 spin ROS 回调，尽快接收 /cmd_vel 且不阻塞 Isaac 主循环。"""
        callback_count = 0
        while callback_count < max(int(max_callbacks), 1):
            timeout_sec = 0.001 if callback_count == 0 else 0.0
            self._rclpy.spin_once(self.node, timeout_sec=timeout_sec)
            callback_count += 1
