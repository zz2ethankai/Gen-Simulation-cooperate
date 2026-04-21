#!/usr/bin/env python3
"""Wait until Odometry is flowing before starting the Nav2 lifecycle stack."""

from __future__ import annotations

import argparse
import time

from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy


class OdomWaiter(Node):
    def __init__(self, *, topic: str):
        super().__init__("interndata_wait_for_odom")
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._count = 0
        self._last_pose = None
        self.create_subscription(Odometry, str(topic), self._on_odom, qos)

    @property
    def count(self) -> int:
        return int(self._count)

    @property
    def last_pose(self):
        return self._last_pose

    def _on_odom(self, msg: Odometry):
        self._count += 1
        self._last_pose = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            str(msg.child_frame_id),
            str(msg.header.frame_id),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/odom")
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--min-messages", type=int, default=3)
    args = parser.parse_args()

    rclpy.init(args=None)
    node = OdomWaiter(topic=args.topic)
    deadline = time.monotonic() + max(float(args.timeout_sec), 1.0)
    min_messages = max(int(args.min_messages), 1)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.count >= min_messages:
                pose = node.last_pose or (0.0, 0.0, "", "")
                node.get_logger().info(
                    f"received {node.count} odom messages on {args.topic} "
                    f"last_pose=({float(pose[0]):.3f}, {float(pose[1]):.3f}) "
                    f"frame={str(pose[3])}->{str(pose[2])}"
                )
                return 0
            if time.monotonic() >= deadline:
                node.get_logger().error(
                    f"timed out waiting for {min_messages} odom messages on {args.topic}"
                )
                return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
