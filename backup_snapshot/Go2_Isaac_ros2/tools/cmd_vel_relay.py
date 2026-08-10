#!/usr/bin/env python3
"""Relay Nav2 cmd_vel to the Go2 low-level controller topic."""

from __future__ import annotations

import argparse

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CmdVelRelay(Node):
    def __init__(self, source_topic: str, target_topic: str):
        super().__init__("go2_cmd_vel_relay")
        self.pub = self.create_publisher(Twist, target_topic, 10)
        self.create_subscription(Twist, source_topic, self.pub.publish, 10)
        self.get_logger().info(f"Relaying {source_topic} -> {target_topic}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/cmd_vel")
    parser.add_argument("--target", default="/unitree_go2_0/cmd_vel")
    args = parser.parse_args()

    rclpy.init()
    node = CmdVelRelay(args.source, args.target)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
