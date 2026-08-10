#!/usr/bin/env python3
"""Relay Go2 namespaced odometry to Nav2's default /odom topic."""

from __future__ import annotations

import argparse

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class OdomRelay(Node):
    def __init__(self, source_topic: str, target_topic: str):
        super().__init__("go2_odom_relay")
        self.pub = self.create_publisher(Odometry, target_topic, 10)
        self.create_subscription(Odometry, source_topic, self.pub.publish, 10)
        self.get_logger().info(f"Relaying {source_topic} -> {target_topic}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/unitree_go2_0/odom")
    parser.add_argument("--target", default="/odom")
    args = parser.parse_args()

    rclpy.init()
    node = OdomRelay(args.source, args.target)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
