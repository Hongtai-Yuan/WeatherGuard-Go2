#!/usr/bin/env python3
import argparse
import math

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2


class ElevationObstacleCloud(Node):
    def __init__(
        self,
        map_topic: str,
        cloud_topic: str,
        clearing_cloud_topic: str,
        threshold: int,
        z: float,
        clear_stride: int,
    ):
        super().__init__("elevation_obstacle_cloud")
        self.threshold = threshold
        self.z = z
        self.clear_stride = max(1, clear_stride)
        self.obstacle_publisher = self.create_publisher(PointCloud2, cloud_topic, 10)
        self.clearing_publisher = self.create_publisher(PointCloud2, clearing_cloud_topic, 10)
        self.create_subscription(OccupancyGrid, map_topic, self._on_map, 10)
        self.get_logger().info(
            f"Publishing obstacle clouds: {map_topic} -> {cloud_topic}, {clearing_cloud_topic}"
        )

    def _on_map(self, msg: OccupancyGrid):
        obstacle_points = []
        clearing_points = []
        width = msg.info.width
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        for index, value in enumerate(msg.data):
            mx = index % width
            my = index // width
            x = origin_x + (mx + 0.5) * resolution
            y = origin_y + (my + 0.5) * resolution
            if math.isfinite(x) and math.isfinite(y):
                if value >= self.threshold:
                    obstacle_points.append((x, y, self.z))
                elif value >= 0 and not (mx % self.clear_stride or my % self.clear_stride):
                    clearing_points.append((x, y, self.z))

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        obstacle_cloud = point_cloud2.create_cloud(
            msg.header,
            fields,
            obstacle_points,
        )
        clearing_cloud = point_cloud2.create_cloud(msg.header, fields, clearing_points)
        self.obstacle_publisher.publish(obstacle_cloud)
        self.clearing_publisher.publish(clearing_cloud)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-topic", default="/unitree_go2_0/elevation/obstacle_map")
    parser.add_argument("--cloud-topic", default="/unitree_go2_0/elevation/obstacle_cloud")
    parser.add_argument("--clearing-cloud-topic", default="/unitree_go2_0/elevation/clearing_cloud")
    parser.add_argument("--threshold", type=int, default=80)
    parser.add_argument("--z", type=float, default=0.25)
    parser.add_argument("--clear-stride", type=int, default=2)
    args = parser.parse_args()

    rclpy.init()
    node = ElevationObstacleCloud(
        args.map_topic,
        args.cloud_topic,
        args.clearing_cloud_topic,
        args.threshold,
        args.z,
        args.clear_stride,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
