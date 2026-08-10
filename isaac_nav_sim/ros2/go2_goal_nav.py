"""RViz goal follower for the Isaac Go2 ROS2 simulation.

This is a lightweight navigation bridge for this repository's current ROS2
interface. RViz publishes /goal_pose, this node reads simulated odometry and
publishes /unitree_go2_0/cmd_vel for the low-level RL locomotion policy.
"""

from __future__ import annotations

import argparse
import math
import threading

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class Go2GoalNavigator(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("go2_goal_navigator")
        self.args = args
        self.goal: PoseStamped | None = None
        self.odom: Odometry | None = None
        self.last_scan = {"front": False, "left": 0, "right": 0, "best_angle": 0.0, "best_clearance": 0.0}
        self.last_map = {"front": False, "low_conf": False, "left": 0, "right": 0}
        self.scan_lock = threading.Lock()
        self.map_lock = threading.Lock()
        self.obstacle_map: OccupancyGrid | None = None
        self.confidence_map: OccupancyGrid | None = None

        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.path_pub = self.create_publisher(Path, args.path_topic, 10)

        self.create_subscription(PoseStamped, args.goal_topic, self.goal_callback, 10)
        self.create_subscription(Odometry, args.odom_topic, self.odom_callback, 10)
        if args.lidar_topic:
            self.create_subscription(PointCloud2, args.lidar_topic, self.lidar_callback, 5)
        if args.obstacle_map_topic:
            self.create_subscription(OccupancyGrid, args.obstacle_map_topic, self.obstacle_map_callback, 5)
        if args.confidence_map_topic:
            self.create_subscription(OccupancyGrid, args.confidence_map_topic, self.confidence_map_callback, 5)

        self.timer = self.create_timer(1.0 / args.rate, self.control_step)
        self.get_logger().info(
            "Ready: RViz goal %s -> odom %s -> cmd_vel %s"
            % (args.goal_topic, args.odom_topic, args.cmd_topic)
        )

    def goal_callback(self, msg: PoseStamped) -> None:
        self.goal = msg
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        self.get_logger().info(f"New goal: x={gx:.2f}, y={gy:.2f}")
        self.publish_path()

    def odom_callback(self, msg: Odometry) -> None:
        self.odom = msg

    def lidar_callback(self, msg: PointCloud2) -> None:
        front = False
        left = 0
        right = 0
        count = 0
        sector_count = int(self.args.local_planner_sectors)
        sector_fov = float(self.args.local_planner_fov)
        sector_edges = [
            -sector_fov + 2.0 * sector_fov * i / sector_count
            for i in range(sector_count + 1)
        ]
        sector_clearance = [float(self.args.local_planner_range)] * sector_count

        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = float(point[0]), float(point[1]), float(point[2])
            if z >= self.args.obstacle_min_z and z <= self.args.obstacle_max_z and x > self.args.obstacle_min_x:
                point_range = math.hypot(x, y)
                angle = math.atan2(y, x)
                if point_range <= self.args.local_planner_range and abs(angle) <= sector_fov:
                    sector_idx = min(
                        sector_count - 1,
                        max(0, int((angle + sector_fov) / (2.0 * sector_fov) * sector_count)),
                    )
                    sector_clearance[sector_idx] = min(sector_clearance[sector_idx], point_range)
            if x < self.args.obstacle_min_x or x > self.args.obstacle_max_x:
                continue
            if abs(y) > self.args.obstacle_half_width:
                continue
            if z < self.args.obstacle_min_z or z > self.args.obstacle_max_z:
                continue
            count += 1
            if y >= 0.0:
                left += 1
            else:
                right += 1
            if count >= self.args.obstacle_points:
                front = True
                break

        sector_centers = [
            0.5 * (sector_edges[i] + sector_edges[i + 1])
            for i in range(sector_count)
        ]
        best_idx = max(range(sector_count), key=lambda idx: sector_clearance[idx] - 0.25 * abs(sector_centers[idx]))
        with self.scan_lock:
            self.last_scan = {
                "front": front,
                "left": left,
                "right": right,
                "best_angle": sector_centers[best_idx],
                "best_clearance": sector_clearance[best_idx],
            }

    def obstacle_map_callback(self, msg: OccupancyGrid) -> None:
        with self.map_lock:
            self.obstacle_map = msg
        self.update_map_hazards()

    def confidence_map_callback(self, msg: OccupancyGrid) -> None:
        with self.map_lock:
            self.confidence_map = msg
        self.update_map_hazards()

    def update_map_hazards(self) -> None:
        with self.map_lock:
            obstacle_msg = self.obstacle_map
            confidence_msg = self.confidence_map
        if obstacle_msg is None:
            return

        width = int(obstacle_msg.info.width)
        height = int(obstacle_msg.info.height)
        if width <= 0 or height <= 0:
            return

        obstacle = list(obstacle_msg.data)
        confidence = list(confidence_msg.data) if confidence_msg is not None else None
        resolution = float(obstacle_msg.info.resolution)
        origin_x = float(obstacle_msg.info.origin.position.x)
        origin_y = float(obstacle_msg.info.origin.position.y)

        high_conf_hits = 0
        low_conf_hits = 0
        left = 0
        right = 0
        for iy in range(height):
            y = origin_y + (iy + 0.5) * resolution
            if abs(y) > self.args.map_obstacle_half_width:
                continue
            for ix in range(width):
                x = origin_x + (ix + 0.5) * resolution
                if x < self.args.map_obstacle_min_x or x > self.args.map_obstacle_max_x:
                    continue
                idx = iy * width + ix
                occ = int(obstacle[idx])
                if occ < self.args.map_obstacle_value:
                    continue
                conf = 100 if confidence is None or idx >= len(confidence) else int(confidence[idx])
                if conf >= self.args.map_confidence_min:
                    high_conf_hits += 1
                    if y >= 0.0:
                        left += 1
                    else:
                        right += 1
                elif conf >= 0:
                    low_conf_hits += 1

        with self.map_lock:
            self.last_map = {
                "front": high_conf_hits >= self.args.map_obstacle_cells,
                "low_conf": low_conf_hits >= self.args.map_low_conf_cells,
                "left": left,
                "right": right,
            }

    def control_step(self) -> None:
        if self.goal is None or self.odom is None:
            self.stop()
            return

        pose = self.odom.pose.pose
        x = pose.position.x
        y = pose.position.y
        yaw = yaw_from_quat(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

        gx = self.goal.pose.position.x
        gy = self.goal.pose.position.y
        gyaw = yaw_from_quat(
            self.goal.pose.orientation.x,
            self.goal.pose.orientation.y,
            self.goal.pose.orientation.z,
            self.goal.pose.orientation.w,
        )

        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)
        if dist <= self.args.xy_tolerance:
            yaw_error = wrap_pi(gyaw - yaw)
            if abs(yaw_error) <= self.args.yaw_tolerance:
                self.get_logger().info("Goal reached.")
                self.goal = None
                self.stop()
                return
            cmd = Twist()
            cmd.angular.z = clamp(self.args.k_yaw * yaw_error, -self.args.max_wz, self.args.max_wz)
            self.cmd_pub.publish(cmd)
            return

        target_yaw = math.atan2(dy, dx)
        heading_error = wrap_pi(target_yaw - yaw)
        cmd = Twist()

        if abs(heading_error) > self.args.turn_in_place_yaw:
            cmd.linear.x = 0.0
        else:
            heading_scale = max(0.0, math.cos(heading_error))
            cmd.linear.x = clamp(self.args.k_linear * dist, 0.0, self.args.max_vx) * heading_scale

        cmd.angular.z = clamp(self.args.k_yaw * heading_error, -self.args.max_wz, self.args.max_wz)

        with self.scan_lock:
            scan = dict(self.last_scan)
        with self.map_lock:
            grid = dict(self.last_map)

        if grid["front"] or scan["front"]:
            best_angle = float(scan.get("best_angle", 0.0))
            best_clearance = float(scan.get("best_clearance", 0.0))
            if abs(best_angle) < 0.08:
                left = grid["left"] + scan["left"]
                right = grid["right"] + scan["right"]
                best_angle = -self.args.local_planner_escape_angle if left > right else self.args.local_planner_escape_angle
            desired_heading = wrap_pi(
                (1.0 - self.args.local_planner_goal_bias) * best_angle
                + self.args.local_planner_goal_bias * heading_error
            )
            slow_vx = self.args.obstacle_slow_vx
            if best_clearance > self.args.local_planner_free_distance:
                slow_vx = max(slow_vx, self.args.local_planner_free_vx)
            cmd.linear.x = min(cmd.linear.x, slow_vx)
            cmd.angular.z = clamp(self.args.k_yaw * desired_heading, -self.args.max_wz, self.args.max_wz)
        elif grid["low_conf"]:
            cmd.linear.x = min(cmd.linear.x, self.args.low_conf_slow_vx)

        self.cmd_pub.publish(cmd)
        self.publish_path()

    def publish_path(self) -> None:
        if self.goal is None or self.odom is None:
            return
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.args.map_frame

        current = PoseStamped()
        current.header = path.header
        current.pose = self.odom.pose.pose

        goal = PoseStamped()
        goal.header = path.header
        goal.pose = self.goal.pose

        path.poses = [current, goal]
        self.path_pub.publish(path)

    def stop(self) -> None:
        self.cmd_pub.publish(Twist())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RViz goal navigator for Unitree Go2 in Isaac.")
    parser.add_argument("--robot_namespace", default="unitree_go2_0")
    parser.add_argument("--goal_topic", default="/goal_pose")
    parser.add_argument("--path_topic", default="/go2_goal_nav/path")
    parser.add_argument("--map_frame", default="map")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--max_vx", type=float, default=1.0)
    parser.add_argument("--max_wz", type=float, default=1.2)
    parser.add_argument("--k_linear", type=float, default=0.8)
    parser.add_argument("--k_yaw", type=float, default=1.8)
    parser.add_argument("--xy_tolerance", type=float, default=0.25)
    parser.add_argument("--yaw_tolerance", type=float, default=0.35)
    parser.add_argument("--turn_in_place_yaw", type=float, default=0.75)
    parser.add_argument("--obstacle_slow_vx", type=float, default=0.10)
    parser.add_argument("--obstacle_turn_wz", type=float, default=0.7)
    parser.add_argument("--obstacle_min_x", type=float, default=0.15)
    parser.add_argument("--obstacle_max_x", type=float, default=1.35)
    parser.add_argument("--obstacle_half_width", type=float, default=0.42)
    parser.add_argument("--obstacle_min_z", type=float, default=-0.25)
    parser.add_argument("--obstacle_max_z", type=float, default=0.8)
    parser.add_argument("--obstacle_points", type=int, default=20)
    parser.add_argument("--low_conf_slow_vx", type=float, default=0.35)
    parser.add_argument("--map_obstacle_min_x", type=float, default=0.20)
    parser.add_argument("--map_obstacle_max_x", type=float, default=1.45)
    parser.add_argument("--map_obstacle_half_width", type=float, default=0.46)
    parser.add_argument("--map_obstacle_value", type=int, default=50)
    parser.add_argument("--map_confidence_min", type=int, default=35)
    parser.add_argument("--map_obstacle_cells", type=int, default=4)
    parser.add_argument("--map_low_conf_cells", type=int, default=10)
    parser.add_argument("--local_planner_sectors", type=int, default=15)
    parser.add_argument("--local_planner_fov", type=float, default=1.25)
    parser.add_argument("--local_planner_range", type=float, default=2.2)
    parser.add_argument("--local_planner_goal_bias", type=float, default=0.35)
    parser.add_argument("--local_planner_escape_angle", type=float, default=0.65)
    parser.add_argument("--local_planner_free_distance", type=float, default=1.4)
    parser.add_argument("--local_planner_free_vx", type=float, default=0.22)
    parser.add_argument("--no_lidar_avoidance", action="store_true")
    parser.add_argument("--no_map_avoidance", action="store_true")
    args = parser.parse_args()
    ns = args.robot_namespace.strip("/")
    args.odom_topic = f"/{ns}/odom"
    args.cmd_topic = f"/{ns}/cmd_vel"
    args.lidar_topic = None if args.no_lidar_avoidance else f"/{ns}/lidar/point_cloud"
    args.obstacle_map_topic = None if args.no_map_avoidance else f"/{ns}/elevation/obstacle_map"
    args.confidence_map_topic = None if args.no_map_avoidance else f"/{ns}/elevation/confidence_map"
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = Go2GoalNavigator(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
