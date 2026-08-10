import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import numpy as np
from cv_bridge import CvBridge
import cv2
import omni
import omni.graph.core as og
import subprocess
import time
import quadruped.go2.go2_ctrl as go2_ctrl
from ros2.lidar_weather_sim import LocalElevationMapper, RealtimeFogSimulator, WeatherConfig
#For YOLO detection
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, \
    BoundingBox2D, Pose2D
from visualization_msgs.msg import MarkerArray, Marker

ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
from isaacsim.ros2.bridge import collect_namespace, read_camera_info


def yaw_from_wxyz(quat) -> float:
    w = float(quat[0])
    x = float(quat[1])
    y = float(quat[2])
    z = float(quat[3])
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class RobotDataManager(Node):
    def __init__(self, env, lidar_annotators, cameras, cfg):
        super().__init__("robot_data_manager")
        self.cfg = cfg
        ros_cfg = getattr(cfg, "ros", None)
        self.use_sim_time_enabled = bool(getattr(ros_cfg, "use_sim_time", False))
        if self.use_sim_time_enabled:
            if self.create_ros_time_graph():
                self.use_sim_time()
            else:
                self.use_sim_time_enabled = False

        self.env = env
        self.num_envs = env.unwrapped.scene.num_envs
        self.lidar_annotators = lidar_annotators
        self.cameras = cameras
        self.lidar_backend = getattr(self.cfg.sensor, "lidar_backend", "rtx")
        self.geometry_world_obstacles = [
            (2.2, -0.7, 0.55),
            (3.4, 0.9, 0.85),
            (5.1, -1.6, 0.65),
            (-1.8, 1.2, 0.50),
        ]
        self.points = []
        self.step_size = 1
        self.node_namespace = ""         
        self.queue_size = 1
        self.confidence_threshold = 0.5  # Set your desired confidence threshold for YOLO detection
        
        # ROS2 Broadcaster
        self.broadcaster= TransformBroadcaster(self)
        
        # ROS2 Publisher
        self.odom_pub = []
        self.pose_pub = []
        self.lidar_pub = []
        self.lidar_raw_pub = []
        self.confidence_map_pub = []
        self.height_variance_map_pub = []
        self.obstacle_map_pub = []
        self.semantic_seg_img_vis_pub = []
        self.det_pub = []
        self.vis_pub = []
        self.marker_pub = []

        # ROS2 Subscriber
        self.cmd_vel_sub = []
        self.color_img_sub = []
        self.depth_img_sub = []
        self.semantic_seg_img_sub = []
        self.image_sub = []

        # ROS2 Timer
        self.lidar_publish_timer = []

        #ROS2 SLAM
        # Mini-map parameters
        self.map_size = 500  # pixels
        self.map_scale = 50  # 1 meter = 50 pixels
        self.robot_radius = 5
        self.obj_radius = 3

        
        
        for i in range(self.num_envs):
            prefix= f"unitree_go2_{i}"
            
            self.odom_pub.append(self.create_publisher(Odometry, f"{prefix}/odom", 10))
            self.pose_pub.append(self.create_publisher(PoseStamped, f"{prefix}/pose", 10))
            self.lidar_pub.append(self.create_publisher(PointCloud2, 
                                                        f"{prefix}/lidar/point_cloud", 10))
            self.lidar_raw_pub.append(self.create_publisher(PointCloud2,
                                                        f"{prefix}/lidar/point_cloud_raw", 10))
            self.confidence_map_pub.append(self.create_publisher(
                                    OccupancyGrid, f"{prefix}/elevation/confidence_map", 10))
            self.height_variance_map_pub.append(self.create_publisher(
                                    OccupancyGrid, f"{prefix}/elevation/height_variance_map", 10))
            self.obstacle_map_pub.append(self.create_publisher(
                                    OccupancyGrid, f"{prefix}/elevation/obstacle_map", 10))
            # Publishers for Yolo detections and visualization
            self.det_pub.append(self.create_publisher(Detection2DArray, 
                                    f"{prefix}/front_cam/detections",10))
            self.vis_pub.append(self.create_publisher(Image, 
                                    f"{prefix}/front_cam/detection_image", 10))
            self.marker_pub.append(self.create_publisher(MarkerArray,
                                    f"{prefix}/front_cam/detection_markers",10))
            # For semantic segmentation visualization
            self.semantic_seg_img_vis_pub.append(self.create_publisher(Image, 
                        f"{prefix}/front_cam/semantic_segmentation_image_vis", 10))
            
            self.cmd_vel_sub.append(self.create_subscription(Twist, f"{prefix}/cmd_vel", 
                    lambda msg, idx=i: self.cmd_vel_callback(msg, idx), 10))
            #For yolo
            self.image_sub.append(self.create_subscription(Image, f"{prefix}/front_cam/color_image",
                lambda msg, idx=i: self.image_callback(msg, idx), 10))
            
            self.semantic_seg_img_sub.append(self.create_subscription(Image, 
                    f"{prefix}/front_cam/semantic_segmentation_image", 
                    lambda msg, idx=i: self.semantic_segmentation_callback(msg, idx), 10))
        
        # self.create_timer(0.1, self.pub_ros2_data_callback)
        # self.create_timer(0.1, self.pub_lidar_data_callback)

        # use wall time for lidar and odom pub
        self.odom_pose_freq = 50.0
        self.lidar_freq = 15.0
        self.odom_pose_pub_time = time.time()
        self.lidar_pub_time = time.time() 
        self.create_static_transform()
        self.create_camera_publisher()  

        self.weather_cfg = WeatherConfig.from_cfg(cfg)
        self.fog_sim = RealtimeFogSimulator(self.weather_cfg)
        self.elevation_mapper = LocalElevationMapper()
        self.rng = np.random.default_rng(2026)
        if self.weather_cfg.enabled:
            params = self.weather_cfg.params
            self.get_logger().info(
                f"Weather LiDAR enabled: preset={self.weather_cfg.preset}, "
                f"alpha={params['alpha']}, friction={params['friction']}"
            )

        # YOLO setup
        self.bridge = CvBridge()
        self.model = None
        if self.cfg.sensor.enable_camera and self.cfg.sensor.color_image:
            self.model = YOLO('ckpts/yolo/yolov8n.pt')  # model path




    def get_robot_position(self, env_idx):
    # Fetch robot pose from latest odometry (map frame)
        try:
            msg = self.pose_pub[env_idx]  # Or store latest PoseStamped in a variable
            x = msg.pose.position.x
            y = msg.pose.position.y
            return np.array([x, y])
        except:
            return np.array([0, 0])

    def world_to_map(self, pos):
        # Convert world coordinates (meters) to map pixels
        map_x = int(self.map_size / 2 + pos[0]*self.map_scale)
        map_y = int(self.map_size / 2 - pos[1]*self.map_scale)
        return (map_x, map_y)

    def get_depth_at_pixel(self, env_idx, px, py):
        # Return depth value at given pixel from latest depth image
        # For simplicity, you can use a constant depth or subscribe to depth image
        return 2.0  # meters

    def transform_camera_to_map(self, robot_pos, depth, box):
        # Convert camera detection to world coordinates (simplified)
        # Assumes objects are in front of robot
        # For real implementation, use TF2 + camera intrinsics + depth
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        # Simple projection forward
        obj_x = robot_pos[0] + depth
        obj_y = robot_pos[1] + (center_x - 320)/320.0  # scale horizontal
        return np.array([obj_x, obj_y])

 
    def create_ros_time_graph(self):
        try:
            og.Controller.edit(
                {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishClock.inputs:topicName", "/clock"),
                    ],
                },
            )
            return True
        except Exception as exc:
            self.get_logger().warn(f"ROS clock graph unavailable; using system time fallback: {exc}")
            return False

    def use_sim_time(self):
        # Define the command as a list
        command = ["ros2", "param", "set", "/robot_data_manager", "use_sim_time", "true"]

        # Run the command in a non-blocking way
        subprocess.Popen(command)  
        return True

    def create_static_transform(self):
        for i in range(self.num_envs):
            # LiDAR
            # Create and publish the transform
            lidar_broadcaster = StaticTransformBroadcaster(self)
            base_lidar_transform = TransformStamped()
            base_lidar_transform.header.stamp = self.get_clock().now().to_msg()
            base_lidar_transform.header.frame_id = f"unitree_go2_{i}/base_link"
            base_lidar_transform.child_frame_id = f"unitree_go2_{i}/lidar_frame"

            # Translation
            base_lidar_transform.transform.translation.x = 0.2
            base_lidar_transform.transform.translation.y = 0.0
            base_lidar_transform.transform.translation.z = 0.2
            
            # Rotation 
            base_lidar_transform.transform.rotation.x = 0.0
            base_lidar_transform.transform.rotation.y = 0.0
            base_lidar_transform.transform.rotation.z = 0.0
            base_lidar_transform.transform.rotation.w = 1.0
            
            # Publish the transform
            lidar_broadcaster.sendTransform(base_lidar_transform)
    
            # -------------------------------------------------------------
            # Camera
            # Create and publish the transform
            camera_broadcaster = StaticTransformBroadcaster(self)
            base_cam_transform = TransformStamped()
            # base_cam_transform.header.stamp = self.get_clock().now().to_msg()
            base_cam_transform.header.frame_id = f"unitree_go2_{i}/base_link"
            base_cam_transform.child_frame_id = f"unitree_go2_{i}/front_cam"

            # Translation
            base_cam_transform.transform.translation.x = 0.4
            base_cam_transform.transform.translation.y = 0.0
            base_cam_transform.transform.translation.z = 0.2
            
            # Rotation 
            base_cam_transform.transform.rotation.x = -0.5
            base_cam_transform.transform.rotation.y = 0.5
            base_cam_transform.transform.rotation.z = -0.5
            base_cam_transform.transform.rotation.w = 0.5
            
            # Publish the transform
            camera_broadcaster.sendTransform(base_cam_transform)
    
    def create_camera_publisher(self):
        # self.pub_image_graph()
        if (self.cfg.sensor.enable_camera):
            if (self.cfg.sensor.color_image):
                self.pub_color_image()
            if (self.cfg.sensor.depth_image):
                self.pub_depth_image()
            if (self.cfg.sensor.semantic_segmentation):
                self.pub_semantic_image()
            # self.pub_cam_depth_cloud()
            self.publish_camera_info()
    
    def publish_odom(self, base_pos, base_rot, base_lin_vel_b, base_ang_vel_b, env_idx):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = f"unitree_go2_{env_idx}/base_link"
        odom_msg.pose.pose.position.x = base_pos[0].item()
        odom_msg.pose.pose.position.y = base_pos[1].item()
        odom_msg.pose.pose.position.z = base_pos[2].item()
        odom_msg.pose.pose.orientation.x = base_rot[1].item()
        odom_msg.pose.pose.orientation.y = base_rot[2].item()
        odom_msg.pose.pose.orientation.z = base_rot[3].item()
        odom_msg.pose.pose.orientation.w = base_rot[0].item()
        odom_msg.twist.twist.linear.x = base_lin_vel_b[0].item()
        odom_msg.twist.twist.linear.y = base_lin_vel_b[1].item()
        odom_msg.twist.twist.linear.z = base_lin_vel_b[2].item()
        odom_msg.twist.twist.angular.x = base_ang_vel_b[0].item()
        odom_msg.twist.twist.angular.y = base_ang_vel_b[1].item()
        odom_msg.twist.twist.angular.z = base_ang_vel_b[2].item()
        self.odom_pub[env_idx].publish(odom_msg)

        # transform
        map_base_trans = TransformStamped()
        map_base_trans.header.stamp = self.get_clock().now().to_msg()
        map_base_trans.header.frame_id = "map"
        map_base_trans.child_frame_id = f"unitree_go2_{env_idx}/base_link"
        map_base_trans.transform.translation.x = base_pos[0].item()
        map_base_trans.transform.translation.y = base_pos[1].item()
        map_base_trans.transform.translation.z = base_pos[2].item()
        map_base_trans.transform.rotation.x = base_rot[1].item()
        map_base_trans.transform.rotation.y = base_rot[2].item()
        map_base_trans.transform.rotation.z = base_rot[3].item()
        map_base_trans.transform.rotation.w = base_rot[0].item()
        self.broadcaster.sendTransform(map_base_trans)


    # def publish_odom(self, base_pos, base_rot, base_lin_vel_b, base_ang_vel_b, env_idx):
    #     odom_msg = Odometry()
    #     odom_msg.header.stamp = self.get_clock().now().to_msg()
    #     # odom_msg.header.frame_id = "map"
    #     # odom_msg.child_frame_id = f"unitree_go2_{env_idx}/base_link"
    #     odom_msg.header.frame_id = "odom"            # parent
    #     odom_msg.child_frame_id = f"unitree_go2_{env_idx}/base_link"

    #     odom_msg.pose.pose.position.x = base_pos[0].item()
    #     odom_msg.pose.pose.position.y = base_pos[1].item()
    #     odom_msg.pose.pose.position.z = base_pos[2].item()
    #     odom_msg.pose.pose.orientation.x = base_rot[1].item()
    #     odom_msg.pose.pose.orientation.y = base_rot[2].item()
    #     odom_msg.pose.pose.orientation.z = base_rot[3].item()
    #     odom_msg.pose.pose.orientation.w = base_rot[0].item()
    #     odom_msg.twist.twist.linear.x = base_lin_vel_b[0].item()
    #     odom_msg.twist.twist.linear.y = base_lin_vel_b[1].item()
    #     odom_msg.twist.twist.linear.z = base_lin_vel_b[2].item()
    #     odom_msg.twist.twist.angular.x = base_ang_vel_b[0].item()
    #     odom_msg.twist.twist.angular.y = base_ang_vel_b[1].item()
    #     odom_msg.twist.twist.angular.z = base_ang_vel_b[2].item()
    #     self.odom_pub[env_idx].publish(odom_msg)

    #     # transform
    #     map_base_trans = TransformStamped()
    #     map_base_trans.header.stamp = self.get_clock().now().to_msg()
    #     map_base_trans.header.frame_id = "map"
    #     map_base_trans.child_frame_id = f"unitree_go2_{env_idx}/base_link"
    #     map_base_trans.transform.translation.x = base_pos[0].item()
    #     map_base_trans.transform.translation.y = base_pos[1].item()
    #     map_base_trans.transform.translation.z = base_pos[2].item()
    #     map_base_trans.transform.rotation.x = base_rot[1].item()
    #     map_base_trans.transform.rotation.y = base_rot[2].item()
    #     map_base_trans.transform.rotation.z = base_rot[3].item()
    #     map_base_trans.transform.rotation.w = base_rot[0].item()
    #     self.broadcaster.sendTransform(map_base_trans)
    
    def publish_pose(self, base_pos, base_rot, env_idx):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = base_pos[0].item()
        pose_msg.pose.position.y = base_pos[1].item()
        pose_msg.pose.position.z = base_pos[2].item()
        pose_msg.pose.orientation.x = base_rot[1].item()
        pose_msg.pose.orientation.y = base_rot[2].item()
        pose_msg.pose.orientation.z = base_rot[3].item()
        pose_msg.pose.orientation.w = base_rot[0].item()
        self.pose_pub[env_idx].publish(pose_msg)

    def publish_lidar_data(self, points, env_idx):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        stamp = self.get_clock().now().to_msg()
        frame_id = f"unitree_go2_{env_idx}/lidar_frame"

        if self.weather_cfg.publish_raw:
            raw_cloud = PointCloud2()
            raw_cloud.header.frame_id = frame_id
            raw_cloud.header.stamp = stamp
            raw_fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
            raw_cloud = point_cloud2.create_cloud(raw_cloud.header, raw_fields, points.tolist())
            self.lidar_raw_pub[env_idx].publish(raw_cloud)

        degraded_points, fog_info = self.fog_sim.apply(points)
        point_cloud = PointCloud2()
        point_cloud.header.frame_id = frame_id
        point_cloud.header.stamp = stamp
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud = point_cloud2.create_cloud(point_cloud.header, fields, degraded_points.tolist())
        
        self.lidar_pub[env_idx].publish(point_cloud)

        confidence, height_variance, obstacle = self.elevation_mapper.build(
            degraded_points,
            frame_id=frame_id,
            stamp=stamp,
            fog_alpha=float(fog_info.get("alpha", 0.0)),
        )
        self.confidence_map_pub[env_idx].publish(confidence)
        self.height_variance_map_pub[env_idx].publish(height_variance)
        self.obstacle_map_pub[env_idx].publish(obstacle)

    def pub_ros2_data_callback(self):
        robot_data = self.env.unwrapped.scene["unitree_go2"].data
        for i in range(self.num_envs):
            self.publish_odom(robot_data.root_state_w[i, :3],
                              robot_data.root_state_w[i, 3:7],
                              robot_data.root_lin_vel_b[i],
                              robot_data.root_ang_vel_b[i],
                              i)
            self.publish_pose(robot_data.root_state_w[i, :3],
                              robot_data.root_state_w[i, 3:7], i)

    def _read_rtx_lidar_points(self, env_idx):
        data = self.lidar_annotators[env_idx].get_data()
        if isinstance(data, dict):
            data = data.get("data", data.get("pointCloud", []))
        points = np.asarray(data, dtype=np.float32)
        if points.size == 0:
            return points.reshape(0, 3)
        return points.reshape(-1, points.shape[-1])[:, :3]

    def _generate_geometry_lidar_points(self, env_idx):
        n_points = int(getattr(self.cfg.sensor, "geometry_points", 2500))
        lidar_range = float(getattr(self.cfg.sensor, "geometry_range", 8.0))
        n_ground = max(360, int(n_points * 0.72))
        n_angles = 180
        n_rings = max(2, n_ground // n_angles)
        robot_data = self.env.unwrapped.scene["unitree_go2"].data
        base_pos = robot_data.root_state_w[env_idx, :3].detach().cpu().numpy()
        base_quat = robot_data.root_state_w[env_idx, 3:7].detach().cpu().numpy()
        base_yaw = yaw_from_wxyz(base_quat)
        cos_yaw = np.cos(base_yaw)
        sin_yaw = np.sin(base_yaw)
        lidar_world_x = float(base_pos[0] + 0.2 * cos_yaw)
        lidar_world_y = float(base_pos[1] + 0.2 * sin_yaw)
        lidar_world_z = float(base_pos[2] + 0.2)

        angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False, dtype=np.float32)
        ranges = np.linspace(0.35, lidar_range, n_rings, dtype=np.float32)
        rr, aa = np.meshgrid(ranges, angles, indexing="ij")
        ground = np.stack([
            rr.reshape(-1) * np.cos(aa.reshape(-1)),
            rr.reshape(-1) * np.sin(aa.reshape(-1)),
            np.full(rr.size, -lidar_world_z, dtype=np.float32),
        ], axis=1)

        fog_alpha = float(self.weather_cfg.params.get("alpha", 0.0)) if self.weather_cfg.enabled else 0.0
        if fog_alpha > 0.0:
            z_noise = self.rng.normal(0.0, 0.01 + 0.08 * fog_alpha, size=ground.shape[0]).astype(np.float32)
            ground[:, 2] += z_noise

        if not bool(getattr(self.cfg.sensor, "geometry_obstacles", True)):
            return ground[:n_points]

        env_name = str(getattr(self.cfg, "env_name", "flat"))
        obstacle_centers = []
        if env_name in ("nav_demo", "obstacle", "warehouse", "office", "hospital", "rivermark", "terrain"):
            obstacle_centers = self.geometry_world_obstacles

        obstacle_points = []
        per_obstacle = max(80, int((n_points - ground.shape[0]) / max(1, len(obstacle_centers))))
        for world_x, world_y, height in obstacle_centers:
            dx = float(world_x - lidar_world_x)
            dy = float(world_y - lidar_world_y)
            cx = cos_yaw * dx + sin_yaw * dy
            cy = -sin_yaw * dx + cos_yaw * dy
            if cx < -0.5 or cx > lidar_range + 0.5 or abs(cy) > lidar_range:
                continue
            side = max(8, int(np.sqrt(max(per_obstacle, 1))))
            ys = np.linspace(-0.35, 0.35, side, dtype=np.float32)
            zs = np.linspace(0.0, height, side, dtype=np.float32) - lidar_world_z
            yy, zz = np.meshgrid(ys, zs, indexing="ij")
            face = np.stack([
                np.full(yy.size, cx, dtype=np.float32),
                cy + yy.reshape(-1),
                zz.reshape(-1),
            ], axis=1)
            obstacle_points.append(face)

        if obstacle_points:
            points = np.concatenate([ground, *obstacle_points], axis=0)
        else:
            points = ground
        if points.shape[0] > n_points:
            idx = self.rng.choice(points.shape[0], n_points, replace=False)
            points = points[idx]
        return points.astype(np.float32, copy=False)

    def _get_lidar_points(self, env_idx):
        if self.lidar_backend == "rtx" and env_idx < len(self.lidar_annotators):
            return self._read_rtx_lidar_points(env_idx)
        return self._generate_geometry_lidar_points(env_idx)

    def pub_lidar_data_callback(self):
        for i in range(self.num_envs):
            self.publish_lidar_data(self._get_lidar_points(i), i)

    def pub_ros2_data(self):
        pub_odom_pose = False
        pub_lidar = False
        dt_odom_pose = time.time() - self.odom_pose_pub_time
        dt_lidar = time.time() - self.lidar_pub_time
        if (dt_odom_pose >= 1./self.odom_pose_freq):
            pub_odom_pose = True
        
        if (dt_lidar >= 1./self.lidar_freq):
            pub_lidar = True

        if (pub_odom_pose):
            self.odom_pose_pub_time = time.time()
            robot_data = self.env.unwrapped.scene["unitree_go2"].data
            for i in range(self.num_envs):
                self.publish_odom(robot_data.root_state_w[i, :3],
                                robot_data.root_state_w[i, 3:7],
                                robot_data.root_lin_vel_b[i],
                                robot_data.root_ang_vel_b[i],
                                i)
                self.publish_pose(robot_data.root_state_w[i, :3],
                                robot_data.root_state_w[i, 3:7], i)
        if (self.cfg.sensor.enable_lidar):
            if (pub_lidar):
                self.lidar_pub_time = time.time()
                for i in range(self.num_envs):
                    self.publish_lidar_data(self._get_lidar_points(i), i)

    def cmd_vel_callback(self, msg, env_idx):
        go2_ctrl.base_vel_cmd_input[env_idx][0] = msg.linear.x
        go2_ctrl.base_vel_cmd_input[env_idx][1] = msg.linear.y
        go2_ctrl.base_vel_cmd_input[env_idx][2] = msg.angular.z
    
    def semantic_segmentation_callback(self, img, env_idx):
        bridge = CvBridge()
        semantic_image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        semantic_image_normalized = (semantic_image / semantic_image.max() * 255).astype(np.uint8)

        # Apply a predefined colormap
        color_mapped_image = cv2.applyColorMap(semantic_image_normalized, cv2.COLORMAP_JET)
        # Convert to ROS Image
        bridge = CvBridge()
        image_msg = bridge.cv2_to_imgmsg(color_mapped_image, encoding='rgb8')
        self.semantic_seg_img_vis_pub[env_idx].publish(image_msg)


    def pub_image_graph(self):
        for i in range(self.num_envs):

            color_topic_name = f"unitree_go2_{i}/front_cam/color_image"
            depth_topic_name = f"unitree_go2_{i}/front_cam/depth_image"
            # segmentation_topic_name = f"unitree_go2_{i}/front_cam/segmentation_image"
            # depth_cloud_topic_name = f"unitree_go2_{i}/front_cam/depth_cloud"
            frame_id = f"unitree_go2_{i}/front_cam"
            keys = og.Controller.Keys
            og.Controller.edit(
                {
                    "graph_path": "/CameraROS2Graph",
                    "evaluator_name": "execution",
                },
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("IsaacCreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                        ("ROS2CameraHelperColor", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ("ROS2CameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        # ("ROS2CameraHelperSegmentation", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        # ("ROS2CameraHelperDepthCloud", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ],

                    keys.SET_VALUES: [
                        ("IsaacCreateRenderProduct.inputs:cameraPrim", f"/World/envs/env_{i}/Go2/base/front_cam"),
                        ("IsaacCreateRenderProduct.inputs:enabled", True),
                        ("IsaacCreateRenderProduct.inputs:height", 480),
                        ("IsaacCreateRenderProduct.inputs:width", 640),
                        
                        # color camera
                        ("ROS2CameraHelperColor.inputs:type", "rgb"),
                        ("ROS2CameraHelperColor.inputs:topicName", color_topic_name),
                        ("ROS2CameraHelperColor.inputs:frameId", frame_id),
                        ("ROS2CameraHelperColor.inputs:useSystemTime", False),

                        # depth camera
                        ("ROS2CameraHelperDepth.inputs:type", "depth"),
                        ("ROS2CameraHelperDepth.inputs:topicName", depth_topic_name),
                        ("ROS2CameraHelperDepth.inputs:frameId", frame_id),
                        ("ROS2CameraHelperDepth.inputs:useSystemTime", False),

                    ],

                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "IsaacCreateRenderProduct.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperColor.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperColor.inputs:renderProductPath"),
                        ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperDepth.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperDepth.inputs:renderProductPath"),

                    ],

                },
            )

    def _rep_sd(self):
        import omni.replicator.core as rep
        import omni.syntheticdata._syntheticdata as sd
        return rep, sd

    def pub_color_image(self):
        rep, sd = self._rep_sd()
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"unitree_go2_{i}/front_cam/color_image"
            frame_id = f"unitree_go2_{i}/front_cam"

            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
            
            writer = rep.writers.get(rv + "ROS2PublishImage")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=topic_name,
            )
            writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)

    def pub_depth_image(self):
        rep, sd = self._rep_sd()
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"unitree_go2_{i}/front_cam/depth_image"
            frame_id = f"unitree_go2_{i}/front_cam"


            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                                    sd.SensorType.DistanceToImagePlane.name
                                )
            writer = rep.writers.get(rv + "ROS2PublishImage")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=topic_name
            )
            writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)

    def pub_semantic_image(self):
        rep, sd = self._rep_sd()
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"unitree_go2_{i}/front_cam/semantic_segmentation_image"
            label_topic_name = f"unitree_go2_{i}/front_cam/semantic_segmentation_label"                       
            frame_id = f"unitree_go2_{i}/front_cam"


            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                                    sd.SensorType.SemanticSegmentation.name
                                )
            writer = rep.writers.get("ROS2PublishSemanticSegmentation")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=topic_name
            )
            writer.attach([render_product])

            semantic_writer = rep.writers.get(
               "SemanticSegmentationSD" + f"ROS2PublishSemanticLabels"
            )
            semantic_writer.initialize(
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=label_topic_name,
            )
            semantic_writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)

    def pub_cam_depth_cloud(self):
        rep, sd = self._rep_sd()
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"unitree_go2_{i}/front_cam/depth_cloud"
            frame_id = f"unitree_go2_{i}/front_cam"

            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                sd.SensorType.DistanceToImagePlane.name
            )

            writer = rep.writers.get(rv + "ROS2PublishPointCloud")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=topic_name,
            )
            writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )

            og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)   

    def publish_camera_info(self):
        rep, _ = self._rep_sd()
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            topic_name = f"unitree_go2_{i}/front_cam/info"
            frame_id = self.cameras[i].prim_path.split("/")[-1] # This matches what the TF tree is publishing.

            writer = rep.writers.get("ROS2PublishCameraInfo")
            camera_info = read_camera_info(render_product_path=render_product)
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=self.node_namespace,
                queueSize=self.queue_size,
                topicName=topic_name,
                width=camera_info["width"],
                height=camera_info["height"],
                projectionType=camera_info["projectionType"],
                k=camera_info["k"].reshape([1, 9]),
                r=camera_info["r"].reshape([1, 9]),
                p=camera_info["p"].reshape([1, 12]),
                physicalDistortionModel=camera_info["physicalDistortionModel"],
                physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
            )
            writer.attach([render_product])

            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                "PostProcessDispatch" + "IsaacSimulationGate", render_product
            )

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            og.Controller.attribute(gate_path + ".inputs:step").set(self.step_size)      

    def image_callback(self, msg, env_idx):
        """Process camera images and detect objects using YOLO."""
        if self.model is None:
            return
        try:
            # Convert ROS Image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Run YOLO detection
            results = self.model(cv_image, verbose=False)
            
            # Create Detection2DArray message
            det_array_msg = Detection2DArray()
            det_array_msg.header = msg.header
            
            # Create marker array for 3D visualization
            marker_array = MarkerArray()
            
            # Process each detection
            for idx, result in enumerate(results[0].boxes.data):
                box = result[:4].cpu().numpy()  # Convert tensor to numpy array
                score = float(result[4])
                class_id = int(result[5])
                if score >= self.confidence_threshold: 
                    # Create 2D detection message
                    det = Detection2D()
                    det.header = msg.header
                    
                    # Set bounding box
                    det.bbox = BoundingBox2D()
                    center_pose = Pose2D()
                    center_pose.position.x = float((box[0] + box[2]) / 2)
                    center_pose.position.y = float((box[1] + box[3]) / 2)
                    center_pose.theta = 0.0
                    det.bbox.center = center_pose
                    det.bbox.size_x = float(box[2] - box[0])
                    det.bbox.size_y = float(box[3] - box[1])
                    
                    # Add class and score
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = self.model.names[class_id]#str(class_id)
                    hypothesis.hypothesis.score = score
                    det.results.append(hypothesis)
                    det_array_msg.detections.append(det)
                    
                    # Draw detection on image
                    cv2.rectangle(
                        cv_image, 
                        (int(box[0]), int(box[1])), 
                        (int(box[2]), int(box[3])), 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Add label with class name and score
                    label = f"{self.model.names[class_id]}: {score:.2f}"
                    cv2.putText(
                        cv_image,
                        label,
                        (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                    
                    # Create 3D marker for visualization
                    marker = Marker()
                    marker.header.frame_id = f"unitree_go2_{env_idx}/front_cam"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.id = idx
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    
                    # Position marker in 3D space (simplified projection)
                    marker.pose.position.x = 2.0  # 2 meters in front of camera
                    marker.pose.position.y = ((box[0] + box[2])/2 - 320) / 100.0  # scaled x offset from center
                    marker.pose.position.z = ((box[1] + box[3])/2 - 240) / 100.0  # scaled y offset from center
                    
                    # Set marker orientation
                    marker.pose.orientation.w = 1.0
                    
                    # Set marker size
                    marker.scale.x = 0.2
                    marker.scale.y = 0.2
                    marker.scale.z = 0.2
                    
                    # Set marker color
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.8
                    
                    marker_array.markers.append(marker)
            
            # Publish all results
            self.det_pub[env_idx].publish(det_array_msg)
            self.vis_pub[env_idx].publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            self.marker_pub[env_idx].publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f'Error in image_callback: {str(e)}')


    # def image_callback(self, msg, env_idx):
    #     try:
    #         # Convert ROS Image to OpenCV image
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
    #         # Run YOLO detection
    #         results = self.model(cv_image, verbose=False)
            
    #         # Mini-map (black background)
    #         map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
            
    #         # Get robot pose from odometry (map frame)
    #         robot_pos = self.get_robot_position(env_idx)  # implement this function
    #         robot_pixel = self.world_to_map(robot_pos)
    #         cv2.circle(map_img, robot_pixel, self.robot_radius, (0, 255, 0), -1)

    #         # Process each YOLO detection
    #         det_array_msg = Detection2DArray()
    #         det_array_msg.header = msg.header
    #         marker_array = MarkerArray()
    #         for idx, result in enumerate(results[0].boxes.data):
    #             box = result[:4].cpu().numpy()
    #             score = float(result[4])
    #             class_id = int(result[5])
    #             if score >= self.confidence_threshold:
    #                 # Draw detection on camera
    #                 cv2.rectangle(cv_image, (int(box[0]), int(box[1])), 
    #                             (int(box[2]), int(box[3])), (0, 255, 0), 2)
    #                 label = f"{self.model.names[class_id]}: {score:.2f}"
    #                 cv2.putText(cv_image, label, (int(box[0]), int(box[1]) - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #                 # Estimate object position in map frame using depth (simplified)
    #                 depth = self.get_depth_at_pixel(env_idx, int((box[0]+box[2])/2), 
    #                                                 int((box[1]+box[3])/2))
    #                 obj_world_pos = self.transform_camera_to_map(robot_pos, depth, box)
    #                 obj_pixel = self.world_to_map(obj_world_pos)
    #                 cv2.circle(map_img, obj_pixel, self.obj_radius, (0, 0, 255), -1)

    #         # Combine camera feed and mini-map
    #         combined = cv2.hconcat([cv_image, cv2.resize(map_img, (cv_image.shape[1], cv_image.shape[0]))])
    #         cv2.imshow(f"Env {env_idx} Camera + Mini-Map", combined)
    #         cv2.waitKey(1)

    #     except Exception as e:
    #         self.get_logger().error(f'Error in image_callback: {str(e)}')
