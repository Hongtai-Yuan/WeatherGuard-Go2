import os
import hydra
import rclpy
import torch
import time
import math
import argparse
import sys
import omni
from isaaclab.app import AppLauncher
from hydra.utils import to_absolute_path

# add CLI arguments
parser = argparse.ArgumentParser(description="Running the Quadruped RL environment.")
parser.add_argument("--max_steps", type=int, default=0, help="0 runs until the Isaac app is closed.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import carb    # must come after app launcher

import quadruped.go2.go2_ctrl as go2_ctrl
import quadruped.spot.spot_ctrl as spot_ctrl


import quadruped.go2.go2_env as go2_env
import quadruped.spot.spot_env as spot_env


from env.sim_env import create_hospital_env, create_obstacle_env, \
    create_warehouse_env, create_office_env, create_rivermark_env, create_terrain_env, create_nav_demo_env
from ros2.lidar_weather_sim import WeatherConfig

def enable_isaac_extensions(*ext_names):
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    for ext_name in ext_names:
        if not ext_manager.is_extension_enabled(ext_name):
            ext_manager.set_extension_enabled_immediate(ext_name, True)


def enable_sensor_extensions(enable_lidar=False, enable_camera=False, enable_ros2=False):
    ext_names = []
    if enable_lidar or enable_camera:
        ext_names.extend(["omni.replicator.core", "omni.syntheticdata"])
    if enable_lidar:
        ext_names.extend(["omni.sensors.nv.lidar", "isaacsim.sensors.rtx"])
    if enable_camera:
        ext_names.append("isaacsim.sensors.camera")
    if enable_ros2:
        ext_names.append("isaacsim.ros2.bridge")
    enable_isaac_extensions(*ext_names)


FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")
@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)

def main(cfg):

    lidar_backend = getattr(cfg.sensor, "lidar_backend", "rtx")
    use_rtx_lidar = cfg.sensor.enable_lidar and lidar_backend == "rtx"
    if cfg.sensor.enable_lidar or cfg.sensor.enable_camera or cfg.ros.enable:
        enable_sensor_extensions(
            enable_lidar=use_rtx_lidar,
            enable_camera=cfg.sensor.enable_camera,
            enable_ros2=cfg.ros.enable,
        )

    if (cfg.robot == "go2"):
    # Robot Environment setup
        controller = getattr(cfg, "locomotion", {}).get("controller", "rsl_flat")
        if controller == "mujoco_onnx":
            env_cfg = go2_env.Go2MujocoOnnxEnvCfg()
        else:
            env_cfg = go2_env.Go2RSLEnvCfg()
        env_cfg.scene.num_envs = cfg.num_envs
        env_cfg.decimation = math.ceil(1./env_cfg.sim.dt/cfg.freq)
        env_cfg.sim.render_interval = env_cfg.decimation
        go2_ctrl.init_base_vel_cmd(cfg.num_envs)
        if controller == "mujoco_onnx":
            policy_path = to_absolute_path(cfg.locomotion.policy)
            sysid_params_json = to_absolute_path(cfg.locomotion.sysid_params_json)
            weather_cfg = WeatherConfig.from_cfg(cfg)
            ground_friction = cfg.locomotion.ground_friction
            if weather_cfg.enabled:
                ground_friction = weather_cfg.params["friction"]
                print(
                    f"Weather preset '{weather_cfg.preset}' -> alpha={weather_cfg.params['alpha']} "
                    f"ground_friction={ground_friction}",
                    flush=True,
                )
            env, policy = go2_ctrl.get_mujoco_onnx_policy_env(
                env_cfg,
                policy_path=policy_path,
                sysid_params_json=sysid_params_json,
                sysid_overrides={
                    "ground_friction": ground_friction,
                    "policy_command_scale": cfg.locomotion.policy_command_scale,
                    "policy_phase_period": cfg.locomotion.policy_phase_period,
                },
            )
        elif controller == "rsl_flat":
            env, policy = go2_ctrl.get_rsl_flat_policy(env_cfg)
        else:
            raise ValueError(f"Unsupported Go2 locomotion controller: {controller}")
            # Simulation environment
        if (cfg.env_name == "nav_demo"):
            create_nav_demo_env()

        elif (cfg.env_name == "obstacle"):
            create_obstacle_env() # obstacles

        elif (cfg.env_name == "warehouse"):
            create_warehouse_env() # warehouse

        elif (cfg.env_name == "office"):
            create_office_env() # office
        
        elif (cfg.env_name == "hospital"):
            create_hospital_env() # hospital
        
        elif (cfg.env_name == "rivermark"):
            create_rivermark_env() # rivermark

        elif (cfg.env_name == "terrain"):
            create_terrain_env() # terrain

        # Sensor setup
        lidar_annotators = []
        cameras = []
        if use_rtx_lidar or cfg.sensor.enable_camera:
            enable_sensor_extensions(
                enable_lidar=use_rtx_lidar,
                enable_camera=cfg.sensor.enable_camera,
            )
            import quadruped.go2.go2_sensors as go2_sensors

            sm = go2_sensors.SensorManager(cfg.num_envs)
            lidar_annotators = sm.add_rtx_lidar() if use_rtx_lidar else []
            cameras = sm.add_camera(cfg.freq) if cfg.sensor.enable_camera else []

        # Keyboard control
        try:
            system_input = carb.input.acquire_input_interface()
            system_input.subscribe_to_keyboard_events(
                omni.appwindow.get_default_app_window().get_keyboard(), go2_ctrl.sub_keyboard_event)
        except RuntimeError:
            print("[WARN] Keyboard input is unavailable; continuing without keyboard teleop.", flush=True)
        
        # ROS2 Bridge and YOLO setup
        dm = None
        if cfg.ros.enable:
            enable_sensor_extensions(
                enable_lidar=use_rtx_lidar,
                enable_camera=cfg.sensor.enable_camera,
                enable_ros2=True,
            )
            import ros2.go2_ros2_bridge as go2_ros2_bridge

            rclpy.init()
            dm = go2_ros2_bridge.RobotDataManager(env, lidar_annotators, cameras, cfg)
        # yolo_detector = YOLODetectorNode(cfg.num_envs)  # Create YOLO detector


        # Run simulation
        sim_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
        obs, _ = env.reset()
        if hasattr(policy, "reset_to_mujoco_default_pose"):
            policy.reset_to_mujoco_default_pose()

        step = 0
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():            
                # control joints
                actions = policy(obs)

                # step the environment
                obs, _, _, _ = env.step(actions)

                # ROS2 data publish
                if dm is not None:
                    dm.pub_ros2_data()
                    rclpy.spin_once(dm,timeout_sec=0.001)
            
                # Camera follow
                if (cfg.camera_follow and not getattr(args_cli, "headless", False)):
                    go2_env.camera_follow(env)

                # limit loop time
                elapsed_time = time.time() - start_time
                if elapsed_time < sim_step_dt:
                    sleep_duration = sim_step_dt - elapsed_time
                    time.sleep(sleep_duration)
                step += 1
                if args_cli.max_steps and step >= args_cli.max_steps:
                    break





    elif (cfg.robot == "spot"):
        env_cfg = spot_env.SpotRSLEnvCfg()
        env_cfg.scene.num_envs = cfg.num_envs
        env_cfg.decimation = math.ceil(1./env_cfg.sim.dt/cfg.freq)
        env_cfg.sim.render_interval = env_cfg.decimation
        spot_ctrl.init_base_vel_cmd(cfg.num_envs)
        env, policy = spot_ctrl.get_rsl_flat_policy(env_cfg)
    
    # # env, policy = go2_ctrl.get_rsl_flat_policy(go2_env_cfg)
    # env, policy = get_rsl_rough_policy(go2_env_cfg)      #-->Run rough terrain policy

        # Simulation environment
        if (cfg.env_name == "nav_demo"):
            create_nav_demo_env()

        elif (cfg.env_name == "obstacle"):
            create_obstacle_env() # obstacles

        elif (cfg.env_name == "warehouse"):
            create_warehouse_env() # warehouse

        elif (cfg.env_name == "office"):
            create_office_env() # office
        
        elif (cfg.env_name == "hospital"):
            create_hospital_env() # hospital
        
        elif (cfg.env_name == "rivermark"):
            create_rivermark_env() # rivermark

        elif (cfg.env_name == "terrain"):
            create_terrain_env() # terrain

    

        # Sensor setup
        lidar_annotators = []
        cameras = []
        if cfg.sensor.enable_lidar or cfg.sensor.enable_camera:
            enable_sensor_extensions(
                enable_lidar=cfg.sensor.enable_lidar,
                enable_camera=cfg.sensor.enable_camera,
            )
            import quadruped.spot.spot_sensors as spot_sensors

            sm = spot_sensors.SensorManager(cfg.num_envs)
            lidar_annotators = sm.add_rtx_lidar() if cfg.sensor.enable_lidar else []
            cameras = sm.add_camera(cfg.freq) if cfg.sensor.enable_camera else []

        # Keyboard control
        try:
            system_input = carb.input.acquire_input_interface()
            system_input.subscribe_to_keyboard_events(
                omni.appwindow.get_default_app_window().get_keyboard(), spot_ctrl.sub_keyboard_event)
        except RuntimeError:
            print("[WARN] Keyboard input is unavailable; continuing without keyboard teleop.", flush=True)
        
        # ROS2 Bridge and YOLO setup
        dm = None
        if cfg.ros.enable:
            enable_isaac_extensions("omni.replicator.core", "omni.syntheticdata", "isaacsim.ros2.bridge")
            import ros2.spot_ros2_bridge as spot_ros2_bridge

            rclpy.init()
            dm = spot_ros2_bridge.RobotDataManager(env, lidar_annotators, cameras, cfg)
        # yolo_detector = YOLODetectorNode(cfg.num_envs)  # Create YOLO detector


        # Run simulation
        sim_step_dt = float(env_cfg.sim.dt * env_cfg.decimation)
        obs, _ = env.reset()


        step = 0
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():            
                # control joints
                actions = policy(obs)

                # step the environment
                obs, _, _, _ = env.step(actions)

                # ROS2 data publish
                if dm is not None:
                    dm.pub_ros2_data()
                    rclpy.spin_once(dm,timeout_sec=0.001)
            
                # Camera follow
                if (cfg.camera_follow and not getattr(args_cli, "headless", False)):
                    spot_env.camera_follow(env)

                # limit loop time
                elapsed_time = time.time() - start_time
                if elapsed_time < sim_step_dt:
                    sleep_duration = sim_step_dt - elapsed_time
                    time.sleep(sleep_duration)
                step += 1
                if args_cli.max_steps and step >= args_cli.max_steps:
                    break
                # actual_loop_time = time.time() - start_time
                # rtf = min(1.0, sim_step_dt/elapsed_time)
                # print(f"\rStep time: {actual_loop_time*1000:.2f}ms, Real Time Factor: {rtf:.2f} | YOLO active", end='', flush=True)


    else:
        print("Please select a valid robot in cfg/sim.yaml: go2, spot")
    
    # Cleanup
    if "dm" in locals() and dm is not None:
        dm.destroy_node()
        rclpy.shutdown()
    if getattr(args_cli, "headless", False):
        os._exit(0)
    simulation_app.close()

if __name__ == "__main__":
    main()
