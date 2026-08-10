# WeatherGuard-Go2

WeatherGuard-Go2 is a Unitree Go2 robust autonomous navigation simulation project for adverse weather and slippery terrain. It studies a coupled problem in legged robot navigation: wet, icy, or low-friction ground reduces locomotion stability, while fog and rain degrade LiDAR perception through sparse returns, range noise, false echoes, and outliers.

The project integrates low-friction reinforcement learning locomotion, MuJoCo-to-Isaac sim2sim transfer, real-time LiDAR fog degradation, confidence-aware elevation mapping, and ROS2/Nav2 navigation validation.

## Project Structure

```text
WeatherGuard-Go2/
├── locomotion_rl/             # PPO low-friction Go2 locomotion training and evaluation
├── low_friction_benchmark/    # MuJoCo low-friction policy benchmark and exported ONNX policies
├── isaac_nav_sim/             # Isaac Sim / Isaac Lab Go2 backend, ROS2 bridge, Nav2, RViz
├── lidar_fog_sim/             # Physics-based LiDAR fog simulation and lookup tables
└── README.md
```

## Main Contributions

- Trained a Unitree Go2 low-friction velocity-tracking locomotion policy with PPO. The reward design includes linear/angular velocity tracking, body attitude stabilization, joint energy regularization, action smoothness, foot slip suppression, and gait phase constraints, enabling stable walking under different ground friction coefficients. The trained ONNX policy is transferred from MuJoCo to Isaac Sim / Isaac Lab with aligned observations, actions, PD control parameters, and identified dynamics parameters for sim2sim validation.

- Built a coupled adverse-weather simulation pipeline that links LiDAR perception degradation with slippery terrain. The Isaac real-time LiDAR stream is degraded with fog effects such as intensity attenuation, backscatter, point sparsification, range noise, and false returns. Weather severity is also mapped to ground friction so that dense fog and wet/slippery terrain can be evaluated as a combined navigation stress case.

- Constructed confidence-aware local mapping and navigation. Degraded 3D point clouds are converted into local elevation, height variance, confidence, and obstacle maps. Local cell height variance, point density, and spatial consistency are used to estimate perception reliability, producing a confidence-aware cost representation that reduces the impact of low-confidence pseudo-obstacles. The filtered obstacle cloud and clearing cloud are integrated with ROS2/Nav2 so DWB can generate valid velocity commands for Go2 navigation under dense fog.

## Key Modules

`locomotion_rl/` contains the reinforcement learning task, PPO training scripts, Go2 velocity environment configuration, reward terms, observations, and policy evaluation utilities.

`isaac_nav_sim/` contains the Isaac Sim / Isaac Lab Go2 simulation backend, MuJoCo ONNX policy loader, ROS2 bridge, fog-aware LiDAR processing, Nav2 configuration, RViz setup, and run scripts.

`lidar_fog_sim/` contains the physics-based fog simulation implementation and integral lookup tables used by the real-time weather degradation pipeline.

`low_friction_benchmark/` contains benchmark configurations, evaluation scripts, and exported policies from the MuJoCo low-friction locomotion experiments.

## Quick Start

Run the dense-fog Go2 Nav2 demo from the Isaac/Nav2 module:

```bash
cd isaac_nav_sim
./tools/run_dense_fog_nav2_dwb.sh
```

Stop the demo:

```bash
cd isaac_nav_sim
./tools/stop_dense_fog_nav2_dwb.sh
```

The demo launches Isaac Sim / Isaac Lab, loads the MuJoCo-trained Go2 ONNX locomotion policy, enables dense-fog LiDAR degradation, publishes elevation/confidence/obstacle maps, starts ROS2/Nav2, and opens RViz for goal-pose navigation.

## Acknowledgement

This project builds on and adapts components from:

- https://github.com/MartinHahner/LiDAR_fog_sim.git
- https://github.com/xbyyh/Go2-Low-Friction-Locomotion-Benchmark.git
- https://github.com/sallu-786/Go2_Isaac_ros2.git
