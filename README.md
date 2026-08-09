# WeatherGuard-Go2

**WeatherGuard-Go2** is a robust quadruped autonomous navigation simulation system designed for rain, fog, and slippery terrain. Adverse weather often introduces coupled challenges for legged robots: wet or icy ground reduces friction and threatens locomotion stability, while fog, haze, and rain-induced scattering degrade LiDAR perception through sparse returns, noisy measurements, false echoes, and outliers.

To address these challenges, WeatherGuard-Go2 integrates low-friction reinforcement learning locomotion, MuJoCo-to-Isaac sim2sim transfer, physics-based LiDAR fog simulation, confidence-aware elevation mapping, and uncertainty-guided navigation. The system uses a robust Go2 locomotion policy to maintain stable motion under varying ground friction, while degraded LiDAR point clouds are converted into elevation maps and confidence maps to estimate local perception reliability. This confidence information is then incorporated into local costmap construction or planning, reducing unnecessary detours, sudden stops, and path oscillations caused by weather-induced pseudo-obstacles.

WeatherGuard-Go2 aims to provide a unified simulation framework for studying quadruped navigation robustness under both slippery-ground locomotion risk and adverse-weather perception uncertainty.

## Acknowledgement

This repository is based on the following open-source projects:

- https://github.com/MartinHahner/LiDAR_fog_sim.git
- https://github.com/xbyyh/Go2-Low-Friction-Locomotion-Benchmark.git
- https://github.com/sallu-786/Go2_Isaac_ros2.git
我正在开发一个名为 **WeatherGuard-Go2** 的四足机器人鲁棒自主导航仿真系统，目标是在雨雾天气和湿滑地面等恶劣环境下提升 Unitree Go2 的运动稳定性与导航可靠性。

项目关注两个耦合问题：

1. **湿滑地面导致的运动不稳定**
   雨天、积水、结冰或潮湿地面会降低足端与地面的摩擦，使四足机器人更容易出现打滑、速度跟踪误差增大、姿态失稳甚至摔倒。

2. **雨雾天气导致的 LiDAR 感知退化**
   雾霾、水汽、雨滴和散射会使 LiDAR 点云出现稀疏化、随机噪声、虚假回波、异常点和局部空间不一致，从而在导航中产生伪障碍、急停、绕行或路径震荡。

项目整体思路是将低层运动鲁棒性和高层感知不确定性结合起来：

- 低层使用强化学习 locomotion policy，使 Go2 在不同地面摩擦条件下保持稳定运动。
- 仿真后端使用 Isaac Sim / Isaac Lab，将 MuJoCo 中训练得到的低摩擦 Go2 policy 迁移到 Isaac 中进行 sim2sim 验证。
- 感知层引入 LiDAR 雾天点云退化仿真，对 Isaac 产生的点云进行雾天退化，包括强度衰减、后向散射、点云稀疏化、距离噪声、虚假回波和异常点。
- 地图层将退化后的 3D 点云转换为局部 elevation map。
- 在 elevation map 的基础上，根据局部 voxel 内高度方差、点云数量、空间一致性等信息生成 confidence map，用于描述局部环境观测的可靠程度。
- 导航层将 confidence-aware elevation map 融入局部代价地图或局部规划过程。高置信度障碍保持较强避障约束；低置信度区域降低异常点云或伪障碍造成的惩罚，从而减少雨雾天气下不必要绕行、急停和路径震荡。

当前参考/组成仓库如下：

1. `Go2-Low-Friction-Locomotion-Benchmark-main`
   - 作用：提供 MuJoCo 中训练得到的 Go2 低摩擦强化学习 locomotion policy。
   - 已有模型包括 `slippery1.0_checkpoint/policy.onnx`。
   - 该 policy 是速度跟踪型低层控制器，输入包括机器人状态、速度命令、关节状态、历史 action 等，输出 12 维关节位置 action。
   - 目标是将这个 MuJoCo policy 迁移到 Isaac Sim 中，而不是重新训练底层控制器。

2. `Go2_Isaac_ros2-main`
   - 作用：作为 WeatherGuard-Go2 的 Isaac Sim / Isaac Lab 主仿真后端。
   - 推荐替代旧的 `isaac_go2_ros1_nav-main`，因为它是更干净的 ROS2 项目。
   - 环境建议为 Ubuntu 22.04 + ROS2 Humble + Python 3.10 + Isaac Sim 4.5.0 + Isaac Lab 2.1.1。
   - 该仓库已有 Go2 articulation、Isaac 场景、RTX LiDAR、相机、ROS2 bridge、RViz2 可视化、`/unitree_go2_0/cmd_vel`、`/unitree_go2_0/odom`、`/unitree_go2_0/lidar/point_cloud` 等接口。
   - 当前仓库默认加载自己的 RSL-RL policy，需要改造为加载 MuJoCo 导出的 ONNX policy，并严格对齐 observation/action。

3. `LiDAR_fog_sim-main`
   - 作用：作为 LiDAR 雾天退化仿真模块。
   - 核心文件是 `fog_simulation.py` 和 `integral_lookup_tables/`。
   - 它基于物理建模模拟雾天 LiDAR，包括衰减、后向散射、软目标回波、距离噪声等。
   - 项目中可以只移植核心雾化逻辑，不需要使用 GUI viewer。
   - 该模块将插入 Isaac LiDAR 点云发布流程中：Isaac 原始点云 -> fog simulation -> degraded point cloud。

推荐主线架构：

```text
MuJoCo low-friction RL policy
        ↓
policy.onnx
        ↓
Isaac Sim / Isaac Lab Go2 backend
        ↓
MuJoCo-style observation builder
        ↓
ONNX policy inference
        ↓
12 joint position targets
        ↓
Go2 articulation control
        ↓
Isaac RTX LiDAR point cloud
        ↓
LiDAR fog degradation module
        ↓
degraded point cloud
        ↓
elevation map
        ↓
confidence map
        ↓
confidence-aware local costmap / local planner
