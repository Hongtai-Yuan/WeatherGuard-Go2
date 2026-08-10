# WeatherGuard-Go2

[中文](README.md) | [English](README_en.md)

WeatherGuard-Go2 是一个面向恶劣天气和湿滑地面的 Unitree Go2 四足机器人鲁棒自主导航仿真项目。项目关注四足机器人导航中的两个耦合问题：雨天、积水、结冰或低摩擦地面会降低足端与地面的接触稳定性，而雾、雨和水汽散射会导致 LiDAR 点云稀疏化、距离噪声、虚假回波和异常点，从而影响局部建图与路径规划。

本项目将低摩擦强化学习运动控制、MuJoCo 到 Isaac 的 sim2sim 迁移、实时 LiDAR 雾天退化、置信度感知高程建图，以及 ROS2/Nav2 导航验证整合到同一个仿真框架中。

## 项目结构

```text
WeatherGuard-Go2/
├── locomotion_rl/             # PPO 低摩擦 Go2 locomotion 训练与评估
├── low_friction_benchmark/    # MuJoCo 低摩擦 benchmark 与导出的 ONNX policy
├── isaac_nav_sim/             # Isaac Sim / Isaac Lab Go2 后端、ROS2 bridge、Nav2、RViz
├── lidar_fog_sim/             # 基于物理模型的 LiDAR 雾天退化仿真和 lookup table
├── README.md
└── README_en.md
```

## 主要贡献

- 基于 PPO 训练 Unitree Go2 低摩擦速度跟踪型 locomotion policy。奖励设计包括线速度/角速度跟踪、机体姿态稳定、关节能耗约束、动作平滑、足端打滑抑制和步态相位约束，使策略能够在不同地面摩擦系数下学习稳定行走；随后将训练得到的 ONNX policy 从 MuJoCo 迁移到 Isaac Sim / Isaac Lab，并对齐 observation、action、PD 控制参数和系统辨识后的动力学参数，完成低摩擦运动控制的 sim2sim 验证。

- 构建恶劣天气 LiDAR 感知退化与湿滑地面运动风险耦合的仿真流程。Isaac 实时 LiDAR 点云流接入雾天退化模型，模拟强度衰减、后向散射、点云稀疏化、距离噪声和虚假回波；同时将天气严重程度映射到地面摩擦系数，使浓雾和湿滑地面能够作为复合恶劣环境进行导航验证。

- 构建置信度感知的局部建图与导航链路。退化后的 3D 点云被转换为局部高程图、高度方差图、置信度图和障碍图，并利用局部栅格内高度方差、点云数量和空间一致性估计观测可靠性，生成 confidence-aware cost 表达，降低低置信度伪障碍对规划的影响；同时将过滤后的 obstacle cloud 和 clearing cloud 接入 ROS2/Nav2，使 DWB 能够在浓雾场景下生成有效速度指令并驱动 Go2 完成目标点导航。

## 核心模块

`locomotion_rl/` 包含强化学习任务、PPO 训练脚本、Go2 速度跟踪环境配置、奖励函数、观测项和策略评估工具。

`isaac_nav_sim/` 包含 Isaac Sim / Isaac Lab Go2 仿真后端、MuJoCo ONNX policy 加载器、ROS2 bridge、雾天 LiDAR 实时处理、Nav2 配置、RViz 配置和运行脚本。

`lidar_fog_sim/` 包含基于物理建模的 LiDAR 雾天退化仿真实现，以及实时雾化流程使用的 integral lookup tables。

`low_friction_benchmark/` 包含 MuJoCo 低摩擦 locomotion 实验中的 benchmark 配置、评估脚本和导出的策略文件。

## 快速运行

从 Isaac/Nav2 模块启动浓雾场景下的 Go2 导航 demo：

```bash
cd isaac_nav_sim
./tools/run_dense_fog_nav2_dwb.sh
```

停止 demo：

```bash
cd isaac_nav_sim
./tools/stop_dense_fog_nav2_dwb.sh
```

该 demo 会启动 Isaac Sim / Isaac Lab，加载 MuJoCo 训练得到的 Go2 ONNX locomotion policy，启用浓雾 LiDAR 退化，发布 elevation/confidence/obstacle maps，启动 ROS2/Nav2，并打开 RViz 用于目标点导航验证。

## 致谢

本项目基于并改造了以下开源项目中的组件：

- https://github.com/MartinHahner/LiDAR_fog_sim.git
- https://github.com/xbyyh/Go2-Low-Friction-Locomotion-Benchmark.git
- https://github.com/sallu-786/Go2_Isaac_ros2.git
