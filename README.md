# WeatherGuard-Go2

**WeatherGuard-Go2** is a robust quadruped autonomous navigation simulation system designed for rain, fog, and slippery terrain. Adverse weather often introduces coupled challenges for legged robots: wet or icy ground reduces friction and threatens locomotion stability, while fog, haze, and rain-induced scattering degrade LiDAR perception through sparse returns, noisy measurements, false echoes, and outliers.

To address these challenges, WeatherGuard-Go2 integrates low-friction reinforcement learning locomotion, MuJoCo-to-Isaac sim2sim transfer, physics-based LiDAR fog simulation, confidence-aware elevation mapping, and uncertainty-guided navigation. The system uses a robust Go2 locomotion policy to maintain stable motion under varying ground friction, while degraded LiDAR point clouds are converted into elevation maps and confidence maps to estimate local perception reliability. This confidence information is then incorporated into local costmap construction or planning, reducing unnecessary detours, sudden stops, and path oscillations caused by weather-induced pseudo-obstacles.

WeatherGuard-Go2 aims to provide a unified simulation framework for studying quadruped navigation robustness under both slippery-ground locomotion risk and adverse-weather perception uncertainty.

## Acknowledgement

This repository is based on the following open-source projects:

- https://github.com/MartinHahner/LiDAR_fog_sim.git
- https://github.com/xbyyh/Go2-Low-Friction-Locomotion-Benchmark.git
- https://github.com/sallu-786/Go2_Isaac_ros2.git
