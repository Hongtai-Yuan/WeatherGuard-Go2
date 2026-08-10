#!/usr/bin/env bash
set -euo pipefail

REPO=/home/yuan/WeatherGuard-Go2/Go2_Isaac_ros2
PY=/home/yuan/anaconda3/envs/env_isaaclab/bin/python

cd "$REPO"
if [ -f /opt/ros/humble/setup.bash ]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

if ! command -v rviz2 >/dev/null 2>&1; then
  echo "rviz2 not found. Please source ROS2 Humble or install rviz2." >&2
  exit 1
fi

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

"$PY" main.py \
  env_name=nav_demo \
  sensor.enable_lidar=true \
  sensor.lidar_backend=geometry \
  sensor.enable_camera=false \
  weather.preset=dense_fog \
  ros.enable=true \
  ros.use_sim_time=false &
PIDS+=("$!")

sleep 12

"$PY" ros2/go2_goal_nav.py \
  --max_vx 0.85 \
  --max_wz 1.2 \
  --obstacle_slow_vx 0.10 \
  --obstacle_max_x 1.35 \
  --obstacle_half_width 0.42 \
  --map_obstacle_max_x 1.45 \
  --map_obstacle_half_width 0.46 \
  --map_obstacle_cells 4 \
  --map_confidence_min 35 \
  --local_planner_sectors 15 \
  --local_planner_fov 1.25 \
  --local_planner_range 2.2 \
  --local_planner_goal_bias 0.35 \
  --local_planner_free_vx 0.22 \
  --low_conf_slow_vx 0.30 &
PIDS+=("$!")

rviz2 -d "$REPO/rviz/go2.rviz"
