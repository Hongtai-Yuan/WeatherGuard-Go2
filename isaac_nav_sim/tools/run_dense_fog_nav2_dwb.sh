#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PYTHON:-/home/yuan/anaconda3/envs/env_isaaclab/bin/python}

cd "$REPO"
if [ -f /opt/ros/humble/setup.bash ]; then
  set +u
  source /opt/ros/humble/setup.bash
  set -u
fi

if ! ros2 pkg prefix nav2_controller >/dev/null 2>&1; then
  echo "Nav2 is not installed in this ROS2 environment." >&2
  echo "Install it with: sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup" >&2
  exit 1
fi

existing="$(pgrep -af 'main.py env_name=nav_demo|nav2_controller controller_server|nav2_planner planner_server|nav2_smoother smoother_server|nav2_behaviors behavior_server|nav2_bt_navigator bt_navigator|nav2_waypoint_follower waypoint_follower|nav2_lifecycle_manager lifecycle_manager|tools/odom_relay.py|tools/elevation_obstacle_cloud.py' || true)"
if [ -n "$existing" ]; then
  echo "Existing Go2/Nav2 processes are still running. Stop them before starting a new demo:" >&2
  echo "$existing" >&2
  echo "A quick cleanup command is:" >&2
  echo "  pkill -f 'main.py env_name=nav_demo|nav2_.*server|bt_navigator|waypoint_follower|lifecycle_manager|tools/odom_relay.py|tools/elevation_obstacle_cloud.py'" >&2
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

sleep 8
"$PY" tools/wait_for_go2_tf.py --timeout 40

"$PY" tools/odom_relay.py \
  --source /unitree_go2_0/odom \
  --target /odom &
PIDS+=("$!")

"$PY" tools/elevation_obstacle_cloud.py \
  --map-topic /unitree_go2_0/elevation/obstacle_map \
  --cloud-topic /unitree_go2_0/elevation/obstacle_cloud \
  --threshold 80 \
  --z 0.25 &
PIDS+=("$!")

ros2 run nav2_controller controller_server \
  --ros-args \
  --params-file "$REPO/nav2/dense_fog_dwb_nav2_params.yaml" \
  -r /tf:=tf \
  -r /tf_static:=tf_static \
  -r cmd_vel:=/unitree_go2_0/cmd_vel &
PIDS+=("$!")

ros2 run nav2_planner planner_server \
  --ros-args \
  --params-file "$REPO/nav2/dense_fog_dwb_nav2_params.yaml" \
  -r /tf:=tf \
  -r /tf_static:=tf_static &
PIDS+=("$!")

ros2 run nav2_smoother smoother_server \
  --ros-args \
  --params-file "$REPO/nav2/dense_fog_dwb_nav2_params.yaml" \
  -r /tf:=tf \
  -r /tf_static:=tf_static &
PIDS+=("$!")

ros2 run nav2_behaviors behavior_server \
  --ros-args \
  --params-file "$REPO/nav2/dense_fog_dwb_nav2_params.yaml" \
  -r /tf:=tf \
  -r /tf_static:=tf_static &
PIDS+=("$!")

ros2 run nav2_bt_navigator bt_navigator \
  --ros-args \
  --params-file "$REPO/nav2/dense_fog_dwb_nav2_params.yaml" \
  -r /tf:=tf \
  -r /tf_static:=tf_static &
PIDS+=("$!")

ros2 run nav2_waypoint_follower waypoint_follower \
  --ros-args \
  --params-file "$REPO/nav2/dense_fog_dwb_nav2_params.yaml" &
PIDS+=("$!")

sleep 10

ros2 run nav2_lifecycle_manager lifecycle_manager \
  --ros-args \
  --params-file "$REPO/nav2/dense_fog_dwb_nav2_params.yaml" \
  -p use_sim_time:=false \
  -p autostart:=true \
  -p node_names:="[controller_server,smoother_server,planner_server,behavior_server,bt_navigator,waypoint_follower]" &
PIDS+=("$!")

rviz2 -d "$REPO/rviz/go2.rviz"
