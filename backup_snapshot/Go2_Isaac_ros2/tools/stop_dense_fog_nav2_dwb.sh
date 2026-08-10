#!/usr/bin/env bash
set -euo pipefail

patterns=(
  "main.py env_name=nav_demo"
  "tools/odom_relay.py"
  "tools/elevation_obstacle_cloud.py"
  "nav2_controller controller_server"
  "nav2_planner planner_server"
  "nav2_smoother smoother_server"
  "nav2_behaviors behavior_server"
  "nav2_bt_navigator bt_navigator"
  "nav2_waypoint_follower waypoint_follower"
  "nav2_lifecycle_manager lifecycle_manager"
  "/opt/ros/humble/lib/nav2_controller/controller_server"
  "/opt/ros/humble/lib/nav2_planner/planner_server"
  "/opt/ros/humble/lib/nav2_smoother/smoother_server"
  "/opt/ros/humble/lib/nav2_behaviors/behavior_server"
  "/opt/ros/humble/lib/nav2_bt_navigator/bt_navigator"
  "/opt/ros/humble/lib/nav2_waypoint_follower/waypoint_follower"
  "/opt/ros/humble/lib/nav2_lifecycle_manager/lifecycle_manager"
)

for pattern in "${patterns[@]}"; do
  pkill -f "$pattern" 2>/dev/null || true
done

sleep 1

for pattern in "${patterns[@]}"; do
  pkill -9 -f "$pattern" 2>/dev/null || true
done

echo "Stopped dense fog Nav2 demo processes."
