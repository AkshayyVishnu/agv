#!/bin/bash
# Map a recorded drive with slam_toolbox, offline on the laptop (WSL + Docker).
#
#   docker build -t turing/mapping:humble -f docker/mapping.Dockerfile docker      # once
#   python scripts/scan_odometry.py <bag dir> --teleop <teleop csv>               # writes <bag>_odom.csv
#   bash scripts/make_map.sh <bag dir> <out map path without extension>
#
# e.g. bash scripts/make_map.sh data/mapping/20261030_1103/map_20261030_110341 maps/TuringTrack2Real/TuringTrack2Real_map
# Paths are relative to turing-ml/. Writes <out>.pgm/.yaml (ROS map_saver format) and <out>_slam.log.
set -eu
cd "$(dirname "$0")/.."
BAG=$1; OUT=$2; RATE=${RATE:-0.5}
mkdir -p "$(dirname "$OUT")"
rm -rf "${BAG}_slam"
docker run --rm -v "$PWD":/w -w /w turing/mapping:humble bash -c "
  set -e; source /opt/ros/humble/setup.bash
  python3 scripts/prepare_slam_bag.py $BAG ${BAG}_odom.csv ${BAG}_slam
  # lidar mount (car_spec.yaml); a bag-played /tf_static has the wrong QoS for tf listeners
  ros2 run tf2_ros static_transform_publisher --x 0.225 --y 0 --z 0.13 --frame-id base_link \
      --child-frame-id \$(cat ${BAG}_slam.frame) --ros-args -p use_sim_time:=true > /dev/null 2>&1 &
  ros2 run slam_toolbox sync_slam_toolbox_node --ros-args --params-file ${SLAM_PARAMS:-configs/slam_toolbox.yaml} > ${OUT}_slam.log 2>&1 &
  sleep 3
  ros2 bag play ${BAG}_slam --clock -r $RATE > /dev/null
  sleep 5
  ros2 run nav2_map_server map_saver_cli -f $OUT --ros-args -p use_sim_time:=true -p save_map_timeout:=10.0 2>&1 | grep -iE 'saving|error|wrote|fail' || true
  kill %1 %2 2>/dev/null || true
  grep -ciE 'loop closure|closed loop' ${OUT}_slam.log | sed 's/^/loop closure log lines: /' || true
"
ls -l "$OUT".*
