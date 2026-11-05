#!/bin/bash
# Start the lidar (if it isn't running) and the disparity-extender driver in the `car` container.
#   ssh -t pi ./drive_car.sh [--max-speed 1.0] [--set KEY=VALUE ...]
# Copy to the Pi's home: it runs scripts/drive_car.py from the mounted ~/agv/turing-ml.
set -u
C=car
ROS='source /opt/ros/humble/setup.bash; source /root/car_ws/install/setup.bash; export ROS_DOMAIN_ID=7'
docker start $C >/dev/null || exit 1
if ! docker exec $C pgrep -f ldlidar_stl_ros2_node >/dev/null; then
  docker exec -d $C bash -c "$ROS; ros2 launch ldlidar_stl_ros2 ld19.launch.py > /tmp/lidar.log 2>&1"
  echo "LiDAR driver starting..."; sleep 6
fi
docker exec -it -e DRIVE_LOGS=/root/car_ws/drive_logs $C bash -c "$ROS; python3 /root/turing-ml/scripts/drive_car.py $*"
