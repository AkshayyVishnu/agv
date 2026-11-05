#!/bin/bash
# Record the LiDAR for mapping, on the Pi (runs the D500 driver and ros2 bag inside the `car` container).
#
#   ./map_record.sh start     # LiDAR on, check /scan, start recording to ~/car_ws/bags/map_<time>
#   ./map_record.sh status    # is it recording, how big is the bag
#   ./map_record.sh stop      # stop recording (bag closed cleanly), LiDAR off, print bag info
#
# Drive with teleop_keyboard.py in a second SSH window meanwhile; its CSV log (ERPM, tacho, clock time)
# is the wheel odometry for building the map offline.
set -u
C=car
ROS='source /opt/ros/humble/setup.bash; source /root/car_ws/install/setup.bash; export ROS_DOMAIN_ID=7'
BAGS=/root/car_ws/bags

case "${1:-}" in
start)
  docker start $C >/dev/null || exit 1
  if docker exec $C pgrep -f "ros2 bag record" >/dev/null; then echo "already recording"; exit 1; fi
  docker exec $C mkdir -p $BAGS
  if ! docker exec $C pgrep -f ldlidar_stl_ros2_node >/dev/null; then
    docker exec -d $C bash -c "$ROS; ros2 launch ldlidar_stl_ros2 ld19.launch.py > $BAGS/lidar.log 2>&1"
    echo "LiDAR driver starting..."; sleep 6
  fi
  hz=$(docker exec $C bash -c "$ROS; timeout 6 ros2 topic hz /scan 2>/dev/null | grep -m1 'average rate'")
  if [ -z "$hz" ]; then
    echo "NO /scan messages. LiDAR log:"; docker exec $C tail -20 $BAGS/lidar.log; exit 1
  fi
  echo "/scan OK: $hz"
  name=map_$(date +%Y%m%d_%H%M%S)
  docker exec -d $C bash -c "$ROS; cd $BAGS && ros2 bag record -o $name /scan /tf_static > $BAGS/$name.log 2>&1"
  sleep 3
  docker exec $C pgrep -f "ros2 bag record" >/dev/null && echo "RECORDING -> ~/car_ws/bags/$name  (stop with: ./map_record.sh stop)" \
    || { echo "recorder failed:"; docker exec $C cat $BAGS/$name.log; exit 1; }
  ;;
status)
  docker exec $C pgrep -f "ros2 bag record" >/dev/null && echo "recording" || echo "not recording"
  docker exec $C pgrep -f ldlidar_stl_ros2_node >/dev/null && echo "lidar driver running" || echo "lidar driver off"
  ls -t ~/car_ws/bags 2>/dev/null | grep '^map_' | grep -v log | head -1 | while read b; do du -sh ~/car_ws/bags/$b; done
  ;;
stop)
  docker exec $C pkill -INT -f "ros2 bag record" && echo "stopping recorder..." && sleep 3
  docker exec $C pkill -INT -f "ros2 launch ldlidar_stl_ros2" ; docker exec $C pkill -INT -f ldlidar_stl_ros2_node
  b=$(ls -t ~/car_ws/bags | grep '^map_' | grep -v log | head -1)
  [ -n "$b" ] && docker exec $C bash -c "$ROS; ros2 bag info $BAGS/$b" | grep -E "Duration|Messages|Topic|Bag size"
  ;;
*) sed -n 2,9p "$0" ;;
esac
