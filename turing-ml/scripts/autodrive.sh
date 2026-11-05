#!/bin/bash
# Car power switch = start/stop. On the Pi (copy to ~): when the VESC appears on USB (car switched
# on), run the disparity extender with no keyboard (drive_car.py --auto: warning wiggle, 5 s, drive).
# Switching the car off cuts the motor; the VESC disappears and the driver stops and centres.
#
#   ./autodrive.sh enable [SPEED]   # turn the feature on (default 0.7 m/s cap); survives reboots
#   ./autodrive.sh disable          # off: stops a running drive; keyboard/SSH driving work again
#   ./autodrive.sh status           # on/off, driving or waiting, last log lines
#   ./autodrive.sh record on|off    # also record /scan during every run (~/car_ws/bags/map_auto_*), for mapping
#   ./autodrive.sh run              # the watcher loop (started by enable and at boot by cron)
#
# After any stop (switched off, no scans, run ended) it waits for the VESC to disappear: to drive
# again, switch the car off and on. While disabled, the watcher does nothing.
set -u
FLAG=~/.autodrive_enabled
RECFLAG=~/.autodrive_record
LOGDIR=~/car_ws/drive_logs
LOG=$LOGDIR/autodrive.log
C=car
ROS='source /opt/ros/humble/setup.bash; source /root/car_ws/install/setup.bash; export ROS_DOMAIN_ID=7'
SELF=$(readlink -f "$0")
CRON="@reboot $SELF run >> $LOG 2>&1"

say() { echo "$(date '+%F %T') $*"; }
vesc_present() { ls /dev/ttyACM* >/dev/null 2>&1; }
watcher_pid() { pgrep -f "autodrive.sh run" | grep -v "^$$\$" | head -1; }
# keyboard teleop on the Pi, or drive_car.py without --auto in the container ([p] keeps grep from matching itself)
manual_running() { pgrep -f "[t]eleop_keyboard.py" >/dev/null ||
                   docker exec $C bash -c 'pgrep -af "[p]ython3 .*drive_car.py" | grep -qv -- --auto' 2>/dev/null; }

case "${1:-}" in
enable)
  echo "${2:-0.7}" > "$FLAG"
  mkdir -p "$LOGDIR"
  docker update --restart unless-stopped $C >/dev/null
  (crontab -l 2>/dev/null | grep -vF "$SELF run"; echo "$CRON") | crontab -
  [ -n "$(watcher_pid)" ] || { setsid nohup "$SELF" run >> "$LOG" 2>&1 < /dev/null & }
  echo "autodrive ON (cap $(cat "$FLAG") m/s). Switch the car on: steering wiggles, 5 s later it drives."
  vesc_present && echo "NOTE: the VESC is on now. It will NOT start until you switch the car off and on."
  ;;
disable)
  rm -f "$FLAG"
  docker exec $C pkill -TERM -f "drive_car.py --auto" 2>/dev/null && echo "stopped the running drive"
  echo "autodrive OFF (the watcher stays idle; keyboard and SSH driving work)"
  ;;
record)
  case "${2:-}" in
    on) touch "$RECFLAG"; echo "recording ON: every autonomous run is recorded to ~/car_ws/bags/map_auto_<time>" ;;
    off) rm -f "$RECFLAG"; echo "recording OFF" ;;
    *) echo "usage: $0 record on|off" ;;
  esac
  ;;
status)
  [ -f "$FLAG" ] && echo "autodrive ON (cap $(cat "$FLAG") m/s)" || echo "autodrive OFF"
  [ -f "$RECFLAG" ] && echo "recording ON" || echo "recording OFF"
  [ -n "$(watcher_pid)" ] && echo "watcher running" || echo "watcher NOT running"
  docker exec $C pgrep -f "drive_car.py --auto" >/dev/null 2>&1 && echo "DRIVING (or arming)" || echo "not driving"
  vesc_present && echo "VESC present (car on)" || echo "VESC absent (car off)"
  tail -8 "$LOG" 2>/dev/null
  ;;
run)
  say "watcher started"
  # a VESC already on when the watcher starts does NOT count as switching on, also at boot: the Pi
  # reboots on under-voltage, and the car must not drive off by itself after that
  armed_needs_off=1
  while true; do
    if [ ! -f "$FLAG" ]; then sleep 2; continue; fi
    if ! vesc_present; then armed_needs_off=0; sleep 0.5; continue; fi
    if [ $armed_needs_off = 1 ]; then sleep 0.5; continue; fi
    if manual_running; then say "manual driving program running: not starting"; armed_needs_off=1; continue; fi
    say "VESC on: starting"
    sleep 2                                       # let USB enumeration settle
    docker start $C >/dev/null
    # the container sees only the /dev it started with: recreate the VESC's port inside it
    for dev in /dev/ttyACM*; do
      [ -e "$dev" ] || continue
      docker exec $C sh -c "rm -f $dev; mknod $dev c $((0x$(stat -c %t "$dev"))) $((0x$(stat -c %T "$dev"))); chmod 660 $dev"
    done
    if ! docker exec $C pgrep -f ldlidar_stl_ros2_node >/dev/null; then
      docker exec -d $C bash -c "$ROS; ros2 launch ldlidar_stl_ros2 ld19.launch.py > /tmp/lidar.log 2>&1"
      sleep 6
    fi
    if [ -f "$RECFLAG" ]; then
      bag=map_auto_$(date +%Y%m%d_%H%M%S)
      docker exec $C mkdir -p /root/car_ws/bags
      docker exec -d $C bash -c "$ROS; cd /root/car_ws/bags && ros2 bag record -o $bag /scan > $bag.log 2>&1"
      say "recording /scan -> ~/car_ws/bags/$bag"
    fi
    docker exec -e DRIVE_LOGS=/root/car_ws/drive_logs $C bash -c \
      "$ROS; python3 /root/turing-ml/scripts/drive_car.py --auto --max-speed $(cat "$FLAG" 2>/dev/null || echo 0.7)" < /dev/null
    docker exec $C pkill -INT -f "ros2 bag record" && say "recording stopped"
    say "drive ended; switch the car off and on to drive again"
    armed_needs_off=1
  done
  ;;
*) sed -n 2,13p "$0" ;;
esac
