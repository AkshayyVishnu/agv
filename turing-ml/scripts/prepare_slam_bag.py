"""Rewrite a mapping bag for slam_toolbox (runs inside the turing/mapping container, needs ROS).

- /scan resampled to a fixed BEAMS beams from 0 rad (the D500 gives 498-505 per turn; slam_toolbox
  wants the same count every scan), range_min raised to 0.15 m (car body returns), no-return = inf
- /tf odom -> base_link from <bag>_odom.csv (scripts/scan_odometry.py), one per scan, scan's stamp
- the recorded /tf_static (old 0, 0, 0.18 mount) is dropped; make_map.sh publishes the measured
  mount (0.225, 0, 0.13) with static_transform_publisher (a bag-played /tf_static has the wrong QoS).
  The scan frame name is written to <out bag>.frame for it.

    python3 prepare_slam_bag.py <bag dir> <odom.csv> <out bag dir>
"""
import math
import sys

import numpy as np
import rosbag2_py
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

BEAMS = 500
R_MIN = 0.15
BODY_R = 0.4         # behind the lidar (cos(angle) < -0.7) closer than this is the car's rear body


def tf(stamp, parent, child, x, y, z, yaw):
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
    t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return t


def main():
    src, odom_csv, dst = sys.argv[1:4]
    odom = np.loadtxt(odom_csv, delimiter=",", comments="#")

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=src, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("cdr", "cdr"))
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=dst, storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    for name, typ in (("/scan", "sensor_msgs/msg/LaserScan"), ("/tf", "tf2_msgs/msg/TFMessage")):
        w.create_topic(rosbag2_py.TopicMetadata(name=name, type=typ, serialization_format="cdr"))

    target = 2 * math.pi * np.arange(BEAMS) / BEAMS
    n, frame = 0, None
    while r.has_next():
        topic, data, t_ns = r.read_next()
        if topic != "/scan":
            continue
        s = deserialize_message(data, LaserScan)
        stamp = s.header.stamp.sec + s.header.stamp.nanosec * 1e-9
        frame = s.header.frame_id
        ranges = np.asarray(s.ranges, float)
        ang = (s.angle_min + s.angle_increment * np.arange(len(ranges))) % (2 * math.pi)
        # nearest original beam for each target angle (wrapping), none if further than one increment
        order = np.argsort(ang)
        ang, ranges = ang[order], ranges[order]
        ext_a = np.concatenate([ang[-1:] - 2 * math.pi, ang, ang[:1] + 2 * math.pi])
        ext_r = np.concatenate([ranges[-1:], ranges, ranges[:1]])
        i = np.clip(np.searchsorted(ext_a, target), 1, len(ext_a) - 1)
        left_closer = (target - ext_a[i - 1]) < (ext_a[i] - target)
        k = np.where(left_closer, i - 1, i)
        out = ext_r[k]
        out[np.abs(ext_a[k] - target) > s.angle_increment] = np.nan
        out[~np.isfinite(out) | (out < R_MIN) | ((np.cos(target) < -0.7) & (out < BODY_R))] = np.inf
        s.angle_min, s.angle_increment = 0.0, 2 * math.pi / BEAMS
        s.angle_max = s.angle_min + s.angle_increment * (BEAMS - 1)
        s.range_min = R_MIN
        s.ranges = out.astype(np.float32).tolist()
        s.intensities = []
        w.write("/scan", serialize_message(s), t_ns)

        j = int(np.argmin(np.abs(odom[:, 0] - stamp)))
        if abs(odom[j, 0] - stamp) > 0.02:
            raise SystemExit(f"no odometry within 20 ms of scan at {stamp:.3f}")
        _, x, y, yaw = odom[j]
        w.write("/tf", serialize_message(TFMessage(transforms=[tf(s.header.stamp, "odom", "base_link", x, y, 0.0, yaw)])),
                t_ns)
        n += 1
    del w
    with open(dst.rstrip("/") + ".frame", "w") as f:
        f.write(frame)
    print(f"wrote {dst}: {n} scans resampled to {BEAMS} beams, scan frame '{frame}', odom tf per scan")


if __name__ == "__main__":
    main()
