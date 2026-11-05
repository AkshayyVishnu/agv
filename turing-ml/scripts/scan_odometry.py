"""Odometry for offline mapping: chain scan-to-scan matches of a recorded drive (laptop, no ROS).

Each scan is matched to the previous one (point-to-line ICP); the motion guess and a weak prior
come from the teleop log (VESC tachometer + commanded steering). Locally this is accurate to a few
cm per scan; it still drifts over laps, which slam_toolbox's loop closure then removes.

    python scripts/scan_odometry.py data/mapping/20261030_1103/map_20261030_110341 \
        --teleop data/mapping/20261030_1103/teleop_20261030_110337.csv

Writes <bag>/../<bag name>_odom.csv: stamp (scan header time), x, y, yaw of base_link (rear axle)
in the odom frame. scripts/make_map.sh turns it into /tf for slam_toolbox.
"""
import argparse
import glob
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bag_to_map import (LIDAR_X, ODO_SIGMA, WHEELBASE, PointMap, compose, fit_tacho,  # noqa: E402
                        odometry, scan_points, voxel)
from turing_ml.bagscan import read_scans                                                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="bag directory")
    ap.add_argument("--teleop", required=True, help="teleop_keyboard.py CSV log of the same drive")
    a = ap.parse_args()

    bag = pathlib.Path(a.bag)
    scans = read_scans(glob.glob(str(bag / "*.db3"))[0])
    ot, counts, steer = odometry(a.teleop, 1.0)

    def odo_motion(t0, t1, k):
        d = (np.interp(t1, ot, counts) - np.interp(t0, ot, counts)) * k
        return d, d * math.tan(np.interp(0.5 * (t0 + t1), ot, steer)) / WHEELBASE

    k = fit_tacho(scans, odo_motion)
    print(f"{len(scans)} scans; tachometer {k * 1000:.3f} mm per count")

    lidar = np.zeros(3)
    out, weak = [], 0
    prev = voxel(scan_points(*scans[0][1:]), 0.03)
    for j in range(len(scans)):
        if j:
            d, dyaw = odo_motion(scans[j - 1][0], scans[j][0], k)
            # rear axle moves along an arc; the lidar sits LIDAR_X ahead of it, so it also swings sideways
            guess = np.array([d * math.cos(dyaw / 2) + LIDAR_X * (math.cos(dyaw) - 1),
                              d * math.sin(dyaw / 2) + LIDAR_X * math.sin(dyaw), dyaw])
            cur = voxel(scan_points(*scans[j][1:]), 0.03)
            pm = PointMap()
            pm.add(prev)
            step, inl, _ = pm.match(cur, guess, prior_sigma=ODO_SIGMA)
            if inl < 0.7:
                step, weak = guess, weak + 1                              # no good match: wheel odometry only
            lidar = compose(lidar, step)
            prev = cur
        base = compose(lidar, [-LIDAR_X, 0.0, 0.0])
        out.append((scans[j][0], *base))
    out = np.array(out)
    path = np.sum(np.hypot(*np.diff(out[:, 1:3], axis=0).T))
    print(f"path {path:.1f} m, total turn {math.degrees(out[-1, 3]):+.0f} deg, {weak} scans used wheel odometry only")
    dst = bag.parent / f"{bag.name}_odom.csv"
    np.savetxt(dst, out, delimiter=", ", fmt=["%.6f", "%.4f", "%.4f", "%.5f"],
               header="stamp_s, x_m, y_m, yaw_rad (base_link = rear axle, in odom)")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
