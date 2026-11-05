"""Build an occupancy map of a real track from a mapping drive: ros2 bag of /scan (+ teleop log).

No ROS needed. Each scan is aligned to the map built so far (point-to-line ICP); the motion guess
between scans comes from the teleop log (VESC tachometer + commanded steering) when given, else
from constant velocity. Scan-to-map matching closes small loops by itself on a track this size:
on lap 2 every scan is matched against the walls seen on lap 1.

    python scripts/bag_to_map.py data/mapping/20261030_1103/map_20261030_110341 \
        --teleop data/mapping/20261030_1103/teleop_20261030_110337.csv --out maps/TuringTrack2Real/TuringTrack2Real_map

Writes <out>.png/.yaml (gym/ROS map: white free, black wall or unknown), <out>_points.png (all
scan points + the car's path: check walls are thin and lap 2 lies on lap 1) and <out>_path.csv.
"""
import argparse
import glob
import math
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turing_ml.bagscan import read_scans          # noqa: E402

R_MIN, R_MAX = 0.15, 8.0      # m; < 0.15 m is the car itself (150-210 deg, measured in this bag)
BODY_R = 0.4                  # m; behind the lidar (cos(angle) < -0.7) the car's rear body returns at 0.25-0.35 m
WHEELBASE = 0.28
LIDAR_X = 0.225               # lidar ahead of the rear axle (car_spec.yaml)
SIGMA_PT = 0.03               # m, scan point noise in the matcher
ODO_SIGMA = (0.01, 0.01, 0.02)  # m, m, rad per scan (0.1 s): wheel odometry uncertainty


def scan_points(a0, inc, r):
    a = a0 + inc * np.arange(len(r))
    ok = np.isfinite(r) & (r > R_MIN) & (r < R_MAX) & ~((np.cos(a) < -0.7) & (r < BODY_R))
    return np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])], 1)


def voxel(p, size):
    _, i = np.unique(np.floor(p / size).astype(np.int64), axis=0, return_index=True)
    return p[np.sort(i)]


def transform(p, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return p @ np.array([[c, s], [-s, c]]) + pose[:2]


def compose(a, d):
    """pose a (x, y, yaw) followed by motion d expressed in a's frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * d[0] - s * d[1], a[1] + s * d[0] + c * d[1], a[2] + d[2]])


class PointMap:
    def __init__(self, voxel_size=0.03):
        self.v = voxel_size
        self.pts = np.zeros((0, 2))

    def add(self, p):
        self.pts = voxel(np.vstack([self.pts, p]), self.v)
        self.tree = cKDTree(self.pts)
        # line direction at each map point from its neighbours (for point-to-line distances)
        _, nb = self.tree.query(self.pts, k=min(6, len(self.pts)))
        q = self.pts[nb] - self.pts[nb].mean(1, keepdims=True)
        cov = np.einsum("nki,nkj->nij", q, q)
        w, vec = np.linalg.eigh(cov)
        self.normal = vec[:, :, 0]                                      # smallest eigenvector
        self.flat = w[:, 0] < 0.15 * np.maximum(w[:, 1], 1e-12)         # a line, not a corner/blob

    def match(self, p, guess, iters=25, prior_sigma=None):
        """ICP, point-to-line. prior_sigma (m, m, rad): pull towards `guess` (odometry) with this
        uncertainty, which decides the directions the walls can't (along a straight corridor).
        Returns (pose, inlier fraction, rms)."""
        pose = guess.copy()
        P = np.zeros((3, 3)) if prior_sigma is None else np.diag(1.0 / np.square(prior_sigma))
        for it in range(iters):
            gate = 0.4 if it < 5 else 0.2 if it < 12 else 0.1
            q = transform(p, pose)
            d, i = self.tree.query(q, distance_upper_bound=gate)
            ok = np.isfinite(d)
            ok[ok] &= self.flat[i[ok]]
            if ok.sum() < 30:
                return pose, ok.mean(), np.inf
            n, m, qq = self.normal[i[ok]], self.pts[i[ok]], q[ok]
            e = np.einsum("ij,ij->i", qq - m, n)
            # d e / d(x, y, yaw): n . (dx, dy) + n . R'(yaw) p
            rel = qq - pose[:2]
            J = np.stack([n[:, 0], n[:, 1], -n[:, 0] * rel[:, 1] + n[:, 1] * rel[:, 0]], 1)
            w = 1.0 / np.maximum(1.0, np.abs(e) / SIGMA_PT) / SIGMA_PT ** 2  # Huber, per point sigma
            H = J.T @ (J * w[:, None]) + P + np.diag([1e-4, 1e-4, 1e-4])
            step = -np.linalg.solve(H, J.T @ (w * e) + P @ (pose - guess))
            pose = pose + step
            if np.abs(step).max() < 1e-4:
                break
        q = transform(p, pose)
        d, _ = self.tree.query(q, distance_upper_bound=0.1)
        return pose, float(np.isfinite(d).mean()), float(np.sqrt(np.mean(d[np.isfinite(d)] ** 2)))


def odometry(teleop_csv, tacho_m):
    """(epoch, distance travelled (m), steer rad) from the teleop log; None if not given."""
    import csv
    rows = [r for r in csv.DictReader(open(teleop_csv)) if r["tacho"] not in ("", None)]
    t = np.array([float(r["epoch_s"]) for r in rows])
    tach = np.array([float(r["tacho"]) for r in rows])
    steer = np.array([float(r["steer_rad"]) for r in rows])
    return t, (tach - tach[0]) * tacho_m, steer


def fit_tacho(scans, odo_motion, n=300):
    """Metres per tachometer count: scan-to-scan ICP distance vs counts on n spread-out scan pairs."""
    fd, ft = [], []
    for j in np.linspace(1, len(scans) - 1, n).astype(int):
        counts = abs(odo_motion(scans[j - 1][0], scans[j][0], 1.0)[0])
        if counts < 5:
            continue
        pm = PointMap()
        pm.add(voxel(scan_points(*scans[j - 1][1:]), 0.03))
        est, inl, _ = pm.match(voxel(scan_points(*scans[j][1:]), 0.03), np.zeros(3))
        if inl > 0.9:
            fd.append(counts)
            ft.append(math.hypot(*est[:2]))
    fd, ft = np.array(fd), np.array(ft)
    return float(np.median(ft / fd))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", help="bag directory (or .db3)")
    ap.add_argument("--teleop", help="teleop_keyboard.py CSV log of the same drive")
    ap.add_argument("--tacho-m", type=float, default=None,
                    help="metres per tachometer count (default: fitted from the scan matching)")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--res", type=float, default=0.05)
    a = ap.parse_args()

    db = a.bag if a.bag.endswith(".db3") else glob.glob(str(pathlib.Path(a.bag) / "*.db3"))[0]
    scans = read_scans(db)
    print(f"{len(scans)} scans, {scans[-1][0] - scans[0][0]:.0f} s")

    odo = odometry(a.teleop, 1.0) if a.teleop else None              # tacho counts for now; scaled below

    def odo_motion(t0, t1, k):
        if odo is None:
            return None
        ot, dist, steer = odo
        d = (np.interp(t1, ot, dist) - np.interp(t0, ot, dist)) * k
        st = np.interp(0.5 * (t0 + t1), ot, steer)
        dyaw = d * math.tan(st) / WHEELBASE
        return d, dyaw

    k = a.tacho_m
    if odo is not None and k is None:
        k = fit_tacho(scans, odo_motion)
        print(f"tachometer scale fitted from scan-to-scan matches: {k * 1000:.3f} mm per count")

    pm = PointMap()
    pose = np.zeros(3)
    poses, quality = [pose.copy()], []
    pm.add(transform(voxel(scan_points(*scans[0][1:]), 0.03), pose))
    last_key = pose.copy()
    fit_d, fit_t = [], []
    vel = np.zeros(3)
    for j in range(1, len(scans)):
        t0, t1 = scans[j - 1][0], scans[j][0]
        p = voxel(scan_points(*scans[j][1:]), 0.03)
        om = odo_motion(t0, t1, 1.0)
        if om is not None:
            d, dyaw = om[0] * k, om[1] * k
            # rear axle moves along an arc; the lidar sits LIDAR_X ahead of it, so it also swings sideways
            guess = compose(pose, [d * math.cos(dyaw / 2) + LIDAR_X * (math.cos(dyaw) - 1),
                                   d * math.sin(dyaw / 2) + LIDAR_X * math.sin(dyaw), dyaw])
            new, inl, rms = pm.match(p, guess, prior_sigma=ODO_SIGMA)
            if inl < 0.5:                                                # walls not recognised: trust odometry
                new = guess
        else:
            new, inl, rms = pm.match(p, compose(pose, vel))
            if inl < 0.5:                                                # retry from no motion
                cand = pm.match(p, pose.copy())
                if cand[1] > inl:
                    new, inl, rms = cand
        c, s = math.cos(pose[2]), math.sin(pose[2])
        dx = new[:2] - pose[:2]
        vel = np.array([c * dx[0] + s * dx[1], -s * dx[0] + c * dx[1], new[2] - pose[2]])
        if om is not None:
            fit_d.append(abs(om[0]))
            fit_t.append(math.hypot(*dx))
        pose = new
        poses.append(pose.copy())
        quality.append((inl, rms))
        if math.hypot(*(pose[:2] - last_key[:2])) > 0.1 or abs(pose[2] - last_key[2]) > 0.1:
            pm.add(transform(p, pose))
            last_key = pose.copy()

    poses = np.array(poses)
    q = np.array(quality)
    bad = (q[:, 0] < 0.6).sum()
    print(f"scan match: inliers median {np.median(q[:, 0]):.2f}, min {q[:, 0].min():.2f}; "
          f"{bad} scans below 0.6; rms median {np.median(q[np.isfinite(q[:, 1]), 1]) * 100:.1f} cm")
    path_len = np.sum(np.hypot(*np.diff(poses[:, :2], axis=0).T))
    print(f"path {path_len:.1f} m; end is {np.hypot(*(poses[-1, :2] - poses[0, :2])):.2f} m and "
          f"{math.degrees((poses[-1, 2] - poses[0, 2] + math.pi) % (2 * math.pi) - math.pi):+.0f} deg from the start")
    if odo is not None and fit_d:
        fd, ft = np.array(fit_d), np.array(fit_t)
        mv = fd > 0
        print(f"tachometer: {np.sum(ft[mv]) / np.sum(fd[mv]) * 1000:.3f} mm per count over the whole drive "
              f"(lidar path {np.sum(ft):.1f} m / {np.sum(fd):.0f} counts)")

    render(scans, poses, a.out, a.res, np.vstack([[1.0, 0.0], q]))


def render(scans, poses, out, res, quality):
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    world = [transform(scan_points(*s[1:]), p) for s, p in zip(scans, poses)]
    allp = np.vstack(world)
    lo, hi = np.percentile(allp, 0.05, 0) - 0.5, np.percentile(allp, 99.95, 0) + 0.5
    w, h = int(math.ceil((hi[0] - lo[0]) / res)), int(math.ceil((hi[1] - lo[1]) / res))

    def rc(p):
        return (h - 1 - np.floor((p[:, 1] - lo[1]) / res)).astype(int), np.floor((p[:, 0] - lo[0]) / res).astype(int)

    hits = np.zeros((h, w), np.int32)
    seen = np.zeros((h, w), np.int32)                 # rays passing through
    for wp, p in zip(world, poses):
        r, c = rc(wp)
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        np.add.at(hits, (r[ok], c[ok]), 1)
        # free cells along each ray, stopping one cell short of the hit
        shorten = (1 - res / np.maximum(np.hypot(*(wp - p[:2]).T), res))[:, None]
        for f in np.linspace(0.0, 1.0, 40, endpoint=False):
            r2, c2 = rc(p[:2] + (wp - p[:2]) * f * shorten)
            ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
            np.add.at(seen, (r2[ok], c2[ok]), 1)
    wall = hits >= 3
    free = (seen >= 3) & ~ndimage.binary_dilation(wall)
    # keep the free region the car drove in
    lab, _ = ndimage.label(free)
    pr, pc = rc(poses[:, :2])
    ids = np.unique(lab[pr, pc])
    free = np.isin(lab, ids[ids > 0])
    free = ndimage.binary_closing(free, iterations=1) & ~wall
    Image.fromarray(np.where(free, 255, 0).astype(np.uint8)).save(str(out) + ".png")
    out.with_suffix(".yaml").write_text(
        f"image: {out.name}.png\nresolution: {res}\norigin: [{lo[0]:.3f}, {lo[1]:.3f}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    np.savetxt(str(out) + "_path.csv", np.hstack([np.array([sc[0] for sc in scans])[:, None], poses, quality]),
               delimiter=", ", fmt="%.4f", header="epoch_s, x_m, y_m, yaw_rad (lidar pose per scan), inlier_frac, rms_m")

    s = 4
    img = Image.new("RGB", (w * s, h * s), (255, 255, 255))
    d = ImageDraw.Draw(img)
    n = len(world)
    for i, wp in enumerate(world[::3]):
        f = i * 3 / n
        col = (int(200 * f), 0, int(200 * (1 - f)))                       # blue early -> red late
        r, c = rc(wp)
        for y, x in zip(r, c):
            d.point((x * s + 1, y * s + 1), fill=col)
    pr, pc = rc(poses[:, :2])
    d.line(list(zip(pc * s + 2, pr * s + 2)), fill=(0, 160, 0), width=1)
    d.text((6, 6), "scan points: blue = early, red = late; green = car path", fill=(0, 0, 0))
    img.save(str(out) + "_points.png")
    print(f"wrote {out}.png ({w} x {h} px, {w * res:.2f} x {h * res:.2f} m, {free.sum() * res * res:.1f} m^2 free), "
          f"{out}.yaml, {out}_points.png, {out}_path.csv")


if __name__ == "__main__":
    main()
