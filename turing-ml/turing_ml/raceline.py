"""Racing line for a known map: centre line -> minimum-curvature line -> speed profile.

    track = load_map("maps/TuringTrack/TuringTrack_map.yaml")
    centre = trace_centreline(track, start_pose)                    # (N, 2) closed loop, even spacing
    line = min_curvature(track, centre, half_car=0.125, margin=0.08)
    prof = speed_profile(line, a_lat=7.5, a_accel=5.0, a_brake=5.0, v_max=5.0)

Minimum curvature as bounded linear least squares: points p_i = c_i + a_i * n_i (c centre line,
n left normal, a lateral offset), minimise sum |p_(i-1) - 2 p_i + p_(i+1)|^2 subject to the track
bounds, then resample and repeat. Same idea as TUM's global_racetrajectory_optimization (which
is tuned for full-size cars: 1-3 m steps on a ~50 m track won't work). Settings: configs/raceline.yaml.
"""
import dataclasses
import math
import pathlib
from typing import Optional, Tuple

import numpy as np
import yaml
from scipy import ndimage
from scipy.optimize import lsq_linear


@dataclasses.dataclass
class TrackMap:
    free: np.ndarray        # bool, image rows top to bottom
    res: float
    origin: Tuple[float, float]
    dist: np.ndarray        # metres from each free cell to the nearest wall surface

    def to_px(self, xy: np.ndarray) -> np.ndarray:
        """world (x, y) -> fractional (row, col)"""
        xy = np.atleast_2d(xy)
        col = (xy[:, 0] - self.origin[0]) / self.res - 0.5
        row = self.free.shape[0] - (xy[:, 1] - self.origin[1]) / self.res - 0.5
        return np.stack([row, col], 1)

    def clearance(self, xy: np.ndarray) -> np.ndarray:
        """Distance to the nearest wall (m), bilinear; <= 0 means inside a wall."""
        rc = self.to_px(xy).T
        return ndimage.map_coordinates(self.dist, rc, order=1, mode="constant", cval=-1.0)


def load_map(yaml_path: str) -> TrackMap:
    from PIL import Image
    p = pathlib.Path(yaml_path)
    meta = yaml.safe_load(p.read_text())
    img = np.asarray(Image.open(p.parent / meta["image"]).convert("L"), dtype=float) / 255.0
    occ = (1.0 - img) if not meta.get("negate", 0) else img
    free = occ < float(meta.get("free_thresh", 0.196))
    res = float(meta["resolution"])
    # cell centre to nearest wall cell centre, minus half a cell = distance to the wall surface
    dist = ndimage.distance_transform_edt(free) * res - res / 2
    dist[~free] = -res / 2
    return TrackMap(free, res, (float(meta["origin"][0]), float(meta["origin"][1])), dist)


def trace_centreline(track: TrackMap, start: Tuple[float, float, float], step: float = 0.05,
                     max_turn_deg: float = 45.0, max_len: float = 2000.0) -> np.ndarray:
    """Follow the distance ridge (the line farthest from both walls) from `start` (x, y, yaw).
    Each step tries headings within +-max_turn_deg of the current one and moves to the point with
    the most clearance, so it gets round U-turns but can never reverse. A step is short (5 cm)
    compared with the thinnest wall, so it can't hop through one. Returns an evenly spaced
    closed loop (N, 2) starting near the start pose."""
    p = np.array(start[:2], float)
    h0 = h = float(start[2])
    if track.clearance(p[None, :])[0] <= 0:
        raise RuntimeError(f"start pose {start} is not on the track")
    pts = [p.copy()]
    turns = np.radians(np.linspace(-max_turn_deg, max_turn_deg, 61))
    travelled = 0.0
    while travelled < max_len:
        cand = p + step * np.stack([np.cos(h + turns), np.sin(h + turns)], 1)
        c = track.clearance(cand) - 0.01 * np.abs(turns)       # tie-break: prefer going straight
        k = int(np.argmax(c))
        if c[k] <= 0:
            raise RuntimeError(f"centre-line trace left the track near {p}")
        h = h + turns[k]
        p = cand[k]
        travelled += step
        pts.append(p.copy())
        heading_ok = math.cos(h - h0) > 0
        if travelled > 5.0 and heading_ok and np.hypot(*(p - pts[0])) < 0.6 \
                and np.dot(p - pts[0], [math.cos(h0), math.sin(h0)]) >= 0:
            break                                                # back across the start line
    else:
        raise RuntimeError("centre-line trace did not close")
    loop = np.array(pts[:-1])
    loop = _smooth_closed(loop, 9)
    return resample_closed(loop, 0.1)


def _smooth_closed(pts: np.ndarray, k: int) -> np.ndarray:
    w = np.ones(k) / k
    return np.stack([ndimage.convolve1d(pts[:, d], w, mode="wrap") for d in range(2)], 1)


def resample_closed(pts: np.ndarray, ds: float) -> np.ndarray:
    """Evenly spaced points (spacing ~ds) along a closed polyline, starting at pts[0]."""
    seg = np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / ds)), 8)
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    return np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], 1)


def normals(pts: np.ndarray) -> np.ndarray:
    """Unit left normals of a closed loop (central differences)."""
    t = np.roll(pts, -1, 0) - np.roll(pts, 1, 0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return np.stack([-t[:, 1], t[:, 0]], 1)


def curvature(pts: np.ndarray) -> np.ndarray:
    """Signed curvature (1/m, + = turning left) of a closed loop from three-point circles."""
    a, b, c = np.roll(pts, 1, 0), pts, np.roll(pts, -1, 0)
    ab, bc, ca = b - a, c - b, a - c
    cross = ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]
    den = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1) * np.linalg.norm(ca, axis=1)
    return 2.0 * cross / np.maximum(den, 1e-12)


def min_curvature(track: TrackMap, centre: np.ndarray, half_car: float, margin: float,
                  iterations: int = 3, ds: float = 0.1, r_min: float = 0.0) -> np.ndarray:
    """Minimum-curvature line inside the track, keeping half_car + margin from the walls.
    Minimising total squared curvature can make one hairpin tighter to straighten the rest;
    with r_min > 0, points tighter than r_min get a growing weight on later passes, so the
    line opens the hairpins up to the car's turning circle where the track allows it."""
    keep = half_car + margin
    line = centre.copy()
    weight = None
    for it in range(iterations + (6 if r_min > 0 else 0)):
        line = resample_closed(line, ds)
        n = len(line)
        nrm = normals(line)
        # free distance to the left and right of each point, along its normal
        left, right = _free_extent(track, line, nrm)
        hi = np.maximum(left - keep, 0.0)
        lo = -np.maximum(right - keep, 0.0)
        eye = np.eye(n)
        D = np.roll(eye, -1, 1) - 2 * eye + np.roll(eye, 1, 1)
        if r_min > 0 and it >= iterations:
            k = ndimage.convolve1d(np.abs(curvature(line)), np.ones(5) / 5, mode="wrap")
            over = np.maximum(k * r_min * 1.1 - 1.0, 0.0)          # 10 % safety on the limit
            if weight is not None and len(weight) != n:              # resampling changed n: map by
                weight = np.interp(np.linspace(0, 1, n, endpoint=False),       # position on the lap
                                   np.linspace(0, 1, len(weight), endpoint=False), weight, period=1.0)
            weight = (1.0 if weight is None else weight) * (1.0 + 20.0 * over)
            weight = ndimage.maximum_filter1d(weight, 15, mode="wrap")
            D = D * weight[:, None]
        A = np.vstack([D * nrm[:, 0], D * nrm[:, 1]])
        b = -np.concatenate([D @ line[:, 0], D @ line[:, 1]])
        res = lsq_linear(A, b, bounds=(lo, hi + 1e-9), lsmr_tol="auto", max_iter=5000)
        line = line + res.x[:, None] * nrm
    return resample_closed(line, ds)


def _free_extent(track: TrackMap, pts: np.ndarray, nrm: np.ndarray, reach: float = 1.5):
    """Distance from each point to the wall surface along +normal (left) and -normal (right),
    following only connected free space."""
    offs = np.arange(0.0, reach + 1e-9, track.res / 4)
    out = []
    for sign in (1.0, -1.0):
        q = pts[:, None, :] + sign * offs[None, :, None] * nrm[:, None, :]
        c = track.clearance(q.reshape(-1, 2)).reshape(len(pts), len(offs))
        blocked = c <= 0
        first = np.where(blocked.any(1), blocked.argmax(1), len(offs) - 1)
        out.append(offs[np.maximum(first - 1, 0)] + track.res / 8)
    return out[0], out[1]


def speed_profile(line: np.ndarray, a_lat: float, a_accel: float, a_brake: float, v_max: float,
                  curv_smooth: int = 5) -> dict:
    """Fastest speed at each point: v <= sqrt(a_lat / |kappa|), then acceleration and braking
    limits, wrapped around the loop."""
    kappa = curvature(line)
    if curv_smooth > 1:
        kappa = ndimage.convolve1d(kappa, np.ones(curv_smooth) / curv_smooth, mode="wrap")
    ds = np.hypot(*(np.roll(line, -1, 0) - line).T)
    v = np.minimum(np.sqrt(a_lat / np.maximum(np.abs(kappa), 1e-6)), v_max)
    n = len(v)
    for _ in range(2):                                         # twice: the loop wraps
        for i in range(n):                                     # forward: acceleration
            j = (i - 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * a_accel * ds[j]))
        for i in range(n - 1, -1, -1):                         # backward: braking
            j = (i + 1) % n
            v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * a_brake * ds[i]))
    s = np.concatenate([[0.0], np.cumsum(ds[:-1])])
    lap = float(np.sum(ds / np.maximum(0.5 * (v + np.roll(v, -1)), 1e-3)))
    return dict(x=line[:, 0], y=line[:, 1], s=s, kappa=kappa, v=v, length=float(ds.sum()), lap_time=lap)


def save_raceline(path: str, prof: dict):
    arr = np.stack([prof["x"], prof["y"], prof["s"], prof["kappa"], prof["v"]], 1)
    np.savetxt(path, arr, delimiter=", ", fmt="%.4f", header="x_m, y_m, s_m, kappa_radpm, v_mps")


def load_raceline(path: str) -> dict:
    a = np.loadtxt(path, delimiter=",", comments="#")
    return dict(x=a[:, 0], y=a[:, 1], s=a[:, 2], kappa=a[:, 3], v=a[:, 4])


def line_start_pose(prof: dict, lr: float) -> Tuple[float, float, float]:
    """Car pose (gym pose = centre of gravity) with the rear axle on the line's first point,
    facing along the line. The line starts at the start line (make_raceline.py)."""
    hd = math.atan2(prof["y"][1] - prof["y"][0], prof["x"][1] - prof["x"][0])
    return (float(prof["x"][0] + lr * math.cos(hd)), float(prof["y"][0] + lr * math.sin(hd)), hd)


def start_index(prof: dict, pose) -> int:
    return int(np.argmin(np.hypot(prof["x"] - pose[0], prof["y"] - pose[1])))


def load_raceline_config(path: str, section: str = "raceline") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)[section]


def find_raceline(map_path: str) -> Optional[str]:
    """<dir>/<Name>_raceline.csv next to <dir>/<Name>_map(.yaml)."""
    p = pathlib.Path(map_path[:-5] if map_path.endswith(".yaml") else map_path)
    name = p.name[:-4] if p.name.endswith("_map") else p.name
    cand = p.parent / f"{name}_raceline.csv"
    return str(cand) if cand.exists() else None
