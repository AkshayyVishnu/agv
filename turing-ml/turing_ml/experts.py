"""Hand-written drivers (experts). The disparity extender is PERSON A's deliverable.

Written by B on 2026-10-11 at the user's request so the harness and the live viewer
have a real driver. It starts from the Execution Manual's version and adds four
changes, each explained in docs/DECISIONS.md:
  D-31 parameters from configs/expert.yaml, no hard-coded 1.5 m/s
  D-32 the D500 seam straight ahead is not treated as a wall edge
  D-33 side-clearance check (don't steer into a wall that's beside the car)
  D-34 speed limited by stopping distance inside the 4 m the sensor can see
  D-36 safety bubble around every point, not only disparities
  D-40 scan-age compensation: rotate the grid by yaw_rate * yaw_comp_dt
  D-41 headings considered within +-120 deg (hairpins)
Pure functions of (grid, speed, yaw_rate), all available on the car, so DAgger can query them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Dict, Optional, Tuple

import numpy as np

from .sensor import GRID, RMAX

INC = float(GRID[1] - GRID[0])                       # 1 degree in rad
ZERO = int(np.argmin(np.abs(GRID)))                  # index of the 0 deg beam


@dataclass(frozen=True)
class DisparityConfig:
    half_width: float = 0.22          # half car width + margin (gym car is 0.31 m wide)
    bubble_width: float = 0.25        # D-36: clearance kept from every point; 0 = off
    disp_thresh: float = 0.35         # range jump that counts as a wall edge (>= 0.3, noise)
    fov_deg: float = 120.0            # D-41: only steer toward points within +-fov
    steer_max: float = 0.4
    v_min: float = 0.6
    v_max: float = 3.0                # NOT capped at 1.5 (user, 2026-10-11)
    stop_dist: float = 0.5            # front closer than this -> speed 0
    turn_slowdown: float = 0.5        # speed *= 1 - turn_slowdown * |steer| / steer_max
    # D-32 seam
    seam_period: float = 0.1          # D500 revolution time
    seam_margin: float = 0.15
    # D-33 side clearance
    side_deg: Tuple[float, float] = (60.0, 100.0)
    side_clear: float = 0.45          # don't steer toward a side wall closer than this
    # D-34 stopping distance
    brake_decel: float = 4.0          # m/s^2 we trust the car to brake at
    latency: float = 0.15             # scan age + control period (D-05 table)
    # D-40 scan age
    yaw_comp_dt: float = 0.15         # s; rotate the grid by yaw_rate * this before deciding; 0 = off


def disparity_extender(grid: np.ndarray, speed_now: float = 0.0,
                       cfg: Optional[DisparityConfig] = None,
                       yaw_rate: float = 0.0) -> Tuple[float, float]:
    """grid: 271 ranges (m) from to_grid, index 0 = -135 deg (right), 135 = straight ahead,
    270 = +135 deg (left). yaw_rate: rad/s, positive = turning left (odometry/IMU).
    Returns (steer rad, speed m/s); positive steer = left."""
    c = cfg or DisparityConfig()
    g = np.asarray(grid, dtype=np.float32)
    if c.yaw_comp_dt > 0.0 and yaw_rate != 0.0:
        # D-40: the scan is ~0.1-0.2 s old; in a corner the car has turned since. A wall seen at
        # angle a is now at a - yaw_rate * dt, so new(phi) = old(phi + yaw_rate * dt).
        g = np.interp(GRID + yaw_rate * c.yaw_comp_dt, GRID, g).astype(np.float32)
    ext = g.copy()
    seam_tol = abs(speed_now) * c.seam_period + c.seam_margin
    for i in range(len(g) - 1):
        a, b = float(g[i]), float(g[i + 1])
        jump = abs(a - b)
        if jump < c.disp_thresh:
            continue
        if abs(i - ZERO) <= 1 and jump < seam_tol:
            continue                                   # D-32: rotating-sensor seam, not an edge
        near = min(a, b)
        n = int(math.ceil(math.atan2(c.half_width, max(near, 1e-3)) / INC))
        if a < b:                                      # wall edge at i: pad toward larger angles
            ext[i + 1:i + 1 + n] = np.minimum(ext[i + 1:i + 1 + n], near)
        else:                                          # wall edge at i+1: pad toward smaller angles
            lo = max(0, i - n + 1)
            ext[lo:i + 1] = np.minimum(ext[lo:i + 1], near)

    # D-36: safety bubble. Pad EVERY point, not just disparities, so the chosen heading
    # keeps bubble_width from smooth walls too (the inside of a bend has no disparity).
    if c.bubble_width > 0.0:
        nb = np.ceil(np.arctan2(c.bubble_width, np.maximum(g, 1e-3)) / INC).astype(int)
        for i in range(len(g)):
            lo, hi = max(0, i - nb[i]), min(len(g), i + nb[i] + 1)
            ext[lo:hi] = np.minimum(ext[lo:hi], g[i])

    cand =np.where(np.abs(GRID) <= math.radians(c.fov_deg), ext, -1.0)
    ok = cand >= cand.max() - 0.1                      # many beams tie at the 4 m cap
    best, start = (ZERO, ZERO), None
    for i, v in enumerate(np.append(ok, False)):       # centre of the widest near-max run
        if v and start is None:
            start = i
        if not v and start is not None:
            if i - 1 - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    steer = float(np.clip(GRID[(best[0] + best[1]) // 2], -c.steer_max, c.steer_max))

    # D-33: never turn toward a wall that is already beside the car
    lo, hi = math.radians(c.side_deg[0]), math.radians(c.side_deg[1])
    lsel, rsel = (GRID >= lo) & (GRID <= hi), (GRID <= -lo) & (GRID >= -hi)
    left = float(g[lsel].min()) if lsel.any() else RMAX
    right = float(g[rsel].min()) if rsel.any() else RMAX
    if steer > 0 and left < c.side_clear:
        steer = 0.0
    elif steer < 0 and right < c.side_clear:
        steer = 0.0

    front = float(g[np.abs(GRID) < math.radians(10)].min())
    if front < c.stop_dist:
        return steer, 0.0
    v = c.v_min + (c.v_max - c.v_min) * min(front / RMAX, 1.0)
    v *= 1.0 - c.turn_slowdown * abs(steer) / c.steer_max
    # D-34: stop within what we can see, after the latency
    usable = max(front - c.stop_dist, 0.0)
    a, t = c.brake_decel, c.latency
    v_stop = -a * t + math.sqrt((a * t) ** 2 + 2 * a * usable)
    return steer, float(max(min(v, v_stop, c.v_max), 0.0))


def load_expert_config(path: str, name: str = "disparity") -> DisparityConfig:
    import yaml
    with open(path, "r") as f:
        d: Dict = yaml.safe_load(f)[name]
    known = {f.name for f in fields(DisparityConfig)}
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"unknown disparity parameters: {sorted(unknown)}")
    d = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
    return replace(DisparityConfig(), **d)
