"""Simulation model of the LDRobot D500 (STL-19P) LiDAR as seen through ldlidar_stl_ros2.

Feed it the simulator's perfect scan at every physics step (push), and it hands back
LaserScan-shaped scans (read / poll) that look like what the real driver publishes on
the car: 360 deg, ~500 points per revolution, NaN for "no return", angles 0..2*pi,
a variable point count, motion skew within each revolution, latency, range-dependent
noise and the failure modes of a cheap sensor next to 20 cm walls.

Pure NumPy (>= 1.18), no ROS imports, so the gym harness and the ROS relay can share it.
Every choice below is explained in docs/DECISIONS.md (D-numbers in comments).
"""
from __future__ import annotations

import collections
import math
from dataclasses import dataclass, fields, replace
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

TWO_PI = 2.0 * math.pi

# Message metadata the real driver hard-codes (ldlidar_stl_ros2 src/demo.cpp), D-03.
DRIVER_RANGE_MIN = 0.02
DRIVER_RANGE_MAX = 25.0


@dataclass(frozen=True)
class D500Config:
    """All sensor parameters. Defaults are the 'nominal' preset in configs/sensor.yaml."""

    # timing (D-05, D-06)
    rate_hz: float = 10.0                 # revolutions per second (spec 6-13, typ 10)
    rate_jitter: float = 0.01             # relative std of each revolution's period
    sample_rate_hz: float = 5000.0        # measurements per second (spec)
    latency_s: float = 0.05               # end of revolution -> scan usable by our code
    latency_jitter_s: float = 0.01
    spin_direction: str = "cw"            # physical rotation seen from above (D-05)

    # geometry (D-04, D-12)
    mount_yaw: float = 0.0                # rad, sensor zero relative to car forward, CCW +
    angle_jitter: float = 0.0035          # rad std per point (~0.2 deg)
    flip: bool = False                    # True = driver run with laser_scan_dir False

    # ranging (D-07)
    range_min: float = 0.03
    range_max: float = 12.0
    noise_bands: Tuple[Tuple[float, float], ...] = (
        (0.5, 0.010), (2.0, 0.020), (math.inf, 0.030))  # (upper bound m, std m)
    noise_scale: float = 1.0
    range_bias: float = 0.0
    base_noise_std: float = 0.01          # noise already present in the input scan

    # 20 cm walls (D-08)
    wall_visible_range: float = 4.5       # beyond this the beam passes over the wall
    overshoot_nan_prob: float = 0.5       # overshooting beam: NaN, else a far return
    pitch_up_prob: float = 0.0            # per scan: nose-up sector overshoots close walls
    pitch_up_visible_range: Tuple[float, float] = (2.0, 3.5)
    pitch_down_prob: float = 0.0          # per scan: nose-down sector hits the floor
    pitch_down_floor_range: Tuple[float, float] = (2.0, 3.5)
    pitch_sector_deg: Tuple[float, float] = (30.0, 90.0)

    # dropouts and artefacts (D-09, D-10, D-11)
    dropout_prob: float = 0.0             # independent per point
    burst_rate: float = 0.0               # expected dark/shiny sectors per scan (Poisson)
    burst_width_deg: Tuple[float, float] = (3.0, 15.0)
    mixed_pixel_prob: float = 0.0
    mixed_pixel_jump: float = 0.3
    self_hit_sectors: Tuple[Tuple[float, float, float], ...] = ()  # (deg0, deg1, range m)

    @property
    def period(self) -> float:
        return 1.0 / self.rate_hz


@dataclass
class Scan:
    """Mirror of sensor_msgs/LaserScan as the real driver fills it (D-03)."""

    seq: int
    stamp: float            # header.stamp: publish time = END of the revolution
    available_at: float     # when our code can use it (stamp + latency)
    angle_min: float
    angle_max: float
    angle_increment: float
    time_increment: float
    scan_time: float
    range_min: float
    range_max: float
    ranges: np.ndarray      # float32, NaN = no return / empty bin


def _wrap_2pi(a: np.ndarray) -> np.ndarray:
    return np.mod(a, TWO_PI)


def _in_sector(theta: np.ndarray, start: float, width: float) -> np.ndarray:
    """theta and start in rad on [0, 2pi); sector runs CCW from start for width."""
    return _wrap_2pi(theta - start) < width


class D500Model:
    """Usage per episode: reset(t0); every physics step push(t, ranges, angle_min, inc);
    every control step read(t) (latest usable scan) or poll(t) (all newly usable scans)."""

    def __init__(self, cfg: Optional[D500Config] = None, seed: int = 0):
        self.cfg = cfg or D500Config()
        self.reset(0.0, seed)

    # ------------------------------------------------------------------ episode API
    def reset(self, t0: float = 0.0, seed: Optional[int] = None, cfg: Optional[D500Config] = None):
        if cfg is not None:
            self.cfg = cfg
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._times: List[float] = []
        self._scans: List[Tuple[np.ndarray, float, float]] = []   # ranges, angle_min, inc
        self._rev_start = float(t0)
        self._period = self._draw_period()
        self._seq = 0
        self._pending: Deque[Scan] = collections.deque()
        self._latest: Optional[Scan] = None
        self._last_avail = -math.inf

    def push(self, t: float, ranges, angle_min: float, angle_increment: float) -> None:
        """Perfect simulated scan taken at time t, angles in the CAR frame (CCW +)."""
        if self._times and t < self._times[-1]:
            raise ValueError("push() times must be non-decreasing")
        self._times.append(float(t))
        self._scans.append((np.asarray(ranges, dtype=np.float64), float(angle_min),
                            float(angle_increment)))
        while self._rev_start + self._period <= t:
            self._pending.append(self._build_revolution())
            self._rev_start += self._period
            self._period = self._draw_period()
        self._prune()

    def poll(self, t: float) -> List[Scan]:
        """All scans that became usable at or before t, oldest first (for the ROS relay)."""
        out = []
        while self._pending and self._pending[0].available_at <= t:
            self._latest = self._pending.popleft()
            out.append(self._latest)
        return out

    def read(self, t: float) -> Optional[Scan]:
        """Newest scan usable at time t (may be the same object as last call), or None."""
        self.poll(t)
        return self._latest

    # ------------------------------------------------------------------ internals
    def _draw_period(self) -> float:
        c = self.cfg
        p = c.period * (1.0 + c.rate_jitter * self.rng.standard_normal())
        return float(np.clip(p, 1.0 / 13.0, 1.0 / 6.0))  # spec rate range 6-13 Hz

    def _prune(self) -> None:
        # keep the newest push taken at or before the current revolution start
        k = int(np.searchsorted(self._times, self._rev_start, side="right")) - 1
        if k > 0:
            del self._times[:k]
            del self._scans[:k]

    def _sample_truth(self, tau: np.ndarray, theta_car: np.ndarray) -> np.ndarray:
        """Range each point would have hit: input scan at the point's time, nearest beam
        at its angle (D-06, D-04). NaN where the input scan has no beam (outside its FOV)."""
        src = np.searchsorted(self._times, tau, side="right") - 1
        src = np.clip(src, 0, len(self._times) - 1)
        out = np.full(tau.shape, np.nan)
        for k in np.unique(src):
            sel = src == k
            r, a0, inc = self._scans[k]
            pos = _wrap_2pi(theta_car[sel] - a0) / inc
            idx = np.rint(pos).astype(np.int64)
            ok = idx <= len(r) - 1
            vals = np.full(idx.shape, np.nan)
            vals[ok] = r[idx[ok]]
            out[sel] = vals
        return out

    def _noise_std(self, r: np.ndarray) -> np.ndarray:
        c = self.cfg
        std = np.full(r.shape, c.noise_bands[-1][1])
        for upper, s in reversed(c.noise_bands):
            std = np.where(r < upper, s, std)
        std = std * c.noise_scale
        # the input already carries base_noise_std; add only the remainder (D-07)
        return np.sqrt(np.maximum(std ** 2 - c.base_noise_std ** 2, 0.0))

    def _random_sector(self, width_deg: Tuple[float, float]) -> Tuple[float, float]:
        return (self.rng.uniform(0.0, TWO_PI),
                math.radians(self.rng.uniform(width_deg[0], width_deg[1])))

    def _build_revolution(self) -> Scan:
        c, rng = self.cfg, self.rng
        period, t0 = self._period, self._rev_start
        n = max(8, int(round(c.sample_rate_hz * period)))
        frac = np.arange(n) / n
        tau = t0 + frac * period                                   # measurement times
        spin = -1.0 if c.spin_direction == "cw" else 1.0           # D-05
        theta = _wrap_2pi(spin * TWO_PI * frac + c.angle_jitter * rng.standard_normal(n))
        r = self._sample_truth(tau, theta + c.mount_yaw)           # sensor -> car frame

        # 20 cm walls: beyond the visible range the beam goes over the wall (D-08)
        visible = np.full(n, c.wall_visible_range)
        if rng.random() < c.pitch_up_prob:
            s0, w = self._random_sector(c.pitch_sector_deg)
            visible[_in_sector(theta, s0, w)] = rng.uniform(*c.pitch_up_visible_range)
        over = np.isfinite(r) & (r > visible)
        # numpy <= 1.22 raises on NaN bounds, so only draw for overshooting beams
        far = np.where(over, np.minimum(r, c.range_max), c.range_max)
        far = rng.uniform(far, c.range_max)
        r = np.where(over, np.where(rng.random(n) < c.overshoot_nan_prob, np.nan, far), r)
        if rng.random() < c.pitch_down_prob:
            s0, w = self._random_sector(c.pitch_sector_deg)
            floor = rng.uniform(*c.pitch_down_floor_range)
            hit = _in_sector(theta, s0, w) & ~(r < floor)          # NaN/far rays hit floor too
            r = np.where(hit, floor, r)

        # ranging noise (D-07); NaN stays NaN
        r = r + c.range_bias + self._noise_std(np.nan_to_num(r, nan=0.0)) * rng.standard_normal(n)
        r[(r < c.range_min) | (r > c.range_max)] = np.nan

        # dropouts (D-09)
        r[rng.random(n) < c.dropout_prob] = np.nan
        for _ in range(rng.poisson(c.burst_rate)):
            s0, w = self._random_sector(c.burst_width_deg)
            r[_in_sector(theta, s0, w)] = np.nan

        # mixed pixels at depth edges, consecutive points in spin order (D-10)
        if c.mixed_pixel_prob > 0.0:
            a, b = r[:-1], r[1:]
            jump = np.isfinite(a) & np.isfinite(b) & (np.abs(a - b) > c.mixed_pixel_jump)
            pick = np.flatnonzero(jump & (rng.random(n - 1) < c.mixed_pixel_prob))
            r[pick] = a[pick] + rng.random(pick.size) * (b[pick] - a[pick])

        # the car's own body, fixed in the sensor frame (D-11)
        for d0, d1, rng_m in c.self_hit_sectors:
            sel = _in_sector(theta, math.radians(d0 % 360.0), math.radians((d1 - d0) % 360.0))
            r[sel] = rng_m + 0.01 * rng.standard_normal(int(sel.sum()))

        ranges = self._bin_like_driver(theta, r, n)
        latency = max(0.0, c.latency_s + c.latency_jitter_s * rng.standard_normal())
        stamp = t0 + period
        avail = max(stamp + latency, self._last_avail)              # keep delivery ordered
        self._last_avail = avail
        self._seq += 1
        inc = TWO_PI / (n - 1)
        return Scan(seq=self._seq, stamp=stamp, available_at=avail, angle_min=0.0,
                    angle_max=TWO_PI, angle_increment=inc, time_increment=period / (n - 1),
                    scan_time=period, range_min=DRIVER_RANGE_MIN, range_max=DRIVER_RANGE_MAX,
                    ranges=ranges)

    def _bin_like_driver(self, theta: np.ndarray, r: np.ndarray, n: int) -> np.ndarray:
        """Replicates ToLaserscanMessagePublish: n bins over [0, 2pi] with
        inc = 2pi/(n-1), min per bin, empty bins NaN (D-03).
        laser_scan_dir=True (CCW, the shipped default) reduces to floor(theta/inc);
        laser_scan_dir=False (flip) is ceil(device_angle/inc) with device angle = -theta."""
        inc = TWO_PI / (n - 1)
        if self.cfg.flip:
            idx = np.ceil(_wrap_2pi(-theta) / inc).astype(np.int64)
        else:
            idx = np.floor(theta / inc).astype(np.int64)
        keep = (idx >= 0) & (idx < n)
        out = np.full(n, np.nan)
        np.fmin.at(out, idx[keep], r[keep])
        return out.astype(np.float32)


# ---------------------------------------------------------------------- config I/O
def _coerce(name: str, value):
    """YAML lists -> the tuple shapes D500Config expects."""
    if name in ("noise_bands", "self_hit_sectors"):
        return tuple(tuple(float(x) for x in row) for row in value)
    if isinstance(value, list):
        return tuple(float(x) for x in value)
    return value


def config_from_dict(d: Dict, base: Optional[D500Config] = None) -> D500Config:
    known = {f.name for f in fields(D500Config)}
    unknown = set(d) - known
    if unknown:
        raise KeyError(f"unknown D500 parameters: {sorted(unknown)}")
    return replace(base or D500Config(), **{k: _coerce(k, v) for k, v in d.items()})


def load_sensor_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)["d500"]


def load_preset(path: str, preset: str = "nominal") -> D500Config:
    return config_from_dict(load_sensor_yaml(path)["presets"][preset])


def sample_config(spec: Dict, rng: np.random.Generator,
                  base: Optional[D500Config] = None) -> D500Config:
    """Per-episode domain randomization (D-13). spec values: [lo, hi] -> uniform,
    {choice: [...]} -> pick one."""
    draws = {}
    for k, v in spec.items():
        if isinstance(v, dict):
            opts = v["choice"]
            draws[k] = opts[int(rng.integers(len(opts)))]
        else:
            draws[k] = float(rng.uniform(v[0], v[1]))
    return config_from_dict(draws, base)
