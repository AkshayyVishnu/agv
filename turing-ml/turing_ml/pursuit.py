"""Pure pursuit on a racing line (map-based driver).

    drv = PurePursuit(load_raceline(".../TuringTrack_raceline.csv"), cfg["pursuit"], vehicle_params, seed=0)
    steer, speed = drv(obs["pose"], obs["speed"], obs["t"], obs["yaw_rate"])

Needs the car's POSE on the map. In sim that is the ground truth `obs["pose"]` (privileged; the
reactive driver never sees it). On the car it has to come from localization (a particle filter on
the LiDAR), which is late and noisy: pose_delay / pose_noise_* in configs/raceline.yaml imitate it.
"""
import collections
import math
from typing import Dict, Tuple

import numpy as np


class PurePursuit:
    def __init__(self, line: Dict[str, np.ndarray], cfg: Dict, vehicle: Dict[str, float], seed: int = 0):
        self.xy = np.stack([line["x"], line["y"]], 1)
        self.v = np.asarray(line["v"], float) * float(cfg.get("speed_scale", 1.0))
        seg = np.hypot(*(np.roll(self.xy, -1, 0) - self.xy).T)
        self.ds = float(np.mean(seg))
        self.cfg = cfg
        self.wb = vehicle["lf"] + vehicle["lr"]
        self.lr = vehicle["lr"]
        self.s_max = vehicle["s_max"]
        self.rng = np.random.default_rng(seed)
        self.idx = None
        self.history = collections.deque()

    def _observed_pose(self, pose, t):
        """Ground-truth pose made late and noisy, like a particle filter's output.
        Returns (x, y, yaw, age_s)."""
        delay = float(self.cfg.get("pose_delay", 0.0))
        self.history.append((t, tuple(pose)))
        while len(self.history) > 1 and self.history[1][0] <= t - delay + 1e-9:
            self.history.popleft()
        t_pose, (x, y, yaw) = self.history[0]
        sxy, syaw = float(self.cfg.get("pose_noise_xy", 0.0)), float(self.cfg.get("pose_noise_yaw", 0.0))
        if sxy > 0:
            x, y = x + self.rng.normal(0, sxy), y + self.rng.normal(0, sxy)
        if syaw > 0:
            yaw = yaw + self.rng.normal(0, syaw)
        return x, y, yaw, t - t_pose

    def _nearest(self, p) -> int:
        n = len(self.xy)
        if self.idx is None:                                   # first call: search everything
            return int(np.argmin(np.hypot(*(self.xy - p).T)))
        win = (self.idx + np.arange(-10, 40)) % n               # then only near the last match
        return int(win[np.argmin(np.hypot(*(self.xy[win] - p).T))])

    def __call__(self, pose, speed: float, t: float = 0.0, yaw_rate: float = 0.0) -> Tuple[float, float]:
        x, y, yaw, age = self._observed_pose(pose, t)
        if self.cfg.get("predict", True) and age > 0:
            # roll the late pose forward by its age with the CURRENT speed and yaw rate (VESC
            # odometry and IMU are fresh on the car; the localization timestamp gives the age)
            mid = yaw + 0.5 * yaw_rate * age
            x, y, yaw = x + speed * age * math.cos(mid), y + speed * age * math.sin(mid), yaw + yaw_rate * age
        c, s = math.cos(yaw), math.sin(yaw)
        rear = np.array([x - self.lr * c, y - self.lr * s])     # gym pose is the CoG; pursuit uses the rear axle
        self.idx = self._nearest(rear)
        n = len(self.xy)
        look = float(np.clip(abs(speed) * self.cfg["lookahead_time"], self.cfg["lookahead_min"],
                             self.cfg["lookahead_max"]))
        j = self.idx
        for _ in range(n):                                      # first line point >= look away
            j = (j + 1) % n
            if np.hypot(*(self.xy[j] - rear)) >= look:
                break
        dx, dy = self.xy[j] - rear
        lx, ly = c * dx + s * dy, -s * dx + c * dy              # target in the car frame
        dist2 = max(lx * lx + ly * ly, 1e-6)
        steer = math.atan(2.0 * self.wb * ly / dist2)           # pure pursuit: curvature 2y / L^2
        steer = float(np.clip(steer, -self.s_max, self.s_max))
        # slowest planned speed between here and speed_preview ahead: brakes early for a corner
        # (the gym's speed control lags) without speeding up before the car is out of one
        ahead = int(round(abs(speed) * self.cfg["speed_preview"] / self.ds))
        v = float(self.v[(self.idx + np.arange(ahead + 1)) % n].min())
        return steer, v
