"""TuringEnv: f1tenth_gym (main branch) made to behave like our car's sense-act loop.

- control at 20 Hz: each action is held for 5 physics steps of 10 ms          (D-19)
- sensing through the D500 model: every physics step's perfect scan is pushed,
  the policy only ever sees the newest scan the car could have by now         (D-20)
- observations in one dict, true pose kept separate from what a policy may use (D-22)
- the ONLY file that calls the gym, so a gym version change touches one file  (D-18)

    env = TuringEnv("maps/my_track", sensor_cfg=load_preset("configs/sensor.yaml", "nominal"))
    obs = env.reset((x, y, yaw))
    while not (obs["done"] or obs["truncated"]):
        obs = env.step(policy(obs["grid"], obs["speed"]))

Design reasons: docs/DECISIONS.md, D-18 to D-29.
"""
from __future__ import annotations

import os
import weakref
from typing import Dict, Optional, Sequence

import numpy as np

from .d500 import D500Config, D500Model, sample_config
from .sensor import GRID, RMAX, to_grid

SIM_DT = 0.01                    # gym physics step (s)
GYM_FOV, GYM_BEAMS = 4.7, 1080   # gym main's fixed LiDAR (RaceCar defaults)
GYM_ANGLE_MIN = -GYM_FOV / 2
GYM_ANGLE_INC = GYM_FOV / (GYM_BEAMS - 1)
LIDAR_DIST = 0.275               # f1tenth_gym_ros scan_distance_to_base_link (D-23)
BUNDLED_MAPS = {"berlin", "skirk", "levine", "vegas"}


class StaleEnvError(RuntimeError):
    pass


class TuringEnv:
    _active: Optional["weakref.ref"] = None   # gym main keeps the map at class level (D-27)

    def __init__(self, map_path: str, map_ext: Optional[str] = None,
                 sensor_cfg: Optional[D500Config] = None,
                 sensor_randomization: Optional[Dict] = None,
                 control_hz: float = 20.0, max_laps: int = 2, max_time: float = 60.0,
                 lidar_dist: float = LIDAR_DIST, params: Optional[Dict] = None,
                 seed: int = 0, render: bool = False):
        import gym  # imported here so the rest of turing_ml works without the gym installed

        map_path, map_ext = _resolve_map(map_path, map_ext)
        self.k_ctrl = int(round(1.0 / (control_hz * SIM_DT)))
        if abs(self.k_ctrl * SIM_DT * control_hz - 1.0) > 1e-6:
            raise ValueError(f"control_hz must divide {1 / SIM_DT:.0f} Hz physics, got {control_hz}")
        self.control_dt = self.k_ctrl * SIM_DT
        self.max_laps, self.max_time, self.render_on = max_laps, max_time, render
        self.base_sensor_cfg = sensor_cfg or D500Config()
        self.sensor_randomization = sensor_randomization
        self._rng = np.random.default_rng(seed)

        prev = TuringEnv._active() if TuringEnv._active else None
        if prev is not None:
            prev._stale = True                     # its map is about to be overwritten
        kwargs = dict(map=map_path, map_ext=map_ext, num_agents=1, timestep=SIM_DT,
                      lidar_dist=lidar_dist, seed=seed)
        if params is not None:
            kwargs["params"] = params
        self.env = gym.make("f110_gym:f110-v0", **kwargs)
        self.map_path, self.map_ext = map_path, map_ext
        TuringEnv._active = weakref.ref(self)
        self._stale = False

        self.sensor = D500Model(self.base_sensor_cfg, seed=seed)
        self.sensor_cfg = self.base_sensor_cfg
        self._obs = None

    # ------------------------------------------------------------------ API
    def reset(self, pose: Sequence[float], seed: Optional[int] = None) -> Dict:
        """Place the car at (x, y, yaw), restart the sensor, and wait until the first scan
        is usable. Returns the first observation."""
        self._check_live()
        ep_seed = int(self._rng.integers(2 ** 31)) if seed is None else int(seed)
        ep_rng = np.random.default_rng(ep_seed)
        self.sensor_cfg = (sample_config(self.sensor_randomization, ep_rng, self.base_sensor_cfg)
                           if self.sensor_randomization else self.base_sensor_cfg)
        self.sensor.reset(0.0, seed=ep_seed, cfg=self.sensor_cfg)
        self.episode_seed = ep_seed

        obs, _, _, _ = self.env.reset(np.array([pose], dtype=np.float64))
        self._obs, self._t = obs, 0.0
        self._push()
        self._grid = np.full(len(GRID), RMAX, np.float32)
        self._scan, self._scan_seq = None, 0
        self._collided, self._prev_laps = False, 0
        self._lap_t0, self.lap_times = 0.0, []
        self._last_action = (0.0, 0.0)

        # warm-up: car held still until the first scan exists (D-21), max 1 s
        while self.sensor.read(self._t) is None and self._t < 1.0:
            self._physics((0.0, 0.0))
        self._t0 = self._t                       # episode clock starts at the first scan
        self._lap_t0 = self._t
        self._refresh_scan()
        return self._pack()

    def step(self, action: Sequence[float]) -> Dict:
        """Hold (steer rad, speed m/s) for one control period (5 physics steps at 20 Hz)."""
        self._check_live()
        if self._obs is None:
            raise RuntimeError("call reset() before step()")
        steer, speed = float(action[0]), float(action[1])
        if not (np.isfinite(steer) and np.isfinite(speed)):
            raise ValueError(f"non-finite action {action}")
        self._last_action = (steer, speed)
        for _ in range(self.k_ctrl):
            self._physics((steer, speed))
            if self._collided:
                break
        self._refresh_scan()
        if self.render_on:
            self.env.render(mode="human_fast")
        return self._pack()

    def add_render_callback(self, fn):
        """fn(env_renderer) is called every rendered frame (gym main's EnvRenderer: pyglet,
        50 px per metre). Used by scripts/drive_live.py to draw the D500 points."""
        self.env.add_render_callback(fn)

    def close(self):
        self._stale = True
        if TuringEnv._active is not None and TuringEnv._active() is self:
            TuringEnv._active = None

    # ------------------------------------------------------------------ internals
    def _check_live(self):
        if self._stale:
            raise StaleEnvError("another TuringEnv was created after this one; f1tenth_gym main "
                                "shares its map between instances, so this one would silently "
                                "see the other map. Create one TuringEnv at a time (D-27).")

    def _physics(self, action):
        obs, _, _, _ = self.env.step(np.array([action], dtype=np.float64))
        self._obs = obs
        self._t = round(self._t + SIM_DT, 9)
        self._push()
        if obs["collisions"][0]:
            self._collided = True
        laps = int(obs["lap_counts"][0])
        if laps > self._prev_laps:               # D-25: per-lap times from count increments
            self.lap_times.append(self._t - self._lap_t0)
            self._lap_t0, self._prev_laps = self._t, laps

    def _push(self):
        self.sensor.push(self._t, self._obs["scans"][0], GYM_ANGLE_MIN, GYM_ANGLE_INC)

    def _refresh_scan(self):
        scan = self.sensor.read(self._t)
        if scan is not None and scan.seq != self._scan_seq:
            self._grid = to_grid(scan.ranges, scan.angle_min, scan.angle_increment)
            self._scan, self._scan_seq = scan, scan.seq

    def _pack(self) -> Dict:
        o, t = self._obs, self._t - self._t0
        laps = len(self.lap_times)
        done = self._collided or laps >= self.max_laps
        return dict(
            # what a policy may use (the car has these too)
            grid=self._grid.copy(),
            speed=float(o["linear_vels_x"][0]),
            yaw_rate=float(o["ang_vels_z"][0]),
            # ground truth: pure pursuit (privileged expert), metrics and plots only
            pose=(float(o["poses_x"][0]), float(o["poses_y"][0]), float(o["poses_theta"][0])),
            collision=self._collided,
            laps=laps,
            lap_times=list(self.lap_times),
            # timing / bookkeeping
            t=t,
            scan_seq=self._scan_seq,
            scan_age=(self._t - self._scan.stamp) if self._scan is not None else float("nan"),
            last_action=self._last_action,
            done=bool(done),
            truncated=bool(not done and t >= self.max_time),
        )

    @property
    def last_scan(self):
        """The raw D500-format scan behind the current grid (for logging / debugging)."""
        return self._scan


def _resolve_map(map_path: str, map_ext: Optional[str]):
    """Accept 'name', 'path/name', or 'path/name.yaml'; find the image extension (D-26)."""
    if map_path in BUNDLED_MAPS:
        # always hand the gym a full path: its name lookup misses 'vegas', and the
        # renderer needs a path, not a name
        import f110_gym
        map_path = os.path.join(os.path.dirname(f110_gym.__file__), "envs", "maps", map_path)
    base = map_path[:-5] if map_path.endswith(".yaml") else map_path
    if not os.path.exists(base + ".yaml"):
        raise FileNotFoundError(f"no map yaml at {base}.yaml")
    if map_ext is None:
        map_ext = _image_ext(base)
    return os.path.abspath(base), map_ext


def _image_ext(base: str) -> str:
    for ext in (".png", ".pgm", ".jpg"):
        if os.path.exists(base + ext):
            return ext
    raise FileNotFoundError(f"no .png/.pgm/.jpg image next to {base}.yaml")
