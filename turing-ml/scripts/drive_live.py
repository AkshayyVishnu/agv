"""Drive a policy in TuringEnv and (optionally) watch it live.

The window shows the map (white points), the car (purple) and, in red, the D500 points the
driver actually saw, drawn at the car's CURRENT pose. When the car moves fast you can watch
them run ahead of the real walls: that gap is the sensor's age (DECISIONS D-05).

    python scripts/drive_live.py --map Spielberg                 # f1tenth_racetracks name
    python scripts/drive_live.py --map ring --preset harsh
    python scripts/drive_live.py --map Monza --v-max 4 --no-render --laps 2
Keys: close the window to stop. Mouse: scroll = zoom, drag = pan (camera follows the car
unless --free-camera).
"""
import argparse
import dataclasses
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turing_ml import testmaps, tracks                         # noqa: E402
from turing_ml.d500 import load_preset, load_sensor_yaml        # noqa: E402
from turing_ml.env import LIDAR_DIST, TuringEnv                 # noqa: E402
from turing_ml.experts import disparity_extender, load_expert_config  # noqa: E402

PX = 50.0            # gym main renderer: pixels per metre
MAX_POINTS = 800


def resolve(name, start=None):
    if name == "ring":
        path, (cx, cy) = testmaps.ring(radius=5.0, width=2.0)
        return path, (cx, cy - 5.0, 0.0)
    if name == "corridor":
        return testmaps.corridor(), (2.0, 2.0, 0.0)
    if pathlib.Path(name + ".yaml").exists() or name.endswith(".yaml"):
        if start is None:
            raise SystemExit("for a custom map also pass --start x y yaw")
        return name[:-5] if name.endswith(".yaml") else name, tuple(start)
    return tracks.racetrack(name)


def make_scan_drawer(env, state, follow=True):
    """Render callback: draw the latest D500 scan in red, move the camera with the car."""
    from pyglet import gl
    handle = {}

    def draw(r):
        gl.glPointSize(3)
        if "pts" not in handle:
            handle["pts"] = r.batch.add(MAX_POINTS, gl.GL_POINTS, None,
                                        ("v3f/stream", [0.0] * 3 * MAX_POINTS),
                                        ("c3B/stream", [255, 60, 60] * MAX_POINTS))
        x, y, yaw = state["pose"]
        scan = env.last_scan
        verts = np.zeros((MAX_POINTS, 3))
        verts[:, 0], verts[:, 1] = x * PX, y * PX               # unused points hide under the car
        if scan is not None:
            r_ = scan.ranges.astype(np.float64)
            ang = scan.angle_min + np.arange(len(r_)) * scan.angle_increment + env.sensor_cfg.mount_yaw
            ok = np.isfinite(r_)
            lx, ly = x + LIDAR_DIST * math.cos(yaw), y + LIDAR_DIST * math.sin(yaw)
            px = lx + r_[ok] * np.cos(yaw + ang[ok])
            py = ly + r_[ok] * np.sin(yaw + ang[ok])
            n = min(len(px), MAX_POINTS)
            verts[:n, 0], verts[:n, 1] = px[:n] * PX, py[:n] * PX
        handle["pts"].vertices = verts.ravel().tolist()
        if follow:
            half_w, half_h = 900, 600
            r.left, r.right = x * PX - half_w, x * PX + half_w
            r.bottom, r.top = y * PX - half_h, y * PX + half_h
        r.score_label.x, r.score_label.y = x * PX, y * PX - 500
        r.score_label.font_size = 18
        r.score_label.text = (f"t {state['t']:5.1f}s  v {state['v']:.2f} m/s  laps {state['laps']}  "
                              f"scan age {state['age'] * 1000:3.0f} ms  [{state['preset']}]")
    return draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="Spielberg", help="racetrack name, 'ring', 'corridor', or a map path (with --start)")
    ap.add_argument("--start", type=float, nargs=3, help="x y yaw (overrides the map's start)")
    ap.add_argument("--preset", default="nominal", help="sensor preset: nominal | harsh")
    ap.add_argument("--randomize", action="store_true", help="per-episode sensor randomization")
    ap.add_argument("--v-max", type=float, help="override expert v_max")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=400.0, help="sim seconds (a harsh Sepang lap is ~181 s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="pace the window at wall-clock speed")
    ap.add_argument("--free-camera", action="store_true")
    a = ap.parse_args()

    if a.preset == "clean":
        raise SystemExit("'clean' is for unit tests only (team rule: never judge a driver on it)")
    map_path, start = resolve(a.map, a.start)
    if a.start:
        start = tuple(a.start)
    sensor_yaml = str(ROOT / "configs" / "sensor.yaml")
    cfg = load_expert_config(str(ROOT / "configs" / "expert.yaml"))
    if a.v_max:
        cfg = dataclasses.replace(cfg, v_max=a.v_max)

    env = TuringEnv(map_path, sensor_cfg=load_preset(sensor_yaml, a.preset),
                    sensor_randomization=load_sensor_yaml(sensor_yaml)["randomization"] if a.randomize else None,
                    max_laps=a.laps, max_time=a.max_time, seed=a.seed, render=not a.no_render)
    state = dict(pose=start, t=0.0, v=0.0, laps=0, age=0.0, preset=a.preset)
    if not a.no_render:
        env.add_render_callback(make_scan_drawer(env, state, follow=not a.free_camera))

    obs = env.reset(start, seed=a.seed)
    speeds, steers, wall0 = [], [], time.time()
    try:
        while not (obs["done"] or obs["truncated"]):
            action = disparity_extender(obs["grid"], obs["speed"], cfg, obs["yaw_rate"])
            obs = env.step(action)
            speeds.append(obs["speed"]), steers.append(action[0])
            state.update(pose=obs["pose"], t=obs["t"], v=obs["speed"], laps=obs["laps"],
                         age=obs["scan_age"] if np.isfinite(obs["scan_age"]) else 0.0)
            if a.realtime and not a.no_render:
                lag = obs["t"] - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
    except Exception as ex:                       # closing the window raises in gym main
        if "window was closed" not in str(ex):
            raise
        print("window closed")

    steers = np.array(steers)
    print(f"map={a.map} preset={a.preset} v_max={cfg.v_max} seed={a.seed}")
    print(f"  result: {'COLLISION' if obs['collision'] else 'laps ' + str(obs['laps'])} "
          f"after {obs['t']:.1f} s sim ({time.time() - wall0:.1f} s wall)")
    print(f"  lap times: {[round(x, 2) for x in obs['lap_times']]}")
    print(f"  mean speed {np.mean(speeds):.2f} m/s, max {np.max(speeds):.2f}, "
          f"smoothness (mean |d steer|) {np.mean(np.abs(np.diff(steers))):.3f} rad/step")
    env.close()


if __name__ == "__main__":
    main()
