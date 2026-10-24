"""Drive a policy in TuringEnv and (optionally) watch it live.

The window shows the map (white points), the car (purple) and, in red, the D500 points the
driver actually saw, drawn at the car's CURRENT pose. When the car moves fast you can watch
them run ahead of the real walls: that gap is the sensor's age (DECISIONS D-05).

    python scripts/drive_live.py --map Spielberg                 # f1tenth_racetracks name
    python scripts/drive_live.py --map ring --preset harsh
    python scripts/drive_live.py --map Monza --v-max 4 --no-render --laps 2
    python scripts/drive_live.py --map maps/TuringTrack/TuringTrack_map.yaml --start 0.875 -4.225 0 --driver pursuit
With --driver pursuit the car follows <Name>_raceline.csv (scripts/make_raceline.py), drawn in green.
It uses the sim's TRUE pose; --pose-delay / --pose-noise imitate a particle filter.
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

from turing_ml import raceline as rl                           # noqa: E402
from turing_ml import testmaps, tracks                         # noqa: E402
from turing_ml.d500 import load_preset, load_sensor_yaml        # noqa: E402
from turing_ml.env import TuringEnv                             # noqa: E402
from turing_ml.experts import disparity_extender, load_expert_config  # noqa: E402
from turing_ml.pursuit import PurePursuit                        # noqa: E402
from turing_ml.vehicle import load_vehicle, vehicle_lidar_dist   # noqa: E402

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


def make_scan_drawer(env, state, follow=True, line=None):
    """Render callback: draw the latest D500 scan in red (and the racing line in green), move the
    camera with the car."""
    from pyglet import gl
    handle = {}

    def draw(r):
        gl.glPointSize(3)
        if line is not None and "line" not in handle:
            handle["line"] = r.batch.add(len(line), gl.GL_POINTS, None,
                                         ("v3f/static", np.c_[line * PX, np.zeros(len(line))].ravel().tolist()),
                                         ("c3B/static", [40, 200, 40] * len(line)))
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
            lx, ly = x + env.lidar_dist * math.cos(yaw), y + env.lidar_dist * math.sin(yaw)
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
    ap.add_argument("--vehicle", default="sbk10", help="vehicle preset in configs/vehicle.yaml")
    ap.add_argument("--randomize", action="store_true", help="per-episode sensor randomization")
    ap.add_argument("--v-max", type=float, help="override expert v_max")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override an expert.yaml value")
    ap.add_argument("--laps", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=400.0, help="sim seconds (a harsh Sepang lap is ~181 s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="pace the window at wall-clock speed")
    ap.add_argument("--free-camera", action="store_true")
    ap.add_argument("--driver", choices=["extender", "pursuit"], default="extender",
                    help="extender = reactive LiDAR driver; pursuit = follow the racing line (needs the pose)")
    ap.add_argument("--raceline", help="racing line csv (default: <Name>_raceline.csv next to the map)")
    ap.add_argument("--speed-scale", type=float, help="pursuit: multiply the racing-line speeds")
    ap.add_argument("--pose-delay", type=float, help="pursuit: pose arrives this late (s), like a particle filter")
    ap.add_argument("--pose-noise", type=float, help="pursuit: pose position noise std (m); yaw noise = this / 5 rad")
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
    if a.set:
        cfg = dataclasses.replace(cfg, **{k: float(v) for k, v in (kv.split("=", 1) for kv in a.set)})

    env = TuringEnv(map_path, sensor_cfg=load_preset(sensor_yaml, a.preset),
                    sensor_randomization=load_sensor_yaml(sensor_yaml)["randomization"] if a.randomize else None,
                    params=load_vehicle(str(ROOT / "configs" / "vehicle.yaml"), a.vehicle),
                    lidar_dist=vehicle_lidar_dist(str(ROOT / "configs" / "vehicle.yaml"), a.vehicle),
                    max_laps=a.laps, max_time=a.max_time, seed=a.seed, render=not a.no_render)
    state = dict(pose=start, t=0.0, v=0.0, laps=0, age=0.0, preset=a.preset)
    drv, line_xy = None, None
    if a.driver == "pursuit":
        line_file = a.raceline or rl.find_raceline(map_path)
        if not line_file:
            raise SystemExit("no racing line for this map: run scripts/make_raceline.py first")
        pcfg = rl.load_raceline_config(str(ROOT / "configs" / "raceline.yaml"), "pursuit")
        for key, val in (("speed_scale", a.speed_scale), ("pose_delay", a.pose_delay),
                         ("pose_noise_xy", a.pose_noise),
                         ("pose_noise_yaw", None if a.pose_noise is None else a.pose_noise / 5)):
            if val is not None:
                pcfg[key] = val
        line = rl.load_raceline(line_file)
        veh = load_vehicle(str(ROOT / "configs" / "vehicle.yaml"), a.vehicle)
        drv = PurePursuit(line, pcfg, veh, seed=a.seed)
        line_xy = np.stack([line["x"], line["y"]], 1)
        start = rl.line_start_pose(line, veh["lr"])       # on the start line, where the racing line crosses it
    if not a.no_render:
        env.add_render_callback(make_scan_drawer(env, state, follow=not a.free_camera, line=line_xy))

    obs = env.reset(start, seed=a.seed)
    speeds, steers, wall0 = [], [], time.time()
    try:
        while not (obs["done"] or obs["truncated"]):
            if drv is not None:
                action = drv(obs["pose"], obs["speed"], obs["t"], obs["yaw_rate"])
            else:
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
    who = (f"driver=pursuit pose_delay={drv.cfg['pose_delay']} pose_noise={drv.cfg['pose_noise_xy']}"
           if drv is not None else f"driver=extender v_max={cfg.v_max}")
    print(f"map={a.map} vehicle={a.vehicle} preset={a.preset} {who} seed={a.seed}")
    print(f"  result: {'COLLISION' if obs['collision'] else 'laps ' + str(obs['laps'])} "
          f"after {obs['t']:.1f} s sim ({time.time() - wall0:.1f} s wall)")
    print(f"  lap times: {[round(x, 2) for x in obs['lap_times']]}")
    print(f"  mean speed {np.mean(speeds):.2f} m/s, max {np.max(speeds):.2f}, "
          f"smoothness (mean |d steer|) {np.mean(np.abs(np.diff(steers))):.3f} rad/step")
    env.close()


if __name__ == "__main__":
    main()
