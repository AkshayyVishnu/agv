"""Integration check: D500 model on f1tenth_gym main (the team simulator's physics).

Drives straight at a wall and measures how stale the model's front reading is versus the
gym's ground truth. With clockwise spin the front beam is measured at the START of each
revolution, so its age when usable should be ~ one period + latency (~0.15 s nominal).
A wrong skew or latency sign shows up here as a lag near 0 or negative.

    python scripts/d500_gym_smoke.py --preset clean --speed 4
"""
import argparse
import pathlib

import gym
import numpy as np

from turing_ml import testmaps
from turing_ml.d500 import D500Model, config_from_dict, load_preset

ROOT = pathlib.Path(__file__).resolve().parents[1]
FOV, NB, DT = 4.7, 1080, 0.01   # f1tenth_gym main defaults


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="clean")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = load_preset(str(ROOT / "configs" / "sensor.yaml"), a.preset)
    if a.preset == "clean":
        cfg = config_from_dict({"latency_s": 0.05}, cfg)  # isolate skew + latency
    map_base = testmaps.corridor(length=40.0, width=4.0)
    env = gym.make("f110_gym:f110-v0", map=map_base, map_ext=".png", num_agents=1, timestep=DT)
    obs, *_ = env.reset(np.array([[1.0, 2.0, 0.0]]))   # centre of the corridor, facing the end wall
    model = D500Model(cfg, seed=a.seed)
    model.reset(0.0)

    t, truth_t, truth_r = 0.0, [], []
    sides = {"right of seam (-3..-0.5 deg)": (-3.0, -0.5), "left of seam (+0.5..+3 deg)": (0.5, 3.0)}
    rows = {k: [] for k in sides}
    steps = []
    while True:
        front_true = float(obs["scans"][0][NB // 2])
        truth_t.append(t), truth_r.append(front_true)
        model.push(t, obs["scans"][0], -FOV / 2, FOV / (NB - 1))
        for s in model.poll(t):
            ang = s.angle_min + np.arange(len(s.ranges)) * s.angle_increment
            wrapped = np.degrees((ang + np.pi) % (2 * np.pi) - np.pi)
            med = {}
            for k, (lo, hi) in sides.items():
                v = s.ranges[(wrapped >= lo) & (wrapped <= hi)]
                v = v[np.isfinite(v)]
                if v.size:
                    med[k] = float(np.median(v))
                    rows[k].append((s.available_at, med[k]))
            if len(med) == 2:
                steps.append((s.available_at, med[list(sides)[0]] - med[list(sides)[1]]))
        if front_true < 1.0 or t > 30:
            break
        obs, _, done, _ = env.step(np.array([[0.0, a.speed]]))
        t = round(t + DT, 6)
        if done:
            break

    truth_t, truth_r = np.array(truth_t), np.array(truth_r)
    v_end = float(obs["linear_vels_x"][0])
    # compare only while the wall is in the sensor's visible range (and the gym's 30 m range)
    limit = 25.0 if a.preset == "clean" else min(3.5, cfg.wall_visible_range - 0.5)
    print(f"preset={a.preset} seed={a.seed} speed={v_end:.2f} m/s, period {cfg.period*1000:.0f} ms, "
          f"latency {cfg.latency_s*1000:.0f} ms, spin {cfg.spin_direction}")
    lags = np.arange(0.0, 0.4, 0.005)
    for k, rs in rows.items():
        rs = [(ta, rf) for ta, rf in rs if ta > 1.0 and np.interp(ta, truth_t, truth_r) < limit]
        if len(rs) < 3:
            print(f"  {k}: too few scans ({len(rs)})")
            continue
        err = [np.median([abs(np.interp(ta - L, truth_t, truth_r) - rf) for ta, rf in rs]) for L in lags]
        best = lags[int(np.argmin(err))]
        print(f"  {k}: lag {best*1000:.0f} ms over {len(rs)} scans (median |err| {min(err):.3f} m)"
              f" -> {best*v_end:.2f} m stale")
    st = [d for ta, d in steps if ta > 1.0 and np.interp(ta, truth_t, truth_r) < limit]
    if st:
        print(f"  seam step straight ahead (right - left): median {np.median(st):.2f} m"
              f" (expected ~ speed x period = {v_end*cfg.period:.2f} m)")


if __name__ == "__main__":
    main()
