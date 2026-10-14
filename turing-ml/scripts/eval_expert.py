"""Evaluate the disparity extender: map x preset x seed -> lap time or failure (D-37, D-38).

    python scripts/eval_expert.py                                  # tuning + held-out maps, nominal+harsh, seeds 0,1,2
    python scripts/eval_expert.py --maps Spielberg,Monza --presets harsh --seeds 0
    python scripts/eval_expert.py --set bubble_width=0.4 --set v_max=2.5   # try parameters without editing the yaml

One run = one TuringEnv in its own worker process (D-27). A failure prints the last steps
before it (grid at -90/-45/0/+45/+90 deg, steer, speed) so the cause can be read, not guessed.
Never judge a driver on the `clean` preset (team rule); it is refused here.
"""
import argparse
import dataclasses
import math
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TUNING = ["Spielberg", "Monza", "Silverstone"]
HELD_OUT = ["Catalunya", "Hockenheim", "Budapest", "Sepang", "Zandvoort"]   # D-38: never tuned on
DIAG_DEG = (-90, -45, 0, 45, 90)


def run_one(job):
    """(map, preset, seed, overrides, max_time) -> dict. Runs in a worker process."""
    from turing_ml import testmaps, tracks
    from turing_ml.d500 import load_preset
    from turing_ml.env import TuringEnv
    from turing_ml.experts import disparity_extender, load_expert_config
    from turing_ml.sensor import GRID

    import warnings
    warnings.filterwarnings("ignore", message="Chosen integrator is RK4")
    name, preset, seed, overrides, max_time = job
    cfg = dataclasses.replace(load_expert_config(str(ROOT / "configs" / "expert.yaml")), **overrides)
    if name == "ring":
        path, (cx, cy) = testmaps.ring(radius=5.0, width=2.0)
        start = (cx, cy - 5.0, 0.0)
    else:
        path, start = tracks.racetrack(name)
    env = TuringEnv(path, sensor_cfg=load_preset(str(ROOT / "configs" / "sensor.yaml"), preset),
                    max_laps=1, max_time=max_time, seed=seed)
    idx = [int(np.argmin(np.abs(GRID - math.radians(d)))) for d in DIAG_DEG]
    obs, hist, still, wall0 = env.reset(start, seed=seed), [], 0, time.time()
    while not (obs["done"] or obs["truncated"]):
        steer, speed = disparity_extender(obs["grid"], obs["speed"], cfg, obs["yaw_rate"])
        hist.append((obs["t"], [float(obs["grid"][i]) for i in idx], steer, speed, obs["speed"]))
        hist = hist[-12:]
        obs = env.step((steer, speed))
        still = still + 1 if obs["speed"] < 0.05 and obs["t"] > 2.0 else 0
        if still > 40:                                   # 2 s standing still: the driver gave up
            break
    env.close()
    if obs["laps"]:
        result = f"{obs['lap_times'][0]:.0f}s"
    elif obs["collision"]:
        result = f"crash@{obs['t']:.0f}"
    elif still > 40:
        result = f"stuck@{obs['t']:.0f}"
    else:
        result = "timeout"
    return dict(map=name, preset=preset, seed=seed, ok=bool(obs["laps"]), result=result,
                pose=obs["pose"], hist=hist, wall=time.time() - wall0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", default=",".join(TUNING + HELD_OUT), help="comma list (racetrack names or 'ring')")
    ap.add_argument("--presets", default="nominal,harsh")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override an expert.yaml value")
    ap.add_argument("--max-time", type=float, default=400.0, help="sim seconds per lap attempt")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--quiet", action="store_true", help="no per-failure diagnostics")
    a = ap.parse_args()

    maps = [m for m in a.maps.split(",") if m]
    presets = [p for p in a.presets.split(",") if p]
    seeds = [int(s) for s in a.seeds.split(",") if s]
    if "clean" in presets:
        raise SystemExit("'clean' is for unit tests only (team rule: never judge a driver on it)")
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        overrides[k] = float(v) if v.lower() not in ("true", "false") else v.lower() == "true"

    jobs = [(m, p, s, overrides, a.max_time) for m in maps for p in presets for s in seeds]
    t0 = time.time()
    print(f"{len(jobs)} runs on {a.workers} workers, overrides {overrides or 'none'} ...", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    by = {(r["map"], r["preset"], r["seed"]): r for r in res}

    cols = [(p, s) for p in presets for s in seeds]
    head = f"{'map':12s} {'set':8s} | " + " | ".join(f"{p[:4]} s{s}".rjust(9) for p, s in cols)
    print("\n" + head + "\n" + "-" * len(head))
    for m in maps:
        tag = "tuning" if m in TUNING else ("held-out" if m in HELD_OUT else "other")
        print(f"{m:12s} {tag:8s} | " + " | ".join(by[(m, p, s)]["result"].rjust(9) for p, s in cols))
    n_ok = sum(r["ok"] for r in res)
    print(f"\nlapped {n_ok}/{len(res)}  ({time.time() - t0:.0f} s wall)")

    if not a.quiet:
        for r in res:
            if r["ok"]:
                continue
            x, y, yaw = r["pose"]
            print(f"\n--- {r['map']} {r['preset']} seed {r['seed']}: {r['result']} at "
                  f"x={x:.1f} y={y:.1f} yaw={yaw:.2f}")
            print("     t   " + " ".join(f"{d:>+6d}" for d in DIAG_DEG) + "   steer  cmd_v  v")
            for t, g, st, sp, v in r["hist"]:
                print(f"  {t:6.2f} " + " ".join(f"{x_:6.2f}" for x_ in g) + f"  {st:+.3f} {sp:5.2f} {v:5.2f}")
    return 0 if n_ok == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
