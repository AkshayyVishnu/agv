# turing-ml

ML / algorithms code for Turing Track: sensor model, experts, gym harness, policies.
Layout follows the team Execution Manual. Design decisions: [docs/DECISIONS.md](docs/DECISIONS.md).
Install steps and all commands: [top-level README](../README.md). Testing on other tracks, sensors or
simulators: [docs/SIMULATION_GUIDE.md](../docs/SIMULATION_GUIDE.md).

## What's here so far (Person B, Day 0)
| Path | What |
|---|---|
| `turing_ml/d500.py` | D500 LiDAR model: feed it perfect sim scans, get back what the real driver publishes |
| `configs/sensor.yaml` | presets `nominal`, `harsh`, `clean` (tests only) + training randomization |
| `tests/test_d500.py` | 25 unit tests |
| `scripts/d500_gym_smoke.py` | integration check on f1tenth_gym main: scan staleness and seam vs ground truth |
| `turing_ml/env.py` | `TuringEnv`: gym main at 20 Hz control, sensing through the D500 model, one obs dict |
| `turing_ml/sensor.py` | `to_grid`: **Person A's file**, placeholder with the NaN fix until A pushes |
| `turing_ml/testmaps.py` | generated corridor / ring maps for tests |
| `tests/test_env.py` | 11 env tests (skipped if `f110_gym` isn't installed) |
| `turing_ml/experts.py` | disparity extender (**Person A's deliverable**, written by B on request), settings in `configs/expert.yaml` |
| `turing_ml/tracks.py` | `racetrack(name)`: map path + start pose from `../f1tenth_racetracks` |
| `tests/test_experts.py` | expert tests on synthetic grids (no gym needed) |
| `scripts/eval_expert.py` | one command: map × preset × seed → lap time or failure, with crash diagnostics |
| `scripts/drive_live.py` | drive the extender and watch it in a window |

## Using TuringEnv
```python
from turing_ml.env import TuringEnv
from turing_ml.d500 import load_preset
env = TuringEnv("path/to/map", sensor_cfg=load_preset("configs/sensor.yaml", "nominal"), seed=0)
obs = env.reset((x, y, yaw))            # waits for the first scan
while not (obs["done"] or obs["truncated"]):
    obs = env.step((steer, speed))       # held for one 20 Hz period
# policies may use obs["grid"], obs["speed"], obs["yaw_rate"]; obs["pose"] is ground truth only
```
One `TuringEnv` per process at a time (gym main shares the map between instances).

## Evaluate the expert (one command)
    python scripts/eval_expert.py                       # 3 tuning + 5 held-out maps, nominal+harsh, seeds 0,1,2
    python scripts/eval_expert.py --maps Monza --presets harsh --seeds 0,1,2,3
    python scripts/eval_expert.py --maps Spielberg --set bubble_width=0.2   # try a value without editing the yaml
It prints a table (lap time in sim seconds, or `crash@t` / `stuck@t` / `timeout`). For every failure it also
prints the last 0.6 s before it: grid ranges at -90/-45/0/+45/+90 deg, steer, and commanded and actual speed.
Tuning maps: Spielberg, Monza, Silverstone. Held out (never tune on them, D-38): Catalunya, Hockenheim,
Budapest, Sepang, Zandvoort. Tracks come from `../f1tenth_racetracks` (or `$TURING_RACETRACKS`).

## How to watch it live
    python scripts/drive_live.py --map Spielberg --laps 1                   # opens a window
    python scripts/drive_live.py --map Monza --preset harsh --realtime      # paced at wall-clock speed
    python scripts/drive_live.py --map ring --no-render                     # headless, just the summary
The window shows the map (white), the car, and **in red the D500 points the driver actually saw**, drawn at the
car's current pose. When the car is fast they run ahead of the real walls: that is the scan age (D-05).
The camera follows the car; use `--free-camera` to pan and zoom yourself (scroll = zoom, drag = pan). Close the
window to stop; a summary (laps, lap times, mean speed) is printed. Without `--realtime` the sim runs as fast as
it can draw. `--preset clean` is refused (team rule).

## Setup (team container, or any Python 3.8+)
    pip install -e .
    python -m pytest tests
    python scripts/d500_gym_smoke.py --preset nominal --speed 4   # needs f110_gym (gym main)

## Using the D500 model
```python
from turing_ml.d500 import D500Model, load_preset
m = D500Model(load_preset("configs/sensor.yaml", "nominal"), seed=0)
m.reset(t0=0.0)
# every physics step (10 ms):
m.push(t, obs["scans"][0], angle_min=-2.35, angle_increment=4.7 / 1079)
# every control step (50 ms):
scan = m.read(t)          # None until the first revolution + latency
if scan is not None and scan.seq != last_seq:
    ...                   # scan.ranges: float32, NaN = no return, angles 0..2pi
```
Rule: evaluate and tune at `nominal` **and** `harsh`. Never use `clean` for results.
