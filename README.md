# AGV: Turing Track autonomous racing (ML harness)

Workspace for the **Turing Track** team's ML/algorithms work on a 1/10-scale F1TENTH/RoboRacer-style car
(Raspberry Pi 5 + LDRobot D500 LiDAR). The main code is in [`turing-ml/`](turing-ml/). Simulators,
race tracks and reference stacks are pinned as git submodules next to it.

- **What's implemented:** see the [list below](#whats-implemented). For the full write-up see
  [`turing-ml/docs/PROJECT_REPORT.md`](turing-ml/docs/PROJECT_REPORT.md).
- **How to run simulations:** see [Quick start](#quick-start) and [Commands](#commands).
- **How to test a model in a different track, sensor setup or simulator:**
  see [`docs/SIMULATION_GUIDE.md`](docs/SIMULATION_GUIDE.md).
- **Why each design choice was made (D-01…D-41):** see [`turing-ml/docs/DECISIONS.md`](turing-ml/docs/DECISIONS.md).

---

## What's implemented

| Component | File | Status |
|---|---|---|
| **D500 LiDAR model.** Turns the gym's perfect scan into what the real `ldlidar` driver publishes: 10 Hz rotation with motion skew, 50–190 ms latency, range noise, dropouts, dark-wall bursts, mixed pixels, over-the-wall and pitch events, NaN = no return, angles 0…2π | `turing_ml/d500.py`, `configs/sensor.yaml` | ✅ 25 tests, checked against the gym (scan age within 5 ms of the design value) |
| **`TuringEnv`.** Wraps `f1tenth_gym` (main) to behave like the car: 100 Hz physics, 20 Hz control, sensing only through the D500 model, one observation dict | `turing_ml/env.py` | ✅ 11 tests, ~140× real time |
| **`to_grid`.** Any scan → 271 ranges on a 1° grid (−135°…+135°), capped at 4 m, with NaN treated as unknown | `turing_ml/sensor.py` | ✅ (placeholder for Person A's version, includes the NaN fix) |
| **Disparity-extender driver.** Reactive, LiDAR-only. Adds scan-age (yaw) compensation, a seam rule, a safety bubble, side clearance and a stopping-distance speed cap | `turing_ml/experts.py`, `configs/expert.yaml` | ✅ **48/48 laps**: 8 tracks × {nominal, harsh} × 3 seeds, up to 3 m/s |
| Track loader (`f1tenth_racetracks` → map + start pose) and generated test maps (ring, corridor) | `turing_ml/tracks.py`, `turing_ml/testmaps.py` | ✅ |
| One-command evaluation with crash diagnostics | `scripts/eval_expert.py` | ✅ |
| Live viewer that draws the D500 points the driver actually saw | `scripts/drive_live.py` | ✅ |
| Parameter sweeps, sensor timing smoke test | `scripts/sweep_expert.py`, `scripts/d500_gym_smoke.py` | ✅ |
| ROS 2 node for the car / `f1tenth_gym_ros` | – | ❌ **not yet.** Procedure in the [simulation guide](docs/SIMULATION_GUIDE.md#4-ros-2-simulator-f1tenth_gym_ros) |
| Imitation learning (`collect.py`, BC/DAgger), neural policy | – | ❌ next phase |

**Main lesson so far:** almost every crash in simulation came from the sensor being **old** (100–190 ms on a
10 Hz rotating LiDAR), not from noise or speed. Rotating the scan by `yaw_rate × 0.15 s` (D-40) was the fix
that worked best.

### Evaluation results (lap time in sim seconds, seeds 0/1/2)

| Track | Set | nominal | harsh |
|---|---|---|---|
| Spielberg | tuning | 124 / 124 / 124 | 129 / 128 / 129 |
| Monza | tuning | 157 / 157 / 157 | 162 / 163 / 162 |
| Silverstone | tuning | 167 / 167 / 167 | 172 / 171 / 172 |
| Catalunya | held-out | 146 / 146 / 146 | 150 / 150 / 149 |
| Hockenheim | held-out | 135 / 135 / 135 | 139 / 138 / 139 |
| Budapest | held-out | 140 / 140 / 141 | 146 / 147 / 146 |
| Sepang | held-out | 176 / 176 / 176 | 181 / 182 / 180 |
| Zandvoort | held-out | 146 / 146 / 145 | 149 / 151 / 149 |

---

## Repository layout

```
agv/
├── turing-ml/                         our code (Python package turing_ml, configs, scripts, tests, docs)
├── docs/SIMULATION_GUIDE.md           how to test a model on other tracks / sensors / simulators
│
├── f1tenth_gym/          (submodule)  simulator physics, main branch: the version the team runs
├── f1tenth_racetracks/   (submodule)  25 scaled F1 circuits (maps + centre lines)
├── f1tenth_gym_ros/      (submodule)  ROS 2 Humble bridge for the gym (dev-humble)
├── turing-track-ros2/    (submodule)  team infra: topic contract, Docker sim setup guide, d500_relay
├── ldlidar_stl_ros2/     (submodule)  real D500 driver (source of truth for the scan format)
├── race_stack/           (submodule)  ForzaETH race stack (reference)
├── f1tenth_ws/           (submodule)  UWaterloo reference nodes
└── global_racetrajectory_optimization/ (submodule)  TUM raceline optimizer (reference)
```

Only `f1tenth_gym` and `f1tenth_racetracks` are needed to run anything in `turing-ml`. The others are
for reference or for the ROS 2 setup.

---

## Quick start

Requirements: **Python 3.10** (the team container's version), git, and a display if you want the live viewer.
The simulator needs the old `gym==0.19.0` and `numpy==1.22.0`, which need a few install workarounds.

### 1. Clone

```bash
git clone https://github.com/AkshayyVishnu/agv.git
cd agv
git submodule update --init f1tenth_gym f1tenth_racetracks     # the two you need
# optional, everything (large: race_stack pulls its own submodules):
# git submodule update --init --recursive
```

### 2. Create the environment

Linux / macOS:
```bash
python3.10 -m venv .venv-main
source .venv-main/bin/activate
python -m pip install "pip<24.1" "setuptools<66" "wheel<0.40"
python -m pip install --no-build-isolation gym==0.19.0      # gym 0.19 won't build any other way today
python -m pip install -e f1tenth_gym                         # pulls numpy 1.22, numba, pyglet 1.4
python -m pip install -e turing-ml pytest
```

Windows (PowerShell):
```powershell
py -3.10 -m venv .venv-main
.\.venv-main\Scripts\Activate.ps1
python -m pip install "pip<24.1" "setuptools<66" "wheel<0.40"
python -m pip install --no-build-isolation gym==0.19.0
python -m pip install -e f1tenth_gym
python -m pip install -e turing-ml pytest
```

The exact versions known to work are in [`turing-ml/requirements-sim.txt`](turing-ml/requirements-sim.txt).

### 3. Check it works

```bash
cd turing-ml
python -m pytest -q                                   # expect: 57 passed
python scripts/drive_live.py --map ring --no-render   # expect: laps 1 after ~14.4 s sim
```

---

## Commands

Run everything from `turing-ml/` with the venv active.

### Watch the driver live
| What | Command |
|---|---|
| Ring track, real-time (15 s laps) | `python scripts/drive_live.py --map ring --realtime` |
| A real circuit | `python scripts/drive_live.py --map Spielberg --realtime` |
| Pessimistic sensor | add `--preset harsh` |
| New random sensor every episode | add `--randomize` |
| Several laps / pan and zoom yourself | add `--laps 3` / `--free-camera` (scroll = zoom, drag = pan) |
| Faster or slower top speed | add `--v-max 2.5` |
| Headless, print the summary only | add `--no-render` |
| Your own map file | `python scripts/drive_live.py --map path/to/my_map.yaml --start X Y YAW` |

In the window, **red dots** are the D500 points the driver used, drawn at the car's *current* pose. At speed
they sit ahead of the real walls: that gap is the sensor's age. Close the window to stop and print a summary.

### Evaluate
| What | Command |
|---|---|
| Full table: 8 tracks × nominal/harsh × seeds 0,1,2 (~1 min) | `python scripts/eval_expert.py` |
| Selected tracks / presets / seeds | `python scripts/eval_expert.py --maps Monza,Sepang --presets harsh --seeds 0,1,2,3` |
| Try a parameter without editing the yaml | `python scripts/eval_expert.py --maps Spielberg --set bubble_width=0.3 --set v_max=2.5` |
| Table only, no crash diagnostics | add `--quiet` |
| Parameter sweep | `python scripts/sweep_expert.py Spielberg,Monza "preset=harsh;bubble_width=0.3" "preset=harsh;bubble_width=0.4"` |

Each cell is a lap time in sim seconds, or `crash@t`, `stuck@t`, `timeout`. For every failure it prints the
last 0.6 s: ranges at −90/−45/0/+45/+90°, steer, commanded and actual speed.

### Tests and checks
| What | Command |
|---|---|
| All unit tests (57) | `python -m pytest -q` |
| Sensor timing against the gym (scan age, seam) | `python scripts/d500_gym_smoke.py --preset nominal --speed 4` |

### Team rules for results
- Report results on **both `nominal` and `harsh`**. `clean` is for unit tests only, and the scripts refuse it.
- **Tune only on Spielberg, Monza, Silverstone.** Catalunya, Hockenheim, Budapest, Sepang and Zandvoort are
  held out: never tune on them (D-38).
- Change settings in `configs/*.yaml`, not in code.

---

## Using the pieces from Python

```python
from turing_ml.env import TuringEnv
from turing_ml.d500 import load_preset
from turing_ml.tracks import racetrack
from turing_ml.experts import disparity_extender, load_expert_config

map_path, start = racetrack("Spielberg")
env = TuringEnv(map_path, sensor_cfg=load_preset("configs/sensor.yaml", "nominal"),
                max_laps=1, max_time=400, seed=0)
cfg = load_expert_config("configs/expert.yaml")
obs = env.reset(start)
while not (obs["done"] or obs["truncated"]):
    action = disparity_extender(obs["grid"], obs["speed"], cfg, obs["yaw_rate"])   # (steer rad, speed m/s)
    obs = env.step(action)                                                         # held for 50 ms
print(obs["laps"], obs["lap_times"], obs["collision"])
env.close()
```

A policy may use `obs["grid"]`, `obs["speed"]` and `obs["yaw_rate"]`. `obs["pose"]` is ground truth, for
scoring only. Run only one `TuringEnv` per process (the gym shares the map between instances).

---

## Status and next steps

The full list is in [PROJECT_REPORT.md §8](turing-ml/docs/PROJECT_REPORT.md).

1. Get yaw rate into the driver on the car, and model IMU noise and delay in the sim.
2. Measure the real D500 (M-1…M-8) once the car arrives, then update `configs/sensor.yaml`.
3. Add the real track map as soon as it's available.
4. Add start poses and a frozen train/val/test map split for evaluation.
5. Write the ROS 2 policy node and test it in `f1tenth_gym_ros`, then on the Pi 5.
6. Imitation learning, using the disparity extender as the teacher.
