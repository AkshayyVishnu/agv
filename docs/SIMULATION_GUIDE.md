# Testing a model in different simulations

This guide covers how to check that a driver (the disparity extender or any new policy) still works when
something about the simulation changes. Go from the cheapest change to the most expensive:

| # | What changes | Effort | Section |
|---|---|---|---|
| 1 | **Track**: another circuit, your own map | none to small | [1](#1-a-different-track) |
| 2 | **Sensor conditions**: noise, latency, dropouts, randomization | edit a yaml | [2](#2-different-sensor-conditions) |
| 3 | **Vehicle physics**: grip, mass, speed limits | a few lines | [3](#3-different-vehicle-physics) |
| 4 | **Driver/model**: your own policy in the same harness | a function | [4](#4-plugging-in-your-own-model) |
| 5 | **Simulator**: another gym version or another simulator | an adapter | [5](#5-a-different-simulator) |
| 6 | **ROS 2 simulator** (`f1tenth_gym_ros`, the team's Docker setup) | a ROS node | [6](#6-ros-2-simulator-f1tenth_gym_ros) |

All commands run from `turing-ml/` with the venv active (see the main [README](../README.md#quick-start)).

**Whatever you change, the bar for "it works" is the same** (see the [checklist](#7-checklist-before-calling-a-model-working)):
it laps on **nominal and harsh**, over **several seeds**, on tracks it was **not tuned on**.

---

## 1. A different track

### 1a. Another circuit from `f1tenth_racetracks`
Any folder name in `f1tenth_racetracks/` that has a `<Name>_centerline.csv` works directly (22 of the 25 do):

```bash
python -c "from turing_ml.tracks import list_racetracks; print(list_racetracks())"   # what's available
python scripts/drive_live.py --map Austin --realtime
python scripts/eval_expert.py --maps Austin,Shanghai,YasMarina --presets nominal,harsh --seeds 0,1,2
```

The start pose is the first centre-line point, facing along the line. Tracks are ~2.2 m wide, scaled from F1.

### 1b. Your own map, quick look (`drive_live.py` only)
A map is a ROS-style pair: an image (white = free, black = wall) plus a yaml:

```yaml
image: my_track.png        # .png, .pgm or .jpg next to the yaml
resolution: 0.05           # metres per pixel
origin: [-10.0, -5.0, 0.0] # world position of the image's bottom-left pixel
negate: 0
occupied_thresh: 0.45
free_thresh: 0.196
```

Run it with a start pose `x y yaw` (metres, radians, map frame) on the track, facing the driving direction:

```bash
python scripts/drive_live.py --map path/to/my_track.yaml --start 0.0 0.0 0.0
```

Maps from SLAM (`slam_toolbox`, `.pgm` + `.yaml`) work as-is. **Walls must be closed:** if the car can drive
off the image, the gym reports no collision (D-16).

### 1c. Your own map, full evaluation (`eval_expert.py`)
`eval_expert.py --maps` only accepts `f1tenth_racetracks`-style names (and `ring`). To evaluate a new track,
put it in that layout:

```
my_tracks/
└── MyTrack/
    ├── MyTrack_map.png
    ├── MyTrack_map.yaml           # image: MyTrack_map.png
    └── MyTrack_centerline.csv     # "# x_m, y_m, w_tr_right_m, w_tr_left_m", one point per row, in driving order
```

The centre line sets the start pose (first row, heading toward the point ≥ 1 m further along). Two rows are
enough if you only need a start. Then point the loader at the folder:

```bash
# Linux/macOS
TURING_RACETRACKS=/path/to/my_tracks python scripts/eval_expert.py --maps MyTrack
# PowerShell
$env:TURING_RACETRACKS="C:\path\to\my_tracks"; python scripts/eval_expert.py --maps MyTrack
```

`TURING_RACETRACKS` replaces the default folder, so either copy your track into `f1tenth_racetracks/` or keep
a separate folder and switch between them.

**When the real competition map arrives** (~10 days before the event): add it this way, evaluate on it
immediately, and keep the held-out circuits for honest reporting.

---

## 2. Different sensor conditions

The D500 model is configured entirely in `configs/sensor.yaml`. Every value is explained in `docs/DECISIONS.md`.

| What | How |
|---|---|
| Realistic sensor (default) | `--preset nominal` |
| Pessimistic but plausible sensor | `--preset harsh` |
| New random sensor every episode (rate, latency, noise, spin direction, dropouts…) | `drive_live.py --randomize`, ranges under `randomization:` in the yaml |
| Your own condition, e.g. "very laggy" | add a preset under `d500.presets` (any key you leave out keeps its default), then `--preset laggy` |
| Real measured values (after the car arrives) | edit `nominal` with the measurements M-1…M-8, rerun everything |

Example preset:

```yaml
    laggy:
      latency_s: 0.12
      latency_jitter_s: 0.03
      rate_hz: 8.5
      dropout_prob: 0.05
```

```bash
python scripts/eval_expert.py --maps Spielberg,Monza --presets nominal,harsh,laggy --seeds 0,1,2
```

Rules: `clean` (perfect sensor) is refused by the scripts. A result that only holds on `nominal` doesn't count.

---

## 3. Different vehicle physics

`TuringEnv` passes `params` straight to the gym, so grip, mass, steering limits and so on can be changed.
f1tenth_gym main **replaces** the whole parameter dict, so start from the defaults and change single values:

```python
from turing_ml.env import TuringEnv
from turing_ml.d500 import load_preset
from turing_ml.tracks import racetrack

params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
          'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2,
          'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
params['mu'] *= 0.8          # 20 % less grip

path, start = racetrack("Monza")
env = TuringEnv(path, sensor_cfg=load_preset("configs/sensor.yaml", "harsh"),
                params=params, max_laps=1, max_time=400, seed=0)
```

(The defaults above are f1tenth_gym main's; check `f1tenth_gym/gym/f110_gym/envs/f110_env.py` if it's updated.)
Actuation delay and odometry/IMU noise are **not** modelled yet (report §7.2, E-3).

---

## 4. Plugging in your own model

A driver is any function `(grid, speed, yaw_rate) -> (steer_rad, speed_mps)`:

- `grid`: 271 ranges in metres on a 1° grid from −135° to +135° (`turing_ml.sensor.GRID`), capped at 4 m;
- `speed`: forward speed (m/s); `yaw_rate`: turn rate (rad/s);
- called at 20 Hz; the action is held for 50 ms.

Don't feed it `obs["pose"]`: the car has no ground-truth pose (a map-based driver must localize first).

Minimal evaluation loop for a new model (`my_eval.py` in `turing-ml/`):

```python
import sys
from turing_ml.env import TuringEnv
from turing_ml.d500 import load_preset
from turing_ml.tracks import racetrack

def my_policy(grid, speed, yaw_rate):
    ...                                   # your model here
    return steer, target_speed

for track in sys.argv[1].split(","):
    for preset in ("nominal", "harsh"):
        for seed in (0, 1, 2):
            path, start = racetrack(track)
            env = TuringEnv(path, sensor_cfg=load_preset("configs/sensor.yaml", preset),
                            max_laps=1, max_time=400, seed=seed)
            obs = env.reset(start, seed=seed)
            while not (obs["done"] or obs["truncated"]):
                obs = env.step(my_policy(obs["grid"], obs["speed"], obs["yaw_rate"]))
            env.close()
            result = f"{obs['lap_times'][0]:.0f}s" if obs["laps"] else ("crash" if obs["collision"] else "timeout")
            print(track, preset, seed, result)
```

```bash
python my_eval.py Spielberg,Monza,Silverstone          # tune here
python my_eval.py Catalunya,Hockenheim,Budapest,Sepang,Zandvoort   # report here, never tune
```

Create **one `TuringEnv` at a time** in a process (the gym keeps the map at class level; an older env raises
`StaleEnvError`). For parallel runs use one process per run, as `eval_expert.py` does.

To compare against the baseline, run `python scripts/eval_expert.py` on the same tracks, presets and seeds.
A new driver should beat the disparity extender before it replaces it.

---

## 5. A different simulator

`turing_ml/env.py` is the **only** file that talks to the gym (D-18). Everything else (sensor model, `to_grid`,
drivers, scripts) only sees the observation dict. To run the same drivers in another simulator (gym
`dev-humble`/gymnasium API, a custom physics model, Isaac, Gazebo…), write an adapter class with the same
interface:

| Method / key | Contract |
|---|---|
| `reset(pose, seed=None) -> obs` | place the car at `(x, y, yaw)`, reset the sensor, step until the first scan is usable |
| `step((steer, speed)) -> obs` | hold the action for one 50 ms control period |
| `close()` | release the simulator |
| `obs["grid"]` | `to_grid(scan.ranges, scan.angle_min, scan.angle_increment)` of the newest **D500-model** scan |
| `obs["speed"]`, `obs["yaw_rate"]` | forward speed (m/s), yaw rate (rad/s) |
| `obs["pose"]`, `obs["collision"]`, `obs["laps"]`, `obs["lap_times"]` | ground truth for scoring |
| `obs["t"]`, `obs["done"]`, `obs["truncated"]` | episode time and end flags |

The sensing pattern to copy from `TuringEnv`:

```python
from turing_ml.d500 import D500Model, load_preset
from turing_ml.sensor import to_grid

sensor = D500Model(load_preset("configs/sensor.yaml", "nominal"), seed=0)
sensor.reset(t0=0.0)
# every physics step (as fine as possible, <= 10 ms): push the simulator's PERFECT scan
sensor.push(t, perfect_ranges, angle_min, angle_increment)
# every control step (50 ms): read what the car would have by now
scan = sensor.read(t)                     # None until the first revolution + latency
if scan is not None:
    grid = to_grid(scan.ranges, scan.angle_min, scan.angle_increment)
```

Things to keep the same, or the results aren't comparable:
- feed the model the **perfect** scan (no extra noise from the simulator) at **≥ 100 Hz**, or the motion
  skew is wrong;
- LiDAR **0.275 m ahead of the rear axle** (same as `f1tenth_gym_ros`);
- control at **20 Hz**; collisions judged from the simulator's physics, not from the scan;
- then check the adapter: `python scripts/d500_gym_smoke.py` is the reference for scan age and seam size
  (~155 ms and ~0.2 m at 2 m/s on nominal).

`f1tenth_gym` **dev-humble** (gymnasium API, Python 3.13 and numpy 2 work) is the obvious next target. The
sensor tests already pass there (`.venv-gym`), but `TuringEnv` targets the main branch.

---

## 6. ROS 2 simulator (`f1tenth_gym_ros`)

This is the closest thing to the car: the same gym physics behind ROS 2 topics, inside the team's Docker
container. **Status: no `turing_ml` ROS node exists yet**, so this section is the procedure to follow, not a
tested recipe.

### 6.1 Bring up the simulator
Follow [`turing-track-ros2/docs/turing_gym_setup_guide.md`](../turing-track-ros2/docs/turing_gym_setup_guide.md)
(Linux host, Docker, ROS 2 Humble). In short:

```bash
xhost +local:docker
sudo docker start ros-humble-gui && sudo docker exec -it ros-humble-gui bash
cd ~/sim_ws && source install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py      # RViz opens with the map and the car
ros2 topic hz /scan                                   # second terminal: check the topics
```

To use another map, set `map_path` (full path, no extension) and `map_img_ext` in
`f1tenth_gym_ros/config/sim.yaml`, then `colcon build` again (the config is copied at build time).

### 6.2 Run a model as a ROS 2 node
1. Copy or install `turing-ml` into the container (`pip install -e turing-ml`), so the node can import
   `turing_ml.sensor.to_grid`, the driver and `configs/*.yaml`.
2. Write a node (an `ament_python` package next to `turing_drive`) that:
   - subscribes to `/scan` with **best-effort QoS** (the real D500 driver publishes best-effort; a reliable
     subscriber gets nothing on the car, R-3);
   - converts each scan with `to_grid(msg.ranges, msg.angle_min, msg.angle_increment)`;
   - reads speed and yaw rate from `/ego_racecar/odom` in sim (`/odom` on the car; yaw rate needs a topic, R-7);
   - runs the driver on a **20 Hz timer** and publishes `ackermann_msgs/AckermannDriveStamped`:
     to **`/drive`** in the bare sim, to **`/drive_raw`** in the team setup (the gate and watchdog forward it to `/drive`).
3. Make the sim's `/scan` look like the D500: run `turing_drive`'s `d500_relay` between the bridge and the node
   (it is currently a simpler, unseeded model, R-1/R-2) or, better, change it to use `turing_ml.d500` and
   `configs/sensor.yaml` so the ROS sim and the Python harness test the same sensor.
4. Launch: simulator (6.1), relay, your node. Watch it in RViz and log lap times. The result should match
   `eval_expert.py` on the same map and preset. A large difference means the two setups have drifted apart.

Topic names, rates and QoS: [`turing-track-ros2/docs/turing_topic_contract.md`](../turing-track-ros2/docs/turing_topic_contract.md).

---

## 7. Checklist before calling a model "working"

- [ ] `python -m pytest -q` passes (57 tests, plus your own).
- [ ] Laps on all **tuning** tracks (Spielberg, Monza, Silverstone) on **nominal and harsh**, seeds 0, 1, 2.
- [ ] Then, once and without further tuning, laps on the **held-out** tracks (Catalunya, Hockenheim, Budapest,
      Sepang, Zandvoort). If you tune after seeing held-out results, they are no longer held out.
- [ ] Survives `--randomize` (several seeds).
- [ ] Compared with the disparity extender (`eval_expert.py`) on the same tracks, presets and seeds.
- [ ] Settings in `configs/*.yaml`, not hard-coded; the reasoning added to `docs/DECISIONS.md`.
- [ ] Before the car: same behaviour in the ROS 2 sim (section 6), and CPU time measured on a Pi 5.
