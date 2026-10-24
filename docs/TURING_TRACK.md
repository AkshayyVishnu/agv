# Turing Track: pure pursuit on our map, our car, and how to use it on yours

This page covers three things:

1. how to run and test the **pure pursuit** driver on **TuringTrack**, our 14 × 10 m track;
2. how the simulation was changed to match **our car**, the Rlaarlo Storm Breaker SBK10 (metal version);
3. **exactly what to do** to try the driver on your own track, car or simulator.

All commands run from `turing-ml/` with the venv active. Setup is in the [top-level README](../README.md#quick-start).
On Windows use PowerShell. On Linux, replace `$env:X="y";` with `X=y`.

---

## 1. What the driver is

| | Reactive driver (disparity extender) | **Pure pursuit on a racing line** (new) |
|---|---|---|
| Needs | LiDAR scan only | the car's **pose on the map** + a prebuilt racing line |
| On TuringTrack | **10/10 laps, ~58–60 s**, with `--set stop_dist=0.25 --set side_clear=0.25` (D-47) | **laps in ~18 s**, mean 3.2 m/s, top 5 m/s |
| Files | `turing_ml/experts.py`, `configs/expert.yaml` | `turing_ml/raceline.py`, `turing_ml/pursuit.py`, `configs/raceline.yaml` |

It works in two steps:

1. **Offline: `scripts/make_raceline.py`**
   - traces the track's centre line from the map image;
   - computes a **minimum-curvature racing line**: it swings wide into a bend, clips the inside and exits wide, so bends are as gentle as possible and can be taken faster;
   - builds a **speed profile** along it from a cornering limit (`a_lat`), an acceleration limit and a braking limit.
2. **Live: `turing_ml/pursuit.py`**
   - each step finds the car on the line and steers toward a point 0.4–1.0 m ahead (pure pursuit, rear axle);
   - commands the **slowest** planned speed in the next 0.25 s, so it brakes before a bend instead of in it.

In sim the pose is the simulator's ground truth. On the real car it must come from **localization**
(e.g. a particle filter on the LiDAR), which is late and noisy. `--pose-delay` and `--pose-noise` imitate
that, and the driver **predicts** the pose forward by its age using the current speed and yaw rate.

---

## 2. Testing pure pursuit on TuringTrack

The map, centre line and racing line are committed in `maps/TuringTrack/`:

| File | What |
|---|---|
| `turing_track_drawing.png` | the original CAD drawing (1400 × 1000 mm grid → 14 × 10 m) |
| `TuringTrack_map.png` / `.yaml` | occupancy map, 5 cm/pixel, origin (−7, −5) |
| `TuringTrack_centerline.csv` | centre line with track half-widths (58.0 m) |
| `TuringTrack_raceline.csv` | racing line: x, y, s, curvature, speed (55.5 m) |
| `TuringTrack_raceline.png` | picture: grey = centre line, coloured = racing line (blue slow → red fast), green = start |

The start pose is `0.875 -4.225 0`: on the start/finish line of the bottom straight, facing +x
(counter-clockwise; see the [assumptions](#5-assumptions-and-what-is-not-known)).

### 2.1 Watch it drive

```powershell
# 2 laps in a window at real speed (close the window to stop)
python scripts/drive_live.py --map maps/TuringTrack/TuringTrack_map.yaml --start 0.875 -4.225 0 --driver pursuit --laps 2 --max-time 60 --realtime

# same, no window, just the summary
python scripts/drive_live.py --map maps/TuringTrack/TuringTrack_map.yaml --start 0.875 -4.225 0 --driver pursuit --laps 2 --max-time 60 --no-render
```

Expected: `laps 2`, lap times about `[18.05, 17.77]`, mean speed 3.24 m/s, max 5.00. The green dots are
the racing line. For comparison, the reactive LiDAR-only driver (needs the two narrow-track settings, D-47):

```powershell
python scripts/drive_live.py --map maps/TuringTrack/TuringTrack_map.yaml --start 0.875 -4.225 0 --set stop_dist=0.25 --set side_clear=0.25 --laps 2 --max-time 150 --realtime
```

If you get `ModuleNotFoundError: gym`, the venv isn't active: run `..\.venv-main\Scripts\Activate.ps1` first (the prompt shows `(.venv-main)`, `python --version` says 3.10).

### 2.2 Make it realistic: late, noisy pose

```powershell
# pose 0.1 s late with 3 cm noise (a reasonable particle filter)
python scripts/drive_live.py --map maps/TuringTrack/TuringTrack_map.yaml --start 0.875 -4.225 0 --driver pursuit --laps 2 --max-time 60 --realtime --pose-delay 0.1 --pose-noise 0.03

# back off to 80 % of the planned speeds
... --speed-scale 0.8
```

### 2.3 Evaluate over many seeds

`eval_expert.py` finds tracks by folder name in `TURING_RACETRACKS`, so point it at `maps`:

```powershell
$env:TURING_RACETRACKS="maps"
python scripts/eval_expert.py --maps TuringTrack --driver pursuit --presets nominal --seeds 0 --max-time 60
python scripts/eval_expert.py --maps TuringTrack --driver pursuit --presets nominal --seeds 0,1,2,3,4,5,6,7,8,9 --max-time 60 --workers 3 --pose-delay 0.1 --pose-noise 0.03
```

Use `--workers 3` or fewer on a laptop: each worker holds a simulator.

### 2.4 Results (sbk10 car, 2026-10-24)

| Pose | Full speed | `--speed-scale 0.8` |
|---|---|---|
| perfect | 18.05 / 17.77 s | not run |
| 0.05 s late + 3 cm noise | 10/10, 18 s | 10/10, 22 s |
| 0.10 s late + 3 cm noise | 8/10 | 10/10, 22 s |
| 0.15 s late + 3 cm noise | 0/10 | 8/10 |
| 0.10 s late, **no prediction** | 0/10 (even at 0.7×) | |

**Takeaway for the car:** localization must deliver a pose **less than ~0.1 s old**, with its timestamp,
and the driver needs fresh speed (VESC/encoder) and yaw rate (IMU) for prediction.

### 2.5 Unit tests (fast, TuringTrack only)

```powershell
python -m pytest -q tests/test_raceline.py tests/test_experts.py
```

### 2.6 Rebuilding the racing line after changing settings

```powershell
python scripts/make_raceline.py maps/TuringTrack/TuringTrack_map.yaml --start 0.875 -4.225 0
```

It prints checks and overwrites the three csv/png files. Expected output:
- centre line 58.0 m, half-width 0.42–0.47 m;
- racing line ~55.5 m, wall clearance ≥ 0.325 m;
- `WARNING: tightest bend 0.57 m is below the car's turning radius 0.62 m`. This is expected: the
  hairpin is tighter than the car can turn at the assumed steering lock, so the car runs a little wide there and uses part of the margin;
- estimated lap 15.5 s (the real sim lap is ~18 s because steering and speed control lag).

Settings live in `configs/raceline.yaml`:

| Key | Value | Effect of raising it |
|---|---|---|
| `direction` | `ccw` | `cw` reverses the race direction |
| `margin` | 0.20 m | line stays further from walls: safer, slightly slower. 0.12 crashed |
| `a_lat` | 6.0 m/s² | faster in bends. 7.0 crashed. On real tyres probably lower: start ~4 |
| `a_accel` / `a_brake` | 5.0 / 5.0 | harder acceleration / later braking |
| `v_max` | 5.0 m/s | top speed on the straights |
| `lookahead_min` / `_max` | 0.4 / 1.0 m | smoother but cuts corners; too long goes through the hairpin wall |
| `speed_scale` | 1.0 | multiplies every speed (same as `--speed-scale`) |
| `pose_delay`, `pose_noise_*` | 0 | defaults for the localization imitation |

---

## 3. How the simulation was changed for our car

### 3.1 Vehicle preset `sbk10` (`configs/vehicle.yaml`, now the default)

Every script takes `--vehicle sbk10` (default) or `--vehicle f1tenth_default` (the stock F1TENTH car).

| Parameter | F1TENTH default | **SBK10** | Source |
|---|---|---|---|
| wheelbase (`lf + lr`) | 0.330 m | **0.275 m** | Rlaarlo spec (275 mm) |
| `lf` / `lr` | 0.159 / 0.171 | **0.1375 / 0.1375** | estimate (50/50): weigh the axles |
| width | 0.31 m | **0.25 m** | spec |
| length | 0.58 m | **0.40 m** | spec |
| mass `m` | 3.74 kg | **2.5 kg** | estimate, includes Pi 5 + D500: **weigh it** |
| yaw inertia `I` | 0.0471 | **0.016** | estimate (scaled by mass and size) |
| CoG height `h` | 0.074 m | **0.065 m** | estimate |
| `v_max` (gym) | 20 m/s | **12.5 m/s** | measured 28 mph by a reviewer |
| `lidar_dist` | 0.275 m | **0.05 m** | estimate: places the collision box on the real body (D-44). **Measure the D500 mount** |
| steering lock `s_max` | ±0.419 rad (24°) | unchanged | **not published: measure** |
| servo speed `sv_max` | 3.2 rad/s | unchanged | not published |
| grip `mu`, `C_Sf`, `C_Sr` | defaults | unchanged | depend on tyres and floor |
| `a_max` | 9.51 | unchanged | in practice limited by grip |

The full table with explanations of every parameter is in [`CAR_PARAMETERS.md`](CAR_PARAMETERS.md).
To try a different value, add a preset (the scripts' `--set` changes driver settings in `expert.yaml`, not the car).

### 3.2 Fixes made while bringing up our track

| ID | Change | Why it matters |
|---|---|---|
| D-42 | The reactive driver now follows a one-beam opening | before, it drove straight into the wall when the only gap into a bend was one beam wide (common on a 1 m track) |
| D-43 | Laps counted at a real start line | the gym counted false laps on our small track. **All TuringTrack lap times before this were wrong** |
| D-44 | `lidar_dist` per vehicle | the gym centres the collision box on the LiDAR; with the default the SBK10's simulated nose was 0.2 m too far forward |
| D-45 | `sbk10` preset; collision box rebuilt per env | the gym built the car's footprint once per process |
| D-46 | Racing line + pure pursuit | 3× faster than the reactive driver; uses the map |
| D-47 | Reactive driver: `stop_dist=0.25 side_clear=0.25`, keep `bubble_width` 0.35 | a 0.2 bubble made it hug the inside of the hairpin and crash; defaults stop it at the first corner |

Details and evidence: [`turing-ml/docs/DECISIONS.md`](../turing-ml/docs/DECISIONS.md).

---

## 4. Trying it in your own scenario: step by step

### Step 1: Get a map of your track

You need an occupancy-grid map (`.png` + `.yaml`, ROS `map_server` format): free space light, walls dark.

**From a CAD drawing or a plan image** (how TuringTrack was made):
```powershell
python scripts/drawing_to_map.py path/to/drawing.png maps/MyTrack/MyTrack_map --size 14 10 --origin -7 -5 --start-px 665 552 --direction ccw
```
- `--size W H`: the drawing's real size in metres (required);
- `--origin X Y`: map coordinate of the bottom-left corner (default: centred);
- `--start-px X Y`: pixel on the start line in the drawing, used to print the start pose;
- `--direction`: race direction.

The script keeps the track surface grey as free space, removes labels, adds a border wall and prints a
start pose. **Open the output png and check it**: walls must be closed, and no text or dimension lines
may be left on the track.

**From the real car** (recommended once it drives): drive slowly round with `slam_toolbox` and save the
map with `map_saver`. That map matches what the LiDAR sees, including walls the drawing doesn't show.

Put the files in `maps/<Name>/<Name>_map.png` and `<Name>_map.yaml`. The folder name and file prefix
must match: the scripts find files by that name.

### Step 2: Choose the start pose and direction

- Pick `X Y YAW`: a point on the start line, in map metres, with YAW pointing the way the race goes.
- Set `direction` in `configs/raceline.yaml` (`ccw` / `cw`). `--start` must face counter-clockwise; for `cw` the script flips it.
- Check: `python scripts/drive_live.py --map maps/MyTrack/MyTrack_map.yaml --start X Y YAW --v-max 1 --realtime` should show the car on the track facing along it.

### Step 3: Describe your car

Add a preset under `vehicle: presets:` in `configs/vehicle.yaml`. Copy `sbk10` and change what you know:

```yaml
mycar:
  lf: 0.16           # CoG to front axle (m): weigh the axles, lf = wheelbase × rear_weight / total
  lr: 0.16
  width: 0.28
  length: 0.45
  m: 3.0
  lidar_dist: 0.10   # LiDAR mount ahead of the CoG (m)
  s_max: 0.36        # steering lock (rad): measure the full-lock turning circle R, s_max = atan(wheelbase / R)
  # anything left out keeps the f1tenth_default value
```

The most important values to measure: **wheelbase, width, steering lock, LiDAR position**, then mass and top speed.

### Step 4: Build the racing line

```powershell
python scripts/make_raceline.py maps/MyTrack/MyTrack_map.yaml --start X Y YAW --vehicle mycar
```

Open `maps/MyTrack/MyTrack_raceline.png` and check:
- the grey centre line goes all the way round and follows the track;
- the coloured line stays inside the track, swings wide into bends and is blue (slow) in bends;
- the script printed no `CHECK FAILED`.

If it fails:
- **"centre line does not pass the start pose"**: the start pose is off the track or faces the wrong way.
- **"line comes too close to a wall"**: lower `margin` a little or fix the map.
- **Centre line takes a shortcut**: there is a gap in a wall on the map. Close it in the png.
- **WARNING about turning radius**: a bend is tighter than your car can turn. Fine if small; if it crashes there, raise `margin` or check `s_max`.

### Step 5: Test in sim, from easy to realistic

```powershell
# 1. perfect pose, slow
python scripts/drive_live.py --map maps/MyTrack/MyTrack_map.yaml --start X Y YAW --vehicle mycar --driver pursuit --speed-scale 0.5 --laps 2 --max-time 120 --realtime
# 2. full planned speed
... --speed-scale 1.0
# 3. realistic localization
... --pose-delay 0.1 --pose-noise 0.03
# 4. many seeds
$env:TURING_RACETRACKS="maps"
python scripts/eval_expert.py --maps MyTrack --vehicle mycar --driver pursuit --presets nominal --seeds 0,1,2,3,4,5,6,7,8,9 --max-time 120 --workers 3 --pose-delay 0.1 --pose-noise 0.03
```

`eval_expert.py` needs `maps/MyTrack/MyTrack_centerline.csv`; `make_raceline.py` writes it.

### Step 6: Tune when it crashes

The live window shows where; the eval prints the last 0.6 s before every crash.

| Symptom | Change |
|---|---|
| Runs wide in a bend and hits the outside wall | lower `a_lat` (or `--speed-scale 0.8`), raise `margin` |
| Clips the inside of a hairpin | lower `lookahead_min` / `lookahead_max` |
| Weaves on straights | raise `lookahead_min` |
| Fine with a perfect pose, crashes with `--pose-delay` | keep `predict: true`; lower the speed; get a faster localization |
| Crashes at the very start | the start pose must be on the track facing the race direction |

After changing `raceline:` settings, rerun Step 4. `pursuit:` settings take effect on the next run.

### Step 7: Another simulator or the real car

The driver is plain Python with no gym dependency:

```python
from turing_ml import raceline as rl
from turing_ml.pursuit import PurePursuit
from turing_ml.vehicle import load_vehicle

line = rl.load_raceline("maps/MyTrack/MyTrack_raceline.csv")
cfg = rl.load_raceline_config("configs/raceline.yaml", "pursuit")
car = load_vehicle("configs/vehicle.yaml", "mycar")
drv = PurePursuit(line, cfg, car)

# every control step (e.g. 20 Hz):
steer_rad, speed_mps = drv((x, y, yaw), speed, t, yaw_rate)
```

- `(x, y, yaw)`: the car's pose in the **same map frame** as the racing line. The driver treats it as
  the CoG and shifts to the rear axle itself. In ROS 2 that is the `map → base_link` transform from your
  localization. If `base_link` is the rear axle, add `lr` along the heading first.
- `speed` (m/s) and `yaw_rate` (rad/s): from odometry (VESC or encoder) and the IMU. Used for the speed
  preview and pose prediction.
- `t`: time in seconds. Set `pose_delay: 0` in the config on the real car; prediction then needs the
  real pose age, so wire your localization timestamp into the prediction step.
- Output: steering angle (rad, positive = left) and target speed. Send as an `AckermannDriveStamped`
  on `/drive`. The procedure for the ROS 2 simulator is in [`SIMULATION_GUIDE.md`](SIMULATION_GUIDE.md).

Before the real car: measure the steering lock and LiDAR mount, start at `speed_scale: 0.3–0.5` and
`a_lat` ≈ 3–4, and raise them a step at a time.

---

## 5. Assumptions and what is not known

- **Race direction is assumed counter-clockwise.** The drawing doesn't say. If it is clockwise, set
  `direction: cw` and rebuild the line.
- **SBK10 values not published:** weight, steering lock, servo speed, CoG position and height, inertia,
  tyre grip and braking. Gym defaults or estimates are used; see §3.1.
- **The stock 45 A ESC has no speed feedback.** The driver needs speed: add a VESC or wheel encoder.
- **The pose is perfect in sim by default.** No particle filter is integrated yet; `--pose-delay` and `--pose-noise` only imitate one.
- The gym's collision box is wheelbase-long (0.275 m), shorter than the real 0.40 m body.
- The F1-track results in the README used the `f1tenth_default` car before D-42..D-44 and have not been re-run.
