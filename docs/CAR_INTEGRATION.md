# Putting the driver on the car: interfaces, assumptions, open items

Status 2026-10-30. **Nothing in this file has run on the real car yet.** Everything here was tested in
simulation only (`f1tenth_gym`, TuringTrack 1.0 m wide). The ROS 2 node in §4 is a sketch to build from,
not tested code.

Contents:
1. [The plan: map first, fallback always](#1-the-plan-map-first-fallback-always)
2. [How the pieces connect](#2-how-the-pieces-connect)
3. [Interface spec: what our node receives and sends](#3-interface-spec-what-our-node-receives-and-sends)
4. [Putting the driver on the Pi](#4-putting-the-driver-on-the-pi)
5. [Assumed, unfilled, and what each one breaks](#5-assumed-unfilled-and-what-each-one-breaks)
6. [Questions for the ROS bridge developer](#6-questions-for-the-ros-bridge-developer)
7. [When the map arrives](#7-when-the-map-arrives)

---

## 1. The plan: map first, fallback always

There are two drivers. Both are plain Python with no simulator dependency.

| | Reactive (disparity extender) | Pure pursuit on a racing line |
|---|---|---|
| Needs | `/scan` + speed + yaw rate | the above + **the car's pose on a map** + a racing line built from that map |
| Sim result (1.0 m track) | 10/10 laps, ~59 s | ~18 s per lap |
| Fails when | settings don't suit the track width | map wrong, pose late (> ~0.1 s) or wrong |
| Role | **day 1 on the car, and the fallback forever** | race driver once localization works |

**Plan A: approximate map → pure pursuit.** An approximate map (drawing with dimensions) is turned
into a sim map (`scripts/drawing_to_map.py`). We build the racing line and test both drivers in sim at
the real width (1.25–1.5 m), so the settings are ready before the track day.
**Do not race on an approximate map.** If it is off by more than the line's safety margin (0.20 m),
pure pursuit drives into the real wall, and localization against a wrong map gives a wrong or jumping pose.

**Plan B (fallback, do it anyway): map the real track on the day.**
1. Drive 2–3 laps by hand at walking pace with `slam_toolbox` (online async) running on the Pi.
2. **At the same time record a bag:** `ros2 bag record /scan /odom /imu /tf /tf_static`. With it, maps
   and both drivers can be rebuilt and tested at home on real sensor data. This is the most valuable
   thing to bring back.
3. Save: `ros2 run nav2_map_server map_saver_cli -f MyTrack_map` gives `MyTrack_map.pgm/.png` + `.yaml`.
4. On the laptop: `maps/MyTrack/`, then `python scripts/make_raceline.py maps/MyTrack/MyTrack_map.yaml --start X Y YAW`,
   then check `MyTrack_raceline.png` and test in sim ([TURING_TRACK.md §4](TURING_TRACK.md#4-trying-it-in-your-own-scenario-step-by-step)).
5. Localize on that map while driving: F1TENTH `particle_filter`, AMCL, or `slam_toolbox` localization mode.
6. Pure pursuit at `speed_scale` 0.3, then step up.

Budget 30–60 min at the track for steps 1–5. Bring the laptop with the repo and `.venv-main` working.

**Order on the car:** bench (wheels off the ground) → reactive at ≤ 1 m/s → reactive faster → mapping
→ pure pursuit slow → pure pursuit faster.

---

## 2. How the pieces connect

```
D500 LiDAR ─── ldlidar_stl_ros2 ───► /scan ─────────────────────┬────────────► slam_toolbox (mapping day)
IMU ────────── IMU driver ─────────► /imu ──┐                    │                   │
Motor / wheel RPM ──────────────────────────┼─► odometry node ─► /odom + TF odom→base_link
Steering command (fed back) ────────────────┘                    │
                                                                 ▼
                                    localization (particle filter) ─► TF map→odom   (pose + timestamp)
                                                                 ▼
                    OUR NODE  turing_driver:  reactive (scan only)  |  pure pursuit (needs pose)
                                                                 ▼
                     /drive_raw ─► command gate ─► /drive_gated ─► watchdog ─► /drive ─► motor + servo driver
                       (ours)       (clamps speed,                 (zero speed if
                                    steer, steer rate)              nothing for 200 ms)

Static TF:  base_link → laser   (the measured LiDAR mount position and orientation)
```

This follows the team contract, [`turing-track-ros2/docs/turing_topic_contract.md`](../turing-track-ros2/docs/turing_topic_contract.md)
(Draft v0.1). Our node publishes **only** `/drive_raw`, never `/drive`.

Who owns what:

| Piece | Owner |
|---|---|
| LiDAR driver, IMU driver, odometry node, motor/servo driver, TF tree | ROS bridge (your friend) |
| Command gate, watchdog, mode manager | team infra (contract) |
| `turing_driver` node, `turing_ml` package, configs, maps, racing line | us |
| Mapping (`slam_toolbox`), localization | shared: decide who (open item O-14) |

---

## 3. Interface spec: what our node receives and sends

ROS 2 conventions (REP-103): metres, seconds, radians; x forward, y left, z up; angles and yaw rate
**counter-clockwise positive**. Every message carries `header.stamp` set to **when the data was measured**,
not when it was published.

### 3.1 Inputs (what we receive)

| # | Topic / source | Type | Rate | Fields we use | Must be |
|---|---|---|---|---|---|
| I-1 | `/scan` | `sensor_msgs/LaserScan` | 10 Hz | `ranges`, `angle_min`, `angle_increment`, `header.stamp`, `time_increment`, `scan_time`, `range_min`, `range_max` | ranges in m; **no return = NaN or +inf, not 0**; angle 0 = straight ahead, CCW positive (0…2π is fine, we wrap it); `frame_id: laser` |
| I-2 | `/odom` | `nav_msgs/Odometry` | 20–50 Hz | `twist.twist.linear.x`, `twist.twist.angular.z` | **signed** speed m/s (negative in reverse), yaw **rate** rad/s; `child_frame_id: base_link` |
| I-3 | `/imu` | `sensor_msgs/Imu` | ~50 Hz | `angular_velocity.z` (only if `/odom` has no yaw rate) | rad/s (not deg/s), z up, bias removed |
| I-4 | TF `map → base_link` (pure pursuit only) | tf2 | ≥ 20 Hz | x, y, yaw + the transform's stamp | pose of the **rear axle** in the map frame; stamp = time of the scan it was computed from |
| I-5 | `/mode` | `std_msgs/String` | on change | `STANDBY / ARMED / RUN / STOPPED` | publish zero speed unless `RUN` |

What each driver needs:
- **Reactive:** I-1 + I-2 (speed for the braking cap, yaw rate for scan-age compensation).
- **Pure pursuit:** I-2 + I-4. I-1 is only used by localization. Speed and yaw rate are also used to
  roll a late pose forward by its age.

### 3.2 Output (what we send)

| Topic | Type | Rate | Fields |
|---|---|---|---|
| `/drive_raw` | `ackermann_msgs/AckermannDriveStamped` | 20 Hz (reactive: one per scan, 10 Hz, is also acceptable; contract says ≥ 20 Hz) | `drive.steering_angle` rad, **positive = left**; `drive.speed` m/s target (≥ 0); `header.stamp` = now; others 0 |

**`speed` is a target speed, not throttle.** Something between `/drive` and the ESC has to turn m/s
into throttle using the RPM feedback (a speed controller). See O-6.

### 3.3 Frames

| Frame | Where | Note |
|---|---|---|
| `map` | fixed, from the saved map | origin = map yaml `origin` |
| `odom` | dead-reckoning frame | drifts; localization corrects it via `map→odom` |
| `base_link` | **rear axle centre, on the ground** (F1TENTH convention) | the contract text says "center of the car (rear axle)": these are two points. **Agree on rear axle** (O-1) |
| `laser` | D500 optical centre | static TF from `base_link`, **measured** (O-3) |

---

## 4. Putting the driver on the Pi

### 4.1 What goes onto the Pi

| What | From the repo | Size | Notes |
|---|---|---|---|
| `turing_ml` package | `turing-ml/turing_ml/` | small | car-side modules: `sensor.py`, `experts.py`, `pursuit.py`, `vehicle.py`, `raceline.py` (for `load_raceline`). **No gym needed** (`env.py` imports gym only when used) |
| Configs | `turing-ml/configs/expert.yaml`, `raceline.yaml`, `vehicle.yaml` | small | edit on the laptop, copy over |
| Racing line (pure pursuit only) | `turing-ml/maps/<Track>/<Track>_raceline.csv` | ~50 KB | built **on the laptop** with `make_raceline.py`. The Pi only reads it |
| Map (for localization) | `<Track>_map.png/.pgm + .yaml` | small | the one saved on the mapping day |
| Our ROS 2 node | `turing_driver` package (**to be written**, sketch in §4.3) | small | lives next to `turing_drive` in `turing-track-ros2/src/` |

Python dependencies on the Pi: `numpy`, `pyyaml`, `scipy` (only because `raceline.py` imports it;
`apt install python3-numpy python3-yaml python3-scipy` or pip). Install: `pip install -e turing-ml`
inside the ROS 2 environment/container.

What stays on the laptop: the simulator, `make_raceline.py`, `drawing_to_map.py`, eval/sweep scripts, tests.

### 4.2 Data flow inside our node

**Reactive, run on every `/scan`:**
```
LaserScan ─► to_grid(ranges, angle_min, angle_increment, range_min)   # 271 ranges, 1°, −135…+135°, capped 4 m
          ─► disparity_extender(grid, speed, cfg, yaw_rate)           # speed, yaw_rate from /odom
          ─► (steer rad, speed m/s) ─► /drive_raw
```

**Pure pursuit, 20 Hz timer:**
```
TF map→base_link (rear axle) + its stamp ─► convert to CoG: x += lr·cos(yaw), y += lr·sin(yaw)   # see O-1
                                         ─► age = now − stamp
                                         ─► PurePursuit(pose, speed, t, yaw_rate) ─► (steer, speed) ─► /drive_raw
```

The current `PurePursuit` code assumes the pose is the centre of gravity, because the gym's pose is.
It only knows the pose age when it imitates delay itself (`pose_delay`). On the car:
- set `pose_delay: 0`, `pose_noise_*: 0`;
- **to-do (code change, small):** let the node pass the real pose age so prediction uses it. Without
  prediction, a 0.1 s old pose crashed 10/10 in sim.

### 4.3 Node sketch (untested: shape only)

```python
# turing_driver/driver_node.py  (ROS 2 Humble/Jazzy, rclpy)
import dataclasses, rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from ackermann_msgs.msg import AckermannDriveStamped
from turing_ml.sensor import to_grid
from turing_ml.experts import disparity_extender, load_expert_config

class Driver(Node):
    def __init__(self):
        super().__init__("turing_driver")
        self.declare_parameter("expert_config", "/opt/turing/configs/expert.yaml")
        self.cfg = load_expert_config(self.get_parameter("expert_config").value)
        # narrow-track settings (D-47): re-check at the real track width
        self.cfg = dataclasses.replace(self.cfg, stop_dist=0.25, side_clear=0.25)
        self.speed, self.yaw_rate, self.mode = 0.0, 0.0, "STANDBY"
        self.pub = self.create_publisher(AckermannDriveStamped, "/drive_raw", 1)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
        self.create_subscription(String, "/mode", lambda m: setattr(self, "mode", m.data), 10)

    def on_odom(self, m):
        self.speed = m.twist.twist.linear.x
        self.yaw_rate = m.twist.twist.angular.z

    def on_scan(self, s):
        grid = to_grid(s.ranges, s.angle_min, s.angle_increment, s.range_min)
        steer, v = disparity_extender(grid, self.speed, self.cfg, self.yaw_rate)
        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.drive.steering_angle = float(steer)
        out.drive.speed = float(v) if self.mode == "RUN" else 0.0
        self.pub.publish(out)

def main():
    rclpy.init(); rclpy.spin(Driver())
```

The pure-pursuit variant adds a `tf2_ros.Buffer`, a 20 Hz timer, `lookup_transform("map", "base_link", Time())`,
and `PurePursuit(load_raceline(csv), pursuit_cfg, load_vehicle(...))`.

### 4.4 Bring-up checks, in order

| Step | Check | Pass when |
|---|---|---|
| 1 | `ros2 topic hz /scan /odom /imu` | 10 / 20–50 / ~50 Hz, steady |
| 2 | Put a box straight ahead, at +90° (left), at −90° (right) | `/scan` shows it at angle ≈ 0, +π/2, −π/2 (or 3π/2). **Mirrored = scan direction wrong** |
| 3 | Push the car forward by hand, then backward | `/odom` speed positive then negative, magnitude ≈ real speed |
| 4 | Turn the car left by hand | yaw rate **positive** |
| 5 | Wheels up: publish `steering_angle +0.2` | wheels turn **left**; measure the real angle |
| 6 | Wheels up: `speed 1.0` | wheels spin forward; RPM-derived speed ≈ 1.0 |
| 7 | Stop publishing | wheels stop within 200 ms (watchdog) |
| 8 | Driver node, wheels up, walk an object around the car | steering follows the openings |
| 9 | On the floor, speed cap 1.0 m/s, reactive | completes laps |

---

## 5. Assumed, unfilled, and what each one breaks

**Status:** `measured` / `spec` (manufacturer) / `assumed` (our guess, used in sim) / `UNFILLED` (needed, nobody has it).

### 5.1 Car

| # | Item | Current value | Status | If wrong |
|---|---|---|---|---|
| C-1 | Steering lock (max angle) | ±0.419 rad (24°), driver caps 0.4 | assumed (gym default) | **Biggest one.** Sets the turning circle (0.62 m). Smaller lock: hairpins fail, the racing line is too tight. Larger: margin unused. Measure a full-lock circle each way: `s_max = atan(0.275 / R)` |
| C-2 | Steering left/right symmetry, centre trim | symmetric, 0 | assumed | car pulls to one side; pure pursuit fights it constantly |
| C-3 | Servo speed | 3.2 rad/s | assumed | slower servo = turns in late = runs wide in quick bends |
| C-4 | LiDAR mount position (ahead of CoG) | 0.05 m | assumed (D-44) | sim collision box in the wrong place; on the car, the `base_link→laser` TF wrong means localization and the map are offset |
| C-5 | LiDAR mount height | – | **UNFILLED** | walls lower than the scan plane are invisible |
| C-6 | Mass | 2.5 kg | assumed | braking distance and grip in sim; small effect at our speeds |
| C-7 | Weight split front/rear (`lf`/`lr`) | 50/50 | assumed | pursuit's rear-axle shift; mild handling change |
| C-8 | Inertia, CoG height | 0.016, 0.065 m | assumed | sim handling detail only |
| C-9 | Tyre grip on the race floor | gym default (μ ≈ 1.05) | **UNFILLED** | **Second biggest.** `a_lat` 6.0 m/s² assumes good grip. On a dusty or smooth floor the car slides wide in bends at the planned speeds. Start `a_lat` 3–4 |
| C-10 | Braking deceleration | 5 m/s² planned | assumed | late braking: overshoots corner entries |
| C-11 | Top speed | 12.5 m/s gym, line capped 5 m/s, gate 1.5 m/s | measured (28 mph) | none at current caps |
| C-12 | Wheel diameter under load | 83 mm | spec | speed from RPM off by the same %: braking cap and pose prediction off |
| C-13 | Gear ratio motor → wheel | – | **UNFILLED** (needed if RPM is motor RPM) | speed wrong by the ratio: everything speed-based fails |

### 5.2 Sensors and actuation

| # | Item | Current value | Status | If wrong |
|---|---|---|---|---|
| S-1 | Scan angle direction / zero | CCW, 0 = forward | assumed | **mirrored scan: the car steers into walls immediately**. Bring-up step 2 catches it |
| S-2 | Scan timestamp meaning | start of rotation | assumed | scan-age compensation off by up to 0.1 s: the car cuts or runs wide in bends |
| S-3 | Scan latency | 50–190 ms modelled | assumed (sensor model) | the reactive driver's `yaw_comp_dt` 0.15 s is tuned to it |
| S-4 | "No return" encoding | NaN | assumed | if the driver gives 0, open space looks like a wall at 0 m: the car stops. `to_grid` treats `< range_min` as unknown, so 0 is safe only if `range_min > 0` |
| S-5 | Dark/black wall returns | dropouts modelled | assumed | black walls can vanish: the reactive driver sees an opening that isn't there |
| S-6 | Speed source (RPM) | – | **UNFILLED** | the stock 45 A ESC has no speed output. No speed means no braking cap, no prediction, no speed control |
| S-7 | Speed sign (reverse) | – | **UNFILLED** | unsigned speed breaks odometry when reversing |
| S-8 | Speed resolution at low speed | – | **UNFILLED** | reads 0 below some speed: the stuck detection and slow starts misbehave |
| S-9 | Yaw: rate or heading? | rate | assumed | integrated heading passed as rate: compensation goes wildly wrong |
| S-10 | IMU units and axes | rad/s, z up | assumed | deg/s = 57× too big; z down = sign flipped. Bring-up step 4 |
| S-11 | m/s → throttle (speed controller) | – | **UNFILLED** | without it `drive.speed` has no meaning: the car's speed depends on the battery voltage |
| S-12 | Steering PWM ↔ angle calibration | – | **UNFILLED** | commanded 0.2 rad gives some other angle: pursuit overshoots or undershoots every bend |
| S-13 | ESC reverse/brake behaviour | – | **UNFILLED** | many hobby ESCs need brake → neutral → reverse; commands to slow down may brake, coast, or do nothing |

### 5.3 System and track

| # | Item | Current value | Status | If wrong |
|---|---|---|---|---|
| X-1 | `base_link` = rear axle | rear axle | assumed (contract ambiguous) | pursuit shifts the pose twice or not at all: 0.14 m systematic error, cuts corners |
| X-2 | Localization pose age | ≤ 0.1 s | assumed (sim requirement) | 0.15 s: 0/10 at full speed in sim. Measure it on the Pi |
| X-3 | ROS 2 distro on the Pi 5 | Humble (contract) | **UNFILLED** | Humble needs Ubuntu 22.04, which doesn't officially support the Pi 5: Docker, or Jazzy on 24.04 |
| X-4 | One clock for all nodes | yes | assumed | pose age and scan age computed wrong |
| X-5 | Scan → command latency | ≤ 100 ms target | **UNFILLED** | more latency = slower safe speed |
| X-6 | Pi 5 CPU headroom (SLAM + PF + driver) | – | **UNFILLED** | dropped scans, late poses |
| X-7 | Track width | 1.0 m in sim, real 1.25–1.5 m | real value from you | all driver settings were tuned at 1.0 m; re-test at the real width |
| X-8 | Race direction | ccw | assumed | the racing line is built for the other direction: rebuild with `direction: cw` |
| X-9 | Wall height and material | 300 mm (drawing) | assumed | low or black walls → LiDAR misses them |
| X-10 | Exact track geometry | approximate map coming | – | see §1: don't race on an approximate map |

---

## 6. Questions for the ROS bridge developer

Copy this list and fill in the answers.

**General**
- [ ] O-1: Is `base_link` the rear axle centre? (The contract says "center of the car (rear axle)".)
- [ ] O-2: ROS 2 Humble in Docker, or Jazzy on Ubuntu 24.04? Python version?
- [ ] O-14: Who runs mapping and localization: you, or us?
- [ ] Are all nodes on the Pi using one clock? Is `use_sim_time` false on the car?

**LiDAR (D500)**
- [ ] O-3: Mount position x, y, z from `base_link`, and its yaw. Is it upside down? Is the `base_link→laser` TF published?
- [ ] Driver: `ldlidar_stl_ros2`? Which settings (`laser_scan_dir`, `enable_angle_crop_func`, `range_threshold`)?
- [ ] Angles: direction and where 0 is. `angle_min`, `angle_max`, `angle_increment`?
- [ ] `range_min`, `range_max`. What does a missing return look like: 0, NaN or inf?
- [ ] Does `header.stamp` mark the first beam or the last? Are `time_increment` and `scan_time` filled in?
- [ ] Measured delay from the rotation to the message arriving.

**IMU**
- [ ] Model, rate, mount orientation (which IMU axis is car-forward, is z up?).
- [ ] Units of `angular_velocity` (rad/s?). Is the gyro bias removed at startup? Noise level?
- [ ] Is the "yaw" you get a **rate** or an integrated heading?

**Speed / RPM**
- [ ] O-6: Where does RPM come from: motor sensor (hall), ESC telemetry, or a wheel encoder?
- [ ] Motor or wheel RPM? Gear ratio? Wheel diameter used?
- [ ] Signed (knows reverse)? Rate? Lowest speed it can read? Delay?
- [ ] Is `/odom` published with `twist.linear.x` and `twist.angular.z` filled in? Is pose from dead reckoning?

**Actuation**
- [ ] How does the Pi drive the ESC and servo: GPIO PWM, a PCA9685 board, or a VESC?
- [ ] Is there a speed controller turning `drive.speed` (m/s) into throttle with the RPM feedback? Who tunes it?
- [ ] Steering calibration: PWM at centre, full left, full right, and the matching angles in rad.
- [ ] Servo speed. Deadband?
- [ ] ESC: braking and reverse behaviour (double tap?), max reverse speed.
- [ ] Watchdog: what happens when `/drive` stops (zero throttle within 200 ms)? Where is the hardware E-stop?

**Performance**
- [ ] Scan → `/drive` latency measured end to end.
- [ ] CPU load with SLAM or particle filter + driver running.

---

## 7. When the map arrives

Send:
1. the map (drawing with dimensions, or a png with its scale: m per pixel, and where (0, 0) is);
2. track width (1.25–1.5 m, per section if it varies);
3. race direction and where the start/finish line is;
4. wall height and colour/material.

Then:
1. `drawing_to_map.py` → sim map;
2. `make_raceline.py` → racing line, at the real width with a recheck of `margin`/`a_lat`;
3. re-test both drivers on it in sim (TuringTrack-style test, 10 seeds, nominal/harsh, pose delay);
4. update this file's §5 with whatever got measured.

Related: [TURING_TRACK.md](TURING_TRACK.md) (how to run and adapt the drivers), [CAR_PARAMETERS.md](CAR_PARAMETERS.md)
(every car parameter), [SIMULATION_GUIDE.md](SIMULATION_GUIDE.md) (other simulators, ROS 2 sim).
