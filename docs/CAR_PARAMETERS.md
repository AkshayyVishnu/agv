# Car parameters in the simulation

Every parameter that describes the car, what it is set to now, where it is set, and what changing it
does to the race. Values were read from the code (f1tenth_gym `main` @ `4fdb9c7`, `turing-ml`), not from
documentation.

The car in the simulation is made of four layers. A race result depends on all four:

| Layer | Where it lives | How to change it |
|---|---|---|
| 1. **Vehicle physics**: size, mass, tyres, steering, motor | `turing-ml/configs/vehicle.yaml` (presets `sbk10`, the default, and `f1tenth_default`) | edit the yaml, or `--vehicle NAME` on `drive_live.py` / `eval_expert.py` (see [§1.6](#16-how-to-change-them)) |
| 2. **Simulation and timing**: physics step, control rate, delays, collision check | `turing_ml/env.py`, gym internals | `TuringEnv` arguments; some fixed in the gym |
| 3. **Sensor (D500 LiDAR) model** | `turing-ml/configs/sensor.yaml` | edit the yaml or add a preset |
| 4. **Driver (disparity extender) settings** | `turing-ml/configs/expert.yaml` | edit the yaml, or `eval_expert.py --set key=value` |

Numbers used for examples: our competition track is **14 × 10 m, 1.0 m wide, 300 mm walls**, with
U-turns whose centre-line radius is roughly 0.6–1.0 m (estimated from the drawing).

---

## 0. Our car: Rlaarlo Storm Breaker SBK10 (metal version)

The `sbk10` preset in `configs/vehicle.yaml` is the default for all scripts since 2026-10-24. Values and
where they come from:

| Param | f1tenth_default | **sbk10** | Status | Source |
|---|---|---|---|---|
| wheelbase `lf + lr` | 0.330 m | **0.275 m** | spec | Rlaarlo: 275 mm; Big Squid RC measured 10.7″ (272 mm) |
| `lf` / `lr` split | 0.159 / 0.171 | **0.1375 / 0.1375** | estimate | 50/50 assumed: weigh front and rear axles |
| `width` | 0.31 m | **0.25 m** | spec | Rlaarlo: 250 mm, tyres included; BSRC measured 9.9″ |
| `length` | 0.58 m | **0.40 m** | spec | Rlaarlo: 400 mm (height 133 mm) |
| `m` | 3.74 kg | **2.5 kg** | estimate | **No reliable weight published** (listings say 1.0 kg or 3.5 kg shipping). Includes Pi 5, D500, mounts |
| `I` | 0.04712 | **0.016** | estimate | default scaled by mass × (length² + width²) |
| `h` | 0.074 m | **0.065 m** | estimate | not published |
| `v_max` (gym) | 20 m/s | **12.5 m/s** | measured | BSRC: 28 mph on the stock 3S pack |
| `s_min`/`s_max` | ±0.4189 | ±0.4189 | **unknown** | steering lock not published: measure |
| `sv_min`/`sv_max` | ±3.2 | ±3.2 | estimate | stock servo is 11 kg (metal version); speed not published |
| `mu`, `C_Sf`, `C_Sr` | defaults | defaults | **unknown** | depend on tyres (Ø83×32 mm front, Ø83×39 mm rear, off-road buggy tyres) and our floor |
| `a_max` | 9.51 | 9.51 | **unknown** | grip-limited in practice: keep ≤ μ·g |

Other SBK10 facts (not used by the physics model): 4WD shaft drive with 3 gear differentials;
2852 3100 kV brushless motor; 45 A ESC; 3S 2200 mAh LiPo; 2.5 mm 6061 aluminium chassis. The stock ESC
has no speed feedback, so the driver's `speed` input needs a VESC or a wheel encoder.

| `lidar_dist` (not a gym param) | 0.275 m | **0.05 m** | estimate | D500 mount ahead of the CoG. Gym main centres the collision box on the LiDAR, so this places the simulated body (D-44). Measure the mount |

Other SBK10 facts (not used by the physics model): 4WD shaft drive with 3 gear differentials;
2852 3100 kV brushless motor; 45 A ESC; 3S 2200 mAh LiPo; 2.5 mm 6061 aluminium chassis. The stock ESC
has no speed feedback, so the driver's `speed` input needs a VESC or a wheel encoder.

**Results on TuringTrack with the sbk10 car** (after the lap-counting fix D-43; earlier numbers in this
file counted false laps and are withdrawn):

| Driver | Result |
|---|---|
| Reactive disparity extender, `v_max` 3.0 or 1.5 | **0/10**: always crashes at the bottom-left hairpin (centre line 0.53 m radius, tighter than the car's 0.62 m turning circle) |
| Pure pursuit on the racing line, perfect pose | 2/2 laps, **18.05 / 17.77 s**, mean 3.24 m/s, max 5.0 |
| Pure pursuit, pose 0.05 s late + 3 cm noise | 10/10, 18 s |
| Pure pursuit, pose 0.10 s late + 3 cm noise | 8/10 at full speed, 10/10 at `--speed-scale 0.8` (22 s) |
| Pure pursuit, pose 0.15 s late + 3 cm noise | 0/10 at full speed, 8/10 at 0.8× |

The SBK10 is narrower (0.125 m half-width instead of 0.155 m) and turns tighter (0.62 m instead of
0.74 m). The F1-track results in the README were run with the `f1tenth_default` car before D-42..D-44 and
have not been re-run.

Verified in sim: the collision box and the full-lock turning circle (0.628 m measured vs 0.618 m
expected at 0.3 m/s) follow the preset.

---

## 1. Vehicle physics (f1tenth_gym `params`)

The gym simulates a **single-track ("bicycle") model with tyre slip**. Below 0.5 m/s it switches to a
pure kinematic model (no slip). The tables below explain each parameter using the generic F1TENTH
defaults (`f1tenth_default`). Our car's values are in [§0](#0-our-car-rlaarlo-storm-breaker-sbk10-metal-version).

### 1.1 Size and geometry

| Param | Value | Meaning | Effect of changing it |
|---|---|---|---|
| `lf` | 0.15875 m | centre of gravity → front axle | With `lr` sets the **wheelbase** (0.330 m). Longer wheelbase = larger turning circle (see below) and more stable at speed. Moving the CoG forward (bigger `lr`, smaller `lf`) makes the car understeer more |
| `lr` | 0.17145 m | centre of gravity → rear axle | See `lf` |
| `width` | 0.31 m | body width | Used for **wall collisions** (half-width 0.155 m to each side). Wider = less room: on our 1.0 m track the car has 0.345 m of space on each side when centred |
| `length` | 0.58 m | body length | **Not used for wall collisions**, only for car-vs-car collisions and drawing. See §2.3 |

**Turning circle.** Minimum radius (rear axle) = wheelbase / tan(max steer):

| Max steer | f1tenth_default (0.330 m) | **sbk10 (0.275 m)** |
|---|---|---|
| 0.4189 rad (gym `s_max`) | 0.74 m | **0.62 m** |
| 0.40 rad (driver `steer_max`) | 0.78 m | **0.65 m** |
| 0.50 rad | 0.60 m | 0.50 m |

With the default car, the tightest U-turn on our track (centre-line radius ~0.6 m) was barely possible.
With the SBK10's shorter wheelbase it fits, with a margin that depends on the real steering lock, which
is not published. **Measure the steering lock first.**

### 1.2 Mass and inertia

| Param | Value | Meaning | Effect of changing it |
|---|---|---|---|
| `m` | 3.74 kg | total mass | Appears in the slip dynamics. Heavier = slower to respond in yaw. Acceleration and braking are set by `a_max`, not by `m`, so mass alone does not change straight-line speed here (on the real car, a heavier battery does) |
| `I` | 0.04712 kg·m² | yaw moment of inertia | Higher = the car rotates into a corner more slowly, so it runs wide in quick direction changes (S-bends, hairpin entries). Lower = twitchier |
| `h` | 0.074 m | centre-of-gravity height | Load transfer: braking loads the front tyres, accelerating loads the rears. Higher CoG = more understeer when accelerating out of a corner and more oversteer when braking into one. (The sim never tips over) |

### 1.3 Tyres and grip

| Param | Value | Meaning | Effect of changing it |
|---|---|---|---|
| `mu` | 1.0489 | tyre–floor friction coefficient | **The most important unknown for top speed.** Maximum cornering speed ≈ √(μ·g·R): with μ = 1.05, a 1 m radius allows ~3.2 m/s and a 0.78 m radius ~2.8 m/s. On a smooth indoor floor with plastic or foam tyres, μ may be 0.5–0.8, which gives ~2.2–2.8 m/s at 1 m. Lower μ = the car slides wide in corners and crashes at speeds that work in sim |
| `C_Sf` | 4.718 /rad | front cornering stiffness (normalised) | How much grip the front tyres produce per radian of slip. Higher = sharper turn-in. Lower = understeer: the car turns less than the steering command |
| `C_Sr` | 5.4562 /rad | rear cornering stiffness | Higher than front → the default car **understeers** slightly (stable). Rear lower than front = oversteer (the tail steps out) |

### 1.4 Steering

| Param | Value | Meaning | Effect of changing it |
|---|---|---|---|
| `s_min`, `s_max` | ±0.4189 rad (±24°) | steering lock | Sets the turning circle (table in §1.1). The driver caps itself lower, at `steer_max` = 0.40 |
| `sv_min`, `sv_max` | ±3.2 rad/s | steering rate (servo speed) | 0 → full lock takes 0.13 s, lock to lock 0.26 s. A slower servo makes the car late into corners and through S-bends. Real hobby servos are often 0.1–0.2 s per 60°, i.e. ~5–10 rad/s unloaded, slower under load. Measure it |

The gym steers **bang-bang**: it always turns the wheels at the full `sv_max` toward the target angle
(`dynamic_models.py:191`). A real servo slows down as it gets close.

### 1.5 Motor and speed

| Param | Value | Meaning | Effect of changing it |
|---|---|---|---|
| `a_max` | 9.51 m/s² | maximum acceleration **and braking** | Reaching 3 m/s from standstill takes ~0.7 s (full `a_max` until 1 m/s short of the target, then easing in); braking from 3 m/s to 0 takes 0.47 m. A real VESC car on a smooth floor will brake worse (wheel lock-up), so stops take longer than in sim. The driver deliberately assumes only 4 m/s² (`brake_decel`) |
| `v_switch` | 7.319 m/s | above this speed acceleration falls off (power limit) | Irrelevant below ~7 m/s, so it doesn't matter on our track |
| `v_min` | −5.0 m/s | top reverse speed | Also sets the braking gain (below) |
| `v_max` | 20.0 m/s | top speed of the car | It also sets how fast the gym's speed controller reacts: **accelerating gain = 10·a_max / v_max** = 4.76 (m/s²)/(m/s). Lowering `v_max` to 5 makes acceleration 4× more aggressive. Change it only together with a check of the acceleration behaviour. The driver's own `v_max` (3.0) is what actually limits speed |

Speed control is a simple proportional controller (`dynamic_models.py:179`): target speed → acceleration,
clipped to ±`a_max`. Braking gain = 10·a_max / |v_min| = 19, so braking is almost always at full `a_max`.

### 1.6 How to change them

Edit `turing-ml/configs/vehicle.yaml` or add a preset there (any key you leave out takes the
`f1tenth_default` value), then pick it with `--vehicle NAME`:

```bash
python scripts/eval_expert.py --vehicle sbk10            # default
python scripts/eval_expert.py --vehicle f1tenth_default  # the old generic car
python scripts/drive_live.py --vehicle sbk10 --map ring
```

From Python, `turing_ml.vehicle.load_vehicle("configs/vehicle.yaml", "sbk10")` returns the full dict.
`TuringEnv` passes it to the gym, which **replaces** the whole dict, so a hand-written dict must contain
every key:

```python
from turing_ml.env import TuringEnv
from turing_ml.d500 import load_preset

params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074,
          'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2,
          'v_switch': 7.319, 'a_max': 9.51, 'v_min': -5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
params.update(mu=0.7, s_max=0.5, s_min=-0.5)      # e.g. less grip, more steering lock

env = TuringEnv("maps/TuringTrack/TuringTrack_map", sensor_cfg=load_preset("configs/sensor.yaml", "harsh"),
                params=params, max_laps=1, max_time=120, seed=0)
```

`TuringEnv` rebuilds the gym's collision box for each new env, so different vehicles can be used one after another in one process.

---

## 2. Simulation and timing

| Setting | Value | Where | Effect of changing it |
|---|---|---|---|
| Physics step | 10 ms (100 Hz) | `env.py` `SIM_DT` | Fixed. The D500 model needs ≥ 100 Hz to reproduce motion skew |
| Integrator | RK4 | gym default | RK4 is the accurate option. Euler is the older, less accurate one |
| Control rate | 20 Hz (each command held 50 ms) | `TuringEnv(control_hz=20)` | Must divide 100 Hz (10, 20, 25, 50). The car's contract is ≥ 20 Hz. A lower rate means coarser steering and later reactions, which shows first in hairpins |
| Steering delay | 20 ms | gym `steer_buffer_size = 2` (`base_classes.py:109`) | Fixed in the gym. The real chain (Pi → VESC → servo) is likely 30–80 ms. **Not modelled yet** (report §7.2, E-3). More delay makes the car turn late and overshoot corner exits |
| Throttle delay | none | – | Not modelled. Real motors take time to spin up and down |
| LiDAR position | 0.275 m ahead of the rear axle | `env.py` `LIDAR_DIST` (from `f1tenth_gym_ros`) | Should match where the D500 sits on our car. Further forward = sees corners earlier but the collision box moves too (below) |
| Gym LiDAR | 1080 beams over 4.7 rad (269°), max 30 m, 0.01 m noise | gym defaults | The input to the D500 model. Leave it alone; the D500 model reshapes it to 360°, ~500 points, 10 Hz |
| Episode | `max_laps`, `max_time` | `TuringEnv` args | Only bookkeeping |
| Odometry (`speed`, `yaw_rate`) | perfect and instant | `env.py` | Real VESC odometry and IMU are noisy and delayed. The driver's scan-age compensation relies on `yaw_rate`, so it looks better in sim than it will be on the car (R-7, E-3) |

### 2.3 Collision check (important for narrow tracks)

The gym reports a crash when, for any LiDAR beam, the time until the car's body reaches the wall is
under 5 ms (`ttc_thresh = 0.005`, `laser_models.py:189`). The body used for this is a box **`width` (0.31 m)
by wheelbase (0.33 m), centred on the LiDAR**, i.e. from 0.11 m to 0.44 m ahead of the rear axle.

Consequences:
- **`length` (0.58 m) is not used.** The rear 0.11 m and any bumper overhang beyond 0.44 m can pass
  through walls without a crash. The sim is a little forgiving in tight corners, where the real car's
  corners would touch.
- A **stationary** car never "collides" (the check needs speed toward the wall). Touching a wall and
  stopping counts as stuck, not crashed.
- No damage, no bounce: a collision ends the episode.

---

## 3. Sensor (D500 LiDAR model, `configs/sensor.yaml`)

This is the sensor on the car, so it shapes the race as much as the physics does. Full reasoning is in
[PROJECT_REPORT.md §5](../turing-ml/docs/PROJECT_REPORT.md) and DECISIONS.md.

| Param | nominal | harsh | Meaning | Effect on the race |
|---|---|---|---|---|
| `rate_hz` | 10 | 9 | revolutions per second | Lower = older data and bigger motion skew. The driver reacts to walls later |
| `rate_jitter` | 1 % | 5 % | per-revolution rate variation | Small effect. Makes scan age vary |
| `latency_s` (± `latency_jitter_s`) | 50 ± 10 ms | 80 ± 20 ms | revolution end → scan usable | **The biggest effect in sim so far.** Total scan age is 100–190 ms: at 3 m/s the car has moved 0.3–0.6 m since the scan, which is half our track width |
| `spin_direction` | cw | cw | rotation direction | Decides which side of the seam is oldest. **Not measured yet** |
| `mount_yaw` | 0 | 0.02 rad | sensor misalignment | The car steers slightly to one side. On a 1 m track a 1° error is ~2 cm per metre |
| `angle_jitter` | 0.2° | 0.5° | per-point angle noise | Small |
| `range_min` / `range_max` | 0.03 / 12 m | 0.03 / 8 m | valid range | Irrelevant on a 14 × 10 m track: walls are always within range |
| `noise_bands`, `noise_scale`, `range_bias` | 10–30 mm ×1, bias 0 | ×2, +20 mm | range noise and offset | Noisy ranges create fake "disparities" (wall edges); on a narrow track they make steering jumpy. A bias makes walls look farther (+) or nearer (−) |
| `wall_visible_range` | 4.5 m | 3.5 m | beyond this the beam passes over the wall | Assumes **20 cm walls**. Our walls are **30 cm**, so walls should stay visible further. Little effect on our small track |
| `pitch_up_prob`, `pitch_down_prob` | 2 %, 1 % | 10 %, 5 % | scans where acceleration or braking tilts the beam | **False openings** (walls vanish) and **phantom walls** (floor seen). Dangerous for a gap follower at corner exits |
| `dropout_prob` | 1 % | 5 % | random missing points | Mostly filled in by `to_grid` |
| `burst_rate`, `burst_width_deg` | 0.05, 3–15° | 0.5, 5–25° | dark or shiny wall sections returning nothing | Fake gaps. Check our wall material |
| `mixed_pixel_prob` | 10 % | 30 % | points between two surfaces at edges | Blurs wall edges exactly where the driver looks |
| `self_hit_sectors` | none | none | angles blocked by the car's own parts | Measure on the car and mask them. Otherwise the driver sees a permanent "wall" |

---

## 4. Driver settings (`configs/expert.yaml`)

These are not physical properties, but they decide how hard the car is driven. Try values with
`python scripts/eval_expert.py --maps TuringTrack --set key=value` (with `TURING_RACETRACKS=maps`).

| Param | Value | Meaning | Effect, especially on our 1.0 m track |
|---|---|---|---|
| `v_max` | 3.0 m/s | top speed the driver asks for | **Must come down on our track.** In tests: 3.0 → crashes within seconds; 1.0 → 1 of 6 laps (54 s) |
| `v_min` | 0.6 m/s | slowest driving speed (when not stopping) | Too high and the car can't creep through hairpins |
| `steer_max` | 0.40 rad | driver's steering limit | Raise toward the car's real lock to tighten the turning circle (§1.1) |
| `turn_slowdown` | 0.75 | speed cut at full steering (up to 75 %) | Higher = slower in corners, safer in hairpins |
| `stop_dist` | 0.5 m | stop if anything within ±10° is closer | **Causes the car to stop for good on our track**: corner walls are often within 0.5 m and there is no reverse. 0.2–0.25 m works better on a narrow track |
| `brake_decel`, `latency` | 4.0 m/s², 0.15 s | assumed braking and delay for the "can I stop in time?" speed cap | Bigger latency or smaller decel = slower, safer |
| `half_width` | 0.22 m | how far wall edges are "fattened" (about half the car width plus margin) | Too big closes gaps on a 1 m track. Too small clips wall ends |
| `bubble_width` | 0.35 m | margin kept from **every** wall point | On a 1 m track two 0.35 m bubbles leave almost no free corridor. 0.15–0.2 m fits better |
| `disp_thresh` | 0.35 m | range jump that counts as a wall edge | Lower = more edges, more cautious, but noise triggers it |
| `fov_deg` | 120° | how far to the side a heading can be picked | Needed for hairpin exits. Too wide on an open area picks side openings |
| `side_deg`, `side_clear` | 60–100°, 0.45 m | don't turn toward a wall already beside the car | 0.45 m is more than the gap to each wall on a 1 m track (0.345 m), so it blocks turns there. Use ~0.25 m |
| `seam_period`, `seam_margin` | 0.1 s, 0.15 m | ignore the rotating sensor's seam straight ahead | Tied to the sensor rate. Update after measuring it |
| `yaw_comp_dt` | 0.15 s | rotate the scan by yaw_rate × this to undo scan age | Should equal the real scan age. Needs a real yaw-rate signal on the car |

Settings tried on our track so far (nominal + harsh, 3 seeds each). Results for both cars are in [§0](#0-our-car-rlaarlo-storm-breaker-sbk10-metal-version).
Best so far with the SBK10: `stop_dist=0.25 bubble_width=0.2 side_clear=0.25 v_max=1.5`, 4/6 laps, ~40 s.

---

## 5. What to measure on the real car (priority order)

| # | Measure | Updates | Why |
|---|---|---|---|
| 1 | Max steering angle each way (drive a slow full-lock circle: `s_max = atan(0.275 / R)`) | `s_min`, `s_max`, driver `steer_max` | Decides how much margin the hairpins have |
| 2 | Total mass and front/rear split (two scales under the axles: `lf = 0.275 × rear / total`), LiDAR position from the rear axle | `m`, `lf`, `lr`, `LIDAR_DIST` | Balance, sensor origin, collision box |
| 3 | Floor grip: slowest speed at which the car slides on a fixed circle | `mu` | Corner speed limit |
| 4 | Servo speed (time 0 → full lock) | `sv_max` | Response in S-bends |
| 5 | Command-to-motion delay (steering and throttle) | delay model (to add) | Late turn-in |
| 6 | Braking distance from 1, 2, 3 m/s | `a_max`, driver `brake_decel` | Safe speed |
| 7 | CoG height | `h`, `I` (`I` grows with mass × size²; hard to measure, keep the estimate) | Smaller effects |
| 8 | D500: spin direction, latency, noise, self-hits | `configs/sensor.yaml` (M-1…M-8) | Scan age is the biggest error source in sim so far |
