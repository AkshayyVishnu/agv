# Turing Track: ML harness and disparity-extender report

*Status as of 2026-10-11 · Person B (harness) · `turing-ml/` in [github.com/AkshayyVishnu/agv](https://github.com/AkshayyVishnu/agv)*

This report explains what has been built, how simulated sensor data is produced, why the noise looks the
way it does, which driving algorithm is implemented, where it fails, and how to improve it. Every design
choice has a numbered entry (D-xx) with evidence in [`DECISIONS.md`](DECISIONS.md); this report is the
readable overview, that file is the detailed record.

---

## 1. Summary

| What | Status |
|---|---|
| Realistic D500 LiDAR model for simulation | ✅ done, 25 tests, validated against the gym (scan age within 5 ms of the design value) |
| `TuringEnv`: the simulator wrapped to behave like the car (20 Hz control, D500 sensing) | ✅ done, 11 tests, ~140× real time |
| Disparity-extender driver (reactive, LiDAR-only) | ✅ laps **48/48** runs: 8 tracks × 2 sensor presets × 3 seeds, at up to 3 m/s |
| One-command evaluation (`eval_expert.py`) | ✅ prints map × preset × seed table with crash diagnostics |
| Live viewer (`drive_live.py`) | ✅ window with the car, the map and the D500 points the driver saw |
| Decision record | ✅ D-01…D-41, measurement list, action items for teammates |
| Tested on the team's exact versions | ✅ Python 3.10, numpy 1.22.0, gym 0.19.0, f1tenth_gym `main`; 57/57 tests pass |
| Runs on the real car | ❌ not yet. Car arrives after Oct 20; several sensor parameters are still unmeasured (§8) |

**The headline lesson:** in simulation, almost every failure came from the sensor being **old** (100–190 ms per
reading on a 10 Hz rotating LiDAR), not from noise or speed. Compensating the scan for how far the car has
turned since it was taken (D-40) was the single most effective fix.

---

## 2. Context and constraints

- **The race:** RoboRacer/F1TENTH-style 1/10-scale autonomous car, our team "Turing Track".
- **Our job (ML team):** one ROS 2 node, LiDAR scan in → `AckermannDriveStamped` (steer, speed) out on
  `/drive_raw` at ≥ 20 Hz. Infra (Aayush) owns the simulator, safety gate, watchdog and car drivers.
- **Hardware that shapes everything:**
  - **Raspberry Pi 5:** CPU only (no GPU, unlike the Jetson boards other teams use). A Hailo AI HAT+ (NPU)
    may come later. Every on-car algorithm must be cheap.
  - **LDRobot D500 LiDAR:** 360°, ~10 Hz, ~500 points per revolution, ±10–30 mm accuracy. Much weaker than
    the Hokuyo-class sensors (40 Hz, 1080 beams) most published F1TENTH work assumes.
  - **20 cm-tall track walls** with the beam ~10 cm up, so walls are reliably visible only to ~4 m.
- **Corrections to the original plan PDFs:** speed is **not** capped at 1.5 m/s (higher speeds may be tried;
  the safety gate's clamp still applies on the car), and the real track map is available **~10 days before**
  the event, not 1 day.
- **Team rule:** never trust results on a clean or perfect simulated sensor. Every result must hold on a
  realistic and a pessimistic sensor model.
- **Team roles (sprint Oct 12–14):** A = reactive driver, **B = harness (me)**, C = map-based driver
  (pure pursuit), D = localization test + random tracks.

---

## 3. Where the simulations run

### 3.1 Simulators
| Simulator | Used for | Notes |
|---|---|---|
| **`f1tenth_gym` main branch** (old `gym` 0.19 API) | Everything in this report: training, evaluation, live view | This is what the team sim installs (Aayush's setup guide clones the default branches). Physics at 100 Hz (10 ms steps), single-track vehicle model, RK4 integrator. |
| `f1tenth_gym_ros` (ROS 2 bridge, inside Aayush's Docker container) | Testing the real ROS node later | Same physics behind ROS topics. Not used yet by this code. |
| `f1tenth_gym` dev-humble branch (gymnasium API) | Not used | Newer, nicer API, but **not** what the team runs. Installed locally only for exploration. |

### 3.2 Python environments on this machine
| Path | Versions | Purpose |
|---|---|---|
| `E:\ML\AGV\.venv-main` | Python 3.10, numpy 1.22.0, gym 0.19.0, f110_gym main | **Use this.** Matches the team container. |
| `E:\ML\AGV\.venv-gym` | Python 3.13, numpy 2.x, f1tenth_gym dev-humble | Cross-version check only (sensor tests pass here too). |

Installing gym 0.19 today needs `pip<24.1` **and** `pip install --no-build-isolation gym==0.19.0`.
The setup guide's `setuptools<66` trick alone no longer works (action R-4 for Aayush).

### 3.3 Tracks
- **`f1tenth_racetracks`** (`E:\ML\AGV\f1tenth_racetracks`): 25 real circuits scaled to ~300–500 m with
  ~2.2 m wide corridors; 22 have the centre-line file needed to start a car.
  - **Tuning tracks** (parameters chosen here): Spielberg, Monza, Silverstone.
  - **Held-out tracks** (named before any run, never tuned on): Catalunya, Hockenheim, Budapest, Sepang, Zandvoort.
- **Generated maps** (`turing_ml/testmaps.py`): a straight corridor (40 m × 4 m) and a ring (5 m radius,
  2 m wide) for tests and quick demos.
- The bundled gym maps are poor test fixtures: from Levine's default start the car drives off the map and
  **no collision is ever reported** (D-16).

### 3.4 Other repos cloned for reference (`E:\ML\AGV\`)
`turing-track-ros2` (Aayush's infra), `ldlidar_stl_ros2` (the real D500 driver, used to copy its output
format), `race_stack` (ForzaETH winning stack, ROS 2 Humble branch), `f1tenth_ws` (reference nodes),
`global_racetrajectory_optimization` (TUM raceline optimizer), `f1tenth_gym_ros`.

---

## 4. How the data is produced (the sensing pipeline)

```
 f1tenth_gym physics (100 Hz)
   │  every 10 ms: a PERFECT scan, 1080 beams over ±135°, from the LiDAR position
   │  (0.275 m ahead of the rear axle, same as the team's ROS sim)
   ▼
 D500 model  (turing_ml/d500.py, push() every physics step)
   │  builds one 360° revolution every ~100 ms: each of ~500 points is taken from the gym scan
   │  at ITS OWN measurement time → motion skew; then noise, wall effects, dropouts, binning
   │  exactly like the real ldlidar driver; the scan becomes usable ~50 ms after the revolution ends
   ▼
 to_grid  (turing_ml/sensor.py, Person A's function)
   │  any scan → 271 ranges on a fixed 1° grid from −135° to +135°, capped at 4 m;
   │  NaN ("no return") is treated as UNKNOWN and filled from neighbours (fix A-1)
   ▼
 driver  (disparity_extender(grid, speed, cfg, yaw_rate))
   │  → (steer, speed)
   ▼
 TuringEnv holds that action for 5 physics steps (50 ms = 20 Hz control, like the car)
```

### 4.1 What the driver gets each control step (`TuringEnv` observation)
| Key | Meaning | Allowed as driver input? |
|---|---|---|
| `grid` | 271 degraded ranges (m) | **yes**: the main input |
| `speed` | forward speed (car: VESC odometry) | **yes** |
| `yaw_rate` | turn rate (car: IMU) | **yes** (needs a topic on the car, R-7) |
| `pose` | true x, y, heading | **no**: ground truth for scoring, plots and the map-based expert |
| `collision`, `laps`, `lap_times`, `done`, `truncated` | bookkeeping | no |
| `scan_seq`, `scan_age`, `t`, `last_action` | debugging | no |

Collisions are always judged from the gym's perfect physics, never from the noisy scan: hitting a wall is
a fact regardless of what the sensor reported.

### 4.2 Timing, the most important property
Because the D500 rotates at 10 Hz and the scan is only usable after the revolution ends:
- a point is **50–150 ms old** when the driver uses it (nominal), and up to ~190 ms on the harsh preset;
- the beams just **right** of straight ahead are the **oldest**, the beams just **left** the newest, so a
  moving car sees a **seam** (a step) dead ahead of about speed × 0.1 s.

Measured in the gym driving at a wall (`scripts/d500_gym_smoke.py`):

| preset, speed | oldest side (right of seam) | newest side (left) | seam step |
|---|---|---|---|
| nominal, 2 m/s | 155 ms → 0.31 m stale | 50 ms | 0.20 m |
| nominal, 4 m/s | 165 ms → 0.66 m stale | 55 ms | 0.35 m |
| harsh, 4 m/s | 210 ms → 0.84 m stale | 90 ms | 0.48 m |

At 4 m/s the car acts on a picture of the wall that is up to 0.8 m out of date.

---

## 5. How the noise is added, and why each effect

The sensor model reproduces what the **real** D500 driver publishes, not an idealised scan. The source of
truth for the format was the driver's own code (`ldlidar_stl_ros2/src/demo.cpp`); the numbers come from the
D500 spec sheet and the team's measurements so far.

### 5.1 Output format (copied from the real driver)
| Property | Value | Why it matters |
|---|---|---|
| Coverage | 360°, angles **0…2π** (not −π…π) | code must wrap angles |
| Points per scan | ≈ 5000 Hz × period, **varies every scan** | never hard-code the beam count |
| No return | **NaN** (not 0, not ∞) | the manual's `to_grid` treated NaN as open space, see §7.3 |
| Binning | driver's own `floor(θ/inc)` with min per bin; empty bins NaN | NaN holes appear even with a perfect sensor |
| Timestamp | **end** of the revolution | the oldest point is up to 100 ms older than the stamp |

Copying this exactly means format bugs show up in simulation instead of on the car.

### 5.2 Effects, values and reasons
| Effect | How it's modelled | nominal | harsh | Why |
|---|---|---|---|---|
| **Rotation / motion skew** | each point sampled from the gym scan at its own time in the revolution | 10 Hz, cw spin | 9 Hz | a revolution takes 100 ms; the car moves meanwhile |
| **Rate jitter** | each revolution's period varies, clipped to the spec's 6–13 Hz | 1 % | 5 % | the real motor speed isn't constant |
| **Latency** | revolution end → usable | 50 ± 10 ms | 80 ± 20 ms | team's working figure (not yet measured) |
| **Range noise** | Gaussian, σ by distance: 10 mm < 0.5 m, 20 mm < 2 m, 30 mm beyond | ×1 | ×2 | D500 spec accuracy bands. The gym already adds 10 mm, so only the remainder `√(σ² − 0.01²)` is added |
| **Range bias** | constant offset | 0 | +20 mm | calibration error |
| **Angle jitter** | per point | 0.2° | 0.5° | encoder / timing jitter |
| **Mount error** | sensor zero vs car forward | 0 | 1.1° | imperfect mounting |
| **Over-the-wall** | true range beyond the visible range → NaN (50 %) or a far reading | 4.5 m | 3.5 m | beam 10 cm up, walls 20 cm tall: the beam clears the wall at a distance |
| **Pitch-up event** | per scan, a 30–90° sector where walls beyond 2–3.5 m vanish | 2 % | 10 % | acceleration lifts the nose. Produces **false openings**, the most dangerous case for a gap follower |
| **Pitch-down event** | per scan, a sector where the floor appears at 2–3.5 m | 1 % | 5 % | braking dips the nose. Produces **phantom walls** |
| **Dropouts** | independent NaN per point | 1 % | 5 % | weak returns |
| **Dark-wall bursts** | Poisson number per scan of 3–25° NaN sectors | 0.05 | 0.5 | matte black or shiny surfaces return nothing (spec: only 8 m range at 4 % reflectivity) |
| **Mixed pixels** | at depth jumps > 0.3 m, a point lands between the two surfaces | 10 % | 30 % | cheap time-of-flight sensors blend edges, exactly where the disparity extender looks |
| **Max range** | beyond → NaN | 12 m | 8 m | spec (light vs dark targets) |
| **Self-hits** | fixed short readings at given angles | off | off | real angles depend on the mount; measure first (M-3), then mask in `to_grid` |

**Presets.** `nominal` is the best current estimate and the default for evaluation. `harsh` is pessimistic
but physically plausible; every driver must lap here before going on the car. `clean` exists **only** for
unit tests, and the scripts refuse it.

**Randomization for training.** `configs/sensor.yaml` also holds ranges (rate 8.5–11 Hz, latency 30–100 ms,
noise ×0.8–2, both spin directions, …). `TuringEnv` can draw a new "sensor" every episode, so a trained
policy sees the real car as just one more variation.

**Determinism.** Each model has its own seeded random generator, so the same seed gives an identical
episode (needed for fair comparisons). Aayush's current ROS relay uses unseeded noise (R-2).

### 5.3 How it was validated
- 25 unit tests on synthetic scans. These include the two direction checks done separately: an object on the
  left appears at +90°, and the skew sign flips with spin direction.
- Integration on the real gym: the measured scan age matched the design value within 5 ms on the clean
  setting (150 ms expected, 155 ms measured) and the seam step matched speed × period (§4.2).
- Tested on numpy 1.22 **and** 2.x. One crash appeared only on 1.22 (`uniform()` with NaN bounds) and was fixed.

---

## 6. The algorithm: disparity extender (with our changes)

**What it is.** A reactive driver: it uses only the current scan, with no map and no position, so it works on
a track it has never seen and nothing can "get lost". It is the team's fallback driver on the car, the
teacher for the neural net (imitation learning), and the baseline every other driver must beat.

**Ownership.** `experts.py` is Person A's deliverable. B wrote it on request so the harness had a real
driver. A should adopt or replace it, keeping the tests and `eval_expert.py` green (A-4).

### 6.1 Steps, in the order the code runs
1. **Scan-age compensation (D-40, our addition).** Rotate the grid by `yaw_rate × 0.15 s`. The scan shows
   the world as it was ~0.15 s ago; in a corner the car has turned since, so walls are shifted back to where
   they are now.
2. **Disparity extension (the manual's core).** Where two neighbouring beams jump by more than 0.35 m (a wall
   edge), make the near wall "fatter" by half a car width (0.22 m) on the open side, so the car won't clip the edge.
3. **Seam rule (D-32, our addition).** A jump across 0° smaller than speed × 0.1 s + 0.15 m is ignored. It is
   the rotating sensor's seam, not a wall.
4. **Safety bubble (D-36, our addition).** Pad **every** scan point by 0.35 m, not just wall edges. The inside
   of a smooth bend has no "edge", so without this the car aimed along the inner wall and cut the corner.
5. **Choose a heading.** Among beams within ±120° (D-41, was ±90°), take the farthest readings (within 0.1 m of
   the maximum, since many tie at the 4 m cap) and steer to the **centre of the widest such run**. A plain
   `argmax` would always pick the rightmost tie and drift right.
6. **Side clearance (D-33, our addition).** If the wall beside the car (60–100°) on the side we would turn
   toward is closer than 0.45 m, don't turn that way.
7. **Speed.**
   - Base speed: 0.6–3.0 m/s, scaled by the free distance straight ahead.
   - Turning: speed is cut by up to 75 % in hard turns.
   - Stopping distance (D-34, our addition): capped so the car can stop within what it sees, after 0.15 s of
     latency at 4 m/s² braking.
   - Stop: below 0.5 m to the front, speed is 0.

All settings live in `configs/expert.yaml`; nothing is hard-coded. The function is pure (same input → same
output), so imitation learning (DAgger) can ask it "what would you do here?" in any state.

### 6.2 How we got here (each step measured, tuning tracks)
| Version | Result |
|---|---|
| Manual's extender as written, 3 m/s | crashed on every real track, **even with a perfect sensor** (corner cutting) |
| + safety bubble, 2 m/s | lapped all tuning tracks on nominal; 2 harsh crashes |
| + bubble 0.35, slowdown 0.75, 3 m/s | 48/54 laps over 9 seeds |
| two extra speed caps (rejected, D-39) | 44–52/54, and 10–20 s slower: crashes happened at only 1.3–1.5 m/s, so speed wasn't the cause |
| + scan-age compensation (D-40) + ±120° (D-41) | **54/54** tuning, then **48/48** including held-out, once |

### 6.3 Final results (`python scripts/eval_expert.py`, lap time in sim seconds, seeds 0 / 1 / 2)
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

Harsh is consistently ~3–4 % slower than nominal. The held-out tracks behave like the tuning tracks, which is
evidence the settings aren't overfitted to three circuits.

---

## 7. Failure points and risks

### 7.1 Known weaknesses of the driver
| Weakness | Why | Evidence / status |
|---|---|---|
| **Depends on yaw rate** | D-40's compensation needs the car's turn rate; in sim it is perfect and instant | Without it: 48/54 instead of 54/54. On the car it must come from the IMU; the team contract has no such topic yet (R-7) |
| **Narrow sections** | the 0.35 m bubble on both walls could close a gap under ~1.2 m | not seen on 2.2 m tracks; check on our real map |
| **Getting stuck** | below 0.5 m to the front it stops for good; there is no reverse or recovery | one hairpin stall was fixed by D-41; recovery not implemented |
| **Sideways openings** | ±120° could pick a side opening on a wide-open area | not seen on 8 tracks |
| **Top speed 3 m/s, not explored above** | the 4 m sensor horizon limits safe speed (stopping distance at 5 m/s ≈ 3.9 m) | faster runs need a map-based driver (Person C) |
| **No racing line** | reactive drivers follow the free space, not the fastest line | expected; this is what the map-based driver is for |
| **Side-clearance rule is not load-bearing in sim** | with D-40, removing it still laps 18/18 | kept as a safety net; drop or keep based on car data |
| **Total sensor dropout reads as "all open"** | `to_grid` returns 4 m everywhere when no bin has data | should mean "stop"; open point for A |

### 7.2 Gaps between this simulation and the real car
| Gap | Consequence | Plan |
|---|---|---|
| D500 parameters unmeasured: spin direction, seam position, latency, noise, real rate | the model may be wrong in sign or size | measurement list M-1…M-8 in DECISIONS.md, after Oct 20 |
| No odometry/IMU noise or delay on `speed` / `yaw_rate` | D-40 looks better in sim than it will on the car | E-3: add noise and delay, rerun the table |
| Only 20 ms steering delay (gym built-in); no VESC/servo delay | the car responds faster in sim | add a randomized action queue once measured |
| Gym walls are infinitely tall; over-the-wall is only modelled statistically | real overshoot depends on actual pitch and floor | log IMU + scan on the car (M-4) |
| Gym physics: simple tyre model, a rectangle car body, perfectly flat floor, nominal grip | real grip, battery sag and bumps differ | system identification on the car, then domain randomization of physics |
| Tracks are scaled F1 circuits at 2.2 m width | our real track's widths and corners are unknown | the real map arrives ~10 days early; add it to training/eval immediately |
| No other cars or obstacles | head-to-head behaviour untested | out of scope for the sprint |
| Aayush's ROS relay is a different, simpler sensor model | ROS-sim and harness results can disagree ("twin drift") | R-1: make the relay import this model and `sensor.yaml` |

### 7.3 Bugs found in the team's plan or infra (reported, not all fixed)
- **The manual's `to_grid` treats NaN as open space.** With the real driver's NaN dropouts, 55 of 271 beams
  read ≥ 1 m too open in a corridor test (a wall shown as open space). The fixed version is used here (A-1).
- **The rotating-sensor seam is a fake wall edge** at ≥ ~3.5 m/s with the manual's threshold (A-2; handled by D-32).
- The manual's code hard-codes 1.5 m/s in several places (A-3).
- The relay subscribes to `/scan` with reliable QoS. On the car (best-effort driver) it would receive nothing (R-3).
- The ROS bridge's scan angle increment is off by one beam, ~0.25° at the edges (R-5).
- The topic contract mentions a Jetson; our computer is a Raspberry Pi 5 (R-6).

---

## 8. How to improve (in priority order)

1. **Get yaw rate into the driver node on the car (R-7)** and model its noise and delay in sim (E-3), then rerun
   `eval_expert.py`. This is the biggest risk to the current result.
2. **Measure the real D500 (M-1…M-8)** as soon as the car arrives: spin direction, seam angle, latency, noise per
   range, scan rate, self-hit angles, dark wall sections. Put the numbers in `configs/sensor.yaml` (no code
   changes) and rerun everything.
3. **Add the real track map** the day it arrives: evaluate on it, tune on it, keep the held-out tracks for honest reporting.
4. **Start poses and a frozen map split for `eval.py` (E-2):** several starts per track and `configs/maps.yaml`
   with train/val/test. This is the next harness task.
5. **Actuation delay and physics randomization** (steering lag, grip ±20 %, mass, battery sag) in `TuringEnv`.
6. **Driver robustness:** a stuck-recovery behaviour (back off and re-plan), a narrow-section mode
   (smaller bubble when both walls are close), and treating "no data at all" as stop.
7. **Speed beyond 3 m/s:** only with better look-ahead. Combine with Person C's map-based driver
   (localization + raceline), whose viability depends on Person D's localization test.
8. **Imitation learning (next phase):** use this extender as the teacher. `collect.py` records (grid, speed,
   yaw_rate) → expert action at 20 Hz, then behavioural cloning and DAgger. The expert is already a pure
   function, so DAgger can query it on any state. Because it's reactive, it can also label **real** car scans
   offline.
9. **Unify the ROS relay with this model (R-1, R-2)** so the ROS sim and the harness test the same sensor.
10. **Pi deployment:** port the extender into `policy_node` (ROS 2), then measure its CPU time on the Pi 5. The
    bubble loop is ~271 iterations in Python, cheap but worth vectorizing if the budget is tight.

---

## 9. How to run things

All commands from `E:\ML\AGV\turing-ml` in PowerShell (`python` = `..\.venv-main\Scripts\python`):

| Task | Command |
|---|---|
| Watch the driver live on the ring (15 s laps) | `..\.venv-main\Scripts\python scripts\drive_live.py --map ring --realtime` |
| Watch it on a real circuit | `..\.venv-main\Scripts\python scripts\drive_live.py --map Spielberg --realtime` |
| Same with the pessimistic sensor | add `--preset harsh` |
| Several laps / free camera | add `--laps 3` / `--free-camera` (scroll = zoom, drag = pan) |
| Headless, summary only | add `--no-render` |
| Full evaluation table (8 tracks × 2 presets × 3 seeds, ~1 min) | `..\.venv-main\Scripts\python scripts\eval_expert.py` |
| Try a setting without editing the yaml | `..\.venv-main\Scripts\python scripts\eval_expert.py --maps Monza --set bubble_width=0.3` |
| Sensor timing check against the gym | `..\.venv-main\Scripts\python scripts\d500_gym_smoke.py --preset nominal --speed 4` |
| All tests (57) | `..\.venv-main\Scripts\python -m pytest -q` |

In the live window, the **red dots** are the D500 points the driver used, drawn at the car's current position.
When the car is fast they sit ahead of the real walls; that gap is the sensor's age (§4.2).

---

## 10. Files

| File | What |
|---|---|
| `turing_ml/d500.py` | D500 sensor model |
| `turing_ml/env.py` | `TuringEnv` (the only code that touches the gym) |
| `turing_ml/sensor.py` | `to_grid` (Person A's; placeholder with the NaN fix) |
| `turing_ml/experts.py` | disparity extender (Person A's; written by B on request) |
| `turing_ml/tracks.py` | racetrack loader and start pose |
| `turing_ml/testmaps.py` | generated corridor and ring |
| `configs/sensor.yaml` | sensor presets + training randomization |
| `configs/expert.yaml` | driver settings |
| `scripts/drive_live.py` | live viewer / single run |
| `scripts/eval_expert.py` | evaluation table with crash diagnostics |
| `scripts/sweep_expert.py` | parameter sweeps |
| `scripts/d500_gym_smoke.py` | sensor timing validation |
| `tests/test_d500.py`, `test_env.py`, `test_experts.py` | 57 tests in total |
| `docs/DECISIONS.md` | every decision (D-01…D-41), measurement list, action items |
| `docs/PROJECT_REPORT.md` | this report |

---

## 11. Open items for the team (full list in DECISIONS.md)

- **Person A:** adopt or replace `experts.py` and `sensor.py`; keep the NaN fix (A-1); note the seam (A-2),
  the speed cap (A-3) and the new `yaw_rate` input (A-4).
- **Aayush:** yaw-rate topic for the ML node (R-7); relay to use this sensor model, seeded, with best-effort QoS
  (R-1…R-3); update the gym install steps (R-4); bridge angle increment (R-5); Jetson mention in the contract (R-6).
- **Person B (next):** start poses + frozen map split + `eval.py` (E-2); odometry noise and actuation delay (E-3);
  then `collect.py` for imitation learning.
- **Everyone, after Oct 20:** the sensor measurements M-1…M-8 on the real car.
- **Housekeeping:** done. The code is in [github.com/AkshayyVishnu/agv](https://github.com/AkshayyVishnu/agv);
  join the team repo when there is one.
