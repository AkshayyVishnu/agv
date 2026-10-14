# Decision record: turing-ml

Owner of this file so far: Person B (harness). Each decision says **what** was decided, **why**, the
**evidence**, what was **rejected**, and what would **reopen** it. Newest decisions go at the bottom.
Code comments reference these by number (e.g. `# D-05`).

Context: Raspberry Pi 5 (CPU only, Hailo AI HAT+ maybe later), LDRobot D500 LiDAR (much weaker than the
Hokuyo-class sensors other teams use), 20 cm track walls, team sim = `f1tenth_gym` **main** + `f1tenth_gym_ros`
**main** (per `adp-sudo/turing-track-ros2/docs/turing_gym_setup_guide.md`). Team rule: never trust clean-sim
results; every scenario runs with realistic D500 degradation.

---

## D500 sensor model (`turing_ml/d500.py`, `configs/sensor.yaml`): 2026-10-11

### D-01 · Separate module, pure NumPy, old-NumPy safe
- **What:** the model lives in `turing_ml/d500.py`, not in `sensor.py`. No ROS imports. Python ≥ 3.8, NumPy ≥ 1.18.
- **Why:** `sensor.py`/`to_grid` belongs to Person A, and a separate file avoids merge conflicts on Day 1. Without ROS
  imports the gym harness *and* Aayush's ROS relay can import the same code (one source of truth, see D-17). The
  team container pins `numpy<=1.22.0` through `f110_gym`.
- **Evidence:** tests pass on Python 3.10 + NumPy 1.22.0 (team versions) and Python 3.13 + NumPy 2.x. NumPy 1.22
  raised `OverflowError` on `Generator.uniform` with NaN bounds where 2.x silently returned NaN. That bug was
  found and fixed only because of the second environment.
- **Rejected:** writing into `sensor.py`, because of an ownership clash with A.
- **Reopen if:** the team decides `sensor.py` should hold everything. Then move the class there unchanged.

### D-02 · Emulate the *real driver's* output, not Aayush's relay
- **What:** scans look exactly like what `ldlidar_stl_ros2` publishes on the car.
- **Why:** the policy, `to_grid` and the ROS node will consume the real driver's messages. If sim feeds them a
  different format, format bugs only appear on the car.
- **Evidence:** `ldlidar_stl_ros2/src/demo.cpp`, `ToLaserscanMessagePublish()` (cloned to `E:\ML\AGV\ldlidar_stl_ros2`).
- **Rejected:** copying `d500_relay.py` (inf for no-return, 270°, a fixed 360 beams, one instantaneous scan + delay).
  It is a reasonable first cut, but it differs from the car on every point in D-03…D-06.

### D-03 · Output format details (all copied from the driver)
| Field | Value | Consequence for consumers |
|---|---|---|
| `angle_min`, `angle_max` | `0`, `2π` (not −π…π) | angles must be wrapped (`to_grid` already does) |
| beam count `n` | points in this revolution ≈ 5000 Hz × period, **varies per scan** | never hard-code the beam count; use `angle_increment` from each message |
| `angle_increment` | `2π/(n−1)` | |
| binning | `floor(θ/inc)` (the `laser_scan_dir=True` CCW branch reduces to this), **min** per bin | |
| empty bins | **NaN**, and the last bin is always NaN | NaN appears even with a perfect sensor |
| no return / out of range | **NaN** (driver sets NaN when distance=0 & intensity=0) | NaN means *unknown*, not far (see action A-1) |
| far returns | finite metres (up to 12 m, or 8 m on dark targets) | **not** capped at 4 m and **not** `inf` |
| `header.stamp` | publish time = **end** of the revolution | the stamp is up to 100 ms later than the oldest point in it |
| `range_min`/`range_max` | 0.02 / 25 (driver constants) | don't use them as physical limits |

### D-04 · 360° coverage, nearest-beam sampling
- **What:** each revolution covers 360°. The gym only simulates ±134.6°, so the rear sector comes out NaN. Each
  point takes the gym beam nearest its angle (no interpolation).
- **Why:** NaN is the honest output for "not simulated", and `to_grid` only uses ±135° anyway. Linear interpolation
  would create a ghost value at *every* depth edge. Ghosts are modelled deliberately instead (D-10).
- **Rejected:** raising the gym FOV to 2π. That changes the gym's own iTTC collision geometry, which belongs to `env.py`.

### D-05 · Motion skew and spin direction
- **What:** every point is sampled from the gym scan at **its own measurement time** within the revolution. The
  default physical spin is **clockwise** seen from above, with the revolution seam at sensor 0°. Training
  randomizes cw/ccw until the spin is measured.
- **Why:** a revolution takes 100 ms. At 4 m/s the car moves 0.4 m while one scan is taken, so walls bend and a
  **seam** appears where the newest and oldest points meet. With the sensor mounted forward, the seam is **straight
  ahead**. The relay's "instant scan + 50 ms" hides this and under-states the age of most points by up to 100 ms.
- **Evidence for cw:** the driver flips index order to produce CCW output when `laser_scan_dir=True` (the shipped
  default), which implies the device reports angles clockwise. This is an *inference*; measure it (M-1).
- **Measured in sim** (`scripts/d500_gym_smoke.py`, 40 m corridor, gym main, driving at the end wall):

  | preset, speed | right of seam (oldest) | left of seam (newest) | seam step straight ahead |
  |---|---|---|---|
  | clean, 4 m/s | 150 ms → 0.60 m stale | 65 ms → 0.26 m | 0.35 m |
  | nominal, 4 m/s | 165 ms → 0.66 m | 55 ms → 0.22 m | 0.35 m |
  | harsh, 4 m/s | 210 ms → 0.84 m | 90 ms → 0.36 m | 0.48 m |
  | nominal, 2 m/s | 155 ms → 0.31 m | 50 ms → 0.10 m | 0.20 m |

  The expected right-side age is period + latency (150 ms nominal), and the seam step should be ≈ speed × period.
  Both match.
- **Rejected:** instantaneous scans (the relay approach).
- **Reopen if:** the on-car spin check (M-1) or the seam location (M-2) says otherwise. Change `spin_direction` /
  `mount_yaw` in the yaml, not the code.

### D-06 · Timing: rate, jitter, latency
- **What:** 10 Hz nominal, 1 % period jitter, clipped to the spec's 6–13 Hz. 5000 points/s. Latency (end of
  revolution → usable by us) 50 ± 10 ms.
- **Why:** these are the spec (rate, sample rate) and the team's working figure (50 ms, **not measured**).
  Including skew, each point is 50–150 ms old when usable.
- **Note for consumers:** do not treat `header.stamp` as the measurement time of the scan.
- **Requirement on the harness:** call `push()` at **every physics step** (10 ms). Pushing only at 20 Hz control
  steps makes the skew coarser.

### D-07 · Range noise
- **What:** Gaussian noise with the spec's accuracy bands as σ: 10 mm (< 0.5 m), 20 mm (0.5–2 m), 30 mm (> 2 m),
  times `noise_scale`, plus a per-episode `range_bias`. The gym already adds σ = 0.01 m, so the model adds only
  `sqrt(σ² − 0.01²)`. Readings outside 0.03–12 m become NaN.
- **Why:** this matches the spec, and the team's "30 mm" figure is the far band. The subtraction stops the noise
  being counted twice.
- **Assumption:** the spec's "accuracy" was treated as 1σ, which is conservative if it is really 2σ.
  `harsh` doubles it anyway.
- **Rejected:** a flat 0.03 m (relay). It is too noisy close up and gives no way to stress-test.

### D-08 · The 20 cm walls (the D500-specific failure)
- **What:**
  1. Beyond `wall_visible_range` (4.5 m nominal, 3.5 m harsh) the beam passes over the wall. Half of those beams
     return NaN; the other half return a far reading (something behind the track).
  2. **Pitch-up event** (nominal 2 %, harsh 10 % of scans): over a 30–120° sector, walls beyond 1.5–3.5 m vanish.
     The scan suddenly shows a **false opening**.
  3. **Pitch-down event** (nominal 1 %, harsh 5 %): over a sector the beam hits the **floor** at 1.5–3.5 m, a
     **phantom wall**.
- **Why:** the beam sits ~10 cm up and the walls are 20 cm tall. A 1.4° nose-up pitch lifts the beam 10 cm over
  4 m and it clears the wall; a ~2° nose-down pitch puts the floor at ~3 m. Acceleration and braking cause exactly
  these pitches. False openings are the most dangerous case for any gap-following driver.
- **Rejected:** a single static 4 m cap (relay). It does nothing downstream because `to_grid` caps at 4 m anyway.
- **Reopen with:** M-4 (on-car pitch/overshoot logging).

### D-09 · Dropouts
- **What:** independent NaN per point (1 % nominal, 5 % harsh) plus **bursts**: Poisson 0.05 (nominal) / 0.5
  (harsh) per scan, each a 3–25° sector of NaN.
- **Why:** matte black or shiny surfaces return nothing, and the spec gives only 8 m range on 4 % reflective
  targets. Bursts reproduce "a dark stretch of wall reads as unknown".

### D-10 · Mixed pixels
- **What:** at depth jumps > 0.3 m between consecutive points, a point takes a value between the two with
  probability 0.1 (nominal) / 0.3 (harsh).
- **Why:** cheap ToF sensors report blended ranges at edges. These are exactly the edges the disparity extender
  acts on.

### D-11 · Self-hits: mechanism in, values out
- **What:** `self_hit_sectors` (angle range plus a fixed short distance) exists but is empty in every preset.
- **Why:** the angles depend on the real mount (mast, wires). Guessing them would teach policies wrong things.
  On the car the fix is to *mask* them in `to_grid`; the sim feature exists to test that masking.
- **Reopen with:** M-3.

### D-12 · Mounting yaw and mirror
- **What:** `mount_yaw` (the sensor's 0° relative to car forward) and `flip` (the driver run with
  `laser_scan_dir=False`) are sensor parameters. Harsh adds a 0.02 rad (~1.1°) mounting error. Training randomizes
  ±0.03 rad.
- **Why:** the manual's rule is to fix orientation in the sensor config, never in the policy. The parameters let us
  test that `to_grid` and the drivers handle it.

### D-13 · Presets and randomization
- **What:**
  - `nominal`: the best current estimate. The default for `eval.py` and `collect.py`.
  - `harsh`: pessimistic but plausible. Every driver must lap here before going on the car.
  - `clean`: unit tests **only**; it is the "perfect sim" the team rule forbids for evaluation.
  - `randomization`: per-episode ranges for training, on top of nominal. It includes the unknown spin direction.
- **Why:** report every result at nominal **and** harsh, and train across the range so the real car is "one more
  sample".

### D-14 · Determinism
- **What:** each model owns a `numpy.random.Generator`; `reset(t0, seed)` reseeds it. There is no global `random`.
- **Why:** `eval.py` must print identical numbers on two runs. Aayush's relay uses Python's unseeded global
  `random`, so it can't be reproduced (see action R-2).
- **Limit:** identical only within one NumPy version. Don't compare exact values across versions.

### D-15 · API (differs from the manual's sketch on purpose)
- `D500Model(cfg, seed)`, `reset(t0, seed=None, cfg=None)`
- `push(t, ranges, angle_min, angle_increment)`: the angle info is needed because sources differ (gym 1080 beams
  over 270°, a relay could be anything).
- `read(t)`: the newest usable `Scan` or `None`. `poll(t)`: every newly usable scan (for a relay that must publish each).
- `Scan` mirrors `LaserScan` plus `seq` and `available_at`.
- **Consequence for `env.py`:** don't refresh the grid on a fixed "every 10 physics steps" schedule (the rate
  jitters). Check `read(t).seq` at every control step.

### D-16 · How it was verified
- `tests/test_d500.py` (25 tests): format, timing, rate, the two direction tests run separately (object at +90°
  appears at +90° for either spin; the skew sign flips with spin), mount yaw, flip, noise per band,
  noise-subtraction, dropout rate, overshoot, floor hits, self-hits, determinism, reset, presets, randomization
  bounds.
- `scripts/d500_gym_smoke.py`: an integration test on the real gym `main` (table in D-05). It generates its own
  corridor map, because on the bundled Levine the car drives off the map from (0, 0, 0) and **no collision is ever
  reported**. An earlier run on Levine looked like it validated the lag but meant nothing: the true range was a
  flat 30 m, so any lag fit.
- Local environments: `E:\ML\AGV\.venv-main` (Python 3.10, NumPy 1.22.0, gym 0.19.0, `f110_gym` main = team
  versions) and `E:\ML\AGV\.venv-gym` (Python 3.13, newest packages + `f1tenth_gym` dev-humble).

### D-17 · Relay and harness must share this module (proposal, not done)
- **What:** propose that `d500_relay` imports `turing_ml.d500` and reads the same `configs/sensor.yaml`.
- **Why:** this is the manual's "twin drift" pitfall. The current relay already differs from this model on:
  NaN vs inf, 360° vs 270°, variable vs fixed beam count, skew, range-dependent noise, overshoot, dropouts, seeding.
- **Not done:** the relay is Aayush's code, so this is a coordination item (R-1).

---

## TuringEnv (`turing_ml/env.py`): 2026-10-11

### D-18 · The only file that touches the gym
- **What:** every `gym` / `f110_gym` call is inside `env.py`. `eval.py`, `collect.py`, DAgger and RL use `TuringEnv` only.
- **Why:** the manual's rule. The gym has two incompatible branches (main vs dev-humble); a switch must touch one file.
- **Target:** gym **main** (old `gym` API), because that is what the team sim installs (setup guide clones the
  default branches).

### D-19 · Control at 20 Hz: each action held for 5 physics steps
- **What:** `step(action)` runs 5 × 10 ms physics steps with the same `(steer, speed)`. `control_hz` must divide
  100 Hz exactly, otherwise `ValueError`.
- **Why:** the car's contract is ≥ 20 Hz on `/drive_raw`. Acting every 10 ms would make a policy react 5× faster
  than it can on the car and look better than it is.
- **Not added yet:** extra actuation delay. Gym main already delays steering by 2 physics steps (20 ms, its internal
  `steer_buffer`). Real VESC/servo delay is unmeasured. Add it as a randomized action queue when measured (Phase 7).

### D-20 · Sensing only through the D500 model, read at every control step
- **What:** every physics step's perfect gym scan is pushed into `D500Model`. At the end of each control step
  the env takes `read(t)`, the newest scan usable by then, and only re-runs `to_grid` when `seq` changed.
- **Why:** the D500 rate jitters (D-06), so the manual's "refresh every 10 physics steps" would sometimes show a
  scan before it exists or keep an old one too long. The grid updates at ~10 Hz while control runs at 20 Hz,
  exactly as on the car.
- **Verified:** `test_grid_updates_at_scan_rate_not_control_rate`: 18–22 new scans in 2 s of 20 Hz control.

### D-21 · Warm-up on reset
- **What:** `reset()` steps physics with zero action until the first scan is usable (~150 ms, capped at 1 s), then
  starts the episode clock (`obs["t"] = 0`) and the lap clock.
- **Why:** without it the first observations are an "all 4 m, fully open" placeholder grid, and an expert labels
  that as "full speed straight" (finding #4 from the manual review). The real car also sits at the start line
  with its LiDAR already running.

### D-22 · Observation dict: car-available vs ground truth kept apart
| Key | Source | May a policy use it? |
|---|---|---|
| `grid` | D500 model → `to_grid` (271 float32, metres, capped 4 m) | **yes**, the main input |
| `speed` | gym `linear_vels_x` (car: VESC odometry) | **yes** |
| `yaw_rate` | gym `ang_vels_z` (car: IMU) | **yes** (the roadmap suggests feeding it) |
| `pose` | gym true x, y, yaw | **no**: privileged expert (pure pursuit), metrics, plots |
| `collision`, `laps`, `lap_times`, `done`, `truncated` | env bookkeeping | no, used for scoring |
| `t`, `scan_seq`, `scan_age`, `last_action` | env bookkeeping | for debugging and logging |
- **Why:** one dict keeps call sites simple. The table makes it explicit which keys would be cheating if fed to the
  student net.
- **Note:** `speed` and `yaw_rate` are perfect in sim. Odometry noise is not modelled yet. Add it with domain
  randomization (Phase 7), and log it here when it is.

### D-23 · LiDAR 0.275 m ahead of the rear axle
- **What:** `lidar_dist=0.275` is passed to the gym.
- **Why:** this is `scan_distance_to_base_link` in the team's `f1tenth_gym_ros` bridge, so harness and ROS sim see
  from the same point.
- **Side effect:** gym main's iTTC collision check uses the scan from that offset point with car-centred side
  distances, so collision timing differs slightly from `lidar_dist=0`. Matching the team sim is the point.
- **Reopen with:** the measured LiDAR position on the real car.

### D-24 · Collisions are ground truth from the gym
- **What:** `collision` comes from the gym's own check on its perfect scan, never from the degraded scan.
  An episode ends on the first collision.
- **Why:** whether the car hit a wall is a fact, regardless of what the D500 reported. Scoring must not depend on
  sensor noise.

### D-25 · Lap times from lap-count increments
- **What:** `lap_times` lists each lap's duration, recorded when the gym's `lap_counts` goes up.
  `done` = collision or `laps >= max_laps` (default 2). `truncated` = `max_time` (default 60 s) reached without
  being done.
- **Why:** gym main's `lap_times` is cumulative time until 2 laps finish, not a lap time.
- **Verified:** `test_lap_time_is_per_lap_on_a_ring`: 5 m-radius ring at 2 m/s gives two laps of ≈ 15.7 s each (±10 %).
- **Known gym behaviour (unchanged):** a lap counts when the car leaves and re-enters a zone around its *start
  pose*, so starts must face the driving direction (manual pitfall).

### D-26 · Map loading
- **What:** `map_path` may be a bundled name (`levine`, `berlin`, `skirk`, `vegas`), a path, or a path ending in
  `.yaml`. `map_ext` defaults to whichever of `.png`/`.pgm`/`.jpg` exists. The gym always gets an absolute path.
- **Why:** bundled Levine is `.pgm` (its yaml says `.png`). The gym's own name lookup misses `vegas`, and its
  renderer breaks when given a name instead of a path.
- **Verified:** `test_bundled_pgm_map_loads_without_naming_the_extension`.

### D-27 · One TuringEnv at a time, enforced
- **What:** creating a new `TuringEnv` marks the previous one stale; any later `reset`/`step` on it raises
  `StaleEnvError`.
- **Why:** gym main keeps the scan simulator (and therefore the map) at class level. A second env silently swaps the
  first one's map, which would corrupt evaluation without any error. Use one env per map, sequentially; use
  separate processes for parallelism.

### D-28 · Seeds and per-episode sensor randomization
- **What:** `TuringEnv(seed)` seeds an episode-seed generator. `reset(pose, seed=None)` uses the given seed or draws
  one. If `sensor_randomization` (the yaml `randomization` block) is given, every episode draws a new sensor
  config on top of `sensor_cfg`. The env also passes `seed` to the gym.
- **Why:** `eval.py` needs identical results for the same seed (`test_same_seed_same_episode`), and training needs a
  new "sensor" every episode (D-13).

### D-29 · `sensor.py` placeholder (Person A's file)
- **What:** `turing_ml/sensor.py` holds the manual's `to_grid` with the A-1 NaN fix, marked as A's file.
- **Why:** `TuringEnv` must call `to_grid` (one scan-processing implementation everywhere). The manual's hand-off
  plan has B build against a stub until A pushes.
- **Action:** when A's version lands, replace the file and rerun `tests/test_env.py`. Keep the NaN fix (A-1).
- **Open point for A:** if *every* bin is unknown (total dropout), `to_grid` returns all 4 m, i.e. fully open.
  On the car that should probably mean "stop" instead.

### D-30 · Performance
- Measured on the ring with the nominal preset: ~2800 control steps/s ≈ **140× real time** on one core (Python
  3.10, NumPy 1.22, gym main). That is enough for `eval.py` (10 starts × 60 s ≈ 4 s) and data collection.

---

## Disparity extender (`turing_ml/experts.py`, `configs/expert.yaml`): 2026-10-11

**Ownership:** `experts.py` is **Person A's deliverable** (Execution Manual). B wrote this version at the user's
request, so that the harness, the evaluation and the live viewer have a real driver to run. A may replace it.
If A does, keep `tests/test_experts.py` and `scripts/eval_expert.py` green, or log here why a test no longer
applies. It starts from the manual's extender. Every change from the manual is listed below.

### D-31 · Parameters in `configs/expert.yaml`, no hard-coded 1.5 m/s
- **What:** every constant is a field of the frozen `DisparityConfig` dataclass. `load_expert_config` reads
  `configs/expert.yaml` and **rejects unknown keys**. `v_max` defaults to 3.0, not the manual's 1.5.
- **Why:** action A-3. The manual hard-codes 1.5 m/s in three places, and every D-05 timing number gets worse with
  speed, so we must tune at the speed we will race. A typo in the yaml must fail loudly rather than silently
  use a default. A frozen config can be passed to worker processes and DAgger without being mutated.
- **Evidence:** `test_config_file_has_no_unknown_keys_and_no_15_cap`. `eval_expert.py --set key=value` sweeps
  values without editing code.
- **Rejected:** module-level constants (the manual), since they can't be swept or logged per run.
- **Reopen if:** the team centralises all configs in one file. Then move the block there unchanged.

### D-32 · The D500 seam straight ahead is not a wall edge
- **What:** a range jump across the beams at 0° (±1 beam) is ignored as a disparity when it is smaller than
  `speed × seam_period + seam_margin` (0.1 s, 0.15 m).
- **Why:** action A-2 / D-05. A revolution's first and last points are taken 0.1 s apart, so straight ahead the
  scan has a fake step of ≈ speed × 0.1 s. At 3.5 m/s that is 0.35 m, which equals `disp_thresh`, so the
  manual's extender would see an edge dead ahead in every scan and swerve.
- **Evidence:** D-05 table (`scripts/d500_gym_smoke.py`): the measured seam step is 0.20 m at 2 m/s and 0.35 m
  (nominal) / 0.48 m (harsh) at 4 m/s. The tolerance at 4 m/s is 0.4 + 0.15 = 0.55 m, which covers all of them.
  A real wall edge straight ahead is still padded when its jump exceeds the tolerance.
- **Rejected:** scaling `disp_thresh` with speed. That would also hide real edges everywhere else.
- **Reopen if:** M-2 shows the seam is not at 0° on our car (`mount_yaw`), or `spin_direction` changes.

### D-33 · Side clearance: never steer toward a wall that is already beside the car
- **What:** if the minimum range in 60–100° on the side we would steer toward is below `side_clear` (0.45 m),
  steer is set to 0. `side_clear` is a separate parameter, not `half_width`.
- **Why:** the extender only reasons about headings, not about the car's body. Steering toward a gap whose
  edge is beside the car swings the body into that edge. We first used `half_width` (0.22 m) as the threshold;
  it triggered too late, because 0.22 m from the LiDAR is only about 7 cm from the side of a 0.31 m car,
  less than one control period of drift.
- **Evidence:** `test_side_clearance_blocks_turning_into_a_wall_beside_the_car` and
  `test_side_clearance_allows_turning_away_from_the_side_wall` (both fail if `side_clear=0`, checked by
  mutation).
- **Rejected:** reusing `half_width` (too late, above), or a hard 60–100° "no-go" padding (it also blocks
  legitimate tight turns).
- **Status after D-40:** with scan-age compensation in place, `--set side_clear=0` also lapped **18/18** on the
  tuning maps (seeds 0–2, nominal + harsh), 1–5 s faster. So in sim the rule is now a safety net, not
  load-bearing. It was kept because removing it would change the frozen config after the held-out run (D-38).
- **Reopen if:** crashes show the car side-swiping a wall that sat at 30–60°, which is not checked. Or drop
  the rule if on-car data agrees it is unnecessary.

### D-34 · Speed limited by stopping distance inside what the D500 can see
- **What:** `v ≤ −a·t + √((a·t)² + 2·a·(front − stop_dist))` with `a = brake_decel` (4 m/s²), `t = latency`
  (0.15 s), `front` = the minimum range within ±10°. Below `stop_dist` (0.5 m) the speed is 0.
- **Why:** the D500 sees a 20 cm wall only to about 4 m (D-08), and the scan is up to ~0.15 s old (D-05 table).
  The car must be able to stop within the range it can actually see, after the latency.
- **Evidence:** `test_wall_ahead_slows_with_stopping_distance` and `test_wall_dead_ahead_stops`.
- **Rejected:** a fixed speed-versus-front-distance ramp (the manual). It ignores latency, which is the
  dominant term at speed.
- **Reopen if:** braking is measured on the car (`brake_decel`) or M-5 gives a different latency.

### D-35 · Start pose from the centre-line heading
- **What:** `tracks.racetrack(name)` starts the car on centre-line point 0, heading toward the first centre-line
  point ≥ 1 m further along.
- **Why:** f1tenth_racetracks gives no start yaw. The heading between adjacent points (≈ 0.1 m apart) is
  noisy, and a 10°-off start on a 2.2 m track can crash before the first scan arrives.
- **Evidence:** all 22 tracks with a `_centerline.csv` (of the 25 folders) load: map yaml and image found,
  start pose computed. Every lap in D-37/D-38 starts this way.
- **Rejected:** a hand-written start yaw per map (doesn't scale to 25 maps).
- **Reopen if:** a frozen map split (`configs/maps.yaml`, E-2) adds several start points per map.

### D-36 · Safety bubble around every scan point
- **What:** after disparity padding, every grid point at range `r` also caps the beams within
  `atan(bubble_width / r)` of it at `r` (`bubble_width` 0.35 m). So the chosen heading keeps about 0.35 m
  from **any** wall point, not only from wall edges.
- **Why:** the inside of a smooth bend has no disparity, so the manual's extender aims right along the inner
  wall and cuts the corner. That was the main failure before this change.
- **Evidence:** before the bubble, the car crashed on Spielberg and Silverstone **even with the clean sensor**.
  With it, it lapped all tuning maps at 2 m/s. Re-measured on the final code (seeds 0–2, nominal + harsh):
  with the bubble **18/18**. With `--set bubble_width=0`, **9/18**: crashes on all three maps (Spielberg n s0/s1,
  h s0/s1; Monza n s0–s2, h s0; Silverstone h s0). Command:
  `python scripts/eval_expert.py --maps Spielberg,Monza,Silverstone --set bubble_width=0`.
  Unit tests: `test_bubble_widens_the_line_past_a_tight_inside_wall` (fails with `bubble_width=0`, checked by
  mutation), `test_bend_heading_keeps_bubble_clearance_from_the_inside_wall`, and
  `test_bubble_keeps_heading_clear_of_a_lone_close_point`. A gentle synthetic bend seen through the 4 m cap
  does **not** discriminate the bubble (the far 4 m run decides), so the discriminating test uses a tight bend.
- **Rejected:** raising `half_width` instead. It only pads edges, so smooth inner walls stay unprotected.
- **Reopen if:** narrow sections (< ~1.2 m) appear on our track. The bubble on both walls could close the gap.

### D-37 · Tuning results
- **What:** `bubble_width` 0.35, `turn_slowdown` 0.75 and `v_max` 3.0 were tuned on the three **tuning** maps
  (Spielberg, Monza, Silverstone) only, with `scripts/sweep_expert.py` and later `scripts/eval_expert.py`.
- **Evidence (before D-40/D-41)**, seed 0, 1 lap, nominal/harsh: Spielberg 160/185 s, Monza 205/239 s,
  Silverstone 212/249 s. Across seeds 0–8 the same config lapped **48/54**. Failures: Monza nominal s1 and s5;
  Spielberg harsh s3 (stuck at a hairpin) and s6; Monza harsh s7; Silverstone harsh s8. That led to
  D-39…D-41.
- **Final config (D-40 + D-41)**, `python scripts/eval_expert.py` (seeds 0–2): **48/48 including the held-out
  maps** (table in D-38). Tuning maps over seeds 3–8 as well: **36/36**, so 54/54 over seeds 0–8.
- **Reopen if:** `v_max` is raised, or the car's real braking or latency (M-5) differs.

### D-38 · Held-out maps, named before any held-out run
- **What:** tuning maps are Spielberg, Monza and Silverstone. **Held out:** Catalunya, Hockenheim, Budapest,
  Sepang and Zandvoort. They were named in `scripts/eval_expert.py` before any of them was run. No parameter
  was ever chosen by looking at a held-out result, and the held-out maps were run **once**, after the config
  was frozen.
- **Why:** 25 tracks share a style. A driver tuned on all of them would hide overfitting.
- **Evidence:** the single held-out run on the frozen config (`python scripts/eval_expert.py`, lap time in sim
  seconds):

  | map | set | nominal s0 / s1 / s2 | harsh s0 / s1 / s2 |
  |---|---|---|---|
  | Spielberg | tuning | 124 / 124 / 124 | 129 / 128 / 129 |
  | Monza | tuning | 157 / 157 / 157 | 162 / 163 / 162 |
  | Silverstone | tuning | 167 / 167 / 167 | 172 / 171 / 172 |
  | Catalunya | held-out | 146 / 146 / 146 | 150 / 150 / 149 |
  | Hockenheim | held-out | 135 / 135 / 135 | 139 / 138 / 139 |
  | Budapest | held-out | 140 / 140 / 141 | 146 / 147 / 146 |
  | Sepang | held-out | 176 / 176 / 176 | 181 / 182 / 180 |
  | Zandvoort | held-out | 146 / 146 / 145 | 149 / 151 / 149 |
- **Reopen if:** anyone tunes on a held-out map. Then pick new held-out maps from the remaining 17 tracks.

### D-39 · Corner-exit speed caps: tried, rejected
- **What:** two speed-side fixes for the 48/54 failures were tried and **not** adopted. (a) `slow_on_wanted`:
  compute `turn_slowdown` from the steer before the D-33 override. (b) `ttc_min`: `v ≤ r / (ttc_min·cos θ)` for
  every beam within ±85°.
- **Why tried:** the crash logs showed `cmd_v` jumping to 2.5–3 m/s on corner exits while the inner wall was
  0.5–0.7 m away at −45°.
- **Evidence (tuning maps, seeds 0–8, nominal + harsh, 54 runs each):** baseline 48; (a) **44**; (b) 0.5 s
  50; `fov_deg=120` alone 48; (b) 0.5 s + fov 120: 50; (b) 0.7 s + fov 120: 51, but **10–20 s slower** per
  lap; all three: 51–52. The remaining crashes happened at 1.3–1.5 m/s, so speed was not the cause (D-40).
  The code for (a) and (b) was removed.
- **Reopen if:** crashes at high speed appear once `v_max` goes above 3 m/s.

### D-40 · Scan-age compensation: rotate the grid by `yaw_rate × yaw_comp_dt`
- **What:** `disparity_extender(grid, speed, cfg, yaw_rate)` first resamples the grid so that
  `new(φ) = old(φ + yaw_rate·yaw_comp_dt)`, with `yaw_comp_dt` = 0.15 s (the same scan age already used for
  `latency`). `yaw_rate` is `obs["yaw_rate"]`, which a policy may use (D-22); on the car it is IMU or odometry.
  All callers pass it. It defaults to 0, so old calls are unchanged in behaviour apart from the new defaults.
- **Why (root cause):** the harsh crashes all came at 1.3–1.5 m/s in right-handers. Comparing the D500 grid
  with the gym's true scan (Spielberg harsh s7) showed scan ages of 100–190 ms while the car was yawing at
  ~1 rad/s. The grid was therefore rotated ~9° behind reality: true −30° range 1.31 m while the grid still
  said 4.0 m, then a true 0.29–0.33 m to the right-side wall while the grid showed 0.6–0.7 m. The extender
  kept steering into a turn the car had already made. Speed caps (D-39) could not fix a wrong heading.
- **Evidence:** tuning maps, seeds 0–8, with `fov_deg=120`: `yaw_comp_dt` 0 → 48/54, 0.1 → 53/54,
  **0.15 → 54/54**, 0.2 → 54/54. Laps also got much faster because steering stopped oscillating (seed 0, nominal:
  Spielberg 160 → 124 s, Monza 205 → 157 s, Silverstone 212 → 167 s; harsh 185 → 129, 239 → 162, 249 → 172 s).
  0.15 was picked as the value already measured as the scan age (D-05, `latency`), not as the sweep optimum.
  Unit test: `test_yaw_compensation_rotates_the_grid_the_right_way`.
- **Rejected:** full motion compensation (also translating points by `speed × dt`). The rotation was enough,
  and D-34 already accounts for the forward motion.
- **Risk / reopen if:** in sim `yaw_rate` is perfect. On the car it will be noisy and delayed (E-3). Rerun this
  table with odometry noise when E-3 exists, and if the scan age measured in M-5 differs from 0.15 s.

### D-41 · Headings considered within ±120°, not ±90°
- **What:** `fov_deg` = 120. The steering command is still clipped to `steer_max`.
- **Why:** Spielberg harsh s3 got stuck at a hairpin. The only opening was at −90° and beyond, and the
  bubble/disparity padding from the wall at −85° erased it inside ±90°. Steer dithered between −0.4 and 0,
  the car drove into the wall face, and `stop_dist` held it at speed 0 for good.
- **Evidence:** with `fov_deg=120` that run no longer stalls at the hairpin. Alone, it then crashed at t = 96 s
  in the D-40 corner; with D-40 as well it laps. Every other run in the 54-run sweep had the **identical**
  result and lap time with and without it, so the change only acts in hairpins. Part of the final 54/54 and 48/48 results.
- **Rejected:** a creep-and-turn recovery when stopped. It is more code, and it drives with a wall 0.5 m ahead.
- **Reopen if:** the car picks a sideways opening on a wide-open section (none seen on 8 maps).

---

## Needs measurement on the car (owner: B + Aayush, after Oct 20)
| # | What | How | Updates |
|---|---|---|---|
| M-1 | Physical spin direction | slow-motion phone video of the rotor, or a box swept past with the car moving | `spin_direction` |
| M-2 | Where the revolution seam is | log `/scan` while the car drives at a wall; look for the step | `mount_yaw` / seam handling |
| M-3 | Self-hit angles | car static in the open, look for constant short readings | `self_hit_sectors`, the `to_grid` mask |
| M-4 | Overshoot/floor behaviour | bag `/scan` + IMU during hard acceleration and braking | `wall_visible_range`, pitch probabilities |
| M-5 | Real latency | header stamp vs receive time (infra latency harness) | `latency_s`, `latency_jitter_s` |
| M-6 | Real noise per range | static car, boxes at 0.3/1/3 m, 200 scans each | `noise_bands`, `noise_scale` |
| M-7 | Real scan rate and point count | `ros2 topic hz /scan`, `len(ranges)` | `rate_hz`, `sample_rate_hz` |
| M-8 | Dark/shiny wall sections | walk the track with `scan_check` | `burst_rate` |

## Action items for others
| # | Who | What |
|---|---|---|
| A-1 | Person A | **The manual's `to_grid` treats NaN as far:** `far = ~np.isfinite(r) \| (r > rmax)`. Real-driver dropouts (NaN) become open 4 m space, the car's worst case. Fix: `far = np.isposinf(r) \| (np.isfinite(r) & (r > rmax))` and `bad = np.isnan(r) \| (np.isfinite(r) & (r < range_min))`, then interpolate `bad` like the existing holes. **Checked 2026-10-11:** 1 m corridor, nominal preset with dark-wall bursts. The manual's version read 55/271 grid beams ≥ 1 m more open than the truth (worst +3.0 m, a wall shown as open); with the fix, 0/271 (worst +0.14 m). |
| A-2 | Person A | **The seam straight ahead is a fake disparity at speed.** The step is ≈ speed × 0.1 s, i.e. 0.35 m at 3.5 m/s, equal to the manual's `disp_thresh`. Above ~3.5 m/s the extender will "see" an edge dead ahead in every scan. Options: ignore the disparity at the seam index pair, or scale `disp_thresh` with speed. Tune on `nominal` *and* `harsh` at the target speed. |
| A-3 | Person A / all | Speed in the manual's code is hard-coded to 1.5 m/s (`v_max`, `v_cap`, `V_MAX`). Faster runs make every number in the D-05 table worse; tune at the speeds we'll race. |
| R-1 | Aayush | Make `d500_relay` import `turing_ml.d500` + `configs/sensor.yaml` (D-17). |
| R-2 | Aayush | The relay's noise uses unseeded global `random`; use the model's seeded generator. |
| R-3 | Aayush | The relay subscribes to `/scan` with default (reliable) QoS. That works in sim, but on the car the driver publishes best-effort (per the contract) and a reliable subscriber receives nothing. Use `qos_profile_sensor_data`. |
| R-4 | Aayush | The setup guide's gym install fix is out of date: `gym==0.19.0` needs `pip<24.1` **and** `pip install --no-build-isolation gym==0.19.0`; `setuptools<66` alone fails with current pip. |
| R-5 | Aayush | `gym_bridge.py` sets `angle_inc = fov / beams`, but the gym uses `fov / (beams − 1)`, so `/scan` angles drift up to ~0.25° at the edges. Small, but `to_grid` bins at 1°. |
| R-6 | Aayush | The topic contract's open items mention "JetPack version on the Jetson"; our compute is a Raspberry Pi 5. |
| A-4 | Person A | `turing_ml/experts.py` was written by B at the user's request (D-31…D-41). It's yours: replace or adopt it, keeping `tests/test_experts.py` and `python scripts/eval_expert.py` (48/48) green. Note the new `yaw_rate` argument (D-40); A-2 and A-3 are handled there (D-32, D-31). |
| R-7 | Aayush + A | D-40 needs the car's **yaw rate** (IMU or `/odom` `twist.angular.z`) in the ML node. The contract (`/scan_d500` → `/drive_raw`) has no such input yet. Without it the expert is back to 48/54 on the tuning maps. Add a topic to the contract and log its latency. |
| E-1 | B (me), `env.py` | **Done 2026-10-11**, see D-18…D-30. |
| E-2 | B (me), next | Start poses + frozen map split (`configs/maps.yaml`), then `eval.py` on `TuringEnv` (Day 2). |
| E-3 | B (me), later | Odometry noise on `speed`/`yaw_rate`, and actuation delay (D-19, D-22), when Phase 7 measurements exist. |
