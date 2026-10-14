"""TuringEnv tests on f1tenth_gym main with generated maps. Skipped if f110_gym is missing."""
import math
import pathlib

import numpy as np
import pytest

pytest.importorskip("f110_gym")

from turing_ml import testmaps                                   # noqa: E402
from turing_ml.d500 import config_from_dict, load_preset, load_sensor_yaml  # noqa: E402
from turing_ml.env import StaleEnvError, TuringEnv              # noqa: E402
from turing_ml.sensor import GRID                                # noqa: E402

CFG = str(pathlib.Path(__file__).resolve().parents[1] / "configs" / "sensor.yaml")
NOMINAL = load_preset(CFG, "nominal")


@pytest.fixture(scope="module")
def corridor():
    return testmaps.corridor(length=40.0, width=4.0)


def beam(grid, deg):
    return float(grid[int(np.argmin(np.abs(GRID - math.radians(deg))))])


def drive(env, action, seconds):
    obs = None
    for _ in range(int(round(seconds / env.control_dt))):
        obs = env.step(action)
        if obs["done"] or obs["truncated"]:
            break
    return obs


def test_reset_waits_for_first_scan_and_returns_a_grid(corridor):
    env = TuringEnv(corridor, sensor_cfg=NOMINAL, seed=0)
    obs = env.reset((2.0, 2.0, 0.0))
    assert obs["grid"].shape == (271,) and obs["grid"].dtype == np.float32
    assert np.isfinite(obs["grid"]).all()
    assert obs["scan_seq"] >= 1 and obs["t"] == 0.0
    assert 0.0 < obs["scan_age"] < 0.3
    assert obs["speed"] == pytest.approx(0.0, abs=1e-6)
    # corridor is 4 m wide, car on the centre line: walls ~1.8 m left and right
    assert beam(obs["grid"], 90) == pytest.approx(1.8, abs=0.15)
    assert beam(obs["grid"], -90) == pytest.approx(1.8, abs=0.15)
    env.close()


def test_one_step_is_one_20hz_control_period(corridor):
    env = TuringEnv(corridor, sensor_cfg=NOMINAL, seed=0)
    obs = env.reset((2.0, 2.0, 0.0))
    t0 = obs["t"]
    obs = env.step((0.0, 1.0))
    assert obs["t"] - t0 == pytest.approx(0.05)
    assert obs["last_action"] == (0.0, 1.0)
    env.close()


def test_grid_updates_at_scan_rate_not_control_rate(corridor):
    env = TuringEnv(corridor, sensor_cfg=NOMINAL, seed=0)
    env.reset((2.0, 2.0, 0.0))
    seqs = [env.step((0.0, 2.0))["scan_seq"] for _ in range(40)]       # 2 s
    new_scans = len(set(seqs))
    assert 18 <= new_scans <= 22                                        # ~10 Hz
    env.close()


def test_scan_lags_truth_while_driving(corridor):
    env = TuringEnv(corridor, sensor_cfg=NOMINAL, seed=0)
    env.reset((2.0, 2.0, 0.0))
    obs = drive(env, (0.0, 3.0), 10.0)
    ages, checked = [], False
    while not obs["done"]:
        obs = env.step((0.0, 3.0))
        ages.append(obs["scan_age"])
        x = obs["pose"][0]
        if x > 36.0:                                    # wall at 39.8 m; check front reading
            truth_front = 39.8 - (x + 0.275)
            assert beam(obs["grid"], -1) > truth_front  # the sensor always sees the past
            checked = True
            break
    assert checked, "never reached the wall-check zone"
    assert 0.05 <= np.median(ages) <= 0.2
    env.close()


def test_driving_into_a_wall_ends_with_collision(corridor):
    env = TuringEnv(corridor, sensor_cfg=NOMINAL, seed=0)
    env.reset((2.0, 2.0, 0.0))
    obs = drive(env, (0.0, 4.0), 20.0)
    assert obs["collision"] and obs["done"] and not obs["truncated"]
    assert obs["pose"][0] > 38.0
    env.close()


def test_time_limit_truncates(corridor):
    env = TuringEnv(corridor, sensor_cfg=NOMINAL, seed=0, max_time=1.0)
    env.reset((2.0, 2.0, 0.0))
    obs = drive(env, (0.0, 0.0), 5.0)
    assert obs["truncated"] and not obs["done"]
    env.close()


def test_same_seed_same_episode(corridor):
    def episode():
        env = TuringEnv(corridor, sensor_cfg=NOMINAL, seed=7)
        obs = env.reset((2.0, 2.0, 0.0))
        grids = [obs["grid"]]
        for k in range(60):
            grids.append(env.step((0.1 * math.sin(k / 5), 2.0))["grid"])
        env.close()
        return np.stack(grids)
    assert np.array_equal(episode(), episode())


def test_creating_a_second_env_makes_the_first_stale(corridor):
    a = TuringEnv(corridor, sensor_cfg=NOMINAL, seed=0)
    a.reset((2.0, 2.0, 0.0))
    b = TuringEnv(corridor, sensor_cfg=NOMINAL, seed=0)
    with pytest.raises(StaleEnvError):
        a.step((0.0, 1.0))
    b.reset((2.0, 2.0, 0.0))
    b.step((0.0, 1.0))
    b.close()


def test_sensor_randomization_draws_a_new_sensor_each_episode(corridor):
    spec = load_sensor_yaml(CFG)["randomization"]
    env = TuringEnv(corridor, sensor_cfg=NOMINAL, sensor_randomization=spec, seed=0)
    cfgs = []
    for _ in range(5):
        env.reset((2.0, 2.0, 0.0))
        cfgs.append(env.sensor_cfg)
    assert len({(c.rate_hz, c.latency_s) for c in cfgs}) == 5
    env.close()


def test_lap_time_is_per_lap_on_a_ring():
    path, (cx, cy) = testmaps.ring(radius=5.0, width=2.0)
    env = TuringEnv(path, sensor_cfg=NOMINAL, seed=0, max_laps=2, max_time=60.0)
    # start on the bottom of the ring heading +x, i.e. counter-clockwise around the centre
    obs = env.reset((cx, cy - 5.0, 0.0))
    wheelbase = 0.15875 + 0.17145
    while not (obs["done"] or obs["truncated"]):
        x, y, yaw = obs["pose"]
        r = math.hypot(x - cx, y - cy)
        steer = math.atan(wheelbase / 5.0) + 0.5 * (r - 5.0)          # circle + radius hold
        obs = env.step((steer, 2.0))
    assert not obs["collision"]
    assert obs["laps"] == 2
    expected = 2 * math.pi * 5.0 / 2.0                                 # ~15.7 s per lap
    for lt in obs["lap_times"]:
        assert lt == pytest.approx(expected, rel=0.1)
    env.close()


def test_bundled_pgm_map_loads_without_naming_the_extension():
    env = TuringEnv("levine", sensor_cfg=NOMINAL, seed=0)
    assert env.map_ext == ".pgm"
    env.reset((0.0, 0.0, 0.0))
    env.close()
