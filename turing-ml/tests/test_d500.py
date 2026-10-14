"""Tests for turing_ml.d500. Synthetic scans only; the gym integration check is
scripts/d500_gym_smoke.py. Run: python -m pytest tests"""
import math
import pathlib

import numpy as np
import pytest

from turing_ml.d500 import (D500Model, config_from_dict, load_preset, load_sensor_yaml,
                            sample_config)

CFG_PATH = str(pathlib.Path(__file__).resolve().parents[1] / "configs" / "sensor.yaml")
GYM_N, GYM_FOV = 1080, 4.7                      # f1tenth_gym main defaults
GYM_AMIN, GYM_INC = -GYM_FOV / 2, GYM_FOV / (GYM_N - 1)
GYM_ANG = GYM_AMIN + np.arange(GYM_N) * GYM_INC


def clean(**over):
    return config_from_dict(over, load_preset(CFG_PATH, "clean"))


def run(model, scene, t_end, dt=0.01):
    """Push a gym-like scan every physics step; scene(t) -> 1080 ranges. Returns all scans."""
    scans = []
    for k in range(int(round(t_end / dt)) + 1):
        t = k / round(1 / dt)
        model.push(t, scene(t), GYM_AMIN, GYM_INC)
        scans += model.poll(t)
    return scans


def at(scan, deg, half_width_deg=3.0):
    """Finite readings of a scan within +-half_width of a ROS angle (deg, CCW +)."""
    ang = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
    d = np.abs((ang - math.radians(deg) + math.pi) % (2 * math.pi) - math.pi)
    v = scan.ranges[d < math.radians(half_width_deg)]
    return v[np.isfinite(v)]


def covered(scan, max_deg=130.0):
    ang = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
    wrapped = (ang + math.pi) % (2 * math.pi) - math.pi
    return scan.ranges[np.abs(wrapped) < math.radians(max_deg)]


def const(r):
    return lambda t: np.full(GYM_N, r)


def box_left(t):
    r = np.full(GYM_N, 4.0)
    r[np.abs(GYM_ANG - math.pi / 2) < math.radians(8)] = 1.0
    return r


# --------------------------------------------------------------------- format & timing
def test_output_matches_real_driver_format():
    s = run(D500Model(clean(), seed=0), const(2.0), 0.3)[0]
    n = len(s.ranges)
    assert s.ranges.dtype == np.float32
    assert 490 <= n <= 510                                     # 5000 Hz / 10 Hz
    assert s.angle_min == 0.0 and s.angle_max == pytest.approx(2 * math.pi)
    assert s.angle_increment == pytest.approx(2 * math.pi / (n - 1))
    assert s.stamp == pytest.approx(0.1)                       # stamped at END of revolution
    assert s.range_min == 0.02 and s.range_max == 25.0         # driver constants


def test_nothing_usable_before_revolution_plus_latency():
    m = D500Model(clean(latency_s=0.05), seed=0)
    for k in range(0, 15):
        m.push(k / 100, const(2.0)(0), GYM_AMIN, GYM_INC)
        assert m.read(k / 100) is None                          # t <= 0.14
    m.push(0.15, const(2.0)(0), GYM_AMIN, GYM_INC)
    s = m.read(0.15 + 1e-9)                                     # 0.1 + 0.05 > 0.15 in floats
    assert s is not None and s.available_at == pytest.approx(0.15)


def test_rate_about_10hz_with_jitter():
    scans = run(D500Model(clean(rate_jitter=0.01), seed=1), const(2.0), 10.0)
    assert 97 <= len(scans) <= 103
    assert [s.seq for s in scans] == list(range(1, len(scans) + 1))
    assert all(b.available_at >= a.available_at for a, b in zip(scans, scans[1:]))


def test_poll_returns_each_scan_once_and_read_keeps_latest():
    m = D500Model(clean(), seed=0)
    seen = run(m, const(2.0), 1.0)
    assert len({s.seq for s in seen}) == len(seen)
    assert m.read(1.0).seq == seen[-1].seq


def test_push_times_must_not_go_backwards():
    m = D500Model(clean(), seed=0)
    m.push(0.1, const(1.0)(0), GYM_AMIN, GYM_INC)
    with pytest.raises(ValueError):
        m.push(0.05, const(1.0)(0), GYM_AMIN, GYM_INC)


# --------------------------------------------------------------------- geometry
def test_rear_outside_gym_fov_is_nan_front_is_covered():
    s = run(D500Model(clean(), seed=0), const(2.0), 0.3)[0]
    assert np.isnan(at(s, 180.0, 20.0)).all() and at(s, 180.0, 20.0).size == 0
    front = covered(s)
    assert np.isfinite(front).mean() > 0.95                    # only driver binning holes
    assert np.allclose(front[np.isfinite(front)], 2.0, atol=1e-6)


@pytest.mark.parametrize("spin", ["cw", "ccw"])
def test_object_on_left_appears_at_plus_90_for_either_spin(spin):
    s = run(D500Model(clean(spin_direction=spin), seed=0), box_left, 0.3)[0]
    assert np.allclose(at(s, 90.0), 1.0, atol=1e-6)
    assert np.allclose(at(s, -90.0), 4.0, atol=1e-6)


def test_flip_mirrors_like_wrong_laser_scan_dir():
    s = run(D500Model(clean(flip=True), seed=0), box_left, 0.3)[0]
    assert np.allclose(at(s, -90.0), 1.0, atol=1e-6)


def test_mount_yaw_rotates_the_scene():
    # sensor zero points left: the box on the car's left is straight ahead of the sensor
    s = run(D500Model(clean(mount_yaw=math.pi / 2), seed=0), box_left, 0.3)[0]
    assert np.allclose(at(s, 0.0), 1.0, atol=1e-6)


@pytest.mark.parametrize("spin,sign", [("cw", 1.0), ("ccw", -1.0)])
def test_motion_skew_sign_follows_spin_order(spin, sign):
    # world changes at 10 m/s; one revolution is 0.1 s, so +90 and -90 deg are
    # measured half a revolution apart: |difference| ~= 10 * 0.05 = 0.5 m
    s = run(D500Model(clean(spin_direction=spin), seed=0), lambda t: np.full(GYM_N, 1.0 + 10 * t), 0.3)[0]
    diff = at(s, 90.0).mean() - at(s, -90.0).mean()
    assert sign * diff == pytest.approx(0.5, abs=0.12)        # 10 ms push quantization


# --------------------------------------------------------------------- noise & failures
@pytest.mark.parametrize("r,std", [(0.3, 0.010), (1.0, 0.020), (3.0, 0.030)])
def test_noise_follows_spec_bands(r, std):
    scans = run(D500Model(clean(noise_scale=1.0), seed=2), const(r), 3.0)
    v = np.concatenate([covered(s) for s in scans])
    v = v[np.isfinite(v)]
    assert np.std(v) == pytest.approx(std, rel=0.1)
    assert np.mean(v) == pytest.approx(r, abs=0.002)


def test_noise_already_in_gym_scan_is_not_double_counted():
    scans = run(D500Model(clean(noise_scale=1.0, base_noise_std=0.01), seed=3), const(3.0), 3.0)
    v = np.concatenate([covered(s) for s in scans])
    assert np.nanstd(v) == pytest.approx(math.sqrt(0.03 ** 2 - 0.01 ** 2), rel=0.1)


def test_dropout_fraction():
    scans = run(D500Model(clean(dropout_prob=0.2), seed=4), const(2.0), 2.0)
    v = np.concatenate([covered(s) for s in scans])
    assert np.isnan(v).mean() == pytest.approx(0.2, abs=0.03)


def test_beyond_visible_range_is_never_a_close_wall():
    scans = run(D500Model(clean(wall_visible_range=4.0, range_max=12.0), seed=5), const(6.0), 1.0)
    v = np.concatenate([covered(s) for s in scans])
    assert np.isnan(v).mean() > 0.3                            # some beams lost over the wall
    assert (v[np.isfinite(v)] >= 6.0).all()                   # the rest hit something further


def test_pitch_down_puts_floor_in_front_of_far_walls():
    cfg = clean(pitch_down_prob=1.0, pitch_down_floor_range=(2.0, 2.5), pitch_sector_deg=(60.0, 60.0))
    s = run(D500Model(cfg, seed=6), const(6.0), 0.3)[0]
    v = covered(s)
    floor = v[np.isfinite(v) & (v < 3.0)]
    assert floor.size > 0 and ((floor >= 2.0) & (floor <= 2.5)).all()


def test_self_hit_sector_reads_short_range():
    s = run(D500Model(clean(self_hit_sectors=((40.0, 50.0, 0.08),)), seed=7), const(2.0), 0.3)[0]
    assert np.allclose(at(s, 45.0, 3.0), 0.08, atol=0.05)


def test_same_seed_same_scans_different_seed_differs():
    cfg = load_preset(CFG_PATH, "nominal")
    a = run(D500Model(cfg, seed=11), box_left, 1.0)
    b = run(D500Model(cfg, seed=11), box_left, 1.0)
    c = run(D500Model(cfg, seed=12), box_left, 1.0)
    assert all(np.array_equal(x.ranges, y.ranges, equal_nan=True) for x, y in zip(a, b))
    assert not all(np.array_equal(x.ranges, y.ranges, equal_nan=True) for x, y in zip(a, c))


def test_reset_restarts_cleanly():
    m = D500Model(load_preset(CFG_PATH, "nominal"), seed=0)
    first = run(m, box_left, 0.5)
    m.reset(0.0, seed=0)
    again = run(m, box_left, 0.5)
    assert [s.seq for s in again] == [s.seq for s in first]
    assert all(np.array_equal(x.ranges, y.ranges, equal_nan=True) for x, y in zip(first, again))


# --------------------------------------------------------------------- config
def test_all_presets_load_and_drive():
    for name in load_sensor_yaml(CFG_PATH)["presets"]:
        scans = run(D500Model(load_preset(CFG_PATH, name), seed=0), box_left, 0.5)
        assert len(scans) >= 3


def test_randomization_stays_in_bounds():
    spec = load_sensor_yaml(CFG_PATH)["randomization"]
    rng = np.random.default_rng(0)
    base = load_preset(CFG_PATH, "nominal")
    spins = set()
    for _ in range(200):
        c = sample_config(spec, rng, base)
        spins.add(c.spin_direction)
        for k, v in spec.items():
            if isinstance(v, list):
                assert v[0] <= getattr(c, k) <= v[1]
        D500Model(c, seed=0)
    assert spins == {"cw", "ccw"}


def test_unknown_parameter_is_rejected():
    with pytest.raises(KeyError):
        config_from_dict({"rate_hzz": 10.0})
