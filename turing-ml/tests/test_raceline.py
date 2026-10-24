"""Racing line (turing_ml.raceline) and pure pursuit (turing_ml.pursuit) on synthetic maps."""
import math
import pathlib

import numpy as np

from turing_ml import raceline as rl
from turing_ml import testmaps
from turing_ml.pursuit import PurePursuit

CAR = {"lf": 0.1375, "lr": 0.1375, "s_max": 0.4189}
PCFG = {"lookahead_min": 0.4, "lookahead_max": 1.0, "lookahead_time": 0.25, "speed_preview": 0.25}


def ring_track(tmp_path):
    path, (cx, cy) = testmaps.ring(radius=5.0, width=2.0, outdir=str(tmp_path))
    return rl.load_map(path + ".yaml"), (cx, cy)


def test_centre_line_of_a_ring_closes_on_the_middle(tmp_path):
    track, (cx, cy) = ring_track(tmp_path)
    centre = rl.trace_centreline(track, (cx, cy - 5.0, 0.0))
    length = len(centre) * 0.1
    assert abs(length - 2 * math.pi * 5.0) < 0.03 * 2 * math.pi * 5.0, length
    r = np.hypot(centre[:, 0] - cx, centre[:, 1] - cy)
    assert np.abs(r - 5.0).max() < 0.06, np.abs(r - 5.0).max()


def test_min_curvature_line_stays_inside_the_margin(tmp_path):
    track, (cx, cy) = ring_track(tmp_path)
    centre = rl.trace_centreline(track, (cx, cy - 5.0, 0.0))
    line = rl.min_curvature(track, centre, half_car=0.125, margin=0.2)
    assert track.clearance(line).min() >= 0.325 - 0.03
    # a ring is already a constant bend: the optimised line must not be much tighter than it
    assert np.abs(rl.curvature(line)).max() < 1.3 / 5.0


def test_speed_profile_on_a_circle_is_the_lateral_limit():
    th = np.linspace(0, 2 * math.pi, 300, endpoint=False)
    circle = np.stack([2.0 * np.cos(th), 2.0 * np.sin(th)], 1)
    prof = rl.speed_profile(circle, a_lat=4.5, a_accel=5.0, a_brake=5.0, v_max=10.0)
    assert np.allclose(prof["v"], math.sqrt(4.5 * 2.0), rtol=0.02)
    assert abs(prof["length"] - 2 * math.pi * 2.0) < 0.05


def straight_line(v=2.0):
    x = np.arange(0.0, 50.0, 0.1)
    return {"x": x, "y": np.zeros_like(x), "v": np.full_like(x, v)}


def test_pursuit_steers_back_to_the_line():
    drv = PurePursuit(straight_line(), PCFG, CAR)
    steer_left_of_line, _ = drv((5.0 + CAR["lr"], 0.3, 0.0), 2.0)
    drv = PurePursuit(straight_line(), PCFG, CAR)
    steer_right_of_line, _ = drv((5.0 + CAR["lr"], -0.3, 0.0), 2.0)
    assert steer_left_of_line < -0.05 and steer_right_of_line > 0.05


def test_pursuit_prediction_undoes_pose_delay():
    """Driving straight along the line at 2 m/s with the pose 0.1 s late: with prediction the
    driver steers like with a fresh pose; the line is offset so a lag would show up."""
    line = straight_line()
    line["y"] = np.where(line["x"] > 6.1, 0.2, 0.0)          # a 0.2 m step: a fresh pose sees it, a 0.1 s old one not
    cfg = dict(PCFG, pose_delay=0.1)
    fresh = PurePursuit(line, PCFG, CAR)
    late = PurePursuit(line, cfg, CAR)
    late_np = PurePursuit(line, dict(cfg, predict=False), CAR)
    for k in range(3):                                         # 50 ms steps along x
        t, x = 0.05 * k, 5.6 + 0.1 * k
        a_f = fresh((x, 0.0, 0.0), 2.0, t)
        a_l = late((x, 0.0, 0.0), 2.0, t)
        a_n = late_np((x, 0.0, 0.0), 2.0, t)
    assert abs(a_l[0] - a_f[0]) < 0.02, (a_l, a_f)
    assert abs(a_n[0] - a_f[0]) > abs(a_l[0] - a_f[0])


def test_turing_track_line_file_is_on_the_track():
    root = pathlib.Path(__file__).resolve().parents[1] / "maps" / "TuringTrack"
    if not (root / "TuringTrack_raceline.csv").exists():
        import pytest
        pytest.skip("racing line not built")
    track = rl.load_map(str(root / "TuringTrack_map.yaml"))
    line = rl.load_raceline(str(root / "TuringTrack_raceline.csv"))
    clear = track.clearance(np.stack([line["x"], line["y"]], 1))
    assert clear.min() > 0.25                                   # half car 0.125 + most of the margin
