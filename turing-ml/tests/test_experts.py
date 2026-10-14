"""Disparity extender on synthetic grids (no gym needed). Reasons: docs/DECISIONS.md D-31..D-36."""
import dataclasses
import math
import pathlib

import numpy as np
import pytest

from turing_ml.experts import DisparityConfig, disparity_extender, load_expert_config
from turing_ml.sensor import GRID, RMAX

CFG = load_expert_config(str(pathlib.Path(__file__).resolve().parents[1] / "configs" / "expert.yaml"))
DEG = math.radians(1.0)


def walls(left, right, front=None):
    """Grid for a car parallel to two straight walls `left` / `right` metres away
    (optionally a wall across the track `front` metres ahead), capped at RMAX."""
    s = np.sin(GRID)
    with np.errstate(divide="ignore"):
        r = np.where(s > 0, left / s, np.where(s < 0, right / -s, np.inf))
    if front is not None:
        c = np.cos(GRID)
        with np.errstate(divide="ignore"):
            r = np.minimum(r, np.where(c > 0, front / c, np.inf))
    return np.minimum(r, RMAX).astype(np.float32)


def ray_circle(cx, cy, rad, outside):
    """Range along each grid beam from the origin to a circle; first hit going outward when
    the origin is outside the circle, the exit point when inside."""
    d = np.stack([np.cos(GRID), np.sin(GRID)], 1).astype(np.float64)
    b = d @ np.array([cx, cy])
    disc = b ** 2 - (cx ** 2 + cy ** 2 - rad ** 2)
    out = np.full(len(GRID), np.inf)
    ok = disc >= 0
    t = b[ok] - np.sqrt(disc[ok]) if outside else b[ok] + np.sqrt(disc[ok])
    out[ok] = np.where(t > 0, t, np.inf)
    return out


def bend(radius=6.0, width=2.2, offset=0.0):
    """Left-hand bend of constant radius (track centre line), car on the centre line + offset
    (positive = toward the inside), heading along the track. Returns (grid, inner wall points)."""
    cy = radius                                         # bend centre is to the left
    inner = ray_circle(0.0, cy - offset, radius - width / 2, outside=True)
    outer = ray_circle(0.0, cy - offset, radius + width / 2, outside=False)
    r = np.minimum(np.minimum(inner, outer), RMAX).astype(np.float32)
    hit = np.isfinite(inner) & (inner < RMAX)
    pts = np.stack([inner[hit] * np.cos(GRID[hit]), inner[hit] * np.sin(GRID[hit])], 1)
    return r, pts


def test_straight_corridor_goes_straight():
    steer, v = disparity_extender(walls(1.1, 1.1), 2.0, CFG)
    assert abs(steer) <= 1.5 * DEG
    assert v > CFG.v_min


def test_wall_close_on_the_left_steers_right():
    steer, _ = disparity_extender(walls(0.5, 1.7), 2.0, CFG)
    assert steer < -2 * DEG


def test_wall_close_on_the_right_steers_left():
    steer, _ = disparity_extender(walls(1.7, 0.5), 2.0, CFG)
    assert steer > 2 * DEG


def test_wall_dead_ahead_stops():
    _, v = disparity_extender(walls(1.1, 1.1, front=0.4), 1.0, CFG)
    assert v == 0.0


def test_wall_ahead_slows_with_stopping_distance():
    """D-34: at 1.5 m to a wall the speed must allow stopping inside what we see."""
    _, v = disparity_extender(walls(1.1, 1.1, front=1.5), 3.0, CFG)
    usable = 1.5 - CFG.stop_dist
    a, t = CFG.brake_decel, CFG.latency
    assert v <= -a * t + math.sqrt((a * t) ** 2 + 2 * a * usable) + 1e-6
    assert v < CFG.v_max


@pytest.mark.parametrize("seed", range(5))
def test_mirrored_grid_gives_mirrored_steer(seed):
    rng = np.random.default_rng(seed)
    left, right = rng.uniform(0.6, 2.0, 2)
    g = walls(left, right)
    g = np.minimum(g, RMAX * (1 - 0.3 * (np.abs(GRID) > math.radians(100)))).astype(np.float32)
    s1, v1 = disparity_extender(g, 1.0, CFG)
    s2, v2 = disparity_extender(g[::-1].copy(), 1.0, CFG)
    assert abs(s1 + s2) <= 1.01 * DEG                  # run centre rounds to a whole beam
    assert abs(v1 - v2) < 0.1


@pytest.mark.parametrize("offset", [0.0, 0.3])
def test_bend_heading_keeps_bubble_clearance_from_the_inside_wall(offset):
    """D-36: the inside of a smooth bend has no disparity. The chosen heading ray must pass
    the inner wall by about bubble_width, over the stretch the extender judged free."""
    g, pts = bend(offset=offset)
    steer, _ = disparity_extender(g, 2.0, CFG)
    assert abs(steer) < CFG.steer_max                   # not clipped, so steer = chosen heading
    d = np.array([math.cos(steer), math.sin(steer)])
    along = pts @ d
    free = float(g[int(np.argmin(np.abs(GRID - steer)))])
    near = pts[(along > 0) & (along < free)]
    clearance = np.min(np.abs(near @ np.array([-d[1], d[0]])))
    assert clearance >= 0.30, clearance                 # bubble 0.35 at ~1 m -> >= 0.33 m
    # NB: a gentle bend seen through a 4 m cap rarely discriminates the bubble (the far 4 m run
    # decides); the tight bend below does.


def inner_clearance(g, pts, cfg):
    steer, _ = disparity_extender(g, 2.0, cfg)
    d = np.array([math.cos(steer), math.sin(steer)])
    along = pts @ d
    free = float(g[int(np.argmin(np.abs(GRID - steer)))])
    near = pts[(along > 0) & (along < free)]
    return float(np.min(np.abs(near @ np.array([-d[1], d[0]])))) if len(near) else np.inf


def test_bubble_widens_the_line_past_a_tight_inside_wall():
    """D-36: tight left bend (centre radius 2.5 m, 1.5 m wide). Without the bubble the heading
    grazes the inner wall (~0.55 m clear); with it the car keeps ~0.75 m."""
    g, pts = bend(radius=2.5, width=1.5)
    on = inner_clearance(g, pts, CFG)
    off = inner_clearance(g, pts, dataclasses.replace(CFG, bubble_width=0.0))
    assert on >= off + 0.1, (on, off)


@pytest.mark.parametrize("post_deg", [-10.0, 0.0, 8.0])
def test_bubble_keeps_heading_clear_of_a_lone_close_point(post_deg):
    """D-36: every point is padded, not only edges. A post 1 m ahead (3 beams wide) must be
    passed by about the bubble (0.35 m)."""
    g = walls(2.5, 2.5)
    k = int(np.argmin(np.abs(GRID - math.radians(post_deg))))
    g[k - 1:k + 2] = 1.0

    def gap(bw):
        steer, _ = disparity_extender(g, 2.0, dataclasses.replace(CFG, bubble_width=bw))
        return 1.0 * abs(math.sin(steer - math.radians(post_deg)))
    assert gap(0.35) >= 0.95 * 0.35 / math.sqrt(1 + 0.35 ** 2)
    assert gap(CFG.bubble_width) >= 0.30


def test_speed_never_exceeds_v_max_and_steer_is_bounded():
    rng = np.random.default_rng(0)
    for k in range(300):
        cfg = dataclasses.replace(CFG, v_max=float(rng.uniform(0.8, 6.0)))
        g = rng.uniform(0.05, RMAX, len(GRID)).astype(np.float32)
        if k % 3 == 0:
            g[:] = RMAX                                  # wide open: the case that wants to go fastest
        steer, v = disparity_extender(g, float(rng.uniform(0, 8)), cfg)
        assert 0.0 <= v <= cfg.v_max + 1e-9
        assert abs(steer) <= cfg.steer_max + 1e-9


def side_case():
    """Open gap ahead-left, everything else 1.5 m, and a wall right beside the car on the left."""
    g = np.full(len(GRID), 1.5, np.float32)
    g[(GRID > math.radians(5)) & (GRID < math.radians(60))] = RMAX
    g[(GRID >= math.radians(85)) & (GRID <= math.radians(100))] = 0.4
    return g


def test_side_clearance_blocks_turning_into_a_wall_beside_the_car():
    """D-33: the gap is to the left, but a wall 0.4 m beside the car on the left -> don't turn left."""
    g = side_case()
    free_steer, _ = disparity_extender(g, 1.0, dataclasses.replace(CFG, side_clear=0.0))
    assert free_steer > 0                                # without the rule it would turn left
    steer, _ = disparity_extender(g, 1.0, CFG)
    assert steer == 0.0


def test_side_clearance_allows_turning_away_from_the_side_wall():
    g = side_case()[::-1].copy()                        # mirror: gap right, wall right -> blocked
    assert disparity_extender(g, 1.0, CFG)[0] == 0.0
    g2 = side_case()
    g2[(GRID >= math.radians(85)) & (GRID <= math.radians(100))] = 1.5   # wall gone -> turns left
    assert disparity_extender(g2, 1.0, CFG)[0] > 0


def test_config_file_has_no_unknown_keys_and_no_15_cap(tmp_path):
    """D-31: everything comes from the yaml, a typo fails loudly, v_max is not the manual's 1.5 m/s."""
    assert CFG.v_max > 1.5
    assert isinstance(CFG, DisparityConfig)
    bad = tmp_path / "expert.yaml"
    bad.write_text("\n".join(["disparity:", "  v_max: 2.0", "  bubble_widht: 0.3", ""]))
    with pytest.raises(KeyError, match="bubble_widht"):
        load_expert_config(str(bad))


def test_yaw_compensation_rotates_the_grid_the_right_way():
    """D-40: turning left (yaw_rate > 0) since the scan means walls are now further right in the car
    frame, so the chosen heading moves right; the reverse for turning right; off when dt = 0."""
    g = walls(0.8, 1.4)
    cfg = dataclasses.replace(CFG, yaw_comp_dt=0.15)
    s0 = disparity_extender(g, 1.5, cfg, yaw_rate=0.0)[0]
    left = disparity_extender(g, 1.5, cfg, yaw_rate=1.0)[0]
    right = disparity_extender(g, 1.5, cfg, yaw_rate=-1.0)[0]
    assert left < s0 < right
    assert abs((s0 - left) - 0.15) <= 2 * DEG                # shifted by yaw_rate * dt
    off = dataclasses.replace(CFG, yaw_comp_dt=0.0)
    assert disparity_extender(g, 1.5, off, yaw_rate=1.0)[0] == disparity_extender(g, 1.5, off)[0]
