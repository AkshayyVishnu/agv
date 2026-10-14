"""Scan -> fixed 271-beam grid. OWNED BY PERSON A.

PLACEHOLDER written by B (2026-10-11) so TuringEnv can run before A pushes the real file
(the manual's hand-off plan: "B builds against a stub"). It is the Execution Manual's
to_grid with ONE change, action A-1 in docs/DECISIONS.md: NaN (what the real D500 driver
publishes for "no return") is treated as unknown and interpolated, not as far/open space.
When A's version lands, replace this file and keep tests/test_env.py green.
"""
import numpy as np

RMAX = 4.0
GRID = np.radians(np.arange(-135, 136, 1.0)).astype(np.float32)   # 271 beams, 1 deg apart


def to_grid(ranges, angle_min, angle_inc, range_min=0.05, rmax=RMAX):
    """Any LaserScan -> 271 ranges on a fixed 1-degree grid, capped at rmax."""
    r = np.asarray(ranges, dtype=np.float32)
    ang = angle_min + np.arange(len(r), dtype=np.float32) * angle_inc
    ang = (ang + np.pi) % (2 * np.pi) - np.pi                     # wrap 0..2pi sensors to [-pi, pi)
    far = np.isposinf(r) | (np.isfinite(r) & (r > rmax))         # nothing within rmax
    bad = np.isnan(r) | (np.isfinite(r) & (r < range_min))       # no return: unknown (A-1)
    r = np.where(far, rmax, r)
    idx = np.round((ang - GRID[0]) / np.radians(1.0)).astype(int)
    keep = (idx >= 0) & (idx < len(GRID)) & ~bad
    out = np.full(len(GRID), np.nan, np.float32)
    np.fmin.at(out, idx[keep], r[keep])                           # min per bin keeps thin obstacles
    holes = np.isnan(out)
    if holes.any() and (~holes).any():                            # fill empty bins from neighbours
        out[holes] = np.interp(np.flatnonzero(holes), np.flatnonzero(~holes), out[~holes])
    return np.nan_to_num(out, nan=rmax)
