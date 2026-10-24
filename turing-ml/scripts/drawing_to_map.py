"""Convert a track plan drawing (grey track, black wall outlines, white page) into a gym map.

The organisers' drawing is not an occupancy grid: the track is grey (gym main reads <= 128 as
wall), the page is white (read as free), and dimension lines and text are drawn in black. This
keeps only the track surface as free space, rescales to the real size, and writes PNG + yaml.

    python scripts/drawing_to_map.py turing_track.png maps/TuringTrack/TuringTrack_map \
        --size 14 10 --resolution 0.05 --origin -7 -5

--size is the track's outer extent in metres (the drawing's dimension lines). Optional
--start-px X Y marks the start/finish line in drawing pixels; a 2-point centre line file is
then written next to the map so tracks.racetrack() and eval_expert.py can start a car there.
"""
import argparse
import math
import pathlib

import numpy as np
from PIL import Image
from scipy import ndimage

DARK = 60          # outline / start line pixels are darker than this
BG = 235           # the page is lighter than this


def track_mask(gray: np.ndarray) -> np.ndarray:
    """Free space in drawing pixels: the grey track plus dark lines drawn across it."""
    light = gray >= DARK
    labels, _ = ndimage.label(light)
    sizes = np.bincount(labels.ravel())
    grey_frac = ndimage.mean((gray < BG).astype(float), labels, np.arange(len(sizes)))
    sizes[0] = 0
    sizes[grey_frac < 0.5] = 0                    # skip the white page and white islands
    track = labels == int(np.argmax(sizes))
    page = light & ~track & (gray >= BG)
    near_page = ndimage.binary_dilation(page, iterations=3)
    # dark pixels not touching the page lie on the track (start line), not on a wall outline
    across = ~light & ~near_page & ndimage.binary_dilation(track, iterations=3)
    return track | across


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drawing")
    ap.add_argument("out", help="output path without extension, e.g. maps/TuringTrack/TuringTrack_map")
    ap.add_argument("--size", type=float, nargs=2, required=True, metavar=("W_M", "H_M"))
    ap.add_argument("--resolution", type=float, default=0.05)
    ap.add_argument("--origin", type=float, nargs=2, metavar=("X", "Y"),
                    help="map bottom-left in metres (default: centre the track on 0,0)")
    ap.add_argument("--start-px", type=float, nargs=2, metavar=("X", "Y"),
                    help="a point on the start/finish line, in drawing pixels")
    ap.add_argument("--direction", choices=["ccw", "cw"], default="ccw",
                    help="driving direction around the track (sets the start heading)")
    a = ap.parse_args()

    gray = np.asarray(Image.open(a.drawing).convert("L")).astype(np.int16)
    free = track_mask(gray)
    walls = ~(gray >= DARK) & ndimage.binary_dilation(free, iterations=6) & ~free
    ys, xs = np.nonzero(free | walls)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    sx, sy = a.size[0] / (x1 - x0), a.size[1] / (y1 - y0)
    print(f"drawing extent {x1 - x0} x {y1 - y0} px -> {sx:.5f} / {sy:.5f} m per px "
          f"(aspect mismatch {abs(sx / sy - 1) * 100:.1f} %)")

    res = a.resolution
    w, h = int(round(a.size[0] / res)), int(round(a.size[1] / res))
    # min-pool: a map cell is free only if all drawing pixels in it are free (keeps thin walls closed)
    xe = np.linspace(x0, x1, w + 1).round().astype(int)
    ye = np.linspace(y0, y1, h + 1).round().astype(int)
    grid = np.zeros((h, w), bool)
    for i in range(h):
        rows = free[ye[i]:ye[i + 1]]
        for j in range(w):
            blk = rows[:, xe[j]:xe[j + 1]]
            grid[i, j] = blk.size > 0 and blk.all()
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = False

    origin = a.origin if a.origin else (-a.size[0] / 2, -a.size[1] / 2)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(grid, 255, 0).astype(np.uint8)).save(str(out) + ".png")
    out.with_suffix(".yaml").write_text(
        f"image: {out.name}.png\nresolution: {res}\norigin: [{origin[0]}, {origin[1]}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    print(f"wrote {out}.png ({w} x {h} px, {grid.mean() * 100:.0f} % free) and {out}.yaml")

    if a.start_px:
        _write_start(a, grid, res, origin, out, (x0, y0), (sx, sy))


def _write_start(a, grid, res, origin, out, xy0, scale):
    """Start pose on the start line, centred across the track, heading along it."""
    h, w = grid.shape
    j = int((a.start_px[0] - xy0[0]) * scale[0] / res)
    i = int((a.start_px[1] - xy0[1]) * scale[1] / res)
    dist = ndimage.distance_transform_edt(grid)
    win = dist[max(i - 30, 0):i + 31, max(j - 30, 0):j + 31]       # nearest centre-of-track cell
    ii, jj = np.unravel_index(np.argmax(win), win.shape)
    i, j = ii + max(i - 30, 0), jj + max(j - 30, 0)
    # track direction = along the ridge of the distance map; take the larger in-track extent
    best = max(np.linspace(0, math.pi, 36, endpoint=False),
               key=lambda t: sum(grid[int(i - k * math.sin(t)), int(j + k * math.cos(t))]
                                 for k in range(-20, 21)
                                 if 0 <= int(i - k * math.sin(t)) < h and 0 <= int(j + k * math.cos(t)) < w))
    x = origin[0] + (j + 0.5) * res
    y = origin[1] + (h - 1 - i + 0.5) * res
    # ccw around the track: on the bottom side that means driving +x; pick the sign by which
    # side of the track centre we are on
    cy = origin[1] + h * res / 2
    cx = origin[0] + w * res / 2
    d = np.array([math.cos(best), math.sin(best)])
    to_c = np.array([cx - x, cy - y])
    ccw_dir = np.array([to_c[1], -to_c[0]])                          # ccw tangent at this point
    if np.dot(d, ccw_dir) < 0:
        d = -d
    if a.direction == "cw":
        d = -d
    name = out.name[:-4] if out.name.endswith("_map") else out.name
    cl = out.parent / f"{name}_centerline.csv"
    half = float(dist[i, j] * res)
    cl.write_text("# x_m, y_m, w_tr_right_m, w_tr_left_m\n"
                  f"{x:.3f}, {y:.3f}, {half:.2f}, {half:.2f}\n"
                  f"{x + 1.5 * d[0]:.3f}, {y + 1.5 * d[1]:.3f}, {half:.2f}, {half:.2f}\n")
    print(f"start pose x={x:.2f} y={y:.2f} yaw={math.atan2(d[1], d[0]):.3f} ({a.direction}); wrote {cl}")


if __name__ == "__main__":
    main()
