"""Build a gym map from a hand-measured track plan: walls given in metres in a yaml file.

For tracks we only have a sketch with dimensions for (no CAD, no lidar map yet). The plan lists
walls as lines, elliptical arcs and filled boxes; everything reachable from the start pose is
track, the rest is wall. Edit the plan when better measurements come in and run this again.

    python scripts/plan_to_map.py maps/TuringTrack2/TuringTrack2_plan.yaml

Writes <Name>_map.png/.yaml and a 2-point <Name>_centerline.csv (start pose) next to the plan,
and prints the track width at every `check` in the plan next to the sketch's number. Then:

    python scripts/make_raceline.py maps/TuringTrack2/TuringTrack2_map.yaml --start X Y YAW
"""
import argparse
import math
import pathlib

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage


def wall_points(w: dict, step: float) -> np.ndarray:
    """Dense (N, 2) points along one wall entry."""
    if "line" in w:
        p = np.asarray(w["line"], float)
        segs = [np.linspace(a, b, max(2, int(np.hypot(*(b - a)) / step) + 1)) for a, b in zip(p[:-1], p[1:])]
        return np.vstack(segs)
    if "arc" in w:
        c, (rx, ry), (d0, d1) = np.asarray(w["arc"]["centre"], float), w["arc"]["radii"], w["arc"]["deg"]
        n = max(2, int(abs(math.radians(d1 - d0)) * max(rx, ry) / step) + 1)
        t = np.radians(np.linspace(d0, d1, n))
        return np.stack([c[0] + rx * np.cos(t), c[1] + ry * np.sin(t)], 1)
    raise ValueError(f"unknown wall entry {w}")


def width_at(free: np.ndarray, res: float, rc, reach: float = 4.0) -> float:
    """Narrowest free chord through a cell (m): the track width there."""
    best = np.inf
    for t in np.radians(np.arange(0, 180, 2)):
        run = 0
        for sgn in (1, -1):
            for k in range(1, int(reach / res)):
                r = int(round(rc[0] - sgn * k * math.sin(t)))
                c = int(round(rc[1] + sgn * k * math.cos(t)))
                if not (0 <= r < free.shape[0] and 0 <= c < free.shape[1]) or not free[r, c]:
                    break
                run += 1
        best = min(best, (run + 1) * res)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    a = ap.parse_args()

    plan_path = pathlib.Path(a.plan)
    plan = yaml.safe_load(plan_path.read_text())
    res, thick, pad = float(plan["resolution"]), float(plan["wall_thickness"]), 0.5
    # dense samples: every cell a wall crosses gets marked, so a one-cell wall has no diagonal leaks
    pts = [wall_points(w, res / 20) for w in plan["walls"] if "box" not in w]
    boxes = [np.asarray(w["box"], float) for w in plan["walls"] if "box" in w]
    allp = np.vstack(pts + [b.reshape(2, 2) for b in boxes])
    x0, y0 = allp.min(0) - pad
    x1, y1 = allp.max(0) + pad
    w, h = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))

    def to_rc(xy):
        xy = np.atleast_2d(xy)
        return np.stack([h - 1 - np.floor((xy[:, 1] - y0) / res), np.floor((xy[:, 0] - x0) / res)], 1).astype(int)

    wall = np.zeros((h, w), bool)
    for p in pts:
        rc = to_rc(p)
        wall[rc[:, 0], rc[:, 1]] = True
    if thick > res:                                   # cells crossed are already one cell (res) thick
        wall = ndimage.binary_dilation(wall, iterations=int(round((thick - res) / 2 / res)))
    for b in boxes:
        (r1, c0), (r0, c1) = to_rc(b[:2])[0], to_rc(b[2:])[0]
        wall[r0:r1 + 1, c0:c1 + 1] = True

    start = plan["start"]
    labels, _ = ndimage.label(~wall)
    s_rc = to_rc(start[:2])[0]
    if wall[s_rc[0], s_rc[1]]:
        raise SystemExit(f"start {start} is inside a wall")
    free = labels == labels[s_rc[0], s_rc[1]]
    if free[0, :].any() or free[:, 0].any() or free[-1, :].any() or free[:, -1].any():
        raise SystemExit("the track is not closed: free space reaches the map edge (a gap in the walls)")

    name = plan_path.stem[:-5] if plan_path.stem.endswith("_plan") else plan_path.stem
    out = plan_path.parent / f"{name}_map"
    origin = (round(float(x0), 3), round(float(y0), 3))
    Image.fromarray(np.where(free, 255, 0).astype(np.uint8)).save(str(out) + ".png")
    out.with_suffix(".yaml").write_text(
        f"image: {out.name}.png\nresolution: {res}\norigin: [{origin[0]}, {origin[1]}, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    print(f"wrote {out}.png ({w} x {h} px, {(x1 - x0):.2f} x {(y1 - y0):.2f} m, origin {origin}) and {out}.yaml")

    print("track width at the sketch's dimensions (measured narrowest chord vs sketch):")
    for c in plan.get("checks", []):
        rc = to_rc(c["at"])[0]
        got = width_at(free, res, rc) if free[rc[0], rc[1]] else float("nan")
        print(f"  {c['name']:<34} {got:5.2f} m   sketch {c['sketch']}")

    dist = ndimage.distance_transform_edt(free) * res
    half = float(dist[s_rc[0], s_rc[1]])
    x, y, yaw = start
    cl = plan_path.parent / f"{name}_centerline.csv"
    cl.write_text("# x_m, y_m, w_tr_right_m, w_tr_left_m\n"
                  f"{x:.3f}, {y:.3f}, {half:.2f}, {half:.2f}\n"
                  f"{x + 1.5 * math.cos(yaw):.3f}, {y + 1.5 * math.sin(yaw):.3f}, {half:.2f}, {half:.2f}\n")
    print(f"start pose x={x} y={y} yaw={yaw}; wrote {cl}")


if __name__ == "__main__":
    main()
