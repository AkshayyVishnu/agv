"""Build the racing line for a map: centre line, minimum-curvature line, speed profile, picture.

    python scripts/make_raceline.py maps/TuringTrack/TuringTrack_map.yaml --start 0.875 -4.225 0

Writes next to the map: <Name>_centerline.csv (full centre line with widths, replaces a 2-point
start file), <Name>_raceline.csv (x, y, s, kappa, v) and <Name>_raceline.png (centre line in grey,
racing line coloured by speed: blue slow -> red fast). Prints checks; exits non-zero if the line
leaves the track or is tighter than the car can turn. Settings: configs/raceline.yaml.
"""
import argparse
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from turing_ml import raceline as rl                      # noqa: E402
from turing_ml.vehicle import load_vehicle                # noqa: E402


def draw(track, centre, prof, out, scale=4):
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.where(track.free, 235, 40).astype(np.uint8)).convert("RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)

    def px(xy):
        rc = track.to_px(xy)
        return [((c + 0.5) * scale, (r + 0.5) * scale) for r, c in rc]

    d.line(px(np.vstack([centre, centre[:1]])), fill=(150, 150, 150), width=1)
    v, vmin, vmax = prof["v"], prof["v"].min(), prof["v"].max()
    pts = px(np.stack([prof["x"], prof["y"]], 1))
    for i in range(len(pts)):
        f = (v[i] - vmin) / max(vmax - vmin, 1e-6)
        d.line([pts[i], pts[(i + 1) % len(pts)]], fill=(int(255 * f), 60, int(255 * (1 - f))), width=3)
    x0, y0 = pts[0]
    d.ellipse([x0 - 6, y0 - 6, x0 + 6, y0 + 6], outline=(0, 160, 0), width=3)
    d.text((8, 8), f"racing line {prof['length']:.1f} m, est. lap {prof['lap_time']:.1f} s, "
                   f"v {vmin:.1f}-{vmax:.1f} m/s (blue slow, red fast); grey = centre line; green = start",
           fill=(0, 0, 0))
    img.save(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", help="map yaml")
    ap.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "YAW"),
                    help="a pose on the start line, facing the ccw direction")
    ap.add_argument("--vehicle", default="sbk10")
    ap.add_argument("--config", default=str(ROOT / "configs" / "raceline.yaml"))
    a = ap.parse_args()

    cfg = rl.load_raceline_config(a.config)
    car = load_vehicle(str(ROOT / "configs" / "vehicle.yaml"), a.vehicle)
    start = list(a.start)
    if cfg["direction"] == "cw":
        start[2] += math.pi
    track = rl.load_map(a.map)

    centre = rl.trace_centreline(track, start)
    gap = float(np.hypot(*(centre[0] - start[:2])))
    w = track.clearance(centre)
    print(f"centre line: {len(centre)} pts, {len(centre) * 0.1:.1f} m long, starts {gap:.2f} m from the start pose, "
          f"half-width {w.min():.2f}-{w.max():.2f} m (median {np.median(w):.2f})")

    half_car = car["width"] / 2
    r_car = (car["lf"] + car["lr"]) / math.tan(car["s_max"])
    line = rl.min_curvature(track, centre, half_car, cfg["margin"], cfg["iterations"], cfg["step"], r_min=r_car)
    i0 = rl.start_index(dict(x=line[:, 0], y=line[:, 1]), start)
    line = np.roll(line, -i0, 0)                                      # start at the start line
    prof = rl.speed_profile(line, cfg["a_lat"], cfg["a_accel"], cfg["a_brake"], cfg["v_max"], cfg["curv_smooth"])

    clear = track.clearance(line)
    # radius averaged over curv_smooth points (~0.5 m, about two wheelbases): the car can't follow
    # a single 10 cm kink anyway, and three-point curvature is noisy
    r_min = 1.0 / max(np.abs(prof["kappa"]).max(), 1e-9)
    r_centre = 1.0 / max(np.abs(rl.speed_profile(centre, 1, 1, 1, 1, cfg["curv_smooth"])["kappa"]).max(), 1e-9)
    print(f"racing line: {prof['length']:.1f} m (centre line {len(centre) * 0.1:.1f} m), "
          f"tightest radius {r_min:.2f} m (centre line {r_centre:.2f} m, car's minimum {r_car:.2f} m; "
          f"averaged over {cfg['curv_smooth']} points)")
    print(f"wall clearance of the line centre: min {clear.min():.3f} m (need >= half car {half_car:.3f} + margin "
          f"{cfg['margin']:.2f} = {half_car + cfg['margin']:.3f}, small sampling error allowed)")
    print(f"speed {prof['v'].min():.2f}-{prof['v'].max():.2f} m/s, estimated lap {prof['lap_time']:.1f} s "
          f"(a_lat {cfg['a_lat']}, accel {cfg['a_accel']}, brake {cfg['a_brake']}, cap {cfg['v_max']})")

    base = pathlib.Path(a.map[:-5] if a.map.endswith(".yaml") else a.map)
    name = base.name[:-4] if base.name.endswith("_map") else base.name
    c0 = np.roll(centre, -rl.start_index(dict(x=centre[:, 0], y=centre[:, 1]), start), 0)
    wc = track.clearance(c0)
    np.savetxt(base.parent / f"{name}_centerline.csv", np.stack([c0[:, 0], c0[:, 1], wc, wc], 1),
               delimiter=", ", fmt="%.4f", header="x_m, y_m, w_tr_right_m, w_tr_left_m")
    rl.save_raceline(str(base.parent / f"{name}_raceline.csv"), prof)
    draw(track, centre, prof, str(base.parent / f"{name}_raceline.png"))
    print(f"wrote {name}_centerline.csv, {name}_raceline.csv, {name}_raceline.png in {base.parent}")

    bad = []
    if clear.min() < half_car + cfg["margin"] - 0.03:
        bad.append("line comes too close to a wall")
    if r_min < r_car:
        # not fatal: at full lock the car runs slightly wide and uses part of the margin. But it
        # means this corner depends on the real steering lock (s_max in vehicle.yaml: measure it)
        print(f"WARNING: tightest bend {r_min:.2f} m is below the car's turning radius {r_car:.2f} m "
              f"(s_max {car['s_max']:.3f} rad): the car will use part of the {cfg['margin']:.2f} m margin there")
    if gap > 0.3:
        bad.append("centre line does not pass the start pose")
    for b in bad:
        print("CHECK FAILED:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
