"""Tiny generated maps for tests and smoke checks (f1tenth_gym main format: PNG + yaml).
Bundled maps are bad test fixtures: from Levine's (0, 0, 0) the car leaves the map without
a collision ever being reported (DECISIONS D-16)."""
import pathlib
import tempfile

import numpy as np

RES = 0.05


def _write(img: np.ndarray, name: str, outdir=None) -> str:
    from PIL import Image
    out = pathlib.Path(outdir or tempfile.mkdtemp()) / name
    Image.fromarray(img).save(str(out) + ".png")
    out.with_suffix(".yaml").write_text(
        f"image: {name}.png\nresolution: {RES}\norigin: [0.0, 0.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")
    return str(out)          # path WITHOUT extension, as gym main's `map` argument expects


def corridor(length=40.0, width=4.0, wall=0.2, outdir=None) -> str:
    """Straight corridor closed at both ends. Centre line is y = width / 2."""
    w, h, k = int(length / RES), int(width / RES), int(wall / RES)
    img = np.full((h, w), 255, np.uint8)
    img[:k, :] = img[-k:, :] = img[:, :k] = img[:, -k:] = 0
    return _write(img, "corridor", outdir)


def ring(radius=5.0, width=2.0, margin=1.0, outdir=None):
    """Circular track (annulus). Returns (map_path, centre_xy). Centre-line radius = radius."""
    size = 2 * (radius + width / 2 + margin)
    n = int(size / RES)
    c = size / 2
    yy, xx = np.mgrid[0:n, 0:n]
    d = np.hypot((xx + 0.5) * RES - c, (yy + 0.5) * RES - c)
    img = np.where(np.abs(d - radius) <= width / 2, 255, 0).astype(np.uint8)
    return _write(img, "ring", outdir), (c, c)
