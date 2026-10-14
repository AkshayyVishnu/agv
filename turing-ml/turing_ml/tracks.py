"""Find maps and start poses. f1tenth_racetracks layout: <root>/<Name>/<Name>_map.{yaml,png}
plus <Name>_centerline.csv (x_m, y_m, w_tr_right_m, w_tr_left_m)."""
import math
import os
import pathlib
from typing import Optional, Tuple

import numpy as np

DEFAULT_ROOT = os.environ.get(
    "TURING_RACETRACKS",
    str(pathlib.Path(__file__).resolve().parents[2] / "f1tenth_racetracks"))


def racetrack(name: str, root: Optional[str] = None, start_index: int = 0,
              look_ahead_m: float = 1.0) -> Tuple[str, Tuple[float, float, float]]:
    """Returns (map path without extension, start pose). The heading points from the chosen
    centre-line point to the first point >= look_ahead_m further along (D-35)."""
    d = pathlib.Path(root or DEFAULT_ROOT) / name
    cl = np.loadtxt(d / f"{name}_centerline.csv", delimiter=",", comments="#")[:, :2]
    i = start_index % len(cl)
    j = i
    for k in range(1, len(cl)):
        j = (i + k) % len(cl)
        if np.hypot(*(cl[j] - cl[i])) >= look_ahead_m:
            break
    yaw = math.atan2(cl[j, 1] - cl[i, 1], cl[j, 0] - cl[i, 0])
    return str(d / f"{name}_map"), (float(cl[i, 0]), float(cl[i, 1]), yaw)


def list_racetracks(root: Optional[str] = None):
    r = pathlib.Path(root or DEFAULT_ROOT)
    return sorted(p.name for p in r.iterdir() if (p / f"{p.name}_centerline.csv").exists())
