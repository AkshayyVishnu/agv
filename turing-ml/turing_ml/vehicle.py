"""Vehicle physics presets (configs/vehicle.yaml) -> the full `params` dict f1tenth_gym main expects.

    params = load_vehicle("configs/vehicle.yaml", "sbk10")
    env = TuringEnv(map_path, params=params, ...)
"""
from typing import Dict, Optional

import yaml

BASE = "f1tenth_default"
KEYS = ("mu", "C_Sf", "C_Sr", "lf", "lr", "h", "m", "I", "s_min", "s_max", "sv_min", "sv_max",
        "v_switch", "a_max", "v_min", "v_max", "width", "length")


def load_vehicle(path: str, name: str = "sbk10", overrides: Optional[Dict] = None) -> Dict[str, float]:
    with open(path, "r") as f:
        presets = yaml.safe_load(f)["vehicle"]["presets"]
    if name not in presets:
        raise KeyError(f"no vehicle preset '{name}' in {path}; have {sorted(presets)}")
    params = {**presets[BASE], **presets[name], **(overrides or {})}
    params.pop("lidar_dist", None)                       # not a gym param: see vehicle_lidar_dist()
    missing = [k for k in KEYS if k not in params]
    unknown = [k for k in params if k not in KEYS]
    if missing or unknown:
        raise ValueError(f"vehicle '{name}': missing {missing}, unknown {unknown}")
    return {k: float(params[k]) for k in KEYS}


def vehicle_lidar_dist(path: str, name: str = "sbk10") -> float:
    """LiDAR position ahead of the gym pose point (m) for TuringEnv(lidar_dist=...). The gym also
    centres its collision box there (D-44)."""
    with open(path, "r") as f:
        presets = yaml.safe_load(f)["vehicle"]["presets"]
    return float({**presets[BASE], **presets[name]}["lidar_dist"])
