import dataclasses, sys, itertools
from turing_ml import tracks, testmaps
from turing_ml.env import TuringEnv
from turing_ml.d500 import load_preset
from turing_ml.experts import disparity_extender, load_expert_config
from turing_ml.vehicle import load_vehicle, vehicle_lidar_dist
VEHICLE = load_vehicle("configs/vehicle.yaml", "sbk10")
def run(m, cfg, preset, seed=0):
    if m=="ring":
        p,(cx,cy)=testmaps.ring(); start=(cx,cy-5,0.0)
    else: p,start=tracks.racetrack(m)
    env=TuringEnv(p, sensor_cfg=load_preset("configs/sensor.yaml",preset), params=VEHICLE, lidar_dist=vehicle_lidar_dist("configs/vehicle.yaml", "sbk10"), max_laps=1, max_time=400, seed=seed)
    obs=env.reset(start, seed=seed); still=0
    while not(obs["done"] or obs["truncated"]):
        obs=env.step(disparity_extender(obs["grid"],obs["speed"],cfg,obs["yaw_rate"]))
        still = still+1 if obs["speed"]<0.05 and obs["t"]>2 else 0
        if still>40: break
    env.close()
    return f"{obs['lap_times'][0]:5.0f}s" if obs["laps"] else ("crash@%.0f"%obs["t"] if obs["collision"] else ("stuck@%.0f"%obs["t"] if still>40 else "timeout"))
base=load_expert_config("configs/expert.yaml")
maps=sys.argv[1].split(",")
for spec in sys.argv[2:]:
    kv=dict(x.split("=") for x in spec.split(";") if x and not x.startswith("preset"))
    pr=[x.split("=")[1] for x in spec.split(";") if x.startswith("preset")][0]
    c=dataclasses.replace(base, **{k:float(v) for k,v in kv.items()})
    print(f"{spec:42s} | "+" | ".join(f"{m[:5]}:{run(m,c,pr):>9s}" for m in maps), flush=True)
