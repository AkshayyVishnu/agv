"""Drive the real car with the disparity extender (lidar only: no map, no pose, no IMU).

Runs in the `car` container on the Pi (ROS 2 /scan from the D500 driver; servo + VESC as in
teleop_keyboard.py). Start it through scripts/drive_car.sh, which starts the lidar first:

    ssh -t pi ./drive_car.sh                        # 1.0 m/s cap
    ssh -t pi ./drive_car.sh --max-speed 1.5 --set stop_dist=0.3

Keys:   hold G   drive autonomously (release: stop within DEADMAN_S)     SPACE  stop
        Q        quit (stop, centre, release)
The car only moves while G is held, so a person must watch it. It also stops when no scan arrives
for SCAN_TIMEOUT, when the battery is low, and when the program ends for any reason.

--auto (no keyboard; started by scripts/autodrive.sh when the car's power switch turns the VESC on):
checks the VESC and the lidar, wiggles the steering left-right as a warning, waits ARM_DELAY, then
drives until the VESC disappears (car switched off), --run-time runs out, or it is stopped. The
car's power switch is the kill switch: switching off cuts the motor in hardware.

Driver settings: configs/expert.yaml, then CAR_SETTINGS (what passed TuringTrack in sim at 1 m/s,
D-47), then --set. Speed (m/s) becomes VESC duty through MPS_PER_DUTY (open loop, from the mapping
drive). Every decision is logged to $DRIVE_LOGS (drive_car.sh: ~/car_ws/drive_logs on the Pi) (time, steer, speed cmd, duty, measured speed, grid).
"""
import argparse
import curses
import dataclasses
import math
import os
import pathlib
import signal
import sys
import threading
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from teleop_keyboard import (LOW_V, MPS_PER_DUTY, RAMP_PER_S, SPEED_MODES, TACHO_M, Servo,  # noqa: E402
                             Vesc, find_vesc)
from turing_ml.experts import disparity_extender, load_expert_config                          # noqa: E402
from turing_ml.sensor import to_grid                                                          # noqa: E402

WHEELBASE = 0.28
DEADMAN_S = 0.8
SCAN_TIMEOUT = 0.3
RATE_HZ = 20                       # VESC commands (it times out without them); decisions run per scan (10 Hz)
ARM_DELAY = 5.0                    # s between the warning wiggle and moving (--auto)
VESC_LOST_S = 1.0                  # no VESC reply for this long: it was switched off (--auto)
CAR_SETTINGS = dict(stop_dist=0.25, side_clear=0.25, steer_max=0.5)   # D-47 + measured lock 0.506 rad


class ScanSource:
    """Latest /scan from ROS 2, received on a background thread."""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan
        rclpy.init()
        self.node = Node("drive_car")
        self.lock, self.msg, self.t = threading.Lock(), None, 0.0
        self.node.create_subscription(LaserScan, "/scan", self._cb, qos_profile_sensor_data)
        self.th = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.th.start()

    def _cb(self, m):
        with self.lock:
            self.msg, self.t = m, time.time()

    def latest(self):
        with self.lock:
            return self.msg, self.t


class Controller:
    """One control step: poll the VESC, decide on each new scan, ramp the motor. Shared by the
    keyboard (hold G) and --auto modes; `go` says whether driving is allowed right now."""

    def __init__(self, a, cfg, vesc, servo, scans, log):
        self.cfg, self.vesc, self.servo, self.scans, self.log = cfg, vesc, servo, scans, log
        self.max_duty = min(a.max_speed / MPS_PER_DUTY, SPEED_MODES[-1])
        self.d = self.steer = self.v_cmd = self.speed = 0.0
        self.vals, self.last_tacho, self.next_poll, self.last_scan_t = None, None, 0.0, 0.0
        self.last_reply, self.n_dec, self.fresh, self.low, self.driving = time.time(), 0, False, False, False

    def step(self, now, go):
        if now >= self.next_poll:
            self.vals, self.next_poll = self.vesc.values(), now + 0.1
            if self.vals:
                self.last_reply = now
                if self.last_tacho:
                    self.speed = (self.vals["tacho"] - self.last_tacho[1]) * TACHO_M / max(now - self.last_tacho[0], 1e-3)
                self.last_tacho = (now, self.vals["tacho"])

        m, t_scan = self.scans.latest()
        self.fresh = m is not None and now - t_scan < SCAN_TIMEOUT
        if self.fresh and t_scan != self.last_scan_t:          # one decision per new scan
            self.last_scan_t = t_scan
            grid = to_grid(m.ranges, m.angle_min, m.angle_increment)
            yaw_rate = self.speed * math.tan(self.steer) / WHEELBASE   # no IMU: kinematic estimate
            self.steer, self.v_cmd = disparity_extender(grid, max(self.speed, 0.0), self.cfg, yaw_rate)
            self.servo.set(self.steer)
            self.n_dec += 1
            v = self.vals
            self.log.write(f"{now:.3f},{self.steer:.3f},{self.v_cmd:.3f},{self.d:.4f},{self.speed:.3f},"
                           f"{v['tacho'] if v else ''}," + " ".join(f"{x:.2f}" for x in grid[::5]) + "\n")

        self.low = self.vals is not None and self.vals["v_in"] < LOW_V
        self.driving = go and self.fresh and not self.low
        target = 0.0
        if self.driving and self.v_cmd > 0.05:
            target = max(min(self.v_cmd / MPS_PER_DUTY, self.max_duty), SPEED_MODES[0])   # 3 % is the least that moves it
        self.d = target if target < self.d else self.d + min(RAMP_PER_S / RATE_HZ, target - self.d)
        if self.d == 0.0:
            self.vesc.brake(2.0) if self.vals and abs(self.vals["erpm"]) > 300 else self.vesc.duty(0.0)
        else:
            self.vesc.duty(self.d)


def run(scr, a, cfg, vesc, servo, scans, log):
    curses.curs_set(0)
    scr.nodelay(True)
    c = Controller(a, cfg, vesc, servo, scans, log)
    go_until, msg = 0.0, ""
    while True:
        now = time.time()
        k = scr.getch()
        while k != -1:
            ch = chr(k).lower() if 0 <= k < 256 else ""
            if ch == "q":
                return
            if ch == " ":
                go_until, msg = 0.0, "STOP"
            elif ch == "g":
                go_until, msg = now + DEADMAN_S, ""
            k = scr.getch()
        c.step(now, now < go_until)
        if c.low:
            msg = f"BATTERY LOW ({c.vals['v_in']:.1f} V < {LOW_V}): driving disabled"
        elif now < go_until and not c.fresh:
            msg = "NO LIDAR SCANS: stopped"

        vals = c.vals
        scr.erase()
        scr.addstr(0, 0, "SBK10 disparity extender   HOLD G = drive   SPACE stop   Q quit")
        scr.addstr(2, 0, f"{'DRIVING' if c.driving else 'stopped'}   steer {c.steer:+.2f} rad   speed cmd {c.v_cmd:.2f} m/s "
                         f"(cap {a.max_speed:.1f})   duty {c.d:.3f}   measured {c.speed:+.2f} m/s")
        scr.addstr(3, 0, f"scans {'OK' if c.fresh else 'NONE'}  decisions {c.n_dec}  "
                         + (f"battery {vals['v_in']:.1f} V  fault {vals['fault']}" if vals else "VESC: no reply"))
        scr.addstr(4, 0, f"settings: stop_dist {cfg.stop_dist} side_clear {cfg.side_clear} steer_max {cfg.steer_max} "
                         f"bubble {cfg.bubble_width} fov {cfg.fov_deg}")
        scr.addstr(6, 0, msg[:100])
        scr.refresh()
        time.sleep(max(0.0, 1.0 / RATE_HZ - (time.time() - now)))


def run_auto(a, cfg, vesc, servo, scans, log):
    """No keyboard. Returns the reason it stopped."""
    def say(text):
        print(time.strftime("%H:%M:%S"), text, flush=True)

    def hold(seconds):                                    # keep the VESC fed (stopped) while waiting
        end = time.time() + seconds
        while time.time() < end:
            vesc.duty(0.0)
            time.sleep(1.0 / RATE_HZ)

    def scans_ok():
        m, t = scans.latest()
        return m is not None and time.time() - t < SCAN_TIMEOUT

    fw = vesc.fw()
    if not fw:
        return "VESC does not answer"
    say(f"VESC {fw}; waiting for lidar scans")
    t_end = time.time() + 20
    while time.time() < t_end and not scans_ok():
        hold(0.2)
    if not scans_ok():
        return "no lidar scans within 20 s"
    say(f"lidar OK; warning wiggle, driving in {ARM_DELAY:.0f} s")
    for st in (0.3, -0.3, 0.3, -0.3, 0.0):
        servo.set(st)
        hold(0.35)
    hold(ARM_DELAY)

    c = Controller(a, cfg, vesc, servo, scans, log)
    t0, next_say = time.time(), 0.0
    say(f"DRIVING (cap {a.max_speed} m/s)")
    while True:
        now = time.time()
        if a.run_time and now - t0 > a.run_time:
            return f"run time {a.run_time:.0f} s reached"
        try:
            c.step(now, True)
        except OSError as e:                              # USB gone: the car was switched off
            return f"VESC disconnected ({e.strerror})"
        if now - c.last_reply > VESC_LOST_S:
            return "VESC stopped answering (switched off?)"
        if now >= next_say:
            v = c.vals
            say(f"{'driving' if c.driving else 'STOPPED'} steer {c.steer:+.2f} v_cmd {c.v_cmd:.2f} meas {c.speed:+.2f} m/s "
                + (f"bat {v['v_in']:.1f} V" if v else "") + ("" if c.fresh else " NO SCANS") + (" LOW BATTERY" if c.low else ""))
            next_say = now + 1.0
        time.sleep(max(0.0, 1.0 / RATE_HZ - (time.time() - now)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-speed", type=float, default=1.0, help="m/s cap (default 1.0; max 2.9)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override an expert.yaml value")
    ap.add_argument("--port", help="VESC serial port (default: auto)")
    ap.add_argument("--auto", action="store_true", help="no keyboard: wiggle, wait, drive until the VESC goes away")
    ap.add_argument("--run-time", type=float, default=0.0, help="--auto: stop after this many seconds (0 = no limit)")
    a = ap.parse_args()
    if not 0 < a.max_speed <= SPEED_MODES[-1] * MPS_PER_DUTY:
        sys.exit(f"--max-speed must be in (0, {SPEED_MODES[-1] * MPS_PER_DUTY:.1f}]")

    cfg = load_expert_config(str(ROOT / "configs" / "expert.yaml"))
    over = dict(CAR_SETTINGS, **{k: float(v) for k, v in (kv.split("=", 1) for kv in a.set)})
    cfg = dataclasses.replace(cfg, **over, v_max=min(cfg.v_max, a.max_speed))

    port = find_vesc(a.port)
    if not port:
        sys.exit("VESC not found (no /dev/ttyACM*)")
    vesc, servo, scans = Vesc(port), Servo(), ScanSource()
    logdir = pathlib.Path(os.environ.get("DRIVE_LOGS", "~/drive_logs")).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    path = logdir / f"drive_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    log = open(path, "w")
    # epoch_s / steer_rad / tacho named as in teleop_keyboard.py's log: either works as mapping odometry
    log.write("epoch_s,steer_rad,v_cmd,duty,v_meas,tacho,grid_every_5deg_from_-135\n")
    def shutdown(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGHUP, shutdown)
    reason = "quit"
    try:
        if a.auto:
            reason = run_auto(a, cfg, vesc, servo, scans, log)
        else:
            curses.wrapper(run, a, cfg, vesc, servo, scans, log)
    finally:
        for _ in range(5):
            try:
                vesc.duty(0.0)
            except OSError:
                break                                        # VESC already gone (switched off)
            time.sleep(0.02)
        servo.release()
        log.close()
        print(f"stopped ({reason}); motor off, steering centred and released. log: {path}", flush=True)
        os._exit(0)                                          # don't wait for the ROS spin thread


if __name__ == "__main__":
    main()
