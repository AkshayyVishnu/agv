"""Keyboard driving for the SBK10 on the Pi 5: steering servo (GPIO12 hardware PWM) + VESC over USB.
No ROS. Run on the Pi in an SSH terminal (needs a real terminal: ssh -t):

    python3 teleop_keyboard.py --check           # read the VESC only (firmware, voltage, faults); motor never driven
    python3 teleop_keyboard.py --no-motor        # keys move the steering only
    python3 teleop_keyboard.py                   # drive (first time: wheels OFF the ground)

Keys:   hold W  forward        hold S  reverse         (release: the motor stops after DEADMAN_S)
        A / D   steer left / right in 0.1 rad steps    X  steering straight
        1 .. 9  speed mode (SPEED_MODES; starts at 1, ~0.35 m/s)
        SPACE  stop now      Q  quit (stop, centre, release)

Safety: duty is capped by --max-duty (default 0.25, mode 9) and rises at most RAMP_PER_S, so a fast
mode doesn't jerk the car or lift its nose; releasing W/S or SPACE stops at once. The motor stops when no W/S key arrives for DEADMAN_S, when the program ends for any
reason (Q, Ctrl+C, SSH dropped), and refuses to drive with the battery below LOW_V. Steering stays inside
the measured 1050-2100 us. Configure a timeout in VESC Tool as a second layer.
A CSV log (clock time to match a ros2 bag, commands, ERPM, voltage, current, tachometer) is written for calibration and odometry.
"""
import argparse
import curses
import glob
import os
import signal
import struct
import sys
import time

# --- steering: measured 2026-10-30 (configs/car_spec.yaml) -------------------------------------------
PWM = "/sys/class/pwm/pwmchip0/pwm0"          # GPIO12, dtoverlay=pwm,pin=12,func=4
CENTRE_US, LEFT_US, RIGHT_US = 1550, 2100, 1050
LEFT_RAD, RIGHT_RAD = 0.519, -0.506
STEER_STEP = 0.1

# --- motor -----------------------------------------------------------------------------------------
DEADMAN_S = 0.8                               # keyboard auto-repeat must arrive faster than this
LOW_V = 13.2                                  # 4S: 3.3 V/cell under load
RATE_HZ = 20
# duty per mode; m/s estimated from the mapping drive (3 % duty = 0.35 m/s, ~linear). The screen shows
# the measured speed from the tachometer (TACHO_M per count, fitted from that drive's lidar).
SPEED_MODES = [0.03, 0.045, 0.06, 0.085, 0.11, 0.14, 0.17, 0.21, 0.25]
MPS_PER_DUTY = 11.7
TACHO_M = 0.0017
RAMP_PER_S = 0.15                             # duty/s while speeding up (0 -> 0.25 in ~1.7 s)


def steer_to_us(rad):
    rad = max(RIGHT_RAD, min(LEFT_RAD, rad))
    if rad >= 0:
        return int(round(CENTRE_US + rad / LEFT_RAD * (LEFT_US - CENTRE_US)))
    return int(round(CENTRE_US + rad / RIGHT_RAD * (RIGHT_US - CENTRE_US)))


class Servo:
    def __init__(self):
        if not os.path.exists(PWM):
            with open(os.path.dirname(PWM) + "/export", "w") as f:
                f.write("0")
            time.sleep(0.2)
        self._w("period", 20_000_000)
        self.set(0.0)
        self._w("enable", 1)

    def _w(self, name, value):
        for _ in range(50):
            try:
                with open(f"{PWM}/{name}", "w") as f:
                    f.write(str(value))
                return
            except PermissionError:
                time.sleep(0.05)
        raise PermissionError(f"{PWM}/{name}")

    def set(self, rad):
        self.us = steer_to_us(rad)
        self._w("duty_cycle", self.us * 1000)

    def release(self):
        self.set(0.0)
        time.sleep(0.3)
        self._w("enable", 0)


# --- VESC serial protocol (short packets: 0x02, len, payload, crc16-xmodem, 0x03) ------------------
COMM_FW_VERSION, COMM_GET_VALUES, COMM_SET_DUTY, COMM_SET_CURRENT_BRAKE = 0, 4, 5, 7


def crc16(data):
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


class RawSerial:
    """Minimal stand-in for pyserial (the car container doesn't have it). A USB CDC port ignores baud."""

    def __init__(self, port, timeout):
        import termios
        import tty
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY)
        tty.setraw(self.fd)
        termios.tcflush(self.fd, termios.TCIOFLUSH)
        self.timeout = timeout

    def write(self, b):
        os.write(self.fd, b)

    def read(self, n):
        import select
        r, _, _ = select.select([self.fd], [], [], self.timeout)
        return os.read(self.fd, n) if r else b""

    def reset_input_buffer(self):
        import termios
        termios.tcflush(self.fd, termios.TCIFLUSH)


class Vesc:
    def __init__(self, port):
        try:
            import serial
            self.s = serial.Serial(port, 115200, timeout=0.05)
        except ImportError:
            self.s = RawSerial(port, 0.05)
        self.buf = b""

    def send(self, payload):
        self.s.write(bytes([2, len(payload)]) + payload + struct.pack(">H", crc16(payload)) + b"\x03")

    def request(self, cmd, timeout=0.3):
        self.s.reset_input_buffer()
        self.send(bytes([cmd]))
        end, buf = time.time() + timeout, b""
        while time.time() < end:
            buf += self.s.read(256)
            i = buf.find(b"\x02")
            if i >= 0 and len(buf) >= i + 2:
                n = buf[i + 1]
                if len(buf) >= i + 2 + n + 3:
                    p = buf[i + 2:i + 2 + n]
                    if struct.unpack(">H", buf[i + 2 + n:i + 4 + n])[0] == crc16(p) and p[:1] == bytes([cmd]):
                        return p[1:]
                    buf = buf[i + 1:]
        return None

    def duty(self, d):
        self.send(struct.pack(">Bi", COMM_SET_DUTY, int(d * 100000)))

    def brake(self, amps=3.0):
        self.send(struct.pack(">Bi", COMM_SET_CURRENT_BRAKE, int(amps * 1000)))

    def fw(self):
        p = self.request(COMM_FW_VERSION)
        if p is None:
            return None
        hw = p[2:].split(b"\x00")[0].decode(errors="replace")
        return f"{p[0]}.{p[1]:02d} {hw}"

    def values(self):
        p = self.request(COMM_GET_VALUES, 0.1)
        if p is None or len(p) < 53:
            return None
        f = struct.unpack(">hhiiiihihiiiiiiB", p[:53])
        return dict(t_fet=f[0] / 10, t_motor=f[1] / 10, i_motor=f[2] / 100, i_in=f[3] / 100,
                    duty=f[6] / 1000, erpm=f[7], v_in=f[8] / 10, tacho=f[13], fault=f[15])


def find_vesc(port):
    if port:
        return port
    ids = glob.glob("/dev/serial/by-id/*STMicro*") + glob.glob("/dev/serial/by-id/*VESC*") + glob.glob("/dev/ttyACM*")
    return ids[0] if ids else None


def check(port):
    v = Vesc(port)
    print("firmware:", v.fw() or "NO REPLY")
    for _ in range(3):
        print("values:", v.values() or "NO REPLY")
        time.sleep(0.2)


def run(scr, a, vesc, servo, log):
    curses.curs_set(0)
    scr.nodelay(True)
    steer, mode, drive_dir, last_drive_key = 0.0, 1, 0, 0.0       # mode 1 = 3 % duty, the lowest that starts the motor
    vals, msg, t0, next_poll = None, "", time.time(), 0.0
    d, speed, last_tacho = 0.0, 0.0, None
    while True:
        now = time.time()
        k = scr.getch()
        while k != -1:
            c = chr(k).lower() if 0 <= k < 256 else ""
            if c == "q":
                return
            if c == " ":
                drive_dir, msg = 0, "STOP"
            elif c == "w":
                drive_dir, last_drive_key = 1, now
            elif c == "s":
                drive_dir, last_drive_key = -1, now
            elif c == "a":
                steer = min(LEFT_RAD, steer + STEER_STEP)
            elif c == "d":
                steer = max(RIGHT_RAD, steer - STEER_STEP)
            elif c == "x":
                steer = 0.0
            elif c in "123456789":
                mode = int(c)
            k = scr.getch()
        if drive_dir and now - last_drive_key > DEADMAN_S:
            drive_dir = 0                              # key released: stop
        servo.set(steer)

        if vesc:
            if now >= next_poll:
                vals, next_poll = vesc.values(), now + 0.1
                if vals:
                    if last_tacho:
                        speed = (vals["tacho"] - last_tacho[1]) * TACHO_M / max(now - last_tacho[0], 1e-3)
                    last_tacho = (now, vals["tacho"])
            low = vals is not None and vals["v_in"] < LOW_V
            target = 0.0
            if drive_dir and not low:
                target = drive_dir * min(SPEED_MODES[mode - 1], a.max_duty) * (1.0 if drive_dir > 0 else 0.6)
            if target * d < 0:
                d = 0.0                                        # direction change: brake first, then ramp up
            elif target == 0.0 or abs(target) < abs(d):
                d = target                                     # stopping or slowing: at once
            else:
                d += max(-RAMP_PER_S / RATE_HZ, min(RAMP_PER_S / RATE_HZ, target - d))
            if d == 0.0:
                vesc.brake(2.0) if vals and abs(vals["erpm"]) > 300 else vesc.duty(0.0)
            else:
                vesc.duty(d)
            if low:
                msg = f"BATTERY LOW ({vals['v_in']:.1f} V < {LOW_V}): driving disabled"
        if log:
            v = vals or {}
            log.write(f"{now:.3f},{now - t0:.3f},{d:.4f},{steer:.3f},{servo.us},{v.get('erpm', '')},{v.get('v_in', '')},"
                      f"{v.get('i_motor', '')},{v.get('tacho', '')},{v.get('fault', '')}\n")

        scr.erase()
        scr.addstr(0, 0, "SBK10 keyboard drive   W/S hold = forward/reverse   A/D steer   X straight   1-9 speed   SPACE stop   Q quit")
        md = min(SPEED_MODES[mode - 1], a.max_duty)
        scr.addstr(2, 0, f"steer {steer:+.2f} rad ({servo.us} us)    speed mode {mode}/9: duty {md:.3f}, "
                         f"~{md * MPS_PER_DUTY:.1f} m/s{'  (capped by --max-duty)' if md < SPEED_MODES[mode - 1] else ''}")
        scr.addstr(3, 0, f"motor: {'FORWARD' if drive_dir > 0 else 'REVERSE' if drive_dir < 0 else 'stopped'}   duty cmd {d:+.3f}"
                   f"   measured {speed:+.2f} m/s"
                   + ("" if vesc else "    [--no-motor: motor not connected]"))
        if vals:
            scr.addstr(4, 0, f"VESC: {vals['v_in']:.1f} V  ERPM {vals['erpm']:+d}  motor {vals['i_motor']:+.1f} A  "
                             f"FET {vals['t_fet']:.0f} C  motor {vals['t_motor']:.0f} C  fault {vals['fault']}")
        elif vesc:
            scr.addstr(4, 0, "VESC: no reply")
        scr.addstr(6, 0, msg[:100])
        scr.refresh()
        time.sleep(max(0.0, 1.0 / RATE_HZ - (time.time() - now)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", help="VESC serial port (default: auto)")
    ap.add_argument("--max-duty", type=float, default=0.25, help="duty cap (0..0.35); 0.25 = mode 9, ~2.9 m/s")
    ap.add_argument("--no-motor", action="store_true", help="steering only")
    ap.add_argument("--check", action="store_true", help="read the VESC and exit; never drives")
    ap.add_argument("--log", default=os.path.expanduser(f"~/teleop_logs/teleop_{time.strftime('%Y%m%d_%H%M%S')}.csv"))
    a = ap.parse_args()
    if not 0 < a.max_duty <= 0.35:
        sys.exit("--max-duty must be in (0, 0.35]")

    port = None if a.no_motor else find_vesc(a.port)
    if not a.no_motor and not port:
        sys.exit("VESC not found (no /dev/ttyACM*). Check USB, or use --no-motor")
    if a.check:
        return check(port)

    vesc = Vesc(port) if port else None
    servo = Servo()
    os.makedirs(os.path.dirname(a.log), exist_ok=True)
    log = open(a.log, "w")
    log.write("epoch_s,t_s,duty_cmd,steer_rad,servo_us,erpm,v_in,i_motor,tacho,fault\n")

    def shutdown(*_):
        raise SystemExit(0)
    signal.signal(signal.SIGHUP, shutdown)            # SSH dropped
    signal.signal(signal.SIGTERM, shutdown)
    try:
        curses.wrapper(run, a, vesc, servo, log)
    finally:
        if vesc:
            for _ in range(5):
                vesc.duty(0.0)
                time.sleep(0.02)
        servo.release()
        log.close()
        print(f"stopped; motor off, steering centred and released. log: {a.log}")


if __name__ == "__main__":
    main()
