"""Read sensor_msgs/LaserScan messages from a ros2 bag (sqlite3 .db3) without ROS installed.

    scans = read_scans("data/mapping/.../map_2026..._0.db3")   # list of (t, angle_min, angle_inc, ranges)
"""
import sqlite3
import struct
from typing import List, Tuple

import numpy as np


def _parse_laserscan(buf: bytes) -> Tuple[float, float, float, np.ndarray]:
    """CDR (little endian, 4-byte encapsulation header; alignment counts from after it)."""
    b = memoryview(buf)[4:]
    sec, nsec = struct.unpack_from("<iI", b, 0)
    n = struct.unpack_from("<I", b, 8)[0]                 # frame_id length incl. NUL
    o = 12 + n
    o += -o % 4
    a_min, a_max, a_inc, t_inc, scan_t, r_min, r_max = struct.unpack_from("<7f", b, o)
    o += 28
    k = struct.unpack_from("<I", b, o)[0]
    ranges = np.frombuffer(b, "<f4", k, o + 4).astype(float)
    return sec + nsec * 1e-9, a_min, a_inc, ranges


def read_scans(db3: str, topic: str = "/scan") -> List[Tuple[float, float, float, np.ndarray]]:
    con = sqlite3.connect(db3)
    tid = con.execute("SELECT id FROM topics WHERE name = ?", (topic,)).fetchone()
    if tid is None:
        raise ValueError(f"{topic} not in {db3}")
    rows = con.execute("SELECT data FROM messages WHERE topic_id = ? ORDER BY timestamp", tid).fetchall()
    con.close()
    return [_parse_laserscan(r[0]) for r in rows]
