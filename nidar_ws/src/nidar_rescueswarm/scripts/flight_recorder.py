#!/usr/bin/env python3
"""
NIDAR RescueSwarm - onboard flight recorder (Raspberry Pi 5)

Records the downward webcam and the flight controller's position/attitude on
ONE clock (the Pi's time.time()), so offline processing never has to align a
video against a separate log. Survivor detection happens afterwards on a GPU
with process_flight.py.

Output, one directory per flight:
    frames/000000.jpg ...  one JPEG per frame (survives a crash or power cut,
                           unlike a video container that is finalised on close)
    frames.csv             frame, t_unix
    gps.csv                t_unix, lat, lon, alt_msl, rel_alt, vn, ve, vd, hdg
    attitude.csv           t_unix, roll, pitch, yaw   (radians, NED / FRD)
    meta.json              camera + connection settings

Usage on the Pi (PX4 TELEM2 wired to the Pi's UART):
    python3 flight_recorder.py --mavlink /dev/ttyAMA0 --baud 921600 \
        --camera 0 --width 1280 --height 720 --fps 15 --out ~/flights

Stop with Ctrl-C (or SIGTERM from a systemd unit); files are flushed per row.
"""

import argparse
import csv
import json
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from pymavlink import mavutil

STOP = threading.Event()


def telemetry_loop(conn_str, baud, out_dir, rate_hz):
    mav = mavutil.mavlink_connection(conn_str, baud=baud, source_system=245)
    # Frames without telemetry cannot be geotagged, so keep complaining
    # loudly rather than letting a whole flight record blind.
    while not STOP.is_set():
        print(f"[REC] Waiting for heartbeat on {conn_str} ...")
        if mav.wait_heartbeat(timeout=5) is not None:
            break
        print("[REC] WARNING: NO TELEMETRY - frames recorded now cannot be geotagged")
    if STOP.is_set():
        return
    print(f"[REC] Connected to system {mav.target_system}")

    # Ask the autopilot to stream exactly what geotagging needs, at fixed rates.
    for msg_id, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT, rate_hz),
                       (mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, rate_hz * 2)):
        mav.mav.command_long_send(mav.target_system, mav.target_component,
                                  mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                  msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)

    with open(out_dir / "gps.csv", "w", newline="") as fg, \
         open(out_dir / "attitude.csv", "w", newline="") as fa:
        g, a = csv.writer(fg), csv.writer(fa)
        g.writerow(["t_unix", "lat", "lon", "alt_msl", "rel_alt", "vn", "ve", "vd", "hdg"])
        a.writerow(["t_unix", "roll", "pitch", "yaw"])
        while not STOP.is_set():
            m = mav.recv_match(type=["GLOBAL_POSITION_INT", "ATTITUDE"], blocking=True, timeout=1.0)
            if m is None:
                continue
            t = time.time()
            if m.get_type() == "GLOBAL_POSITION_INT":
                hdg = m.hdg / 100.0 if m.hdg != 65535 else float("nan")
                g.writerow([f"{t:.4f}", m.lat / 1e7, m.lon / 1e7, m.alt / 1000.0,
                            m.relative_alt / 1000.0, m.vx / 100.0, m.vy / 100.0, m.vz / 100.0, hdg])
                fg.flush()
            else:
                a.writerow([f"{t:.4f}", f"{m.roll:.5f}", f"{m.pitch:.5f}", f"{m.yaw:.5f}"])
                fa.flush()


def camera_loop(cam, width, height, fps, out_dir, jpeg_quality):
    cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
    # Most UVC webcams only reach full frame rate in MJPG mode.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # A deep driver queue would hand us old frames stamped with the current
    # time; at 4 m/s every 100 ms of lag is 0.4 m of geotag error.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {cam}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[REC] Camera {cam} at {w}x{h}")

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    period = 1.0 / fps
    n = 0
    with open(out_dir / "frames.csv", "w", newline="") as ff:
        wr = csv.writer(ff)
        wr.writerow(["frame", "t_unix"])
        next_t = time.time()
        while not STOP.is_set():
            ok, img = cap.read()
            t = time.time()
            if not ok:
                time.sleep(0.01)
                continue
            if t < next_t:  # webcam faster than requested: drop, don't queue
                continue
            next_t = max(next_t + period, t)
            cv2.imwrite(str(frames_dir / f"{n:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            wr.writerow([n, f"{t:.4f}"])
            ff.flush()
            n += 1
            if n % (fps * 10) == 0:
                print(f"[REC] {n} frames")
    cap.release()
    return w, h, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mavlink", default="/dev/ttyAMA0",
                    help="serial device or e.g. udpin:0.0.0.0:14540")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--camera", default="0", help="V4L2 index or /dev/videoN")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--telemetry-hz", type=int, default=20)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--out", type=Path, default=Path.home() / "flights")
    a = ap.parse_args()

    out_dir = a.out.expanduser() / datetime.now().strftime("flight_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    cam = int(a.camera) if a.camera.isdigit() else a.camera

    # Ctrl-C and SIGTERM both just end the capture loop, so the CSVs and
    # meta.json are always closed out properly.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())
    tele = threading.Thread(target=telemetry_loop, args=(a.mavlink, a.baud, out_dir, a.telemetry_hz),
                            daemon=True)
    tele.start()
    meta = {"started": datetime.now().isoformat(timespec="seconds"), "mavlink": a.mavlink,
            "camera": a.camera, "requested": [a.width, a.height, a.fps]}
    print(f"[REC] Recording to {out_dir}  (Ctrl-C to stop)")
    try:
        w, h, n = camera_loop(cam, a.width, a.height, a.fps, out_dir, a.jpeg_quality)
    finally:
        STOP.set()
        tele.join(timeout=3)
    meta.update({"stopped": datetime.now().isoformat(timespec="seconds"),
                 "resolution": [w, h], "frames": n})
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[REC] Done: {out_dir}")


if __name__ == "__main__":
    main()
