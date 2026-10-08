#!/usr/bin/env python3
"""
NIDAR RescueSwarm - webcam intrinsic calibration for geotagging

Print a chessboard (default 9x6 inner corners, 25 mm squares), then either
  live:    python3 calibrate_camera.py --camera 0 --width 1280 --height 720
           (SPACE grabs a view, Q finishes; aim for 20+ views covering the
            corners and edges of the frame at different tilts)
  offline: python3 calibrate_camera.py --images 'calib/*.jpg'

Writes webcam_calib.yaml (K, dist, width, height) for process_flight.py
--calib. Calibrate at the SAME resolution you record at, with autofocus off
if the webcam has it: refocusing changes the focal length.
"""

import argparse
import glob

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default=None)
    ap.add_argument("--images", default=None)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--cols", type=int, default=9, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=6, help="inner corners per column")
    ap.add_argument("--square", type=float, default=0.025, help="square size in metres")
    ap.add_argument("--out", default="webcam_calib.yaml")
    a = ap.parse_args()

    pattern = (a.cols, a.rows)
    objp = np.zeros((a.cols * a.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    obj_pts, img_pts, size = [], [], None

    def add(gray):
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        if found:
            obj_pts.append(objp)
            img_pts.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit))
        return found, corners

    if a.images:
        for f in sorted(glob.glob(a.images)):
            gray = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            print(f"{f}: {'ok' if add(gray)[0] else 'no board'}")
    else:
        cam = int(a.camera) if str(a.camera).isdigit() else a.camera
        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(gray, pattern, cv2.CALIB_CB_FAST_CHECK)
            view = frame.copy()
            if found:
                cv2.drawChessboardCorners(view, pattern, corners, found)
            cv2.putText(view, f"views: {len(obj_pts)}  SPACE=grab  Q=finish", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("calibrate", view)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" ") and found:
                add(gray)
            elif k == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    if len(obj_pts) < 10:
        raise SystemExit(f"only {len(obj_pts)} usable views; need at least 10 (20+ is better)")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0])))
    print(f"RMS reprojection error {rms:.3f} px (good: < 0.5)  |  horizontal FOV {hfov:.1f} deg")

    fs = cv2.FileStorage(a.out, cv2.FILE_STORAGE_WRITE)
    fs.write("K", K)
    fs.write("dist", dist)
    fs.write("width", size[0])
    fs.write("height", size[1])
    fs.write("rms", rms)
    fs.release()
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
