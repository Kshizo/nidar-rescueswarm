#!/usr/bin/env python3
"""
NIDAR RescueSwarm - YOLO overhead person detector

Drop-in alternative to SurvivorDetectorAndDropper.detect_red_survivor() for
real flights, where survivors are people rather than red sim capsules. It
returns the same (detected, u_c, v_c, (x, y, w, h), debug_img) tuple, so the
depth geolocation and survivor registry downstream are unchanged.

The weights are trained by training/train_person_detector.py on VisDrone +
C2A. Any Ultralytics format loads. On the Raspberry Pi 5 use the NCNN export
(models/person_aerial_ncnn_model/), which is picked up automatically when it
exists and is several times faster on the Pi's CPU than the .pt file.

Select the detector with environment variables:
    NIDAR_DETECTOR=yolo            (default: hsv, for the red-capsule sim)
    NIDAR_YOLO_WEIGHTS=/path/to/person_aerial.pt
    NIDAR_YOLO_CONF=0.35
"""

import os
from datetime import datetime
from pathlib import Path

import cv2

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def default_weights():
    ncnn = MODELS_DIR / "person_aerial_ncnn_model"
    return ncnn if ncnn.is_dir() else MODELS_DIR / "person_aerial.pt"


class YoloPersonDetector:
    def __init__(self, weights=None, conf=None, imgsz=640, device=None):
        # Imported here so the HSV-only sim path never needs torch installed.
        from ultralytics import YOLO

        self.weights = str(weights or os.environ.get("NIDAR_YOLO_WEIGHTS") or default_weights())
        self.conf = float(conf if conf is not None else os.environ.get("NIDAR_YOLO_CONF", 0.35))
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(self.weights, task="detect")
        print(f"[YOLO] Loaded {self.weights} (conf={self.conf}, imgsz={self.imgsz})")

    def get_timestamp(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def detect_all(self, bgr_image):
        """All person boxes in the frame as [(x, y, w, h, conf), ...], best first."""
        res = self.model.predict(bgr_image, imgsz=self.imgsz, conf=self.conf,
                                 device=self.device, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(c)))
        out.sort(key=lambda d: d[4], reverse=True)
        return out

    def detect_survivor(self, bgr_image, drone_name="Drone-0", draw_debug=True):
        if bgr_image is None:
            return False, 0, 0, (0, 0, 0, 0), None

        dets = self.detect_all(bgr_image)
        debug_img = bgr_image.copy() if draw_debug else None
        if debug_img is not None:
            for x, y, w, h, c in dets:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(debug_img, f"person {c:.2f}", (x, max(12, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        if not dets:
            return False, 0, 0, (0, 0, 0, 0), debug_img

        x, y, w, h, c = dets[0]
        u_c, v_c = x + w // 2, y + h // 2
        if debug_img is not None:
            cv2.circle(debug_img, (u_c, v_c), 5, (0, 0, 255), -1)
        print(f"[{self.get_timestamp()}] [{drone_name}] [NADIR CAMERA] Person in Frame | "
              f"BBox: ({x}, {y}, {w}, {h}) | Centroid: ({u_c}, {v_c}) | Conf: {c:.2f} | Count: {len(dets)}")
        return True, u_c, v_c, (x, y, w, h), debug_img


def make_detect_fn(perception):
    """Return the frame detector selected by NIDAR_DETECTOR.

    `perception` is the SurvivorDetectorAndDropper used for HSV detection and
    for everything after detection (depth geolocation, payload drop).
    """
    if os.environ.get("NIDAR_DETECTOR", "hsv").lower() == "yolo":
        return YoloPersonDetector().detect_survivor
    return perception.detect_red_survivor


if __name__ == "__main__":
    # Quick check on a folder or single image: python3 person_detector.py img.jpg [weights]
    import sys
    det = YoloPersonDetector(weights=sys.argv[2] if len(sys.argv) > 2 else None)
    src = Path(sys.argv[1])
    files = sorted(src.glob("*.[jp][pn]g")) if src.is_dir() else [src]
    for f in files:
        img = cv2.imread(str(f))
        found, u, v, bbox, dbg = det.detect_survivor(img, draw_debug=True)
        out = f.with_name(f.stem + "_det.jpg")
        cv2.imwrite(str(out), dbg)
        print(f"{f.name}: {len(det.detect_all(img))} people -> {out}")
