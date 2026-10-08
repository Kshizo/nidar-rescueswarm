#!/usr/bin/env python3
"""
Fine-tune a YOLO model to detect people from an overhead / nadir drone camera.

Run prepare_dataset.py first. Weights land in runs/person_aerial/<name>/weights.

Usage:
    python3 train_person_detector.py --data ~/nidar/datasets/person_aerial/person_aerial.yaml
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default="yolo26n.pt",
                    help="pretrained COCO checkpoint to start from (yolo26n for a Pi 5 CPU, yolo26s for a Jetson)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="yolo26n_person_aerial")
    ap.add_argument("--project", type=Path, default=Path(__file__).resolve().parent / "runs")
    ap.add_argument("--resume", action="store_true",
                    help="continue the --name run from its last.pt (all other settings are restored)")
    a = ap.parse_args()

    if a.resume:
        model = YOLO(str(a.project / a.name / "weights" / "last.pt"))
        model.train(resume=True)
        report_test(model, a)
        return

    model = YOLO(a.model)
    model.train(
        data=str(a.data.expanduser()),
        imgsz=a.imgsz,
        epochs=a.epochs,
        batch=a.batch,
        patience=25,
        project=str(a.project),
        name=a.name,
        exist_ok=True,
        single_cls=True,
        # A nadir camera has no preferred "up": vertical flips and free
        # in-plane rotation are realistic, perspective warps are not.
        flipud=0.5,
        fliplr=0.5,
        degrees=15.0,
        perspective=0.0,
        scale=0.5,
        mosaic=1.0,
        close_mosaic=10,
        # Flood/disaster light varies a lot; widen HSV jitter a little.
        hsv_v=0.5,
        workers=8,
        cos_lr=True,
        plots=True,
    )
    report_test(model, a)


def report_test(model, a):
    metrics = model.val(data=str(a.data.expanduser()), split="test", imgsz=a.imgsz)
    print(f"TEST mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f} "
          f"P={metrics.box.mp:.3f} R={metrics.box.mr:.3f}")


if __name__ == "__main__":
    main()
