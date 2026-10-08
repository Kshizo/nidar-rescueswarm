#!/usr/bin/env python3
"""
Build a single-class overhead 'person' dataset for YOLO from four sources:

  VisDrone2019-DET  real drone footage; classes 0 (pedestrian) + 1 (people)
                    become 'person', everything else is dropped.
  C2A               humans composited into disaster scenes (already 1 class).
  SARD              search-and-rescue actors in grass, quarry, forest. The
                    Kaggle/Roboflow export is 3x3 tiles of 1920x1080 frames
                    stretched to 640x640; they are squashed back to 640x360.
  HERIDAL           4000x3000 wilderness SAR images at ~2 cm/px. Downscaling
                    to 640 leaves people 2-3 px wide, so they are tiled at
                    native resolution instead.

Frames or tiles without people are kept as negatives (all of them for
VisDrone/SARD, a sample for HERIDAL, which would otherwise be ~95% empty
tiles). VisDrone and C2A images are symlinked; SARD and HERIDAL are written.

Usage:
    python3 prepare_dataset.py --visdrone ~/nidar/datasets/VisDrone \
        --c2a ~/nidar/datasets/C2A \
        --sard ~/nidar/datasets/SARD_raw/search-and-rescue \
        --heridal ~/nidar/datasets/HERIDAL_raw \
        --out ~/nidar/datasets/person_aerial
"""

import argparse
import os
import random
from pathlib import Path

import cv2

VISDRONE_PERSON = {0, 1}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "test": "test"}


def read_boxes(lbl, keep_classes):
    out = []
    if lbl.exists():
        for row in lbl.read_text().splitlines():
            p = row.split()
            if len(p) == 5 and int(p[0]) in keep_classes:
                out.append(tuple(map(float, p[1:])))
    return out


def write_label(path, boxes):
    path.write_text("".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in boxes))


def out_dirs(out_root, split):
    oi, ol = out_root / "images" / split, out_root / "labels" / split
    oi.mkdir(parents=True, exist_ok=True)
    ol.mkdir(parents=True, exist_ok=True)
    return oi, ol


def images_in(d):
    return [f for f in sorted(d.iterdir()) if f.suffix.lower() in IMG_EXT] if d.is_dir() else []


def report(prefix, split, n_img, n_obj, n_empty):
    print(f"  {prefix:8s} {split:5s} images={n_img:6d} persons={n_obj:7d} negatives={n_empty}")


def link_split(img_dir, lbl_dir, out_root, split, prefix, keep_classes):
    oi, ol = out_dirs(out_root, split)
    n_img = n_obj = n_empty = 0
    for img in images_in(img_dir):
        boxes = read_boxes(lbl_dir / (img.stem + ".txt"), keep_classes)
        dst = oi / f"{prefix}_{img.name}"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(img.resolve(), dst)
        write_label(ol / f"{prefix}_{img.stem}.txt", boxes)
        n_img, n_obj, n_empty = n_img + 1, n_obj + len(boxes), n_empty + (not boxes)
    report(prefix, split, n_img, n_obj, n_empty)


def sard_split(src_split, out_root, split, size=(640, 360)):
    """Undo Roboflow's 640x640 stretch. Normalised labels are unaffected."""
    oi, ol = out_dirs(out_root, split)
    n_img = n_obj = n_empty = 0
    for img in images_in(src_split / "images"):
        boxes = read_boxes(src_split / "labels" / (img.stem + ".txt"), {0})
        im = cv2.imread(str(img))
        if im is None:
            continue
        cv2.imwrite(str(oi / f"sard_{img.stem}.jpg"), cv2.resize(im, size, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        write_label(ol / f"sard_{img.stem}.txt", boxes)
        n_img, n_obj, n_empty = n_img + 1, n_obj + len(boxes), n_empty + (not boxes)
    report("sard", split, n_img, n_obj, n_empty)


def tile_starts(length, tile, stride):
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    return starts + [length - tile]


def heridal_split(src_split, out_root, split, tile=640, overlap=0.2, neg_keep=0.05, min_vis=0.5):
    """Tile full-resolution images. A box is kept in a tile if at least
    min_vis of its area falls inside; it is then clipped to the tile."""
    oi, ol = out_dirs(out_root, split)
    rng = random.Random(0)
    stride = int(tile * (1 - overlap))
    n_img = n_obj = n_empty = 0
    for img in images_in(src_split / "images"):
        im = cv2.imread(str(img))
        if im is None:
            continue
        H, W = im.shape[:2]
        boxes = [((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)
                 for x, y, w, h in read_boxes(src_split / "labels" / (img.stem + ".txt"), {0})]
        for ty in tile_starts(H, tile, stride):
            for tx in tile_starts(W, tile, stride):
                th, tw = min(tile, H), min(tile, W)
                kept = []
                for x1, y1, x2, y2 in boxes:
                    cx1, cy1 = max(x1, tx), max(y1, ty)
                    cx2, cy2 = min(x2, tx + tw), min(y2, ty + th)
                    if cx2 <= cx1 or cy2 <= cy1:
                        continue
                    if (cx2 - cx1) * (cy2 - cy1) < min_vis * (x2 - x1) * (y2 - y1):
                        continue
                    kept.append((((cx1 + cx2) / 2 - tx) / tw, ((cy1 + cy2) / 2 - ty) / th,
                                 (cx2 - cx1) / tw, (cy2 - cy1) / th))
                if not kept and rng.random() > neg_keep:
                    continue
                name = f"heridal_{img.stem}_{tx}_{ty}"
                cv2.imwrite(str(oi / f"{name}.jpg"), im[ty:ty + th, tx:tx + tw],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                write_label(ol / f"{name}.txt", kept)
                n_img, n_obj, n_empty = n_img + 1, n_obj + len(kept), n_empty + (not kept)
    report("heridal", split, n_img, n_obj, n_empty)


def roboflow_splits(root):
    """Yield (src_split_dir, our_split) for a Roboflow YOLO export."""
    for name in ("train", "valid", "val", "test"):
        if (root / name / "images").is_dir():
            yield root / name, SPLIT_ALIASES[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--visdrone", type=Path)
    ap.add_argument("--c2a", type=Path)
    ap.add_argument("--sard", type=Path, help="Roboflow YOLO export root (has train/ valid/ test/)")
    ap.add_argument("--heridal", type=Path, help="Roboflow YOLO export root, full resolution")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out.expanduser().resolve()

    print(f"Building {out}")
    for split in ("train", "val", "test"):
        if a.visdrone:
            v = a.visdrone.expanduser()
            if (v / "images" / split).is_dir():
                link_split(v / "images" / split, v / "labels" / split, out, split, "visdrone", VISDRONE_PERSON)
        if a.c2a:
            c = a.c2a.expanduser() / split
            if (c / "images").is_dir():
                link_split(c / "images", c / "labels", out, split, "c2a", {0})
    if a.sard:
        for src, split in roboflow_splits(a.sard.expanduser()):
            sard_split(src, out, split)
    if a.heridal:
        for src, split in roboflow_splits(a.heridal.expanduser()):
            heridal_split(src, out, split)

    (out / "person_aerial.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n")
    print(f"Wrote {out / 'person_aerial.yaml'}")


if __name__ == "__main__":
    main()
