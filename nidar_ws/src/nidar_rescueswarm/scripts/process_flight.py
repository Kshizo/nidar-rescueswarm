#!/usr/bin/env python3
"""
NIDAR RescueSwarm - offline survivor detection + geotagging

Runs the overhead person model over a flight recorded by flight_recorder.py
and turns per-frame detections into a de-duplicated, geotagged survivor list.

Per frame:
  1. detect people (optionally also on overlapping tiles, for high altitude)
  2. interpolate GPS / relative altitude / attitude at the frame's timestamp
  3. cast a ray through the box centre and intersect it with flat ground at
     the take-off elevation (the real drone has no depth camera)
  4. fold the ground point into SurvivorRegistry, which clusters repeat
     sightings of the same person and rejects inconsistent flickers

Output in <flight>/results/:
    survivors.csv / .geojson / .kml   confirmed survivors (open the KML in
                                      Google Earth, the GeoJSON in QGIS)
    detections.csv                    every raw detection with its ground point
    crops/survivor_<id>.jpg           best-confidence crop of each survivor
    annotated/ (with --save-annotated)

Usage:
    python3 process_flight.py ~/flights/flight_20261008_101500 \
        --weights ../models/person_aerial.pt --calib webcam_calib.yaml
    # no calibration yet: --hfov 78 (degrees, from the webcam's datasheet)
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from person_detector import YoloPersonDetector
from survivor_registry import EARTH_RADIUS_M, SurvivorRegistry


# ------------------------------------------------------------------ telemetry

def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path} is empty")
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


class Telemetry:
    """Linear interpolation of GPS and attitude at arbitrary timestamps."""

    def __init__(self, flight_dir, max_gap=0.5):
        self.gps = load_csv(flight_dir / "gps.csv")
        self.att = load_csv(flight_dir / "attitude.csv")
        self.att["yaw"] = np.unwrap(self.att["yaw"])
        self.max_gap = max_gap

    def _interp(self, table, key, t):
        ts = table["t_unix"]
        i = np.searchsorted(ts, t)
        if i == 0 or i >= len(ts) or ts[i] - ts[i - 1] > self.max_gap:
            return None
        return float(np.interp(t, ts, table[key]))

    def at(self, t):
        vals = {k: self._interp(self.gps, k, t) for k in ("lat", "lon", "rel_alt")}
        vals.update({k: self._interp(self.att, k, t) for k in ("roll", "pitch", "yaw")})
        return None if any(v is None for v in vals.values()) else vals

    def first_fix(self):
        return float(self.gps["lat"][0]), float(self.gps["lon"][0])


def geo_to_ne(lat0, lon0, lat, lon):
    north = math.radians(lat - lat0) * EARTH_RADIUS_M
    east = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    return north, east


# --------------------------------------------------------------------- camera

class Camera:
    """Pinhole + distortion model. Nadir mount, image top toward the nose
    (the same convention as the sim's estimate_3d_location)."""

    def __init__(self, width, height, calib=None, hfov_deg=None, yaw_offset_deg=0.0):
        self.dist = None
        if calib:
            fs = cv2.FileStorage(str(calib), cv2.FILE_STORAGE_READ)
            self.K = fs.getNode("K").mat()
            self.dist = fs.getNode("dist").mat()
            cw, ch = int(fs.getNode("width").real()), int(fs.getNode("height").real())
            fs.release()
            if (cw, ch) != (width, height):  # calibrated at another resolution
                self.K = self.K.copy()
                self.K[0] *= width / cw
                self.K[1] *= height / ch
        else:
            f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
            self.K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
        self.yaw_offset = math.radians(yaw_offset_deg)

    def ray_body(self, u, v):
        """Pixel -> unit-depth ray in body FRD."""
        pt = np.array([[[u, v]]], dtype=np.float64)
        x, y = cv2.undistortPoints(pt, self.K, self.dist).reshape(2) if self.dist is not None else \
            ((u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1])
        fwd, right = -y, x
        c, s = math.cos(self.yaw_offset), math.sin(self.yaw_offset)
        return np.array([c * fwd - s * right, s * fwd + c * right, 1.0])


def body_to_ned(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def ground_point(cam, u, v, pose):
    """Ground offset (north, east) of pixel (u, v) from the drone, or None if
    the ray does not hit the ground in front of the camera."""
    d = body_to_ned(pose["roll"], pose["pitch"], pose["yaw"]) @ cam.ray_body(u, v)
    if d[2] < 0.2:  # ray within ~78 deg of horizontal: hopelessly ill-conditioned
        return None
    s = pose["rel_alt"] / d[2]
    return d[0] * s, d[1] * s


# ------------------------------------------------------------------- detector

def tiles(w, h, size, overlap):
    def starts(n):
        if n <= size:
            return [0]
        step = int(size * (1 - overlap))
        return list(range(0, n - size, step)) + [n - size]
    return [(x, y, min(size, w), min(size, h)) for y in starts(h) for x in starts(w)]


def detect(det, img, tile, overlap, iou=0.5):
    """[(x, y, w, h, conf)] from the full frame plus, if tile > 0, every tile,
    merged with NMS so a person straddling a seam is reported once."""
    import torch
    from torchvision.ops import nms

    h, w = img.shape[:2]
    crops, offsets = [img], [(0, 0)]
    if tile:
        for x, y, tw, th in tiles(w, h, tile, overlap):
            crops.append(img[y:y + th, x:x + tw])
            offsets.append((x, y))
    boxes, scores = [], []
    for res, (ox, oy) in zip(det.model.predict(crops, imgsz=det.imgsz, conf=det.conf,
                                               device=det.device, verbose=False), offsets):
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
            scores.append(c)
    if not boxes:
        return []
    b, s = torch.tensor(boxes), torch.tensor(scores)
    keep = nms(b, s, iou).tolist()
    return [(int(b[i, 0]), int(b[i, 1]), int(b[i, 2] - b[i, 0]), int(b[i, 3] - b[i, 1]), float(s[i]))
            for i in keep]


# --------------------------------------------------------------------- output

def write_outputs(out, survivors, flight_name):
    with open(out / "survivors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lat", "lon", "north_m", "east_m", "hits", "max_conf", "first_t", "crop"])
        for s in survivors:
            w.writerow([s["id"], f"{s['lat']:.7f}", f"{s['lon']:.7f}", f"{s['north']:.2f}",
                        f"{s['east']:.2f}", s["hits"], f"{s['max_conf']:.2f}", f"{s['first_t']:.2f}", s["crop"]])

    (out / "survivors.geojson").write_text(json.dumps({
        "type": "FeatureCollection",
        "features": [{"type": "Feature",
                      "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
                      "properties": {k: s[k] for k in ("id", "hits", "max_conf", "crop")}}
                     for s in survivors]}, indent=2))

    marks = "".join(
        f"<Placemark><name>Survivor {s['id']}</name><description>{s['hits']} sightings, "
        f"max conf {s['max_conf']:.2f}</description><Point><coordinates>{s['lon']:.7f},"
        f"{s['lat']:.7f},0</coordinates></Point></Placemark>\n" for s in survivors)
    (out / "survivors.kml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">'
        f"<Document><name>{flight_name}</name>\n{marks}</Document></kml>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flight", type=Path)
    ap.add_argument("--weights", default=None, help="default: models/person_aerial.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--tile", type=int, default=0,
                    help="also detect on NxN tiles (e.g. 640) for frames much larger than imgsz")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--calib", type=Path, help="calibrate_camera.py output (preferred)")
    ap.add_argument("--hfov", type=float, default=78.0, help="horizontal FOV in degrees, if no --calib")
    ap.add_argument("--cam-yaw", type=float, default=0.0,
                    help="degrees the image top is rotated clockwise from the nose")
    ap.add_argument("--cam-latency", type=float, default=0.0,
                    help="seconds between exposure and the recorded timestamp")
    ap.add_argument("--min-alt", type=float, default=3.0, help="skip frames below this AGL (take-off/landing)")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--home", help="lat,lon of take-off point (default: first GPS fix)")
    ap.add_argument("--cluster-radius", type=float, default=2.0,
                    help="sightings closer than this are one person; 2 m keeps groups apart")
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--save-annotated", action="store_true")
    a = ap.parse_args()

    flight = a.flight.expanduser()
    out = flight / "results"
    (out / "crops").mkdir(parents=True, exist_ok=True)
    if a.save_annotated:
        (out / "annotated").mkdir(exist_ok=True)

    tele = Telemetry(flight)
    lat0, lon0 = map(float, a.home.split(",")) if a.home else tele.first_fix()
    frames = load_csv(flight / "frames.csv")
    det = YoloPersonDetector(weights=a.weights, conf=a.conf, imgsz=a.imgsz)
    registry = SurvivorRegistry(cluster_radius=a.cluster_radius, min_hits=a.min_hits)
    best = {}        # cluster id -> (conf, crop)
    first_seen = {}  # cluster id -> t_unix of the first sighting
    cam = None
    skipped = {"no_telemetry": 0, "low_alt": 0, "unreadable": 0}

    with open(out / "detections.csv", "w", newline="") as fd:
        dw = csv.writer(fd)
        dw.writerow(["frame", "t_unix", "x", "y", "w", "h", "conf", "lat", "lon",
                     "drone_lat", "drone_lon", "rel_alt", "cluster"])
        idxs = range(0, len(frames["frame"]), a.stride)
        for n, i in enumerate(idxs):
            fi, t = int(frames["frame"][i]), frames["t_unix"][i] - a.cam_latency
            pose = tele.at(t)
            if pose is None:
                skipped["no_telemetry"] += 1
                continue
            if pose["rel_alt"] < a.min_alt:
                skipped["low_alt"] += 1
                continue
            img = cv2.imread(str(flight / "frames" / f"{fi:06d}.jpg"))
            if img is None:
                skipped["unreadable"] += 1
                continue
            if cam is None:
                cam = Camera(img.shape[1], img.shape[0], a.calib, a.hfov, a.cam_yaw)

            dn, de = geo_to_ne(lat0, lon0, pose["lat"], pose["lon"])
            dets = detect(det, img, a.tile, a.overlap)
            for x, y, w, h, c in dets:
                gp = ground_point(cam, x + w / 2, y + h / 2, pose)
                if gp is None:
                    continue
                north, east = dn + gp[0], de + gp[1]
                lat, lon = SurvivorRegistry.to_geodetic(lat0, lon0, north, east)
                cid = registry.add_estimate(east, north, 0.0, lat=lat, lon=lon)
                dw.writerow([fi, f"{t:.3f}", x, y, w, h, f"{c:.3f}", f"{lat:.7f}", f"{lon:.7f}",
                             f"{pose['lat']:.7f}", f"{pose['lon']:.7f}", f"{pose['rel_alt']:.1f}", cid])
                first_seen.setdefault(cid, t)
                if cid not in best or c > best[cid][0]:
                    pad = max(w, h)
                    crop = img[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad].copy()
                    cv2.rectangle(crop, (x - max(0, x - pad), y - max(0, y - pad)),
                                  (x - max(0, x - pad) + w, y - max(0, y - pad) + h), (0, 255, 0), 1)
                    best[cid] = (c, crop)
            if a.save_annotated and dets:
                for x, y, w, h, c in dets:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, f"{c:.2f}", (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)
                cv2.imwrite(str(out / "annotated" / f"{fi:06d}.jpg"), img)
            if n % 200 == 0:
                print(f"[PROC] frame {fi} ({n + 1}/{len(idxs)}), {len(registry.all_tracks())} candidate tracks")

    survivors = []
    for s in sorted(registry.confirmed_survivors(), key=lambda s: s["id"]):
        conf, crop = best[s["id"]]
        name = f"crops/survivor_{s['id']:03d}.jpg"
        cv2.imwrite(str(out / name), crop)
        survivors.append({**s, "max_conf": conf, "first_t": first_seen[s["id"]], "crop": name})
    write_outputs(out, survivors, flight.name)

    print(f"[PROC] Skipped frames: {skipped}")
    print(f"[PROC] {len(survivors)} confirmed survivors "
          f"({len(registry.all_tracks()) - len(survivors)} unconfirmed tracks) -> {out}")
    for s in survivors:
        print(f"   #{s['id']:3d}  {s['lat']:.7f}, {s['lon']:.7f}  hits={s['hits']}  conf={s['max_conf']:.2f}")


if __name__ == "__main__":
    main()
