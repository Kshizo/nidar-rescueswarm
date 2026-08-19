#!/usr/bin/env python3
"""
NIDAR RescueSwarm - Survivor Registry (detection fusion + geotagging)
====================================================================
Turns the stream of noisy per-frame depth estimates into the confirmed,
de-duplicated survivor list the mission brief asks for ("detect survivors;
geotag survivor locations"), and converts each one to WGS-84 lat/lon for the
single-operator GCS report.

Estimates are held in the shared FIELD frame (Gazebo world convention:
+X = East, +Y = North, +Z = Up) rather than a drone's local NED, because each
drone's PX4 EKF origin sits on its own spawn pad and the two origins are 10 m
apart. Clustering in a per-drone frame would file the same survivor twice.
"""

import json
import math
import threading
from datetime import datetime

import numpy as np

EARTH_RADIUS_M = 6378137.0


class SurvivorRegistry:
    """Thread-safe. add_estimate() is called from the ROS executor thread while
    the asyncio mission loop reads confirmed_survivors() concurrently."""

    def __init__(self, cluster_radius=3.0, min_hits=4, max_spread=3.5):
        # Two estimates within cluster_radius metres are treated as the same
        # survivor. 3 m comfortably exceeds our expected geolocation error but
        # stays under the world generator's 6 m minimum survivor separation.
        self.cluster_radius = cluster_radius
        self.min_hits = min_hits
        self.max_spread = max_spread

        self._clusters = []
        self._lock = threading.Lock()
        self._next_id = 1

    # ---------------------------------------------------------------- ingest

    def add_estimate(self, east, north, alt, lat=None, lon=None, drone_idx=None):
        """Fold one frame's estimate into the registry. Returns the cluster id."""
        if not all(math.isfinite(v) for v in (east, north, alt)):
            return None

        with self._lock:
            cluster = self._nearest_cluster(east, north)

            if cluster is None:
                cluster = {
                    "id": self._next_id,
                    "east": [], "north": [], "alt": [],
                    "lat": [], "lon": [],
                    "seen_by": set(),
                    "claimed_by": None,
                    "delivered": False,
                    "first_seen": datetime.now().isoformat(timespec="seconds"),
                }
                self._next_id += 1
                self._clusters.append(cluster)

            cluster["east"].append(east)
            cluster["north"].append(north)
            cluster["alt"].append(alt)
            if lat is not None and lon is not None:
                cluster["lat"].append(lat)
                cluster["lon"].append(lon)
            if drone_idx is not None:
                cluster["seen_by"].add(drone_idx)

            return cluster["id"]

    def _nearest_cluster(self, east, north):
        """Caller must hold the lock."""
        best, best_dist = None, self.cluster_radius
        for c in self._clusters:
            d = math.hypot(np.median(c["east"]) - east, np.median(c["north"]) - north)
            if d < best_dist:
                best, best_dist = c, d
        return best

    # ----------------------------------------------------------------- query

    @staticmethod
    def _fuse(cluster):
        """Median over all estimates. Median rather than mean so a single bad
        depth read (a roof edge, a gap in the point cloud) cannot drag the
        fused position."""
        return {
            "id": cluster["id"],
            "east": float(np.median(cluster["east"])),
            "north": float(np.median(cluster["north"])),
            "alt": float(np.median(cluster["alt"])),
            "lat": float(np.median(cluster["lat"])) if cluster["lat"] else None,
            "lon": float(np.median(cluster["lon"])) if cluster["lon"] else None,
            "hits": len(cluster["east"]),
            "seen_by": sorted(cluster["seen_by"]),
            "delivered": cluster["delivered"],
            "first_seen": cluster["first_seen"],
        }

    def _spread(self, cluster):
        if len(cluster["east"]) < 2:
            return 0.0
        return float(max(np.std(cluster["east"]), np.std(cluster["north"])))

    def confirmed_survivors(self):
        """Clusters with enough mutually-consistent hits to act on. The spread
        test rejects a cluster whose estimates disagree with each other, which
        is what a flickering false positive on red debris looks like."""
        with self._lock:
            return [
                self._fuse(c) for c in self._clusters
                if len(c["east"]) >= self.min_hits and self._spread(c) <= self.max_spread
            ]

    def all_tracks(self):
        with self._lock:
            return [self._fuse(c) for c in self._clusters]

    # ------------------------------------------------------------ claim/state

    def try_claim(self, survivor_id, drone_idx):
        """Atomically claim a survivor for delivery. Returns True to exactly one
        caller, so the two drones can never both divert to the same target."""
        with self._lock:
            for c in self._clusters:
                if c["id"] == survivor_id:
                    if c["claimed_by"] is None and not c["delivered"]:
                        c["claimed_by"] = drone_idx
                        return True
                    return False
        return False

    def mark_delivered(self, survivor_id):
        with self._lock:
            for c in self._clusters:
                if c["id"] == survivor_id:
                    c["delivered"] = True
                    return

    # ---------------------------------------------------------------- geodesy

    @staticmethod
    def to_geodetic(lat0_deg, lon0_deg, north_m, east_m):
        """Local tangent plane -> WGS-84. Flat-earth is exact to well under a
        centimetre over the brief's 10-hectare (~316 m) search box."""
        lat = lat0_deg + math.degrees(north_m / EARTH_RADIUS_M)
        lon = lon0_deg + math.degrees(
            east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat0_deg)))
        )
        return lat, lon

    # ----------------------------------------------------------------- report

    def export(self, path):
        """Write the geotag report consumed by the GCS."""
        payload = {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "confirmed_count": len(self.confirmed_survivors()),
            "survivors": self.confirmed_survivors(),
        }
        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass
        return payload
