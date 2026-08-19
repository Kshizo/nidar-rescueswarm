#!/usr/bin/env python3
"""
NIDAR RescueSwarm - Perception & Dynamic Falling Package Spawner Module
"""

import os
import time
import math
import numpy as np
import cv2
import subprocess
from datetime import datetime

# Drone spawn locations in Gazebo World coordinates (X, Y)
DRONE_SPAWN_POSES = [
    (0.0, -5.0), # Drone 0 spawn
    (0.0, 5.0),  # Drone 1 spawn
]

class SurvivorDetectorAndDropper:
    # Intrinsics derived from the sensor definition rather than guessed:
    # fx = fy = (width/2) / tan(hfov/2) = 320 / tan(1.204/2) = 465.7 for the
    # 640x480 IMX214. The depth camera's hfov is matched to the RGB one in the
    # OakD-Lite SDF, so a single intrinsic set is valid for both.
    def __init__(self, fx=465.7, fy=465.7, cx=320.0, cy=240.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.dropped_packages_count = 0

    def get_timestamp(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def detect_red_survivor(self, rgb_image, drone_name="Drone-0", draw_debug=True):
        if rgb_image is None:
            return False, 0, 0, (0, 0, 0, 0), None

        # The debug overlay is only needed by the viewer HUD. Callers that throw
        # it away (the mission controller) pass draw_debug=False to skip a full
        # frame copy on every callback.
        debug_img = rgb_image.copy() if draw_debug else None
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([12, 255, 255])
        lower_red2 = np.array([168, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_target = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 80:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(h) / w if w > 0 else 0
                if 0.4 <= aspect_ratio <= 4.5:
                    if area > max_area:
                        max_area = area
                        best_target = (x, y, w, h, cnt)

        if best_target is not None:
            x, y, w, h, cnt = best_target
            u_c = x + w // 2
            v_c = y + h // 2

            if debug_img is not None:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(debug_img, (u_c, v_c), 5, (0, 0, 255), -1)

            print(f"[{self.get_timestamp()}] [{drone_name}] [NADIR CAMERA] Red Survivor in Frame | BBox: ({x}, {y}, {w}, {h}) | Centroid: ({u_c}, {v_c}) | Area: {int(max_area)}px")
            return True, u_c, v_c, (x, y, w, h), debug_img

        return False, 0, 0, (0, 0, 0, 0), debug_img

    def estimate_3d_location(self, u_c, v_c, depth_image, drone_pos_ned, drone_att_rad,
                             drone_name="Drone-0", verbose=True):
        """Pixel + depth -> survivor position in the drone's local NED frame.

        drone_att_rad is (roll, pitch, yaw) in radians. All three are required:
        a yaw-only rotation is correct in level hover, but at SEARCH_SPEED the
        airframe carries real pitch, and ignoring a tilt of alpha displaces the
        target by roughly Z*sin(alpha) -- over a metre at cruise.
        """
        if depth_image is None:
            return None

        h, w = depth_image.shape[:2]
        if not (0 <= u_c < w and 0 <= v_c < h):
            return None

        u_min, u_max = max(0, u_c - 3), min(w, u_c + 4)
        v_min, v_max = max(0, v_c - 3), min(h, v_c + 4)
        depth_patch = depth_image[v_min:v_max, u_min:u_max]
        valid_depths = depth_patch[np.isfinite(depth_patch) & (depth_patch > 0.2)]

        if len(valid_depths) == 0:
            return None

        z_depth = float(np.median(valid_depths))

        # Deproject the pixel through the pinhole model into the body FRD frame.
        # The camera is nadir with the image top toward the nose, so image "up"
        # maps to Forward, image "right" to Right, and the optical axis to Down.
        dx_body = (self.cy - v_c) * z_depth / self.fy   # Forward
        dy_body = (u_c - self.cx) * z_depth / self.fx   # Right
        dz_body = z_depth                               # Down

        # Full 3-2-1 (yaw-pitch-roll) body->NED direction cosine matrix.
        roll, pitch, yaw = drone_att_rad
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy_, sy_ = math.cos(yaw), math.sin(yaw)

        R = np.array([
            [cy_ * cp, cy_ * sp * sr - sy_ * cr, cy_ * sp * cr + sy_ * sr],
            [sy_ * cp, sy_ * sp * sr + cy_ * cr, sy_ * sp * cr - cy_ * sr],
            [-sp,      cp * sr,                  cp * cr                 ],
        ])

        offset_north, offset_east, offset_down = R @ np.array([dx_body, dy_body, dz_body])

        survivor_north = drone_pos_ned[0] + float(offset_north)
        survivor_east  = drone_pos_ned[1] + float(offset_east)
        survivor_down  = drone_pos_ned[2] + float(offset_down)

        if verbose:
            print(f"[{self.get_timestamp()}] [{drone_name}] [NADIR DEPTH ESTIMATION] Measured Depth z={z_depth:.2f}m -> Target Local NED (North: {survivor_north:.2f}m, East: {survivor_east:.2f}m, Alt: {-survivor_down:.2f}m)")
        return (survivor_north, survivor_east, survivor_down)

    def drop_rescue_package(self, survivor_world_pos_ned, drone_hover_alt=None, drone_id=0, drone_name="Drone-0"):
        self.dropped_packages_count += 1
        pkg_name = f"aid_package_d{drone_id}_{self.dropped_packages_count}"

        spawn_x, spawn_y = DRONE_SPAWN_POSES[drone_id]
        target_north, target_east, target_down = survivor_world_pos_ned

        # Offset package 0.5m to the SIDE of the survivor so it lands next to them
        gz_x = spawn_x + target_east + 0.5 
        gz_y = spawn_y + target_north
        target_surface_z = max(0.15, -target_down)

        if drone_hover_alt is not None:
            gz_z_spawn = max(target_surface_z + 1.2, drone_hover_alt - 0.25)
        else:
            gz_z_spawn = target_surface_z + 1.8

        print(f"[{self.get_timestamp()}] [{drone_name}] [ACTION] Dropping Aid Package #{self.dropped_packages_count} from Altitude Z={gz_z_spawn:.2f}m -> Falling to Survivor Site (X: {gz_x:.2f}m, Y: {gz_y:.2f}m)")

        sdf_str = f"<?xml version='1.0'?><sdf version='1.9'><model name='{pkg_name}'><pose>{gz_x} {gz_y} {gz_z_spawn} 0 0 0</pose><static>false</static><link name='box_link'><visual name='v'><geometry><box><size>0.40 0.40 0.30</size></box></geometry><material><ambient>1 0.4 0 1</ambient><diffuse>1 0.5 0 1</diffuse><specular>0.8 0.8 0.8 1</specular></material></visual><collision name='c'><geometry><box><size>0.40 0.40 0.30</size></box></geometry><material><ambient>1 0.4 0 1</ambient><diffuse>1 0.5 0 1</diffuse></material></collision><inertial><mass>0.5</mass><inertia><ixx>0.005</ixx><ixy>0</ixy><ixz>0</ixz><iyy>0.005</iyy><iyz>0</iyz><izz>0.005</izz></inertia></inertial></link></model></sdf>"

        gz_cmd = f"source /opt/ros/${{ROS_DISTRO:-jazzy}}/setup.bash 2>/dev/null; gz service -s /world/rescueswarm_flood_zone/create --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 3000 --req \"sdf: \\\"{sdf_str}\\\", name: \\\"{pkg_name}\\\", pose: {{position: {{x: {gz_x}, y: {gz_y}, z: {gz_z_spawn}}}}}\" || ros2 run ros_gz_sim create -world rescueswarm_flood_zone -string \"{sdf_str}\" -name {pkg_name} -x {gz_x} -y {gz_y} -z {gz_z_spawn}"

        try:
            subprocess.Popen(gz_cmd, shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[{self.get_timestamp()}] [{drone_name}] [PACKAGE RELEASE] Aid Package '{pkg_name}' released! Falling down onto survivor site!")
        except Exception as e:
            print(f"[{self.get_timestamp()}] [{drone_name}] [PACKAGE SPAWN WARN] {e}")

        return pkg_name
