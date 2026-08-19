#!/usr/bin/env python3
"""
NIDAR RescueSwarm - Dual-Tier LiDAR & Spatial Safety Collision Avoidance
========================================================================
1. Tier 1: Real-Time 360° LiDAR Processing with Body-to-NED Rotational Transform.
2. Tier 2: Spatial Safety Boundary Guard for all generated obstacles (houses, trees, towers).
3. Emergency Push-Back and Vertical Climb-Over when proximity < 1.8m.
4. Inter-Drone Swarm Evasion via peer position telemetry.
"""

import math
import os
import json
import numpy as np
from datetime import datetime

# Potential-field tuning. The mission cruises at SEARCH_SPEED (2.5 m/s); these
# ceilings keep total evasion meaningfully below that so the goal always wins in
# steady state and the drone deflects around obstacles instead of stalling
# against them. Ratio matters more than absolute value: raising safety_distance
# makes avoidance start sooner, raising these would make it fight the mission.
REPULSION_MAX = 1.8     # per-tier lidar repulsion magnitude (m/s)
EVASION_MAX_H = 1.5     # combined horizontal evasion ceiling (m/s)
EVASION_MAX_V = 1.2     # combined vertical (climb-over) ceiling (m/s)

class ObstacleAvoidanceModule:
    """
    Robust dual-tier obstacle avoidance module.
    """
    def __init__(self, spawn_offset=(0.0, 0.0), safety_distance=3.5, emergency_distance=1.8, swarm_safety_dist=4.5):
        self.spawn_x, self.spawn_y = spawn_offset
        self.safety_distance = safety_distance
        self.emergency_distance = emergency_distance
        self.swarm_safety_dist = swarm_safety_dist

        self.closest_obstacle_dist = float('inf')
        self.closest_obstacle_angle_ned = 0.0
        self.lidar_evasion_vector_ned = np.array([0.0, 0.0, 0.0])
        self.collision_imminent = False
        self.last_log_time = 0.0

        # Load obstacle list from /tmp/world_layout.json for Tier 2 safety guard
        self.all_obstacles_gz = []
        if os.path.exists("/tmp/world_layout.json"):
            try:
                with open("/tmp/world_layout.json") as f:
                    data = json.load(f)
                    if "houses" in data:
                        for h in data["houses"]:
                            self.all_obstacles_gz.append({
                                "name": h.get("name", "house"),
                                "x": h["x"], "y": h["y"],
                                "sx": h.get("sx", 6.0), "sy": h.get("sy", 6.0),
                                "h": h.get("h", 5.0)
                            })
                    if "trees" in data:
                        for t in data["trees"]:
                            cr = t.get("canopy_r", 3.0)
                            self.all_obstacles_gz.append({
                                "name": t.get("name", "tree"),
                                "x": t["x"], "y": t["y"],
                                "sx": cr * 2.2, "sy": cr * 2.2,
                                "h": t.get("h", 10.0)
                            })
                    if "towers" in data:
                        for tw in data["towers"]:
                            self.all_obstacles_gz.append({
                                "name": tw.get("name", "tower"),
                                "x": tw["x"], "y": tw["y"],
                                "sx": 2.0, "sy": 2.0,
                                "h": tw.get("h", 12.0)
                            })
            except Exception:
                pass

        if not self.all_obstacles_gz:
            self.all_obstacles_gz = [
                {"name": "house_1", "x": 10.0,  "y": 8.0,   "sx": 6.0, "sy": 6.0, "h": 5.0},
                {"name": "house_2", "x": -12.0, "y": 14.0,  "sx": 7.0, "sy": 5.0, "h": 4.0},
                {"name": "house_3", "x": 14.0,  "y": -10.0, "sx": 8.0, "sy": 8.0, "h": 6.0},
                {"name": "house_4", "x": -12.0, "y": -14.0, "sx": 5.0, "sy": 4.0, "h": 3.0},
            ]

    def get_timestamp(self):
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]

    def process_lidar_scan(self, ranges, angle_min, angle_increment, range_min, range_max, drone_yaw_rad=0.0):
        """
        Processes 360-degree LiDAR scan ranges and transforms them from
        Drone Body Frame into PX4 World NED Frame using drone_yaw_rad.
        """
        ranges_arr = np.array(ranges)
        angles_body = angle_min + np.arange(len(ranges_arr)) * angle_increment

        valid_mask = np.isfinite(ranges_arr) & (ranges_arr >= range_min) & (ranges_arr <= range_max)

        if not np.any(valid_mask):
            self.closest_obstacle_dist = float('inf')
            self.lidar_evasion_vector_ned = np.array([0.0, 0.0, 0.0])
            self.collision_imminent = False
            return self.lidar_evasion_vector_ned

        valid_ranges = ranges_arr[valid_mask]
        valid_angles_body = angles_body[valid_mask]

        min_idx = np.argmin(valid_ranges)
        self.closest_obstacle_dist = valid_ranges[min_idx]

        closest_body_angle = valid_angles_body[min_idx]
        self.closest_obstacle_angle_ned = (closest_body_angle + drone_yaw_rad)

        repulsion_ned = np.array([0.0, 0.0, 0.0])
        close_mask = valid_ranges < self.safety_distance

        if np.any(close_mask):
            close_ranges = valid_ranges[close_mask]
            close_angles_body = valid_angles_body[close_mask]

            for r, a_body in zip(close_ranges, close_angles_body):
                x_body = r * math.cos(a_body)
                y_body = r * math.sin(a_body)

                x_ned = x_body * math.cos(drone_yaw_rad) - y_body * math.sin(drone_yaw_rad)
                y_ned = x_body * math.sin(drone_yaw_rad) + y_body * math.cos(drone_yaw_rad)

                weight = ((self.safety_distance - r) / self.safety_distance) ** 2.0

                repulsion_ned[0] -= x_ned * weight
                repulsion_ned[1] -= y_ned * weight

            norm = np.linalg.norm(repulsion_ned[:2])
            if norm > 1e-4:
                # Deflection strength must stay well below cruise speed or the
                # goal-seeking velocity and the repulsion cancel and the drone
                # parks in a local minimum. Reacting EARLIER is the job of
                # safety_distance (how far out this triggers); it is not the job
                # of a bigger push. Genuine near-misses are handled by the
                # collision_imminent override below, which ignores this cap.
                repulsion_ned[:2] = (repulsion_ned[:2] / norm) * REPULSION_MAX

            self.collision_imminent = self.closest_obstacle_dist < self.emergency_distance
            if self.collision_imminent:
                repulsion_ned[2] = -2.0  # Upward climb
        else:
            self.collision_imminent = False

        self.lidar_evasion_vector_ned = repulsion_ned
        return self.lidar_evasion_vector_ned

    def compute_spatial_boundary_repulsion(self, my_pos, drone_name="Drone-0"):
        """
        Tier 2 Safety Guard: Spatial boundary repulsion for all houses, trees, and towers.
        """
        my_n, my_e, my_d = my_pos[:3]
        my_alt = -my_d
        spatial_rep = np.array([0.0, 0.0, 0.0])

        for obs in self.all_obstacles_gz:
            obs_n = obs["y"] - self.spawn_y
            obs_e = obs["x"] - self.spawn_x
            obs_h = obs["h"]

            # Check if drone is at or below obstacle height (+ 2.0m clearance margin)
            if my_alt < (obs_h + 2.0):
                margin_n = obs["sy"] / 2.0 + self.safety_distance
                margin_e = obs["sx"] / 2.0 + self.safety_distance

                dist_n = abs(my_n - obs_n)
                dist_e = abs(my_e - obs_e)

                if dist_n < margin_n and dist_e < margin_e:
                    diff_n = my_n - obs_n
                    diff_e = my_e - obs_e
                    d_center = math.hypot(diff_n, diff_e)

                    if d_center > 0.1:
                        # Smooth penetration-based scaling (0 at margin edge -> 1.2 m/s near wall)
                        pen_n = max(0.0, (margin_n - dist_n) / self.safety_distance)
                        pen_e = max(0.0, (margin_e - dist_e) / self.safety_distance)
                        penetration = min(1.0, max(pen_n, pen_e)) ** 1.5

                        rep_n = (diff_n / d_center) * (1.2 * penetration)
                        rep_e = (diff_e / d_center) * (1.2 * penetration)

                        # Gentle climb-over component (max -0.8 m/s UP in NED)
                        climb_urgency = max(0.0, ((obs_h + 2.0) - my_alt) / (obs_h + 2.0))
                        rep_d = -climb_urgency * (0.8 * penetration)

                        spatial_rep += np.array([rep_n, rep_e, rep_d])

        # Cap total spatial repulsion vector
        h_norm = np.linalg.norm(spatial_rep[:2])
        if h_norm > 1.2:
            spatial_rep[:2] = (spatial_rep[:2] / h_norm) * 1.2
        spatial_rep[2] = max(-1.0, min(1.0, spatial_rep[2]))

        return spatial_rep

    def compute_swarm_repulsion(self, my_pos, other_drone_positions, drone_name="Drone-0"):
        swarm_repulsion = np.array([0.0, 0.0, 0.0])
        my_pos_arr = np.array(my_pos[:3])

        for peer_pos in other_drone_positions:
            peer_pos_arr = np.array(peer_pos[:3])
            diff = my_pos_arr - peer_pos_arr
            dist = np.linalg.norm(diff)

            if 0.1 < dist < self.swarm_safety_dist:
                repulsion_mag = (self.swarm_safety_dist - dist) / self.swarm_safety_dist
                direction = diff / dist
                swarm_repulsion += direction * repulsion_mag * 2.0

        return swarm_repulsion

    def get_avoidance_velocity(self, desired_velocity, my_pos=None, other_drone_positions=None, drone_name="Drone-0", active_rescue=False, cruise_alt=8.5):
        desired_arr = np.array(desired_velocity, dtype=float)

        if active_rescue:
            if my_pos is not None and other_drone_positions is not None:
                swarm_ev = self.compute_swarm_repulsion(my_pos, other_drone_positions, drone_name)
                desired_arr += swarm_ev
            speed = np.linalg.norm(desired_arr)
            if speed > 6.0:
                desired_arr = (desired_arr / speed) * 6.0
            return desired_arr.tolist()

        # Combine Tier 1 (LiDAR) + Tier 2 (Spatial Guard) + Swarm Evasion
        combined_evasion = self.lidar_evasion_vector_ned.copy()

        if my_pos is not None:
            spatial_ev = self.compute_spatial_boundary_repulsion(my_pos, drone_name=drone_name)
            combined_evasion += spatial_ev

        if my_pos is not None and other_drone_positions is not None:
            swarm_ev = self.compute_swarm_repulsion(my_pos, other_drone_positions, drone_name)
            combined_evasion += swarm_ev

        # Cap the combined evasion offset so it deflects the commanded velocity
        # rather than overpowering it (see REPULSION_MAX note above).
        ev_norm = np.linalg.norm(combined_evasion[:2])
        if ev_norm > EVASION_MAX_H:
            combined_evasion[:2] = (combined_evasion[:2] / ev_norm) * EVASION_MAX_H
        combined_evasion[2] = max(-EVASION_MAX_V, min(EVASION_MAX_V, combined_evasion[2]))

        if np.linalg.norm(combined_evasion) > 0.0:
            now = math.floor(datetime.now().timestamp() * 2) / 2
            if now > self.last_log_time:
                self.last_log_time = now
                print(f"[{self.get_timestamp()}] [{drone_name}] [AVOIDANCE GUARD] Evasion Offset [{combined_evasion[0]:.2f}, {combined_evasion[1]:.2f}, {combined_evasion[2]:.2f}]")

            if self.collision_imminent:
                desired_arr[0] = combined_evasion[0]
                desired_arr[1] = combined_evasion[1]
                desired_arr[2] = min(desired_arr[2], combined_evasion[2])
            else:
                desired_arr += combined_evasion

        speed = np.linalg.norm(desired_arr)
        if speed > 6.0:
            desired_arr = (desired_arr / speed) * 6.0

        return desired_arr.tolist()
