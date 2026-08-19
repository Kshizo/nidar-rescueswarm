#!/usr/bin/env python3
"""
NIDAR RescueSwarm - Strict Visual Camera Verification & Geofence Mission Controller
===================================================================================
1. Subscribes to live ROS 2 camera feeds (/drone_0/camera/image_raw, /drone_1/camera/image_raw).
2. Runs real-time OpenCV HSV color segmentation on incoming camera frames.
3. Drone ONLY initiates rescue hover & aid package drop when OpenCV VISUALLY DETECTS a red survivor in the camera frame!
"""

import asyncio
import json
import math
import os
import time
import numpy as np
import cv2
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from datetime import datetime

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

from obstacle_avoidance import ObstacleAvoidanceModule
from perception_and_rescue import SurvivorDetectorAndDropper
from survivor_registry import SurvivorRegistry
import mission_map
import live_map

DRONE_PORTS = [14540, 14541]
GRPC_PORTS  = [50051, 50052]

DRONE_SPAWN_POSES = [
    (0.0, -5.0), # Drone 0 spawn pad
    (0.0, 5.0),  # Drone 1 spawn pad
]

CRUISING_ALT = 8.5
# 4.0 m/s against a 2.0 m safety ring left ~0.5 s to react, so the drone was
# already inside an obstacle's margin by the time repulsion built up. Slower
# cruise + a wider ring (see AVOID_* below) gives ~1.6 s instead.
SEARCH_SPEED = 2.5
DETECTION_RADIUS = 6.0

# ------------------------------------------------------------------ avoidance
# These override ObstacleAvoidanceModule's constructor defaults. The mission
# used to pass values TIGHTER than the module's own (2.0/1.2/3.5 vs 3.5/1.8/4.5),
# which is what made obstacles register too late to avoid smoothly.
AVOID_SAFETY_DIST = 4.0     # start pushing away this far out
AVOID_EMERGENCY_DIST = 2.0  # hard override + climb inside this
AVOID_SWARM_DIST = 4.5      # drone-to-drone separation

# ---------------------------------------------------------- stall escape
# Potential-field avoidance has a known failure mode: repulsion from an
# obstacle (or a corner, where two obstacles push back at once) exactly cancels
# attraction to the goal and the drone parks there. Detect no-progress and
# slide sideways to walk around it rather than pressing into it.
STALL_ESCAPE_SPEED = 2.0
STALL_ESCAPE_BURST = 2.5
STALL_PROGRESS_EPS = 0.3
WP_STALL_TIMEOUT = 5.0      # sweep leg: escape after this long without progress
WP_MAX_ESCAPES = 4          # then give up on the leg

# ---------------------------------------------------------------- return home
# The world generator builds houses up to 12.5 m, trees 8.5-12 m and towers
# 10-13 m. Transiting home at CRUISING_ALT (8.5 m) therefore routes the drone
# straight through the obstacle field: avoidance pushes it away from an
# obstacle exactly as the goal pulls it through, the two cancel, and the drone
# parks in that local minimum forever. Climb above everything first.
RTL_ALT = 16.0              # clears the tallest generated obstacle + ~3 m
RTL_CLIMB_TIMEOUT = 25.0    # give up on the climb and transit anyway
RTL_STALL_TIMEOUT = 6.0     # no progress for this long => wedged
RTL_TOTAL_TIMEOUT = 120.0   # hand over to PX4's own RTL past this
RTL_PROGRESS_EPS = 0.3      # closing by less than this does not count
WAYPOINT_TIMEOUT = 90.0     # abandon a leg rather than stall the whole sweep

from sensor_msgs.msg import Image, LaserScan

swarm_drone_positions = {}
swarm_drone_yaws = {}
# (vn, ve, vd) m/s per drone -- the telemetry stream already carries velocity,
# and the live map shows real ground speed rather than differencing positions.
swarm_drone_velocities = {}
# Flown path per drone, in the shared field frame, for the live map.
swarm_drone_trails = {}
# Full (roll, pitch, yaw) in radians. Yaw alone is not enough to project a
# camera detection into NED once the airframe is tilted at cruise speed.
swarm_drone_attitudes = {}
# Home lat/lon per drone, captured after GPS lock; the origin for geotagging.
swarm_home_geo = {}
avoidance_modules = {}
shared_rescued_survivors = set()

# Fused, de-duplicated survivor list shared by the whole swarm. This is what
# active geotagging produces and what the mission now flies to.
survivor_registry = SurvivorRegistry()

LIVE_MAP_PORT = int(os.environ.get("NIDAR_LIVE_MAP_PORT", "8080"))
LIVE_MAP_ENABLED = os.environ.get("NIDAR_LIVE_MAP", "1") != "0"
MISSION_START_TS = None

GEOTAG_REPORT_PATH = "/tmp/nidar_survivor_geotags.json"
WORLD_LAYOUT_PATH = "/tmp/world_layout.json"

# Every run leaves a timestamped, self-contained HTML map + CSV geotag table
# here, so the operator record survives the next launch overwriting /tmp.
# scripts/ -> nidar_rescueswarm -> src -> nidar_ws
MISSION_REPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))),
    "mission_reports")

# Ground truth is loaded for ACCURACY SCORING ONLY -- it is never used to
# select or navigate to a target. Set NIDAR_GROUND_TRUTH=1 to fall back to the
# old ground-truth-driven behaviour for comparison.
USE_GROUND_TRUTH_TARGETS = os.environ.get("NIDAR_GROUND_TRUTH", "0") == "1"

# Live camera detection states from ROS 2 subscription
latest_camera_detections = {
    0: {"detected": False, "timestamp": 0.0},
    1: {"detected": False, "timestamp": 0.0}
}

def get_timestamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


async def escape_local_minimum(drone, name, err_n, err_e, attempt, dt, context):
    """Break out of a potential-field local minimum (obstacle head-on, or a
    corner where two obstacles repel at once).

    Pressing harder toward the goal cannot work -- repulsion scales with
    proximity, so it just cancels harder. Instead move PERPENDICULAR to the
    goal bearing, which the repulsion field does not oppose, while climbing.
    Sides alternate per attempt so a blocked left slide tries right next.
    """
    side = 1.0 if (attempt % 2) else -1.0
    ang = math.atan2(err_e, err_n) + side * (math.pi / 2.0)
    print(f"[{get_timestamp()}] [{name}] [STALL ESCAPE] {context} -- attempt "
          f"#{attempt} sliding {'left' if side > 0 else 'right'} + climbing")

    t0 = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - t0) < STALL_ESCAPE_BURST:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(
            STALL_ESCAPE_SPEED * math.cos(ang),
            STALL_ESCAPE_SPEED * math.sin(ang),
            -0.8,               # climb while sliding
            0.0))
        await asyncio.sleep(dt)


def log_geotag_accuracy(sv, ground_truth, drone_name):
    """Score a fused geotag against the world generator's ground truth.

    Ground truth is used ONLY to measure error and never to select or navigate
    to a target -- at the real mission there is no world_layout.json.
    """
    if not ground_truth:
        return
    best, best_d = None, float("inf")
    for (gz_x, gz_y, gz_z) in ground_truth:
        d = math.hypot(gz_x - sv["east"], gz_y - sv["north"])
        if d < best_d:
            best, best_d = (gz_x, gz_y, gz_z), d
    if best is None:
        return

    v_err = abs(best[2] - sv["alt"])
    geo = ""
    if sv.get("lat") is not None:
        geo = f" | geotag {sv['lat']:.7f}, {sv['lon']:.7f}"
    print(f"[{get_timestamp()}] [{drone_name}] [GEOTAG ACCURACY] Survivor #{sv['id']}: "
          f"est=(E {sv['east']:.2f}, N {sv['north']:.2f}, alt {sv['alt']:.2f}) "
          f"truth=(E {best[0]:.2f}, N {best[1]:.2f}, alt {best[2]:.2f}) "
          f"horiz_err={best_d:.2f}m vert_err={v_err:.2f}m{geo}")


class RosSensorSubscriberNode(Node):
    def __init__(self):
        super().__init__('ros_sensor_subscriber_node')
        self.bridge = CvBridge()
        self.perception = SurvivorDetectorAndDropper()

        # Sensor QoS = best-effort, depth 1. With the default reliable depth-10
        # queue, image messages pile up faster than OpenCV can drain them and
        # the backlog grows without bound until the machine swaps. Dropping
        # stale frames is the correct behaviour for live detection anyway.
        #
        # RGB and depth are paired by timestamp rather than read independently:
        # geolocation samples depth at the pixel where the RGB detector fired,
        # so the two frames must describe the same instant. At 4 m/s a 100 ms
        # mismatch is already 0.4 m of error.
        self._sync = {}
        for idx in (0, 1):
            rgb = Subscriber(self, Image, f'/drone_{idx}/camera/image_raw',
                             qos_profile=qos_profile_sensor_data)
            depth = Subscriber(self, Image, f'/drone_{idx}/depth/image_raw',
                               qos_profile=qos_profile_sensor_data)
            sync = ApproximateTimeSynchronizer([rgb, depth], queue_size=5, slop=0.05)
            sync.registerCallback(
                lambda rgb_msg, depth_msg, i=idx: self.rgbd_cb(rgb_msg, depth_msg, i)
            )
            # Keep references alive; message_filters does not own them.
            self._sync[idx] = (rgb, depth, sync)

        self.sub_scan0 = self.create_subscription(
            LaserScan, '/drone_0/scan', lambda msg: self.scan_cb(msg, 0),
            qos_profile_sensor_data
        )
        self.sub_scan1 = self.create_subscription(
            LaserScan, '/drone_1/scan', lambda msg: self.scan_cb(msg, 1),
            qos_profile_sensor_data
        )

    def rgbd_cb(self, rgb_msg, depth_msg, drone_idx):
        """Detect in RGB, then geolocate through the time-matched depth frame."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            detected, u_c, v_c, bbox, _ = self.perception.detect_red_survivor(
                cv_img, f"Drone-{drone_idx}", draw_debug=False
            )
            latest_camera_detections[drone_idx] = {
                "detected": detected,
                "timestamp": time.time()
            }
            if not detected:
                return

            # Need pose and attitude before a pixel means anything in NED.
            pos = swarm_drone_positions.get(drone_idx)
            att = swarm_drone_attitudes.get(drone_idx)
            if pos is None or att is None:
                return

            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            est = self.perception.estimate_3d_location(
                u_c, v_c, depth_img, pos, att,
                drone_name=f"Drone-{drone_idx}", verbose=False
            )
            if est is None:
                return

            s_north, s_east, s_down = est

            # Local NED (origin = this drone's own spawn pad) -> shared field
            # frame, so both drones' estimates cluster into one registry.
            spawn_x, spawn_y = DRONE_SPAWN_POSES[drone_idx]
            field_east = s_east + spawn_x
            field_north = s_north + spawn_y
            field_alt = -s_down

            # Reject physically impossible estimates before they pollute a
            # cluster: below ground or above the drone itself.
            if field_alt < -1.0 or field_alt > 30.0:
                return

            lat = lon = None
            home = swarm_home_geo.get(drone_idx)
            if home is not None:
                lat, lon = SurvivorRegistry.to_geodetic(home[0], home[1], s_north, s_east)

            survivor_registry.add_estimate(
                field_east, field_north, field_alt,
                lat=lat, lon=lon, drone_idx=drone_idx
            )
        except Exception:
            pass

    def scan_cb(self, msg, drone_idx):
        try:
            if drone_idx in avoidance_modules:
                yaw_rad = swarm_drone_yaws.get(drone_idx, 0.0)
                avoidance_modules[drone_idx].process_lidar_scan(
                    msg.ranges, msg.angle_min, msg.angle_increment,
                    msg.range_min, msg.range_max, drone_yaw_rad=yaw_rad
                )
        except Exception as e:
            pass


class PIDController:
    def __init__(self, kp=0.85, ki=0.02, kd=0.35, max_output=0.8):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt=0.1):
        self.integral += error * dt
        self.integral = float(np.clip(self.integral, -0.5, 0.5))
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return float(np.clip(output, -self.max_output, self.max_output))

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class CoverageTracker:
    def __init__(self, e_min, e_max, n_min, n_max):
        self.e_min = e_min
        self.e_max = e_max
        self.n_min = n_min
        self.n_max = n_max
        self.grid_size = 1.0

        self.e_cells = int(math.ceil((e_max - e_min) / self.grid_size))
        self.n_cells = int(math.ceil((n_max - n_min) / self.grid_size))

        self.grid = np.zeros((self.n_cells, self.e_cells), dtype=bool)

    def mark_visited(self, current_n, current_e, radius=3.0):
        cell_n_idx = int((current_n - self.n_min) / self.grid_size)
        cell_e_idx = int((current_e - self.e_min) / self.grid_size)

        r_cells = int(math.ceil(radius / self.grid_size))

        for n in range(max(0, cell_n_idx - r_cells), min(self.n_cells, cell_n_idx + r_cells + 1)):
            for e in range(max(0, cell_e_idx - r_cells), min(self.e_cells, cell_e_idx + r_cells + 1)):
                cell_center_n = self.n_min + (n + 0.5) * self.grid_size
                cell_center_e = self.e_min + (e + 0.5) * self.grid_size
                dist = math.hypot(cell_center_n - current_n, cell_center_e - current_e)
                if dist <= radius:
                    self.grid[n, e] = True

    def get_coverage_percent(self):
        total = self.n_cells * self.e_cells
        if total == 0:
            return 100.0
        visited = np.sum(self.grid)
        return (visited / total) * 100.0


async def run_drone_rescue_mission(port: int, grpc_port: int, drone_index: int):
    name = f"Drone-{drone_index}"
    spawn_x, spawn_y = DRONE_SPAWN_POSES[drone_index]

    if drone_index > 0:
        stagger = drone_index * 5.0
        print(f"[{get_timestamp()}] [{name}] Staggering startup by {stagger:.0f}s ...")
        await asyncio.sleep(stagger)

    # Sector bounds
    n_min = -25.0 - spawn_y
    n_max = 25.0 - spawn_y

    if drone_index == 0:
        e_min = -25.0 - spawn_x
        e_max = 0.0 - spawn_x
        leg_e_positions = [-22.0, -18.0, -14.0, -10.0, -6.0, -2.0]
    else:
        e_min = 0.0 - spawn_x
        e_max = 25.0 - spawn_x
        leg_e_positions = [2.0, 6.0, 10.0, 14.0, 18.0, 22.0]

    geo_n_min = n_min + 1.0
    geo_n_max = n_max - 1.0

    avoidance = ObstacleAvoidanceModule(
        spawn_offset=(spawn_x, spawn_y),
        safety_distance=AVOID_SAFETY_DIST,
        emergency_distance=AVOID_EMERGENCY_DIST,
        swarm_safety_dist=AVOID_SWARM_DIST)
    avoidance_modules[drone_index] = avoidance
    rescue_perception = SurvivorDetectorAndDropper()
    tracker = CoverageTracker(e_min, e_max, n_min, n_max)

    pid_x = PIDController(kp=0.85, ki=0.02, kd=0.35, max_output=SEARCH_SPEED)
    pid_y = PIDController(kp=0.85, ki=0.02, kd=0.35, max_output=SEARCH_SPEED)
    pid_z = PIDController(kp=0.80, ki=0.01, kd=0.30, max_output=0.8)

    print(f"[{get_timestamp()}] [{name}] Connecting MAVSDK gRPC:{grpc_port} -> PX4 udp://:{port}")
    drone = System(port=grpc_port)
    await drone.connect(system_address=f"udp://:{port}")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"[{get_timestamp()}] [{name}] Connected to PX4!")
            break

    print(f"[{get_timestamp()}] [{name}] Waiting for GPS fix ...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print(f"[{get_timestamp()}] [{name}] GPS fix OK!")
            break

    try:
        # Must clear the obstacle field, otherwise PX4's own RTL (our fallback)
        # flies home through the trees too.
        await drone.param.set_param_float("RTL_RETURN_ALT", RTL_ALT)
    except Exception:
        pass

    # Home lat/lon is the origin every geotag is reported against.
    try:
        async for home in drone.telemetry.home():
            swarm_home_geo[drone_index] = (home.latitude_deg, home.longitude_deg)
            print(f"[{get_timestamp()}] [{name}] Geotag origin: "
                  f"{home.latitude_deg:.7f}, {home.longitude_deg:.7f}")
            break
    except Exception as e:
        print(f"[{get_timestamp()}] [{name}] [WARN] No home position, geotags will be local-only: {e}")

    current_pos_ned = [0.0, 0.0, 0.0]
    telemetry_active = False

    async def update_telemetry():
        nonlocal current_pos_ned, telemetry_active
        try:
            async for pos_ned in drone.telemetry.position_velocity_ned():
                current_pos_ned = [
                    pos_ned.position.north_m,
                    pos_ned.position.east_m,
                    pos_ned.position.down_m
                ]
                swarm_drone_positions[drone_index] = current_pos_ned
                swarm_drone_velocities[drone_index] = (
                    pos_ned.velocity.north_m_s,
                    pos_ned.velocity.east_m_s,
                    pos_ned.velocity.down_m_s,
                )
                _sx, _sy = DRONE_SPAWN_POSES[drone_index]
                live_map.record_trail(swarm_drone_trails, drone_index,
                                      current_pos_ned[1] + _sx,
                                      current_pos_ned[0] + _sy)
                telemetry_active = True
        except Exception as e:
            print(f"[{get_timestamp()}] [{name}] [ERROR] Telemetry stream died: {e}")

    async def update_attitude():
        try:
            async for att in drone.telemetry.attitude_euler():
                swarm_drone_yaws[drone_index] = math.radians(att.yaw_deg)
                swarm_drone_attitudes[drone_index] = (
                    math.radians(att.roll_deg),
                    math.radians(att.pitch_deg),
                    math.radians(att.yaw_deg),
                )
        except Exception:
            pass

    asyncio.create_task(update_telemetry())
    asyncio.create_task(update_attitude())

    for _ in range(100):
        if telemetry_active:
            break
        await asyncio.sleep(0.05)

    print(f"[{get_timestamp()}] [{name}] Arming ...")
    await drone.action.arm()
    await drone.action.set_takeoff_altitude(CRUISING_ALT)
    print(f"[{get_timestamp()}] [{name}] Taking off to {CRUISING_ALT}m ...")
    await drone.action.takeoff()

    async for pos in drone.telemetry.position():
        if pos.relative_altitude_m > CRUISING_ALT * 0.85:
            print(f"[{get_timestamp()}] [{name}] Altitude reached: {pos.relative_altitude_m:.1f}m")
            break

    print(f"[{get_timestamp()}] [{name}] Starting offboard mode ...")
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
        print(f"[{get_timestamp()}] [{name}] Offboard mode ACTIVE!")
    except OffboardError as e:
        print(f"[{get_timestamp()}] [{name}] [FATAL] Offboard Start Failed: {e}")
        return

    print(f"[{get_timestamp()}] [{name}] Stabilizing offboard hold (2s) ...")
    stab_start = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - stab_start) < 2.0:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(0.1)

    waypoints = []
    direction = 1
    for e_pos in leg_e_positions:
        if direction == 1:
            waypoints.append((geo_n_min, e_pos, -CRUISING_ALT))
            waypoints.append((geo_n_max, e_pos, -CRUISING_ALT))
        else:
            waypoints.append((geo_n_max, e_pos, -CRUISING_ALT))
            waypoints.append((geo_n_min, e_pos, -CRUISING_ALT))
        direction *= -1

    gazebo_survivor_targets = [
        (10.0, 8.0, 5.5),    # House 1 roof (Top Right)
        (-12.0, 14.0, 4.5),  # House 2 roof (Top Left)
        (-8.0, -10.0, 0.5),  # Ground
        (14.0, -10.0, 6.5),  # House 3 roof (Bottom Right)
        (-15.0, -16.0, 0.5)  # Ground
    ]

    import os, json
    if os.path.exists("/tmp/world_layout.json"):
        try:
            with open("/tmp/world_layout.json") as f:
                layout_data = json.load(f)
                if "survivors" in layout_data:
                    gazebo_survivor_targets = [
                        (s["x"], s["y"], s["z"]) for s in layout_data["survivors"]
                    ]
                    print(f"[{get_timestamp()}] [{name}] Dynamic survivor layout loaded: {len(gazebo_survivor_targets)} targets.")
        except Exception as e:
            print(f"[{get_timestamp()}] [{name}] Failed to load /tmp/world_layout.json: {e}")

    print(f"[{get_timestamp()}] [{name}] ===== STARTING WAYPOINT NAVIGATION WITH VISUAL VERIFICATION =====")
    print(f"[{get_timestamp()}] [{name}]   Sector E: [{e_min:.0f}, {e_max:.0f}]  N: [{n_min:.0f}, {n_max:.0f}]")

    dt = 0.05
    last_log_time = 0.0

    for wp_idx, wp in enumerate(waypoints):
        wp_n, wp_e, wp_d = wp
        wp_n = max(n_min, min(n_max, wp_n))
        wp_e = max(e_min, min(e_max, wp_e))

        print(f"[{get_timestamp()}] [{name}] -> WP {wp_idx+1}/{len(waypoints)}: N={wp_n:.1f} E={wp_e:.1f}")

        pid_x.reset()
        pid_y.reset()

        # Same deadlock risk as the return leg: a leg that runs into an
        # obstacle-induced local minimum would stall the entire sweep. Abandon
        # the leg instead and carry on -- the lawnmower still covers the sector.
        wp_start = asyncio.get_event_loop().time()
        wp_best = float("inf")
        wp_last_progress = wp_start
        wp_escapes = 0

        while True:
            if (asyncio.get_event_loop().time() - wp_start) > WAYPOINT_TIMEOUT:
                print(f"[{get_timestamp()}] [{name}] [WP TIMEOUT] Leg {wp_idx+1} "
                      f"abandoned after {WAYPOINT_TIMEOUT:.0f}s, moving to next")
                break

            # Check survivors the swarm has itself detected and geolocated.
            # These come from the fused registry, NOT from a ground-truth
            # answer key -- this is the active geotagging path.
            for sv in survivor_registry.confirmed_survivors():
                if not sv["delivered"]:
                    s_n = sv["north"] - spawn_y
                    s_e = sv["east"] - spawn_x
                    s_d = -sv["alt"]

                    # Only act on targets inside this drone's assigned sector
                    if e_min <= s_e <= e_max:
                        dist_to_survivor = math.sqrt((current_pos_ned[0] - s_n)**2 + (current_pos_ned[1] - s_e)**2)

                        # try_claim() is atomic, so if both drones confirm the
                        # same survivor only one diverts to deliver.
                        if dist_to_survivor < DETECTION_RADIUS and survivor_registry.try_claim(sv["id"], drone_index):
                            target_hover_alt = max(2.5, sv["alt"] + 2.0)

                            print(f"\n[{get_timestamp()}] [{name}] *** OPENCV RED SURVIVOR VISUALLY DETECTED IN CAMERA FRAME ***")
                            print(f"[{get_timestamp()}] [{name}] Survivor #{sv["id"]} (geotagged, {sv["hits"]} hits) at N={s_n:.2f}m, E={s_e:.2f}m | Descending to Hover Alt {target_hover_alt:.1f}m ...")

                            pid_x.reset()
                            pid_y.reset()
                            pid_z.reset()

                            align_start = asyncio.get_event_loop().time()
                            while (asyncio.get_event_loop().time() - align_start) < 10.0:
                                err_n = s_n - current_pos_ned[0]
                                err_e = s_e - current_pos_ned[1]
                                h_err = math.hypot(err_n, err_e)

                                current_alt = -current_pos_ned[2]
                                err_z = target_hover_alt - current_alt

                                if h_err < 0.12 and abs(err_z) < 0.15:
                                    print(f"[{get_timestamp()}] [{name}] PID converged: h_err={h_err:.3f}m alt_err={err_z:.3f}m")
                                    break

                                vx = pid_x.compute(err_n, dt=dt)
                                vy = pid_y.compute(err_e, dt=dt)
                                vz = -pid_z.compute(err_z, dt=dt)

                                peer_positions = [pos for idx, pos in swarm_drone_positions.items() if idx != drone_index]
                                safe_vel = avoidance.get_avoidance_velocity(
                                    [vx, vy, vz],
                                    my_pos=current_pos_ned,
                                    other_drone_positions=peer_positions,
                                    drone_name=name,
                                    active_rescue=True
                                )

                                await drone.offboard.set_velocity_ned(VelocityNedYaw(safe_vel[0], safe_vel[1], safe_vel[2], 0.0))
                                await asyncio.sleep(dt)

                            await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
                            await asyncio.sleep(3.0)

                            rescue_perception.drop_rescue_package(
                                (s_n, s_e, s_d),
                                drone_hover_alt=target_hover_alt,
                                drone_id=drone_index,
                                drone_name=name
                            )

                            survivor_registry.mark_delivered(sv["id"])
                            log_geotag_accuracy(sv, gazebo_survivor_targets, name)
                            survivor_registry.export(GEOTAG_REPORT_PATH)

                            print(f"[{get_timestamp()}] [{name}] Package dropped! Climbing back to {CRUISING_ALT}m ...")
                            pid_z.reset()
                            climb_back_start = asyncio.get_event_loop().time()
                            while (asyncio.get_event_loop().time() - climb_back_start) < 4.0:
                                current_alt = -current_pos_ned[2]
                                alt_err = CRUISING_ALT - current_alt
                                if abs(alt_err) < 0.3:
                                    break
                                vz_back = -pid_z.compute(alt_err, dt=dt)
                                await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, vz_back, 0.0))
                                await asyncio.sleep(dt)

                            print(f"[{get_timestamp()}] [{name}] Resuming lawnmower sweep")
                            pid_x.reset()
                            pid_y.reset()
                            # A rescue legitimately takes ~20s AND diverts the
                            # drone away from the waypoint. Reset the whole
                            # watchdog, not just the timer: a stale wp_best from
                            # before the detour makes the flight back look like
                            # a stall and fires a spurious escape.
                            wp_start = asyncio.get_event_loop().time()
                            wp_best = math.hypot(wp_n - current_pos_ned[0],
                                                 wp_e - current_pos_ned[1])
                            wp_last_progress = wp_start
                            break

            # Move towards Waypoint
            err_n = wp_n - current_pos_ned[0]
            err_e = wp_e - current_pos_ned[1]
            dist_to_wp = math.hypot(err_n, err_e)

            if dist_to_wp < 1.0:
                cov = tracker.get_coverage_percent()
                print(f"[{get_timestamp()}] [{name}] WP {wp_idx+1} reached! Coverage: {cov:.1f}%")
                break

            # Stall watchdog: closing on the waypoint resets it; standing still
            # against a tree or wedged in a corner triggers a sideways escape.
            now_t = asyncio.get_event_loop().time()
            if dist_to_wp < wp_best - STALL_PROGRESS_EPS:
                wp_best = dist_to_wp
                wp_last_progress = now_t
            elif (now_t - wp_last_progress) > WP_STALL_TIMEOUT and wp_escapes < WP_MAX_ESCAPES:
                wp_escapes += 1
                await escape_local_minimum(
                    drone, name, err_n, err_e, wp_escapes, dt,
                    f"Leg {wp_idx+1} stalled {dist_to_wp:.1f}m short")
                # Re-baseline: the escape deliberately moved us sideways, so the
                # old best distance would instantly re-trigger the watchdog.
                wp_best = math.hypot(wp_n - current_pos_ned[0], wp_e - current_pos_ned[1])
                wp_last_progress = asyncio.get_event_loop().time()
                pid_x.reset()
                pid_y.reset()
                continue

            err_z = CRUISING_ALT - (-current_pos_ned[2])

            vx = pid_x.compute(err_n, dt=dt)
            vy = pid_y.compute(err_e, dt=dt)
            vz = -pid_z.compute(err_z, dt=dt)

            speed = math.hypot(vx, vy)
            if speed > SEARCH_SPEED:
                vx = (vx / speed) * SEARCH_SPEED
                vy = (vy / speed) * SEARCH_SPEED

            peer_positions = [pos for idx, pos in swarm_drone_positions.items() if idx != drone_index]
            safe_vel = avoidance.get_avoidance_velocity([vx, vy, vz], current_pos_ned, peer_positions, drone_name=name, cruise_alt=CRUISING_ALT)

            target_yaw = math.degrees(math.atan2(safe_vel[1], safe_vel[0]))

            await drone.offboard.set_velocity_ned(VelocityNedYaw(safe_vel[0], safe_vel[1], safe_vel[2], target_yaw))

            tracker.mark_visited(current_pos_ned[0], current_pos_ned[1])

            now = time.time()
            if now - last_log_time > 3.0:
                last_log_time = now
                print(f"[{get_timestamp()}] [{name}] pos=({current_pos_ned[0]:.1f},{current_pos_ned[1]:.1f},{current_pos_ned[2]:.1f}) "
                      f"vel=({safe_vel[0]:.2f},{safe_vel[1]:.2f},{safe_vel[2]:.2f}) "
                      f"dist_wp={dist_to_wp:.1f}m VisualDet={latest_camera_detections[drone_index]['detected']}")

            await asyncio.sleep(dt)

    cov = tracker.get_coverage_percent()
    print(f"[{get_timestamp()}] [{name}] Search complete! Final coverage: {cov:.1f}%")
    print(f"[{get_timestamp()}] [{name}] Returning to launch pad ...")
    home_n, home_e = 0.0, 0.0

    # ---- Phase 1: climb above the obstacle field before transiting ----------
    print(f"[{get_timestamp()}] [{name}] Climbing to RTL altitude {RTL_ALT:.1f}m ...")
    pid_z.reset()
    climb_start = asyncio.get_event_loop().time()
    while (asyncio.get_event_loop().time() - climb_start) < RTL_CLIMB_TIMEOUT:
        alt = -current_pos_ned[2]
        if alt >= RTL_ALT - 0.4:
            break
        vz = -pid_z.compute(RTL_ALT - alt, dt=dt)
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, vz, 0.0))
        await asyncio.sleep(dt)
    print(f"[{get_timestamp()}] [{name}] At {-current_pos_ned[2]:.1f}m, transiting home ...")

    # ---- Phase 2: transit, watching for a stall --------------------------
    pid_x.reset()
    pid_y.reset()
    pid_z.reset()

    rtl_start = asyncio.get_event_loop().time()
    best_dist = float("inf")
    last_progress = rtl_start
    escapes = 0
    rtl_timed_out = False

    while True:
        now = asyncio.get_event_loop().time()
        err_n = home_n - current_pos_ned[0]
        err_e = home_e - current_pos_ned[1]
        dist_to_home = math.hypot(err_n, err_e)

        if dist_to_home < 0.8:
            break

        # Closing on home resets the stall watchdog.
        if dist_to_home < best_dist - RTL_PROGRESS_EPS:
            best_dist = dist_to_home
            last_progress = now

        if (now - rtl_start) > RTL_TOTAL_TIMEOUT:
            print(f"[{get_timestamp()}] [{name}] [RTL TIMEOUT] {RTL_TOTAL_TIMEOUT:.0f}s "
                  f"elapsed, still {dist_to_home:.1f}m out -- handing over to PX4 RTL")
            rtl_timed_out = True
            break

        # Wedged: repulsion is cancelling the pull toward home. Slide
        # perpendicular to the home bearing (alternating side each attempt) and
        # climb, which walks the drone around the obstacle instead of into it.
        if (now - last_progress) > RTL_STALL_TIMEOUT:
            escapes += 1
            await escape_local_minimum(
                drone, name, err_n, err_e, escapes, dt,
                f"RTL stalled {dist_to_home:.1f}m from home")

            # Re-baseline so the next stall is measured from here, not from the
            # best distance reached before we deliberately moved sideways.
            best_dist = math.hypot(home_n - current_pos_ned[0], home_e - current_pos_ned[1])
            last_progress = asyncio.get_event_loop().time()
            pid_x.reset()
            pid_y.reset()
            continue

        err_z = RTL_ALT - (-current_pos_ned[2])
        vx = pid_x.compute(err_n, dt=dt)
        vy = pid_y.compute(err_e, dt=dt)
        vz = -pid_z.compute(err_z, dt=dt)

        speed = math.hypot(vx, vy)
        if speed > SEARCH_SPEED:
            vx = (vx / speed) * SEARCH_SPEED
            vy = (vy / speed) * SEARCH_SPEED

        peer_positions = [pos for idx, pos in swarm_drone_positions.items() if idx != drone_index]
        safe_vel = avoidance.get_avoidance_velocity(
            [vx, vy, vz], current_pos_ned, peer_positions,
            drone_name=name, cruise_alt=RTL_ALT)

        await drone.offboard.set_velocity_ned(VelocityNedYaw(safe_vel[0], safe_vel[1], safe_vel[2], 0.0))
        await asyncio.sleep(dt)

    # ---- Phase 3: land ----------------------------------------------------
    try:
        await drone.offboard.stop()
    except Exception:
        pass

    if rtl_timed_out:
        # PX4's built-in RTL has its own path logic and will land at home.
        try:
            await drone.action.return_to_launch()
        except Exception as e:
            print(f"[{get_timestamp()}] [{name}] [WARN] Native RTL failed ({e}), landing in place")
            await drone.action.land()
    else:
        await drone.action.land()

    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print(f"[{get_timestamp()}] [{name}] Landed safely at home pad.")
            break


def start_ros2_node():
    rclpy.init()
    node = RosSensorSubscriberNode()
    rclpy.spin(node)


async def main():
    print("=" * 78)
    print("  NIDAR RescueSwarm - Strict Visual Camera Verification Mission")
    print("=" * 78)

    ros_thread = threading.Thread(target=start_ros2_node, daemon=True)
    ros_thread.start()

    # ---- Live map (browser view of the mission as it flies) ----------------
    global MISSION_START_TS
    MISSION_START_TS = datetime.now().timestamp()
    live_srv = None
    if LIVE_MAP_ENABLED:
        try:
            with open(WORLD_LAYOUT_PATH) as f:
                live_layout = json.load(f)
        except Exception:
            live_layout = {}
        live_srv = live_map.LiveMapServer(
            state_fn=lambda: live_map.build_state(
                swarm_drone_positions, swarm_drone_yaws, swarm_drone_velocities,
                survivor_registry.confirmed_survivors(), DRONE_SPAWN_POSES,
                detections=latest_camera_detections, started=MISSION_START_TS,
                phase="mission in progress", trails=swarm_drone_trails),
            layout=live_layout, port=LIVE_MAP_PORT)
        live_srv.start()

    tasks = [
        asyncio.ensure_future(run_drone_rescue_mission(port, grpc_port, idx))
        for idx, (port, grpc_port) in enumerate(zip(DRONE_PORTS, GRPC_PORTS))
    ]

    await asyncio.gather(*tasks)

    # ---- Final geotag report (what the GCS / operator interface consumes) ----
    report = survivor_registry.export(GEOTAG_REPORT_PATH)
    print("\n" + "=" * 78)
    print(f"  GEOTAG REPORT - {report['confirmed_count']} survivor(s) confirmed")
    print("=" * 78)
    for sv in report["survivors"]:
        geo = (f"{sv['lat']:.7f}, {sv['lon']:.7f}"
               if sv.get("lat") is not None else "no GPS origin")
        print(f"  #{sv['id']:<3} E={sv['east']:8.2f} N={sv['north']:8.2f} "
              f"alt={sv['alt']:6.2f}  hits={sv['hits']:<4} "
              f"drones={sv['seen_by']} delivered={sv['delivered']}  {geo}")
    print("=" * 78)
    print(f"  Written to {GEOTAG_REPORT_PATH}")

    if live_srv is not None:
        print(f"  Live    http://localhost:{LIVE_MAP_PORT} (still serving final state)")

    # ---- Exportable per-run artefacts (HTML plan view + CSV geotag table) ----
    layout = {}
    try:
        with open(WORLD_LAYOUT_PATH) as f:
            layout = json.load(f)
    except Exception:
        pass  # no layout at the real mission; the map renders without it

    html_path, csv_path = mission_map.write_report(
        report, layout, MISSION_REPORT_DIR)
    if html_path:
        print(f"  Map     {html_path}")
        print(f"  CSV     {csv_path}")
        print(f"  Latest  {os.path.join(MISSION_REPORT_DIR, 'latest.html')}")
    print("=" * 78)


if __name__ == "__main__":
    asyncio.run(main())