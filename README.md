# NIDAR RescueSwarm & AirMouse - Autonomous Multi-Drone Rescue Swarm

Official autonomous multi-drone search, perception, and aid delivery system developed for the **NIDAR Innovation Challenge**.

---

## 🚀 Overview

**NIDAR RescueSwarm** is an autonomous multi-drone swarm system engineered for post-disaster flood response and search-and-rescue operations across a 10-hectare search grid.

### Key Mission Capabilities:
* **Collaborative Lawnmower Search**: Sector-based search partitioning between Drone 0 (West Sector) and Drone 1 (East Sector) at 4.0 m/s cruise speed.
* **Strict Visual Camera Verification**: Real-time OpenCV HSV color segmentation on live RGB camera feeds (`/drone_0/camera/image_raw`, `/drone_1/camera/image_raw`). Drones initiate rescue mode **only** when a red survivor capsule is visually detected in the camera frame.
* **Autonomous Aid Delivery**: Pinpoint overhead descent and physical payload dropping over survivors.
* **Sensor-Driven LiDAR & Dual-Tier Collision Avoidance**: 360° LiDAR obstacle processing with Body-to-NED rotational transformations, smooth penetration-based spatial repulsion, emergency push-back, and automatic vertical climb-over.
* **Dynamic World & Survivor Generator**: Launch-time procedural generation of randomized buildings, 8-12m tall trees, power towers, debris, and 10 well-spaced survivors with 10m takeoff clearance zones.
* **Live Nadir HUD Viewer**: OpenCV HUD display (`view_survivor_camera.py`) showing real-time camera feeds, bounding boxes, crosshairs, distance vectors, and detection telemetry.

---

## 📁 Repository Structure

```
nidar/
├── README.md
├── Mission Brief - NIDAR RescueSwarm.pdf
├── Mission Brief - NIDAR AirMouse.pdf
└── nidar_ws/
    └── src/
        ├── nidar_rescueswarm/       # Mission controller, perception & avoidance
        │   └── scripts/
        │       ├── rescueswarm_mission.py     # Main mission controller & ROS 2 node
        │       ├── obstacle_avoidance.py      # Dual-tier LiDAR & spatial avoidance
        │       ├── perception_and_rescue.py   # OpenCV detector & payload release
        │       ├── view_survivor_camera.py    # Real-time OpenCV camera viewer HUD
        │       └── launch_rescueswarm.sh      # SITL multi-drone launcher
        ├── nidar_gazebo/            # Worlds & procedural world generator
        │   ├── scripts/
        │   │   └── generate_random_world.py   # Random world & survivor generator
        │   └── worlds/
        │       └── rescueswarm_flood_zone.sdf # Generated SDF world file
        ├── nidar_airmouse/          # Indoor GPS-denied mapping & search
        ├── nidar_description/       # URDF / SDF robot descriptions
        ├── nidar_gcs/               # Ground control station interface
        └── nidar_interfaces/        # Custom ROS 2 interfaces
```

---

## 🛠️ Quick Start

### 1. Build the Workspace
```bash
cd nidar_ws
colcon build --symlink-install
source install/setup.bash
```

### 2. Launch Simulation (Terminal 1)
```bash
cd nidar_ws/src/nidar_rescueswarm/scripts
./launch_rescueswarm.sh
```

### 3. Open Live OpenCV Nadir Camera Viewer (Terminal 2)
```bash
python3 nidar_ws/src/nidar_rescueswarm/scripts/view_survivor_camera.py
```

### 4. Run Mission Controller (Terminal 3)
```bash
python3 nidar_ws/src/nidar_rescueswarm/scripts/rescueswarm_mission.py
```

---

## 🎥 Real-Flight Survivor Detection (offline)

On the real drone, survivors are people, not red capsules. The Raspberry Pi 5
records the downward webcam and the flight controller's telemetry on one clock;
detection and geotagging run afterwards on a GPU laptop.

```bash
# 1. Once per webcam (same resolution you fly at, autofocus off)
python3 scripts/calibrate_camera.py --camera 0 --width 1280 --height 720

# 2. On the Pi, during the flight (PX4 TELEM2 -> Pi UART)
python3 scripts/flight_recorder.py --mavlink /dev/ttyAMA0 --baud 921600 --fps 15

# 3. On the laptop, after landing
python3 scripts/process_flight.py ~/flights/flight_YYYYmmdd_HHMMSS --calib webcam_calib.yaml
```

Paths are relative to `nidar_ws/src/nidar_rescueswarm/`. Step 3 writes
`results/survivors.{csv,geojson,kml}` plus a crop of every survivor for
visual confirmation. The YOLO26 person model (VisDrone + C2A + SARD +
HERIDAL) and how to retrain it are in
[`training/README.md`](nidar_ws/src/nidar_rescueswarm/training/README.md).
Live detection in the sim stays on HSV unless `NIDAR_DETECTOR=yolo` is set.

---

## 🛡️ Technical Highlights

### Collision Avoidance System
* **Tier 1 (Real-Time 360° LiDAR)**: Processes scan ranges in Body Frame ($+X$ forward, $+Y$ left) and rotates them into World NED coordinates ($+X$ North, $+Y$ East) using attitude yaw telemetry.
* **Tier 2 (Spatial Boundary Guard)**: Evaluates distance to all generated buildings, trees, and towers. Applies penetration-scaled horizontal repulsion ($F_{\text{max}} = 1.2\text{ m/s}$) and gentle climb-over ($V_z = -0.8\text{ m/s}$ UP).
* **Emergency Push-Back**: Immediate reverse push and vertical climb when obstacle distance $< 1.2\text{m}$.

### World Generation & Safety
* **10m Takeoff Corridor**: Guaranteed obstacle-free $20\text{m} \times 20\text{m}$ central launch zone around spawn pads $(0, -5)$ and $(0, +5)$.
* **9m Flight Corridors**: Minimum 9.0m clearance between all placed obstacles.
* **Survivor Spacing**: 10 survivors placed with a minimum 6.0m Euclidean distance separation filter.

---

## 📜 License & Credits

Developed for the NIDAR Autonomous Drone Innovation Challenge.
