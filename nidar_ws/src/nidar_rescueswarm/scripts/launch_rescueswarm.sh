#!/usr/bin/env bash
# ============================================================================
# NIDAR RescueSwarm - Multi-Drone PX4 SITL Launch Script
# Launches Gazebo Sim with flood zone world + 2 PX4 x500_depth drones
# ============================================================================
set -e

PX4_DIR="${HOME}/PX4-Autopilot"
PX4_BUILD="${PX4_DIR}/build/px4_sitl_default"

# World file lives in nidar_gazebo package
# Derive the workspace root from this script's own location (scripts/ -> pkg -> src -> ws)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIDAR_WS="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORLD_FILE="${NIDAR_WS}/src/nidar_gazebo/worlds/rescueswarm_flood_zone.sdf"

# PX4 model paths
PX4_GZ_MODELS="${PX4_DIR}/Tools/simulation/gz/models"
PX4_GZ_WORLDS="${PX4_DIR}/Tools/simulation/gz/worlds"
PX4_SERVER_CONFIG="${PX4_DIR}/Tools/simulation/gz/server.config"

NUM_DRONES=2

# Drone spawn positions (x,y,z,roll,pitch,yaw) — 5 meters apart
DRONE_POSES=(
    "0,-5.0,0.1,0,0,0"
    "0,5.0,0.1,0,0,0"
)

echo "=============================================="
echo "  NIDAR RescueSwarm — PX4 Multi-Drone SITL"
echo "  Drones: ${NUM_DRONES} x x500_depth (OakD Camera + Aid Box)"
echo "  World:  rescueswarm_flood_zone"
echo "=============================================="

# Generate fresh randomized obstacle & survivor world for this launch
python3 "${NIDAR_WS}/src/nidar_gazebo/scripts/generate_random_world.py"

# Check that the world file exists
if [ ! -f "${WORLD_FILE}" ]; then
    echo "[ERROR] World file not found: ${WORLD_FILE}"
    exit 1
fi

# ------------------------------------------------------------------
# 1. Set Gazebo resource paths so it finds PX4 & custom models
# ------------------------------------------------------------------
export GZ_SIM_RESOURCE_PATH="${PX4_GZ_MODELS}:${PX4_GZ_WORLDS}:${GZ_SIM_RESOURCE_PATH:-}"

if [ -d "${HOME}/.simulation-gazebo/models" ]; then
    export GZ_SIM_RESOURCE_PATH="${HOME}/.simulation-gazebo/models:${GZ_SIM_RESOURCE_PATH}"
fi
if [ -d "${HOME}/gz_models" ]; then
    export GZ_SIM_RESOURCE_PATH="${HOME}/gz_models:${GZ_SIM_RESOURCE_PATH}"
fi
if [ -d "${HOME}/gz_worlds" ]; then
    export GZ_SIM_RESOURCE_PATH="${HOME}/gz_worlds:${GZ_SIM_RESOURCE_PATH}"
fi

# OGRE Render & Qt Optimization
export OGRE_RTT_MODE=Copy
export QT_X11_NO_MITSHM=1

# ------------------------------------------------------------------
# Force rendering onto the discrete NVIDIA GPU (hybrid-graphics laptops).
# Without this, gz sim can silently fall back to Mesa llvmpipe and render
# every camera on the CPU, which pegs all cores and freezes the desktop.
# Set NIDAR_NO_PRIME=1 to skip (e.g. on a machine with no NVIDIA GPU).
# ------------------------------------------------------------------
if [ -z "${NIDAR_NO_PRIME:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    export __NV_PRIME_RENDER_OFFLOAD=1
    export __GLX_VENDOR_LIBRARY_NAME=nvidia
    export __VK_LAYER_NV_optimus=NVIDIA_only
    echo "[INFO] NVIDIA PRIME render offload enabled"
fi

# HEADLESS=1 runs the Gazebo server only (no GUI window). Sensors still
# render offscreen, so the mission and the OpenCV viewer work unchanged --
# this just drops the ~1 full render pass the GUI costs.
if [ "${HEADLESS:-0}" = "1" ]; then
    GZ_ARGS="-s -r"
    echo "[INFO] HEADLESS=1 -> running Gazebo server only (no GUI)"
else
    GZ_ARGS="-r"
fi

# Use PX4's server.config and built system plugins
export GZ_SIM_SERVER_CONFIG_PATH="${PX4_SERVER_CONFIG}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${PX4_BUILD}/src/modules/simulation/gz_plugins:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"

echo "[INFO] GZ_SIM_RESOURCE_PATH=${GZ_SIM_RESOURCE_PATH}"
echo "[INFO] GZ_SIM_SERVER_CONFIG_PATH=${GZ_SIM_SERVER_CONFIG_PATH}"
echo "[INFO] GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH}"
echo "[INFO] World file: ${WORLD_FILE}"

# ------------------------------------------------------------------
# 2. Kill any leftover processes from previous runs
# ------------------------------------------------------------------
pkill -f "gz sim" 2>/dev/null || true
pkill -f "ruby.*gz" 2>/dev/null || true
pkill -f "ros_gz_bridge" 2>/dev/null || true
# Stale PX4 SITL instances hold a per-instance lock: a leftover process makes
# the new one exit with "PX4 server already running for instance N", leaving
# the mission talking to a PX4 whose Gazebo no longer exists.
pkill -f "px4_sitl_default/bin/px4" 2>/dev/null || true
sleep 2

# ------------------------------------------------------------------
# 3. Launch Gazebo Sim with the flood zone world (background)
# ------------------------------------------------------------------
echo "[INFO] Starting Gazebo Sim ..."
gz sim ${GZ_ARGS} "${WORLD_FILE}" &
GZ_PID=$!
sleep 6
echo "[INFO] Gazebo started (PID: ${GZ_PID})"

# ------------------------------------------------------------------
# 4. Spawn PX4 SITL instances with x500_depth (Autostart 4002)
# ------------------------------------------------------------------
PX4_PIDS=()

for i in $(seq 0 $((NUM_DRONES - 1))); do
    INSTANCE=$i
    POSE="${DRONE_POSES[$i]}"

    echo "[INFO] Spawning PX4 instance ${INSTANCE} (x500_depth) at pose: ${POSE}"

    INSTANCE_DIR="/tmp/px4_sitl_rescueswarm_${INSTANCE}"
    rm -rf "${INSTANCE_DIR}"
    mkdir -p "${INSTANCE_DIR}"

    PX4_SYS_AUTOSTART=4002 \
    PX4_SIM_MODEL=gz_x500_depth \
    PX4_GZ_MODEL=x500_depth \
    PX4_GZ_MODEL_POSE="${POSE}" \
    PX4_GZ_WORLD=rescueswarm_flood_zone \
    PX4_GZ_STANDALONE=1 \
    PX4_GZ_NO_FOLLOW=1 \
    PX4_GZ_SIM_RENDER_ENGINE=ogre2 \
    ${PX4_BUILD}/bin/px4 \
        -i ${INSTANCE} \
        -d "${PX4_BUILD}/etc" \
        -w "${INSTANCE_DIR}" \
        -s "${PX4_BUILD}/etc/init.d-posix/rcS" \
        > "/tmp/px4_rescueswarm_${INSTANCE}.log" 2>&1 &

    PX4_PIDS+=($!)
    echo "[INFO] PX4 instance ${INSTANCE} started (PID: ${PX4_PIDS[-1]})"
    echo "       MAVSDK port: udp://:$((14540 + INSTANCE))"
    echo "       QGC port:    udp://:$((18570 + INSTANCE))"
    echo "       Log: /tmp/px4_rescueswarm_${INSTANCE}.log"

    sleep 5
done

# ------------------------------------------------------------------
# 5. Start ROS 2 <-> Gazebo Harmonic LiDAR Bridge for Both Drones
# ------------------------------------------------------------------
echo "[INFO] Launching ROS 2 LiDAR Bridge for Drone 0 & Drone 1 ..."
if [ -z "${ROS_DISTRO:-}" ]; then
    for _d in jazzy humble iron rolling; do
        if [ -f "/opt/ros/${_d}/setup.bash" ]; then
            source "/opt/ros/${_d}/setup.bash"
            break
        fi
    done
fi

ros2 run ros_gz_bridge parameter_bridge \
  /world/rescueswarm_flood_zone/model/x500_depth_0/link/link/sensor/lidar_2d_v2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan \
  /world/rescueswarm_flood_zone/model/x500_depth_1/link/link/sensor/lidar_2d_v2/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan \
  /world/rescueswarm_flood_zone/model/x500_depth_0/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /world/rescueswarm_flood_zone/model/x500_depth_1/link/camera_link/sensor/IMX214/image@sensor_msgs/msg/Image[gz.msgs.Image \
  /world/rescueswarm_flood_zone/model/x500_depth_0/link/camera_link/sensor/StereoOV7251/depth_image@sensor_msgs/msg/Image[gz.msgs.Image \
  /world/rescueswarm_flood_zone/model/x500_depth_1/link/camera_link/sensor/StereoOV7251/depth_image@sensor_msgs/msg/Image[gz.msgs.Image \
  --ros-args \
  -r /world/rescueswarm_flood_zone/model/x500_depth_0/link/link/sensor/lidar_2d_v2/scan:=/drone_0/scan \
  -r /world/rescueswarm_flood_zone/model/x500_depth_1/link/link/sensor/lidar_2d_v2/scan:=/drone_1/scan \
  -r /world/rescueswarm_flood_zone/model/x500_depth_0/link/camera_link/sensor/IMX214/image:=/drone_0/camera/image_raw \
  -r /world/rescueswarm_flood_zone/model/x500_depth_1/link/camera_link/sensor/IMX214/image:=/drone_1/camera/image_raw \
  -r /world/rescueswarm_flood_zone/model/x500_depth_0/link/camera_link/sensor/StereoOV7251/depth_image:=/drone_0/depth/image_raw \
  -r /world/rescueswarm_flood_zone/model/x500_depth_1/link/camera_link/sensor/StereoOV7251/depth_image:=/drone_1/depth/image_raw > /tmp/ros_gz_bridge.log 2>&1 &

BRIDGE_PID=$!

echo ""
echo "=============================================="
echo "  All ${NUM_DRONES} drones spawned!"
echo ""
echo "  Drone 0: MAVSDK udp://:14540 (West, initial Y=-5m)"
echo "  Drone 1: MAVSDK udp://:14541 (East, initial Y=+5m)"
echo ""
echo "  Run the RescueSwarm Mission & Perception Controller:"
echo "    python3 ${NIDAR_WS}/src/nidar_rescueswarm/scripts/rescueswarm_mission.py"
echo ""
echo "  Press Ctrl+C to shut down everything."
echo "=============================================="

cleanup() {
    echo ""
    echo "[INFO] Shutting down..."
    kill "$BRIDGE_PID" 2>/dev/null || true
    for pid in "${PX4_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    kill "$GZ_PID" 2>/dev/null || true
    sleep 1
    pkill -f "px4" 2>/dev/null || true
    pkill -f "gz sim" 2>/dev/null || true
    pkill -f "ros_gz_bridge" 2>/dev/null || true
    echo "[INFO] Done."
    exit 0
}
trap cleanup SIGINT SIGTERM

wait $GZ_PID
