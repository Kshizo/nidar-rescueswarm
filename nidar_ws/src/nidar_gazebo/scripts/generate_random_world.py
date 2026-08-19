#!/usr/bin/env python3
"""
NIDAR RescueSwarm - Random World & Aerial Obstacle Generator
============================================================
Generates randomized buildings, trees, power towers, debris, and survivors,
writing the new SDF world file and /tmp/world_layout.json metadata.
"""

import os
import json
import random
import math

def generate_random_world(output_sdf_path, layout_json_path="/tmp/world_layout.json"):
    print("[WORLD GENERATOR] Randomizing obstacle, tree & survivor locations for launch...")

    # Designated Drone Spawn Pad locations (Drone 0 at 0,-5 and Drone 1 at 0,+5)
    drone_spawns = [(0.0, -5.0), (0.0, 5.0)]

    # Strictly enforce min 10.0m clearance around launch pads for vertical takeoff
    TAKEOFF_CLEARANCE = 10.0

    # Physical shared launchpad: a single 10x10 m deck centred on the origin, so
    # both spawn points sit on its north/south edges and RTL has a real surface
    # to come home to. Kept deliberately thin -- PX4 spawns the airframe at
    # z=0.1, so a taller deck would drop the drone inside its own collision box.
    LAUNCHPAD_SIZE = 10.0
    LAUNCHPAD_THICK = 0.05
    LAUNCHPAD_CENTER = (0.0, 0.0)

    def is_on_launchpad(x, y, margin=0.5):
        """Ground survivors only need 5 m of spawn clearance, which leaves the
        pad deck itself reachable (e.g. (4, 0) clears both spawns). Nothing may
        be generated on the deck the drones land on."""
        half = LAUNCHPAD_SIZE / 2.0 + margin
        return (abs(x - LAUNCHPAD_CENTER[0]) <= half and
                abs(y - LAUNCHPAD_CENTER[1]) <= half)

    def is_far_from_spawns(x, y, min_dist=TAKEOFF_CLEARANCE):
        for sx, sy in drone_spawns:
            if math.hypot(x - sx, y - sy) < min_dist:
                return False
        return True

    def is_far_from_others(x, y, existing_poses, min_dist=6.0):
        for ex, ey in existing_poses:
            if math.hypot(x - ex, y - ey) < min_dist:
                return False
        return True

    all_obstacle_poses = []
    obstacle_metadata = []

    # 1. Generate 4 Random Houses (one per quadrant, min 9.0m spacing)
    house_sizes = [
        (6.0, 6.0, random.choice([4.0, 5.5, 7.0])),
        (7.0, 5.0, random.choice([4.5, 6.0, 8.0])),
        (8.0, 8.0, random.choice([5.0, 7.0, 8.5])),
        (5.0, 4.0, random.choice([3.5, 5.0, 6.5])),
    ]

    placed_houses = []

    # Quadrants configured outside the 10m central launch corridor
    quadrants = [
        (10.0, 20.0, 8.0, 20.0),      # Top Right Sector
        (-20.0, -10.0, 8.0, 20.0),    # Top Left Sector
        (10.0, 20.0, -20.0, -8.0),    # Bottom Right Sector
        (-20.0, -10.0, -20.0, -8.0),   # Bottom Left Sector
    ]

    for idx, (sx, sy, h) in enumerate(house_sizes):
        q_xmin, q_xmax, q_ymin, q_ymax = quadrants[idx]
        attempts = 0
        while attempts < 300:
            attempts += 1
            hx = round(random.uniform(q_xmin, q_xmax), 1)
            hy = round(random.uniform(q_ymin, q_ymax), 1)
            if is_far_from_spawns(hx, hy, min_dist=TAKEOFF_CLEARANCE) and is_far_from_others(hx, hy, all_obstacle_poses, min_dist=9.0):
                yaw = round(random.uniform(-0.4, 0.4), 2)
                all_obstacle_poses.append((hx, hy))
                placed_houses.append({
                    "name": f"house_{idx+1}",
                    "x": hx, "y": hy, "z": round(h / 2.0, 2),
                    "sx": sx, "sy": sy, "h": h,
                    "yaw": yaw
                })
                obstacle_metadata.append({
                    "name": f"house_{idx+1}", "x": hx, "y": hy, "sx": sx, "sy": sy, "h": h
                })
                break

    # 2. Generate 3 Tall Aerial Trees (outside 10m takeoff zone, min 9.0m spacing)
    placed_trees = []
    for t_idx in range(3):
        attempts = 0
        while attempts < 300:
            attempts += 1
            tx = round(random.uniform(-22.0, 22.0), 1)
            ty = round(random.uniform(-22.0, 22.0), 1)
            tree_h = round(random.uniform(8.5, 12.0), 1)
            canopy_r = round(random.uniform(2.0, 2.8), 1)

            if is_far_from_spawns(tx, ty, min_dist=TAKEOFF_CLEARANCE + canopy_r) and is_far_from_others(tx, ty, all_obstacle_poses, min_dist=9.0):
                all_obstacle_poses.append((tx, ty))
                placed_trees.append({
                    "name": f"tree_{t_idx+1}",
                    "x": tx, "y": ty, "h": tree_h, "canopy_r": canopy_r
                })
                obstacle_metadata.append({
                    "name": f"tree_{t_idx+1}", "x": tx, "y": ty, "sx": canopy_r * 2.0, "sy": canopy_r * 2.0, "h": tree_h
                })
                break

    # 3. Generate 1 Communication / Power Utility Pole (outside 10m takeoff zone)
    placed_towers = []
    for p_idx in range(1):
        attempts = 0
        while attempts < 200:
            attempts += 1
            px = round(random.uniform(-20.0, 20.0), 1)
            py = round(random.uniform(-20.0, 20.0), 1)
            t_h = round(random.uniform(10.0, 13.0), 1)
            if is_far_from_spawns(px, py, min_dist=TAKEOFF_CLEARANCE) and is_far_from_others(px, py, all_obstacle_poses, min_dist=9.0):
                all_obstacle_poses.append((px, py))
                placed_towers.append({
                    "name": f"tower_{p_idx+1}", "x": px, "y": py, "h": t_h
                })
                obstacle_metadata.append({
                    "name": f"tower_{p_idx+1}", "x": px, "y": py, "sx": 2.0, "sy": 2.0, "h": t_h
                })
                break

    # 4. Generate Random Debris (outside 10m takeoff zone)
    debris_list = []
    for d_idx in range(3):
        attempts = 0
        while attempts < 150:
            attempts += 1
            dx = round(random.uniform(-20.0, 20.0), 1)
            dy = round(random.uniform(-20.0, 20.0), 1)
            if is_far_from_spawns(dx, dy, min_dist=TAKEOFF_CLEARANCE) and is_far_from_others(dx, dy, all_obstacle_poses, min_dist=4.0):
                debris_list.append({
                    "name": f"debris_{d_idx+1}",
                    "x": dx, "y": dy, "z": 0.4,
                    "sx": 3.0, "sy": 1.5, "sz": 0.8,
                    "roll": 0.1, "pitch": 0.05, "yaw": round(random.uniform(0, 3.14), 2)
                })
                break

    # 5. Generate 10 Well-Spaced Survivors (5 on unique house roofs, 5 on ground plane)
    survivors = []
    survivor_poses = []

    def is_far_from_survivors(x, y, min_dist=6.0):
        for sx, sy in survivor_poses:
            if math.hypot(x - sx, y - sy) < min_dist:
                return False
        return True

    # Place 1 survivor per rooftop (up to 5 houses)
    for s_idx, h in enumerate(placed_houses[:5]):
        attempts = 0
        while attempts < 50:
            attempts += 1
            off_x = round(random.uniform(-h["sx"]*0.2, h["sx"]*0.2), 1)
            off_y = round(random.uniform(-h["sy"]*0.2, h["sy"]*0.2), 1)
            surv_x = round(h["x"] + off_x, 1)
            surv_y = round(h["y"] + off_y, 1)
            surv_z = round(h["h"] + 0.5, 2)

            if is_far_from_survivors(surv_x, surv_y, min_dist=6.0):
                survivor_poses.append((surv_x, surv_y))
                survivors.append({
                    "name": f"survivor_{len(survivors)+1}",
                    "x": surv_x, "y": surv_y, "z": surv_z,
                    "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                    "location_type": "rooftop",
                    "house_name": h["name"]
                })
                break

    # Place remaining ground survivors with strict min 6.0m spacing
    needed_ground = 10 - len(survivors)
    for g_idx in range(needed_ground):
        attempts = 0
        while attempts < 200:
            attempts += 1
            gx = round(random.uniform(-22.0, 22.0), 1)
            gy = round(random.uniform(-22.0, 22.0), 1)
            if (is_far_from_spawns(gx, gy, min_dist=5.0) and
                not is_on_launchpad(gx, gy) and
                is_far_from_others(gx, gy, all_obstacle_poses, min_dist=3.0) and
                is_far_from_survivors(gx, gy, min_dist=6.0)):

                survivor_poses.append((gx, gy))
                survivors.append({
                    "name": f"survivor_{len(survivors)+1}",
                    "x": gx, "y": gy, "z": 0.5,
                    "roll": 1.57, "pitch": 0.0, "yaw": round(random.uniform(0, 3.14), 2),
                    "location_type": "ground",
                    "house_name": None
                })
                break

    # Save Metadata JSON
    layout_data = {
        "launchpad": {
            "x": LAUNCHPAD_CENTER[0], "y": LAUNCHPAD_CENTER[1],
            "size": LAUNCHPAD_SIZE, "thickness": LAUNCHPAD_THICK,
            "spawns": [list(sp) for sp in drone_spawns],
        },
        "houses": obstacle_metadata,
        "trees": placed_trees,
        "towers": placed_towers,
        "debris": debris_list,
        "survivors": survivors
    }

    with open(layout_json_path, "w") as f:
        json.dump(layout_data, f, indent=2)

    print(f"[WORLD GENERATOR] Saved world layout to {layout_json_path}")
    print(f"  Houses: {len(placed_houses)} | Trees (8-12m): {len(placed_trees)} | Towers: {len(placed_towers)} | Survivors: {len(survivors)}")

    # Build XML SDF string
    sdf_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<sdf version="1.9">
  <world name="rescueswarm_flood_zone">
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>

    <scene>
      <grid>true</grid>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.55 0.65 0.82 1</background>
      <shadows>false</shadows>
    </scene>

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>28.6139</latitude_deg>
      <longitude_deg>77.2090</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry>
          <surface><friction><ode/></friction></surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>50 50</size></plane></geometry>
          <material>
            <ambient>0.12 0.28 0.38 1</ambient>
            <diffuse>0.15 0.35 0.50 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <light name="sunUTC" type="directional">
      <pose>0 0 100 0 0 0</pose>
      <cast_shadows>true</cast_shadows>
      <intensity>1</intensity>
      <direction>0.001 0.625 -0.78</direction>
      <diffuse>0.904 0.904 0.904 1</diffuse>
      <specular>0.271 0.271 0.271 1</specular>
      <attenuation><range>2000</range><linear>0</linear><constant>1</constant><quadratic>0</quadratic></attenuation>
    </light>

    <!-- Shared 10x10 m launchpad. Both drones spawn on its N/S edges and RTL
         returns them here. Thin deck (see LAUNCHPAD_THICK) so the z=0.1 spawn
         sits above the surface rather than inside it. -->
    <model name="launchpad">
      <static>true</static>
      <pose>{LAUNCHPAD_CENTER[0]} {LAUNCHPAD_CENTER[1]} {LAUNCHPAD_THICK/2.0} 0 0 0</pose>
      <link name="link">
        <collision name="col">
          <geometry><box><size>{LAUNCHPAD_SIZE} {LAUNCHPAD_SIZE} {LAUNCHPAD_THICK}</size></box></geometry>
          <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
        </collision>
        <visual name="deck">
          <geometry><box><size>{LAUNCHPAD_SIZE} {LAUNCHPAD_SIZE} {LAUNCHPAD_THICK}</size></box></geometry>
          <material><ambient>0.16 0.17 0.19 1</ambient><diffuse>0.22 0.23 0.26 1</diffuse></material>
        </visual>
        <visual name="border">
          <pose>0 0 {LAUNCHPAD_THICK*0.55} 0 0 0</pose>
          <geometry><box><size>{LAUNCHPAD_SIZE-0.6} {LAUNCHPAD_SIZE-0.6} {LAUNCHPAD_THICK*0.2}</size></box></geometry>
          <material><ambient>0.85 0.85 0.15 1</ambient><diffuse>0.95 0.95 0.2 1</diffuse></material>
        </visual>
        <visual name="deck_inner">
          <pose>0 0 {LAUNCHPAD_THICK*0.62} 0 0 0</pose>
          <geometry><box><size>{LAUNCHPAD_SIZE-1.4} {LAUNCHPAD_SIZE-1.4} {LAUNCHPAD_THICK*0.2}</size></box></geometry>
          <material><ambient>0.16 0.17 0.19 1</ambient><diffuse>0.22 0.23 0.26 1</diffuse></material>
        </visual>
        <visual name="mark_v">
          <pose>0 0 {LAUNCHPAD_THICK*0.7} 0 0 0</pose>
          <geometry><box><size>1.0 4.4 {LAUNCHPAD_THICK*0.2}</size></box></geometry>
          <material><ambient>0.85 0.85 0.15 1</ambient><diffuse>0.95 0.95 0.2 1</diffuse></material>
        </visual>
        <visual name="mark_h">
          <pose>0 0 {LAUNCHPAD_THICK*0.7} 0 0 0</pose>
          <geometry><box><size>3.0 1.0 {LAUNCHPAD_THICK*0.2}</size></box></geometry>
          <material><ambient>0.85 0.85 0.15 1</ambient><diffuse>0.95 0.95 0.2 1</diffuse></material>
        </visual>
      </link>
    </model>
"""

    # Add Houses to SDF
    for h in placed_houses:
        sdf_content += f"""
    <model name="{h['name']}">
      <static>true</static>
      <pose>{h['x']} {h['y']} {h['z']} 0 0 {h['yaw']}</pose>
      <link name="link">
        <collision name="col"><geometry><box><size>{h['sx']} {h['sy']} {h['h']}</size></box></geometry></collision>
        <visual name="walls"><geometry><box><size>{h['sx']} {h['sy']} {h['h']}</size></box></geometry>
          <material><ambient>0.55 0.38 0.28 1</ambient><diffuse>0.65 0.45 0.35 1</diffuse></material>
        </visual>
      </link>
    </model>
"""

    # Add Trees (Single link containing Trunk + Foliage Canopy collisions for proper Gazebo raycasting)
    for t in placed_trees:
        trunk_h = t['h'] * 0.6
        canopy_z = t['h'] - t['canopy_r'] * 0.5
        sdf_content += f"""
    <model name="{t['name']}">
      <static>true</static>
      <pose>{t['x']} {t['y']} 0 0 0 0</pose>
      <link name="link">
        <!-- Trunk -->
        <pose>0 0 0 0 0 0</pose>
        <collision name="trunk_col">
          <pose>0 0 {trunk_h/2.0} 0 0 0</pose>
          <geometry><cylinder><radius>0.4</radius><length>{trunk_h}</length></cylinder></geometry>
        </collision>
        <visual name="trunk_vis">
          <pose>0 0 {trunk_h/2.0} 0 0 0</pose>
          <geometry><cylinder><radius>0.4</radius><length>{trunk_h}</length></cylinder></geometry>
          <material><ambient>0.3 0.2 0.1 1</ambient><diffuse>0.4 0.25 0.15 1</diffuse></material>
        </visual>
        <!-- Foliage Canopy -->
        <collision name="canopy_col">
          <pose>0 0 {canopy_z} 0 0 0</pose>
          <geometry><sphere><radius>{t['canopy_r']}</radius></sphere></geometry>
        </collision>
        <visual name="canopy_vis">
          <pose>0 0 {canopy_z} 0 0 0</pose>
          <geometry><sphere><radius>{t['canopy_r']}</radius></sphere></geometry>
          <material><ambient>0.1 0.4 0.15 1</ambient><diffuse>0.15 0.55 0.2 1</diffuse></material>
        </visual>
      </link>
    </model>
"""

    # Add Utility / Communication Towers
    for tw in placed_towers:
        tw_h = tw['h']
        sdf_content += f"""
    <model name="{tw['name']}">
      <static>true</static>
      <pose>{tw['x']} {tw['y']} {tw_h/2.0} 0 0 0</pose>
      <link name="link">
        <collision name="col"><geometry><box><size>0.8 0.8 {tw_h}</size></box></geometry></collision>
        <visual name="vis"><geometry><box><size>0.8 0.8 {tw_h}</size></box></geometry>
          <material><ambient>0.2 0.2 0.25 1</ambient><diffuse>0.3 0.3 0.35 1</diffuse></material>
        </visual>
      </link>
    </model>
"""

    # Add Debris to SDF
    for d in debris_list:
        sdf_content += f"""
    <model name="{d['name']}">
      <static>true</static>
      <pose>{d['x']} {d['y']} {d['z']} {d['roll']} {d['pitch']} {d['yaw']}</pose>
      <link name="link">
        <visual name="vis"><geometry><box><size>{d['sx']} {d['sy']} {d['sz']}</size></box></geometry>
          <material><ambient>0.35 0.3 0.25 1</ambient><diffuse>0.4 0.35 0.3 1</diffuse></material>
        </visual>
        <collision name="col"><geometry><box><size>{d['sx']} {d['sy']} {d['sz']}</size></box></geometry></collision>
      </link>
    </model>
"""

    # Add Survivors (Red Capsules) to SDF
    for s in survivors:
        sdf_content += f"""
    <model name="{s['name']}">
      <static>true</static>
      <pose>{s['x']} {s['y']} {s['z']} {s['roll']} {s['pitch']} {s['yaw']}</pose>
      <link name="link">
        <visual name="body"><geometry><capsule><radius>0.25</radius><length>0.9</length></capsule></geometry>
          <material><ambient>0.9 0.1 0.1 1</ambient><diffuse>1.0 0.15 0.15 1</diffuse></material>
        </visual>
        <collision name="col"><geometry><capsule><radius>0.25</radius><length>0.9</length></capsule></geometry></collision>
      </link>
    </model>
"""

    sdf_content += """
  </world>
</sdf>
"""

    with open(output_sdf_path, "w") as f:
        f.write(sdf_content)

    print(f"[WORLD GENERATOR] Successfully generated fresh randomized SDF world -> {output_sdf_path}")


if __name__ == "__main__":
    # Derive the workspace root from this file's location (scripts/ -> pkg -> src -> ws)
    nidar_ws = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    sdf_path = os.path.join(nidar_ws, "src/nidar_gazebo/worlds/rescueswarm_flood_zone.sdf")
    generate_random_world(sdf_path)
