#!/usr/bin/env python3
"""
NIDAR RescueSwarm - Live Nadir Camera Viewer & Visual Red Survivor Detector
=============================================================================
Subscribes to ROS 2 camera topics (/drone_0/camera/image_raw, /drone_1/camera/image_raw)
and displays real-time OpenCV windows with bounding boxes drawn over red survivors.
"""

import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from datetime import datetime


class SurvivorCameraViewerNode(Node):
    def __init__(self):
        super().__init__('survivor_camera_viewer')
        self.bridge = CvBridge()

        # Latest images & detection states
        self.img_drone0 = None
        self.img_drone1 = None
        self.det_drone0 = False
        self.det_drone1 = False

        # ROS 2 Subscribers for Drone 0 & Drone 1 RGB Cameras
        # Best-effort, depth 1: drop stale frames rather than queue them up.
        self.sub_cam0 = self.create_subscription(
            Image,
            '/drone_0/camera/image_raw',
            self.cb_cam0,
            qos_profile_sensor_data
        )
        self.sub_cam1 = self.create_subscription(
            Image,
            '/drone_1/camera/image_raw',
            self.cb_cam1,
            qos_profile_sensor_data
        )

        # Timer to render OpenCV windows at 15 FPS (matches the camera rate --
        # rendering faster than the sensor publishes just burns CPU redrawing
        # identical frames).
        self.timer = self.create_timer(0.066, self.render_gui)

        self.get_logger().info("NIDAR Survivor Camera Viewer Started. Listening on /drone_0/camera/image_raw and /drone_1/camera/image_raw ...")

    def detect_survivor_and_annotate(self, cv_image, drone_name="Drone-0"):
        if cv_image is None:
            return False, cv_image

        annotated = cv_image.copy()
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # HSV red color range (handles hue wrap-around 0-10 & 165-180)
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
            if area > 60:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(h) / w if w > 0 else 0
                if 0.3 <= aspect_ratio <= 4.5:
                    if area > max_area:
                        max_area = area
                        best_target = (x, y, w, h, cnt)

        h_img, w_img = annotated.shape[:2]
        center_x, center_y = w_img // 2, h_img // 2

        # Draw Nadir Camera Crosshair
        cv2.line(annotated, (center_x - 15, center_y), (center_x + 15, center_y), (255, 255, 255), 1)
        cv2.line(annotated, (center_x, center_y - 15), (center_x, center_y + 15), (255, 255, 255), 1)

        detected = False
        if best_target is not None:
            detected = True
            x, y, w, h, cnt = best_target
            u_c = x + w // 2
            v_c = y + h // 2

            # Green Bounding Box around Red Survivor
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Red Target Centroid Dot
            cv2.circle(annotated, (u_c, v_c), 5, (0, 0, 255), -1)

            # Draw vector from camera center to survivor centroid
            cv2.line(annotated, (center_x, center_y), (u_c, v_c), (0, 255, 255), 2)

            # Bounding Box Label
            label = f"SURVIVOR ({u_c},{v_c}) Area:{int(max_area)}px"
            cv2.putText(annotated, label, (x, max(25, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # HUD Banner - SURVIVOR DETECTED
            cv2.rectangle(annotated, (0, 0), (w_img, 35), (0, 180, 0), -1)
            cv2.putText(annotated, f"[{drone_name}] *** SURVIVOR DETECTED IN CAMERA FRAME ***", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            # HUD Banner - SEARCHING...
            cv2.rectangle(annotated, (0, 0), (w_img, 35), (40, 40, 40), -1)
            cv2.putText(annotated, f"[{drone_name}] SEARCHING FOR SURVIVORS...", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)

        return detected, annotated

    def cb_cam0(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.det_drone0, self.img_drone0 = self.detect_survivor_and_annotate(cv_img, "Drone-0")
        except Exception as e:
            self.get_logger().error(f"Cam0 error: {e}")

    def cb_cam1(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.det_drone1, self.img_drone1 = self.detect_survivor_and_annotate(cv_img, "Drone-1")
        except Exception as e:
            self.get_logger().error(f"Cam1 error: {e}")

    def render_gui(self):
        if self.img_drone0 is not None:
            cv2.imshow("NIDAR - Drone 0 Nadir Camera Feed", self.img_drone0)

        if self.img_drone1 is not None:
            cv2.imshow("NIDAR - Drone 1 Nadir Camera Feed", self.img_drone1)

        cv2.waitKey(1)


def main():
    rclpy.init()
    node = SurvivorCameraViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
