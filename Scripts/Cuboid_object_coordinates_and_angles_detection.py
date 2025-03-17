import os
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

# Clear terminal before running anything
os.system('cls' if os.name == 'nt' else 'clear')

# Load YOLO model for detecting the cuboid
model = YOLO("runs/detect/train/weights/best.pt")  # Replace with your trained model path
model.to('cuda')  # Use GPU for faster inference

# Configure Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Define real-world 3D cuboid coordinates (5mm x 5mm x 3mm cuboid)
cuboid_size = [0.005, 0.005, 0.003]  # meters
object_points = np.array([
    [-cuboid_size[0] / 2, -cuboid_size[1] / 2, 0],  # Bottom-left
    [cuboid_size[0] / 2, -cuboid_size[1] / 2, 0],   # Bottom-right
    [cuboid_size[0] / 2, cuboid_size[1] / 2, 0],    # Top-right
    [-cuboid_size[0] / 2, cuboid_size[1] / 2, 0],   # Top-left
], dtype=np.float32)

camera_matrix = np.eye(3)  # Placeholder, replace with actual camera calibration
dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

try:
    print("Starting detection...")

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        results = model(color_image, conf=0.5)
        cube_center = None

        for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            cube_cx, cube_cy = (x1 + x2) // 2, (y1 + y2) // 2
            depth = depth_frame.get_distance(cube_cx, cube_cy)
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            cube_center = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cube_cx, cube_cy], depth)

            image_points = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

            if success:
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                quaternion = R.from_matrix(rotation_matrix).as_quat()
                qx, qy, qz, qw = quaternion
                
                cv2.putText(color_image, f"X: {cube_center[0]:.2f}m", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(color_image, f"Y: {cube_center[1]:.2f}m", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(color_image, f"Z: {cube_center[2]:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(color_image, f"Distance: {depth:.2f}m", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(color_image, f"Quaternion: ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})", (x1, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                print(f"Cuboid center (X, Y, Z): ({cube_center[0]:.2f}, {cube_center[1]:.2f}, {cube_center[2]:.2f})")
                print(f"Distance to camera: {depth:.2f}m")
                print(f"Quaternion (qx, qy, qz, qw): ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})")

                # Draw normal arrow
                start_point = (cube_cx, cube_cy)
                end_point = (cube_cx, cube_cy - 30)  # Upward direction
                cv2.arrowedLine(color_image, start_point, end_point, (0, 255, 255), 2)

        cv2.imshow("Cuboid Detection", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
