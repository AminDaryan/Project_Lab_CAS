import os
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

# Clear terminal before running anything
os.system('cls' if os.name == 'nt' else 'clear')

# Load YOLO model for detecting the white cube
model = YOLO("runs/detect/train2/weights/best.pt")  # Replace with your trained model path
model.to('cuda')  # Use GPU for faster inference

# Configure Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

# Define real-world 3D cube coordinates (assuming a 10 cm cube)
cube_size = 0.1  # meters (adjust based on actual cube size)
object_points = np.array([
    [-cube_size / 2, -cube_size / 2, 0],  # Bottom-left
    [cube_size / 2, -cube_size / 2, 0],   # Bottom-right
    [cube_size / 2, cube_size / 2, 0],    # Top-right
    [-cube_size / 2, cube_size / 2, 0]    # Top-left
], dtype=np.float32)

try:
    print("Starting detection...")

    while True:
        # Capture frames and align
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert to NumPy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        ### 1. YOLO Cube Detection ###
        results = model(color_image, conf=0.5)
        cube_center = None

        for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())

            # Compute cube center
            cube_cx, cube_cy = (x1 + x2) // 2, (y1 + y2) // 2
            depth = depth_frame.get_distance(cube_cx, cube_cy)

            # Convert to 3D coordinates
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            cube_center = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cube_cx, cube_cy], depth)

            # Draw bounding box & cube depth
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, f"Z: {cube_center[2]:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        ### 2. Display Final Result ###
        cv2.imshow("White Cube Detection", color_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
