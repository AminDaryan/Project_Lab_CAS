import os
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

# Clear terminal before running anything
os.system('cls' if os.name == 'nt' else 'clear')

# Load YOLO model for detecting the cuboid
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

# Define real-world 3D cuboid coordinates (5mm x 5mm x 3mm cuboid)
cuboid_size = [0.005, 0.005, 0.003]  # meters (5mm x 5mm x 3mm cuboid)
object_points = np.array([
    [-cuboid_size[0] / 2, -cuboid_size[1] / 2, 0],  # Bottom-left
    [cuboid_size[0] / 2, -cuboid_size[1] / 2, 0],   # Bottom-right
    [cuboid_size[0] / 2, cuboid_size[1] / 2, 0],    # Top-right
    [-cuboid_size[0] / 2, cuboid_size[1] / 2, 0],    # Top-left
], dtype=np.float32)

# Camera matrix and distortion coefficients (assuming no distortion for RealSense)
camera_matrix = np.eye(3)  # Placeholder, replace with actual camera calibration if available
dist_coeffs = np.zeros((4, 1))  # Assuming no distortion

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

        # Convert color image to grayscale
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        ### 1. YOLO Cuboid Detection ###
        results = model(color_image, conf=0.5)
        cube_center = None

        for detection in results[0].boxes:
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())

            # Draw a rectangle around the detected object
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Red rectangle

            # Compute cuboid center
            cube_cx, cube_cy = (x1 + x2) // 2, (y1 + y2) // 2
            depth = depth_frame.get_distance(cube_cx, cube_cy)

            # Convert to 3D coordinates
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            cube_center = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cube_cx, cube_cy], depth)

            # Define image_points using the four corners of the detected bounding box
            image_points = np.array([
                [x1, y1],  # Bottom-left corner
                [x2, y1],  # Bottom-right corner
                [x2, y2],  # Top-right corner
                [x1, y2]   # Top-left corner
            ], dtype=np.float32)

            # Ensure that we have 4 points (for 2D)
            assert len(image_points) == 4, f"Expected 4 points, but got {len(image_points)}"

            # Now, solve the PnP problem
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

            if success:
                # Convert rotation vector to Euler angles (Theta, Omega, Alpha)
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                theta_x = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
                theta_y = np.degrees(np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)))
                theta_z = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))

                # Display angles, coordinates, and distance
                cv2.putText(color_image, f"X: {cube_center[0]:.2f}m", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(color_image, f"Y: {cube_center[1]:.2f}m", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(color_image, f"Z: {cube_center[2]:.2f}m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.putText(color_image, f"Distance: {depth:.2f}m", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(color_image, f"θ: {theta_x:.2f}°", (x1, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(color_image, f"ω: {theta_y:.2f}°", (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(color_image, f"α: {theta_z:.2f}°", (x1, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Print the cuboid's coordinates and angles in the terminal
                print(f"Cuboid center coordinates (X, Y, Z): ({cube_center[0]:.2f}, {cube_center[1]:.2f}, {cube_center[2]:.2f})")
                print(f"Distance to camera: {depth:.2f} meters")
                print(f"Rotation angles (θ, ω, α): ({theta_x:.2f}, {theta_y:.2f}, {theta_z:.2f})")

                # Draw a red dot at the cuboid center
                cv2.circle(color_image, (cube_cx, cube_cy), 5, (0, 0, 255), -1)  # Red dot

        ### 2. Display Final Result ###
        cv2.imshow("Cuboid Detection", color_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
