import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO model with your trained weights
model = YOLO("runs/train/weights/best.pt")  # Replace 'best.pt' with your trained weights
model.to('cuda')  # Use GPU for faster inference

# Configure the Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

# Start the pipeline
pipeline.start(config)

# Align depth to color
align = rs.align(rs.stream.color)

try:
    print("Starting real-time object detection...")
    while True:
        # Wait for a coherent frame
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get color and depth frames
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Perform object detection
        results = model(color_image, conf=0.5)

        # Extract detection results
        for detection in results[0].boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())

            # Compute the center of the bounding box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Get depth at the center of the bounding box
            depth = depth_frame.get_distance(cx, cy)

            # Convert depth to 3D coordinates
            depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
            object_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cx, cy], depth)

            # Draw bounding box and coordinates on the image
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, f"X: {object_3d[0]:.2f}m", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(color_image, f"Y: {object_3d[1]:.2f}m", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(color_image, f"Z: {object_3d[2]:.2f}m", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the image
        cv2.imshow("D455 Object Detection", color_image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
