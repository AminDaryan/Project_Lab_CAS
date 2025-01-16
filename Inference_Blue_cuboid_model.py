import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO


# Load the YOLO model with your trained weights
model = YOLO('runs/train/weights/best.pt')

# Configure the Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

try:
    print("Starting object detection...")
    while True:
        # Wait for a coherent frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert RealSense frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Perform object detection using YOLO
        results = model(color_image)

        # Render results on the image
        annotated_image = results[0].plot()

        # Display the annotated image
        cv2.imshow("D455 Object Detection", annotated_image)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline and close OpenCV windows
    pipeline.stop()
    cv2.destroyAllWindows()
