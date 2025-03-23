import pyrealsense2 as rs
import numpy as np
import cv2
import os

# Initialize the pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure RGB and Depth streams
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Align depth to color stream
align_to = rs.stream.color
align = rs.align(align_to)

# Set up a folder to save data
save_path = "dataset/"
os.makedirs(save_path, exist_ok=True)

frame_count = 0

try:
    while True:
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

        # Save images
        cv2.imwrite(f"{save_path}/color_{frame_count:04d}.png", color_image)
        cv2.imwrite(f"{save_path}/depth_{frame_count:04d}.png", depth_image)

        frame_count += 1

        # Display the color image
        cv2.imshow("Recording RGB-D Data", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
