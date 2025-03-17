import pyrealsense2 as rs
import cv2
import os
import numpy as np
import time

# Set up RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Folder to save images
image_folder = 'datasets/images'
os.makedirs(image_folder, exist_ok=True)

# Capture frames from the RealSense camera
frame_count = 0
start_time = time.time()

try:
    while time.time() - start_time < 50:  # Limit to 10 seconds
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to numpy array (OpenCV compatible)
        color_image = np.asanyarray(color_frame.get_data())

        # Save the frame as an image
        image_filename = f'{image_folder}/frame_{frame_count:04d}.jpg'
        cv2.imwrite(image_filename, color_image)
        frame_count += 1

        # Display the captured frame (optional)
        cv2.imshow("RealSense Camera", color_image)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.1)  # Add a delay of 0.05s between frames

finally:
    # Release the pipeline and close the OpenCV window
    pipeline.stop()
    cv2.destroyAllWindows()
