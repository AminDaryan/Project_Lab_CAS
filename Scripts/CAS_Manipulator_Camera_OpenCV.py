import cv2
import pyrealsense2 as rs
import numpy as np

# Configure RealSense pipeline
pipeline = rs.pipeline()

# Configure streams: color and depth
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)   # Depth stream

# Start streaming
pipeline.start(config)

# Create OpenCV window for display
cv2.namedWindow('Color Stream', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('Depth Stream', cv2.WINDOW_AUTOSIZE)

try:
    while True:
        # Wait for a coherent set of frames: color and depth
        frames = pipeline.wait_for_frames()

        # Get color frame and depth frame
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Convert RealSense frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Apply colormap on depth image for better visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally for display
        images = np.hstack((color_image, depth_colormap))

        # Display the images
        cv2.imshow('Color Stream', color_image)
        cv2.imshow('Depth Stream', depth_colormap)

        # Wait for user input to close (press 'q' to quit)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
finally:
    # Stop streaming and release resources
    pipeline.stop()
    cv2.destroyAllWindows()
