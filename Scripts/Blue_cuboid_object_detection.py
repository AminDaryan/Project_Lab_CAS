import pyrealsense2 as rs
import cv2
import numpy as np

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Define HSV range for blue color
LOWER_BLUE = np.array([100, 50, 50])    # Lower range for blue
UPPER_BLUE = np.array([140, 255, 255])  # Upper range for blue

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert BGR image to HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Mask blue color
        blue_mask = cv2.inRange(hsv_image, LOWER_BLUE, UPPER_BLUE)

        # Find contours in the mask
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter by area (ignore very small or very large objects)
            area = cv2.contourArea(contour)
            if area < 500 or area > 10000:  # Adjust thresholds based on cuboid size
                continue

            # Get bounding box around the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Allow rectangular shapes (adjust aspect ratio range)
            aspect_ratio = float(w) / h
            if 0.5 <= aspect_ratio <= 2.0:  # Accept wider range for cuboids
                # Get depth value at the center of the bounding box
                center_x, center_y = x + w // 2, y + h // 2
                distance = depth_frame.get_distance(center_x, center_y)

                # Draw bounding box and label
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                label = f"Blue Cuboid: {distance:.2f}m"
                cv2.putText(color_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the result
        cv2.imshow("Blue Cuboid Detection", color_image)

        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
