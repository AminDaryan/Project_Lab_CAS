import os
import time
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

# Clear terminal before running anything
os.system('cls' if os.name == 'nt' else 'clear')

# Load YOLO model for detecting the cuboid
model = YOLO("runs/detect/train/weights/best.pt")  # Your trained model path
model.to('cuda')  # Use GPU for faster inference

# Function to safely initialize RealSense
def initialize_realsense():
    # First, ensure no contexts are left open
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"Found {len(devices)} connected RealSense devices")
    
    if len(devices) == 0:
        print("No RealSense devices found. Please check connection.")
        return None, None
    
    # Reset all devices
    for dev in devices:
        print(f"Resetting device: {dev.get_info(rs.camera_info.name)}")
        dev.hardware_reset()
    
    # Wait for reset to complete
    time.sleep(2)
    
    # Create new pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Try to find a device
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No devices found after reset. Please check connection.")
        return None, None
    
    # Get device serial number
    device_serial = devices[0].get_info(rs.camera_info.serial_number)
    print(f"Using device with serial number: {device_serial}")
    
    # Enable streams with a lower resolution and frame rate for reliability
    config.enable_device(device_serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    
    # Set timeout to a larger value
    try:
        print("Starting pipeline...")
        profile = pipeline.start(config)
        print("Pipeline started successfully")
        
        # Get camera intrinsic parameters
        color_stream = profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Create camera matrix from intrinsics
        camera_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # Get distortion coefficients
        dist_coeffs = np.array([
            color_intrinsics.coeffs[0],
            color_intrinsics.coeffs[1],
            color_intrinsics.coeffs[2],
            color_intrinsics.coeffs[3],
            0
        ]).reshape(5, 1)
        
        return pipeline, profile, camera_matrix, dist_coeffs
    
    except Exception as e:
        print(f"Error starting pipeline: {e}")
        return None, None, None, None

# Initialize RealSense with error handling
pipeline, profile, camera_matrix, dist_coeffs = initialize_realsense()

if pipeline is None:
    print("Failed to initialize RealSense camera. Please check connection and try again.")
    exit(1)

# Align depth to color
align = rs.align(rs.stream.color)

# Define real-world 3D cuboid coordinates
cuboid_size = [0.005, 0.005, 0.005]  # meters

# Define 3D model points for the cuboid
object_points = np.array([
    # Bottom face
    [-cuboid_size[0]/2, -cuboid_size[1]/2, -cuboid_size[2]/2],  # Bottom-left-back
    [cuboid_size[0]/2, -cuboid_size[1]/2, -cuboid_size[2]/2],   # Bottom-right-back
    [cuboid_size[0]/2, cuboid_size[1]/2, -cuboid_size[2]/2],    # Top-right-back
    [-cuboid_size[0]/2, cuboid_size[1]/2, -cuboid_size[2]/2],   # Top-left-back
    
    # Top face
    [-cuboid_size[0]/2, -cuboid_size[1]/2, cuboid_size[2]/2],   # Bottom-left-front
    [cuboid_size[0]/2, -cuboid_size[1]/2, cuboid_size[2]/2],    # Bottom-right-front
    [cuboid_size[0]/2, cuboid_size[1]/2, cuboid_size[2]/2],     # Top-right-front
    [-cuboid_size[0]/2, cuboid_size[1]/2, cuboid_size[2]/2]     # Top-left-front
], dtype=np.float32)

# Define edges of the cuboid for visualization
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
    (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
]

# Function to get the corner points from the detected bounding box
def get_corner_points(x1, y1, x2, y2, depth_frame, depth_intrinsics):
    # Get the depth at the corners
    depth_corners = np.array([
        depth_frame.get_distance(x1, y1),  # Top-left
        depth_frame.get_distance(x2, y1),  # Top-right
        depth_frame.get_distance(x2, y2),  # Bottom-right
        depth_frame.get_distance(x1, y2)   # Bottom-left
    ])
    
    # Filter out invalid depth readings
    valid_depth = depth_corners[depth_corners > 0]
    if len(valid_depth) == 0:
        return None
    
    # Use the average depth where valid
    avg_depth = np.mean(valid_depth)
    
    # Create 2D points from bounding box
    image_points = np.array([
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2]   # Bottom-left
    ], dtype=np.float32)
    
    return image_points, avg_depth

try:
    print("Starting detection...")
    
    # Set a retry counter
    retry_count = 0
    max_retries = 5
    
    while True:
        try:
            # Try to get frames with error handling
            frames = pipeline.wait_for_frames(5000)  # 5 second timeout
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            # Reset retry counter on success
            retry_count = 0
            
            if not color_frame or not depth_frame:
                print("Invalid frames received. Skipping...")
                continue
                
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Clone the color image for display
            display_image = color_image.copy()
            
            # Run YOLO detection
            results = model(color_image, conf=0.25)
            
            # Process detections
            for detection in results[0].boxes:
                x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
                
                # Draw the original bounding box
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate bounding box center
                cube_cx, cube_cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Get depth at the center point
                depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
                depth = depth_frame.get_distance(cube_cx, cube_cy)
                
                # Skip invalid depth readings
                if depth <= 0:
                    continue
                
                # Get corner points
                corner_data = get_corner_points(x1, y1, x2, y2, depth_frame, depth_intrinsics)
                if corner_data is None:
                    continue
                    
                image_points, avg_depth = corner_data
                
                # For more accurate pose estimation, use the detected corners as reference
                # but we need 8 points for our 3D model
                full_image_points = np.zeros((8, 2), dtype=np.float32)
                
                # Bottom face points (using the detected bounding box)
                full_image_points[0:4] = image_points
                
                # Calculate approximate top face points
                # We'll estimate the height of the cube in the image
                height_ratio = cuboid_size[2] / avg_depth * depth_intrinsics.fy
                
                # Calculate top face points (shifted up by estimated height)
                for i in range(4):
                    full_image_points[i+4] = [image_points[i][0], image_points[i][1] - height_ratio]
                
                # Try to estimate the pose using PnP with RANSAC for robustness
                try:
                    success, rvec, tvec = cv2.solvePnP(
                        object_points, 
                        full_image_points, 
                        camera_matrix, 
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                    
                    if success:
                        # Convert rotation vector to rotation matrix
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        
                        # Convert rotation matrix to quaternion
                        quaternion = R.from_matrix(rotation_matrix).as_quat()
                        qx, qy, qz, qw = quaternion
                        
                        # Project 3D points to image plane for visualization
                        projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
                        projected_points = projected_points.reshape(-1, 2).astype(np.int32)
                        
                        # Draw edges of the cuboid with different colors
                        colors = [
                            (255, 0, 0),  # Blue for bottom face
                            (0, 255, 0),  # Green for top face
                            (255, 255, 0) # Yellow for connecting edges
                        ]
                        
                        # Draw wireframe
                        for i, edge in enumerate(edges):
                            color_idx = 0 if i < 4 else (1 if i < 8 else 2)
                            cv2.line(display_image, 
                                    tuple(projected_points[edge[0]]), 
                                    tuple(projected_points[edge[1]]), 
                                    colors[color_idx], 2)
                        
                        # Calculate the center of the top face
                        top_face_center = np.mean(projected_points[4:8], axis=0).astype(np.int32)
                        
                        # Draw normal vector from top face center
                        normal_vector = rotation_matrix @ np.array([0, 0, 1])  # Top face normal
                        normal_length = 40  # Fixed length for visualization
                        end_point = (
                            int(top_face_center[0] + normal_vector[0] * normal_length),
                            int(top_face_center[1] - normal_vector[2] * normal_length)  # Negate Z for screen coordinates
                        )
                        cv2.arrowedLine(display_image, tuple(top_face_center), end_point, (0, 255, 255), 2)
                        
                        # Draw red dot at cuboid center
                        cv2.circle(display_image, (cube_cx, cube_cy), 5, (0, 0, 255), -1)
                        
                        # Get 3D position in camera coordinates
                        cube_center_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [cube_cx, cube_cy], depth)
                        
                        # Display quaternion and position
                        position_text = f"Position: ({cube_center_3d[0]:.4f}, {cube_center_3d[1]:.4f}, {cube_center_3d[2]:.4f})"
                        quat_text = f"Quaternion: ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})"
                        
                        # Display texts with background for better visibility
                        cv2.putText(display_image, quat_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(display_image, position_text, (10, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Print info
                        print(f"Quaternion (qx, qy, qz, qw): ({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})")
                        print(f"Position (x, y, z): ({cube_center_3d[0]:.4f}, {cube_center_3d[1]:.4f}, {cube_center_3d[2]:.4f})")
                
                except Exception as e:
                    print(f"Error in pose estimation: {e}")
                    continue
            
            # Display the image
            cv2.imshow("Cuboid Detection", display_image)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        except RuntimeError as e:
            print(f"Runtime error: {e}")
            retry_count += 1
            
            if retry_count >= max_retries:
                print("Too many consecutive errors. Restarting pipeline...")
                # Stop the pipeline
                pipeline.stop()
                time.sleep(1)
                
                # Reinitialize
                pipeline, profile, camera_matrix, dist_coeffs = initialize_realsense()
                if pipeline is None:
                    print("Failed to reinitialize RealSense camera. Exiting...")
                    break
                
                # Reset retry counter
                retry_count = 0
            else:
                print(f"Retrying... ({retry_count}/{max_retries})")
                time.sleep(1)  # Wait before retry

finally:
    # Clean up
    if pipeline:
        pipeline.stop()
    cv2.destroyAllWindows()
    print