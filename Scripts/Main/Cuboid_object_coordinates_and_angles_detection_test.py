import os
import time
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
import trimesh  # Import trimesh to handle CAD files

# Clear terminal before running anything
os.system('cls' if os.name == 'nt' else 'clear')

# Load YOLO model for detecting the cuboid
model = YOLO("runs/detect/train/weights/best.pt")  # Your trained model path
model.to('cuda')  # Use GPU for faster inference

# Function to safely initialize RealSense
def initialize_realsense():
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"Found {len(devices)} connected RealSense devices")
    
    if len(devices) == 0:
        print("No RealSense devices found. Please check connection.")
        return None, None, None, None
    
    # Reset all devices
    for dev in devices:
        print(f"Resetting device: {dev.get_info(rs.camera_info.name)}")
        dev.hardware_reset()
    
    time.sleep(2)
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No devices found after reset. Please check connection.")
        return None, None, None, None
    
    device_serial = devices[0].get_info(rs.camera_info.serial_number)
    print(f"Using device with serial number: {device_serial}")
    
    config.enable_device(device_serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    
    try:
        print("Starting pipeline...")
        profile = pipeline.start(config)
        print("Pipeline started successfully")
        
        color_stream = profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        camera_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ])
        
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

# Load CAD file (e.g., .obj) using trimesh
cad_file_path = "Scripts\\Main\\cuboid.stl"  # Replace with the actual path to your CAD file
mesh = trimesh.load_mesh(cad_file_path)

# Extract 3D vertices (coordinates of the model)
object_points = mesh.vertices  # These are the 3D coordinates of the object model

# If the object model has faces, you can also extract the faces for visualization, if needed.
faces = mesh.faces  # List of faces for the 3D model

# Function to get the corner points from the detected bounding box
def get_corner_points(x1, y1, x2, y2, depth_frame):
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

# Function to project 3D points to 2D
def project_3d_to_2d(object_points, rotation_matrix, translation_vector, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(object_points, rotation_matrix, translation_vector, camera_matrix, dist_coeffs)
    return projected_points.reshape(-1, 2)

try:
    print("Starting detection...")
    
    # Set a retry counter and frame skip
    retry_count = 0
    max_retries = 5
    frame_skip = 5  # Only process every 2nd frame to speed up
    
    frame_counter = 0  # Frame counter to skip frames
    
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
            
            # Skip frames if necessary
            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue  # Skip this frame
            
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
                depth = depth_frame.get_distance(cube_cx, cube_cy)
                
                # Skip invalid depth readings
                if depth <= 0:
                    continue
                
                # Get corner points
                corner_data = get_corner_points(x1, y1, x2, y2, depth_frame)
                if corner_data is None:
                    continue
                    
                image_points, avg_depth = corner_data
                
                # For more accurate pose estimation, use the detected corners as reference
                full_image_points = np.zeros((len(object_points), 2), dtype=np.float32)
                
                # Use the detected bounding box corners for the initial points
                for i in range(min(4, len(object_points))):
                    full_image_points[i] = image_points[i]
                
                # Solve PnP using the 3D object points from the CAD model
                _, rvec, tvec = cv2.solvePnP(object_points, full_image_points, camera_matrix, dist_coeffs)
                
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # Project 3D model points into 2D
                projected_corners = project_3d_to_2d(object_points, rotation_matrix, tvec, camera_matrix, dist_coeffs)
                
                # Draw the axes (X, Y, Z)
                axis_length = 0.05  # Adjust according to your cuboid's size
                axes = np.array([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])  # X, Y, Z axes
                projected_axes = project_3d_to_2d(axes, rotation_matrix, tvec, camera_matrix, dist_coeffs)
                
                # Draw axes
                cv2.line(display_image, tuple(projected_axes[0].astype(int)), tuple(projected_axes[1].astype(int)), (255, 0, 0), 3)  # X-axis
                cv2.line(display_image, tuple(projected_axes[0].astype(int)), tuple(projected_axes[2].astype(int)), (0, 255, 0), 3)  # Y-axis
                cv2.line(display_image, tuple(projected_axes[0].astype(int)), tuple(projected_axes[3].astype(int)), (0, 0, 255), 3)  # Z-axis
                
                # Draw the center point
                center_2d = projected_corners[0]  # This is an example; use the center if needed
                cv2.circle(display_image, (int(center_2d[0]), int(center_2d[1])), 8, (0, 255, 255), -1)  # Yellow center

                # Display translation and quaternion info
                position = tvec.flatten()
                rotation = R.from_rotvec(rvec.flatten())
                quaternion = rotation.as_quat()  # Quaternion [x, y, z, w]
                
                quat_text = f"Quaternion: ({quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f})"
                position_text = f"Position: ({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f})"
                
                cv2.putText(display_image, quat_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display_image, position_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display the result
            cv2.imshow("Pose Estimation", display_image)
            
            # Exit the loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        except Exception as e:
            print(f"Error: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                print("Maximum retries reached. Exiting...")
                break
except KeyboardInterrupt:
    print("Program interrupted.")
finally:
    # Clean up
    pipeline.stop()
    cv2.destroyAllWindows()
