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

# Load CAD file
cad_file_path = "Scripts\\Main\\cuboid.stl"  # Path to your CAD file
try:
    mesh = trimesh.load_mesh(cad_file_path)
    print("CAD file loaded successfully")
    
    # Print vertices for debugging
    print(f"Number of vertices: {len(mesh.vertices)}")
    for i, vertex in enumerate(mesh.vertices):
        print(f"Vertex {i}: {vertex}")
    
except Exception as e:
    print(f"Error loading CAD file: {e}")
    # Create a simple cuboid model based on your 5mm x 5mm x 3mm dimensions
    # Half dimensions for a centered cuboid
    width, height, depth = 2.5, 2.5, 1.5  # in mm
    
    # Define 8 corners of the cuboid (centered at origin)
    mesh = trimesh.Trimesh(vertices=np.array([
        [-width, -height, -depth],  # 0: back bottom left
        [width, -height, -depth],   # 1: back bottom right
        [width, height, -depth],    # 2: back top right
        [-width, height, -depth],   # 3: back top left
        [-width, -height, depth],   # 4: front bottom left
        [width, -height, depth],    # 5: front bottom right
        [width, height, depth],     # 6: front top right
        [-width, height, depth]     # 7: front top left
    ]))
    print("Created cuboid model with dimensions 5mm x 5mm x 3mm")

# Extract 3D vertices
object_points = np.array(mesh.vertices, dtype=np.float32)

# Print the cuboid model dimensions
min_coords = np.min(object_points, axis=0)
max_coords = np.max(object_points, axis=0)
dimensions = max_coords - min_coords
print(f"Model dimensions: {dimensions[0]:.2f}mm x {dimensions[1]:.2f}mm x {dimensions[2]:.2f}mm")

# Define the front face vertices (for better visualization)
# This assumes the cuboid is centered at the origin and aligned with the coordinate axes
front_face_idx = [4, 5, 6, 7]  # Front face vertices (the ones where z is positive)
front_face_points = object_points[front_face_idx]

# Function to get the corner points from the detected bounding box
def get_corner_points(x1, y1, x2, y2, depth_frame, color_intrinsics):
    # Define the four corners of the bounding box
    corners_2d = np.array([
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2]   # Bottom-left
    ], dtype=np.float32)
    
    # Get the center of the bounding box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Get depth at the center point, which is often more reliable
    center_depth = depth_frame.get_distance(center_x, center_y)
    
    # If center depth is invalid, try to get depth from corners
    if center_depth <= 0:
        # Try each corner
        depths = []
        for corner in corners_2d:
            cx, cy = int(corner[0]), int(corner[1])
            d = depth_frame.get_distance(cx, cy)
            if d > 0:
                depths.append(d)
        
        # If we found any valid depths, use their average
        if depths:
            center_depth = np.mean(depths)
        else:
            # If no valid depths, scan the bounding box area
            valid_depths = []
            for x in range(x1, x2, 5):  # Sample every 5 pixels
                for y in range(y1, y2, 5):
                    d = depth_frame.get_distance(x, y)
                    if d > 0:
                        valid_depths.append(d)
            
            if valid_depths:
                center_depth = np.mean(valid_depths)
            else:
                return None  # No valid depth found
    
    print(f"Estimated object depth: {center_depth:.3f}m")
    
    return corners_2d, center_depth

# Function to project 3D points to 2D
def project_3d_to_2d(object_points, rotation_matrix, translation_vector, camera_matrix, dist_coeffs):
    projected_points, _ = cv2.projectPoints(object_points, rotation_matrix, translation_vector, camera_matrix, dist_coeffs)
    return projected_points.reshape(-1, 2)

try:
    print("Starting detection...")
    
    # Set a retry counter and frame skip
    retry_count = 0
    max_retries = 2
    frame_skip = 5  # Only process every 5th frame to speed up
    
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
            
            # Get color intrinsics for depth calculations
            color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            
            # Run YOLO detection
            results = model(color_image, conf=0.25)
            
            # Process detections
            if len(results[0].boxes) > 0:
                print(f"Found {len(results[0].boxes)} detections")
                
            for detection in results[0].boxes:
                x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
                confidence = detection.conf[0].item()
                print(f"Detection confidence: {confidence:.2f}")
                
                # Draw the original bounding box
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Get corner points
                corner_data = get_corner_points(x1, y1, x2, y2, depth_frame, color_intrinsics)
                if corner_data is None:
                    print("Could not get valid depth measurements for this detection")
                    continue
                    
                image_points, depth = corner_data
                
                # Calculate the aspect ratio of the bounding box
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                aspect_ratio = bbox_width / bbox_height if bbox_height != 0 else 0
                print(f"Bounding box aspect ratio: {aspect_ratio:.2f}")
                
                # Select the appropriate 3D points based on the view
                # For simplicity, we're using the front face points, which is appropriate if
                # the cuboid is being viewed from the front
                model_points = np.array([
                    [-2.5, -2.5, 1.5],  # Bottom left, front
                    [2.5, -2.5, 1.5],   # Bottom right, front
                    [2.5, 2.5, 1.5],    # Top right, front
                    [-2.5, 2.5, 1.5]    # Top left, front
                ], dtype=np.float32)
                
                # Solve PnP using the front face points (should be visible in the image)
                success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                
                if not success:
                    print("PnP solver failed")
                    continue
                
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # Project all vertices of the 3D model into 2D for visualization
                projected_points = project_3d_to_2d(object_points, rvec, tvec, camera_matrix, dist_coeffs)
                
                # Draw the projected model points
                for point in projected_points:
                    cv2.circle(display_image, tuple(point.astype(int)), 3, (255, 0, 255), -1)  # Magenta points
                
                # Draw the cuboid edges
                edges = [
                    # Bottom face
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    # Top face
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    # Connecting edges
                    (0, 4), (1, 5), (2, 6), (3, 7)
                ]
                
                for edge in edges:
                    p1 = tuple(projected_points[edge[0]].astype(int))
                    p2 = tuple(projected_points[edge[1]].astype(int))
                    cv2.line(display_image, p1, p2, (255, 165, 0), 1)  # Orange lines
                
                # Draw the axes (X, Y, Z)
                axis_length = 5.0  # 5mm axis length
                axes = np.array([
                    [0, 0, 0],
                    [axis_length, 0, 0],
                    [0, axis_length, 0],
                    [0, 0, axis_length]
                ], dtype=np.float32)  # Origin, X, Y, Z axes
                
                projected_axes = project_3d_to_2d(axes, rvec, tvec, camera_matrix, dist_coeffs)
                
                # Draw axes with thicker lines for better visibility
                origin = tuple(projected_axes[0].astype(int))
                cv2.line(display_image, origin, tuple(projected_axes[1].astype(int)), (0, 0, 255), 2)  # X-axis (red)
                cv2.line(display_image, origin, tuple(projected_axes[2].astype(int)), (0, 255, 0), 2)  # Y-axis (green)
                cv2.line(display_image, origin, tuple(projected_axes[3].astype(int)), (255, 0, 0), 2)  # Z-axis (blue)
                
                # Display translation and quaternion info
                position = tvec.flatten()
                rotation = R.from_rotvec(rvec.flatten())
                quaternion = rotation.as_quat()  # Quaternion [x, y, z, w]
                
                # Format with better precision and more space
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