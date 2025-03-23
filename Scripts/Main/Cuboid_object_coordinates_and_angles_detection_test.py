import os
import time
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
import trimesh
import torch
import threading
from collections import deque


# Clear terminal before running anything
os.system('cls' if os.name == 'nt' else 'clear')

# Load YOLO model for detecting the cuboid
model = YOLO("runs/detect/train/weights/best.pt")  # Your trained model path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

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
    # We don't really need depth, but keep it for compatibility
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
            color_intrinsics.coeffs[0],  # k1
            color_intrinsics.coeffs[1],  # k2
            color_intrinsics.coeffs[2],  # p1
            color_intrinsics.coeffs[3],  # p2
            color_intrinsics.coeffs[4]   # k3
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

# Align depth to color for compatibility with existing code
align = rs.align(rs.stream.color)

# Load CAD file
cad_file_path = "Scripts\\Main\\cuboid.ply"  # Path to your CAD file
try:
    mesh = trimesh.load_mesh(cad_file_path)
    print("CAD file loaded successfully")
    
    # Print vertices for debugging
    print(f"Number of vertices: {len(mesh.vertices)}")
    for i, vertex in enumerate(mesh.vertices[:8]):  # Print first 8 vertices only
        print(f"Vertex {i}: {vertex}")
    
except Exception as e:
    print(f"Error loading CAD file: {e}")
    # Create a simple cuboid model based on corrected 5cm x 5cm x 3cm dimensions
    # Half dimensions for a centered cuboid (in meters)
    width, height = 0.025, 0.025  # 50cm/2, 50cm/2
    
    # Define 8 corners of the cuboid (centered at origin)
    vertices = np.array([
        [-width, -height],  # 0: bottom left
        [width, -height],   # 1: bottom right
        [width, height],    # 2: top right
        [-width, height],   # 3: top left
    ])
    
    # Define faces using 2 triangles to cover the square
    faces = np.array([
        [0, 1, 2],   # First triangle
        [0, 2, 3]    # Second triangle
    ])

    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    print(f"Created cuboid model with dimensions 5cm x 5cm x 3cm")

# Extract 2D vertices
object_points = np.array(mesh.vertices, dtype=np.float32)

# Print the cuboid model dimensions
min_coords = np.min(object_points, axis=0)
max_coords = np.max(object_points, axis=0)
dimensions = (max_coords - min_coords) * 100  # Convert back to mm for display
print(f"Model dimensions: {dimensions[0]:.2f}cm x {dimensions[1]:.2f}cm x")

# Function to get the corner points from the detected bounding box
def get_corner_points(x1, y1, x2, y2):
    # Define the four corners of the bounding box
    corners_2d = np.array([
        [x1, y1],  # Top-left
        [x2, y1],  # Top-right
        [x2, y2],  # Bottom-right
        [x1, y2]   # Bottom-left
    ], dtype=np.float32)
    
    return corners_2d

def solve_pnp_with_plane_constraint(object_points, image_points, camera_matrix, dist_coeffs):
    # Initial pose estimation
    retval = cv2.solvePnP(
        object_points, 
        image_points,
        camera_matrix, 
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # In newer OpenCV versions, solvePnP returns a tuple with the first element being the return value
    if isinstance(retval, tuple):
        if len(retval) == 3:  # Newer OpenCV versions return (success, rvec, tvec)
            success, rvec, tvec = retval
        elif len(retval) == 2:  # Some versions might return (rvec, tvec) with success implied
            rvec, tvec = retval
            success = True
        else:
            print(f"Unexpected return format from solvePnP: {retval}")
            return False, None, None
    else:
        # For older OpenCV versions that returned just a boolean
        success = retval
        # We don't have rvec and tvec in this case, so we can't proceed
        return False, None, None
    
    if not success:
        return False, None, None
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    # Assume the y-axis should be pointing upward
    up_vector = np.array([0, 1, 0])  # Assuming Y is up, adjust as needed
    y_column = rotation_matrix[:, 1]
    
    # Ensure Y axis is close to vertical
    angle = np.arccos(np.clip(np.dot(y_column, up_vector), -1.0, 1.0))
    
    if angle > np.pi/4:  # If more than 45 degrees off, adjust
        # Create a rotation that would align y_column with up_vector
        correction_axis = np.cross(y_column, up_vector)
        correction_axis = correction_axis / np.linalg.norm(correction_axis)
        correction_angle = angle
        correction_rotvec = correction_axis * correction_angle
        
        # Apply correction to rotation matrix
        correction_matrix, _ = cv2.Rodrigues(correction_rotvec)
        corrected_rotation = np.dot(correction_matrix, rotation_matrix)
        
        # Convert back to rotation vector
        rvec, _ = cv2.Rodrigues(corrected_rotation)
    
    # Refine with the constraint in mind
    refined = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec
    )
    
    # Handle the return value from refinement similarly
    if isinstance(refined, tuple):
        if len(refined) == 3:
            success, rvec, tvec = refined
        elif len(refined) == 2:
            rvec, tvec = refined
            success = True
        else:
            print(f"Unexpected return format from solvePnPRefineLM: {refined}")
            return False, None, None
    else:
        success = refined
    
    
    return success, rvec, tvec

def detect_corners_in_roi(color_image, x1, y1, x2, y2):
    # Extract ROI from the bounding box
    roi = color_image[y1:y2, x1:x2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to find corners
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we get 4 points, we probably have the corners
    if len(approx) == 4:
        # Convert back to original image coordinates
        corners = approx.reshape(-1, 2) + np.array([x1, y1])
        
        # Ensure corners is a numpy array with the right shape
        corners = np.array(corners, dtype=np.float32)
        
        # Print for debugging
        print(f"Found corners: {corners.shape}, data: {corners}")
        
        return corners
    else:
        print(f"Approximation yielded {len(approx)} points instead of 4")
    
    return None

def create_full_cuboid_model(width=0.05, height=0.05, depth=0.03):
    """Create a full 8-point cuboid model centered at origin."""
    w, h, d = width/2, height/2, depth/2
    
    # All 8 corners of the cuboid
    points = np.array([
        [-w, -h, -d],  # 0: back bottom left
        [w, -h, -d],   # 1: back bottom right
        [w, h, -d],    # 2: back top right
        [-w, h, -d],   # 3: back top left
        [-w, -h, d],   # 4: front bottom left
        [w, -h, d],    # 5: front bottom right
        [w, h, d],     # 6: front top right
        [-w, h, d]     # 7: front top left
    ], dtype=np.float32)
    
    return points

def project_to_2d(object_points_input, rotation_v, translation_vector):
    rotation_matrix = cv2.Rodrigues(rotation_v)[0]
    projected_points, _ = cv2.projectPoints(object_points_input, rotation_matrix, translation_vector, camera_matrix, dist_coeffs)
    return projected_points.reshape(-1, 2)

def is_outlier(data, value, threshold=1.5):
    """
    Check if a value is an outlier based on interquartile range (IQR).
    
    Args:
        data: numpy array of values
        value: the value to check
        threshold: multiplier for IQR (default 1.5, commonly used for outlier detection)
        
    Returns:
        True if value is an outlier, False otherwise
    """
    if len(data) < 4:  # Need at least a few points to estimate quartiles
        return False
        
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    
    return value < lower_bound or value > upper_bound

def filter_outliers_in_quaternions(quaternions):
    """
    Filter outliers from a list of quaternions.
    
    This uses the quaternion angle as the metric for outlier detection.
    """
    if len(quaternions) <= 4:  # Not enough data to reliably filter outliers
        return quaternions
    
    # Convert quaternions to a common hemisphere (handle the double cover)
    ref_quat = quaternions[0]
    aligned_quats = []
    
    for q in quaternions:
        # Ensure same hemisphere by checking dot product
        if np.dot(ref_quat, q) < 0:
            aligned_quats.append(-q)
        else:
            aligned_quats.append(q)
    
    # Calculate distance of each quaternion from the median
    aligned_quats = np.array(aligned_quats)
    median_quat = np.median(aligned_quats, axis=0)
    median_quat = median_quat / np.linalg.norm(median_quat)  # Normalize
    
    # Calculate angular distances from median
    angular_distances = []
    filtered_quats = []
    
    for i, q in enumerate(aligned_quats):
        # Angular distance (in radians)
        dot_product = np.abs(np.clip(np.dot(q, median_quat), -1.0, 1.0))
        angle = np.arccos(dot_product) * 2.0  # Double because of quaternion representation
        angular_distances.append(angle)
    
    # Filter based on IQR
    angular_distances = np.array(angular_distances)
    q1 = np.percentile(angular_distances, 25)
    q3 = np.percentile(angular_distances, 75)
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    
    # Keep only quaternions within bounds
    filtered_indices = np.where(angular_distances <= upper_bound)[0]
    filtered_quats = [quaternions[i] for i in filtered_indices]
    
    print(f"Filtered {len(quaternions) - len(filtered_quats)} outlier quaternions out of {len(quaternions)}")
    return filtered_quats

def filter_outliers_in_positions(positions):
    """
    Filter outliers from a list of position vectors.
    """
    if len(positions) <= 4:  # Not enough data to reliably filter outliers
        return positions
    
    positions = np.array(positions)
    median_pos = np.median(positions, axis=0)
    
    # Calculate distances from median
    distances = np.linalg.norm(positions - median_pos, axis=1)
    
    # Use IQR to filter outliers
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    
    # Keep only positions within bounds
    filtered_indices = np.where(distances <= upper_bound)[0]
    filtered_positions = positions[filtered_indices]
    
    print(f"Filtered {len(positions) - len(filtered_positions)} outlier positions out of {len(positions)}")
    return filtered_positions

def average_quaternions(quaternions):
    if not quaternions:
        return None
    
    # Filter outliers first
    filtered_quats = filter_outliers_in_quaternions(quaternions)
    
    if not filtered_quats:  # If all were outliers (shouldn't happen)
        return quaternions[0]
    
    # Convert to numpy array for easier manipulation
    quats = np.array(filtered_quats)
    
    # Handle quaternion double cover by ensuring consistent sign
    # Make the first element of the first quaternion positive
    reference = quats[0]
    if reference[0] < 0:
        reference = -reference
    
    # Align all quaternions with the reference
    for i in range(1, len(quats)):
        # Compute dot product
        dot_product = np.sum(quats[i] * reference)
        # If negative, flip sign to ensure shortest path
        if dot_product < 0:
            quats[i] = -quats[i]
    
    # Simple arithmetic mean
    avg_quat = np.mean(quats, axis=0)
    
    # Normalize the result
    norm = np.linalg.norm(avg_quat)
    if norm > 0:
        avg_quat = avg_quat / norm
    
    return avg_quat

def average_positions(positions):
    if not positions:
        return None
    
    # Filter outliers first
    filtered_positions = filter_outliers_in_positions(positions)
    
    if len(filtered_positions) == 0:  # If all were outliers (shouldn't happen)
        return positions[0]
    
    # Calculate mean of filtered positions
    avg_position = np.mean(filtered_positions, axis=0)
    
    return avg_position

# Function to write the average quaternion to a text file
def write_quaternion_to_file(quaternion, position):
    # Specify an absolute path
    file_path = "/tmp/quaternion_average.txt" 

    with open(file_path, "w") as f:
        if quaternion is not None:
            # Format quaternion with 6 decimal places
            quat_str = f"Average Quaternion: ({quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f})"
            f.write(quat_str + "\n")
            
            if position is not None:
                # Format position with 6 decimal places
                pos_str = f"Average Position: ({position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f})"
                f.write(pos_str + "\n")
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            f.write(f"Last Updated: {timestamp}\n")
            
            # Add number of samples used in the average
            f.write(f"Samples in Average: {len(quaternion_buffer)}\n")
        else:
            f.write("No quaternion data available yet.\n")


try:
    print("Starting detection...")
    
    # Set a retry counter and frame skip
    retry_count = 0
    max_retries = 2
    frame_skip = 5  # Only process every Nth frame to speed up
    frame_counter = 0  # Frame counter to skip frames

    # For tracking quaternion averages
    quaternion_buffer = deque(maxlen=100)  # Store last 30 quaternions for averaging
    quaternion_lock = threading.Lock()    # Thread safety when updating the buffer
    last_file_update = 0                  # Track when we last updated the file
    update_interval = 0.5                 # Update file every 0.5 seconds

    position_buffer = deque(maxlen=100)    # Store last 30 positions
    
    while True:
        try:
            # Try to get frames with error handling
            frames = pipeline.wait_for_frames(5000)  # 5 second timeout
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            
            # Reset retry counter on success
            retry_count = 0
            
            if not color_frame:
                print("Invalid frames received. Skipping...")
                continue
                
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # Skip frames if necessary
            frame_counter += 1
            if frame_counter % frame_skip != 0:
                continue  # Skip this frame
            
            # Clone the color image for display
            display_image = color_image.copy()
            
            # Get color intrinsics for calculations
            color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            
            # Run YOLO detection
            results = model(color_image, conf=0.25)
            
            # Process only the highest confidence detection
            if len(results[0].boxes) > 0:
                print(f"Found {len(results[0].boxes)} detections, using highest confidence one")
                
                # Sort by confidence and take the highest one
                best_detection = None
                highest_conf = 0
                
                for detection in results[0].boxes:
                    confidence = detection.conf[0].item()
                    if confidence > highest_conf:
                        highest_conf = confidence
                        best_detection = detection
                
                # Process the best detection
                if best_detection is not None:
                    x1, y1, x2, y2 = map(int, best_detection.xyxy[0].tolist())
                    confidence = best_detection.conf[0].item()
                    print(f"Best detection confidence: {confidence:.2f}")
                    
                    # Draw the original bounding box
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Get corner points (no depth needed)
                    image_points = get_corner_points(x1, y1, x2, y2)
                    
                    # Calculate the aspect ratio of the bounding box
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    aspect_ratio = bbox_width / bbox_height if bbox_height != 0 else 0
                    print(f"Bounding box aspect ratio: {aspect_ratio:.2f}")
                    
                    # First, try to detect better corners using contour detection
                    better_corners = detect_corners_in_roi(color_image, x1, y1, x2, y2)

                    # Use the full cuboid model
                    full_cuboid = create_full_cuboid_model(0.05, 0.05, 0.03)  # 5cm x 5cm x 3cm

                    if better_corners is not None and len(better_corners) == 4:
                        print("Using contour-detected corners for pose estimation")
                        image_points = better_corners
                        
                        # Add this validation
                        if len(image_points) != 4:
                            print(f"Invalid number of corner points: {len(image_points)}, expected 4")
                            continue
                            
                        # Define model points for the detected face (assuming front face)
                        model_points = np.array([
                            [-0.025, -0.025, 0.015],  # Front bottom left
                            [0.025, -0.025, 0.015],   # Front bottom right
                            [0.025, 0.025, 0.015],    # Front top right
                            [-0.025, 0.025, 0.015]    # Front top left
                        ], dtype=np.float32)
                        
                        # Use plane-constrained PnP with fixed Z
                        success, rvec, tvec = solve_pnp_with_plane_constraint(
                            model_points,
                            image_points,
                            camera_matrix,
                            dist_coeffs
                        )
                        
                        if not success:
                            print("Plane-constrained PnP failed, falling back to standard method")
                            # Fall back to standard method
                            success, rvec, tvec = cv2.solvePnP(
                                model_points,
                                image_points,
                                camera_matrix,
                                dist_coeffs,
                                flags=cv2.SOLVEPNP_ITERATIVE
                            )
                            
                    else:
                        print("Using bounding box corners with multiple PnP methods")
                        # Use bounding box corners
                        image_points = get_corner_points(x1, y1, x2, y2)
                        
                        # Define model points for PnP - use the front face of the cuboid
                        model_points = np.array([
                            [-0.025, -0.025, 0.015],  # Front bottom left
                            [0.025, -0.025, 0.015],   # Front bottom right
                            [0.025, 0.025, 0.015],    # Front top right
                            [-0.025, 0.025, 0.015]    # Front top left
                        ], dtype=np.float32)
                        
                        # Try different PnP methods if one fails
                        pnp_methods = [cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP]
                        success = False
                        
                        for method in pnp_methods:
                            try:
                                # Initial PnP estimation
                                success, rvec, tvec = cv2.solvePnP(
                                    model_points,
                                    image_points,
                                    camera_matrix,
                                    dist_coeffs,
                                    flags=method
                                )

                                if not success:
                                    print(f"PnP method {method} failed in initial estimation.")
                                    continue

                                print(f"Initial solvePnP successful with method {method}")
                                

                                # Try plane-constrained refinement
                                success, rvec, tvec = solve_pnp_with_plane_constraint(
                                    model_points,
                                    image_points,
                                    camera_matrix,
                                    dist_coeffs
                                )

                                print("Refined Rotation Vector (rvec):", rvec.flatten())
                                print("Refined Translation Vector (tvec):", tvec.flatten())
                                
                                if success:
                                    print(f"PnP solved with method {method} and plane constraint")
                                    break
                            except Exception as e:
                                print(f"PnP method {method} failed: {e}")
                                continue

                    if not success:
                        print("All PnP methods failed")
                        continue
                    
                    print(f"PnP solver result - success: {success}, rotation: {rvec.flatten()}, translation: {tvec.flatten()}")
                    
                    # Project all vertices of the 2D model into 2D for visualization
                    projected_points = project_to_2d(model_points, rvec, tvec)

                    # First draw the projected model vertices (magenta points)
                    for i, point in enumerate(projected_points):
                        point_int = tuple(np.round(point).astype(int))
                        # Check if point is within image boundaries
                        if 0 <= point_int[0] < display_image.shape[1] and 0 <= point_int[1] < display_image.shape[0]:
                            cv2.circle(display_image, point_int, 3, (255, 0, 255), -1)  # Magenta points
                            # Optional: label points
                            cv2.putText(display_image, str(i), point_int, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                   # Visualize the full cuboid if the pose was estimated successfully
                    if success:
                        # Get all 8 corners of the cuboid for visualization
                        all_cuboid_points = create_full_cuboid_model(0.05, 0.05, 0.03)
                        projected_cuboid = project_to_2d(all_cuboid_points, rvec, tvec)
                        
                        # Define the edges connecting the corners of the cuboid
                        cuboid_edges = [
                            # Bottom face
                            (0, 1), (1, 2), (2, 3), (3, 0),
                            # Top face
                            (4, 5), (5, 6), (6, 7), (7, 4),
                            # Connecting edges
                            (0, 4), (1, 5), (2, 6), (3, 7)
                        ]
                        
                        # Draw the full cuboid wireframe
                        for edge in cuboid_edges:
                            try:
                                p1 = tuple(np.round(projected_cuboid[edge[0]]).astype(int))
                                p2 = tuple(np.round(projected_cuboid[edge[1]]).astype(int))
                                # Check if points are within image boundaries
                                if (0 <= p1[0] < display_image.shape[1] and 0 <= p1[1] < display_image.shape[0] and
                                    0 <= p2[0] < display_image.shape[1] and 0 <= p2[1] < display_image.shape[0]):
                                    cv2.line(display_image, p1, p2, (0, 255, 255), 2)  # Yellow lines
                            except Exception as e:
                                print(f"Error drawing cuboid edge {edge}: {e}")
                    
                # THEN draw the axes (after the yellow cuboid)
                axis_length = 0.1  # 10cm axis length for better visibility with a 5cm object
                axes = np.array([
                    [0, 0, 0],  # Origin
                    [axis_length, 0, 0],  # X-axis
                    [0, axis_length, 0],  # Y-axis
                    [0, 0, axis_length]   # Z-axis
                ], dtype=np.float32)

                projected_axes = project_to_2d(axes, rvec, tvec)

                # Draw axes with thicker lines for better visibility
                origin = tuple(projected_axes[0].astype(int))
                # Check if origin is within image boundaries
                if 0 <= origin[0] < display_image.shape[1] and 0 <= origin[1] < display_image.shape[0]:
                    # X-axis (red)
                    p_x = tuple(projected_axes[1].astype(int))
                    if 0 <= p_x[0] < display_image.shape[1] and 0 <= p_x[1] < display_image.shape[0]:
                        cv2.line(display_image, origin, p_x, (0, 0, 255), 2)
                        cv2.putText(display_image, "X", p_x, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Y-axis (green)
                    p_y = tuple(projected_axes[2].astype(int))
                    if 0 <= p_y[0] < display_image.shape[1] and 0 <= p_y[1] < display_image.shape[0]:
                        cv2.line(display_image, origin, p_y, (0, 255, 0), 2)
                        cv2.putText(display_image, "Y", p_y, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Z-axis (blue)
                    p_z = tuple(projected_axes[3].astype(int))
                    if 0 <= p_z[0] < display_image.shape[1] and 0 <= p_z[1] < display_image.shape[0]:
                        cv2.line(display_image, origin, p_z, (255, 0, 0), 2)
                        cv2.putText(display_image, "Z", p_z, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Display translation and quaternion info
                position = tvec.flatten()
                rotation = R.from_rotvec(rvec.flatten())
                quaternion = rotation.as_quat()  # Quaternion [x, y, z, w]
                
                with quaternion_lock:
                    quaternion_buffer.append(quaternion)
                    position_buffer.append(position)
                
                # Update the text file periodically
                current_time = time.time()
                if current_time - last_file_update > update_interval:
                    with quaternion_lock:
                        avg_quat = average_quaternions(list(quaternion_buffer))
                        avg_pos = average_positions(list(position_buffer)) if position_buffer else None
                        write_quaternion_to_file(avg_quat, avg_pos)
                    last_file_update = current_time
                
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
                print(f"Maximum retries reached ({max_retries}). Exiting...")
                break
except KeyboardInterrupt:
    print("Program interrupted.")
finally:
    # Clean up resources
    print("Cleaning up resources...")
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Program ended.")