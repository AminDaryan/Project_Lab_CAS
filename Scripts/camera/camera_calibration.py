import cv2
import numpy as np
import glob

# Chessboard dimensions (internal corners)
CHECKERBOARD = (7, 6)  # 8x6 grid means (7,6) internal corners
square_size = 25  # Size of a square (in mm or any unit)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []
imgpoints = []

# Load calibration images
images = glob.glob('calibration_images/*.jpg')  # Ensure you have a folder with calibration images

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        cv2.imshow('Chessboard Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Calibration
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the calibration result
np.savez('camera_calibration.npz', cameraMatrix=cameraMatrix, distCoeffs=distCoeffs)

print("Camera matrix:\n", cameraMatrix)
print("Distortion coefficients:\n", distCoeffs)
