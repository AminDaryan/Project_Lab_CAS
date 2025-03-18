data = np.load('camera_calibration.npz')
cameraMatrix = data['cameraMatrix']
distCoeffs = data['distCoeffs']

img = cv2.imread('test_image.jpg')
h, w = img.shape[:2]
newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))

undistorted_img = cv2.undistort(img, cameraMatrix, distCoeffs, None, newCameraMatrix)
cv2.imshow('Undistorted Image', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
