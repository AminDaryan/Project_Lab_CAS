## History

*15.12.2024*
- The D455 Camera is now accessed using the realsense library.
- The OpenCV is used to interpret the image/video recieved by the camera.
- Two files of code examples are added to the repo.

*02.01.2025*
- Added Object detection sample code which detects general objects
- Some files were added so that the devs can install and setup the project more easily even on the windows
- Added Blue Cuboid detection code with YoloV5 or V11 library.

*15.01.2025*
- Using [video_to_dataset_pics file](video_to_dataset_pics.py) to get the cuboid video to images dataset.
- The the data samples (Images) were annotated and then trained in [roboflow.com](roboflow.com).
  - The dataset is in [dataset](dataset) folder
- Using [this video](https://www.youtube.com/watch?v=m9fH9OWn8YM) to train the model in [google python colab](https://colab.research.google.com/drive/1u9PslhvYOG_QUzIM-CZxvZ3ivJpVOuyi#scrollTo=9b8qaAPqL-jr).
  - The relative weights of the trained model and all the other specifications are in the [run](run) folder.
- Added code to use the cuboid detection model in [Inference_Blue_cuboid_model file](Inference_Blue_cuboid_model.py).

*16.01.2025*
- Added object coordinate calculation code for the blue cuboid in [Blue_cuboid_object_coordinate_detection file](Blue_cuboid_object_coordinate_detection.py)

*13.03.2025*
- Using [video_to_dataset_pics file](video_to_dataset_pics.py) to get the cube with April tags video to images dataset.
- The the data samples (Images) were annotated and then trained in [roboflow.com](roboflow.com).

*15.03.2025*
- Using [train_model file](train_model.py) the images recieved in files: train, valid and test from roboflow.com are trained.

*21.03.2025*
- In [Cuboid_object_coordinates_and_angles_detection_test file](Cuboid_object_coordinates_and_angles_detection_test.py) using [Claud.io](https://claude.ai/chat/6d9affa4-3a98-4140-9611-01f581c6f9ca) and some papers, plus understanding the code and tweaking it, I was able to make the code detect the cubes rotation.
- The camera intrinsics can be fetched from the camera itself.

<br />

## About the Project

- [make_cuboid.py](Scripts\make_cuboid.py): In this file, a 3D model of the target cuboid is generated and saved as "cuboid.stl".
- [train_model.py](Scripts\train_model.py):  In this file, images from "datasets/Target_Cube_Object_Detection_3" are trained using the YOLO library.
- [video_to_dataset_pics.py](Scripts\video_to_dataset_pics.py):  In this file, the camera takes a 10 second video and thereafter generates images which they are the video splitted into frames. These frames (images) will be used to train the model. The video is taken by the user from the cuboid object from different angles ideally.
- [Cuboid_object_coordinates_and_angles_detection.py](Scripts\Main\Cuboid_object_coordinates_and_angles_detection.py):  In this file, the code uses the generated model to detect the object real-time. The code detects the object angle and position and then converts it into quaternian. This can also be seen when the code is run; On the real-time video window opened after the code is run, a square will surround the target cuboid object when the object is detected.

### How does the object detection code work
1. Trained model is fetched from "runs/detect/train/weights/best.pt".
2. Safely initialize RealSense
3. Load CAD file made earlier "cuboid.stl"
4. Extract 2D vertices from the model (Line 130)
5. Define functions:
    1. get_corner_points: get the corner points from the detected bounding box model
    2. solve_pnp_with_plane_constraint: Using OpenCV (CV2), it estimates the Initial pose of the object.
    Return rotation vector (rvec) and translation vector (tvec) (Line 151)
    3. detect_corners_in_roi: To detect the four corners of the largest contour within a specified Region of Interest (ROI) of a color image.
    4. create_full_cuboid_model
    5. project_to_2d
    6. filter_outliers_in_quaternions: Filter outliers from a list of quaternions based on interquartile range (IQR).
    7. filter_outliers_in_positions:  Filter outliers from a list of positions based on interquartile range (IQR).
    8. average_quaternions: Get the average quaternians by first filtering the outliers using the aformentioned filter_outliers_in_quaternions function.
    9. average_positions: Get the average positions by first filtering the outliers using the aformentioned filter_outliers_in_positions function.
    10. write_quaternion_to_file: Write the average quaternion to a text file in order for it to be read by the ROS franka robot manipulator code. (Line 422)


## Requirements
- Python version: 3.9.13