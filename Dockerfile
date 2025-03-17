# Use an official Python runtime as a parent image
FROM python:3.9.13-slim

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y libopencv-dev
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libusb-1.0-0-dev
RUN apt-get install -y cmake
RUN apt-get install -y git
RUN apt-get install -y curl
RUN apt-get install -y librealsense2 librealsense2-dev


# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory in the container
WORKDIR /app

# Copy the rest of your application code into the container
COPY . /app

# Run the script
CMD ["python", "Cuboid_object_coordinates_and_angles_detection_test.py"]
