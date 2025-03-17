# Use an official Python runtime as a parent image
FROM python:3.9.13-slim

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    libusb-1.0-0-dev \
    cmake \
    git \
    libopencv-dev \
    libboost-all-dev

# Add Intel RealSense SDK repository and install librealsense2
RUN curl -sSL https://github.com/IntelRealSense/librealsense/releases/download/v2.50.0/Librealsense-2.50.0-Ubuntu-20.04.deb \
    -o librealsense2.deb \
    && dpkg -i librealsense2.deb \
    && apt-get install -f -y \
    && rm librealsense2.deb

# Install Python dependencies from the requirements file
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory in the container
WORKDIR /app

# Copy the rest of your application code into the container
COPY . /app

# Run the script
CMD ["python", "Cuboid_object_coordinates_and_angles_detection_test.py"]
