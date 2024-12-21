cmake_minimum_required(VERSION 3.0.2)
project(my_franka_app)

## Find catkin and required components
find_package(catkin REQUIRED COMPONENTS
  franka_msgs
  gazebo_ros
  roscpp
  rospy
  std_msgs
)

## Declare catkin package
catkin_package()

## Specify include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Add executable and link dependencies
add_executable(print_franka_states src/print_franka_states.cpp)

target_link_libraries(print_franka_states
  ${catkin_LIBRARIES}
)