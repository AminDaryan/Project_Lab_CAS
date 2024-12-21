#include <ros/ros.h>
#include <franka_msgs/FrankaState.h>

// Callback function for receiving Franka joint states
void frankaStateCallback(const franka_msgs::FrankaState::ConstPtr& msg) {
    ROS_INFO("Franka joint state: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
             msg->q[0], msg->q[1], msg->q[2], msg->q[3],
             msg->q[4], msg->q[5], msg->q[6]);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "print_franka_states");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe(
        "/franka_state_controller/franka_states", 10, frankaStateCallback);

    ROS_INFO("Listening to Franka states...");
    ros::spin();

    return 0;
}