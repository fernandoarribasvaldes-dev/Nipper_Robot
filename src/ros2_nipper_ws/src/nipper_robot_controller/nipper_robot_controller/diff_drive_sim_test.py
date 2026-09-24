#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class DirectWheelPublisher(Node):
    def __init__(self):
        super().__init__('direct_wheel_publisher')

        # List of wheel topics
        # wheel_topics = [
        #     "/wheelset_left_left_wheel_joint/cmd_vel",
        #     "/wheelset_left_right_wheel_joint/cmd_vel",
        #     "/wheelset_rear_left_left_wheel_joint/cmd_vel",
        #     "/wheelset_rear_left_right_wheel_joint/cmd_vel",
        #     "/wheelset_rear_right_left_wheel_joint/cmd_vel",
        #     "/wheelset_rear_right_right_wheel_joint/cmd_vel",
        #     "/wheelset_right_left_wheel_joint/cmd_vel",
        #     "/wheelset_right_right_wheel_joint/cmd_vel",
        # ]

        wheel_topics = [
            "/wheelset_right_left_wheel_joint/cmd_vel",
            "/wheelset_right_right_wheel_joint/cmd_vel",
            "/wheelset_rear_right_left_wheel_joint/cmd_vel",
            "/wheelset_rear_right_right_wheel_joint/cmd_vel",
            "/wheelset_rear_left_left_wheel_joint/cmd_vel",
            "/wheelset_rear_left_right_wheel_joint/cmd_vel",
            "/wheelset_left_left_wheel_joint/cmd_vel",
            "/wheelset_left_right_wheel_joint/cmd_vel"
        ]

        # Create publishers for all 8 wheels
        self.wheel_publishers = [self.create_publisher(Float64, topic, 10) for topic in wheel_topics]

        # Example wheel speeds [rad/s]

        # self.wheel_speeds = [
        #     3.0, 3.0,   # left-front  (LF_L, LF_R)
        #     3.0, 3.0,   # left-rear   (LR_L, LR_R)
        #     3.0, 3.0,   # right-rear  (RR_L, RR_R)
        #     3.0, 3.0    # right-front (RF_L, RF_R)
        # ]


        self.wheel_speeds = [-3.2357142857142853, 3.464285714285714, 
                             3.478571428571428, -3.9214285714285713, 
                             4.042857142857142, -4.121428571428571, 
                             -4.414285714285714, 4.828571428571428]


        # Create timer to continuously publish velocities (10 Hz)
        self.timer = self.create_timer(0.1, self.publish_wheel_speeds)
        self.get_logger().info("Direct wheel publisher initialized.")

    def publish_wheel_speeds(self):
        for pub, speed in zip(self.wheel_publishers, self.wheel_speeds):
            msg = Float64()
            msg.data = speed
            pub.publish(msg)

        self.get_logger().info(f"Published wheel speeds: {self.wheel_speeds}")

def main(args=None):
    rclpy.init(args=args)
    node = DirectWheelPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
