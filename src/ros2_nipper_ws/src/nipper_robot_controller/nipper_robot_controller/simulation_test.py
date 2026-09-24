import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import Int32MultiArray

class WheelVelocityNode(Node):
    def __init__(self):
        super().__init__("wheel_velocity_node")

        # Subscriber to full drive data
        self.subscription = self.create_subscription(
            Int32MultiArray,
            "/drive_full_status",
            self.listener_callback,
            10
        )

        # drive1: front left turret right motor

        # drive2: front left turret left motor

        # drive3: rear right turret left motor

        # ​drive4: rear right turret right motor

        # ​drive5: rear left turret left motor

        # ​drive6: rear left turret right motor

        # ​drive7: front right turret right motor

        # ​drive8: front right turret left motor

        wheel_topics = [
            "/wheelset_left_right_wheel_joint/cmd_vel",
            "/wheelset_left_left_wheel_joint/cmd_vel",
            "/wheelset_rear_right_left_wheel_joint/cmd_vel",
            "/wheelset_rear_right_right_wheel_joint/cmd_vel",
            "/wheelset_rear_left_left_wheel_joint/cmd_vel",
            "/wheelset_rear_left_right_wheel_joint/cmd_vel",
            "/wheelset_right_right_wheel_joint/cmd_vel",
            "/wheelset_right_left_wheel_joint/cmd_vel"
        ]

        self.wheel_publishers = [
            self.create_publisher(Float64, topic, 10) 
            for topic in wheel_topics
        ]

        self.gear_ratio = 2.0
        self.radius = 0.07

        self.get_logger().info("Wheel Velocity Node Started")


        # rosbag2_2025_12_03-15_35_08 :- This has few rotations and straight paths 
        # rosbag2_2025_12_03-15_50_35 :- This 0.6 m straight path and back 0.6 m
        # rosbag2_2025_12_03-15_54_14 :- This 0.6 m straight path and back 0.6 m but this doesn't has delays in forward and backward motion
        # rosbag2_2025_12_03-15_57_40 :- This rotation about the geometric center of the robot.


    def listener_callback(self, msg: Int32MultiArray):
        data = msg.data

        # -----------------------------------------------------
        # Extract groups from drive_full_status
        # -----------------------------------------------------
        din_safe_speed = data[0]
        actual_values  = data[1:9]
        target_values  = data[9:17]
        position_vals  = data[17:25]

        # Print everything
        self.get_logger().info(f"dinSafeSpeed: {din_safe_speed}")
        self.get_logger().info(f"Actual Vel:  {actual_values}")
        self.get_logger().info(f"Target Vel:  {target_values}")
        self.get_logger().info(f"Position:    {position_vals}")

        # -----------------------------------------------------
        # Convert ONLY actual velocities → wheel speeds (rad/s)
        # -----------------------------------------------------
        processed = []
        for v_mm_s in actual_values:
            v_m_s = v_mm_s / 1000.0          # mm/s → m/s
            v_m_s = v_m_s / self.gear_ratio  # gear ratio
            rad_s = v_m_s / self.radius      # m/s → rad/s
            processed.append(rad_s)

        # Flip signs for selected wheels
        flip_indices = [0, 3, 5, 6]  # your mapping
        for i in flip_indices:
            processed[i] = -processed[i]

        # -----------------------------------------------------
        # Publish wheel speeds
        # -----------------------------------------------------
        for pub, value in zip(self.wheel_publishers, processed):
            msg_out = Float64()
            msg_out.data = value
            pub.publish(msg_out)

        self.get_logger().info(f"Published wheel speeds (rad/s): {processed}")


def main(args=None):
    rclpy.init(args=args)
    node = WheelVelocityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
