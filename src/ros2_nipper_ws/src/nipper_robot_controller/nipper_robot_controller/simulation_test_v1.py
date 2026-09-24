import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from std_msgs.msg import Int32MultiArray

class WheelVelocityNode(Node):
    def __init__(self):
        super().__init__("wheel_velocity_node")

        # Subscriber to OPC UA values
        self.subscription = self.create_subscription(
            Int32MultiArray,
            "/drive_actual_velocities",
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

        # /home/sarthakshirke/Documents/Nipper_data_sim_test/rosbag2_2025_11_26-16_14_46 :- Forward and backward

        # rosbag2_2025_11_26-16_14_46_0.mcap :- SPin from center axis

        # Wheel topics (8 wheels)
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

        # Create 8 publishers
        self.wheel_publishers = [
            self.create_publisher(Float64, topic, 10) 
            for topic in wheel_topics
        ]

        # Constants
        self.gear_ratio = 2.0
        self.radius = 0.07

        self.get_logger().info("Wheel Velocity Node Started")

    def listener_callback(self, msg: Int32MultiArray):
        raw_values = msg.data  # 8 raw values from PLC

        if len(raw_values) != 8:
            self.get_logger().error("Expected 8 velocities but received something else!")
            return

        processed = []

        for v_mm_s in raw_values:
            v_m_s = v_mm_s / 1000.0          # mm/s → m/s
            v_m_s = v_m_s / self.gear_ratio  # gear reduction
            rad_s = v_m_s / self.radius      # linear → angular speed
            processed.append(rad_s)

        # Flip the sign for wheels 2, 4, 6, 8 (1,3,5,7 index) 4, 6, 7
        flip_indices = [0,3,5,6]
        for i in flip_indices:
            processed[i] = -processed[i]

        # Publish each wheel velocity (FIXED HERE)
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
