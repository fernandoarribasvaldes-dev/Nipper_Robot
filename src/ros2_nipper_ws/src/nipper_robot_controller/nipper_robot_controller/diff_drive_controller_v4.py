#!/usr/bin/env python3
# Written by Sarthak Shirke @Nipper B.V.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
import numpy as np


class MultiTurretKinematics(Node):
    def __init__(self):
        super().__init__('multi_turret_kinematics')

        # Subscribe to cmd_vel
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )

        # Turret positions relative to base_link [m]
        dx_turret = 0.7
        dy_turret = 0.2

        # Turret coordinates (x, y)
        rLF = (dx_turret, -dy_turret)
        rRF = (dx_turret,  dy_turret)
        rLR = (-dx_turret, -dy_turret)
        rRR = (-dx_turret,  dy_turret)
        self.R_turrets = (rLF, rRF, rLR, rRR)

        # 8×3 kinematic mapping matrix (vx, vy, omega)
        self.A = np.array([
            [1, 0,  self.R_turrets[0][1]],
            [0, 1,  self.R_turrets[0][0]],
            [1, 0,  self.R_turrets[1][1]],
            [0, 1,  self.R_turrets[1][0]],
            [1, 0,  self.R_turrets[2][1]],
            [0, 1,  self.R_turrets[2][0]],
            [1, 0,  self.R_turrets[3][1]],
            [0, 1,  self.R_turrets[3][0]],
        ])

        # Wheel parameters
        self.r_wheel = 0.07   # m
        self.d_wheel = 0.15   # m (distance between left and right wheels)

        # Publishers for all 8 wheels
        wheel_topics = [
            "/wheelset_left_left_wheel_joint/cmd_vel",
            "/wheelset_left_right_wheel_joint/cmd_vel",
            "/wheelset_rear_left_left_wheel_joint/cmd_vel",
            "/wheelset_rear_left_right_wheel_joint/cmd_vel",
            "/wheelset_rear_right_left_wheel_joint/cmd_vel",
            "/wheelset_rear_right_right_wheel_joint/cmd_vel",
            "/wheelset_right_left_wheel_joint/cmd_vel",
            "/wheelset_right_right_wheel_joint/cmd_vel",
        ]

        self.wheel_publishers = [self.create_publisher(Float64, topic, 10) for topic in wheel_topics]

        self.get_logger().info("Kinematic node initialized and all wheel publishers created.")

    def cmd_callback(self, msg: Twist):
        # Extract velocities
        vx = msg.linear.x
        vy = msg.linear.y
        omega = msg.angular.z

        self.get_logger().info(f"Received cmd_vel → vx: {vx:.3f}, vy: {vy:.3f}, omega: {omega:.3f}")

        # Robot velocity vector
        V = np.array([[vx], [vy], [omega]])

        # Compute turret linear velocities
        v_turrets = np.dot(self.A, V)
        turret_vels = v_turrets.reshape((4, 2))
        turret_names = ["LF", "RF", "LR", "RR"]

        # Log turret velocities
        self.get_logger().info("\n=== Turret velocities (vx, vy) [m/s] ===")
        for name, (vx_i, vy_i) in zip(turret_names, turret_vels):
            self.get_logger().info(f"{name}: vx={vx_i:.3f}, vy={vy_i:.3f}")

        # Compute wheel angular velocities for each turret
        wheel_omegas = []
        for vx_i, vy_i in turret_vels:
            v_turret = np.sign(vx_i) * np.sqrt(vx_i**2 + vy_i**2)
            v_left = v_turret
            v_right = v_turret
            omega_left = v_left / self.r_wheel
            omega_right = v_right / self.r_wheel
            wheel_omegas.append((omega_left, omega_right))

        # Log wheel angular velocities
        self.get_logger().info("\n=== Wheel angular velocities [rad/s] ===")
        for name, (omega_left, omega_right) in zip(turret_names, wheel_omegas):
            self.get_logger().info(f"{name}: ω_left={omega_left:.3f}, ω_right={omega_right:.3f}")

        # Combine all omegas into one list (8 wheels total)
        all_omegas = [
            wheel_omegas[0][0], wheel_omegas[0][1],  # LF
            wheel_omegas[2][0], wheel_omegas[2][1],  # LR
            wheel_omegas[3][0], wheel_omegas[3][1],  # RR
            wheel_omegas[1][0], wheel_omegas[1][1],  # RF
        ]

        # Publish each omega
        for pub, omega in zip(self.wheel_publishers, all_omegas):
            msg_out = Float64()
            msg_out.data = omega
            pub.publish(msg_out)

        self.get_logger().info("Published all 8 wheel velocities successfully.")


def main(args=None):
    rclpy.init(args=args)
    node = MultiTurretKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
