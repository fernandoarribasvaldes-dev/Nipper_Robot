#!/usr/bin/env python3
# Written by Sarthak shirke @Nipper B.V.
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

        # Publisher for RLR wheels
        self.pub_rlr_left = self.create_publisher(Float64,
            '/wheelset_rear_left_left_wheel_joint/cmd_vel', 10)
        self.pub_rlr_right = self.create_publisher(Float64,
            '/wheelset_rear_left_right_wheel_joint/cmd_vel', 10)

        # Turret positions relative to base_link [m]
        dx_turret = 0.7
        dy_turret = 0.2

        # Turret coordinates (x, y)
        rLF = ( dx_turret, -dy_turret)
        rRF = ( dx_turret,  dy_turret)
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

        self.get_logger().info("Kinematic node initialized.")

    def cmd_callback(self, msg: Twist):
        # Extract linear and angular velocities
        vx = msg.linear.x
        vy = msg.linear.y
        omega = msg.angular.z

        # Log incoming command velocities
        self.get_logger().info(f"Received cmd_vel → vx: {vx:.3f}, vy: {vy:.3f}, omega: {omega:.3f}")

        # Robot velocity vector
        V = np.array([[vx], [vy], [omega]])

        # Compute turret linear velocities (vx_i, vy_i)
        v_turrets = np.dot(self.A, V)
        turret_vels = v_turrets.reshape((4, 2))
        turret_names = ["LF", "RF", "LR", "RR"]

        self.get_logger().info("\n=== Turret velocities (vx, vy) [m/s] ===")
        for name, (vx_i, vy_i) in zip(turret_names, turret_vels):
            self.get_logger().info(f"{name}: vx={vx_i:.3f}, vy={vy_i:.3f}")

        # Compute per-turret speed and direction
        self.get_logger().info("\n=== Wheel velocities [rad/s] ===")
        for name, (vx_i, vy_i) in zip(turret_names, turret_vels):
            # Turret speed and direction
            v_turret = np.sqrt(vx_i**2 + vy_i**2)
            phi_turret = np.arctan2(vy_i, vx_i)  # direction (rad)

            # Differential wheel velocities
            v_left  = v_turret - (self.d_wheel / 2) * 0  # assuming no turret rotation
            v_right = v_turret + (self.d_wheel / 2) * 0

            # Convert to rad/s
            omega_left  = v_left / self.r_wheel
            omega_right = v_right / self.r_wheel

            self.get_logger().info(
                f"{name}: phi={phi_turret:.3f} rad | omega_left={omega_left:.3f} rad/s, omega_right={omega_right:.3f} rad/s"
            )


def main(args=None):
    rclpy.init(args=args)
    node = MultiTurretKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
