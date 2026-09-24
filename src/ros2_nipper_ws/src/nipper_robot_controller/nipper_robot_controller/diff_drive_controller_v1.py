#!/usr/bin/env python3
# Written by Sarthak shirke @Nipper B.V.
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np

class MultiTurretKinematics(Node):
    def __init__(self):
        super().__init__('multi_turret_kinematics')

        # Sub to cmd_vel
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )

        # turret positions relative to base_link [m]
        dx_turret = 0.7
        dy_turret = 0.2

        # turret coordinates (x, y)
        rLF = ( dx_turret, -dy_turret)
        rRF = ( dx_turret,  dy_turret)
        rLR = (-dx_turret, -dy_turret)
        rRR = (-dx_turret,  dy_turret)
        R_turrets = (rLF, rRF, rLR, rRR)

        # 8×3 kinematic mapping matrix (vx, vy, omega)
        self.A = np.array([
            [1, 0,  R_turrets[0][1]],
            [0, 1,  R_turrets[0][0]],
            [1, 0,  R_turrets[1][1]],
            [0, 1,  R_turrets[1][0]],
            [1, 0,  R_turrets[2][1]],
            [0, 1,  R_turrets[2][0]],
            [1, 0,  R_turrets[3][1]],
            [0, 1,  R_turrets[3][0]],
        ])

        self.get_logger().info("Kinematic node initialized.")

    def cmd_callback(self, msg: Twist):
        # extract linear and angular velocities
        vx = msg.linear.x
        vy = msg.linear.y
        omega = msg.angular.z

        # print or log the incoming command velocities
        self.get_logger().info(f"Received cmd_vel → vx: {vx:.3f}, vy: {vy:.3f}, omega: {omega:.3f}")


        # robot velocity vector
        V = np.array([[vx], [vy], [omega]])

        # compute turret linear velocities (vx_i, vy_i)
        v_turrets = np.dot(self.A, V)

        # group into 4 turret velocity pairs (vx_i, vy_i)
        turret_vels = v_turrets.reshape((4, 2))

        self.get_logger().info("\n=== Turret velocities (vx, vy) [m/s] ===")
        turret_names = ["LF", "RF", "LR", "RR"]
        for name, (vx_i, vy_i) in zip(turret_names, turret_vels):
            self.get_logger().info(f"{name}: vx={vx_i:.3f}, vy={vy_i:.3f}")

        # TODO: You can now compute wheel angular velocities per turret if needed
        # e.g. using wheel radius and turret steering angle later.

def main(args=None):
    rclpy.init(args=args)
    node = MultiTurretKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
