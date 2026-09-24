#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import numpy as np
import math
import time


def angle_wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class MultiTurretInverseKinematics(Node):
    """
    Estimate base twist (vx, vy, omega) from:
      - turret steering angles (from joint positions)
      - wheel angular velocities (from joint velocities)

    Publishes: /cmd_vel_estimated (geometry_msgs/Twist)
    """

    def __init__(self):
        super().__init__("multi_turret_inverse_kinematics")

        # ----------------------------
        # Geometry (same as your node)
        # ----------------------------
        dx, dy = 0.7, 0.2
        # Order: LF, LR, RR, RF  (same as your A matrix)
        self.R_turrets = [
            (dx,  -dy),   # LF
            (-dx, -dy),   # LR
            (-dx,  dy),   # RR
            (dx,   dy),   # RF
        ]

        # Build A exactly as in your forward node
        self.A = np.array([
            [1, 0, self.R_turrets[0][1]],
            [0, 1, self.R_turrets[0][0]],
            [1, 0, self.R_turrets[1][1]],
            [0, 1, self.R_turrets[1][0]],
            [1, 0, self.R_turrets[2][1]],
            [0, 1, self.R_turrets[2][0]],
            [1, 0, self.R_turrets[3][1]],
            [0, 1, self.R_turrets[3][0]],
        ])

        # Precompute pseudo-inverse A^+
        self.A_pinv = np.linalg.pinv(self.A)

        # Wheel params (match your forward code)
        self.r_wheel = 0.075
        self.d_wheel = 0.15

        # ----------------------------
        # State
        # ----------------------------
        # turret_angles[0..3] = LF, LR, RR, RF
        self.turret_angles = [0.0, 0.0, 0.0, 0.0]

        # 8 wheel angular velocities in order:
        # LF_L, LF_R, LR_L, LR_R, RR_L, RR_R, RF_L, RF_R
        self.wheel_omegas = [0.0] * 8
        self.last_update_time = None

        # ----------------------------
        # ROS interfaces
        # ----------------------------
        self.create_subscription(JointState, "/joint_states",
                                 self.joint_state_callback, 20)

        self.cmd_est_pub = self.create_publisher(Twist,
                                                 "/cmd_vel_estimated", 10)

        # Run estimation at fixed rate
        self.create_timer(0.02, self.estimate_twist)   # 50 Hz

        self.get_logger().info("MultiTurretInverseKinematics initialized")

    # ------------------------------------------------------
    # JointStates: read turret angles + wheel angular vels
    # ------------------------------------------------------
    def joint_state_callback(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity))

        # Turret steering joints
        # Same order as before: [LF, LR, RR, RF]
        self.turret_angles[0] = name_to_pos.get(
            "wheelset_left_revolute_joint", 0.0
        )
        self.turret_angles[1] = name_to_pos.get(
            "wheelset_rear_left_revolute_joint", 0.0
        )
        self.turret_angles[2] = name_to_pos.get(
            "wheelset_rear_right_revolute_joint", 0.0
        )
        self.turret_angles[3] = name_to_pos.get(
            "wheelset_right_revolute_joint", 0.0
        )

        # Wheel joints – keep the same mapping as your forward node
        # Order: LF_L, LF_R, LR_L, LR_R, RR_L, RR_R, RF_L, RF_R
        self.wheel_omegas[0] = name_to_vel.get(
            "wheelset_left_left_wheel_joint", 0.0
        )
        self.wheel_omegas[1] = name_to_vel.get(
            "wheelset_left_right_wheel_joint", 0.0
        )

        self.wheel_omegas[2] = name_to_vel.get(
            "wheelset_rear_left_left_wheel_joint", 0.0
        )
        self.wheel_omegas[3] = name_to_vel.get(
            "wheelset_rear_left_right_wheel_joint", 0.0
        )

        self.wheel_omegas[4] = name_to_vel.get(
            "wheelset_rear_right_left_wheel_joint", 0.0
        )
        self.wheel_omegas[5] = name_to_vel.get(
            "wheelset_rear_right_right_wheel_joint", 0.0
        )

        self.wheel_omegas[6] = name_to_vel.get(
            "wheelset_right_left_wheel_joint", 0.0
        )
        self.wheel_omegas[7] = name_to_vel.get(
            "wheelset_right_right_wheel_joint", 0.0
        )

        self.last_update_time = time.time()

    # ------------------------------------------------------
    # Main estimation step
    # ------------------------------------------------------
    def estimate_twist(self):
        # Wait until we have at least one joint_states update
        if self.last_update_time is None:
            return

        # Build v_turrets_flat = [vx1, vy1, ..., vx4, vy4]^T
        v_turrets_flat = np.zeros((8, 1))

        for i in range(4):
            # 2 wheels per turret
            omega_left = self.wheel_omegas[2*i]
            omega_right = self.wheel_omegas[2*i + 1]

            # Convert to linear speed at wheel contact
            v_left = omega_left * self.r_wheel
            v_right = omega_right * self.r_wheel

            # Average translational speed for turret
            v_bar = 0.5 * (v_left + v_right)

            phi = self.turret_angles[i]

            # Turret frame velocity in global robot frame
            vx_i = v_bar * math.cos(phi)
            vy_i = v_bar * math.sin(phi)

            v_turrets_flat[2*i, 0] = vx_i
            v_turrets_flat[2*i + 1, 0] = vy_i

        # Solve V = [vx, vy, omega]^T via least squares
        V = self.A_pinv.dot(v_turrets_flat)  # shape (3,1)

        vx_est = float(V[0, 0])
        vy_est = float(V[1, 0])
        omega_est = float(V[2, 0])

        # Publish as a Twist
        msg = Twist()
        msg.linear.x = vx_est
        msg.linear.y = vy_est
        msg.angular.z = omega_est

        self.cmd_est_pub.publish(msg)

        # Optional: debug log
        self.get_logger().info(
            f"Estimated base: vx={vx_est:.3f}, vy={vy_est:.3f}, omega={omega_est:.3f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MultiTurretInverseKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
