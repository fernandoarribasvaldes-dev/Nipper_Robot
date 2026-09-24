#!/usr/bin/env python3
# Written by Sarthak Shirke @Nipper B.V.

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Float64MultiArray
import numpy as np
import math
import time


class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_term = 0
        self.prev_error = 0
        self.prev_time = None

    def step(self, error):
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            return 0.0

        dt = now - self.prev_time
        self.prev_time = now

        p = self.kp * error
        self.i_term += self.ki * error * dt
        d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        return p + self.i_term + d


class MultiTurretKinematics(Node):
    def __init__(self):
        super().__init__('multi_turret_kinematics')

        # -----------------------------
        # Subscriptions
        # -----------------------------
        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(Float64MultiArray, "/turret_angles", self.angles_callback, 10)

        # -----------------------------
        # Robot model
        # -----------------------------
        dx, dy = 0.7, 0.2
        self.R_turrets = [
            (dx, -dy),   # LF
            (dx,  dy),   # RF
            (-dx, -dy),  # LR
            (-dx,  dy)   # RR
        ]

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

        self.r_wheel = 0.07
        self.d_wheel = 0.15

        # -----------------------------
        # Publishers
        # -----------------------------
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
        self.wheel_publishers = [self.create_publisher(Float64, t, 10) for t in wheel_topics]

        # -----------------------------
        # State machine
        # -----------------------------
        self.mode = "ALIGN"
        self.turret_angles = [0.0, 0.0, 0.0, 0.0]
        self.theta_targets = [0.0, 0.0, 0.0, 0.0]

        # PID controllers for turret alignment
        self.pid = [PID(5.0, 0.0, 0.2) for _ in range(4)]

        self.latest_cmd = Twist()

        self.get_logger().info("Multi-turret kinematic controller initialized!")

    # -----------------------------
    # Callback for turret angles
    # -----------------------------
    def angles_callback(self, msg):
        self.turret_angles = list(msg.data)

    # -----------------------------
    # Callback for cmd_vel
    # -----------------------------
    def cmd_callback(self, msg):
        self.latest_cmd = msg

        vx = msg.linear.x
        vy = msg.linear.y
        omega = msg.angular.z

        # Compute turret-local velocities
        V = np.array([[vx], [vy], [omega]])
        v_turrets = np.dot(self.A, V).reshape((4, 2))

        # Compute target turret angles
        self.theta_targets = [
            math.atan2(vy_i, vx_i) for (vx_i, vy_i) in v_turrets
        ]

    # -----------------------------
    # The main control loop
    # -----------------------------
    def control_step(self):
        # Compute angle errors
        errors = [
            self.theta_targets[i] - self.turret_angles[i]
            for i in range(4)
        ]

        max_err = max(abs(e) for e in errors)

        # -----------------------------
        # MODE SELECTION
        # -----------------------------
        if max_err > 0.1:   # tolerance rad
            self.mode = "ALIGN"
        else:
            self.mode = "DRIVE"

        if self.mode == "ALIGN":
            self.run_align_mode(errors)
        else:
            self.run_drive_mode()

    # -----------------------------
    # ALIGN MODE
    # -----------------------------
    def run_align_mode(self, errors):
        wheel_cmds = []

        for i in range(4):
            phi_dot = self.pid[i].step(errors[i])

            v_left = -phi_dot * (self.d_wheel / 2.0)
            v_right = +phi_dot * (self.d_wheel / 2.0)

            wheel_cmds.extend([v_left / self.r_wheel, v_right / self.r_wheel])

        self.publish_wheels(wheel_cmds)

    # -----------------------------
    # DRIVE MODE
    # -----------------------------
    def run_drive_mode(self):
        vx = self.latest_cmd.linear.x
        vy = self.latest_cmd.linear.y
        omega = self.latest_cmd.angular.z

        V = np.array([[vx], [vy], [omega]])
        v_turrets = np.dot(self.A, V).reshape((4, 2))

        wheel_cmds = []
        for (vx_i, vy_i) in v_turrets:
            v_direction = np.sign(vx_i) * math.sqrt(vx_i**2 + vy_i**2)
            omega_left = v_direction / self.r_wheel
            omega_right = v_direction / self.r_wheel
            wheel_cmds.extend([omega_left, omega_right])

        self.publish_wheels(wheel_cmds)

    # -----------------------------
    # Wheel publishing helper
    # -----------------------------
    def publish_wheels(self, cmd_list):
        for pub, val in zip(self.wheel_publishers, cmd_list):
            msg = Float64()
            msg.data = val
            pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MultiTurretKinematics()

    timer = node.create_timer(0.02, node.control_step)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
