#!/usr/bin/env python3
# Continuous steer-and-drive multi-turret controller
# Written by Sarthak Shirke @Nipper B.V. (edited with ChatGPT)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
import math
import time


class PID:
    def __init__(self, kp, ki, kd, i_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_term = 0.0
        self.prev_error = 0.0
        self.prev_time = None
        self.i_limit = i_limit

    def step(self, error: float) -> float:
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            self.prev_error = error
            return 0.0

        dt = now - self.prev_time
        if dt <= 0.0:
            return 0.0

        self.prev_time = now

        # P
        p = self.kp * error

        # I
        self.i_term += self.ki * error * dt
        if self.i_limit is not None:
            self.i_term = max(min(self.i_term, self.i_limit), -self.i_limit)

        # D
        d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        return p + self.i_term + d


def angle_wrap(angle: float) -> float:
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class MultiTurretKinematics(Node):
    def __init__(self):
        super().__init__('multi_turret_kinematics')

        # -----------------------------
        # Subscriptions
        # -----------------------------
        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)


        # -----------------------------
        # Robot geometry
        # -----------------------------
        dx, dy = 0.7, 0.2

        # Order here MUST match:
        #   0: LF, 1: LR, 2: RR, 3: RF
        self.R_turrets = [
            (dx,  -dy),   # LF
            (-dx, -dy),   # LR
            (-dx,  dy),   # RR
            (dx,   dy),   # RF
        ]

        # Build 8x3 mapping matrix A (vx, vy, omega) -> (vx_i, vy_i)
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
        self.r_wheel = 0.07   # [m]
        self.d_wheel = 0.15   # [m] distance between left & right wheels

        # -----------------------------
        # Publishers for 8 wheels
        # -----------------------------
        wheel_topics = [
            "/wheelset_left_left_wheel_joint/cmd_vel",        # LF_L
            "/wheelset_left_right_wheel_joint/cmd_vel",       # LF_R
            "/wheelset_rear_left_left_wheel_joint/cmd_vel",   # LR_L
            "/wheelset_rear_left_right_wheel_joint/cmd_vel",  # LR_R
            "/wheelset_rear_right_left_wheel_joint/cmd_vel",  # RR_L
            "/wheelset_rear_right_right_wheel_joint/cmd_vel", # RR_R
            "/wheelset_right_left_wheel_joint/cmd_vel",       # RF_L
            "/wheelset_right_right_wheel_joint/cmd_vel",      # RF_R
        ]
        self.wheel_publishers = [
            self.create_publisher(Float64, t, 10) for t in wheel_topics
        ]

        # -----------------------------
        # State
        # -----------------------------
        self.turret_angles = [0.0, 0.0, 0.0, 0.0]   # measured [LF, LR, RR, RF]
        self.latest_cmd = Twist()                   # last /cmd_vel

        # PID controllers for each turret heading
        self.heading_pid = [
            PID(kp=10.0, ki=0.0, kd=0.0, i_limit=2.0) for _ in range(4)
        ]

        # Control loop at 50 Hz
        self.create_timer(0.02, self.control_step)

        self.get_logger().info("Continuous multi-turret kinematic controller initialized!")

    # -----------------------------
    # Callbacks
    # -----------------------------


    def joint_state_callback(self, msg: JointState):
        # Build dictionary: {joint_name: position}
        joint_pos = dict(zip(msg.name, msg.position))

        # Read turrets in LF, LR, RR, RF order
        self.turret_angles = [
            joint_pos.get("wheelset_left_revolute_joint"),        # LF
            joint_pos.get("wheelset_rear_left_revolute_joint"),   # LR
            joint_pos.get("wheelset_rear_right_revolute_joint"),  # RR
            joint_pos.get("wheelset_right_revolute_joint"),       # RF
        ]

    # def angles_callback(self, msg: Float64MultiArray):
    #     # Expect 4 angles (rad) in same order as R_turrets
    #     if len(msg.data) >= 4:
    #         self.turret_angles = list(msg.data[:4])

    def cmd_callback(self, msg: Twist):
        self.latest_cmd = msg

    # -----------------------------
    # Main control step
    # -----------------------------
    def control_step(self):
        # Extract commanded base velocity
        vx = self.latest_cmd.linear.x
        vy = self.latest_cmd.linear.y
        omega = self.latest_cmd.angular.z

        # 1) Robot velocity vector
        V = np.array([[vx], [vy], [omega]])

        # 2) Turret-local velocities (vx_i, vy_i)
        v_turrets = np.dot(self.A, V).reshape((4, 2))

        wheel_cmds = []

        # 3) For each turret: compute v_bar, phi_des, phi_dot, then v_L, v_R
        for i, (vx_i, vy_i) in enumerate(v_turrets):
            # Desired magnitude and heading
            v_bar = math.sqrt(vx_i**2 + vy_i**2)
            phi_des = math.atan2(vy_i, vx_i)

            phi_act = self.turret_angles[i]
            e_phi = angle_wrap(phi_des - phi_act)

            # Heading controller output = desired phi_dot
            phi_dot = self.heading_pid[i].step(e_phi)

            # Differential drive combination:
            # v_bar = (vL + vR)/2,  phi_dot = (vL - vR)/d
            v_left  = v_bar - 0.5 * self.d_wheel * phi_dot
            v_right = v_bar + 0.5 * self.d_wheel * phi_dot

            # Convert to angular wheel velocities
            omega_left  = v_left / self.r_wheel
            omega_right = v_right / self.r_wheel

            wheel_cmds.extend([omega_left, omega_right])

        # wheel_cmds order: [LF_L, LF_R, LR_L, LR_R, RR_L, RR_R, RF_L, RF_R]
        self.publish_wheels(wheel_cmds)

    # -----------------------------
    # Helper
    # -----------------------------
    def publish_wheels(self, cmd_list):
        for pub, val in zip(self.wheel_publishers, cmd_list):
            msg = Float64()
            msg.data = float(val)
            pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MultiTurretKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
