#!/usr/bin/env python3
# Continuous steer-and-drive multi-turret controller
# Written by Sarthak Shirke @Nipper B.V. (edited with ChatGPT)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
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

        p = self.kp * error
        self.i_term += self.ki * error * dt

        if self.i_limit is not None:
            self.i_term = max(min(self.i_term, self.i_limit), -self.i_limit)

        d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        return p + self.i_term + d


def angle_wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class MultiTurretKinematics(Node):
    def __init__(self):
        super().__init__('multi_turret_kinematics')

        # -----------------------------
        # Subscriptions
        # -----------------------------
        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)

        # Four joint_state topics
        self.create_subscription(JointState, "/joint_states_wheelset_left",
                                 self.js_left_callback, 10)
        self.create_subscription(JointState, "/joint_states_wheelset_rear_left",
                                 self.js_rl_callback, 10)
        self.create_subscription(JointState, "/joint_states_wheelset_rear_right",
                                 self.js_rr_callback, 10)
        self.create_subscription(JointState, "/joint_states_wheelset_right",
                                 self.js_right_callback, 10)

        # -----------------------------
        # Robot geometry
        # -----------------------------
        dx, dy = 0.7, 0.2

        # turret order: LF, LR, RR, RF
        self.R_turrets = [
            (dx,  -dy),   # LF
            (-dx, -dy),   # LR
            (-dx,  dy),   # RR
            (dx,   dy),   # RF
        ]

        # 8×3 mapping matrix (vx, vy, omega → vx_i, vy_i)
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
        self.r_wheel = 0.075
        self.d_wheel = 0.15

        # -----------------------------
        # Wheel publishers
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

        self.wheel_publishers = [
            self.create_publisher(Float64, t, 10) for t in wheel_topics
        ]

        # -----------------------------
        # State
        # -----------------------------
        self.turret_angles = [0.0, 0.0, 0.0, 0.0]  # LF, LR, RR, RF
        self.latest_cmd = Twist()

        # PID for turret headings
        self.heading_pid = [PID(0.001, 0.0, 0.0, i_limit=2.0) for _ in range(4)]

        # Main control loop (50 Hz)
        self.create_timer(0.02, self.control_step)

        self.get_logger().info("Continuous multi-turret kinematic controller initialized!")

    # -----------------------------
    # Joint State Callbacks
    # -----------------------------
    def js_left_callback(self, msg):
        joint_pos = dict(zip(msg.name, msg.position))
        if "wheelset_left_revolute_joint" in joint_pos:
            self.turret_angles[0] = joint_pos["wheelset_left_revolute_joint"]

    def js_rl_callback(self, msg):
        joint_pos = dict(zip(msg.name, msg.position))
        if "wheelset_rear_left_revolute_joint" in joint_pos:
            self.turret_angles[1] = joint_pos["wheelset_rear_left_revolute_joint"]

    def js_rr_callback(self, msg):
        joint_pos = dict(zip(msg.name, msg.position))
        if "wheelset_rear_right_revolute_joint" in joint_pos:
            self.turret_angles[2] = joint_pos["wheelset_rear_right_revolute_joint"]

    def js_right_callback(self, msg):
        joint_pos = dict(zip(msg.name, msg.position))
        if "wheelset_right_revolute_joint" in joint_pos:
            self.turret_angles[3] = joint_pos["wheelset_right_revolute_joint"]

    # -----------------------------
    # Command Callback
    # -----------------------------
    def cmd_callback(self, msg: Twist):
        self.latest_cmd = msg

    # -----------------------------
    # Main Control Step
    # -----------------------------
    def control_step(self):
        vx = self.latest_cmd.linear.x
        vy = self.latest_cmd.linear.y
        omega = self.latest_cmd.angular.z

        V = np.array([[vx], [vy], [omega]])
        v_turrets = np.dot(self.A, V).reshape((4, 2))

        wheel_cmds = []

        for i, (vx_i, vy_i) in enumerate(v_turrets):
            v_bar = math.sqrt(vx_i**2 + vy_i**2)
            raw_phi = math.atan2(vy_i, vx_i)
            print(f"[Turret {i}] vx_i={vx_i:.3f}, vy_i={vy_i:.3f}, atan2={raw_phi:.3f} rad ({math.degrees(raw_phi):.1f} deg)")

            # Flip backwards direction
            if abs(angle_wrap(raw_phi)) > math.pi / 2:
                phi_des = angle_wrap(raw_phi + math.pi)
                v_bar = -v_bar
            else:
                phi_des = raw_phi

            phi_act = self.turret_angles[i]
            e_phi = angle_wrap(phi_des - phi_act)

            phi_dot = self.heading_pid[i].step(e_phi)

            # phi_dot = 1

            v_left  = v_bar - 0.5 * self.d_wheel * phi_dot
            v_right = v_bar + 0.5 * self.d_wheel * phi_dot


            print(
                f"[Turret {i}] phi_des={phi_des:.3f}, "
                f"phi_act={phi_act:.3f}, "
                f"e_phi={e_phi:.3f}, "
                f"phi_dot={phi_dot:.3f}"
            )

            omega_left  = v_left / self.r_wheel
            omega_right = v_right / self.r_wheel

            wheel_cmds.extend([omega_left, omega_right])

        self.publish_wheels(wheel_cmds)

    # -----------------------------
    # Publisher helper
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
