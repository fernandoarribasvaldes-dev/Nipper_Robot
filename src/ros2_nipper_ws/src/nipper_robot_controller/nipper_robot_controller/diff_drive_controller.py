#!/usr/bin/env python3
# Continuous steer-and-drive multi-turret controller + LIVE e_phi, phi_des, phi_act plot

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
import numpy as np
import math
import time
import matplotlib.pyplot as plt


# ===========================================================
# PID
# ===========================================================
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


# ===========================================================
# MultiTurret + Live Plot
# ===========================================================
class MultiTurretKinematics(Node):
    def __init__(self):
        super().__init__('multi_turret_kinematics')

        # ------------------------------------------
        # ROS Subscriptions
        # ------------------------------------------
        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)

        # ------------------------------------------
        # Geometry
        # ------------------------------------------
        dx, dy = 0.7, 0.2
        self.R_turrets = [
            (dx,  -dy),   # LF
            (-dx, -dy),   # LR
            (-dx,  dy),   # RR
            (dx,   dy),   # RF
        ]

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

        self.r_wheel = 0.075
        self.d_wheel = 0.15

        wheel_topics = [
            "/wheelset_left_left_wheel_joint/cmd_vel",
            "/wheelset_left_right_wheel_joint/cmd_vel",
            "/wheelset_d_left_left_wheel_joint/cmd_vel",
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
        self.turret_angles = [0.0, 0.0, 0.0, 0.0]
        self.latest_cmd = Twist()

        # PID
        self.heading_pid = [PID(10, 0.1, 5, i_limit=2.0) for _ in range(4)]
        # self.heading_pid = [PID(10, 0.0, 5, i_limit=2.0) for _ in range(4)]
        # self.heading_pid = [PID(0.0, 0.0, 0.0, i_limit=2.0) for _ in range(4)]

        # ------------------------------------------
        # LIVE PLOT SETUP
        # ------------------------------------------
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

        self.time_history = []
        self.e_phi_history = [[], [], [], []]
        self.phi_des_history = [[], [], [], []]
        self.phi_act_history = [[], [], [], []]

        colors = ["r", "g", "b", "m"]
        labels = ["LF", "LR", "RR", "RF"]

        self.lines_err = []
        self.lines_phi_des = []
        self.lines_phi_act = []

        for i in range(4):
            # e_phi (error)
            line_err, = self.axs[i].plot([], [], color=colors[i], label=f"e_phi {labels[i]} (deg)")
            self.lines_err.append(line_err)

            # Desired angle (blue dashed)
            line_des, = self.axs[i].plot([], [], '--', color='blue', label='phi_des')
            self.lines_phi_des.append(line_des)

            # Actual angle (green)
            line_act, = self.axs[i].plot([], [], color='green', label='phi_act')
            self.lines_phi_act.append(line_act)

            self.axs[i].axhline(0, color='k', linestyle='--', linewidth=0.7)
            self.axs[i].set_ylabel(labels[i])
            self.axs[i].legend(loc='upper right')

        self.axs[-1].set_xlabel("Time [s]")

        # Timers
        self.create_timer(0.02, self.control_step)   # 50 Hz control
        self.create_timer(0.1, self.update_plot)     # 10 Hz plot

        self.start_time = time.time()

        self.get_logger().info("Controller + live plots initialized!")

    # ---------------------------------------------------------
    def joint_state_callback(self, msg):
        joint_pos = dict(zip(msg.name, msg.position))
        self.turret_angles = [
            joint_pos.get("wheelset_left_revolute_joint", 0.0),
            joint_pos.get("wheelset_rear_left_revolute_joint", 0.0),
            joint_pos.get("wheelset_rear_right_revolute_joint", 0.0),
            joint_pos.get("wheelset_right_revolute_joint", 0.0),
        ]

    def cmd_callback(self, msg):
        self.latest_cmd = msg

    # ---------------------------------------------------------
    def control_step(self):
        vx = self.latest_cmd.linear.x
        vy = self.latest_cmd.linear.y
        omega = self.latest_cmd.angular.z

        V = np.array([[vx], [vy], [omega]])
        v_turrets = np.dot(self.A, V).reshape((4, 2))
        turret_names = ["LF", "RF", "LR", "RR"]

        wheel_cmds = []
        t_now = time.time() - self.start_time
        self.time_history.append(t_now)

        # wheel_omegas = []

        for i, (vx_i, vy_i) in enumerate(v_turrets):

            v_bar = math.sqrt(vx_i**2 + vy_i**2)
            raw_phi = math.atan2(vy_i, vx_i)

            self.get_logger().info(
                f"[{i}] vx_i={vx_i:+.4f}, vy_i={vy_i:+.4f}, "
                f"v_bar={v_bar:.4f}, raw_phi={math.degrees(raw_phi):+.2f} deg"
            )

            if abs(angle_wrap(raw_phi)) > math.pi/2:
                phi_des = angle_wrap(raw_phi + math.pi)
                v_bar = -v_bar
            else:
                phi_des = raw_phi


            self.get_logger().info(
                f"[{i}] vx_i={vx_i:+.4f}, vy_i={vy_i:+.4f}, "
                f"v_bar={v_bar:.4f}, phi_des={math.degrees(phi_des):+.2f} deg"
            )

            phi_act = self.turret_angles[i]
            e_phi = angle_wrap(phi_des - phi_act)

            # Convert to degrees for plotting
            e_phi_deg = math.degrees(e_phi)
            phi_des_deg = math.degrees(phi_des)
            phi_act_deg = math.degrees(phi_act)

            self.e_phi_history[i].append(e_phi_deg)
            self.phi_des_history[i].append(phi_des_deg)
            self.phi_act_history[i].append(phi_act_deg)

            phi_dot = self.heading_pid[i].step(e_phi)

            v_left  = v_bar - 0.5 * self.d_wheel * phi_dot
            v_right = v_bar + 0.5 * self.d_wheel * phi_dot

            omega_left  = v_left / self.r_wheel
            omega_right = v_right / self.r_wheel

            # wheel_omegas.append((omega_left, omega_right))

            wheel_cmds.extend([omega_left, omega_right])

        # Log wheel angular velocities
        # self.get_logger().info("\n=== Wheel angular velocities [rad/s] ===")
        # for name, (omega_left, omega_right) in zip(turret_names, wheel_omegas):
        #     self.get_logger().info(f"{name}: ω_left={omega_left:.3f}, ω_right={omega_right:.3f}")

        self.publish_wheels(wheel_cmds)

    # ---------------------------------------------------------
    def update_plot(self):
        if len(self.time_history) < 2:
            return

        for i in range(4):
            self.lines_err[i].set_data(self.time_history, self.e_phi_history[i])
            self.lines_phi_des[i].set_data(self.time_history, self.phi_des_history[i])
            self.lines_phi_act[i].set_data(self.time_history, self.phi_act_history[i])

            self.axs[i].relim()
            self.axs[i].autoscale_view()

        plt.draw()
        plt.pause(0.001)

    # ---------------------------------------------------------
    def publish_wheels(self, cmd_list):
        for pub, val in zip(self.wheel_publishers, cmd_list):
            msg = Float64()
            msg.data = float(val)
            pub.publish(msg)


# ===========================================================
def main(args=None):
    rclpy.init(args=args)
    node = MultiTurretKinematics()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
