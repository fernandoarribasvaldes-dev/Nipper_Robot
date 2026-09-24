#!/usr/bin/env python3
# Continuous steer-and-drive multi-turret controller + LIVE e_phi plot

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
            (dx,  -dy),   # LF (Left-Front turret)
            (-dx, -dy),   # LR (Left-Rear turret)
            (-dx,  dy),   # RR (Right-Rear turret)
            (dx,   dy),   # RF (Right-Front turret)
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

        self.turret_angles = [0.0, 0.0, 0.0, 0.0]
        self.latest_cmd = Twist()

        # PID
        self.heading_pid = [PID(5, 0.1, 3, i_limit=2.0) for _ in range(4)]  # kp, ki, kd

        # ------------------------------------------
        # LIVE PLOT SETUP
        # ------------------------------------------
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

        self.time_history = []
        self.e_phi_history = [[], [], [], []]  # 4 turrets

        self.lines = []
        # labels = ["LF", "LR", "RR", "RF"]
        labels = ["e_phi LF (deg)", "e_phi LR (deg)", "e_phi RR (deg)", "e_phi RF (deg)"]

        colors = ["r", "g", "b", "m"]

        for i in range(4):
            line, = self.axs[i].plot([], [], color=colors[i], label=f"e_phi {labels[i]}")
            self.lines.append(line)
            self.axs[i].set_ylabel(labels[i])

            # ---- Add zero reference line ----
            self.axs[i].axhline(0, color='k', linestyle='--', linewidth=0.8)
            self.axs[i].legend(loc="upper right")

        self.axs[-1].set_xlabel("Time [s]")

        # Plot update timer (10 Hz)
        self.create_timer(0.1, self.update_plot)

        # Control loop 50 Hz
        self.create_timer(0.02, self.control_step)

        self.start_time = time.time()

        self.get_logger().info("Controller + e_phi live plot initialized!")


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

        wheel_cmds = []

        t_now = time.time() - self.start_time
        self.time_history.append(t_now)

        for i, (vx_i, vy_i) in enumerate(v_turrets):
            v_bar = math.sqrt(vx_i**2 + vy_i**2)
            raw_phi = math.atan2(vy_i, vx_i)

            if abs(angle_wrap(raw_phi)) > math.pi/2:
                phi_des = angle_wrap(raw_phi + math.pi)
                v_bar = -v_bar
            else:
                phi_des = raw_phi

            phi_act = self.turret_angles[i]
            e_phi = angle_wrap(phi_des - phi_act)

            # Store for plot
            # self.e_phi_history[i].append(e_phi)   #This is in radians

            e_phi_deg = math.degrees(e_phi)
            self.e_phi_history[i].append(e_phi_deg)

            phi_dot = self.heading_pid[i].step(e_phi)

            v_left  = v_bar - 0.5 * self.d_wheel * phi_dot
            v_right = v_bar + 0.5 * self.d_wheel * phi_dot

            omega_left  = v_left / self.r_wheel
            omega_right = v_right / self.r_wheel

            wheel_cmds.extend([omega_left, omega_right])

        self.publish_wheels(wheel_cmds)


    # ---------------------------------------------------------
    def update_plot(self):
        if len(self.time_history) < 2:
            return

        for i in range(4):
            self.lines[i].set_data(self.time_history, self.e_phi_history[i])
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
