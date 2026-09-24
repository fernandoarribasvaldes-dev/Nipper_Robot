#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Float64
from sensor_msgs.msg import JointState

import matplotlib.pyplot as plt
import time


class WheelComparePlotNode(Node):
    def __init__(self):
        super().__init__("wheel_compare_plot_node")

        # ----------------------------------------------------
        # Subscribe to REAL ROBOT drive status
        # ----------------------------------------------------
        self.subscription_drive = self.create_subscription(
            Int32MultiArray,
            "/drive_full_status",
            self.drive_callback,
            10
        )

        # ----------------------------------------------------
        # Subscribe to SIMULATION joint states
        # ----------------------------------------------------
        self.subscription_joint = self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10
        )

        # ----------------------------------------------------
        # Publishers (real robot target → sim cmd_vel)
        # ----------------------------------------------------
        wheel_topics = [
            "/wheelset_left_right_wheel_joint/cmd_vel",
            "/wheelset_left_left_wheel_joint/cmd_vel",
            "/wheelset_rear_right_left_wheel_joint/cmd_vel",
            "/wheelset_rear_right_right_wheel_joint/cmd_vel",
            "/wheelset_rear_left_left_wheel_joint/cmd_vel",
            "/wheelset_rear_left_right_wheel_joint/cmd_vel",
            "/wheelset_right_right_wheel_joint/cmd_vel",
            "/wheelset_right_left_wheel_joint/cmd_vel",
        ]
        self.wheel_publishers = [
            self.create_publisher(Float64, t, 10) for t in wheel_topics
        ]

        # ----------------------------------------------------
        # Joint mapping (simulation)
        # ----------------------------------------------------
        self.joint_map = [
            "wheelset_left_right_wheel_joint",       # 0 wheel 1
            "wheelset_left_left_wheel_joint",        # 1 wheel 2
            "wheelset_rear_right_left_wheel_joint",  # 2 wheel 3
            "wheelset_rear_right_right_wheel_joint", # 3 wheel 4
            "wheelset_rear_left_left_wheel_joint",   # 4 wheel 5
            "wheelset_rear_left_right_wheel_joint",  # 5 wheel 6
            "wheelset_right_right_wheel_joint",      # 6 wheel 7
            "wheelset_right_left_wheel_joint",       # 7 wheel 8
        ]

        # ----------------------------------------------------
        # Constants
        # ----------------------------------------------------
        self.gear_ratio = 2.0
        self.radius = 0.035  # wheel radius (m)

        # ----------------------------------------------------
        # Live plot setup
        # ----------------------------------------------------
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 2, figsize=(12, 9), sharex=True)

        labels = ["W1","W2","W3","W4","W5","W6","W7","W8"]
        self.time_history = []
        self.target_history = [[] for _ in range(8)]
        self.sim_history = [[] for _ in range(8)]
        self.real_actual_history = [[] for _ in range(8)]

        self.target_lines = []
        self.sim_lines = []
        self.real_actual_lines = []

        for ax, label in zip(self.axs.flatten(), labels):
            s_line, = ax.plot([], [], 'b-', markersize=2, label=f"{label} sim")
            t_line, = ax.plot([], [], 'r--', markersize=2, label=f"{label} target")
            r_line, = ax.plot([], [], 'g-.', markersize=2, label=f"{label} real")

            ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax.set_ylabel("[rad/s]")
            ax.legend(loc="upper right", fontsize=7)

            self.sim_lines.append(s_line)
            self.target_lines.append(t_line)
            self.real_actual_lines.append(r_line)

        self.axs[-1, -1].set_xlabel("Time [s]")

        self.create_timer(0.1, self.update_plot)

        # Latest data storage
        self.start_time = time.time()
        self.latest_sim_vel = [0.0] * 8
        self.latest_target = [0.0] * 8
        self.latest_real_actual = [0.0] * 8

        self.get_logger().info("Compare real-target, real-actual, sim-actual node started!")

    # =======================================================
    # REAL ROBOT DATA CALLBACK
    # =======================================================
    def drive_callback(self, msg: Int32MultiArray):

        data = msg.data

        # --------------------------------------------------
        # REAL ROBOT ACTUAL VELOCITIES
        # --------------------------------------------------
        actual_mm = data[1:9]     # per-wheel actuals
        real_actual_rs = []

        for a in actual_mm:
            a_m_s = (a / 1000.0) / self.gear_ratio
            real_actual_rs.append(a_m_s / self.radius)

        # apply direction flip
        flip_indices = [0, 3, 5, 6]
        for i in flip_indices:
            real_actual_rs[i] = -real_actual_rs[i]

        self.latest_real_actual = real_actual_rs

        # --------------------------------------------------
        # REAL ROBOT TARGET VELOCITIES
        # --------------------------------------------------
        target_mm = data[9:17]

        target_rs = []
        for t in target_mm:
            t_m_s = (t / 1000.0) / self.gear_ratio
            target_rs.append(t_m_s / self.radius)

        # flip signs
        for i in flip_indices:
            target_rs[i] = -target_rs[i]

        # Only wheel 1 copied to all wheels
        t1 = target_rs[0]
        self.latest_target = [t1] * 8

        # Publish target to simulation
        for pub, val in zip(self.wheel_publishers, self.latest_target):
            msg_out = Float64()
            msg_out.data = float(val)
            pub.publish(msg_out)

    # =======================================================
    # SIMULATION joint velocity callback
    # =======================================================
    def joint_state_callback(self, msg: JointState):

        name_to_index = {name: i for i, name in enumerate(msg.name)}
        sim_vels = []

        for joint in self.joint_map:
            if joint in name_to_index:
                idx = name_to_index[joint]
                sim_vels.append(msg.velocity[idx])
            else:
                sim_vels.append(0.0)

        self.latest_sim_vel = sim_vels

    # =======================================================
    # PLOT UPDATE LOOP
    # =======================================================
    def update_plot(self):
        t = time.time() - self.start_time
        self.time_history.append(t)

        for i in range(8):
            self.target_history[i].append(self.latest_target[i])
            self.sim_history[i].append(self.latest_sim_vel[i])
            self.real_actual_history[i].append(self.latest_real_actual[i])

        for i in range(8):
            ax = self.axs.flatten()[i]
            self.sim_lines[i].set_data(self.time_history, self.sim_history[i])
            self.target_lines[i].set_data(self.time_history, self.target_history[i])
            self.real_actual_lines[i].set_data(self.time_history, self.real_actual_history[i])

            ax.relim()
            ax.autoscale_view()

        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = WheelComparePlotNode()
    rclpy.spin(node)
    node.destroy()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
