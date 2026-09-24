#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
import numpy as np


class WheelLivePlot(Node):
    def __init__(self, reference_speeds):
        super().__init__("wheel_live_plot")

        self.reference_speeds = reference_speeds  # length 8
        self.actual = [0.0] * 8
        self.time = 0.0

        # ---------------------------------------
        # Wheel joint order (8 wheels)
        # ---------------------------------------
        self.joint_map = [
            "wheelset_left_left_wheel_joint",        # 0 LF_L
            "wheelset_left_right_wheel_joint",       # 1 LF_R
            "wheelset_rear_left_left_wheel_joint",   # 2 LR_L
            "wheelset_rear_left_right_wheel_joint",  # 3 LR_R
            "wheelset_rear_right_left_wheel_joint",  # 4 RR_L
            "wheelset_rear_right_right_wheel_joint", # 5 RR_R
            "wheelset_right_left_wheel_joint",       # 6 RF_L
            "wheelset_right_right_wheel_joint"       # 7 RF_R
        ]

        # ---------------------------------------
        # Turret joint order (4 turret angles)
        # ---------------------------------------
        # self.turret_map = [
        #     "wheelset_left_revolute_joint",        # LF turret
        #     "wheelset_rear_left_revolute_joint",   # LR turret
        #     "wheelset_rear_right_revolute_joint",  # RR turret
        #     "wheelset_right_revolute_joint",       # RF turret
        # ]

        # ROS Subscriber
        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.joint_callback, 10
        )

        # ---------------------------------------
        # MATPLOTLIB SETUP: 12 subplots
        # ---------------------------------------
        plt.ion()
        self.fig, self.axes = plt.subplots(8, 1, figsize=(10, 18), sharex=True)

        self.t_data = []
        self.act_data = [[] for _ in range(8)]
        self.ref_data = [[] for _ in range(8)]

        # self.turret_act = [[] for _ in range(4)]

        self.actual_lines = []
        self.reference_lines = []
        # self.turret_lines = []

        colors = plt.cm.tab10(np.linspace(0, 1, 8))
        # turret_colors = plt.cm.Set2(np.linspace(0, 1, 4))

        # ---- Wheel plots (8) ----
        for i in range(8):
            ax = self.axes[i]
            act_line, = ax.plot([], [], label=f"Actual {i}", color=colors[i])
            ref_line, = ax.plot([], [], "--", label=f"Ref {i}", color=colors[i])
            self.actual_lines.append(act_line)
            self.reference_lines.append(ref_line)
            ax.set_ylabel("Vel [rad/s]")
            ax.legend(loc="upper right")
            ax.set_title(f"Wheel {i}")

        # ---- Turret angle plots (4) ----
        # for i in range(4):
        #     ax = self.axes[8 + i]
        #     t_line, = ax.plot([], [], label=f"Turret {i} angle", color=turret_colors[i])
        #     self.turret_lines.append(t_line)
        #     ax.set_ylabel("Angle [rad]")
        #     ax.legend(loc="upper right")
        #     ax.set_title(f"Turret {i}")

        # Time axis label on bottom subplot
        self.axes[-1].set_xlabel("Time [s]")

        # Timer for plot updates
        self.plot_timer = self.create_timer(0.1, self.update_plot)

        self.get_logger().info("Live wheel + turret plot running...")

    # ---------------------------------------
    # Joint State Callback
    # ---------------------------------------
    def joint_callback(self, msg):
        self.time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.t_data.append(self.time)

        # Read wheel velocities
        for i, joint_name in enumerate(self.joint_map):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.actual[i] = msg.velocity[idx]
                self.act_data[i].append(self.actual[i])
                self.ref_data[i].append(self.reference_speeds[i])

        # Read turret angles
        # for i, joint_name in enumerate(self.turret_map):
        #     if joint_name in msg.name:
        #         idx = msg.name.index(joint_name)
        #         angle = msg.position[idx]
        #         self.turret_act[i].append(angle)

    # ---------------------------------------
    # Plot Update
    # ---------------------------------------
    def update_plot(self):
        if len(self.t_data) < 2:
            return

        # Update wheel plots
        for i in range(8):
            self.actual_lines[i].set_data(self.t_data, self.act_data[i])
            self.reference_lines[i].set_data(self.t_data, self.ref_data[i])
            self.axes[i].relim()
            self.axes[i].autoscale_view()

        # Update turret plots
        # for i in range(4):
        #     self.turret_lines[i].set_data(self.t_data, self.turret_act[i])
        #     self.axes[8 + i].relim()
        #     self.axes[8 + i].autoscale_view()

        plt.draw()
        plt.pause(0.001)


# ---------------------------------------
# Main
# ---------------------------------------
def main(args=None):
    rclpy.init(args=args)

    reference_speeds = [
        3.0, 3.0,
        3.0, 3.0,
        3.0, 3.0,
        3.0, 3.0
    ]

    node = WheelLivePlot(reference_speeds)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
