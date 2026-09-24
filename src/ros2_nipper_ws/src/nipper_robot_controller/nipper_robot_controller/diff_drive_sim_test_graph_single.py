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

        # Index of wheel to plot (v1 = wheel index 1)
        self.wheel_index = 7

        # Wheel name mapping (your order → joint_states names)
        self.joint_map = [
            "wheelset_left_left_wheel_joint",        # 0 LF_L
            "wheelset_left_right_wheel_joint",       # 1 LF_R (v1)
            "wheelset_rear_left_left_wheel_joint",   # 2 LR_L
            "wheelset_rear_left_right_wheel_joint",  # 3 LR_R
            "wheelset_rear_right_left_wheel_joint",  # 4 RR_L
            "wheelset_rear_right_right_wheel_joint", # 5 RR_R
            "wheelset_right_left_wheel_joint",       # 6 RF_L
            "wheelset_right_right_wheel_joint"       # 7 RF_R
        ]

        # Subscriber
        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.joint_callback, 10
        )

        # ---- MATPLOTLIB INITIALIZATION ----
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

        self.t_data = []
        self.act_data = [[] for _ in range(8)]
        self.ref_data = [[] for _ in range(8)]

        # Only ONE line for actual + ONE for reference
        color = "tab:red"
        self.actual_line, = self.ax.plot([], [], label=f"Actual v1", color=color)
        self.ref_line, = self.ax.plot([], [], "--", label=f"Ref v1", color=color)

        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Velocity [rad/s]")
        self.ax.legend(loc="upper right")
        self.ax.set_title("Live Velocity of Wheel v1")

        # Timer to update plot
        self.plot_timer = self.create_timer(0.1, self.update_plot)

        self.get_logger().info("Live plot for wheel v1 running...")

    def joint_callback(self, msg):
        """Extract wheel velocities in correct order."""
        self.time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Read velocities
        for i, joint_name in enumerate(self.joint_map):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.actual[i] = msg.velocity[idx]

        # Store for SINGLE wheel v1
        i = self.wheel_index
        self.t_data.append(self.time)
        self.act_data[i].append(self.actual[i])
        self.ref_data[i].append(self.reference_speeds[i])

    def update_plot(self):
        """Refresh the matplotlib plot without blocking ROS."""
        if len(self.t_data) < 2:
            return  # nothing to plot yet

        i = self.wheel_index

        # Update just v1
        self.actual_line.set_data(self.t_data, self.act_data[i])
        self.ref_line.set_data(self.t_data, self.ref_data[i])

        # Adjust axes dynamically
        self.ax.relim()
        self.ax.autoscale_view()

        plt.draw()
        plt.pause(0.001)  # Non-blocking update

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
