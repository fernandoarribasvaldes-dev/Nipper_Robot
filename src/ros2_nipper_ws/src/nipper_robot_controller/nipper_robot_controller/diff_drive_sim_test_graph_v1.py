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

        # Wheel name mapping (your order → joint_states names)
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

        # self.joint_map = [
        #     "wheelset_left_right_wheel_joint",        # 1 LF_R
        #     "wheelset_left_left_wheel_joint",        # 0 LF_L
        #     "wheelset_rear_right_left_wheel_joint",  # 4 RR_L
        #     "wheelset_rear_right_right_wheel_joint", # 5 RR_R
        #     "wheelset_rear_left_left_wheel_joint",   # 2 LR_L
        #     "wheelset_rear_left_right_wheel_joint",  # 3 LR_R
        #     "wheelset_right_right_wheel_joint",      # 7 RF_R 
        #     "wheelset_right_left_wheel_joint"        # 6 RF_L        
        # ]

        # ROS Subscriber
        self.subscription = self.create_subscription(
            JointState, "/joint_states", self.joint_callback, 10
        )

        # ---- MATPLOTLIB INITIALIZATION ----
        plt.ion()
        self.fig, self.axes = plt.subplots(8, 1, figsize=(10, 12), sharex=True)

        self.t_data = []
        self.act_data = [[] for _ in range(8)]
        self.ref_data = [[] for _ in range(8)]

        self.actual_lines = []
        self.reference_lines = []
        colors = plt.cm.tab10(np.linspace(0, 1, 8))

        for i in range(8):
            ax = self.axes[i]
            act_line, = ax.plot([], [], label=f"Actual {i}", color=colors[i])
            ref_line, = ax.plot([], [], "--", label=f"Ref {i}", color=colors[i])
            self.actual_lines.append(act_line)
            self.reference_lines.append(ref_line)
            ax.set_ylabel("Vel [rad/s]")
            ax.legend(loc="upper right")
            ax.set_title(f"Wheel {i}")

        self.axes[-1].set_xlabel("Time [s]")  # only bottom subplot gets x-label

        # Timer to update plot
        self.plot_timer = self.create_timer(0.1, self.update_plot)

        self.get_logger().info("Live plot running...")

    def joint_callback(self, msg):
        """Extract wheel velocities in correct order."""
        self.time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Read velocities
        for i, joint_name in enumerate(self.joint_map):
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                self.actual[i] = msg.velocity[idx]

        # Store for plotting
        self.t_data.append(self.time)
        for i in range(8):
            self.act_data[i].append(self.actual[i])
            self.ref_data[i].append(self.reference_speeds[i])

    def update_plot(self):
        """Refresh the matplotlib plot without blocking ROS."""
        if len(self.t_data) < 2:
            return  # nothing to plot yet

        for i in range(8):
            self.actual_lines[i].set_data(self.t_data, self.act_data[i])
            self.reference_lines[i].set_data(self.t_data, self.ref_data[i])
            self.axes[i].relim()
            self.axes[i].autoscale_view()

        plt.draw()
        plt.pause(0.001)


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
