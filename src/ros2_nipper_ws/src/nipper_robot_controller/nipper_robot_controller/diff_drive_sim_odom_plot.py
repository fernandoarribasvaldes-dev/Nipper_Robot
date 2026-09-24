#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np
import math


class OdometryLivePlot(Node):
    def __init__(self):
        super().__init__("odometry_live_plot")

        # -------------------------------
        # Matplotlib Setup – 6 subplots
        # -------------------------------
        plt.ion()
        self.fig, self.axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

        self.t_data = []

        # position
        self.x_data = []
        self.y_data = []
        self.yaw_data = []

        # velocities
        self.vx_data = []
        self.vy_data = []
        self.omega_data = []

        # Create empty line handles
        self.lines = []
        labels = ["x [m]", "y [m]", "yaw [rad]", "vx [m/s]", "vy [m/s]", "omega [rad/s]"]

        # for i in range(6):
        #     line, = self.axes[i].plot([], [], label=labels[i])
        #     self.lines.append(line)
        #     self.axes[i].set_ylabel(labels[i])
        #     self.axes[i].legend(loc="upper right")

        # Distinct colors for each subplot
        colors = plt.cm.tab10(np.linspace(0, 1, 6))

        for i in range(6):
            line, = self.axes[i].plot([], [], label=labels[i], color=colors[i])
            self.lines.append(line)
            self.axes[i].set_ylabel(labels[i])
            self.axes[i].legend(loc="upper right")


        self.axes[-1].set_xlabel("Time [s]")

        # ROS subscriber
        self.sub = self.create_subscription(
            Odometry, "/odometry", self.odom_callback, 10
        )

        # update timer (10 Hz)
        self.timer = self.create_timer(0.1, self.update_plot)

        self.get_logger().info("Live odometry plot started...")

    # ---------------------------
    # Odometry Callback
    # ---------------------------
    def odom_callback(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.t_data.append(t)

        # Position
        self.x_data.append(msg.pose.pose.position.x)
        self.y_data.append(msg.pose.pose.position.y)

        # Orientation → yaw
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y*q.y + q.z*q.z)
        )
        self.yaw_data.append(yaw)

        # Velocities
        self.vx_data.append(msg.twist.twist.linear.x)
        self.vy_data.append(msg.twist.twist.linear.y)
        self.omega_data.append(msg.twist.twist.angular.z)

    # ---------------------------
    # Update plot
    # ---------------------------
    def update_plot(self):
        if len(self.t_data) < 2:
            return

        data_list = [
            self.x_data,
            self.y_data,
            self.yaw_data,
            self.vx_data,
            self.vy_data,
            self.omega_data
        ]

        for i in range(6):
            self.lines[i].set_data(self.t_data, data_list[i])
            self.axes[i].relim()
            self.axes[i].autoscale_view()

        plt.draw()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)

    node = OdometryLivePlot()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
