#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

import matplotlib.pyplot as plt
from collections import deque

class WheelVelocityPlot(Node):
    def __init__(self):
        super().__init__('wheel_velocity_plot')

        # Parameters
        self.max_points = 100  # number of points to show on graph

        # Rear left wheel indices in joint_states
        self.left_index = None
        self.right_index = None

        # Data buffers
        self.time_buffer = deque(maxlen=self.max_points)
        self.desired_left = deque(maxlen=self.max_points)
        self.desired_right = deque(maxlen=self.max_points)
        self.actual_left = deque(maxlen=self.max_points)
        self.actual_right = deque(maxlen=self.max_points)

        self.current_time = 0

        # Subscribers
        self.create_subscription(Float64,
                                 '/wheelset_rear_left_left_wheel_joint/cmd_vel',
                                 self.desired_left_callback, 10)
        self.create_subscription(Float64,
                                 '/wheelset_rear_left_right_wheel_joint/cmd_vel',
                                 self.desired_right_callback, 10)
        self.create_subscription(JointState,
                                 '/joint_states',
                                 self.joint_states_callback, 10)

        # Setup matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.left_line_desired, = self.ax.plot([], [], 'b-', label='Left Desired')
        self.left_line_actual, = self.ax.plot([], [], 'b--', label='Left Actual')
        self.right_line_desired, = self.ax.plot([], [], 'r-', label='Right Desired')
        self.right_line_actual, = self.ax.plot([], [], 'r--', label='Right Actual')
        self.ax.set_xlabel('Time step')
        self.ax.set_ylabel('Velocity (rad/s)')
        self.ax.legend()
        self.ax.set_title('Rear Left Wheel Velocities')

        # Timer to update plot
        # self.create_timer(0.1, self.update_plot)  # 10 Hz

    # Callbacks
    def desired_left_callback(self, msg):
        self.desired_left.append(msg.data)
        self.time_buffer.append(self.current_time)
        self.current_time += 1
        self.get_logger().info(f"Desired Left: {msg.data:.3f}")

    def desired_right_callback(self, msg):
        self.desired_right.append(msg.data)
        self.get_logger().info(f"Desired Right: {msg.data:.3f}")

    def joint_states_callback(self, msg):
        # Find indices of rear left wheels if not already found
        if self.left_index is None or self.right_index is None:
            try:
                self.left_index = msg.name.index('wheelset_rear_left_left_wheel_joint')
                self.right_index = msg.name.index('wheelset_rear_left_right_wheel_joint')
            except ValueError:
                return

        actual_left_vel = msg.velocity[self.left_index]
        actual_right_vel = msg.velocity[self.right_index]

        self.actual_left.append(actual_left_vel)
        self.actual_right.append(actual_right_vel)

        self.get_logger().info(
            f"Actual Left: {actual_left_vel:.3f}, Actual Right: {actual_right_vel:.3f}"
        )

    # Update matplotlib
    def update_plot(self):
        # Don't update if we have no data yet
        if len(self.time_buffer) == 0:
            return

        # Plot rear left wheel
        self.left_line_desired.set_data(self.time_buffer, list(self.desired_left))
        self.left_line_actual.set_data(self.time_buffer, list(self.actual_left))
        self.right_line_desired.set_data(self.time_buffer, list(self.desired_right))
        self.right_line_actual.set_data(self.time_buffer, list(self.actual_right))

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def main(args=None):
    rclpy.init(args=args)
    node = WheelVelocityPlot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
