#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Float64

import matplotlib.pyplot as plt
import time

class WheelVelocityPlotNode(Node):
    def __init__(self):
        super().__init__("wheel_velocity_plot_node")

        # ----------------------------
        # Subscription to raw PLC data
        # ----------------------------
        self.subscription = self.create_subscription(
            Int32MultiArray,
            "/drive_actual_velocities",
            self.listener_callback,
            10
        )

        # ----------------------------
        # Wheel publishers (rad/s)
        # ----------------------------
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

        # Conversion constants
        self.gear_ratio = 2.0
        self.radius = 0.075

        # ----------------------------
        # LIVE PLOT INIT (like your turret plot)
        # ----------------------------
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 2, figsize=(10, 8), sharex=True)

        labels = [
            "Wheel 1", "Wheel 2", "Wheel 3", "Wheel 4",
            "Wheel 5", "Wheel 6", "Wheel 7", "Wheel 8"
        ]

        self.time_history = []
        self.wheel_history = [[] for _ in range(8)]

        self.lines = []
        for ax, lab in zip(self.axs.flatten(), labels):
            line, = ax.plot([], [], label=lab)
            ax.set_ylabel(f"{lab} [rad/s]")
            ax.axhline(0, color='k', linestyle='--', linewidth=0.7)
            ax.legend(loc="upper right")
            self.lines.append(line)

        self.axs[-1, -1].set_xlabel("Time [s]")

        # Timer for updating plot
        self.create_timer(0.1, self.update_plot)

        self.start_time = time.time()

        self.get_logger().info("Wheel velocity processor + live plot started!")

    # =======================================================
    def listener_callback(self, msg: Int32MultiArray):
        raw_values = msg.data

        if len(raw_values) != 8:
            self.get_logger().error("Expected 8 values but received something else!")
            return

        # ----------------------------
        # PROCESS → rad/s
        # ----------------------------
        processed = []
        for v_mm_s in raw_values:
            v_m_s = v_mm_s / 1000.0           # mm/s → m/s
            v_m_s /= self.gear_ratio         # gear reduction
            rad_s = v_m_s / self.radius      # m/s → rad/s
            processed.append(rad_s)

        # Flip specific wheels
        flip_indices = [0, 3, 5, 6]
        for i in flip_indices:
            processed[i] = -processed[i]

        # ----------------------------
        # PUBLISH → wheel joint topics
        # ----------------------------
        for pub, val in zip(self.wheel_publishers, processed):
            msg_out = Float64()
            msg_out.data = float(val)
            pub.publish(msg_out)

        # ----------------------------
        # LOG
        # ----------------------------
        self.get_logger().info(f"Wheel speeds (rad/s): {processed}")

        # ----------------------------
        # STORE FOR PLOTTING
        # ----------------------------
        t = time.time() - self.start_time
        self.time_history.append(t)

        for i in range(8):
            self.wheel_history[i].append(processed[i])

    # =======================================================
    def update_plot(self):
        if len(self.time_history) < 2:
            return

        for i in range(8):
            self.lines[i].set_data(self.time_history, self.wheel_history[i])
            ax = self.axs.flatten()[i]
            ax.relim()
            ax.autoscale_view()

        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.001)


# ===========================================================
def main(args=None):
    rclpy.init(args=args)
    node = WheelVelocityPlotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
