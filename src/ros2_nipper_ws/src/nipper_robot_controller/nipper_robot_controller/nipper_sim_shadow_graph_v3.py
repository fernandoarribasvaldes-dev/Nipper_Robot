#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Float64

import matplotlib.pyplot as plt
import time


class WheelVelocityPlotNode(Node):
    def __init__(self):
        super().__init__("wheel_velocity_plot_node")

        # ----------------------------------------------------
        # Subscribe to full drive status
        # ----------------------------------------------------
        self.subscription = self.create_subscription(
            Int32MultiArray,
            "/drive_full_status",
            self.listener_callback,
            10
        )

        # ----------------------------------------------------
        # Wheel publishers
        # ----------------------------------------------------
        wheel_topics = [
            "/wheelset_left_right_wheel_joint/cmd_vel",      # 1
            "/wheelset_left_left_wheel_joint/cmd_vel",       # 2
            "/wheelset_rear_right_left_wheel_joint/cmd_vel", # 3
            "/wheelset_rear_right_right_wheel_joint/cmd_vel",# 4
            "/wheelset_rear_left_left_wheel_joint/cmd_vel",  # 5
            "/wheelset_rear_left_right_wheel_joint/cmd_vel", # 6
            "/wheelset_right_right_wheel_joint/cmd_vel",     # 7
            "/wheelset_right_left_wheel_joint/cmd_vel",      # 8
        ]

        self.wheel_publishers = [
            self.create_publisher(Float64, topic, 10)
            for topic in wheel_topics
        ]

        # ----------------------------------------------------
        # Conversion constants
        # ----------------------------------------------------
        self.gear_ratio = 2.0
        # self.gear_ratio = 3.0
        self.radius = (0.07/2)  # wheel radius (m)

        # ----------------------------------------------------
        # Live plot setup
        # ----------------------------------------------------
        plt.ion()
        self.fig, self.axs = plt.subplots(4, 2, figsize=(12, 9), sharex=True)

        labels = [
            "Wheel 1", "Wheel 2", "Wheel 3", "Wheel 4",
            "Wheel 5", "Wheel 6", "Wheel 7", "Wheel 8"
        ]

        self.time_history = []
        self.actual_history = [[] for _ in range(8)]
        self.target_history = [[] for _ in range(8)]

        self.actual_lines = []
        self.target_lines = []

        for ax, label in zip(self.axs.flatten(), labels):
            actual_line, = ax.plot([], [], 'bo-', markersize=3, label=f"{label} Actual")
            target_line, = ax.plot([], [], 'ro-', markersize=3, label=f"{label} Target")

            ax.axhline(0, color='k', linestyle='--', linewidth=0.7)
            ax.set_ylabel("[rad/s]")
            ax.legend(loc="upper right", fontsize=7)

            self.actual_lines.append(actual_line)
            self.target_lines.append(target_line)

        self.axs[-1, -1].set_xlabel("Time [s]")

        # ----------------------------------------------------
        # Dropout correction memory
        # ----------------------------------------------------
        self.prev_actual = [0.0] * 8
        self.prev_target = [0.0] * 8
        self.zero_streak_actual = [0] * 8
        self.zero_streak_target = [0] * 8
        self.zero_streak_threshold = 3  # samples allowed (10 Hz → 300ms)

        # ----------------------------------------------------
        # Warm-up period (ignore dropout correction)
        # ----------------------------------------------------
        self.start_time = time.time()
        self.warmup_duration = 0.3  # seconds

        # Plot refresh timer
        self.create_timer(0.1, self.update_plot)

        self.get_logger().info("Wheel velocity tracking node started!")


    # =======================================================
    def listener_callback(self, msg: Int32MultiArray):

        data = msg.data

        # ----------------------------------------------------
        # Extract actual + target velocities
        # ----------------------------------------------------
        actual_mm = data[1:9]
        target_mm = data[9:17]

        actual_rs = []
        target_rs = []

        # Convert mm/s → rad/s
        for a, t in zip(actual_mm, target_mm):
            a_m_s = (a / 1000.0) / self.gear_ratio
            t_m_s = (t / 1000.0) / self.gear_ratio
            actual_rs.append(a_m_s / self.radius)
            target_rs.append(t_m_s / self.radius)

        # ----------------------------------------------------
        # Sign flipping (motor direction)
        # ----------------------------------------------------
        flip_indices = [0, 3, 5, 6]
        for i in flip_indices:
            actual_rs[i] = -actual_rs[i]
            target_rs[i] = -target_rs[i]

        # ----------------------------------------------------
        # WARM-UP PHASE — no dropout correction
        # ----------------------------------------------------
        elapsed = time.time() - self.start_time
        if elapsed < self.warmup_duration:

            self.prev_actual = actual_rs[:]
            self.prev_target = target_rs[:]

            self.zero_streak_actual = [0]*8
            self.zero_streak_target = [0]*8

            self.time_history.append(elapsed)
            for i in range(8):
                self.actual_history[i].append(actual_rs[i])
                self.target_history[i].append(target_rs[i])

            # Publish TARGET during warm-up
            for pub, val in zip(self.wheel_publishers, target_rs):
                msg_out = Float64()
                msg_out.data = float(val)
                pub.publish(msg_out)

            return


        # ----------------------------------------------------
        # SMART DROPOUT CORRECTION (actual + target)
        # ----------------------------------------------------
        corrected_actual = []
        corrected_target = []

        for i in range(8):

            # ---- Actual dropout fix ----
            a = actual_rs[i]
            if abs(a) < 0.0001:
                self.zero_streak_actual[i] += 1
                if (
                    self.zero_streak_actual[i] <= self.zero_streak_threshold
                    and abs(self.prev_actual[i]) > 0.0001
                ):
                    a = self.prev_actual[i]
                else:
                    a = 0.0
            else:
                self.zero_streak_actual[i] = 0
            corrected_actual.append(a)

            # ---- Target dropout fix ----
            t = target_rs[i]
            if abs(t) < 0.0001:
                self.zero_streak_target[i] += 1
                if (
                    self.zero_streak_target[i] <= self.zero_streak_threshold
                    and abs(self.prev_target[i]) > 0.0001
                ):
                    t = self.prev_target[i]
                else:
                    t = 0.0
            else:
                self.zero_streak_target[i] = 0
            corrected_target.append(t)

        # Save corrected values
        actual_rs = corrected_actual
        target_rs = corrected_target

        self.prev_actual = actual_rs[:]
        self.prev_target = target_rs[:]

        # ----------------------------------------------------
        # **OVERWRITE TARGET VALUES USING PAIR MAPPING**
        # W1 → wheels [1,3,5,7]
        # W2 → wheels [2,4,6,8]
        # ----------------------------------------------------
        t1 = target_rs[0]   # W1_target
        t2 = target_rs[1]   # W2_target

        target_rs = [
            t1,  # wheel 1
            t2,  # wheel 2
            t1,  # wheel 3
            t2,  # wheel 4
            t1,  # wheel 5
            t2,  # wheel 6
            t1,  # wheel 7
            t2,  # wheel 8
        ]

        # ----------------------------------------------------
        # Publish corrected TARGET commands
        # ----------------------------------------------------
        for pub, val in zip(self.wheel_publishers, target_rs):
            msg_out = Float64()
            msg_out.data = float(val)
            pub.publish(msg_out)

        # ----------------------------------------------------
        # Store for plotting
        # ----------------------------------------------------
        t_now = time.time() - self.start_time
        self.time_history.append(t_now)

        for i in range(8):
            self.actual_history[i].append(actual_rs[i])
            self.target_history[i].append(target_rs[i])

        # Logging
        self.get_logger().info(f"Actual: {actual_rs}")
        self.get_logger().info(f"Target: {target_rs}")


    # =======================================================
    def update_plot(self):
        if len(self.time_history) < 2:
            return

        for i in range(8):
            ax = self.axs.flatten()[i]

            # Plot target only (as per your last request)
            self.target_lines[i].set_data(self.time_history, self.target_history[i])

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
    node.destroy()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
