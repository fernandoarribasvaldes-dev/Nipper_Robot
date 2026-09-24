#!/usr/bin/env python3
# Continuous steer-and-drive 3-turret controller + LIVE plots

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

    def step(self, error):
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            self.prev_error = error
            return 0.0

        dt = now - self.prev_time
        self.prev_time = now

        p = self.kp * error
        self.i_term += self.ki * error * dt
        if self.i_limit:
            self.i_term = max(min(self.i_term, self.i_limit), -self.i_limit)

        d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        return p + self.i_term + d


def angle_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ===========================================================
# 3-Turret Kinematics
# ===========================================================
class MultiTurretKinematics(Node):
    def __init__(self):
        super().__init__('multi_turret_kinematics')

        # ------------------------------------------
        # Subscriptions
        # ------------------------------------------
        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)

        # ------------------------------------------
        # Geometry (LF, RR, RF)
        # ------------------------------------------
        self.R_turrets = [
            ( 0.7, -0.2),   # LF
            (-0.7,  0.0),   # RR
            ( 0.7,  0.2),   # RF
        ]

        self.A = np.array([
            [1, 0, -self.R_turrets[0][1]],
            [0, 1, self.R_turrets[0][0]],
            [1, 0, -self.R_turrets[1][1]],
            [0, 1, self.R_turrets[1][0]],
            [1, 0, -self.R_turrets[2][1]],
            [0, 1, self.R_turrets[2][0]],
        ])

        self.r_wheel = 0.075
        self.d_wheel = 0.15

        # ------------------------------------------
        # Wheel publishers (6 wheels)
        # ------------------------------------------
        wheel_topics = [
            "/wheelset_left_left_wheel_joint/cmd_vel",
            "/wheelset_left_right_wheel_joint/cmd_vel",

            "/wheelset_rear_right_left_wheel_joint/cmd_vel",
            "/wheelset_rear_right_right_wheel_joint/cmd_vel",

            "/wheelset_right_left_wheel_joint/cmd_vel",
            "/wheelset_right_right_wheel_joint/cmd_vel",
        ]

        self.wheel_publishers = [
            self.create_publisher(Float64, t, 10) for t in wheel_topics
        ]

        # ------------------------------------------
        # State
        # ------------------------------------------
        self.turret_angles = [0.0, 0.0, 0.0]
        self.latest_cmd = Twist()

        self.heading_pid = [PID(10, 0.1, 5, i_limit=2.0) for _ in range(3)]
        # self.heading_pid = [PID(10, 0.0, 0.0, i_limit=0) for _ in range(3)]

        # ------------------------------------------
        # LIVE PLOT
        # ------------------------------------------
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        self.time_history = []
        self.e_phi_history = [[], [], []]
        self.phi_des_history = [[], [], []]
        self.phi_act_history = [[], [], []]

        labels = ["RF", "Rear", "LF"]
        colors = ["r", "b", "g"]

        self.lines_err = []
        self.lines_phi_des = []
        self.lines_phi_act = []

        for i in range(3):
            e, = self.axs[i].plot([], [], color=colors[i], label="e_phi (deg)")
            d, = self.axs[i].plot([], [], '--', color='blue', label="phi_des")
            a, = self.axs[i].plot([], [], color='green', label="phi_act")

            self.lines_err.append(e)
            self.lines_phi_des.append(d)
            self.lines_phi_act.append(a)

            self.axs[i].axhline(0, color='k', linestyle='--')
            self.axs[i].set_ylabel(labels[i])
            self.axs[i].legend()

        self.axs[-1].set_xlabel("Time [s]")

        # Timers
        self.create_timer(0.02, self.control_step)
        self.create_timer(0.1, self.update_plot)

        self.start_time = time.time()
        self.get_logger().info("3-Turret controller initialized")

    # ---------------------------------------------------------
    def joint_state_callback(self, msg):
        jp = dict(zip(msg.name, msg.position))
        self.turret_angles = [
            jp.get("wheelset_left_revolute_joint", 0.0),
            jp.get("wheelset_rear_right_revolute_joint", 0.0),
            jp.get("wheelset_right_revolute_joint", 0.0),
        ]

    def cmd_callback(self, msg):
        self.latest_cmd = msg

    # ---------------------------------------------------------
    def control_step(self):
        vx = self.latest_cmd.linear.x
        vy = self.latest_cmd.linear.y
        omega = self.latest_cmd.angular.z

        V = np.array([[vx], [vy], [omega]])
        v_turrets = np.dot(self.A, V).reshape((3, 2))

        t = time.time() - self.start_time
        self.time_history.append(t)

        wheel_cmds = []

        for i, (vx_i, vy_i) in enumerate(v_turrets):
            v_bar = math.hypot(vx_i, vy_i)
            raw_phi = math.atan2(vy_i, vx_i)

            self.get_logger().info(
                f"[{i}] vx_i={vx_i:+.4f}, vy_i={vy_i:+.4f}, "
                f"v_bar={v_bar:.4f}, raw_phi={math.degrees(raw_phi):+.2f} deg"
            )

            if abs(angle_wrap(raw_phi)) > math.pi / 2:
                phi_des = angle_wrap(raw_phi + math.pi)
                v_bar = -v_bar
            else:
                phi_des = raw_phi

            # self.get_logger().info(
            #     f"[{i}] vx_i={vx_i:+.4f}, vy_i={vy_i:+.4f}, "
            #     f"v_bar={v_bar:.4f}, phi_des={math.degrees(phi_des):+.2f} deg"
            # )

            phi_act = self.turret_angles[i]
            e_phi = angle_wrap(phi_des - phi_act)


            self.get_logger().info(
                f"[{i}] phi_act={math.degrees(phi_act):+.2f} deg, "
                f"e_phi={math.degrees(e_phi):+.2f} deg"
            )


            self.e_phi_history[i].append(math.degrees(e_phi))
            self.phi_des_history[i].append(math.degrees(phi_des))
            self.phi_act_history[i].append(math.degrees(phi_act))

            phi_dot = self.heading_pid[i].step(e_phi)

            v_l = v_bar - 0.5 * self.d_wheel * phi_dot
            v_r = v_bar + 0.5 * self.d_wheel * phi_dot

            wheel_cmds.extend([
                v_l / self.r_wheel,
                v_r / self.r_wheel
            ])

        self.publish_wheels(wheel_cmds)

    # ---------------------------------------------------------
    def update_plot(self):
        if len(self.time_history) < 2:
            return

        for i in range(3):
            self.lines_err[i].set_data(self.time_history, self.e_phi_history[i])
            self.lines_phi_des[i].set_data(self.time_history, self.phi_des_history[i])
            self.lines_phi_act[i].set_data(self.time_history, self.phi_act_history[i])

            self.axs[i].relim()
            self.axs[i].autoscale_view()

        plt.draw()
        plt.pause(0.001)

    # ---------------------------------------------------------
    def publish_wheels(self, cmds):
        for pub, val in zip(self.wheel_publishers, cmds):
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
#!/usr/bin/env python3
# Continuous steer-and-drive 3-turret controller + LIVE plots

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

    def step(self, error):
        now = time.time()
        if self.prev_time is None:
            self.prev_time = now
            self.prev_error = error
            return 0.0

        dt = now - self.prev_time
        self.prev_time = now

        p = self.kp * error
        self.i_term += self.ki * error * dt
        if self.i_limit:
            self.i_term = max(min(self.i_term, self.i_limit), -self.i_limit)

        d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        return p + self.i_term + d


def angle_wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ===========================================================
# 3-Turret Kinematics
# ===========================================================
class MultiTurretKinematics(Node):
    def __init__(self):
        super().__init__('multi_turret_kinematics')

        # ------------------------------------------
        # Subscriptions
        # ------------------------------------------
        self.create_subscription(Twist, "/cmd_vel", self.cmd_callback, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 10)

        # ------------------------------------------
        # Geometry (RF, Rear , LF)
        # ------------------------------------------
        self.R_turrets = [
            ( 0.7, -0.2),   # RF
            (-0.7,  0.0),   # Rear 
            ( 0.7,  0.2),   # LF
        ]

        self.A = np.array([
            [1, 0, -self.R_turrets[0][1]],
            [0, 1, self.R_turrets[0][0]],
            [1, 0, -self.R_turrets[1][1]],
            [0, 1, self.R_turrets[1][0]],
            [1, 0, -self.R_turrets[2][1]],
            [0, 1, self.R_turrets[2][0]],
        ])

        self.r_wheel = 0.075
        self.d_wheel = 0.15

        # ------------------------------------------
        # Wheel publishers (6 wheels)
        # ------------------------------------------
        wheel_topics = [
            "/wheelset_left_left_wheel_joint/cmd_vel",
            "/wheelset_left_right_wheel_joint/cmd_vel",

            "/wheelset_rear_right_left_wheel_joint/cmd_vel",
            "/wheelset_rear_right_right_wheel_joint/cmd_vel",

            "/wheelset_right_left_wheel_joint/cmd_vel",
            "/wheelset_right_right_wheel_joint/cmd_vel",
        ]

        self.wheel_publishers = [
            self.create_publisher(Float64, t, 10) for t in wheel_topics
        ]

        # ------------------------------------------
        # State
        # ------------------------------------------
        self.turret_angles = [0.0, 0.0, 0.0]
        self.latest_cmd = Twist()

        self.heading_pid = [PID(10, 0.1, 5, i_limit=2.0) for _ in range(3)]
        # self.heading_pid = [PID(5, 0.0, 0.0, i_limit=0) for _ in range(3)]

        # ------------------------------------------
        # LIVE PLOT
        # ------------------------------------------
        plt.ion()
        self.fig, self.axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

        self.time_history = []
        self.e_phi_history = [[], [], []]
        self.phi_des_history = [[], [], []]
        self.phi_act_history = [[], [], []]

        labels = ["RF", "Rear", "LF"]
        colors = ["r", "b", "g"]

        self.lines_err = []
        self.lines_phi_des = []
        self.lines_phi_act = []

        for i in range(3):
            e, = self.axs[i].plot([], [], color=colors[i], label="e_phi (deg)")
            d, = self.axs[i].plot([], [], '--', color='blue', label="phi_des")
            a, = self.axs[i].plot([], [], color='green', label="phi_act")

            self.lines_err.append(e)
            self.lines_phi_des.append(d)
            self.lines_phi_act.append(a)

            self.axs[i].axhline(0, color='k', linestyle='--')
            self.axs[i].set_ylabel(labels[i])
            self.axs[i].legend()

        self.axs[-1].set_xlabel("Time [s]")

        # Timers
        self.create_timer(0.02, self.control_step)
        self.create_timer(0.1, self.update_plot)

        self.start_time = time.time()
        self.get_logger().info("3-Turret controller initialized")

    # ---------------------------------------------------------
    def joint_state_callback(self, msg):
        jp = dict(zip(msg.name, msg.position))
        self.turret_angles = [
            jp.get("wheelset_left_revolute_joint", 0.0),
            jp.get("wheelset_rear_right_revolute_joint", 0.0),
            jp.get("wheelset_right_revolute_joint", 0.0),
        ]

    def cmd_callback(self, msg):
        self.latest_cmd = msg

    # ---------------------------------------------------------
    def control_step(self):
        vx = self.latest_cmd.linear.x
        vy = self.latest_cmd.linear.y
        omega = self.latest_cmd.angular.z

        #I think due to the equations z and y are commands by the user should be fliped.
        # vx = self.latest_cmd.linear.x
        # vy = self.latest_cmd.angular.z 
        # omega = self.latest_cmd.linear.y

        V = np.array([[vx], [vy], [omega]])
        v_turrets = np.dot(self.A, V).reshape((3, 2))

        t = time.time() - self.start_time
        self.time_history.append(t)

        wheel_cmds = []

        for i, (vx_i, vy_i) in enumerate(v_turrets):
            v_bar = math.hypot(vx_i, vy_i)
            raw_phi = math.atan2(vy_i, vx_i)

            self.get_logger().info(
                f"[{i}] vx_i={vx_i:+.4f}, vy_i={vy_i:+.4f}, "
                f"v_bar={v_bar:.4f}, raw_phi={math.degrees(raw_phi):+.2f} deg"
            )

            if abs(angle_wrap(raw_phi)) > math.pi / 2:
                phi_des = angle_wrap(raw_phi + math.pi)
                v_bar = -v_bar
            else:
                phi_des = raw_phi

            self.get_logger().info(
                f"[{i}] vx_i={vx_i:+.4f}, vy_i={vy_i:+.4f}, "
                f"v_bar={v_bar:.4f}, phi_des={math.degrees(phi_des):+.2f} deg"
            )

            # phi_act = self.turret_angles[i]
            # RR turret has inverted encoder sign
            if i == 1:   # RR
                phi_act = -self.turret_angles[i]
            else:
                phi_act = self.turret_angles[i]

            e_phi = angle_wrap(phi_des - phi_act)


            # self.get_logger().info(
            #     f"[{i}] phi_act={math.degrees(phi_act):+.2f} deg, "
            #     f"e_phi={math.degrees(e_phi):+.2f} deg"
            # )


            self.e_phi_history[i].append(math.degrees(e_phi))
            self.phi_des_history[i].append(math.degrees(phi_des))
            self.phi_act_history[i].append(math.degrees(phi_act))

            phi_dot = self.heading_pid[i].step(e_phi)

            v_l = v_bar - 0.5 * self.d_wheel * phi_dot
            v_r = v_bar + 0.5 * self.d_wheel * phi_dot

            wheel_cmds.extend([
                v_l / self.r_wheel,
                v_r / self.r_wheel
            ])

        self.publish_wheels(wheel_cmds)

    # ---------------------------------------------------------
    def update_plot(self):
        if len(self.time_history) < 2:
            return

        for i in range(3):
            self.lines_err[i].set_data(self.time_history, self.e_phi_history[i])
            self.lines_phi_des[i].set_data(self.time_history, self.phi_des_history[i])
            self.lines_phi_act[i].set_data(self.time_history, self.phi_act_history[i])

            self.axs[i].relim()
            self.axs[i].autoscale_view()

        plt.draw()
        plt.pause(0.001)

    # ---------------------------------------------------------
    def publish_wheels(self, cmds):
        for pub, val in zip(self.wheel_publishers, cmds):
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
