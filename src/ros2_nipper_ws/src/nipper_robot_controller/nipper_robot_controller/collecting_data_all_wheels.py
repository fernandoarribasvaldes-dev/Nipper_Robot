import rclpy
from rclpy.node import Node
from opcua import Client
from std_msgs.msg import Int32MultiArray

class OpcUaReader(Node):
    def __init__(self):
        super().__init__('opcua_reader')

        opcua_url = "opc.tcp://10.1.144.123:4840"
        self.get_logger().info(f"Connecting to {opcua_url}")

        self.client = Client(opcua_url)
        self.client.connect()
        self.get_logger().info("OPC UA Connected!")

        self.publisher = self.create_publisher(Int32MultiArray,
                                               "/drive_full_status",
                                               10)

        base = "ns=6;s=::AsGlobalPV:Vehicle.Status.drive"

        # dinSafeSpeed node
        self.dinSafeSpeed_node = self.client.get_node(
            "ns=6;s=::AsGlobalPV:Vehicle.Status.dinSafeSpeed"
        )

        # All 8 actual and target nodes (these exist)
        self.actual_nodes =   [self.client.get_node(f"{base}{i}_actual")   for i in range(1, 9)]
        self.target_nodes =   [self.client.get_node(f"{base}{i}_target")   for i in range(1, 9)]

        # POSITION nodes exist only for odd drives (1,3,5,7)
        self.position_map = {
            1: self.client.get_node(f"{base}1_position"),
            3: self.client.get_node(f"{base}3_position"),
            5: self.client.get_node(f"{base}5_position"),
            7: self.client.get_node(f"{base}7_position")
        }

        # Timer at 10 Hz
        self.timer = self.create_timer(0.1, self.read_opcua)

    def read_opcua(self):
        try:
            din_safe_speed = self.dinSafeSpeed_node.get_value()

            actual   = [node.get_value() for node in self.actual_nodes]
            target   = [node.get_value() for node in self.target_nodes]

            # Build position array: missing = 0
            position = []
            for i in range(1, 9):
                if i in self.position_map:
                    position.append(self.position_map[i].get_value())
                else:
                    position.append(0)  # no position value

            combined = [din_safe_speed] + actual + target + position

            self.get_logger().info(f"dinSafeSpeed: {din_safe_speed}")
            self.get_logger().info(f"Actual:   {actual}")
            self.get_logger().info(f"Target:   {target}")
            self.get_logger().info(f"Position: {position}")

            msg = Int32MultiArray()
            msg.data = combined
            self.publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f"OPC UA read error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = OpcUaReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
