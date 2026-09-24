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

        # Publisher to a topic
        self.publisher = self.create_publisher(
            Int32MultiArray,
            "/drive_actual_velocities",
            10
        )

        # Create list of 8 OPC UA nodes
        base = "ns=6;s=::AsGlobalPV:Vehicle.Status.drive"
        self.nodes = [
            self.client.get_node(f"{base}{i}_actual")
            for i in range(1, 9)
        ]

        # Read at 10 Hz
        self.timer = self.create_timer(0.1, self.read_opcua)

    def read_opcua(self):
        try:
            values = [node.get_value() for node in self.nodes]

            # Log to terminal
            self.get_logger().info(f"Drive Actual Velocities: {values}")

            # Publish to ROS2
            msg = Int32MultiArray()
            msg.data = values
            self.publisher.publish(msg)

        except Exception as e:
            self.get_logger().error(f"OPC UA read error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = OpcUaReader()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
