import rclpy
from rclpy.node import Node
from opcua import Client

class OpcUaReader(Node):
    def __init__(self):
        super().__init__('opcua_reader')

        opcua_url = "opc.tcp://10.1.144.123:4840"
        # opcua_url = "opc.tcp://192.168.8.102:4840"
        self.get_logger().info(f"Connecting to {opcua_url}")

        self.client = Client(opcua_url)
        self.client.connect()
        self.get_logger().info("OPC UA Connected!")

        # Create list of 8 nodes
        base = "ns=6;s=::AsGlobalPV:Vehicle.Status.drive"
        self.nodes = [
            self.client.get_node(f"{base}{i}_actual")
            for i in range(1, 9)
        ]

        self.timer = self.create_timer(0.1, self.read_opcua)

    def read_opcua(self):
        try:
            # Read all values
            values = [node.get_value() for node in self.nodes]
            self.get_logger().info(f"Drive Actual Velocities: {values}")
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
