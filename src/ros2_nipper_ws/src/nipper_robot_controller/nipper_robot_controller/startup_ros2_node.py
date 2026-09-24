#!/usr/bin/env python3  
import rclpy
from rclpy.node import Node

#Nodes communicate with eachother with topics!

class diff_node(Node):  #We define a class diff_node which inherets from the Node 
                        #that is in rclpy.node 
                        

    def __init__(self):
        super().__init__("Diff_drive_node")
        #self.get_logger().info("ROS")
        self.counter_ = 0
        self.create_timer(1.0,self.timer_callback)


    def timer_callback(self):
        self.get_logger().info("Hello " + str(self.counter_))
        self.counter_ += 1


def main(args=None):
    rclpy.init(args=args)
    node = diff_node()


    #Write your node here 
    rclpy.spin(node)  #Make node spin means that the node will continue to run until we press 
    rclpy.shutdown()

if __name__ == '__main__':
    main()