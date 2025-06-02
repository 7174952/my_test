import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
import csv
import os

class OdomVehicleVelLogger(Node):
    def __init__(self):
        super().__init__('odom_vehiclevel_logger')

        self.current_position = [None, None, None]
        self.current_linear_vel = None
        self.current_angular_vel = None

        self.filename = 'odom_vehiclevel_log.csv'
        self._init_file()

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(TwistStamped, '/vehicle_vel', self.vehicle_vel_callback, 10)

    def _init_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'x', 'y', 'z', 'linear_x', 'angular_z'])

    def odom_callback(self, msg):
        self.current_position = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]
        self.save_data(msg.header.stamp)

    def vehicle_vel_callback(self, msg):
        self.current_linear_vel = msg.twist.linear.x
        self.current_angular_vel = msg.twist.angular.z
        # 只更新，不保存

    def save_data(self, stamp):
        if None in self.current_position or self.current_linear_vel is None or self.current_angular_vel is None:
            return
        t = stamp.sec + stamp.nanosec * 1e-9
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                f"{t:.5f}",
                f"{self.current_position[0]:.5f}",
                f"{self.current_position[1]:.5f}",
                f"{self.current_position[2]:.5f}",
                f"{self.current_linear_vel:.5f}",
                f"{self.current_angular_vel:.5f}"
            ])
        self.get_logger().info(
            f"记录: t={t:.5f} x={self.current_position[0]:.5f} y={self.current_position[1]:.5f} "
            f"z={self.current_position[2]:.5f} v={self.current_linear_vel:.5f} w={self.current_angular_vel:.5f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = OdomVehicleVelLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

