import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import csv


class WaypointVisualizer(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer')

        self.publisher = self.create_publisher(Marker, 'visualization_marker', 10)

        # Load all waypoints once at startup
        self.points = self.load_csv('raceline.csv')
        self.get_logger().info(f'Loaded {len(self.points)} waypoints')

        # Index tracking how many points have been published so far
        self.current_index = 0

        # Publish rate: one new point every 0.05 seconds (20 Hz)
        # Adjust this to slow down or speed up the reveal
        self.publish_interval = 0.05
        self.timer = self.create_timer(self.publish_interval, self.publish_next_point)

    def load_csv(self, file_path):
        points = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x = float(row['x_m'])
                    y = float(row['y_m'])
                    points.append((x, y))
                except (KeyError, ValueError) as e:
                    self.get_logger().warn(f'Skipping row: {e}')
        return points

    def publish_next_point(self):
        # Stop once all points have been revealed
        if self.current_index >= len(self.points):
            self.get_logger().info('All waypoints published. Stopping timer.')
            self.timer.cancel()
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.scale.x = 0.15
        marker.scale.y = 0.15

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Include all points up to and including the current index
        for x, y in self.points[:self.current_index + 1]:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)

        self.publisher.publish(marker)
        self.current_index += 1


def main():
    rclpy.init()
    node = WaypointVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()