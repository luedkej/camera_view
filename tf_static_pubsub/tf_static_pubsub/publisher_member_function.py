import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped

class TFStaticPublisherNode(Node):

    def __init__(self):
        super().__init__('tf_static_publisher_node')

        # Erstellen eines Publishers für das /tf_static-Topic
        self.publisher = self.create_publisher(
            TransformStamped,        # Der Nachrichtentyp des Topics
            '/tf_static',             # Der Name des Topics
            10                       # Queue-Größe (Anzahl der zwischengespeicherten Nachrichten)
        )

        # Timer für die Veröffentlichung von Transformationen
        self.timer = self.create_timer(1.0, self.publish_tf_static)

    def publish_tf_static(self):
        # Erstellen einer Beispiel-Transformation und Veröffentlichen auf /tf_static
        transform_msg = TransformStamped()
        transform_msg.header.frame_id = 'base_link'
        transform_msg.child_frame_id = 'camera_link'
        transform_msg.transform.translation.x = 1.0
        transform_msg.transform.translation.y = 0.0
        transform_msg.transform.translation.z = 0.0
        transform_msg.transform.rotation.w = 1.0

        self.publisher.publish(transform_msg)
        self.get_logger().info('Statische Transformation veröffentlicht:\n%s' % str(transform_msg))

def main():
    rclpy.init()
    node = TFStaticPublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

