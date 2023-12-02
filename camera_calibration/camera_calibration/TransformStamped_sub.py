import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node

class TFSubscriber(Node):

    def __init__(self):
        super().__init__('tf_subscriber_node')

        self.subscription = self.create_subscription(
            TransformStamped,
            '/tf_static',
            self.tf_callback,
            10  # Queue-Größe (Anzahl der zwischengespeicherten Nachrichten)
        )

    def tf_callback(self, msg):
        # Diese Funktion wird aufgerufen, wenn eine Nachricht auf dem /tf_static-Topic empfangen wird
        self.get_logger().info(f'Received TransformStamped:\n{msg}')

def main():
    rclpy.init()
    node = TFSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
