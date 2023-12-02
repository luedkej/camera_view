import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped

class TFStaticSubscriberNode(rclpy.node.Node):

    def __init__(self):
        super().__init__('tf_static_subscriber_node')

        # Erstellen eines Subscribers für das /tf_static-Topic
        self.subscription = self.create_subscription(
            TransformStamped,        # Der Nachrichtentyp des Topics
            '/tf_static',             # Der Name des Topics
            self.callback,            # Die Callback-Funktion, die aufgerufen wird, wenn eine Nachricht empfangen wird
            10                       # Queue-Größe (Anzahl der zwischengespeicherten Nachrichten)
        )
        self.subscription  # Referenz auf das erstellte Abonnement speichern

    def callback(self, msg):
        # Diese Funktion wird aufgerufen, wenn eine Nachricht auf dem /tf_static-Topic empfangen wird
        self.get_logger().info('Empfangene Transformationsnachricht:\n%s' % str(msg))

def main():
    rclpy.init()
    node = TFStaticSubscriberNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

