import rclpy
from rclpy.node import Node
import numpy as np

from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage


class TfstaticSubscriber(Node):
    
    def __init__(self):
        super().__init__('tf_static_subscriber')
        self.subscription = self.create_subscription(TFMessage, "/tf_static", self.listener_callback, 10)

    def listener_callback(self, msg):
        for transform_stamped_msg in msg.transforms:
            transformation_matrix = self.transform_stamped_to_matrix(transform_stamped_msg)
            self.print_transformation_matrix(transformation_matrix)

    def transform_stamped_to_matrix(self, transform_stamped):
        translation = np.array([
            transform_stamped.transform.translation.x,
            transform_stamped.transform.translation.y,
            transform_stamped.transform.translation.z
        ])

        rotation = np.array([
            transform_stamped.transform.rotation.x,
            transform_stamped.transform.rotation.y,
            transform_stamped.transform.rotation.z,
            transform_stamped.transform.rotation.w
        ])

        rotation_matrix = self.quaternion_to_matrix(rotation)

        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation

        return transformation_matrix

    def quaternion_to_matrix(self, quaternion):
        x, y, z, w = quaternion
        rotation_matrix = np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])

        return rotation_matrix

    def print_transformation_matrix(self, transformation_matrix):
        self.get_logger().info("Transformationsmatrix:")
        self.get_logger().info(np.array2string(transformation_matrix, separator=', '))

    
    
def main(args=None):
    rclpy.init(args=args)

    tf_static_subscriber = TfstaticSubscriber()

    rclpy.spin(tf_static_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    tf_static_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
