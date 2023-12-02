import rclpy
from rclpy.node import Node

from rclpy.qos import QoSProfile

from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as R

from messages_artur.srv import GetChessboardTransform

import pyzed.sl as sl
import cv2
import numpy as np


# Create a ZED camera object
zed = sl.Camera()

# Define the chessboard parameters
CHESSBOARD_SIZE = (8, 11)  # Number of inner corners on the chessboard
SQUARE_SIZE = 0.03  # Size of each square in meters


def matrix_to_translation_quaternion(transform_matrix):
    # Assuming transform_matrix is a 4x4 transformation matrix
    translation = transform_matrix[:3, 3]

    # Extract the rotation part of the matrix
    rotation_matrix = transform_matrix[:3, :3]
    # Convert to a rotation object to get the quaternion
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat()  # (x, y, z, w)

    return translation, quaternion

def get_intrinsics(): 
    # Get intrinsic parameters from ZED API
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters

    fx = calibration_params.left_cam.fx
    fy = calibration_params.left_cam.fy
    cx = calibration_params.left_cam.cx
    cy = calibration_params.left_cam.cy
    k1 = calibration_params.left_cam.disto[0]
    k2 = calibration_params.left_cam.disto[1]
    k3 = calibration_params.left_cam.disto[2]
    p1 = calibration_params.left_cam.disto[3]
    p2 = calibration_params.left_cam.disto[4]

    # Create a vector containing the known distortion coefficients
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

    # Create the camera matrix mtx
    mtx = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]], dtype=np.float64)

    return mtx, dist_coeffs

def get_chessboard_corners(image_gray):    
    # Generate chessboard world coordinates
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

    ret, corners = cv2.findChessboardCorners(image_gray, CHESSBOARD_SIZE, None)

    if ret:
        # Refine the corners' position
        corners_refined = cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        return corners_refined, objp, ret
    else:
        raise Exception("Chessboard corners not found.")

def get_static_transform(image_gray, camera_matrix, dist_coeffs, visualize=False): 
    corners2, objp, ret = get_chessboard_corners(image_gray)

    # Solve the PnP problem
    _, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Construct the transformation matrix
    rot_mat, _ = cv2.Rodrigues(rvecs)
    transform_matrix = np.hstack((rot_mat, tvecs))

    if visualize:
        # Overlay coordinate axes on the chessboard image
        axis_length = SQUARE_SIZE * 3  # Length of the coordinate axes in meters
        imgpts, jac = cv2.projectPoints(np.float32([0, 0, 0]), rvecs, tvecs, camera_matrix, dist_coeffs)

        imgpts = np.int32(imgpts).reshape(-1, 2)

        img = cv2.drawChessboardCorners(image_gray, CHESSBOARD_SIZE, corners2, ret)
        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs, tvecs, axis_length)

        # Display the image
        cv2.imshow("Chessboard", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rvecs, tvecs, transform_matrix


class CameraChessboardNode(Node):
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

        #super().__init__('camera_chessboard_node')
        #self.srv = self.create_service(GetChessboardTransform, 'get_chessboard_transform', self.handle_service)
        #self.br = StaticTransformBroadcaster(self)

    #def handle_service(self, request, response):
    def publish_tf_static(self):
        # Capture frame, calculate transform
       while True:
        #try:
            # Code pasted from get_transform_matrix.py
            


            # Set initialization parameters
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD1080  # Set resolution
            init_params.camera_fps = 30  # Set FPS
            init_params.depth_mode = sl.DEPTH_MODE.NONE  # No depth for calibration

            # Open the camera
            self.get_logger().info('Opening camera...')
            err = zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                exit(1)
            self.get_logger().info('Camera opened.')

            # Capture frame
            runtime_parameters = sl.RuntimeParameters()

            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                image_zed = sl.Mat()
                zed.retrieve_image(
                    image_zed, sl.VIEW.LEFT
                )  # Get the left image from the camera
                frame = image_zed.get_data()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mtx, dist_coeffs = get_intrinsics()
            rvecs, tvecs, transform_matrix = get_static_transform(gray, mtx, dist_coeffs, False)

            # Convert transform_matrix to a TransformStamped message
            transform_stamped = TransformStamped()
            transform_stamped.header.stamp = self.get_clock().now().to_msg()
            transform_stamped.header.frame_id = 'zed_camera_frame'
            transform_stamped.child_frame_id = 'chessboard_frame'

            # Convert transform_matrix to translation vector and quaternion
            translation, quaternion = matrix_to_translation_quaternion(transform_matrix)

            # Add translation vector and quaternion to transform_stamped message
            transform_stamped.transform.translation.x = translation[0]
            transform_stamped.transform.translation.y = translation[1]
            transform_stamped.transform.translation.z = translation[2]
            
            transform_stamped.transform.rotation.x = quaternion[0]
            transform_stamped.transform.rotation.y = quaternion[1]
            transform_stamped.transform.rotation.z = quaternion[2]
            transform_stamped.transform.rotation.w = quaternion[3]
            
            self.publisher.publish(transform_stamped)
            self.get_logger().info('Statische Transformation veröffentlicht:\n%s' % str(transform_msg))

            # Broadcast static transform
            #self.br.sendTransform(transform_stamped)
            ## self.tf_static_broadcaster.sendTransfrom(transform_stamped)
            #self.get_logger().info('Static Transform broadcasted. Calculated transformation matrix:')
            #self.get_logger().info('\n' + np.array2string(transform_matrix))
            #response.success = True
            #return response

        #except Exception as e:
           #self.get_logger().error('Failed to calculate transform %s' % str(e))
            #response.success = False
            #return response

	
	
	

def main(args=None):
    #rclpy.init(args=args)
    #node = CameraChessboardNode()
    #rclpy.spin(node)
    #rclpy.shutdown()
    
    rclpy.init()
    node = TFStaticPublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()


