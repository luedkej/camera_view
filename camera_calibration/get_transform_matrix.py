# get the static transform from the camera to the chessboard
import pyzed.sl as sl
import cv2
import numpy as np

# Define the chessboard parameters
chessboard_size = (8, 11)  # Number of inner corners on the chessboard
square_size = 0.03  # Size of each square in meters

# Create a ZED camera object
zed = sl.Camera()

# Set initialization parameters
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080  # Set resolution
init_params.camera_fps = 30  # Set FPS
init_params.depth_mode = sl.DEPTH_MODE.NONE  # No depth for calibration

# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

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
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    ret, corners = cv2.findChessboardCorners(image_gray, chessboard_size, None)

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
        axis_length = square_size * 3  # Length of the coordinate axes in meters
        imgpts, jac = cv2.projectPoints(np.float32([0, 0, 0]), rvecs, tvecs, camera_matrix, dist_coeffs)

        imgpts = np.int32(imgpts).reshape(-1, 2)

        img = cv2.drawChessboardCorners(image_gray, chessboard_size, corners2, ret)
        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs, tvecs, axis_length)

        # Display the image
        cv2.imshow("Chessboard", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rvecs, tvecs, transform_matrix
    

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

rvecs, tvecs, transform_matrix = get_static_transform(gray, mtx, dist_coeffs, True)

print(transform_matrix)
