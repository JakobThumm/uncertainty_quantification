"""This file contains helper functions for triangulation and 3D pose estimation."""

import numpy as np
import cv2
import json


def load_camera_parameters(json_path, subject, camera_ids):
    """Load H36M camera parameters from JSON file"""
    with open(json_path, 'r') as f:
        params = json.load(f)

    intrinsics = {}
    extrinsics = {}
    projection_matrices = {}

    for cam_id in camera_ids:
        cam_id_with_dot = f".{cam_id}"
        intrinsics[cam_id] = np.array(params['intrinsics'][cam_id_with_dot]['calibration_matrix'])
        R = np.array(params['extrinsics'][subject][cam_id_with_dot]['R'])
        t = np.array(params['extrinsics'][subject][cam_id_with_dot]['t']).reshape(3, 1)
        extrinsics[cam_id] = np.hstack((R, t))  # [R|t]

        # Compute projection matrix
        K = intrinsics[cam_id]
        RT = extrinsics[cam_id]
        projection_matrices[cam_id] = K @ RT  # P = K[R|t]

    return intrinsics, extrinsics, projection_matrices


def create_joint_covariance(mapped_uncertainty_cam1, mapped_covariance_cam1,
                            mapped_uncertainty_cam2, mapped_covariance_cam2,
                            cross_covariance=None):
    """
    Create a 4x4 joint covariance matrix from two camera observations.

    Args:
        mapped_uncertainty_cam1: (std_x, std_y) for camera 1
        mapped_covariance_cam1: covariance xy for camera 1
        mapped_uncertainty_cam2: (std_x, std_y) for camera 2
        mapped_covariance_cam2: covariance xy for camera 2
        cross_covariance: 2x2 cross-covariance matrix between cameras (optional)

    Returns:
        np.ndarray: 4x4 joint covariance matrix
    """
    C1 = np.array([
        [float(mapped_uncertainty_cam1[0])**2, float(mapped_covariance_cam1)],
        [float(mapped_covariance_cam1), float(mapped_uncertainty_cam1[1])**2]
    ])
    C2 = np.array([
        [float(mapped_uncertainty_cam2[0])**2, float(mapped_covariance_cam2)],
        [float(mapped_covariance_cam2), float(mapped_uncertainty_cam2[1])**2]
    ])

    if cross_covariance is None:
        C12 = np.zeros((2, 2))
    else:
        C12 = cross_covariance

    upper = np.hstack((C1, C12))
    lower = np.hstack((C12.T, C2))
    C_joint = np.vstack((upper, lower))

    # Check if C_joint is positive semi-definite
    eigenvalues = np.linalg.eigvals(C_joint)
    if not np.all(eigenvalues >= 0):
        print("Warning: Joint Covariance Matrix is not Positive Semi-Definite.")

    return C_joint  # Shape: (4, 4)


def triangulate_point_with_covariance(pose_cam1, pose_cam2, P1, P2, C_joint):
    """
    Triangulate a single 3D point and propagate covariance.

    Args:
        pose_cam1: 2D point from camera 1, shape (2,)
        pose_cam2: 2D point from camera 2, shape (2,)
        P1: Projection matrix for camera 1, shape (3, 4)
        P2: Projection matrix for camera 2, shape (3, 4)
        C_joint: 4x4 covariance matrix for the joint's 2D observations

    Returns:
        tuple: (point_3d, C_3d) - 3D point and 3x3 covariance matrix
    """
    # Original 2D points
    x1, y1 = pose_cam1
    x2, y2 = pose_cam2

    # Triangulate using OpenCV
    points_4d_hom = cv2.triangulatePoints(
        P1, P2, np.array([[x1], [y1]]), np.array([[x2], [y2]])
    )

    # Convert to 3D by dividing by the fourth (homogeneous) coordinate
    w = points_4d_hom[3, 0]
    if w == 0:
        w = 1e-8  # Prevent division by zero
    points_3d = points_4d_hom[:3, 0] / w  # Shape: (3,)

    # Numerical differentiation to compute the Jacobian
    epsilon = 1e-3
    J = np.zeros((3, 4))  # Jacobian matrix should be 3x4

    # Define input variables
    input_points = np.array([x1, y1, x2, y2])

    for i in range(4):
        perturbed_points = input_points.copy()
        perturbed_points[i] += epsilon

        # Perturbed 2D points
        px1, py1, px2, py2 = perturbed_points

        # Triangulate with perturbed points
        perturbed_4d_hom = cv2.triangulatePoints(
            P1, P2, np.array([[px1], [py1]]), np.array([[px2], [py2]])
        )

        # Convert to 3D
        w_perturbed = perturbed_4d_hom[3, 0]
        if w_perturbed == 0:
            w_perturbed = 1e-8
        perturbed_3d = perturbed_4d_hom[:3, 0] / w_perturbed

        # Compute the partial derivative
        J[:, i] = (perturbed_3d - points_3d) / epsilon

    # Propagate covariance using the Jacobian
    C_3d = J @ C_joint @ J.T  # Shape: (3, 3)

    # Check if C_3d is positive semi-definite
    if not is_positive_semi_definite(C_3d):
        print("Warning: C_3d is not positive semi-definite.")

    return points_3d, C_3d


def triangulate_points_with_covariance(pose_cam1, pose_cam2, P1, P2, C_joint_list):
    """
    Triangulate 3D points from two camera views with covariance propagation.

    Args:
        pose_cam1: 2D keypoints from camera 1, shape (13, 2)
        pose_cam2: 2D keypoints from camera 2, shape (13, 2)
        P1: Projection matrix for camera 1, shape (3, 4)
        P2: Projection matrix for camera 2, shape (3, 4)
        C_joint_list: List of 4x4 covariance matrices for each joint

    Returns:
        tuple: (points_3d, C_3d_all) - 3D points and covariance matrices
    """
    points_3d = np.zeros((13, 3))
    C_3d_all = np.zeros((13, 3, 3))

    for i in range(13):
        point_3d, C_3d = triangulate_point_with_covariance(
            pose_cam1[i], pose_cam2[i], P1, P2, C_joint_list[i]
        )
        points_3d[i] = point_3d
        C_3d_all[i] = C_3d

    return points_3d, C_3d_all


def is_positive_semi_definite(matrix):
    """Check if a matrix is positive semi-definite."""
    return np.all(np.linalg.eigvals(matrix) >= -1e-8)


def validate_projection_matrices(P1, P2):
    """
    Validate projection matrices by triangulating a known 3D point.
    """
    # Define a known 3D point in front of both cameras
    test_3d = np.array([0, 0, 1000])  # Arbitrary point

    # Convert to homogeneous coordinates
    test_3d_hom = np.append(test_3d, 1)  # Shape: (4,)

    # Project to both cameras
    proj_cam1 = P1 @ test_3d_hom  # Shape: (3,)
    proj_cam2 = P2 @ test_3d_hom  # Shape: (3,)

    # Convert to 2D
    proj_cam1_2d = proj_cam1[:2] / proj_cam1[2]
    proj_cam2_2d = proj_cam2[:2] / proj_cam2[2]

    # Now triangulate back
    points_4d_hom = cv2.triangulatePoints(
        P1, P2, proj_cam1_2d.reshape(2,1), proj_cam2_2d.reshape(2,1)
    )
    triangulated_3d = points_4d_hom[:3, 0] / points_4d_hom[3, 0]

    print(f"Projection validation - Original: {test_3d}, Triangulated: {triangulated_3d}")