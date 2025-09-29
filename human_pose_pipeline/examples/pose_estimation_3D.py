#!/usr/bin/env python3
"""
3D Pose Estimation - JAX Implementation

This implements the JAX version of Marian's 3D_Pose_Estimation.py:
- Load H36M dataset with two camera views
- Detect humans using YOLO
- Perform 2D pose estimation using JAX RegressFlow model
- Triangulate 3D poses with uncertainty propagation
- Visualize results with ground truth and estimated poses

Based on marian_code/Experiment2/3D_Pose_Estimation.py but adapted for JAX.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import Image
import json
from tqdm import tqdm
import jax.numpy as jnp

# Add root directory to path to access src
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from src.datasets.h36m import Human36mDatasetSequence, Human36mDatasetTwoCameras
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    process_frame_2d
)
from human_pose_pipeline.evaluation.pose_metrics import (
    MIRROR_13_JOINT_MODEL_MAP
)
from human_pose_pipeline.pose_estimation.triangulation_helper import (
    load_camera_parameters,
    create_joint_covariance,
    triangulate_points_with_covariance,
    validate_projection_matrices
)
from human_pose_pipeline.utils.visualization import (
    draw_3d_pose_with_covariance
)

# Same mappings as in Marian's code
JOINT_IDX_17 = [0, 1, 2, 3, 6, 7, 8, 12, 16, 14, 15, 17, 18, 19, 25, 26, 27]
JOINT_IDX_13 = [9, 14, 11, 15, 12, 16, 13, 1, 4, 2, 5, 3, 6]

# Skeleton connections for 13-joint visualization
CONNECTIONS_13 = [
    (0, 1), (0, 2),  # Nose to shoulders
    (1, 3), (3, 5),  # Left arm
    (2, 4), (4, 6),  # Right arm
    (1, 2), (1, 7), (2, 8),  # Shoulders to hips
    (7, 8),  # Connect hips
    (7, 9), (9, 11),  # Left leg
    (8, 10), (10, 12)  # Right leg
]

# Dataset splits (same as Marian's)
SPLIT = {
    'train': ['S1'],
    'validation': ['S11'],
    'test': ['S6']
}


# def visualize_pose_with_uncertainty(frame, pose, uncertainty, covariance_scalar, connections):
#     """
#     Visualize 2D pose with uncertainty ellipses on frame.
#     """
#     frame_viz = frame.copy()

#     # Draw skeleton connections
#     for connection in connections:
#         start_idx, end_idx = connection
#         start_point = tuple(pose[start_idx].astype(int))
#         end_point = tuple(pose[end_idx].astype(int))
#         cv2.line(frame_viz, start_point, end_point, color=(0, 255, 0), thickness=2)

#     # Draw keypoints with uncertainty ellipses
#     for idx, (x, y) in enumerate(pose):
#         cv2.circle(frame_viz, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)

#         std_x, std_y = uncertainty[idx]
#         cov_xy = covariance_scalar[idx]

#         if std_x > 0 and std_y > 0:
#             angle = 0.5 * np.arctan2(2 * float(cov_xy), (float(std_x)**2 - float(std_y)**2)) * (180 / np.pi)
#             width = int(2 * std_x)
#             height = int(2 * std_y)
#             cv2.ellipse(frame_viz, (int(x), int(y)), (width, height), float(angle), 0, 360, (0, 0, 255), 1)

#     return frame_viz

def main():
    """
    Main function for running 3D pose estimation on the Human3.6M dataset.
    JAX version of Marian's main function.
    """
    print("=" * 60)
    print("3D Pose Estimation - JAX Implementation")
    print("=" * 60)

    # Configuration
    base_directory = os.path.join(root_dir, "datasets", "H36M", "extracted")
    subject = "S1"
    action = "WalkingDog"
    camera_ids = ['55011271', '60457274']  # Ensure these match your camera IDs

    try:
        # Initialize models
        print("Initializing models...")

        # Initialize JAX pose estimation model with uncertainty estimation
        models_dir = os.path.join(root_dir, "models_tianle", "H36M", "RegressFlow", "seed_420")
        checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_with_unc")
        model, params, batch_stats = initialize_jax_models(checkpoint_path_jax)
        print("Using RegressFlowWithAleatoric model for uncertainty estimation")

        # Initialize YOLO human detector
        human_detector, device_torch = initialize_human_detector('cuda')

        print("Models initialized successfully!")

        # Load camera parameters
        camera_parameters_path = os.path.join(models_dir, 'camera-parameters.json')
        if not os.path.exists(camera_parameters_path):
            print(f"Warning: Camera parameters file not found at {camera_parameters_path}")
            print("Please ensure the camera-parameters.json file is available in the models directory")
            return

        intrinsics, extrinsics = load_camera_parameters(camera_parameters_path, subject, camera_ids)

        # Compute projection matrices
        projection_matrices = {}
        for cam_id in camera_ids:
            K = intrinsics[cam_id]
            RT = extrinsics[cam_id]
            projection_matrices[cam_id] = K @ RT  # P = K[R|t]

        # Validate projection matrices
        validate_projection_matrices(projection_matrices[camera_ids[0]], projection_matrices[camera_ids[1]])

        # Create dataset
        dataset = Human36mDatasetTwoCameras(base_directory, subject, action, camera_ids=camera_ids)

        if len(dataset) == 0:
            print("No data found. Please check the dataset path and camera IDs.")
            return

        print(f"Dataset loaded with {len(dataset)} samples")

        # Get a sample from the dataset
        sample_idx = 0  # Use first sample
        sample = dataset[sample_idx]
        video_paths = sample['video_paths']
        pose_sequence = np.array(sample['pose_sequence'])

        print(f"Processing videos:")
        for i, path in enumerate(video_paths):
            print(f"  Camera {i+1}: {path}")

        # Open video captures
        caps = [cv2.VideoCapture(vp) for vp in video_paths]

        # Define a common frame size
        common_width = 640
        common_height = 480

        # Set up the 3D plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Process a limited number of frames for testing
        frames_to_process = min(100, len(pose_sequence))
        all_3d_points = []
        all_3d_covariances = []

        print(f"Processing {frames_to_process} frames...")

        for frame_idx in tqdm(range(frames_to_process), desc="Processing frames"):
            ret_flags = []
            frames = []

            # Read frames from both cameras
            for cap in caps:
                ret, frame = cap.read()
                ret_flags.append(ret)
                if ret:
                    frame = cv2.resize(frame, (common_width, common_height))
                    frames.append(frame)

            if not all(ret_flags):
                break

            # Process frames from both cameras
            poses_cam1 = None
            poses_cam2 = None
            uncertainties_cam1 = None
            uncertainties_cam2 = None
            cov_cam1 = None
            cov_cam2 = None

            for cam_idx, frame in enumerate(frames):
                pose, uncertainty, covariance_scalar, covariance = process_frame_2d(
                    frame.copy(), model, params, batch_stats, human_detector, device_torch, MIRROR_13_JOINT_MODEL_MAP
                )

                if cam_idx == 0:
                    poses_cam1 = pose
                    uncertainties_cam1 = uncertainty
                    cov_cam1 = covariance
                elif cam_idx == 1:
                    poses_cam2 = pose
                    uncertainties_cam2 = uncertainty
                    cov_cam2 = covariance

            # Triangulate 3D points if both poses are available
            if poses_cam1 is not None and poses_cam2 is not None:
                # Create joint covariance matrices
                C_joint_list = []
                for i in range(13):
                    C_joint = create_joint_covariance(
                        mapped_uncertainty_cam1=uncertainties_cam1[i],
                        mapped_covariance_cam1=cov_cam1[i, 0, 1],
                        mapped_uncertainty_cam2=uncertainties_cam2[i],
                        mapped_covariance_cam2=cov_cam2[i, 0, 1],
                        cross_covariance=np.zeros((2, 2))  # Assume zero cross-covariance
                    )
                    C_joint_list.append(C_joint)

                P1 = projection_matrices[camera_ids[0]]
                P2 = projection_matrices[camera_ids[1]]
                points_3d, C_3d_all = triangulate_points_with_covariance(
                    poses_cam1, poses_cam2, P1, P2, C_joint_list
                )

                all_3d_points.append(points_3d)
                all_3d_covariances.append(C_3d_all)
            else:
                all_3d_points.append(np.zeros((13, 3)))
                all_3d_covariances.append(np.zeros((13, 3, 3)))

        # Release resources
        for cap in caps:
            cap.release()

        print(f"3D pose estimation completed!")
        print(f"Processed {len(all_3d_points)} frames")

        # Convert to numpy arrays
        all_3d_points = np.array(all_3d_points)  # Shape: (num_frames, 13, 3)
        all_3d_covariances = np.array(all_3d_covariances)  # Shape: (num_frames, 13, 3, 3)

        # Visualize a sample frame
        if len(all_3d_points) > 0:
            sample_frame = len(all_3d_points) // 2  # Middle frame
            draw_3d_pose_with_covariance(
                ax, all_3d_points[sample_frame], all_3d_covariances[sample_frame],
                CONNECTIONS_13, scale=1.0
            )
            plt.savefig(f"3d_pose_estimation_frame_{sample_frame}.png", dpi=150, bbox_inches='tight')
            print(f"Sample 3D pose visualization saved as: 3d_pose_estimation_frame_{sample_frame}.png")

        # Compute mean positions across joints for each frame
        mean_3d_points = np.mean(all_3d_points, axis=1)  # Shape: (num_frames, 3)

        # Plot the 3D trajectory
        fig2 = plt.figure(figsize=(10, 8))
        ax2 = fig2.add_subplot(111, projection='3d')
        ax2.plot(mean_3d_points[:, 0], mean_3d_points[:, 1], mean_3d_points[:, 2],
                label='Mean 3D Trajectory', linewidth=2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Mean 3D Pose Trajectory Over Time')
        ax2.legend()
        plt.savefig("3d_trajectory.png", dpi=150, bbox_inches='tight')
        print("3D trajectory saved as: 3d_trajectory.png")

        plt.show()

        print("\n" + "=" * 60)
        print("3D POSE ESTIMATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during 3D pose estimation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()