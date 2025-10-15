#!/usr/bin/env python3
"""
Debug 3D Pose Visualization Script - JAX Implementation

This script provides a simplified version of pose_estimation_3D.py for quick testing:
- Loads a single H36M frame pair from two cameras
- Detects humans using YOLO in both frames
- Performs 2D pose estimation using JAX RegressFlow model with uncertainty
- Triangulates 3D pose with uncertainty propagation
- Visualizes the results with 2D and 3D views
- Computes and displays evaluation metrics

Based on the working pose_estimation_3D.py but simplified for debugging.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    process_frame_2d
)

from human_pose_pipeline.pose_estimation.triangulation_helper import (
    load_camera_parameters,
    create_joint_covariance,
    triangulate_points_with_covariance
)

from human_pose_pipeline.utils.visualization import (
    visualize_poses_matplotlib,
    draw_3d_pose_with_covariance
)

from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_IDX_17,
    JOINT_IDX_13,
    CONNECTIONS_13,
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def load_single_frame_pair(base_directory, subject='S1', action='WalkingDog',
                           camera_ids=['55011271', '60457274'], frame_idx=100):
    """
    Load a single frame pair from two cameras with corresponding poses.

    Args:
        base_directory: Path to H36M extracted dataset
        subject: Subject ID (e.g., 'S1')
        action: Action name (e.g., 'WalkingDog')
        camera_ids: List of two camera IDs
        frame_idx: Frame index to load

    Returns:
        dict: Contains frames, poses, and video paths
    """
    from spacepy.pycdf import CDF

    # Construct paths
    poses_dir = os.path.join(base_directory, subject, 'Poses_D2_Positions')
    videos_dir = os.path.join(base_directory, subject, 'Videos')

    # Find matching pose file
    pose_files = [f for f in os.listdir(poses_dir) if f.startswith(action) and f.endswith('.cdf')]
    if not pose_files:
        raise FileNotFoundError(f"No pose file found for {subject}/{action}")

    pose_file = pose_files[0]
    pose_path = os.path.join(poses_dir, pose_file)

    # Find matching video files for both cameras
    video_files = [f"{action}.{camera_id}.mp4" for camera_id in camera_ids]
    video_paths = []
    for vf in video_files:
        video_path = os.path.join(videos_dir, vf)
        if os.path.exists(video_path):
            video_paths.append(video_path)
        else:
            raise FileNotFoundError(f"Video file not found: {video_path}")

    if len(video_paths) != 2:
        raise ValueError(f"Expected 2 camera videos, found {len(video_paths)}")

    print(f"Loading pose data from: {pose_path}")
    print(f"Loading videos from:")
    for i, path in enumerate(video_paths):
        print(f"  Camera {i+1}: {path}")

    # Load pose data
    with CDF(pose_path) as cdf:
        poses = cdf['Pose'][:]
        poses = poses.reshape(-1, 32, 2)  # (frames, 32 joints, 2 coords)
        poses_17 = poses[:, JOINT_IDX_17, :]
        poses_13 = poses_17[:, JOINT_IDX_13, :]

    # Load specific frames from both videos
    frames = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_idx >= total_frames:
            frame_idx = total_frames - 1
            print(f"Requested frame {frame_idx} exceeds video length, using frame {frame_idx}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

        # Convert frame to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        frames.append(image_pil)

    # Get corresponding pose
    if frame_idx >= len(poses_13):
        frame_idx = len(poses_13) - 1
        print(f"Frame index exceeds pose sequence, using frame {frame_idx}")

    pose_13 = poses_13[frame_idx]

    return {
        'frames': frames,  # List of 2 PIL images
        'pose_13': pose_13,  # Ground truth pose
        'video_paths': video_paths,
        'frame_idx': frame_idx,
        'camera_ids': camera_ids
    }


def print_3d_evaluation_summary(points_3d, covariances):
    """Print detailed evaluation summary for 3D pose."""
    print("\n" + "=" * 60)
    print("3D POSE ESTIMATION EVALUATION SUMMARY")
    print("=" * 60)

    print(f"3D Pose Shape: {points_3d.shape}")
    print(f"Covariances Shape: {covariances.shape}")

    # Compute uncertainty magnitudes
    uncertainty_mags = []
    for i, cov in enumerate(covariances):
        eigenvals = np.linalg.eigvals(cov)
        uncertainty_mag = np.sqrt(np.mean(eigenvals))
        uncertainty_mags.append(uncertainty_mag)

    uncertainty_mags = np.array(uncertainty_mags)

    print(f"\nUncertainty Statistics:")
    print(f"  Mean uncertainty magnitude: {np.mean(uncertainty_mags):.2f} mm")
    print(f"  Min uncertainty magnitude:  {np.min(uncertainty_mags):.2f} mm")
    print(f"  Max uncertainty magnitude:  {np.max(uncertainty_mags):.2f} mm")
    print(f"  Std uncertainty magnitude:  {np.std(uncertainty_mags):.2f} mm")

    # 3D coordinate statistics
    print(f"\n3D Coordinate Statistics:")
    print(f"  X range: [{np.min(points_3d[:, 0]):.1f}, {np.max(points_3d[:, 0]):.1f}] mm")
    print(f"  Y range: [{np.min(points_3d[:, 1]):.1f}, {np.max(points_3d[:, 1]):.1f}] mm")
    print(f"  Z range: [{np.min(points_3d[:, 2]):.1f}, {np.max(points_3d[:, 2]):.1f}] mm")

    print("=" * 60)


def main():
    """Main debug function for 3D pose estimation."""
    print("=" * 60)
    print("Debug 3D Pose Visualization - JAX Implementation")
    print("=" * 60)

    try:
        # Initialize models
        print("Initializing models...")

        # Initialize JAX pose estimation model with uncertainty estimation
        models_dir = os.path.join(root_dir, "models_tianle", "H36M", "RegressFlow", "seed_420")
        checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_with_unc")
        pose_estimation_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path_jax)
        print("Using RegressFlowWithAleatoric model for uncertainty estimation")

        # Initialize YOLO human detector
        human_detector, device_torch = initialize_human_detector('cuda')

        print("Models initialized successfully!")

        # Load camera parameters
        camera_parameters_path = os.path.join(models_dir, 'camera-parameters.json')
        if not os.path.exists(camera_parameters_path):
            print(f"Warning: Camera parameters file not found at {camera_parameters_path}")
            print("Please ensure the camera-parameters.json file is available")
            return

        # Configuration
        subject = 'S1'
        action = 'WalkingDog'
        camera_ids = ['55011271', '60457274']
        frame_idx = 50

        intrinsics, extrinsics = load_camera_parameters(camera_parameters_path, subject, camera_ids)

        # Compute projection matrices
        projection_matrices = {}
        for cam_id in camera_ids:
            K = intrinsics[cam_id]
            RT = extrinsics[cam_id]
            projection_matrices[cam_id] = K @ RT

        print("Camera parameters loaded successfully!")

        # Load sample data
        print("\nLoading H36M sample...")
        base_directory = os.path.join(root_dir, "datasets", "H36M", "extracted")

        sample = load_single_frame_pair(
            base_directory=base_directory,
            subject=subject,
            action=action,
            camera_ids=['55011271', '60457274'],  # Use the dataset camera IDs (without prefix)
            frame_idx=frame_idx
        )

        print(f"Sample loaded successfully!")
        print(f"  - Frame index: {sample['frame_idx']}")
        print(f"  - Number of camera views: {len(sample['frames'])}")

        # Process both camera frames
        print("\nRunning 2D pose estimation on both cameras...")

        poses = []
        uncertainties = []
        covariances = []

        for i, frame in enumerate(sample['frames']):
            print(f"Processing camera {i+1}...")

            pose_predictions = process_frame_2d(
                frame=frame,
                pose_estimation_jit_fn=pose_estimation_jit_fn,
                params=params,
                batch_stats=batch_stats,
                human_detector=human_detector,
                device_torch=device_torch,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                score_fn=None,  # No OOD scoring for now
                human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD
            )
            # Take the first detected person
            pose = pose_predictions[0]['keypoints']
            uncertainty = pose_predictions[0]['uncertainties']
            covariance_scalar = pose_predictions[0]['covariance']

            if pose is None:
                print(f"No human detected in camera {i+1}!")
                poses.append(np.zeros((13, 2)))
                uncertainties.append(np.ones((13, 2)) * 5.0)
                covariances.append(np.ones(13) * 0.1)
            else:
                poses.append(pose)
                uncertainties.append(uncertainty)
                covariances.append(covariance_scalar)

        if len(poses) != 2:
            print("Error: Need exactly 2 camera views for triangulation")
            return

        # Triangulate 3D pose
        print("\nTriangulating 3D pose...")

        # Create joint covariance matrices
        C_joint_list = []
        for i in range(13):
            C_joint = create_joint_covariance(
                mapped_uncertainty_cam1=uncertainties[0][i],
                mapped_covariance_cam1=covariances[0][i],
                mapped_uncertainty_cam2=uncertainties[1][i],
                mapped_covariance_cam2=covariances[1][i],
                cross_covariance=np.zeros((2, 2))  # Assume zero cross-covariance
            )
            C_joint_list.append(C_joint)

        # Triangulate
        P1 = projection_matrices[camera_ids[0]]
        P2 = projection_matrices[camera_ids[1]]
        points_3d, covariances_3d = triangulate_points_with_covariance(
            poses[0], poses[1], P1, P2, C_joint_list
        )

        print("3D triangulation completed!")

        # Print evaluation summary
        print_3d_evaluation_summary(points_3d, covariances_3d)

        # Create comprehensive visualization
        print("\nCreating visualizations...")

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))

        # 2D pose visualizations for both cameras
        for i in range(2):
            ax = fig.add_subplot(2, 3, i+1)
            ax.imshow(sample['frames'][i])
            ax.set_title(f'Camera {i+1} - Original Image')
            ax.axis('off')

        # 2D poses with uncertainty for both cameras
        for i in range(2):
            # Create individual visualization for each camera
            save_path = f"debug_3d_camera_{i+1}_frame_{frame_idx}.png"
            visualize_poses_matplotlib(
                image=sample['frames'][i],
                pred_pose=poses[i],
                pred_uncertainties=uncertainties[i],
                pred_covariances=covariances[i],
                save_path=save_path,
                show_uncertainty=True
            )

        # 3D pose visualization
        ax_3d = fig.add_subplot(2, 3, (5, 6), projection='3d')
        draw_3d_pose_with_covariance(ax_3d, points_3d, covariances_3d, CONNECTIONS_13)

        plt.tight_layout()

        # Save comprehensive visualization
        save_path = f"debug_3d_comprehensive_frame_{frame_idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comprehensive visualization saved: {save_path}")

        # Save individual 3D plot
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d_solo = fig_3d.add_subplot(111, projection='3d')
        draw_3d_pose_with_covariance(ax_3d_solo, points_3d, covariances_3d, CONNECTIONS_13)

        save_path_3d = f"debug_3d_pose_frame_{frame_idx}.png"
        plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
        print(f"3D pose visualization saved: {save_path_3d}")

        plt.show()

        print("\n" + "=" * 60)
        print("DEBUG 3D POSE ESTIMATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Frame processed: {frame_idx}")
        print(f"Action: {action}")
        print(f"Subject: {subject}")
        print(f"Cameras: {camera_ids}")

    except Exception as e:
        print(f"\nError during debug 3D pose estimation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()