#!/usr/bin/env python3
"""
3D Pose Estimation - JAX Implementation

This implements the JAX version of Marian's 3D_Pose_Estimation.py:
- Load H36M dataset with two camera views
- Detect humans using YOLO
- Perform 2D pose estimation using JAX RegressFlow model
- Optional OOD detection on left camera view
- Triangulate 3D poses with uncertainty propagation
- Visualize results with ground truth and estimated poses

Based on marian_code/Experiment2/3D_Pose_Estimation.py but adapted for JAX.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.datasets.h36m import Human36mDatasetTwoCameras
from src.ood_scores.lm_lanczos import load_score_functions
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    process_frame_2d
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

from human_pose_pipeline.pose_estimation.h36m_settings import (
    CONNECTIONS_13,
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
    OOD_THRESHOLD
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Dataset splits (same as Marian's)
SPLIT = {
    'train': ['S1'],
    'validation': ['S11'],
    'test': ['S6']
}


def create_video_frame(frames, poses_3d, covariances_3d, frame_idx, common_width, common_height):
    """
    Create a single video frame combining 2D camera views and 3D pose visualization.

    Args:
        frames: List of 2 camera frames (PIL Images or numpy arrays)
        poses_3d: 3D pose keypoints (13, 3)
        covariances_3d: 3D pose covariances (13, 3, 3)
        frame_idx: Current frame number
        common_width: Width for 2D frames
        common_height: Height for 2D frames

    Returns:
        numpy array: Combined video frame
    """
    # Create figure with subplots for cameras and 3D view
    fig = plt.figure(figsize=(15, 5))

    # Camera 1 view
    ax1 = fig.add_subplot(1, 3, 1)
    if isinstance(frames[0], np.ndarray):
        ax1.imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(frames[0])
    ax1.set_title(f'Camera 1 - Frame {frame_idx}')
    ax1.axis('off')

    # Camera 2 view
    ax2 = fig.add_subplot(1, 3, 2)
    if isinstance(frames[1], np.ndarray):
        ax2.imshow(cv2.cvtColor(frames[1], cv2.COLOR_BGR2RGB))
    else:
        ax2.imshow(frames[1])
    ax2.set_title(f'Camera 2 - Frame {frame_idx}')
    ax2.axis('off')

    # 3D pose view
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    if poses_3d is not None and covariances_3d is not None:
        draw_3d_pose_with_covariance(ax3, poses_3d, covariances_3d, CONNECTIONS_13, scale=1.0)
        ax3.set_title(f'3D Pose - Frame {frame_idx}')
    else:
        ax3.set_title(f'3D Pose - No Detection')

    # Set consistent 3D view limits
    ax3.set_xlim(-1000, 1000)
    ax3.set_ylim(-1000, 1000)
    ax3.set_zlim(0, 2000)

    plt.tight_layout()

    # Convert matplotlib figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Remove alpha channel

    plt.close(fig)

    return buf


def save_3d_pose_video(all_frames, all_3d_points, all_3d_covariances,
                      common_width, common_height, output_path="3d_pose_estimation_video.mp4", fps=15):
    """
    Save a video showing the 3D pose estimation results.

    Args:
        all_frames: List of frame pairs for each time step
        all_3d_points: Array of 3D poses (num_frames, 13, 3)
        all_3d_covariances: Array of 3D covariances (num_frames, 13, 3, 3)
        common_width: Width for resizing frames
        common_height: Height for resizing frames
        output_path: Output video file path
        fps: Frames per second for output video
    """
    if len(all_frames) == 0:
        print("No frames to create video")
        return

    print(f"Creating video with {len(all_frames)} frames...")

    # Create first frame to get dimensions
    sample_frame = create_video_frame(
        all_frames[0], all_3d_points[0], all_3d_covariances[0], 0,
        common_width, common_height
    )

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = sample_frame.shape[:2]
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Create frames for video
    for i, (frame_pair, pose_3d, cov_3d) in enumerate(zip(all_frames, all_3d_points, all_3d_covariances)):
        # Create combined frame
        combined_frame = create_video_frame(
            frame_pair, pose_3d, cov_3d, i, common_width, common_height
        )

        # Convert RGB to BGR for OpenCV
        combined_frame_bgr = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)

        # Write frame to video
        video_writer.write(combined_frame_bgr)

    # Release video writer
    video_writer.release()
    print(f"Video saved as: {output_path}")


def save_3d_trajectory_video(all_3d_points, output_path="3d_trajectory_video.mp4", fps=15):
    """
    Save a video showing the 3D trajectory evolution over time.

    Args:
        all_3d_points: Array of 3D poses (num_frames, 13, 3)
        output_path: Output video file path
        fps: Frames per second for output video
    """
    if len(all_3d_points) == 0:
        print("No 3D points to create trajectory video")
        return

    print(f"Creating trajectory video with {len(all_3d_points)} frames...")

    # Compute mean trajectory for visualization
    mean_trajectory = np.mean(all_3d_points, axis=1)  # (num_frames, 3)

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Set up video writer
    frames = []

    for i in range(len(all_3d_points)):
        ax.clear()

        # Plot current pose
        draw_3d_pose_with_covariance(ax, all_3d_points[i], np.zeros((13, 3, 3)), CONNECTIONS_13, scale=1.0)

        # Plot trajectory up to current frame
        if i > 0:
            ax.plot(mean_trajectory[:i+1, 0], mean_trajectory[:i+1, 1], mean_trajectory[:i+1, 2],
                   'r-', linewidth=2, alpha=0.7, label='Trajectory')

        # Set consistent view
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.set_zlim(0, 2000)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'3D Pose and Trajectory - Frame {i}')

        # Convert to frame
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        buf = buf[:, :, :3]  # Remove alpha channel
        frames.append(buf)

    plt.close(fig)

    # Save as video
    if frames:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        print(f"Trajectory video saved as: {output_path}")


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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='3D Pose Estimation with OOD Detection')
    parser.add_argument('--cache_dir', type=str, default='cache/', help='Cache directory with score functions')
    parser.add_argument('--base_key', type=str, default=None, help='Base key for loading the OOD score functions')
    parser.add_argument('--data_path', type=str, default='datasets/', help='Path to datasets')
    parser.add_argument('--model_save_path', type=str, default='models_tianle', help='Path to saved models')
    parser.add_argument('--run_name', type=str, default='finetuned_h36m_regressflow_with_unc', help='Model run name')
    parser.add_argument('--ood_threshold', type=float, default=OOD_THRESHOLD, help='OOD threshold')
    parser.add_argument('--subject', type=str, default='S1', help='Subject ID (e.g., S1, S6)')
    parser.add_argument('--action', type=str, default='WalkingDog', help='Action to visualize')
    parser.add_argument('--camera_ids', type=str, nargs=2, default=['55011271', '60457274'], help='Camera IDs')
    parser.add_argument('--max_frames', type=int, default=100, help='Maximum number of frames to process')
    parser.add_argument('--enable_ood', action='store_true', help='Enable OOD detection on left camera')
    parser.add_argument('--output_dir', type=str, default='results/pose_3d', help='Output directory for results')

    args = parser.parse_args()

    print("=" * 60)
    print("3D Pose Estimation - JAX Implementation")
    if args.enable_ood:
        print("WITH OOD DETECTION (left camera only)")
    print("=" * 60)

    # Configuration
    base_directory = os.path.join(root_dir, args.data_path, "H36M", "extracted")
    subject = args.subject
    action = args.action
    camera_ids = args.camera_ids

    try:
        # Initialize models
        print("\nInitializing models...")

        # Initialize JAX pose estimation model with uncertainty estimation
        models_dir = os.path.join(root_dir, args.model_save_path, "H36M", "RegressFlow", "seed_420")
        checkpoint_path_jax = os.path.join(models_dir, args.run_name)
        pose_estimation_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path_jax)
        print("Using RegressFlowWithAleatoric model for uncertainty estimation")

        # Initialize YOLO human detector
        human_detector, device_torch = initialize_human_detector('cuda')

        print("Models initialized successfully!")

        # Load OOD score functions if enabled
        score_fn = None
        if args.enable_ood:
            if args.base_key is None:
                print("\nWARNING: OOD detection enabled but no base_key provided. Skipping OOD detection.")
                print("Use --base_key to specify the cache key for OOD score functions.")
            else:
                print(f"\nLoading OOD score functions with cache key: {args.base_key}")
                score_fn, _, _, _ = load_score_functions(args.cache_dir, args.base_key)
                print("OOD score functions loaded successfully!")
                print(f"Using OOD threshold: {args.ood_threshold:.6f}")

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
        frames_to_process = min(args.max_frames, len(pose_sequence))
        all_3d_points = []
        all_3d_covariances = []
        all_frame_pairs = []  # Store frame pairs for video creation
        all_ood_scores = []  # Store OOD scores from left camera
        all_is_ood = []  # Store OOD classifications

        print(f"\nProcessing {frames_to_process} frames...")
        if args.enable_ood and score_fn is not None:
            print("OOD detection will be performed on LEFT camera (camera 0) only")

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

            # Store frame pair for video creation
            all_frame_pairs.append(frames.copy())

            # Process frames from both cameras
            poses_cam1 = None
            poses_cam2 = None
            uncertainties_cam1 = None
            uncertainties_cam2 = None
            cov_cam1 = None
            cov_cam2 = None
            ood_score_left = 0.0
            is_ood_left = False

            for cam_idx, frame in enumerate(frames):
                # Enable OOD detection only for left camera (cam_idx == 0)
                current_score_fn = score_fn if (args.enable_ood and cam_idx == 0) else None

                # Get pose estimations using JAX model
                pose_predictions = process_frame_2d(
                    frame=frame.copy(),
                    pose_estimation_jit_fn=pose_estimation_jit_fn,
                    params=params,
                    batch_stats=batch_stats,
                    human_detector=human_detector,
                    device_torch=device_torch,
                    mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                    score_fn=current_score_fn,
                    human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
                    ood_threshold=args.ood_threshold
                )

                if not pose_predictions:
                    continue

                # Take the first detected person
                pose = pose_predictions[0]['keypoints']
                uncertainty = pose_predictions[0]['uncertainties']
                covariance_matrix = pose_predictions[0]['covariance_matrix']

                # Store OOD results from left camera
                if cam_idx == 0:
                    poses_cam1 = pose
                    uncertainties_cam1 = uncertainty
                    cov_cam1 = covariance_matrix
                    if args.enable_ood and current_score_fn is not None:
                        ood_score_left = pose_predictions[0]['ood_score']
                        is_ood_left = pose_predictions[0]['is_ood']
                elif cam_idx == 1:
                    poses_cam2 = pose
                    uncertainties_cam2 = uncertainty
                    cov_cam2 = covariance_matrix

            # Store OOD information
            all_ood_scores.append(ood_score_left)
            all_is_ood.append(is_ood_left)

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

        print(f"\n3D pose estimation completed!")
        print(f"Processed {len(all_3d_points)} frames")

        # Print OOD statistics if enabled
        if args.enable_ood and score_fn is not None:
            all_ood_scores_arr = np.array(all_ood_scores)
            all_is_ood_arr = np.array(all_is_ood)
            print(f"\nOOD Detection Statistics (Left Camera):")
            print(f"  Mean OOD score: {all_ood_scores_arr.mean():.4f}")
            print(f"  Std OOD score: {all_ood_scores_arr.std():.4f}")
            print(f"  Classified as OOD: {all_is_ood_arr.sum()} / {len(all_is_ood_arr)} ({100*all_is_ood_arr.mean():.1f}%)")
            print(f"  OOD threshold used: {args.ood_threshold:.4f}")

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

        # Generate videos
        print("\nGenerating videos...")

        # Create 3D pose estimation video (combines 2D views + 3D pose)
        save_3d_pose_video(
            all_frame_pairs, all_3d_points, all_3d_covariances,
            common_width, common_height,
            output_path="3d_pose_estimation_video.mp4", fps=15
        )

        # Create 3D trajectory video (3D pose evolution over time)
        save_3d_trajectory_video(
            all_3d_points,
            output_path="3d_trajectory_video.mp4", fps=15
        )

        plt.show()

        print("\n" + "=" * 60)
        print("3D POSE ESTIMATION COMPLETED SUCCESSFULLY!")
        print("Videos generated:")
        print("  - 3d_pose_estimation_video.mp4: Combined 2D views + 3D poses")
        print("  - 3d_trajectory_video.mp4: 3D pose evolution over time")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during 3D pose estimation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()