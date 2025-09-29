#!/usr/bin/env python3
"""
Debug Pose Visualization Script - JAX Implementation

This script provides a simplified version of pose_estimation_2D.py for quick testing:
- Loads a single H36M image and ground truth pose
- Detects humans using YOLO
- Performs 2D pose estimation using JAX RegressFlow model
- Visualizes the results side-by-side with ground truth
- Computes and displays evaluation metrics

Based on the working pose_estimation_2D.py but simplified for debugging.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from spacepy.pycdf import CDF
import jax.numpy as jnp
import cv2
from PIL import Image
from scipy.stats import chi2

# Add root directory to path to access src
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    pose_estimation_2d
)
from human_pose_pipeline.evaluation.pose_metrics import (
    mpjpe_jax,
    JOINT_NAMES_13
)

# Same mappings as pose_estimation_2D.py
JOINT_IDX_17 = [0, 1, 2, 3, 6, 7, 8, 12, 16, 14, 15, 17, 18, 19, 25, 26, 27]
JOINT_IDX_13 = [10, 14, 11, 15, 12, 16, 13, 1, 4, 2, 5, 3, 6]
JOINT_IDX_13_MODEL = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]

# Skeleton connections for visualization
CONNECTIONS_13 = [
    (0, 1), (0, 2),  # Nose to shoulders
    (1, 3), (3, 5),  # Left arm
    (2, 4), (4, 6),  # Right arm
    (1, 2), (1, 7), (2, 8),  # Shoulders to hips
    (7, 8),  # Connect hips
    (7, 9), (9, 11),  # Left leg
    (8, 10), (10, 12)  # Right leg
]

def load_single_sample(base_directory, subject='S1', action='Directions', camera='55011271', frame_idx=100):
    """
    Load a single sample from H36M dataset

    Args:
        base_directory: Path to H36M extracted dataset
        subject: Subject ID (e.g., 'S1')
        action: Action name (e.g., 'Directions')
        camera: Camera ID (e.g., '55011271')
        frame_idx: Frame index to load

    Returns:
        dict: Contains 'image', 'pose_13', 'video_path', 'frame_idx'
    """
    # Construct paths
    poses_dir = os.path.join(base_directory, subject, 'Poses_D2_Positions')
    videos_dir = os.path.join(base_directory, subject, 'Videos')

    # Find matching pose file
    pose_filename = None
    for filename in os.listdir(poses_dir):
        if action in filename and camera in filename and filename.endswith('.cdf'):
            pose_filename = filename
            break

    if not pose_filename:
        raise FileNotFoundError(f"No pose file found for {subject}/{action}/{camera}")

    # Find matching video file
    video_filename = None
    base_name = os.path.splitext(pose_filename)[0]
    for video_name in [f"{base_name}.mp4", f"_{base_name}.mp4"]:
        video_path = os.path.join(videos_dir, video_name)
        if os.path.exists(video_path):
            video_filename = video_name
            break

    if not video_filename:
        raise FileNotFoundError(f"No video file found for {pose_filename}")

    # Load pose data
    pose_file_path = os.path.join(poses_dir, pose_filename)
    video_file_path = os.path.join(videos_dir, video_filename)

    print(f"Loading pose data from: {pose_file_path}")
    print(f"Loading video from: {video_file_path}")

    with CDF(pose_file_path) as cdf:
        poses = cdf['Pose'][:]
        poses = poses.reshape(-1, 32, 2)  # (frames, 32 joints, 2 coords)
        poses_17 = poses[:, JOINT_IDX_17, :]
        poses_13 = poses_17[:, JOINT_IDX_13, :]

    # Load specific frame from video
    cap = cv2.VideoCapture(video_file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_idx >= total_frames:
        frame_idx = total_frames - 1
        print(f"Requested frame {frame_idx} exceeds video length, using frame {frame_idx}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from video")

    # Convert frame to RGB PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(frame_rgb)

    # Get corresponding pose
    if frame_idx >= len(poses_13):
        frame_idx = len(poses_13) - 1
        print(f"Frame index exceeds pose sequence, using frame {frame_idx}")

    pose_13 = poses_13[frame_idx]

    return {
        'image': image_pil,
        'pose_13': pose_13,
        'video_path': video_file_path,
        'frame_idx': frame_idx,
        'image_shape': frame_rgb.shape
    }

def map_17_to_13_joints(pose_17, mapping):
    """Convert a 17-joint pose representation to a 13-joint representation"""
    return pose_17[mapping]

def compute_mpjpe(pred_pose, gt_pose):
    """Compute Mean Per Joint Position Error"""
    pred_jax = jnp.array(pred_pose[None, ...])  # Add batch dimension
    gt_jax = jnp.array(gt_pose[None, ...])      # Add batch dimension
    return float(mpjpe_jax(pred_jax, gt_jax))

def visualize_poses(image, gt_pose, pred_pose, save_path=None):
    """
    Create a side-by-side visualization of ground truth and predicted poses

    Args:
        image: PIL Image
        gt_pose: Ground truth pose (13, 2)
        pred_pose: Predicted pose (13, 2)
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Pose Estimation Debug Visualization', fontsize=16)

    # Convert PIL to numpy
    image_np = np.array(image)

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Ground truth pose
    axes[1].imshow(image_np)
    axes[1].set_title('Ground Truth Pose')
    axes[1].axis('off')

    # Draw GT skeleton
    for connection in CONNECTIONS_13:
        start_idx, end_idx = connection
        if start_idx < len(gt_pose) and end_idx < len(gt_pose):
            start_point = gt_pose[start_idx]
            end_point = gt_pose[end_idx]
            axes[1].plot([start_point[0], end_point[0]],
                        [start_point[1], end_point[1]], 'g-', linewidth=2)

    # Draw GT keypoints
    for i, point in enumerate(gt_pose):
        axes[1].scatter(point[0], point[1], c='red', s=50, zorder=5)
        axes[1].text(point[0]+5, point[1]-5, str(i), fontsize=8, color='white',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.7))

    # Predicted pose
    axes[2].imshow(image_np)
    axes[2].set_title('Predicted Pose')
    axes[2].axis('off')

    # Draw predicted skeleton
    for connection in CONNECTIONS_13:
        start_idx, end_idx = connection
        if start_idx < len(pred_pose) and end_idx < len(pred_pose):
            start_point = pred_pose[start_idx]
            end_point = pred_pose[end_idx]
            axes[2].plot([start_point[0], end_point[0]],
                        [start_point[1], end_point[1]], 'b-', linewidth=2)

    # Draw predicted keypoints
    for i, point in enumerate(pred_pose):
        axes[2].scatter(point[0], point[1], c='yellow', s=50, zorder=5)
        axes[2].text(point[0]+5, point[1]-5, str(i), fontsize=8, color='white',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")

    plt.show()

def print_evaluation_summary(mpjpe_score, pred_pose, gt_pose):
    """Print detailed evaluation summary"""
    print("\n" + "=" * 60)
    print("POSE ESTIMATION EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Overall MPJPE: {mpjpe_score:.3f} pixels")

    # Per-joint errors
    joint_errors = np.sqrt(np.sum((pred_pose - gt_pose) ** 2, axis=1))

    print(f"\nPer-Joint Position Errors:")
    print("-" * 40)
    for i, (joint_name, error) in enumerate(zip(JOINT_NAMES_13, joint_errors)):
        print(f"{i:2d}. {joint_name:15s}: {error:6.2f} pixels")

    print(f"\nError Statistics:")
    print(f"  Min error:  {np.min(joint_errors):.2f} pixels")
    print(f"  Max error:  {np.max(joint_errors):.2f} pixels")
    print(f"  Mean error: {np.mean(joint_errors):.2f} pixels")
    print(f"  Std error:  {np.std(joint_errors):.2f} pixels")

    # Coordinate range analysis
    print(f"\nCoordinate Range Analysis:")
    print(f"  Ground Truth: [{np.min(gt_pose):.1f}, {np.max(gt_pose):.1f}]")
    print(f"  Predicted:    [{np.min(pred_pose):.1f}, {np.max(pred_pose):.1f}]")

    # Quality assessment
    if mpjpe_score < 20:
        quality = "Excellent"
    elif mpjpe_score < 50:
        quality = "Good"
    elif mpjpe_score < 100:
        quality = "Fair"
    else:
        quality = "Poor"

    print(f"\nPose Estimation Quality: {quality}")
    print("=" * 60)

def main():
    """Main debug function"""
    print("=" * 60)
    print("Debug Pose Visualization - JAX Implementation")
    print("=" * 60)

    try:
        # Initialize models
        print("Initializing models...")

        # Initialize JAX pose estimation model
        models_dir = os.path.join(root_dir, "models_tianle", "H36M", "RegressFlow", "seed_420")
        checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_pred")
        model, params, batch_stats = initialize_jax_models(checkpoint_path_jax)

        # Initialize YOLO human detector
        human_detector, device_torch = initialize_human_detector('cuda')

        print("Models initialized successfully!")

        # Load a single sample
        print("\nLoading H36M sample...")
        base_directory = os.path.join(root_dir, "datasets", "H36M", "extracted")

        sample = load_single_sample(
            base_directory=base_directory,
            subject='S1',
            action='Directions',
            camera='55011271',
            frame_idx=100
        )

        print(f"Sample loaded successfully!")
        print(f"  - Image shape: {sample['image_shape']}")
        print(f"  - Frame index: {sample['frame_idx']}")
        print(f"  - GT pose shape: {sample['pose_13'].shape}")

        # Run pose estimation
        print("\nRunning pose estimation...")

        pose_estimations = pose_estimation_2d(
            pil_image=sample['image'],
            model=model,
            params=params,
            batch_stats=batch_stats,
            human_detector=human_detector,
            device_torch=device_torch,
            threshold=0.8
        )

        if not pose_estimations:
            print("No humans detected! Using dummy pose for visualization.")
            pred_pose_17 = np.zeros((17, 2))
        else:
            print(f"Detected {len(pose_estimations)} human(s)")
            first_pose = pose_estimations[0]['keypoints']
            pred_pose_17 = np.array(first_pose)

        # Map from 17 joints to 13 joints
        pred_pose_13 = map_17_to_13_joints(pred_pose_17, JOINT_IDX_13_MODEL)
        gt_pose_13 = sample['pose_13']

        print(f"Pose estimation completed!")
        print(f"  - Predicted pose shape: {pred_pose_13.shape}")
        print(f"  - Predicted coordinate range: [{np.min(pred_pose_13):.1f}, {np.max(pred_pose_13):.1f}]")

        # Compute evaluation metrics
        mpjpe_score = compute_mpjpe(pred_pose_13, gt_pose_13)

        # Print evaluation summary
        print_evaluation_summary(mpjpe_score, pred_pose_13, gt_pose_13)

        # Create visualization
        print("\nCreating visualization...")
        save_path = f"debug_pose_visualization_frame_{sample['frame_idx']}.png"
        visualize_poses(sample['image'], gt_pose_13, pred_pose_13, save_path)

        print("\n" + "=" * 60)
        print("DEBUG COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"MPJPE: {mpjpe_score:.3f} pixels")
        print(f"Visualization saved: {save_path}")

        if mpjpe_score > 100:
            print("\nNote: High MPJPE suggests coordinate system issues.")
            print("Consider checking:")
            print("  - Coordinate range conversion (RegressFlow uses [-0.5, 0.5])")
            print("  - Joint mapping alignment")
            print("  - Image preprocessing pipeline")

    except Exception as e:
        print(f"\nError during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()