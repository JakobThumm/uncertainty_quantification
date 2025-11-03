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
import numpy as np
from spacepy.pycdf import CDF
import jax.numpy as jnp
import cv2
from PIL import Image


from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    process_frame_2d
)
from human_pose_pipeline.evaluation.pose_metrics import (
    mpjpe_jax
)
from human_pose_pipeline.utils.visualization import (
    visualize_poses_matplotlib
)
from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_NAMES_13,
    JOINT_IDX_13,
    JOINT_IDX_17,
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


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


def compute_mpjpe(pred_pose, gt_pose):
    """Compute Mean Per Joint Position Error"""
    pred_jax = jnp.array(pred_pose[None, ...])  # Add batch dimension
    gt_jax = jnp.array(gt_pose[None, ...])      # Add batch dimension
    return float(mpjpe_jax(pred_jax, gt_jax))


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

        # Initialize JAX pose estimation model with uncertainty estimation
        

        # Use 3-joint reduced model for faster OOD detection
        # Change to "finetuned_h36m_regressflow_with_unc" for full 17-joint model
        use_3joint_model = False  # Set to False to use full 17-joint model

        if use_3joint_model:
            models_dir = os.path.join(root_dir, "human_pose_pipeline/models/pose_estimation", "H36M", "RegressFlow_3joints", "seed_420")
            checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_pred_3joints")
            num_output_joints = 3
            print("Using 3-joint reduced model (nose, left wrist, right wrist) for faster inference")
        else:
            models_dir = os.path.join(root_dir, "human_pose_pipeline/models/pose_estimation", "H36M", "RegressFlow", "seed_420")
            checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_with_unc")
            num_output_joints = 17
            print("Using full RegressFlowWithAleatoric model for uncertainty estimation")

        pose_estimation_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path_jax)

        # Initialize YOLO human detector
        human_detector, device_torch = initialize_human_detector('cuda')

        print("Models initialized successfully!")

        # Load a single sample
        print("\nLoading H36M sample...")
        base_directory = os.path.join(root_dir, "datasets", "H36M", "extracted")

        sample = load_single_sample(
            base_directory=base_directory,
            subject='S1',
            action='Sitting 1',
            camera='55011271',
            frame_idx=0
        )

        print(f"Sample loaded successfully!")
        print(f"  - Image shape: {sample['image_shape']}")
        print(f"  - Frame index: {sample['frame_idx']}")
        print(f"  - GT pose shape: {sample['pose_13'].shape}")

        # Run pose estimation
        print("\nRunning pose estimation...")

        pose_predictions = process_frame_2d(
            frame=sample['image'],
            pose_estimation_jit_fn=pose_estimation_jit_fn,
            params=params,
            batch_stats=batch_stats,
            human_detector=human_detector,
            device_torch=device_torch,
            mirror_map=MIRROR_13_JOINT_MODEL_MAP,
            score_fn=None,  # No OOD scoring for now
            human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
            num_output_joints=num_output_joints  # Pass number of joints from model
        )
        # Take the first detected person
        pred_pose_13 = pose_predictions[0]['keypoints']
        pred_uncertainties = pose_predictions[0]['uncertainties']
        pred_covariances = pose_predictions[0]['covariance']

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
        save_path = f"visualizations/debug_pose_visualization_frame_{sample['frame_idx']}.png"
        visualize_poses_matplotlib(
            image=sample['image'],
            gt_pose=gt_pose_13,
            pred_pose=pred_pose_13,
            pred_uncertainties=pred_uncertainties,
            pred_covariances=pred_covariances,
            save_path=save_path,
            show_uncertainty=True
        )

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