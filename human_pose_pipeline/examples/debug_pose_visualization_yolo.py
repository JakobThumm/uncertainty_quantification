#!/usr/bin/env python3
"""
Debug YOLO Pose Visualization Script

This script provides a test for the new YOLO-based pose estimation:
- Loads a single H36M image and ground truth pose
- Performs 2D pose estimation using YOLO11-pose
- Visualizes the results side-by-side with ground truth
- Computes and displays evaluation metrics
- Tests tracking functionality

Similar to debug_pose_visualization.py but using YOLO instead of custom JAX models.
"""

import os
import sys
import numpy as np
from spacepy.pycdf import CDF
import jax.numpy as jnp
import cv2
from PIL import Image
import torch

from ultralytics import YOLO
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_2d_yolo,
    reset_yolo_tracking
)
from human_pose_pipeline.evaluation.pose_metrics import mpjpe_jax
from human_pose_pipeline.utils.visualization import visualize_poses_matplotlib
from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_NAMES_13,
    JOINT_IDX_13,
    JOINT_IDX_17,
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD
)


# Add parent directory to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)


def load_single_sample(base_directory, subject='S1', action='Directions',
                       camera='55011271', frame_idx=100):
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
        raise FileNotFoundError(
            f"No pose file found for {subject}/{action}/{camera}"
        )

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
        print(f"Requested frame exceeds video length, using frame {frame_idx}")

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


def print_evaluation_summary(mpjpe_score, pred_pose, gt_pose, confidence_scores=None):
    """Print detailed evaluation summary"""
    print("\n" + "=" * 60)
    print("YOLO POSE ESTIMATION EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Overall MPJPE: {mpjpe_score:.3f} pixels")

    # Per-joint errors
    joint_errors = np.sqrt(np.sum((pred_pose - gt_pose) ** 2, axis=1))

    print(f"\nPer-Joint Position Errors:")
    print("-" * 50)
    for i, (joint_name, error) in enumerate(zip(JOINT_NAMES_13, joint_errors)):
        conf_str = ""
        if confidence_scores is not None:
            conf_str = f" (conf: {confidence_scores[i]:.3f})"
        print(f"{i:2d}. {joint_name:15s}: {error:6.2f} pixels{conf_str}")

    print(f"\nError Statistics:")
    print(f"  Min error:  {np.min(joint_errors):.2f} pixels")
    print(f"  Max error:  {np.max(joint_errors):.2f} pixels")
    print(f"  Mean error: {np.mean(joint_errors):.2f} pixels")
    print(f"  Std error:  {np.std(joint_errors):.2f} pixels")

    if confidence_scores is not None:
        print(f"\nConfidence Statistics:")
        print(f"  Min confidence:  {np.min(confidence_scores):.3f}")
        print(f"  Max confidence:  {np.max(confidence_scores):.3f}")
        print(f"  Mean confidence: {np.mean(confidence_scores):.3f}")

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


def test_tracking(yolo_model, sample, mirror_map, device):
    """Test YOLO tracking functionality with multiple frames"""
    print("\n" + "=" * 60)
    print("TESTING TRACKING FUNCTIONALITY")
    print("=" * 60)

    # Load 5 consecutive frames
    cap = cv2.VideoCapture(sample['video_path'])
    start_idx = sample['frame_idx']
    frames = []

    for i in range(5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx + i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame_rgb).to(device))

    cap.release()

    if len(frames) < 2:
        print("Not enough frames for tracking test")
        return

    print(f"Loaded {len(frames)} consecutive frames")
    print("\nProcessing frames with tracking enabled...")

    # Process frames sequentially with tracking
    track_ids = []
    for i, frame in enumerate(frames):
        results = process_frame_2d_yolo(
            frames=frame.unsqueeze(0),
            yolo_pose_model=yolo_model,
            mirror_map=mirror_map,
            enable_tracking=True,
            confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
            verbose=False,
            device=device
        )
        track_id = results['track_id'][0].item()
        track_ids.append(track_id)
        detected = results['mask'][0].item()
        print(f"  Frame {i}: Track ID = {track_id}, Detected = {detected}")

    # Check consistency
    if all(tid == track_ids[0] and tid != -1 for tid in track_ids):
        print("\n✓ Tracking SUCCESSFUL: Same person tracked across all frames")
    else:
        print("\n✗ Tracking issue: IDs changed across frames")
        print(f"  Track IDs: {track_ids}")

    # Test reset functionality
    print("\nTesting tracking reset...")
    reset_yolo_tracking(yolo_model)

    # Process first frame again after reset
    results_after_reset = process_frame_2d_yolo(
        frames=frames[0].unsqueeze(0),
        yolo_pose_model=yolo_model,
        mirror_map=mirror_map,
        enable_tracking=True,
        confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
        verbose=False,
        device=device
    )
    new_track_id = results_after_reset['track_id'][0].item()
    print(f"  Track ID after reset: {new_track_id}")

    if new_track_id != -1:
        print("✓ Tracking reset SUCCESSFUL")
    else:
        print("✗ Tracking reset issue")

    print("=" * 60)


def main():
    """Main debug function"""
    print("=" * 60)
    print("Debug YOLO Pose Visualization")
    print("=" * 60)

    try:
        # Configuration
        yolo_model_name = "yolo26n-pose.pt"  # Options: yolo26n/s/m/l/x-pose.pt (auto-downloaded)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        enable_tracking_test = True

        print(f"\nConfiguration:")
        print(f"  YOLO Model: {yolo_model_name}")
        print(f"  Device: {device}")
        print(f"  Tracking test: {enable_tracking_test}")

        # Initialize YOLO model
        print("\nInitializing YOLO pose model...")
        yolo_model = YOLO(yolo_model_name)

        # Move model to specified device
        if device == 'cuda' and torch.cuda.is_available():
            yolo_model.to('cuda')
            print(f"Model loaded successfully on CUDA (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print(f"Model loaded successfully on CPU")

        # Load a single sample
        print("\nLoading H36M sample...")
        base_directory = os.path.join(root_dir, "datasets", "H36M", "extracted")

        sample = load_single_sample(
            base_directory=base_directory,
            subject='S1',
            action='Sitting 1',
            camera='55011271',
            frame_idx=100
        )

        print(f"Sample loaded successfully!")
        print(f"  - Image shape: {sample['image_shape']}")
        print(f"  - Frame index: {sample['frame_idx']}")
        print(f"  - GT pose shape: {sample['pose_13'].shape}")

        # Run pose estimation
        print("\nRunning YOLO pose estimation...")

        pose_predictions = process_frame_2d_yolo(
            frames=sample['image'],
            yolo_pose_model=yolo_model,
            mirror_map=MIRROR_13_JOINT_MODEL_MAP,
            enable_tracking=False,  # Disable tracking for single image
            confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
            verbose=True,
            device=device
        )

        # Extract predictions
        pred_pose_13 = pose_predictions['keypoints'][0].cpu().numpy()
        confidence_scores = pose_predictions['confidence'][0].cpu().numpy()
        detected = pose_predictions['mask'][0].item()

        gt_pose_13 = sample['pose_13']

        if not detected:
            print("\n✗ WARNING: No human detected in the image!")
            print("Try adjusting confidence_threshold or check input image")
            return

        print(f"✓ Pose estimation completed!")
        print(f"  - Predicted pose shape: {pred_pose_13.shape}")
        print(f"  - Coordinate range: [{np.min(pred_pose_13):.1f}, "
              f"{np.max(pred_pose_13):.1f}]")
        print(f"  - Mean confidence: {np.mean(confidence_scores):.3f}")

        # Extract and display sigma uncertainty info
        pred_uncertainties_raw = pose_predictions['uncertainties'][0].cpu().numpy()  # [13, 2]
        print(f"  - Mean sigma_x: {np.mean(pred_uncertainties_raw[:, 0]):.2f} px")
        print(f"  - Mean sigma_y: {np.mean(pred_uncertainties_raw[:, 1]):.2f} px")

        # Compute evaluation metrics
        mpjpe_score = compute_mpjpe(pred_pose_13, gt_pose_13)

        # Print evaluation summary
        print_evaluation_summary(
            mpjpe_score, pred_pose_13, gt_pose_13, confidence_scores
        )

        # Create visualization
        print("\nCreating visualization...")
        os.makedirs("visualizations", exist_ok=True)
        save_path = (f"visualizations/debug_yolo_pose_visualization_"
                    f"frame_{sample['frame_idx']}.png")

        # Extract sigma_x, sigma_y uncertainties from the custom Pose26 model output
        pred_uncertainties = pose_predictions['uncertainties'][0].cpu().numpy()  # [13, 2]
        pred_covariances = np.zeros(13)  # No x-y covariance from the sigma head

        visualize_poses_matplotlib(
            image=sample['image'],
            gt_pose=gt_pose_13,
            pred_pose=pred_pose_13,
            pred_uncertainties=pred_uncertainties,
            pred_covariances=pred_covariances,
            save_path=save_path,
            show_uncertainty=True  # Custom Pose26 model provides sigma_x, sigma_y
        )

        print("\n" + "=" * 60)
        print("SINGLE IMAGE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"MPJPE: {mpjpe_score:.3f} pixels")
        print(f"Visualization saved: {save_path}")

        # Test tracking functionality
        if enable_tracking_test:
            test_tracking(yolo_model, sample, MIRROR_13_JOINT_MODEL_MAP, device)

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during debug: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
