#!/usr/bin/env python3
"""
Debug RGBD 2D Pose Visualization Script - YOLO Implementation

This script provides pose estimation on RGBD data using YOLO:
- Loads RGB images from the HumanRGBD dataset
- Performs 2D pose estimation using YOLO11-pose
- Visualizes the results with optional tracking
- Supports both single frame and batch processing

Usage:
    # Process single frame
    python debug_rgbd_2d_visualization_yolo.py --frame_idx 10

    # Process all frames
    python debug_rgbd_2d_visualization_yolo.py --all --save_dir visualizations/rgbd_2d_yolo_all

    # With tracking enabled
    python debug_rgbd_2d_visualization_yolo.py --all --enable_tracking
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ultralytics import YOLO
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_2d_yolo,
    reset_yolo_tracking
)
from human_pose_pipeline.utils.visualization import (
    visualize_poses_matplotlib
)
from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_NAMES_13,
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD
)
from src.datasets.human_rgbd import HumanRGBDDataset


# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def print_pose_summary(pred_pose, confidence_scores):
    """Print detailed summary of predicted pose and confidence scores."""
    print("\n" + "=" * 60)
    print("YOLO POSE ESTIMATION SUMMARY")
    print("=" * 60)

    print(f"Predicted pose shape: {pred_pose.shape}")
    print(f"Confidence scores shape: {confidence_scores.shape}")

    # Per-joint statistics
    print(f"\nPer-Joint Predictions:")
    print("-" * 55)
    for i, joint_name in enumerate(JOINT_NAMES_13):
        x, y = pred_pose[i]
        conf = confidence_scores[i]
        print(f"{i:2d}. {joint_name:15s}: ({x:6.1f}, {y:6.1f}) conf={conf:.3f}")

    # Coordinate range analysis
    print(f"\nCoordinate Statistics:")
    print(f"  X range: [{np.min(pred_pose[:, 0]):.1f}, {np.max(pred_pose[:, 0]):.1f}]")
    print(f"  Y range: [{np.min(pred_pose[:, 1]):.1f}, {np.max(pred_pose[:, 1]):.1f}]")

    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {np.mean(confidence_scores):.3f}")
    print(f"  Min confidence:  {np.min(confidence_scores):.3f}")
    print(f"  Max confidence:  {np.max(confidence_scores):.3f}")

    print("=" * 60)


def process_single_frame(frame_idx, dataset, yolo_model, mirror_map, save_dir,
                         enable_tracking=False, device='cpu', verbose=True):
    """
    Process a single frame and save visualization.

    Returns:
        dict: Results containing success status, detection info, and any error message
    """
    result = {
        'frame_idx': frame_idx,
        'success': False,
        'human_detected': False,
        'track_id': -1,
        'mean_confidence': 0.0,
        'error': None,
        'filename': None
    }

    try:
        sample = dataset[frame_idx]
        result['filename'] = sample['filename']

        # Convert to PIL Image
        image_pil = Image.fromarray(sample['color_raw'])

        # Run YOLO pose estimation
        pose_predictions = process_frame_2d_yolo(
            frames=image_pil,
            yolo_pose_model=yolo_model,
            mirror_map=mirror_map,
            enable_tracking=enable_tracking,
            confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
            verbose=False,
            device=device
        )

        detected = pose_predictions['mask'][0].item()
        if not detected:
            result['error'] = "No human detected"
            return result

        result['human_detected'] = True
        result['track_id'] = pose_predictions['track_id'][0].item()

        # Extract predictions
        pred_pose_13 = pose_predictions['keypoints'][0].cpu().numpy()
        confidence_scores = pose_predictions['confidence'][0].cpu().numpy()
        result['mean_confidence'] = float(np.mean(confidence_scores))

        # YOLO doesn't provide uncertainties, use placeholder
        pred_uncertainties = np.ones_like(pred_pose_13) * 10.0
        pred_covariances = np.zeros(13)

        # Save visualization
        save_path = os.path.join(save_dir, f"frame_{frame_idx:04d}.png")
        visualize_poses_matplotlib(
            image=image_pil,
            gt_pose=None,
            pred_pose=pred_pose_13,
            pred_uncertainties=pred_uncertainties,
            pred_covariances=pred_covariances,
            save_path=save_path,
            show_uncertainty=False  # YOLO doesn't provide uncertainty
        )

        result['success'] = True

        if verbose:
            track_str = f" (track_id={result['track_id']})" if enable_tracking else ""
            print(f"Frame {frame_idx}: conf={result['mean_confidence']:.3f}{track_str} -> {save_path}")

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"Frame {frame_idx}: ERROR - {e}")

    return result


def main():
    """Main debug function for YOLO 2D pose estimation on RGBD data."""
    parser = argparse.ArgumentParser(description="YOLO 2D pose estimation on RGBD data")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to process (ignored if --all)")
    parser.add_argument("--all", action="store_true", help="Process all frames in the dataset")
    parser.add_argument("--data_path", type=str, default=None, help="Path to RGBD dataset")
    parser.add_argument("--save_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cuda/cpu)")
    parser.add_argument("--yolo_model", type=str, default="yolo11n-pose.pt",
                        help="YOLO model name (e.g., yolo11n-pose.pt, yolo11s-pose.pt)")
    parser.add_argument("--enable_tracking", action="store_true", help="Enable YOLO tracking")
    args = parser.parse_args()

    print("=" * 60)
    print("Debug RGBD 2D Pose Visualization - YOLO Implementation")
    print("=" * 60)

    try:
        # Determine device
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            args.device = 'cpu'

        print(f"\nConfiguration:")
        print(f"  YOLO Model: {args.yolo_model}")
        print(f"  Device: {args.device}")
        print(f"  Tracking: {args.enable_tracking}")

        # Initialize YOLO model
        print("\nInitializing YOLO pose model...")
        yolo_model = YOLO(args.yolo_model)

        if args.device == 'cuda':
            yolo_model.to('cuda')
            print(f"Model loaded on CUDA (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("Model loaded on CPU")

        # Load RGBD dataset
        print("\nLoading RGBD dataset...")
        data_path = args.data_path or os.path.join(root_dir, "datasets", "rgbd_test")

        dataset = HumanRGBDDataset(
            base_directory=data_path,
        )

        # Create save directory
        if args.all:
            save_dir = os.path.join(args.save_dir, "rgbd_2d_yolo_all")
        else:
            save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        if args.all:
            # Process all frames
            print(f"\nProcessing all {len(dataset)} frames...")
            print(f"Saving to: {save_dir}")

            # Reset tracking at the start
            if args.enable_tracking:
                reset_yolo_tracking(yolo_model)

            results = []
            for frame_idx in tqdm(range(len(dataset)), desc="Processing frames"):
                result = process_single_frame(
                    frame_idx=frame_idx,
                    dataset=dataset,
                    yolo_model=yolo_model,
                    mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                    save_dir=save_dir,
                    enable_tracking=args.enable_tracking,
                    device=args.device,
                    verbose=False
                )
                results.append(result)

            # Print summary
            successful = sum(1 for r in results if r['success'])
            detected = sum(1 for r in results if r['human_detected'])
            failed = sum(1 for r in results if r['error'] is not None)
            avg_confidence = np.mean([r['mean_confidence'] for r in results if r['human_detected']])

            print("\n" + "=" * 60)
            print("BATCH PROCESSING COMPLETED")
            print("=" * 60)
            print(f"Total frames:        {len(dataset)}")
            print(f"Successful:          {successful}")
            print(f"Human detected:      {detected}")
            print(f"Failed:              {failed}")
            print(f"Avg confidence:      {avg_confidence:.3f}")
            print(f"Output directory:    {save_dir}")

            # Track ID statistics if tracking enabled
            if args.enable_tracking:
                track_ids = [r['track_id'] for r in results if r['human_detected']]
                unique_tracks = len(set(track_ids))
                print(f"Unique track IDs:    {unique_tracks}")
                if unique_tracks == 1 and track_ids[0] != -1:
                    print(f"✓ Consistent tracking: same person tracked across all frames")

            # List failed frames if any
            if failed > 0:
                print(f"\nFailed frames:")
                for r in results:
                    if r['error'] is not None:
                        print(f"  Frame {r['frame_idx']}: {r['error']}")

        else:
            # Process single frame
            if args.frame_idx >= len(dataset):
                print(f"Frame index {args.frame_idx} exceeds dataset length {len(dataset)}, using frame 0")
                args.frame_idx = 0

            sample = dataset[args.frame_idx]
            print(f"Loaded sample: {sample['filename']}")
            print(f"  - Color shape: {sample['color_raw'].shape}")
            print(f"  - Depth shape: {sample['depth_raw'].shape}")

            # Convert to PIL Image
            image_pil = Image.fromarray(sample['color_raw'])

            # Run YOLO pose estimation
            print("\nRunning YOLO pose estimation...")

            pose_predictions = process_frame_2d_yolo(
                frames=image_pil,
                yolo_pose_model=yolo_model,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                enable_tracking=args.enable_tracking,
                confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
                verbose=True,
                device=args.device
            )

            detected = pose_predictions['mask'][0].item()
            if not detected:
                print("\n✗ WARNING: No human detected in the image!")
                print("Try adjusting confidence_threshold or check input image")
                return

            # Extract predictions
            pred_pose_13 = pose_predictions['keypoints'][0].cpu().numpy()
            confidence_scores = pose_predictions['confidence'][0].cpu().numpy()

            print(f"✓ Pose estimation completed!")

            # Print summary
            print_pose_summary(pred_pose_13, confidence_scores)

            # Create visualization
            print("\nCreating visualization...")
            save_path = os.path.join(save_dir, f"debug_rgbd_2d_yolo_frame_{args.frame_idx}.png")

            # YOLO doesn't provide uncertainties
            pred_uncertainties = np.ones_like(pred_pose_13) * 10.0
            pred_covariances = np.zeros(13)

            visualize_poses_matplotlib(
                image=image_pil,
                gt_pose=None,
                pred_pose=pred_pose_13,
                pred_uncertainties=pred_uncertainties,
                pred_covariances=pred_covariances,
                save_path=save_path,
                show_uncertainty=False
            )

            print("\n" + "=" * 60)
            print("DEBUG COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Mean confidence: {np.mean(confidence_scores):.3f}")
            print(f"Visualization saved: {save_path}")

    except Exception as e:
        print(f"\nError during debug: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
