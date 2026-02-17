#!/usr/bin/env python3
"""
Debug RGBD 2D Pose Visualization Script - JAX Implementation

This script provides a simplified pose estimation pipeline for RGBD data:
- Loads RGB images from the HumanRGBD dataset
- Detects humans using YOLO
- Performs 2D pose estimation using JAX RegressFlow model
- Visualizes the results with uncertainty ellipses

Based on debug_pose_visualization.py but adapted for RGBD data without ground truth.

Usage:
    # Process single frame
    python debug_rgbd_2d_visualization.py --frame_idx 10

    # Process all frames
    python debug_rgbd_2d_visualization.py --all --save_dir visualizations/rgbd_2d_all
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    process_frame_2d
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


def print_pose_summary(pred_pose, pred_uncertainties):
    """Print detailed summary of predicted pose and uncertainties."""
    print("\n" + "=" * 60)
    print("POSE ESTIMATION SUMMARY")
    print("=" * 60)

    print(f"Predicted pose shape: {pred_pose.shape}")
    print(f"Uncertainties shape: {pred_uncertainties.shape}")

    # Per-joint statistics
    print(f"\nPer-Joint Predictions:")
    print("-" * 50)
    for i, joint_name in enumerate(JOINT_NAMES_13):
        x, y = pred_pose[i]
        ux, uy = pred_uncertainties[i]
        print(f"{i:2d}. {joint_name:15s}: ({x:6.1f}, {y:6.1f}) +/- ({ux:5.2f}, {uy:5.2f})")

    # Coordinate range analysis
    print(f"\nCoordinate Statistics:")
    print(f"  X range: [{np.min(pred_pose[:, 0]):.1f}, {np.max(pred_pose[:, 0]):.1f}]")
    print(f"  Y range: [{np.min(pred_pose[:, 1]):.1f}, {np.max(pred_pose[:, 1]):.1f}]")

    # Uncertainty statistics
    print(f"\nUncertainty Statistics:")
    mean_unc = np.mean(pred_uncertainties)
    max_unc = np.max(pred_uncertainties)
    min_unc = np.min(pred_uncertainties)
    print(f"  Mean uncertainty: {mean_unc:.2f} pixels")
    print(f"  Min uncertainty:  {min_unc:.2f} pixels")
    print(f"  Max uncertainty:  {max_unc:.2f} pixels")

    print("=" * 60)


def process_single_frame(frame_idx, dataset, pose_estimation_jit_fn, params, batch_stats,
                         human_detector, device_torch, save_dir, num_output_joints=17, verbose=True):
    """
    Process a single frame and save visualization.

    Returns:
        dict: Results containing success status and any error message
    """
    result = {
        'frame_idx': frame_idx,
        'success': False,
        'human_detected': False,
        'error': None,
        'filename': None
    }

    try:
        sample = dataset[frame_idx]
        result['filename'] = sample['filename']

        # Convert to PIL Image
        image_pil = Image.fromarray(sample['color_raw'])

        # Run pose estimation
        pose_predictions = process_frame_2d(
            frame=image_pil,
            pose_estimation_jit_fn=pose_estimation_jit_fn,
            params=params,
            batch_stats=batch_stats,
            human_detector=human_detector,
            device_torch=device_torch,
            mirror_map=MIRROR_13_JOINT_MODEL_MAP,
            score_fn=None,
            human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
            num_output_joints=num_output_joints
        )

        if len(pose_predictions) == 0 or pose_predictions[0]['keypoints'] is None:
            result['error'] = "No human detected"
            return result

        result['human_detected'] = True

        # Extract predictions
        pred_pose_13 = pose_predictions[0]['keypoints']
        pred_uncertainties = pose_predictions[0]['uncertainties']
        pred_covariances = pose_predictions[0]['covariance']

        # Save visualization
        save_path = os.path.join(save_dir, f"frame_{frame_idx:04d}.png")
        visualize_poses_matplotlib(
            image=image_pil,
            gt_pose=None,
            pred_pose=pred_pose_13,
            pred_uncertainties=pred_uncertainties,
            pred_covariances=pred_covariances,
            save_path=save_path,
            show_uncertainty=True
        )

        result['success'] = True

        if verbose:
            print(f"Frame {frame_idx}: saved to {save_path}")

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"Frame {frame_idx}: ERROR - {e}")

    return result


def main():
    """Main debug function for 2D pose estimation on RGBD data."""
    parser = argparse.ArgumentParser(description="Debug 2D pose estimation on RGBD data")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to process (ignored if --all)")
    parser.add_argument("--all", action="store_true", help="Process all frames in the dataset")
    parser.add_argument("--data_path", type=str, default=None, help="Path to RGBD dataset")
    parser.add_argument("--save_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cuda/cpu)")
    args = parser.parse_args()

    print("=" * 60)
    print("Debug RGBD 2D Pose Visualization - JAX Implementation")
    print("=" * 60)

    try:
        # Initialize models
        print("Initializing models...")

        # Use full RegressFlow model with uncertainty
        models_dir = os.path.join(root_dir, "human_pose_pipeline/models/pose_estimation", "H36M", "RegressFlow", "seed_420")
        checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_with_unc")
        num_output_joints = 17
        print("Using RegressFlowWithAleatoric model for uncertainty estimation")

        pose_estimation_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path_jax)

        # Initialize YOLO human detector
        human_detector, device_torch = initialize_human_detector(args.device)

        print("Models initialized successfully!")

        # Load RGBD dataset
        print("\nLoading RGBD dataset...")
        data_path = args.data_path or os.path.join(root_dir, "datasets", "rgbd_test")

        dataset = HumanRGBDDataset(
            base_directory=data_path,
        )

        # Create save directory
        if args.all:
            save_dir = os.path.join(args.save_dir, "rgbd_2d_all")
        else:
            save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)

        if args.all:
            # Process all frames
            print(f"\nProcessing all {len(dataset)} frames...")
            print(f"Saving to: {save_dir}")

            results = []
            for frame_idx in tqdm(range(len(dataset)), desc="Processing frames"):
                result = process_single_frame(
                    frame_idx=frame_idx,
                    dataset=dataset,
                    pose_estimation_jit_fn=pose_estimation_jit_fn,
                    params=params,
                    batch_stats=batch_stats,
                    human_detector=human_detector,
                    device_torch=device_torch,
                    save_dir=save_dir,
                    num_output_joints=num_output_joints,
                    verbose=False
                )
                results.append(result)

            # Print summary
            successful = sum(1 for r in results if r['success'])
            detected = sum(1 for r in results if r['human_detected'])
            failed = sum(1 for r in results if r['error'] is not None)

            print("\n" + "=" * 60)
            print("BATCH PROCESSING COMPLETED")
            print("=" * 60)
            print(f"Total frames:      {len(dataset)}")
            print(f"Successful:        {successful}")
            print(f"Human detected:    {detected}")
            print(f"Failed:            {failed}")
            print(f"Output directory:  {save_dir}")

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

            # Convert to PIL Image for pose estimation
            image_pil = Image.fromarray(sample['color_raw'])

            # Run pose estimation
            print("\nRunning 2D pose estimation...")

            pose_predictions = process_frame_2d(
                frame=image_pil,
                pose_estimation_jit_fn=pose_estimation_jit_fn,
                params=params,
                batch_stats=batch_stats,
                human_detector=human_detector,
                device_torch=device_torch,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                score_fn=None,
                human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
                num_output_joints=num_output_joints
            )

            if len(pose_predictions) == 0 or pose_predictions[0]['keypoints'] is None:
                print("No human detected in frame!")
                return

            # Extract predictions
            pred_pose_13 = pose_predictions[0]['keypoints']
            pred_uncertainties = pose_predictions[0]['uncertainties']
            pred_covariances = pose_predictions[0]['covariance']

            print(f"Pose estimation completed!")

            # Print summary
            print_pose_summary(pred_pose_13, pred_uncertainties)

            # Create visualization
            print("\nCreating visualization...")
            save_path = os.path.join(save_dir, f"debug_rgbd_2d_frame_{args.frame_idx}.png")

            visualize_poses_matplotlib(
                image=image_pil,
                gt_pose=None,
                pred_pose=pred_pose_13,
                pred_uncertainties=pred_uncertainties,
                pred_covariances=pred_covariances,
                save_path=save_path,
                show_uncertainty=True
            )

            print("\n" + "=" * 60)
            print("DEBUG COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Visualization saved: {save_path}")

    except Exception as e:
        print(f"\nError during debug: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
