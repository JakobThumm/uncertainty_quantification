#!/usr/bin/env python3
"""
3D Pose Estimation - YOLO Implementation

This implements YOLO-based 3D pose estimation for the H36M dataset:
- Load H36M dataset with two camera views
- Perform 2D pose estimation using YOLO11-pose with tracking
- Triangulate 3D poses with uncertainty propagation
- Visualize results with ground truth and estimated poses
- Reset tracking for each new sequence

Based on pose_estimation_3D_full_eval.py but using YOLO instead of custom models.
Note: YOLO does not support OOD detection.
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path for imports
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)

from ultralytics import YOLO

from human_pose_pipeline.utils.eval_utils import (
    evaluate_pose_prediction_scores_np,
    evaluate_uncertainty_coverage_with_covariance,
    print_coverage_stats,
    print_mpjpe_results,
    save_coverage_stats,
    save_mpjpe_results
)
from src.datasets.h36m import Human36mDatasetTwoCameras
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d_yolo,
    reset_yolo_tracking
)
from human_pose_pipeline.pose_estimation.triangulation_helper import (
    load_camera_parameters
)

from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
)


def main():
    """
    Main function for running 3D pose estimation on the Human3.6M dataset using YOLO.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='3D Pose Estimation with YOLO')
    parser.add_argument('--data_path', type=str, default='datasets/',
                       help='Path to datasets')
    parser.add_argument('--yolo_model', type=str, default='yolo11n-pose.pt',
                       help='YOLO model to use (e.g., yolo11n/s/m/l/x-pose.pt)')
    parser.add_argument('--split', type=str, default='validation',
                       help='train, validation, or test')
    parser.add_argument('--action', type=str, default='WalkingDog',
                       help='Action to visualize')
    parser.add_argument('--camera_ids', type=str, nargs=2,
                       default=['55011271', '60457274'],
                       help='Camera IDs')
    parser.add_argument('--max_sequences', type=int, default=10000000000,
                       help='Maximum number of sequences to process')
    parser.add_argument('--output_dir', type=str, default='results/pose_3d_yolo',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--enable_tracking', action='store_true',
                       help='Enable YOLO tracking within sequences')
    parser.add_argument('--confidence_threshold', type=float,
                       default=YOLO_CONFIDENCE_THRESHOLD,
                       help='Confidence threshold for pose detection')
    parser.add_argument('--camera_params_path', type=str, default=None,
                       help='Path to camera parameters JSON file')

    args = parser.parse_args()

    print("=" * 60)
    print("3D Pose Estimation - YOLO Implementation")
    print(f"Model: {args.yolo_model}")
    if args.enable_tracking:
        print("WITH TRACKING (resets per sequence)")
    print("=" * 60)

    # Configuration
    base_directory = os.path.join(root_dir, args.data_path, "H36M", "extracted")
    split = args.split
    action = args.action
    camera_ids = args.camera_ids
    batch_size = args.batch_size
    device = args.device

    # Initialize YOLO model
    print("\nInitializing YOLO pose model...")
    yolo_model = YOLO(args.yolo_model)

    # Move model to specified device
    if device == 'cuda' and torch.cuda.is_available():
        yolo_model.to('cuda')
        print(f"Model loaded on CUDA (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = 'cpu'
        print(f"Model loaded on CPU")

    print("YOLO model initialized successfully!")

    # Load camera parameters
    if args.camera_params_path is None:
        # Default path
        models_dir = os.path.join(
            root_dir,
            "human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420"
        )
        camera_parameters_path = os.path.join(models_dir, 'camera-parameters.json')
    else:
        camera_parameters_path = args.camera_params_path

    if not os.path.exists(camera_parameters_path):
        print(f"Error: Camera parameters file not found at {camera_parameters_path}")
        print("Please ensure the camera-parameters.json file is available")
        return

    print(f"Using camera parameters from: {camera_parameters_path}")

    # Create dataset
    print(f"\nLoading dataset from {base_directory}...")
    dataset = Human36mDatasetTwoCameras(
        base_directory=base_directory,
        split=split,
        camera_ids=camera_ids
    )

    if len(dataset) == 0:
        print("No data found. Please check the dataset path and camera IDs.")
        return

    print(f"Dataset loaded with {len(dataset)} samples")

    # Storage for results
    all_3d_points_list = []
    all_3d_covariances_list = []
    all_gt_points_list = []
    all_batch_sizes = []
    all_track_ids_list = []  # Store track IDs for analysis

    counter = 0
    for sample_idx, sample in enumerate(tqdm(dataset, desc="Processing sequences")):
        if counter >= args.max_sequences:
            break
        counter += 1

        all_camera_frames = sample['all_camera_frames']
        pose_sequence = sample['pose_sequence']
        subject = sample['subject']
        action = sample['action']

        # Reset tracking at the start of each new sequence
        if args.enable_tracking:
            reset_yolo_tracking(yolo_model)
            if sample_idx == 0 or sample_idx % 10 == 0:
                print(f"\n[Sequence {sample_idx}] Reset tracking for: "
                      f"{subject}/{action}")

        # Load camera parameters for this subject
        intrinsics, extrinsics, projection_matrices = load_camera_parameters(
            camera_parameters_path, subject, camera_ids
        )

        # Compute projection matrices
        P1 = projection_matrices[camera_ids[0]]
        P2 = projection_matrices[camera_ids[1]]
        P1 = torch.from_numpy(P1).to(device)
        P2 = torch.from_numpy(P2).to(device)
        projection_matrices_torch = [P1, P2]

        # Process frames
        frames_to_process = min(len(all_camera_frames[0]), len(pose_sequence))

        # Iterate through frames in batches
        for frame_idx in range(0, frames_to_process, batch_size):
            current_batch_size = min(batch_size, frames_to_process - frame_idx)
            all_batch_sizes.append(current_batch_size)

            # Process frames from both cameras
            left_frames = all_camera_frames[0][frame_idx:frame_idx + current_batch_size]
            right_frames = all_camera_frames[1][frame_idx:frame_idx + current_batch_size]

            # Interleave left and right frames for stereo processing
            interleaved_frames = [x for pair in zip(left_frames, right_frames)
                                for x in pair]

            # Run YOLO-based 3D pose estimation
            (points_3d, C_3d_all, ood_score, is_ood, human_detected,
             keypoints_2d, uncertainties_2d, covariance_xy) = process_frame_3d_yolo(
                frames=interleaved_frames,
                projection_matrices=projection_matrices_torch,
                yolo_pose_model=yolo_model,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                enable_tracking=args.enable_tracking,
                confidence_threshold=args.confidence_threshold,
                verbose=False,
                device=device
            )

            # Store results
            all_3d_points_list.append(points_3d.to('cpu').numpy())
            all_3d_covariances_list.append(C_3d_all.to('cpu').numpy())
            all_gt_points_list.append(
                pose_sequence[frame_idx:frame_idx + current_batch_size]
            )

            # Remove GPU tensors to free memory
            del points_3d, C_3d_all, ood_score, keypoints_2d
            del uncertainties_2d, covariance_xy

    # Convert to numpy arrays
    num_frames = sum(all_batch_sizes)
    print(f"\n3D pose estimation completed!")
    print(f"Processed {num_frames} frames from {counter} sequences")

    all_3d_points = np.zeros((num_frames, 13, 3))
    all_3d_covariances = np.zeros((num_frames, 13, 3, 3))
    all_gt_points = np.zeros((num_frames, 13, 3))

    index = 0
    for i, batch_size_i in enumerate(all_batch_sizes):
        all_3d_points[index:index + batch_size_i] = all_3d_points_list[i]
        all_3d_covariances[index:index + batch_size_i] = all_3d_covariances_list[i]
        all_gt_points[index:index + batch_size_i] = all_gt_points_list[i]
        index += batch_size_i

    # Filter out frames with zero predictions (no human detected)
    valid_mask = ~np.all(all_3d_points == 0, axis=(1, 2))
    num_invalid = np.sum(~valid_mask)
    if num_invalid > 0:
        print(f"\nFiltering {num_invalid} frames with no human detected "
              f"({100 * num_invalid / num_frames:.1f}%)")

    all_3d_points = all_3d_points[valid_mask]
    all_3d_covariances = all_3d_covariances[valid_mask]
    all_gt_points = all_gt_points[valid_mask]
    num_frames = all_3d_points.shape[0]

    print(f"Valid frames for evaluation: {num_frames}")

    # Reshape for evaluation (add sequence dimension)
    all_3d_points = all_3d_points.reshape(1, num_frames, 13, 3)
    all_gt_points = all_gt_points.reshape(1, num_frames, 13, 3)

    # Evaluate pose prediction
    print("\nEvaluating pose prediction...")
    (mpjpe, std, per_time_errors, per_time_std,
     per_joint_errors, per_joint_std) = evaluate_pose_prediction_scores_np(
        predictions=all_3d_points,
        targets=all_gt_points,
    )

    # Evaluate uncertainty coverage
    print("Evaluating uncertainty coverage...")
    coverage_stats, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=all_3d_points,
        true_poses=all_gt_points,
        cov_matrices=all_3d_covariances
    )

    # Print and save results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print_mpjpe_results(mpjpe, per_time_errors, per_joint_errors, print_per_time_errors=False)
    print_coverage_stats(coverage_stats, print_per_time_stats=False)

    # Create output directory with model-specific subdirectory
    model_name = args.yolo_model.replace('.pt', '')
    tracking_suffix = "_tracking" if args.enable_tracking else "_notracking"
    output_subdir = os.path.join(args.output_dir, f"{model_name}{tracking_suffix}")
    os.makedirs(output_subdir, exist_ok=True)

    # Save results
    save_mpjpe_results(
        mpjpe, per_time_errors, per_joint_errors,
        split=split,
        output_dir=output_subdir
    )
    save_coverage_stats(
        coverage_stats,
        split=split,
        output_dir=output_subdir
    )

    print(f"\nResults saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
