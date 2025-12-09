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
import torch
from tqdm import tqdm

from human_pose_pipeline.utils.eval_utils import evaluate_pose_prediction_scores_np
from src.datasets.h36m import Human36mDatasetSequenceTwoCameras, SPLIT
from src.ood_scores.lm_lanczos import load_score_functions
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
)
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d
)
from human_pose_pipeline.pose_estimation.triangulation_helper import (
    load_camera_parameters
)

from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
    OOD_THRESHOLD
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


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
    parser.add_argument('--model_save_path', type=str, default='human_pose_pipeline/models/pose_estimation', help='Path to saved models')
    parser.add_argument('--run_name', type=str, default='finetuned_h36m_regressflow_with_unc', help='Model run name')
    parser.add_argument('--ood_threshold', type=float, default=OOD_THRESHOLD, help='OOD threshold')
    parser.add_argument('--split', type=str, default='validation', help='train, validation, or test')
    parser.add_argument('--action', type=str, default='WalkingDog', help='Action to visualize')
    parser.add_argument('--camera_ids', type=str, nargs=2, default=['55011271', '60457274'], help='Camera IDs')
    parser.add_argument('--max_sequences', type=int, default=10000000000, help='Maximum number of sequences to process')
    parser.add_argument('--enable_ood', action='store_true', help='Enable OOD detection on left camera')
    parser.add_argument('--output_dir', type=str, default='results/pose_3d', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    print("=" * 60)
    print("3D Pose Estimation - JAX Implementation")
    if args.enable_ood:
        print("WITH OOD DETECTION (left camera only)")
    print("=" * 60)

    # Configuration
    base_directory = os.path.join(root_dir, args.data_path, "H36M", "extracted")
    split = args.split
    action = args.action
    camera_ids = args.camera_ids
    batch_size = args.batch_size
    device = args.device

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

    subject = SPLIT[args.split][0]
    intrinsics, extrinsics, projection_matrices = load_camera_parameters(camera_parameters_path, subject, camera_ids)

    # Compute projection matrices
    P1 = projection_matrices[camera_ids[0]]
    P2 = projection_matrices[camera_ids[1]]
    P1 = torch.from_numpy(P1).to(device)
    P2 = torch.from_numpy(P2).to(device)
    projection_matrices = [P1, P2]

    # Create dataset
    dataset = Human36mDatasetSequenceTwoCameras(
        base_directory=base_directory,
        split=split,
        camera_ids=camera_ids
    )

    if len(dataset) == 0:
        print("No data found. Please check the dataset path and camera IDs.")
        return

    print(f"Dataset loaded with {len(dataset)} samples")
    counter = 0

    # Get a sample from the dataset
    for sample in tqdm(dataset):
        if counter > args.max_sequences:
            break
        counter += 1
        all_camera_frames = sample['all_camera_frames']
        pose_sequence = sample['pose_sequence']

        # Process a limited number of frames for testing
        frames_to_process = min(args.max_sequences, len(all_camera_frames[0]))
        all_3d_points = []
        all_3d_covariances = []
        all_gt_points = []
        all_ood_scores = []  # Store OOD scores from left camera
        all_is_ood = []  # Store OOD classifications

        if args.enable_ood and score_fn is not None:
            print("OOD detection will be performed on LEFT camera (camera 0) only")

        # for frame_idx in range(frames_to_process):
        # Iterate through frames in a batched manner
        for frame_idx in range(0, frames_to_process, batch_size):
            current_batch_size = min(batch_size, frames_to_process - frame_idx)
            # Process frames from both cameras    
            left_frames = all_camera_frames[0][frame_idx:frame_idx + current_batch_size]
            right_frames = all_camera_frames[1][frame_idx:frame_idx + current_batch_size]
            # Append right frames to the left frames list
            both_frames = left_frames + right_frames

            points_3d, C_3d_all, ood_score, is_ood = process_frame_3d(
                frames=both_frames,
                projection_matrices=projection_matrices,
                pose_estimation_jit_fn=pose_estimation_jit_fn,
                params=params,
                batch_stats=batch_stats,
                human_detector=human_detector,
                device_torch=device_torch,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                score_fn=score_fn,
                human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
                ood_threshold=args.ood_threshold,
                verbose=False,
                device=device
            )

            # Store OOD information
            all_ood_scores.append(ood_score)
            all_is_ood.append(is_ood)
            all_3d_points.append(points_3d)
            all_3d_covariances.append(C_3d_all)
            all_gt_points.append(pose_sequence[frame_idx])

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
    all_gt_points = np.array(all_gt_points)
    num_frames = all_3d_points.shape[0]
    mpjpe, std, per_time_errors, per_time_std, per_joint_errors, per_joint_std = evaluate_pose_prediction_scores_np(
        predictions=np.reshape(all_3d_points, [num_frames, 1, 13, 3]),
        targets=np.reshape(pose_sequence[:num_frames], [num_frames, 1, 13, 3]),
    )
    print(f"MPJPE = {mpjpe:.2f}")
    print(f"per_joint_errors = {per_joint_errors}")


if __name__ == "__main__":
    main()
