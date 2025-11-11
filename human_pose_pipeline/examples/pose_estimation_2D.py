#!/usr/bin/env python3
"""
2D Pose Estimation - JAX Implementation

This implements the JAX version of Marian's 2D_Pose_Estimation.py:
- Load H36M dataset with pose sequences and video frames
- Detect humans using YOLO
- Perform 2D pose estimation using JAX RegressFlow model
- Evaluate pose estimation accuracy using Mahalanobis distance
- Visualize results with ground truth and estimated poses

Based on marian_code/Experiment2/2D_Pose_Estimation.py but adapted for JAX.
"""

import os
import numpy as np
from scipy.stats import chi2

from src.datasets.h36m import Human36mDatasetSequence
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    process_frame_2d
)
from human_pose_pipeline.utils.visualization import (
    visualize_pose_sequence
)
from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Dataset splits (same as Marian's)
SPLIT = {
    'train': ['S1'],
    'validation': ['S11'],
    'test': ['S5']
}


def evaluate_pose_estimation_full(ground_truth, estimated_pose, estimated_uncertainty, estimated_covariance):
    """
    Evaluate pose estimation accuracy using Mahalanobis distance and confidence intervals.
    JAX version of Marian's evaluation function.
    """
    # Calculate the difference between ground truth and estimated pose
    delta = ground_truth - estimated_pose  # Shape: (num_joints, 2)

    # Extract uncertainties and covariance
    std_x = estimated_uncertainty[:, 0]  # Shape: (num_joints,)
    std_y = estimated_uncertainty[:, 1]  # Shape: (num_joints,)
    cov_xy = estimated_covariance  # Shape: (num_joints,)

    # Compute the determinant of the covariance matrix
    det_sigma = (std_x ** 2) * (std_y ** 2) - (cov_xy ** 2)  # Shape: (num_joints,)

    # Add a small epsilon to determinant for numerical stability
    epsilon = 1e-6
    det_sigma += epsilon

    # Compute the inverse of the covariance matrix
    inv_sigma_xx = (std_y ** 2) / det_sigma  # Shape: (num_joints,)
    inv_sigma_yy = (std_x ** 2) / det_sigma  # Shape: (num_joints,)
    inv_sigma_xy = (-cov_xy) / det_sigma    # Shape: (num_joints,)

    # Compute the Mahalanobis distance for each joint
    mahalanobis = (inv_sigma_xx * (delta[:, 0] ** 2) +
                   inv_sigma_yy * (delta[:, 1] ** 2) +
                   2 * inv_sigma_xy * (delta[:, 0] * delta[:, 1]))  # Shape: (num_joints,)

    # Define chi-squared thresholds for 2 degrees of freedom
    thresholds = [chi2.ppf(0.68, df=2),   # 1 std
                  chi2.ppf(0.95, df=2),   # 2 std
                  chi2.ppf(0.9973, df=2), # 3 std
                  chi2.ppf(0.99994, df=2)]# 4 std

    # Determine which keypoints fall within each threshold
    within_std = [mahalanobis <= threshold for threshold in thresholds]

    # Count the number of keypoints within each threshold
    counts = {f'within_{i+1}std': np.sum(within) for i, within in enumerate(within_std)}

    # Prepare detailed results per joint
    joint_results = []
    for i, dist in enumerate(mahalanobis):
        joint_result = {
            'joint_index': i,
            'mahalanobis_distance': dist,
            'within_1std': dist <= thresholds[0],
            'within_2std': dist <= thresholds[1],
            'within_3std': dist <= thresholds[2],
            'within_4std': dist <= thresholds[3]
        }
        joint_results.append(joint_result)

    return {
        'counts': counts,
        'joint_results': joint_results,
        'num_joints': len(ground_truth)
    }


def main():
    """
    Main function for running pose estimation inference on the Human3.6M dataset.
    JAX version of Marian's main function.
    """
    base_directory = os.path.join(root_dir, "datasets", "H36M", "extracted")

    # Initialize models
    print("Initializing models...")

    # Initialize JAX pose estimation model with uncertainty estimation
    models_dir = os.path.join(root_dir, "human_pose_pipeline/models/pose_estimation", "H36M", "RegressFlow", "seed_420")
    checkpoint_path_jax = os.path.join(models_dir, "jax_resnet18_regressflow")
    # checkpoint_path_jax = os.path.join(models_dir, "jax_resnet50_regressflow")
    # checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_with_unc")
    pose_estimation_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path_jax)
    print("Using RegressFlowWithAleatoric model for uncertainty estimation")

    # Initialize YOLO human detector
    human_detector, device_torch = initialize_human_detector('cuda')

    print("Models initialized successfully")

    # Create datasets and dataloaders for each split
    datasets = {}
    for split in ['train']:  # Reduced to just 'train' split
        datasets[split] = Human36mDatasetSequence(base_directory, split=split, sequence_length=500)
        print(f"{split.capitalize()} dataset size: {len(datasets[split])}")

    # Process each split
    for split in ['train']:
        dataset = datasets[split]
        print(f"\nProcessing {split} split...")

        for idx, sample in enumerate(dataset):
            full_sequence = np.array(sample['pose_sequence'])
            frames = sample['frames']

            print(f"\n{split.capitalize()} split:")
            print(f"Full sequence shape: {full_sequence.shape}")
            print(f"Number of frames: {len(frames)}")

            # Initialize statistics variables
            total_frames = 0
            total_joints = 0
            total_within_1std = 0
            total_within_2std = 0
            total_within_3std = 0
            total_within_4std = 0

            # Perform pose estimation on each frame
            estimated_poses = []
            estimated_uncertainties = []
            estimated_covariances = []

            for frame_idx in range(len(frames)):
                frame_image_pil = frames[frame_idx]

                # Get pose estimations using JAX model
                pose_predictions = process_frame_2d(
                    frame=frame_image_pil,
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
                mapped_pose = pose_predictions[0]['keypoints']
                mapped_uncertainty = pose_predictions[0]['uncertainties']
                mapped_covariance = pose_predictions[0]['covariance']

                estimated_poses.append(mapped_pose)
                estimated_uncertainties.append(mapped_uncertainty)
                estimated_covariances.append(mapped_covariance)

                # Evaluate pose estimation
                ground_truth = full_sequence[frame_idx]
                estimated_pose = np.array(estimated_poses[frame_idx])
                estimated_uncertainty = np.array(estimated_uncertainties[frame_idx])
                estimated_covariance = np.array(estimated_covariances[frame_idx])

                evaluation = evaluate_pose_estimation_full(
                    ground_truth=ground_truth,
                    estimated_pose=estimated_pose,
                    estimated_uncertainty=estimated_uncertainty,
                    estimated_covariance=estimated_covariance
                )

                # Update counters
                total_frames += 1
                total_joints += evaluation['num_joints']
                total_within_1std += evaluation['counts']['within_1std']
                total_within_2std += evaluation['counts']['within_2std']
                total_within_3std += evaluation['counts']['within_3std']
                total_within_4std += evaluation['counts']['within_4std']

            # Print evaluation results
            if total_frames > 0:
                avg_within_1std = (total_within_1std / total_joints) * 100
                avg_within_2std = (total_within_2std / total_joints) * 100
                avg_within_3std = (total_within_3std / total_joints) * 100
                avg_within_4std = (total_within_4std / total_joints) * 100

                print(f"\nOverall Evaluation Results:")
                print(f"Total frames processed: {total_frames}")
                print(f"Total joints evaluated: {total_joints}")
                print(f"Average percentage of keypoints within 1 std: {avg_within_1std:.2f}%")
                print(f"Average percentage of keypoints within 2 std: {avg_within_2std:.2f}%")
                print(f"Average percentage of keypoints within 3 std: {avg_within_3std:.2f}%")
                print(f"Average percentage of keypoints within 4 std: {avg_within_4std:.2f}%")

            # Visualize the results
            output_file = f"sample_pose_sequence_with_images_{split}_{idx}.gif"
            visualize_pose_sequence(
                pose_sequence=full_sequence,
                images=frames,
                output_file=output_file,
                num_frames=len(frames),
                estimated_poses=estimated_poses,
                estimated_uncertainties=estimated_uncertainties,
                estimated_covariances=estimated_covariances,
                show_uncertainty=True
            )
            print(f"Visualization saved as {output_file}")

            # Break after first sample
            if idx == 0:
                break

if __name__ == "__main__":
    main()