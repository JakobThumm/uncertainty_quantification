#!/usr/bin/env python3
"""
Evaluate 2D Pose Estimation on Preprocessed H36M Dataset

This script evaluates a trained pose estimation model on the preprocessed H36M dataset.
Unlike the original evaluation that requires human detection at inference time, this uses
pre-computed bounding boxes and directly loads cropped, preprocessed images.

Key differences from pose_estimation_2D.py:
- Uses preprocessed dataset (no human detection needed)
- Loads preprocessed bounding box images directly
- Still applies reverse transformations to compare predictions with GT in original image space

Run with: python human_pose_pipeline/examples/evaluate_preprocessed_h36m.py --preprocessed_dir datasets/H36M/pre_processed --checkpoint models_tianle/H36M/RegressFlow/seed_420 --split validation --visualize --save_dir results/visualizations
"""

import os
import argparse
import json
import pickle
import numpy as np
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.wrapper import model_from_string
from src.datasets.h36m_preprocessed import Human36mPreprocessedDataset
from human_pose_pipeline.utils.transform_utils import transform_predictions_to_original_space

from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_IDX_13_MODEL,
    CONNECTIONS_13,
    JOINT_NAMES_13
)


def load_jax_model(checkpoint_path):
    """
    Load JAX pose estimation model from checkpoint

    Args:
        checkpoint_path: Path to model checkpoint directory or params file

    Returns:
        tuple: (model, params, batch_stats)
    """
    # Find args and params files
    if os.path.isdir(checkpoint_path):
        # Find files in directory
        files = os.listdir(checkpoint_path)
        args_file = [f for f in files if f.endswith('_args.json')][0]
        params_file = [f for f in files if f.endswith('_params.pickle')][0]
        args_path = os.path.join(checkpoint_path, args_file)
        params_path = os.path.join(checkpoint_path, params_file)
    else:
        # Assume checkpoint_path points to params file
        args_path = checkpoint_path.replace('_params.pickle', '_args.json')
        params_path = checkpoint_path

    # Load model configuration
    with open(args_path, 'r') as f:
        args_dict = json.load(f)

    # Load model parameters
    with open(params_path, 'rb') as f:
        params_dict = pickle.load(f)

    # Create model instance
    model = model_from_string(
        model_name=args_dict["model"],
        output_dim=args_dict["output_dim"]
    )

    params = params_dict["params"]
    batch_stats = params_dict.get("batch_stats", None)

    print(f"Loaded model: {args_dict['model']}")
    print(f"  Output dim: {args_dict['output_dim']}")
    print(f"  Has batch stats: {batch_stats is not None}")

    return model, params, batch_stats


def predict_batch(model, params, batch_stats, images):
    """
    Run model inference on a batch of preprocessed images

    Args:
        model: JAX model
        params: Model parameters
        batch_stats: Batch normalization statistics
        images: Batch of preprocessed images, shape (B, 3, 256, 192)

    Returns:
        dict: Model outputs (pred_jts, log_variance, covariance if available)
    """
    with jax.disable_jit(False):
        if batch_stats is not None:
            output = model.apply_test(params, batch_stats, images)
        else:
            output = model.apply_test(params, images)

    return output


def evaluate_preprocessed_dataset(
    model, params, batch_stats, dataset,
    num_samples=None, visualize=True, save_dir=None
):
    """
    Evaluate model on preprocessed H36M dataset

    Args:
        model: JAX pose estimation model
        params: Model parameters
        batch_stats: Batch normalization statistics
        dataset: Human36mPreprocessedDataset instance
        num_samples: Number of samples to evaluate (None = all)
        visualize: Whether to visualize results
        save_dir: Directory to save visualizations

    Returns:
        dict: Evaluation metrics
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_samples = num_samples or len(dataset)
    all_mpjpe = []
    all_per_joint_errors = []

    print(f"Evaluating {num_samples} samples...")

    for idx in tqdm(range(num_samples)):
        # Load preprocessed sample
        image, pose_flat, metadata = dataset[idx]

        # Convert to numpy and add batch dimension
        image_batch = np.expand_dims(np.array(image), axis=0)  # (1, 3, 256, 192)

        # Run model inference
        output = predict_batch(model, params, batch_stats, image_batch)

        # Extract predictions
        if isinstance(output, dict):
            pred_joints_flat = np.array(output['pred_jts'][0])  # (34,) for 17 joints
            pred_joints_17 = pred_joints_flat.reshape(17, 2)

            # Select 13 joints of interest
            pred_joints_13 = pred_joints_17[JOINT_IDX_13_MODEL]  # (13, 2)

            # Extract uncertainties if available
            if 'log_variance' in output or 'pure_sigma' in output:
                log_variance = np.array(output.get('log_variance', output.get('pure_sigma'))[0])
                uncertainties_17 = np.sqrt(np.exp(log_variance.reshape(17, 2)))
                uncertainties_13 = uncertainties_17[JOINT_IDX_13_MODEL]
            else:
                uncertainties_13 = None

            # Extract covariance if available
            if 'covariance' in output:
                covariance_17 = np.array(output['covariance'][0])
                covariance_13 = covariance_17[JOINT_IDX_13_MODEL]
            else:
                covariance_13 = None
        else:
            # Simple regression output
            pred_joints_flat = np.array(output[0])
            pred_joints_17 = pred_joints_flat.reshape(17, 2)
            pred_joints_13 = pred_joints_17[JOINT_IDX_13_MODEL]
            uncertainties_13 = None
            covariance_13 = None

        # Transform predictions back to original image space
        result = transform_predictions_to_original_space(
            pred_joints_13,
            trans=np.array(metadata['trans']),
            scale_x=metadata['scale_factors'][0],
            scale_y=metadata['scale_factors'][1],
            uncertainties=uncertainties_13,
            covariance=covariance_13
        )

        pred_keypoints_original = result['keypoints']

        # Get ground truth in original space (stored as pixel coordinates)
        gt_pose_pixel = np.array(metadata['pose_pixel'])  # Already in original space

        # Compute MPJPE for this sample
        errors = np.linalg.norm(pred_keypoints_original - gt_pose_pixel, axis=1)
        mpjpe = np.mean(errors)

        all_mpjpe.append(mpjpe)
        all_per_joint_errors.append(errors)

        # Visualize if requested
        if visualize and idx < 10:  # Only visualize first 10 samples
            visualize_sample(
                metadata, pred_keypoints_original, gt_pose_pixel,
                errors, idx, save_dir
            )

    # Compute aggregate metrics
    all_per_joint_errors = np.array(all_per_joint_errors)  # (N, 13)

    metrics = {
        'mpjpe_mean': np.mean(all_mpjpe),
        'mpjpe_std': np.std(all_mpjpe),
        'per_joint_errors_mean': np.mean(all_per_joint_errors, axis=0),
        'per_joint_errors_std': np.std(all_per_joint_errors, axis=0),
    }

    # Print results
    print(f"\n{'='*80}")
    print("Evaluation Results")
    print(f"{'='*80}")
    print(f"MPJPE: {metrics['mpjpe_mean']:.2f} ± {metrics['mpjpe_std']:.2f} pixels")
    print(f"\nPer-joint errors (pixels):")
    for i, joint_name in enumerate(JOINT_NAMES_13):
        print(f"  {joint_name:12s}: {metrics['per_joint_errors_mean']:.2f} ± {metrics['per_joint_errors_std']:.2f}")

    return metrics


def visualize_sample(metadata, pred_pose, gt_pose, errors, sample_idx, save_dir):
    """
    Visualize prediction vs ground truth for a single sample

    Args:
        metadata: Sample metadata
        pred_pose: Predicted pose in original image space (13, 2)
        gt_pose: Ground truth pose in original image space (13, 2)
        errors: Per-joint errors (13,)
        sample_idx: Sample index for naming
        save_dir: Directory to save visualization
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Draw ground truth skeleton in blue
    for start_idx, end_idx in CONNECTIONS_13:
        ax.plot([gt_pose[start_idx, 0], gt_pose[end_idx, 0]],
                [gt_pose[start_idx, 1], gt_pose[end_idx, 1]],
                'b-', linewidth=2, label='Ground Truth' if start_idx == 0 and end_idx == 1 else '')

    # Draw predicted skeleton in red
    for start_idx, end_idx in CONNECTIONS_13:
        ax.plot([pred_pose[start_idx, 0], pred_pose[end_idx, 0]],
                [pred_pose[start_idx, 1], pred_pose[end_idx, 1]],
                'r-', linewidth=2, label='Prediction' if start_idx == 0 and end_idx == 1 else '')

    # Draw joints
    ax.scatter(gt_pose[:, 0], gt_pose[:, 1], c='blue', s=100, marker='o', zorder=3)
    ax.scatter(pred_pose[:, 0], pred_pose[:, 1], c='red', s=100, marker='x', zorder=3)

    # Add joint labels with errors
    for i, (joint_name, error) in enumerate(zip(JOINT_NAMES_13, errors)):
        ax.text(gt_pose[i, 0], gt_pose[i, 1], f'{joint_name}\n{error:.1f}px',
                fontsize=8, ha='right')

    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f'Sample {sample_idx} - MPJPE: {np.mean(errors):.2f} pixels\n'
                 f'Subject: {metadata["subject"]}, Action: {metadata["action"]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_dir:
        plt.savefig(os.path.join(save_dir, f'sample_{sample_idx:04d}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate pose estimation on preprocessed H36M dataset')
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                       help='Path to preprocessed H36M dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='validation',
                       choices=['train', 'validation', 'test'],
                       help='Which split to evaluate')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to evaluate (default: all)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save visualizations')
    parser.add_argument('--output_json', type=str, default=None,
                       help='Path to save metrics as JSON')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, params, batch_stats = load_jax_model(args.checkpoint)

    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = Human36mPreprocessedDataset(
        preprocessed_dir=args.preprocessed_dir,
        split=args.split,
        return_metadata=True,
        jax_format=False
    )

    # Run evaluation
    metrics = evaluate_preprocessed_dataset(
        model, params, batch_stats, dataset,
        num_samples=args.num_samples,
        visualize=args.visualize,
        save_dir=args.save_dir
    )

    # Save metrics if requested
    if args.output_json:
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {
            'mpjpe_mean': float(metrics['mpjpe_mean']),
            'mpjpe_std': float(metrics['mpjpe_std']),
            'per_joint_errors_mean': metrics['per_joint_errors_mean'].tolist(),
            'per_joint_errors_std': metrics['per_joint_errors_std'].tolist(),
            'joint_names': JOINT_NAMES_13
        }

        with open(args.output_json, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        print(f"\nMetrics saved to {args.output_json}")


if __name__ == '__main__':
    main()
