#!/usr/bin/env python3
"""
Debug Preprocessed Pose Estimation

Simple test script to visualize pose estimation on a single sample from preprocessed H36M dataset.
This is useful for:
- Quick testing of model inference
- Debugging preprocessing pipeline
- Visualizing predictions vs ground truth
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import json
import pickle

from src.models.wrapper import model_from_string
from src.datasets.h36m_preprocessed import Human36mPreprocessedDataset
from human_pose_pipeline.utils.transform_utils import (
    transform_predictions_to_original_space,
    denormalize_image_regressflow
)

from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_IDX_13_MODEL,
    CONNECTIONS_13,
    JOINT_NAMES_13
)


def load_model(checkpoint_path):
    """Load JAX model from checkpoint"""
    # Find args and params files
    if os.path.isdir(checkpoint_path):
        files = os.listdir(checkpoint_path)
        args_file = [f for f in files if f.endswith('_args.json')][0]
        params_file = [f for f in files if f.endswith('_params.pickle')][0]
        args_path = os.path.join(checkpoint_path, args_file)
        params_path = os.path.join(checkpoint_path, params_file)
    else:
        args_path = checkpoint_path.replace('_params.pickle', '_args.json')
        params_path = checkpoint_path

    with open(args_path, 'r') as f:
        args_dict = json.load(f)

    with open(params_path, 'rb') as f:
        params_dict = pickle.load(f)

    model = model_from_string(
        model_name=args_dict["model"],
        output_dim=args_dict["output_dim"]
    )

    params = params_dict["params"]
    batch_stats = params_dict.get("batch_stats", None)

    print(f"Loaded model: {args_dict['model']}")
    return model, params, batch_stats


def predict_single_sample(model, params, batch_stats, image):
    """Run inference on a single image"""
    # Add batch dimension
    image_batch = np.expand_dims(np.array(image), axis=0)

    # Run inference
    with jax.disable_jit(False):
        if batch_stats is not None:
            output = model.apply_test(params, batch_stats, image_batch)
        else:
            output = model.apply_test(params, image_batch)

    return output


def visualize_pose_comparison(pred_pose, gt_pose, errors, metadata, save_path=None):
    """
    Visualize predicted vs ground truth pose

    Args:
        pred_pose: Predicted pose (13, 2) in original image space
        gt_pose: Ground truth pose (13, 2) in original image space
        errors: Per-joint errors (13,)
        metadata: Sample metadata
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left plot: Ground Truth
    ax1.set_title('Ground Truth Pose', fontsize=14, fontweight='bold')
    for start_idx, end_idx in CONNECTIONS_13:
        ax1.plot([gt_pose[start_idx, 0], gt_pose[end_idx, 0]],
                [gt_pose[start_idx, 1], gt_pose[end_idx, 1]],
                'b-', linewidth=3)
    ax1.scatter(gt_pose[:, 0], gt_pose[:, 1], c='blue', s=150, marker='o', zorder=3)
    for i, joint_name in enumerate(JOINT_NAMES_13):
        ax1.text(gt_pose[i, 0] + 10, gt_pose[i, 1] - 10, joint_name,
                fontsize=9, ha='left', color='blue', fontweight='bold')
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)

    # Right plot: Prediction vs Ground Truth
    ax2.set_title('Prediction (red) vs Ground Truth (blue)', fontsize=14, fontweight='bold')

    # Draw GT skeleton in blue
    for start_idx, end_idx in CONNECTIONS_13:
        ax2.plot([gt_pose[start_idx, 0], gt_pose[end_idx, 0]],
                [gt_pose[start_idx, 1], gt_pose[end_idx, 1]],
                'b-', linewidth=2, alpha=0.5, label='GT' if start_idx == 0 else '')

    # Draw predicted skeleton in red
    for start_idx, end_idx in CONNECTIONS_13:
        ax2.plot([pred_pose[start_idx, 0], pred_pose[end_idx, 0]],
                [pred_pose[start_idx, 1], pred_pose[end_idx, 1]],
                'r-', linewidth=2, label='Pred' if start_idx == 0 else '')

    # Draw joints
    ax2.scatter(gt_pose[:, 0], gt_pose[:, 1], c='blue', s=150, marker='o', zorder=3, alpha=0.5)
    ax2.scatter(pred_pose[:, 0], pred_pose[:, 1], c='red', s=150, marker='x', zorder=3, linewidths=3)

    # Add error annotations
    for i, (joint_name, error) in enumerate(zip(JOINT_NAMES_13, errors)):
        mid_x = (gt_pose[i, 0] + pred_pose[i, 0]) / 2
        mid_y = (gt_pose[i, 1] + pred_pose[i, 1]) / 2
        ax2.text(mid_x, mid_y, f'{error:.1f}px',
                fontsize=8, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax2.set_aspect('equal')
    ax2.invert_yaxis()
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Overall title with metadata
    mpjpe = np.mean(errors)
    fig.suptitle(f'Sample: {metadata["subject"]} - {metadata["action"]} (frame {metadata["original_frame_idx"]})\n'
                 f'MPJPE: {mpjpe:.2f} pixels', fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Debug pose estimation on single preprocessed sample')
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                       help='Path to preprocessed H36M dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='validation',
                       choices=['train', 'validation', 'test'],
                       help='Which split to use')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--save_path', type=str, default=None,
                       help='Path to save visualization')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, params, batch_stats = load_model(args.checkpoint)

    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = Human36mPreprocessedDataset(
        preprocessed_dir=args.preprocessed_dir,
        split=args.split,
        return_metadata=True,
        jax_format=False
    )

    print(f"Dataset has {len(dataset)} samples")

    # Load single sample
    sample_idx = args.sample_idx
    if sample_idx >= len(dataset):
        print(f"Warning: sample_idx {sample_idx} >= dataset size {len(dataset)}, using sample 0")
        sample_idx = 0

    image, pose_flat, metadata = dataset[sample_idx]

    # Visualize preprocessed image (denormalized)
    print("\nVisualizing preprocessed image...")
    image_denorm = denormalize_image_regressflow(image)

    # Display the preprocessed image
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    ax.imshow(image_denorm)
    ax.set_title(f'Preprocessed Image (denormalized)\nSample {sample_idx}', fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('preprocessed_image_debug.png', dpi=150, bbox_inches='tight')
    print(f"Saved preprocessed image to preprocessed_image_debug.png")
    plt.close()

    print(f"\nProcessing sample {sample_idx}:")
    print(f"  Subject: {metadata['subject']}")
    print(f"  Action: {metadata['action']}")
    print(f"  Original frame: {metadata['original_frame_idx']}")
    print(f"  Image shape: {image.shape}")

    # Run inference
    print("\nRunning inference...")
    output = predict_single_sample(model, params, batch_stats, image)

    # Extract predictions
    if isinstance(output, dict):
        pred_joints_flat = np.array(output['pred_jts'][0])
        pred_joints_17 = pred_joints_flat.reshape(17, 2)
        pred_joints_13 = pred_joints_17[JOINT_IDX_13_MODEL]

        # Extract uncertainties if available
        if 'log_variance' in output or 'pure_sigma' in output:
            log_variance = np.array(output.get('log_variance', output.get('pure_sigma'))[0])
            uncertainties_17 = np.sqrt(np.exp(log_variance.reshape(17, 2)))
            uncertainties_13 = uncertainties_17[JOINT_IDX_13_MODEL]
            print("  Model provides uncertainties")
        else:
            uncertainties_13 = None

        # Extract covariance if available
        if 'covariance' in output:
            covariance_17 = np.array(output['covariance'][0])
            covariance_13 = covariance_17[JOINT_IDX_13_MODEL]
        else:
            covariance_13 = None
    else:
        pred_joints_flat = np.array(output[0])
        pred_joints_17 = pred_joints_flat.reshape(17, 2)
        pred_joints_13 = pred_joints_17[JOINT_IDX_13_MODEL]
        uncertainties_13 = None
        covariance_13 = None

    # Transform to original image space
    print("Transforming predictions to original image space...")
    retransformed_prediction = transform_predictions_to_original_space(
        pred_joints_13,
        trans=np.array(metadata['trans']),
        scale_x=metadata['scale_factors'][0],
        scale_y=metadata['scale_factors'][1],
        uncertainties=uncertainties_13,
        covariance=covariance_13
    )
    
    retransformed_ground_truth = transform_predictions_to_original_space(
        np.array(metadata["pose_normalized"]),
        trans=np.array(metadata['trans']),
        scale_x=metadata['scale_factors'][0],
        scale_y=metadata['scale_factors'][1],
        uncertainties=uncertainties_13,  # not used.
        covariance=covariance_13  # not used.
    )

    pred_pose_original = retransformed_prediction['keypoints']
    gt_pose_original = retransformed_ground_truth['keypoints']  # np.array(metadata['pose_pixel'])

    # Compute errors
    errors = np.linalg.norm(pred_pose_original - gt_pose_original, axis=1)
    mpjpe = np.mean(errors)

    print(f"\nResults:")
    print(f"  MPJPE: {mpjpe:.2f} pixels")
    print(f"\n  Per-joint errors:")
    for joint_name, error in zip(JOINT_NAMES_13, errors):
        print(f"    {joint_name:12s}: {error:.2f} pixels")

    # Visualize
    print("\nGenerating visualization...")
    visualize_pose_comparison(
        pred_pose_original, gt_pose_original, errors,
        metadata, save_path=args.save_path
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
