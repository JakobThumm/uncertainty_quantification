#!/usr/bin/env python3
"""
Debug Preprocessed Tiger Pose Estimation

Simple test script to visualize pose estimation on a single sample from preprocessed tiger dataset.
This is useful for:
- Quick testing of model inference on OOD data (tigers)
- Debugging tiger preprocessing pipeline
- Visualizing predictions vs ground truth with tiger skeleton
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import json
import pickle

from src.models.wrapper import model_from_string
from human_pose_pipeline.utils.transform_utils import denormalize_image_regressflow

from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_IDX_13_MODEL,
    TRANSFORM_IMAGE_SIZE
)


# Tiger skeleton connections (mapped to H36M 13-joint format)
# Based on the tiger-to-H36M mapping in tiger_pose_to_h36m_format()
# Original tiger connections: 0-1, 1-2, 2-3, 3-4, 4-5, 3-7, 7-6, 2-8, 8-9, 2-10, 10-11
TIGER_CONNECTIONS_H36M = [
    (0, 1),   # nose to left_eye->left_shoulder
    (1, 2),   # left_eye->left_shoulder to right_eye->right_shoulder
    (2, 3),   # right_eye->right_shoulder to left_ear->left_elbow
    (3, 4),   # left_ear->left_elbow to right_ear->right_elbow
    (4, 5),   # right_ear->right_elbow to front_left_paw->left_wrist
    (3, 7),   # left_ear->left_elbow to back_left_paw->left_hip
    (7, 6),   # back_left_paw->left_hip to front_right_paw->right_wrist
    (2, 8),   # right_eye->right_shoulder to back_right_paw->right_hip
    (8, 9),   # back_right_paw->right_hip to tail_start->left_knee
    (2, 10),  # right_eye->right_shoulder to tail_middle->right_knee
    (10, 11)  # tail_middle->right_knee to tail_end->left_ankle
]

# Tiger joint names mapped to H36M indices
TIGER_JOINT_NAMES_H36M = [
    'Nose', 'LeftEye', 'RightEye', 'LeftEar', 'RightEar',
    'FrontLeftPaw', 'FrontRightPaw', 'BackLeftPaw', 'BackRightPaw',
    'TailStart', 'TailMiddle', 'TailEnd', '(unused)'
]


def load_model(checkpoint_path):
    """Load JAX model from checkpoint"""
    # Find args and params files
    if os.path.isdir(checkpoint_path):
        # Case 1: Directory path - find the first matching files
        files = os.listdir(checkpoint_path)
        args_file = [f for f in files if f.endswith('_args.json')][0]
        params_file = [f for f in files if f.endswith('_params.pickle')][0]
        args_path = os.path.join(checkpoint_path, args_file)
        params_path = os.path.join(checkpoint_path, params_file)
    elif os.path.exists(checkpoint_path):
        # Case 2: Full path to params file exists
        args_path = checkpoint_path.replace('_params.pickle', '_args.json')
        params_path = checkpoint_path
    else:
        # Case 3: Base checkpoint name (without extension)
        args_path = checkpoint_path + '_args.json'
        params_path = checkpoint_path + '_params.pickle'
        if not os.path.exists(args_path) or not os.path.exists(params_path):
            raise FileNotFoundError(
                f"Checkpoint files not found:\n"
                f"  Args: {args_path}\n"
                f"  Params: {params_path}\n"
                f"Please provide either:\n"
                f"  - Directory containing checkpoint files\n"
                f"  - Full path to _params.pickle file\n"
                f"  - Base checkpoint name (without _args.json or _params.pickle)"
            )

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


def visualize_tiger_pose_comparison(image_array, pred_pose_pixels, gt_pose_pixels, valid_mask,
                                   pred_uncertainties_pixels=None, sample_idx=0, save_path=None):
    """
    Visualize predicted vs ground truth tiger pose on preprocessed image

    Args:
        image_array: (H, W, 3) denormalized image array
        pred_pose_pixels: (13, 2) predicted pose in pixel coordinates
        gt_pose_pixels: (13, 2) ground truth pose in pixel coordinates
        valid_mask: (13,) boolean mask for valid keypoints
        pred_uncertainties_pixels: Optional (13, 2) uncertainties in pixel coordinates
        sample_idx: Sample index for title
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Left: Ground truth only
    ax1 = axes[0]
    ax1.imshow(image_array)
    ax1.set_title('Ground Truth Tiger Pose', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Draw GT skeleton
    for start_idx, end_idx in TIGER_CONNECTIONS_H36M:
        if valid_mask[start_idx] and valid_mask[end_idx]:
            ax1.plot([gt_pose_pixels[start_idx, 0], gt_pose_pixels[end_idx, 0]],
                    [gt_pose_pixels[start_idx, 1], gt_pose_pixels[end_idx, 1]],
                    'g-', linewidth=3, alpha=0.8)

    # Draw GT keypoints
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            ax1.scatter(gt_pose_pixels[i, 0], gt_pose_pixels[i, 1],
                       c='green', s=150, marker='o', zorder=3, edgecolors='white', linewidths=2)
            ax1.text(gt_pose_pixels[i, 0] + 5, gt_pose_pixels[i, 1] - 5, str(i),
                    fontsize=8, color='green', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Middle: Prediction only
    ax2 = axes[1]
    ax2.imshow(image_array)
    ax2.set_title('Predicted Pose (Human Model on Tiger)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Draw predicted skeleton
    for start_idx, end_idx in TIGER_CONNECTIONS_H36M:
        ax2.plot([pred_pose_pixels[start_idx, 0], pred_pose_pixels[end_idx, 0]],
                [pred_pose_pixels[start_idx, 1], pred_pose_pixels[end_idx, 1]],
                'r-', linewidth=3, alpha=0.8)

    # Draw predicted keypoints with uncertainties
    for i in range(len(pred_pose_pixels)):
        ax2.scatter(pred_pose_pixels[i, 0], pred_pose_pixels[i, 1],
                   c='red', s=150, marker='x', zorder=3, linewidths=3)
        ax2.text(pred_pose_pixels[i, 0] + 5, pred_pose_pixels[i, 1] + 10, str(i),
                fontsize=8, color='red', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Draw uncertainty ellipse if available
        if pred_uncertainties_pixels is not None:
            from matplotlib.patches import Ellipse
            unc_x, unc_y = pred_uncertainties_pixels[i]
            ellipse = Ellipse(
                (pred_pose_pixels[i, 0], pred_pose_pixels[i, 1]),
                width=unc_x * 2, height=unc_y * 2,
                alpha=0.3, color='red', fill=True
            )
            ax2.add_patch(ellipse)

    # Right: Overlay comparison
    ax3 = axes[2]
    ax3.imshow(image_array)
    ax3.set_title('Overlay: Green=GT, Red=Pred', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Draw GT skeleton (green, thinner, transparent)
    for start_idx, end_idx in TIGER_CONNECTIONS_H36M:
        if valid_mask[start_idx] and valid_mask[end_idx]:
            ax3.plot([gt_pose_pixels[start_idx, 0], gt_pose_pixels[end_idx, 0]],
                    [gt_pose_pixels[start_idx, 1], gt_pose_pixels[end_idx, 1]],
                    'g-', linewidth=2, alpha=0.6, label='GT' if start_idx == 0 else '')

    # Draw predicted skeleton (red, thinner)
    for start_idx, end_idx in TIGER_CONNECTIONS_H36M:
        ax3.plot([pred_pose_pixels[start_idx, 0], pred_pose_pixels[end_idx, 0]],
                [pred_pose_pixels[start_idx, 1], pred_pose_pixels[end_idx, 1]],
                'r-', linewidth=2, alpha=0.8, label='Pred' if start_idx == 0 else '')

    # Draw keypoints
    for i, is_valid in enumerate(valid_mask):
        if is_valid:
            ax3.scatter(gt_pose_pixels[i, 0], gt_pose_pixels[i, 1],
                       c='green', s=100, marker='o', zorder=3, alpha=0.6, edgecolors='white')

    for i in range(len(pred_pose_pixels)):
        ax3.scatter(pred_pose_pixels[i, 0], pred_pose_pixels[i, 1],
                   c='red', s=100, marker='x', zorder=3, linewidths=2)

    # Compute and display errors for valid keypoints
    errors = np.linalg.norm(pred_pose_pixels[valid_mask] - gt_pose_pixels[valid_mask], axis=1)
    mpjpe = np.mean(errors)

    ax3.legend(loc='upper right', fontsize=10)

    # Overall title with error
    fig.suptitle(f'Tiger Pose Sample {sample_idx} - MPJPE: {mpjpe:.2f} pixels\n'
                 f'(OOD test: Human pose model applied to tiger)',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()

    return mpjpe, errors


def main():
    parser = argparse.ArgumentParser(description='Debug pose estimation on preprocessed tiger sample')
    parser.add_argument('--preprocessed_dir', type=str,
                       default='datasets/tiger-pose/preprocessed',
                       help='Path to preprocessed tiger dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val'],
                       help='Which split to use')
    parser.add_argument('--sample_idx', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--save_path', type=str, default='debug_tiger_preprocessed.png',
                       help='Path to save visualization')

    args = parser.parse_args()

    print("=" * 80)
    print("Debug Preprocessed Tiger Pose Estimation")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, params, batch_stats = load_model(args.checkpoint)

    # Load preprocessed data
    print(f"\nLoading preprocessed {args.split} data...")
    images_path = os.path.join(args.preprocessed_dir, args.split, 'PreprocessedImages', f'tiger_pose_{args.split}.npy')
    poses_path = os.path.join(args.preprocessed_dir, args.split, 'PreprocessedPoses', f'tiger_pose_{args.split}.npz')

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Preprocessed images not found: {images_path}")
    if not os.path.exists(poses_path):
        raise FileNotFoundError(f"Preprocessed poses not found: {poses_path}")

    images = np.load(images_path)  # (N, 3, 256, 192)
    poses_data = np.load(poses_path)

    poses_normalized = poses_data['poses_normalized']  # (N, 13, 2)
    poses_pixel = poses_data['poses_pixel']  # (N, 13, 2)
    valid_keypoints = poses_data['valid_keypoints']  # (N, 13)

    print(f"Loaded {len(images)} samples")
    print(f"  Images shape: {images.shape}")
    print(f"  Poses normalized shape: {poses_normalized.shape}")
    print(f"  Poses pixel shape: {poses_pixel.shape}")

    # Get sample
    sample_idx = args.sample_idx
    if sample_idx >= len(images):
        print(f"Warning: sample_idx {sample_idx} >= dataset size {len(images)}, using sample 0")
        sample_idx = 0

    image = images[sample_idx]  # (3, 256, 192)
    pose_norm = poses_normalized[sample_idx]  # (13, 2)
    pose_pix = poses_pixel[sample_idx]  # (13, 2)
    valid_mask = valid_keypoints[sample_idx]  # (13,)

    print(f"\nProcessing sample {sample_idx}:")
    print(f"  Image shape: {image.shape}")
    print(f"  Valid keypoints: {np.sum(valid_mask)}/13")

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
            print("  Model provides covariances")
        else:
            covariance_13 = None
    else:
        pred_joints_flat = np.array(output[0])
        pred_joints_17 = pred_joints_flat.reshape(17, 2)
        pred_joints_13 = pred_joints_17[JOINT_IDX_13_MODEL]
        uncertainties_13 = None
        covariance_13 = None

    # Convert predictions from normalized [-0.5, 0.5] to pixel coordinates
    image_w, image_h = TRANSFORM_IMAGE_SIZE[0], TRANSFORM_IMAGE_SIZE[1]
    pred_pose_pixels = np.zeros_like(pred_joints_13)
    pred_pose_pixels[:, 0] = (pred_joints_13[:, 0] + 0.5) * image_w
    pred_pose_pixels[:, 1] = (pred_joints_13[:, 1] + 0.5) * image_h

    # Convert uncertainties to pixel coordinates if available
    if uncertainties_13 is not None:
        pred_uncertainties_pixels = uncertainties_13.copy()
        pred_uncertainties_pixels[:, 0] = uncertainties_13[:, 0] * image_w
        pred_uncertainties_pixels[:, 1] = uncertainties_13[:, 1] * image_h
    else:
        pred_uncertainties_pixels = None

    # Compute errors for valid keypoints
    valid_pred = pred_pose_pixels[valid_mask]
    valid_gt = pose_pix[valid_mask]
    errors = np.linalg.norm(valid_pred - valid_gt, axis=1)
    mpjpe = np.mean(errors)

    print(f"\nResults:")
    print(f"  MPJPE (valid keypoints): {mpjpe:.2f} pixels")
    print(f"\n  Per-joint errors (valid keypoints only):")
    valid_joint_names = [name for i, name in enumerate(TIGER_JOINT_NAMES_H36M) if valid_mask[i]]
    for joint_name, error in zip(valid_joint_names, errors):
        print(f"    {joint_name:20s}: {error:.2f} pixels")

    # Denormalize image for visualization
    print("\nDenormalizing image...")
    image_denorm = denormalize_image_regressflow(image)  # (H, W, 3)
    image_denorm = np.clip(image_denorm, 0, 1)

    # Visualize
    print("\nGenerating visualization...")
    mpjpe_computed, _ = visualize_tiger_pose_comparison(
        image_array=image_denorm,
        pred_pose_pixels=pred_pose_pixels,
        gt_pose_pixels=pose_pix,
        valid_mask=valid_mask,
        pred_uncertainties_pixels=pred_uncertainties_pixels,
        sample_idx=sample_idx,
        save_path=args.save_path
    )

    print(f"\n{'=' * 80}")
    print("Summary:")
    print(f"  This visualization shows a human pose model applied to tiger images (OOD).")
    print(f"  The model was trained on human poses (H36M) but tested on tigers.")
    print(f"  High errors indicate the model struggles with out-of-distribution data.")
    print(f"  MPJPE: {mpjpe:.2f} pixels")
    print(f"{'=' * 80}")

    print("\nDone!")


if __name__ == '__main__':
    main()
