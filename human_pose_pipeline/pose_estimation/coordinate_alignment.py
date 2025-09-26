"""
Coordinate System Alignment for Pose Estimation

This module handles coordinate transformations between:
1. Model predictions (normalized coordinates)
2. Ground truth poses (pixel coordinates)
3. Different image sizes and aspect ratios

The goal is to align coordinate systems for proper evaluation metrics.
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, Union, Optional

def normalize_to_image_coords(normalized_poses: Union[jnp.ndarray, np.ndarray],
                             image_size: Tuple[int, int] = (256, 192),
                             normalization_range: str = 'centered') -> np.ndarray:
    """
    Convert normalized pose coordinates to image pixel coordinates.

    Args:
        normalized_poses: Poses in normalized coordinates, shape (N, num_joints, 2)
        image_size: Target image size as (height, width)
        normalization_range: Type of normalization used
            - 'centered': [-1, 1] range (default for many models)
            - 'regressflow': [-0.5, 0.5] range (RegressFlow models)
            - 'unit': [0, 1] range
            - 'auto': Auto-detect based on data range

    Returns:
        np.ndarray: Poses in pixel coordinates
    """
    if hasattr(normalized_poses, 'numpy'):
        poses = normalized_poses.numpy()
    else:
        poses = np.array(normalized_poses)

    height, width = image_size

    # Auto-detect normalization if requested
    if normalization_range == 'auto':
        min_val = np.min(poses)
        max_val = np.max(poses)

        if min_val >= -0.1 and max_val <= 1.1:  # Likely [0, 1]
            normalization_range = 'unit'
        elif min_val >= -0.6 and max_val <= 0.6:  # Likely [-0.5, 0.5] RegressFlow range
            normalization_range = 'regressflow'
        elif min_val >= -1.1 and max_val <= 1.1:  # Likely [-1, 1]
            normalization_range = 'centered'
        else:
            print(f"Warning: Unusual coordinate range [{min_val:.3f}, {max_val:.3f}]")
            normalization_range = 'regressflow'  # Default to RegressFlow for our models

    # Convert based on detected/specified range
    if normalization_range == 'centered':  # [-1, 1] → [0, image_size]
        pixel_poses = poses.copy()
        pixel_poses[..., 0] = (poses[..., 0] + 1) * width / 2
        pixel_poses[..., 1] = (poses[..., 1] + 1) * height / 2
    elif normalization_range == 'regressflow':  # [-0.5, 0.5] → [0, image_size] (Marian's approach)
        pixel_poses = poses.copy()
        pixel_poses[..., 0] = (poses[..., 0] + 0.5) * width
        pixel_poses[..., 1] = (poses[..., 1] + 0.5) * height
    elif normalization_range == 'unit':  # [0, 1] → [0, image_size]
        pixel_poses = poses.copy()
        pixel_poses[..., 0] = poses[..., 0] * width
        pixel_poses[..., 1] = poses[..., 1] * height
    else:
        raise ValueError(f"Unknown normalization_range: {normalization_range}")

    return pixel_poses

def scale_poses_to_target_size(poses: Union[jnp.ndarray, np.ndarray],
                              source_size: Tuple[int, int],
                              target_size: Tuple[int, int]) -> np.ndarray:
    """
    Scale pose coordinates from one image size to another.

    Args:
        poses: Pose coordinates, shape (N, num_joints, 2)
        source_size: Original image size as (height, width)
        target_size: Target image size as (height, width)

    Returns:
        np.ndarray: Scaled poses
    """
    if hasattr(poses, 'numpy'):
        poses_np = poses.numpy()
    else:
        poses_np = np.array(poses)

    source_h, source_w = source_size
    target_h, target_w = target_size

    scale_x = target_w / source_w
    scale_y = target_h / source_h

    scaled_poses = poses_np.copy()
    scaled_poses[..., 0] *= scale_x
    scaled_poses[..., 1] *= scale_y

    return scaled_poses

def align_pose_coordinates(pred_poses: Union[jnp.ndarray, np.ndarray],
                          gt_poses: Union[jnp.ndarray, np.ndarray],
                          pred_image_size: Tuple[int, int] = (256, 192),
                          gt_image_size: Optional[Tuple[int, int]] = None,
                          alignment_method: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
    """
    Align prediction and ground truth pose coordinates to the same coordinate system.

    Args:
        pred_poses: Predicted poses (typically normalized)
        gt_poses: Ground truth poses (typically in pixel coordinates)
        pred_image_size: Image size used for predictions (height, width)
        gt_image_size: Original image size for ground truth (height, width)
                      If None, assumes same as pred_image_size
        alignment_method: Method for alignment
            - 'auto': Auto-detect and align coordinate systems
            - 'normalize_predictions': Convert predictions to pixel coordinates
            - 'normalize_gt': Convert GT to normalized coordinates

    Returns:
        Tuple[np.ndarray, np.ndarray]: (aligned_pred_poses, aligned_gt_poses)
    """
    # Convert to numpy
    if hasattr(pred_poses, 'numpy'):
        pred_np = pred_poses.numpy()
    else:
        pred_np = np.array(pred_poses)

    if hasattr(gt_poses, 'numpy'):
        gt_np = gt_poses.numpy()
    else:
        gt_np = np.array(gt_poses)

    # Reshape if flattened
    if pred_np.ndim == 2 and pred_np.shape[-1] % 2 == 0:
        num_joints = pred_np.shape[-1] // 2
        pred_np = pred_np.reshape(-1, num_joints, 2)

    if gt_np.ndim == 2 and gt_np.shape[-1] % 2 == 0:
        num_joints = gt_np.shape[-1] // 2
        gt_np = gt_np.reshape(-1, num_joints, 2)

    # Analyze coordinate ranges
    pred_min, pred_max = np.min(pred_np), np.max(pred_np)
    gt_min, gt_max = np.min(gt_np), np.max(gt_np)

    print(f"Coordinate analysis:")
    print(f"  Predictions: [{pred_min:.3f}, {pred_max:.3f}]")
    print(f"  Ground truth: [{gt_min:.3f}, {gt_max:.3f}]")

    # Auto-detect coordinate systems
    pred_is_normalized = (pred_min >= -1.1 and pred_max <= 1.1)
    gt_is_pixels = (gt_max > 10)  # Assume pixel coords if range > 10

    if alignment_method == 'auto':
        if pred_is_normalized and gt_is_pixels:
            alignment_method = 'normalize_predictions'
            print(f"  Auto-detected: Converting predictions to pixel coordinates")
        elif not pred_is_normalized and gt_is_pixels:
            alignment_method = 'scale_to_common'
            print(f"  Auto-detected: Scaling both to common coordinate system")
        else:
            alignment_method = 'normalize_predictions'
            print(f"  Auto-detected: Default to normalizing predictions")

    # Apply alignment
    if alignment_method == 'normalize_predictions':
        # Convert predictions from normalized to pixel coordinates
        aligned_pred = normalize_to_image_coords(
            pred_np, pred_image_size, normalization_range='auto'
        )

        # Scale GT to prediction image size if needed
        if gt_image_size is not None and gt_image_size != pred_image_size:
            aligned_gt = scale_poses_to_target_size(gt_np, gt_image_size, pred_image_size)
        else:
            aligned_gt = gt_np.copy()

    elif alignment_method == 'normalize_gt':
        # Convert GT to normalized coordinates
        if gt_image_size is None:
            gt_image_size = pred_image_size

        aligned_gt = gt_np.copy()
        aligned_gt[..., 0] = (gt_np[..., 0] / gt_image_size[1]) * 2 - 1  # [0, w] → [-1, 1]
        aligned_gt[..., 1] = (gt_np[..., 1] / gt_image_size[0]) * 2 - 1  # [0, h] → [-1, 1]
        aligned_pred = pred_np.copy()

    elif alignment_method == 'scale_to_common':
        # Scale both to a common coordinate system (pixel coordinates)
        aligned_pred = pred_np.copy()
        aligned_gt = gt_np.copy()

        # If predictions are much smaller, scale them up
        if pred_max < 10 and gt_max > 100:
            aligned_pred = normalize_to_image_coords(pred_np, pred_image_size)

    else:
        raise ValueError(f"Unknown alignment_method: {alignment_method}")

    # Final verification
    final_pred_range = (np.min(aligned_pred), np.max(aligned_pred))
    final_gt_range = (np.min(aligned_gt), np.max(aligned_gt))

    print(f"  After alignment:")
    print(f"    Predictions: [{final_pred_range[0]:.3f}, {final_pred_range[1]:.3f}]")
    print(f"    Ground truth: [{final_gt_range[0]:.3f}, {final_gt_range[1]:.3f}]")

    return aligned_pred, aligned_gt

def compute_alignment_metrics(pred_poses: np.ndarray, gt_poses: np.ndarray) -> dict:
    """
    Compute metrics to assess coordinate alignment quality.

    Args:
        pred_poses: Aligned predicted poses
        gt_poses: Aligned ground truth poses

    Returns:
        dict: Alignment quality metrics
    """
    # Compute coordinate statistics
    pred_center = np.mean(pred_poses, axis=(0, 1))
    gt_center = np.mean(gt_poses, axis=(0, 1))

    pred_std = np.std(pred_poses, axis=(0, 1))
    gt_std = np.std(gt_poses, axis=(0, 1))

    # Compute scale ratio
    pred_scale = np.mean(pred_std)
    gt_scale = np.mean(gt_std)
    scale_ratio = pred_scale / gt_scale if gt_scale > 0 else float('inf')

    # Compute center offset
    center_offset = np.linalg.norm(pred_center - gt_center)

    return {
        'pred_center': pred_center.tolist(),
        'gt_center': gt_center.tolist(),
        'pred_std': pred_std.tolist(),
        'gt_std': gt_std.tolist(),
        'scale_ratio': scale_ratio,
        'center_offset': center_offset,
        'alignment_quality': 'good' if (0.8 <= scale_ratio <= 1.2 and center_offset < 50) else 'poor'
    }