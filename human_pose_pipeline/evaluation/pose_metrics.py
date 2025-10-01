"""
Pose Estimation Evaluation Metrics

Implements standard pose estimation evaluation metrics:
- MPJPE (Mean Per Joint Position Error)
- PCK (Percentage of Correct Keypoints)
- Mahalanobis distance with uncertainty (from Marian's Experiment 2)

These metrics are implemented in JAX for compatibility with the pose estimation pipeline.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Dict, Optional, Union
from scipy.stats import chi2

from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_NAMES_13
)


def mpjpe_jax(pred_poses: jnp.ndarray, gt_poses: jnp.ndarray) -> float:
    """
    Compute Mean Per Joint Position Error (MPJPE) in JAX.

    Args:
        pred_poses: Predicted poses of shape (N, num_joints, 2) or (N, num_joints*2)
        gt_poses: Ground truth poses of shape (N, num_joints, 2) or (N, num_joints*2)

    Returns:
        float: MPJPE value (mean Euclidean distance across all joints and samples)
    """
    # Ensure both arrays have the same shape
    if pred_poses.ndim == 2 and pred_poses.shape[1] % 2 == 0:
        # Reshape from (N, num_joints*2) to (N, num_joints, 2)
        num_joints = pred_poses.shape[1] // 2
        pred_poses = pred_poses.reshape(-1, num_joints, 2)

    if gt_poses.ndim == 2 and gt_poses.shape[1] % 2 == 0:
        num_joints = gt_poses.shape[1] // 2
        gt_poses = gt_poses.reshape(-1, num_joints, 2)

    # Compute Euclidean distance for each joint
    joint_errors = jnp.sqrt(jnp.sum((pred_poses - gt_poses) ** 2, axis=2))

    # Return mean error across all joints and samples
    return float(jnp.mean(joint_errors))

def pck_jax(pred_poses: jnp.ndarray, gt_poses: jnp.ndarray,
           threshold: float = 0.05, normalize: bool = True) -> float:
    """
    Compute Percentage of Correct Keypoints (PCK) in JAX.

    Args:
        pred_poses: Predicted poses of shape (N, num_joints, 2)
        gt_poses: Ground truth poses of shape (N, num_joints, 2)
        threshold: Distance threshold as fraction of torso size (default: 0.05 = 5%)
        normalize: Whether to normalize by torso size

    Returns:
        float: PCK percentage (0-100)
    """
    # Reshape if needed
    if pred_poses.ndim == 2 and pred_poses.shape[1] % 2 == 0:
        num_joints = pred_poses.shape[1] // 2
        pred_poses = pred_poses.reshape(-1, num_joints, 2)

    if gt_poses.ndim == 2 and gt_poses.shape[1] % 2 == 0:
        num_joints = gt_poses.shape[1] // 2
        gt_poses = gt_poses.reshape(-1, num_joints, 2)

    # Compute joint distances
    joint_distances = jnp.sqrt(jnp.sum((pred_poses - gt_poses) ** 2, axis=2))

    if normalize and gt_poses.shape[1] >= 9:  # Need at least shoulder and hip joints
        # Compute torso size (shoulder to hip distance) for normalization
        # Assuming standard joint order: shoulders at indices 1,2 and hips at 7,8
        try:
            left_shoulder = gt_poses[:, 1, :]   # Left shoulder
            right_shoulder = gt_poses[:, 2, :]  # Right shoulder
            left_hip = gt_poses[:, 7, :]        # Left hip
            right_hip = gt_poses[:, 8, :]       # Right hip

            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            torso_size = jnp.sqrt(jnp.sum((shoulder_center - hip_center) ** 2, axis=1))

            # Normalize distances by torso size
            normalized_distances = joint_distances / torso_size[:, None]
            threshold_distances = normalized_distances
        except (IndexError, ValueError):
            # Fallback: use absolute threshold if normalization fails
            threshold_distances = joint_distances
            threshold = threshold * 100  # Scale up for absolute coordinates
    else:
        threshold_distances = joint_distances
        if normalize:
            threshold = threshold * 100  # Scale up for absolute coordinates

    # Count joints within threshold
    correct_joints = threshold_distances < threshold
    pck_score = jnp.mean(correct_joints) * 100

    return float(pck_score)

def mahalanobis_accuracy_numpy(gt_poses: np.ndarray, pred_poses: np.ndarray,
                              uncertainties: np.ndarray, covariances: np.ndarray) -> Dict:
    """
    Evaluate pose estimation using Mahalanobis distance and confidence intervals.

    This implements Marian's uncertainty-aware evaluation from Experiment 2.

    Args:
        gt_poses: Ground truth poses of shape (N, num_joints, 2)
        pred_poses: Predicted poses of shape (N, num_joints, 2)
        uncertainties: Standard deviations of shape (N, num_joints, 2)
        covariances: Covariance values of shape (N, num_joints)

    Returns:
        dict: Dictionary with accuracy statistics for different confidence levels
    """
    # Ensure numpy arrays
    gt_poses = np.array(gt_poses)
    pred_poses = np.array(pred_poses)
    uncertainties = np.array(uncertainties)
    covariances = np.array(covariances)

    # Reshape if needed
    if gt_poses.ndim == 2 and gt_poses.shape[1] % 2 == 0:
        num_joints = gt_poses.shape[1] // 2
        gt_poses = gt_poses.reshape(-1, num_joints, 2)

    if pred_poses.ndim == 2 and pred_poses.shape[1] % 2 == 0:
        num_joints = pred_poses.shape[1] // 2
        pred_poses = pred_poses.reshape(-1, num_joints, 2)

    N, num_joints, _ = gt_poses.shape

    # Calculate differences
    delta = gt_poses - pred_poses  # Shape: (N, num_joints, 2)

    # Extract uncertainties
    std_x = uncertainties[:, :, 0]  # Shape: (N, num_joints)
    std_y = uncertainties[:, :, 1]  # Shape: (N, num_joints)
    cov_xy = covariances  # Shape: (N, num_joints)

    # Compute determinant of covariance matrix with numerical stability
    det_sigma = (std_x ** 2) * (std_y ** 2) - (cov_xy ** 2)
    epsilon = 1e-6
    det_sigma = det_sigma + epsilon

    # Compute inverse of covariance matrix
    inv_sigma_xx = (std_y ** 2) / det_sigma
    inv_sigma_yy = (std_x ** 2) / det_sigma
    inv_sigma_xy = (-cov_xy) / det_sigma

    # Compute Mahalanobis distance for each joint
    mahalanobis = (inv_sigma_xx * (delta[:, :, 0] ** 2) +
                   inv_sigma_yy * (delta[:, :, 1] ** 2) +
                   2 * inv_sigma_xy * (delta[:, :, 0] * delta[:, :, 1]))

    # Chi-squared thresholds for different confidence levels
    # For 2D pose estimation (degrees of freedom = 2)
    confidence_levels = [0.68, 0.95, 0.997]  # 1σ, 2σ, 3σ
    chi2_thresholds = [chi2.ppf(level, df=2) for level in confidence_levels]

    results = {
        'total_joints': N * num_joints,
        'mahalanobis_distances': mahalanobis,
        'confidence_accuracy': {}
    }

    # Count joints within each confidence interval
    for i, (level, threshold) in enumerate(zip(confidence_levels, chi2_thresholds)):
        within_confidence = mahalanobis < threshold
        accuracy = np.mean(within_confidence) * 100
        results['confidence_accuracy'][f'{level:.1%}'] = {
            'accuracy': accuracy,
            'count': int(np.sum(within_confidence)),
            'threshold': threshold
        }

    # Overall statistics
    results['mean_mahalanobis'] = float(np.mean(mahalanobis))
    results['std_mahalanobis'] = float(np.std(mahalanobis))

    return results

def comprehensive_pose_evaluation(pred_poses: Union[jnp.ndarray, np.ndarray],
                                 gt_poses: Union[jnp.ndarray, np.ndarray],
                                 uncertainties: Optional[np.ndarray] = None,
                                 covariances: Optional[np.ndarray] = None) -> Dict:
    """
    Comprehensive pose estimation evaluation with multiple metrics.

    Args:
        pred_poses: Predicted poses
        gt_poses: Ground truth poses
        uncertainties: Optional uncertainty estimates
        covariances: Optional covariance estimates

    Returns:
        dict: Complete evaluation results
    """
    # Convert to numpy for compatibility
    if hasattr(pred_poses, 'numpy'):
        pred_poses_np = pred_poses.numpy()
    else:
        pred_poses_np = np.array(pred_poses)

    if hasattr(gt_poses, 'numpy'):
        gt_poses_np = gt_poses.numpy()
    else:
        gt_poses_np = np.array(gt_poses)

    # Convert to JAX for computation
    pred_jax = jnp.array(pred_poses_np)
    gt_jax = jnp.array(gt_poses_np)

    results = {
        'mpjpe': mpjpe_jax(pred_jax, gt_jax),
        'pck_5': pck_jax(pred_jax, gt_jax, threshold=0.05),
        'pck_10': pck_jax(pred_jax, gt_jax, threshold=0.10),
        'pck_20': pck_jax(pred_jax, gt_jax, threshold=0.20),
    }

    # Add uncertainty-based evaluation if available
    if uncertainties is not None and covariances is not None:
        mahalanobis_results = mahalanobis_accuracy_numpy(
            gt_poses_np, pred_poses_np, uncertainties, covariances
        )
        results['uncertainty_evaluation'] = mahalanobis_results

    return results

def joint_wise_analysis(pred_poses: jnp.ndarray, gt_poses: jnp.ndarray) -> Dict:
    """
    Compute per-joint error analysis.

    Args:
        pred_poses: Predicted poses of shape (N, num_joints, 2)
        gt_poses: Ground truth poses of shape (N, num_joints, 2)

    Returns:
        dict: Per-joint error statistics
    """
    # Reshape if needed
    if pred_poses.ndim == 2 and pred_poses.shape[1] % 2 == 0:
        num_joints = pred_poses.shape[1] // 2
        pred_poses = pred_poses.reshape(-1, num_joints, 2)

    if gt_poses.ndim == 2 and gt_poses.shape[1] % 2 == 0:
        num_joints = gt_poses.shape[1] // 2
        gt_poses = gt_poses.reshape(-1, num_joints, 2)

    # Compute per-joint errors
    joint_errors = jnp.sqrt(jnp.sum((pred_poses - gt_poses) ** 2, axis=2))

    joint_stats = {}
    num_joints = joint_errors.shape[1]

    for joint_idx in range(num_joints):
        joint_name = JOINT_NAMES_13[joint_idx] if joint_idx < len(JOINT_NAMES_13) else f"Joint_{joint_idx}"
        errors = joint_errors[:, joint_idx]

        joint_stats[joint_name] = {
            'mean_error': float(jnp.mean(errors)),
            'std_error': float(jnp.std(errors)),
            'min_error': float(jnp.min(errors)),
            'max_error': float(jnp.max(errors)),
            'median_error': float(jnp.median(errors))
        }

    return joint_stats

def print_evaluation_summary(results: Dict, title: str = "Pose Estimation Evaluation"):
    """Pretty print evaluation results."""
    print("=" * 60)
    print(f"{title}")
    print("=" * 60)

    print(f"MPJPE: {results['mpjpe']:.3f}")
    print(f"PCK@5%: {results['pck_5']:.1f}%")
    print(f"PCK@10%: {results['pck_10']:.1f}%")
    print(f"PCK@20%: {results['pck_20']:.1f}%")

    if 'uncertainty_evaluation' in results:
        unc_eval = results['uncertainty_evaluation']
        print(f"\nUncertainty-aware evaluation:")
        print(f"Total joints evaluated: {unc_eval['total_joints']}")
        print(f"Mean Mahalanobis distance: {unc_eval['mean_mahalanobis']:.3f}")

        for level, stats in unc_eval['confidence_accuracy'].items():
            print(f"  {level} confidence: {stats['accuracy']:.1f}% ({stats['count']}/{unc_eval['total_joints']})")