"""Utilities for evaluating uncertainty estimates in human pose predictions."""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import chi2


def evaluate_pose_prediction_scores_np(predictions, targets):
    """Evaluate MPJPE scores using numpy.

    Args:
        predictions: predicted poses, shape = [B, T, J, 3]
        targets: target poses, shape = [B, T, J, 3]
    Returns:
        MPJPE: Mean per joint position error
        STD of MPJPE: Std per joint position error
        Per time MPJPE: Mean per joint position error per time step, shape = [T]
        STD of per time MPJPE, shape = [T]
        Per joint MPJPE: Mean per joint position error per time step, shape = [J]
        STD of per joint MPJPE, shape = [J]
    """
    errors = np.linalg.norm(predictions - targets, axis=-1)  # Shape = [B, T, J]
    mpjpe = np.mean(errors)  # Shape = [1]
    std = np.std(errors)  # Shape = [1]
    per_time_errors = np.mean(errors, axis=(0, 2))  # Shape = [T]
    per_time_std = np.std(errors, axis=(0, 2))  # Shape = [T]
    per_joint_errors = np.mean(errors, axis=(0, 1))  # Shape = [J]
    per_joint_std = np.std(errors, axis=(0, 1))  # Shape = [J]
    return mpjpe, std, per_time_errors, per_time_std, per_joint_errors, per_joint_std


def evaluate_pose_prediction_scores_jax(predictions, targets):
    """Evaluate MPJPE scores using JAX.

    Args:
        predictions: predicted poses, shape = [B, T, J, 3]
        targets: target poses, shape = [B, T, J, 3]
    Returns:
        MPJPE: Mean per joint position error
        STD of MPJPE: Std per joint position error
        Per time MPJPE: Mean per joint position error per time step, shape = [T]
        STD of per time MPJPE, shape = [T]
        Per joint MPJPE: Mean per joint position error per time step, shape = [J]
        STD of per joint MPJPE, shape = [J]
    """
    errors = jnp.linalg.norm(predictions - targets, axis=-1)  # Shape = [B, T, J]
    mpjpe = jnp.mean(errors)  # Shape = [1]
    std = jnp.std(errors)  # Shape = [1]
    per_time_errors = jnp.mean(errors, axis=(0, 2))  # Shape = [T]
    per_time_std = jnp.std(errors, axis=(0, 2))  # Shape = [T]
    per_joint_errors = jnp.mean(errors, axis=(0, 1))  # Shape = [J]
    per_joint_std = jnp.std(errors, axis=(0, 1))  # Shape = [J]
    return mpjpe, std, per_time_errors, per_time_std, per_joint_errors, per_joint_std


def evaluate_uncertainty_coverage_jax(pred_poses, true_poses, L, std_multipliers=[1, 2, 3, 4]):
    """
    Evaluate how well predicted Gaussian covariances match empirical coverage.

    Args:
        pred_poses: [B, T, J, 3]
        true_poses: [B, T, J, 3]
        L: Cholesky decomposition of covariance, [B, T, J, 3, 3]
        std_multipliers: list of std multipliers to evaluate

    Returns:
        List of coverage errors for each multiplier:
        error = expected_coverage - empirical_coverage
    """
    # Diff
    diff = true_poses - pred_poses             # [B, T, J, 3]
    B, T, J, C = diff.shape
    N = B * T * J
    diff = diff.reshape(N, C, 1)               # [N, 3, 1]
    L_flat = L.reshape(N, C, C)                # [N, 3, 3]

    # Solve L m = diff  →  m = L^{-1} diff
    m = jax.lax.linalg.triangular_solve(L_flat, diff, lower=True, left_side=True)
    m = m[..., 0]                              # [N, 3]

    # Mahalanobis distances: m^T m
    mahal = jnp.sum(m**2, axis=-1)             # [N]

    # Dimension = 3
    df = 3
    results = []

    for k in std_multipliers:
        # Expected Gaussian coverage for k std in 3D:
        # Probability that chi-square(df) < k^2
        expected = chi2.cdf(k * k, df=df)

        # Empirical coverage
        inside = (mahal < (k * k))               # ellipsoid boundary
        empirical = inside.mean()

        # Error = expected - empirical
        results.append(expected - empirical)

    return results


def evaluate_uncertainty_coverage_with_covariance(pred_poses, true_poses, cov_matrices):
    """
    Modified to evaluate coverage per frame
    """
    from scipy.stats import chi2
    # Convert to numpy for easier manipulation
    pred_poses = np.asarray(pred_poses)
    true_poses = np.asarray(true_poses)
    cov_matrices = np.asarray(cov_matrices)

    expected_coverage = {
        1: 0.682,  # 68.2% for 1 standard deviation
        2: 0.954,  # 95.4% for 2 standard deviations
        3: 0.997,  # 99.7% for 3 standard deviations
        4: 0.9999,  # 99.99% for 4 standard deviations
    }
    thresholds = [chi2.ppf(expected_coverage[i + 1], df=3) for i in range(4)]

    batch_size, n_frames, total_dims = pred_poses.shape
    n_joints = total_dims // 3

    # Reshape poses
    pred_poses = pred_poses.reshape(batch_size, n_frames, n_joints, 3)
    true_poses = true_poses.reshape(batch_size, n_frames, n_joints, 3)

    # Compute errors (B, T, J, 3)
    errors = true_poses - pred_poses

    # Add small epsilon to diagonal of covariance matrices for numerical stability
    cov_matrices = cov_matrices + np.eye(3)[None, None, None, :, :] * 1e-6

    # Compute inverse of covariance matrices (vectorized)
    try:
        # Compute Cholesky decomposition
        L = np.linalg.cholesky(cov_matrices)

        # Reshape errors for batch operations
        errors_reshaped = errors.reshape(batch_size, n_frames, n_joints, 3, 1)

        # Solve triangular system
        whitened_errors = np.linalg.solve(L, errors_reshaped)

        # Compute Mahalanobis distances
        mahalanobis_distances_squared = np.sum(whitened_errors**2, axis=3).squeeze()  # Shape: (B, T, J)

    except np.linalg.LinAlgError:
        print("Warning: Cholesky decomposition failed, adding more regularization")
        # Add more regularization and retry
        cov_matrices = cov_matrices + np.eye(3)[None, None, None, :, :] * 1e-4
        L = np.linalg.cholesky(cov_matrices)
        errors_reshaped = errors.reshape(batch_size, n_frames, n_joints, 3, 1)
        whitened_errors = np.linalg.solve(L, errors_reshaped)
        mahalanobis_distances_squared = np.sum(whitened_errors**2, axis=3).squeeze()

    within_stds = [mahalanobis_distances_squared <= threshold for threshold in thresholds]
    # Compute per-joint coverage
    within_std_joint = [within_std.mean(axis=(0, 1)) for within_std in within_stds]  # List of arrays shape (J,)
    within_std_frame = [within_std.mean(axis=(0, 2)) for within_std in within_stds]  # List of arrays shape (T,)
    overall_coverage = [within_std.mean() for within_std in within_stds]  # List of scalars

    # Create results dictionary
    coverage_stats = {}
    for i in range(len(within_stds)):
        coverage_stats[f'overall_within_{i + 1}std'] = overall_coverage[i]
        coverage_stats[f'per_joint_within_{i + 1}std'] = within_std_joint[i]
        coverage_stats[f'per_frame_within_{i + 1}std'] = within_std_frame[i]

    return coverage_stats
