"""Utilities for evaluating uncertainty estimates in human pose predictions."""

import numpy as np
import jax.numpy as jnp


def evaluate_pose_prediction_scores_np(predictions, targets):
    """Evaluate MPJPE scores."""
    errors = np.linalg.norm(predictions - targets, axis=-1)
    mpjpe = np.mean(errors)
    std = np.std(errors)
    per_time_errors = np.mean(np.mean(errors, axis=-1), axis=0)
    per_time_std = np.std(np.mean(errors, axis=-1), axis=0)
    per_joint_errors = np.mean(np.mean(errors, axis=1), axis=0)
    per_joint_std = np.std(np.mean(errors, axis=1), axis=0)
    return mpjpe, std, per_time_errors, per_time_std, per_joint_errors, per_joint_std


def evaluate_pose_prediction_scores_jax(predictions, targets):
    """Evaluate MPJPE scores using JAX."""
    errors = jnp.linalg.norm(predictions - targets, axis=-1)
    mpjpe = jnp.mean(errors)
    std = jnp.std(errors)
    per_time_errors = jnp.mean(jnp.mean(errors, axis=-1), axis=0)
    per_time_std = jnp.std(jnp.mean(errors, axis=-1), axis=0)
    per_joint_errors = jnp.mean(jnp.mean(errors, axis=1), axis=0)
    per_joint_std = jnp.std(jnp.mean(errors, axis=1), axis=0)
    return mpjpe, std, per_time_errors, per_time_std, per_joint_errors, per_joint_std


def evaluate_uncertainty_coverage_with_covariance(pred_poses, true_poses, cov_matrices, std_multipliers=[1, 2, 3, 4]):
    """
    Modified to evaluate coverage per frame
    """
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

    batch_size, n_frames, total_dims = pred_poses.shape
    n_joints = total_dims // 3

    # Reshape poses
    pred_poses = pred_poses.reshape(batch_size, n_frames, n_joints, 3)
    true_poses = true_poses.reshape(batch_size, n_frames, n_joints, 3)

    # Initialize results dictionary
    coverage_stats = {
        "per_joint": {joint: {mult: 0.0 for mult in std_multipliers} for joint in range(n_joints)},
        "overall": {mult: 0.0 for mult in std_multipliers},
        "per_frame": {frame: {mult: 0.0 for mult in std_multipliers} for frame in range(n_frames)},
        "expected": {mult: expected_coverage[mult] for mult in std_multipliers},
    }

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
        mahalanobis_distances = np.sqrt(np.sum(whitened_errors**2, axis=3)).squeeze()

    except np.linalg.LinAlgError:
        print("Warning: Cholesky decomposition failed, adding more regularization")
        # Add more regularization and retry
        cov_matrices = cov_matrices + np.eye(3)[None, None, None, :, :] * 1e-4
        L = np.linalg.cholesky(cov_matrices)
        errors_reshaped = errors.reshape(batch_size, n_frames, n_joints, 3, 1)
        whitened_errors = np.linalg.solve(L, errors_reshaped)
        mahalanobis_distances = np.sqrt(np.sum(whitened_errors**2, axis=3)).squeeze()

    # Initialize per-frame coverage stats
    frame_coverage_stats = {frame: {mult: 0.0 for mult in std_multipliers} for frame in range(n_frames)}

    # Compute coverage per frame
    for frame in range(n_frames):
        for mult in std_multipliers:
            frame_coverage = (mahalanobis_distances[:, frame, :] <= mult).mean()
            frame_coverage_stats[frame][mult] = frame_coverage

    print("\nPer-frame coverage statistics:")
    for frame in range(n_frames):
        print(f"\nFrame {frame + 1}:")
        for mult in std_multipliers:
            expected = expected_coverage[mult]
            actual = frame_coverage_stats[frame][mult]
            error = abs(actual - expected)
            print(f"{mult}σ - Actual: {actual:.1%}, Expected: {expected:.1%}, Error: {error:.1%}")

    return coverage_stats
