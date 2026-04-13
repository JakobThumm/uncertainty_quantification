"""Utilities for evaluating uncertainty estimates in human pose predictions."""

from typing import Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp
import csv
import os
from pathlib import Path


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
    # Mask out all zero predictions
    mask_predictions = np.all(predictions == 0.0, axis=(2, 3))  # [B, T]
    mask_targets = np.all(targets == 0.0, axis=(2, 3))
    mask = np.logical_or(mask_predictions, mask_targets)
    errors = np.linalg.norm(predictions - targets, axis=-1)  # Shape = [B, T, J]
    masked_errors = np.ma.array(errors, mask=np.repeat(mask[:, :, np.newaxis], errors.shape[-1], axis=-1))
    mpjpe = float(masked_errors.mean())
    std = float(masked_errors.std())
    per_time_errors = np.array(masked_errors.mean(axis=(0, 2)))  # Shape = [T]
    per_time_std = np.array(masked_errors.std(axis=(0, 2)))  # Shape = [T]
    per_joint_errors = np.array(masked_errors.mean(axis=(0, 1)))  # Shape = [J]
    per_joint_std = np.array(masked_errors.std(axis=(0, 1)))  # Shape = [J]
    per_prediction_error = np.array(masked_errors.mean(axis=(1, 2)))
    per_dimension_error = np.mean(np.abs(predictions - targets), axis=(0, 1, 2))
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
    # Mask out all zero predictions: full_mask is True where valid
    valid_predictions = ~jnp.all(predictions == 0.0, axis=(2, 3))  # [B, T]
    valid_targets = ~jnp.all(targets == 0.0, axis=(2, 3))
    valid = jnp.logical_and(valid_predictions, valid_targets)
    errors = jnp.linalg.norm(predictions - targets, axis=-1)  # [B, T, J]
    full_mask = jnp.repeat(valid[:, :, jnp.newaxis], errors.shape[-1], axis=-1)  # [B, T, J]
    mpjpe = jnp.where(full_mask, errors, 0).sum() / full_mask.sum()
    std = jnp.sqrt(jnp.where(full_mask, (errors - mpjpe) ** 2, 0).sum() / full_mask.sum())
    n_per_t = full_mask.sum(axis=(0, 2))  # [T]
    per_time_errors = jnp.where(full_mask, errors, 0).sum(axis=(0, 2)) / n_per_t  # [T]
    per_time_std = jnp.sqrt(
        jnp.where(full_mask, (errors - per_time_errors[None, :, None]) ** 2, 0).sum(axis=(0, 2)) / n_per_t
    )
    n_per_j = full_mask.sum(axis=(0, 1))  # [J]
    per_joint_errors = jnp.where(full_mask, errors, 0).sum(axis=(0, 1)) / n_per_j  # [J]
    per_joint_std = jnp.sqrt(
        jnp.where(full_mask, (errors - per_joint_errors[None, None, :]) ** 2, 0).sum(axis=(0, 1)) / n_per_j
    )
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
    from jax.scipy.stats import chi2
    # Mask out all zero predictions: full_mask is True where valid
    valid_predictions = ~jnp.all(pred_poses == 0.0, axis=(2, 3))  # [B, T]
    valid_targets = ~jnp.all(true_poses == 0.0, axis=(2, 3))  # [B, T]
    valid = jnp.logical_and(valid_predictions, valid_targets)
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
    mahal = jnp.sum(m**2, axis=-1).reshape(B, T, J)  # [B, T, J]
    full_mask = jnp.repeat(valid[:, :, jnp.newaxis], J, axis=-1)  # [B, T, J]

    # Dimension = 3
    df = 3
    results = []

    for k in std_multipliers:
        # Expected Gaussian coverage for k std in 3D:
        # Probability that chi-square(df) < k^2
        expected = chi2.cdf(k * k, df=df)

        # Empirical coverage over valid entries only
        inside = mahal < (k * k)                 # ellipsoid boundary [B, T, J]
        empirical = jnp.where(full_mask, inside, 0).sum() / full_mask.sum()

        # Error = expected - empirical
        results.append(expected - empirical)

    return results


def evaluate_uncertainty_coverage_with_covariance(pred_poses, true_poses, cov_matrices):
    """Evaluate the uncertainty coverage for 3D poses.

    Args:
        pred_poses: predicted 3D poses, [batch_size, n_frames, n_joints, 3]
        true_poses: ground truth 3D poses, [batch_size, n_frames, n_joints, 3]
        cov_matrices: predicted covariance matrices, [batch_size, n_frames, n_joints, 3, 3]
    Returns:
        coverage_stats dict with keys (i = 1, 2, 3, 4):
            "overall_within_{i}std", "per_joint_within_{i}std", "per_frame_within_{i}std"
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

    batch_size, n_frames, n_joints, _ = pred_poses.shape

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
        mahalanobis_distances_squared = np.sum(whitened_errors**2, axis=3).squeeze(axis=-1)  # Shape: (B, T, J)

    except np.linalg.LinAlgError:
        print("Warning: Cholesky decomposition failed, adding more regularization")
        # Add more regularization and retry
        cov_matrices = cov_matrices + np.eye(3)[None, None, None, :, :] * 1e-4
        L = np.linalg.cholesky(cov_matrices)
        errors_reshaped = errors.reshape(batch_size, n_frames, n_joints, 3, 1)
        whitened_errors = np.linalg.solve(L, errors_reshaped)
        mahalanobis_distances_squared = np.sum(whitened_errors**2, axis=3).squeeze()

    # Mask out all zero predictions: full_mask True = invalid (matches np.ma convention)
    inv_mask_predictions = np.all(pred_poses == 0.0, axis=(2, 3))  # [B, T]
    inv_mask_targets = np.all(true_poses == 0.0, axis=(2, 3))  # [B, T]
    inv_mask = np.logical_or(inv_mask_predictions, inv_mask_targets)
    full_mask = np.repeat(inv_mask[:, :, np.newaxis], n_joints, axis=-1)  # [B, T, J]

    within_stds = [mahalanobis_distances_squared <= threshold for threshold in thresholds]
    # Compute per-joint/frame coverage over valid entries only
    within_std_joint = [np.ma.array(ws, mask=full_mask).mean(axis=(0, 1)) for ws in within_stds]
    within_std_frame = [np.ma.array(ws, mask=full_mask).mean(axis=(0, 2)) for ws in within_stds]
    overall_coverage = [float(np.ma.array(ws, mask=full_mask).mean()) for ws in within_stds]

    # Create results dictionary
    coverage_stats = {}
    for i in range(len(within_stds)):
        coverage_stats[f'overall_within_{i + 1}std'] = overall_coverage[i]
        coverage_stats[f'per_joint_within_{i + 1}std'] = within_std_joint[i]
        coverage_stats[f'per_frame_within_{i + 1}std'] = within_std_frame[i]

    return coverage_stats, within_stds


def print_mpjpe_results(
    mpjpe,
    per_time_errors,
    per_joint_errors,
    print_per_time_errors=True,
    print_per_joint_errors=True
):
    """Print evaluation results."""
    print(f"\nOverall MPJPE: {mpjpe:.2f} mm")

    # Per-time errors
    if print_per_time_errors:
        print("\nPer-Time Errors:")
        for i, error in enumerate(per_time_errors):
            print(f"  Time point {i + 1} error = {error:7.2f} mm")

    # Per-joint errors
    if print_per_joint_errors:
        print("\nPer-Joint Errors:")
        for i, error in enumerate(per_joint_errors):
            print(f"  Joint {i + 1} error = {error:7.2f} mm")


def print_coverage_stats(
    coverage_stats,
    print_per_time_stats=True,
    print_per_joint_stats=True
):
    """Print coverage statistics."""
    print("\nUncertainty Coverage Stats:")
    for mult in [1, 2, 3, 4]:
        overall_cov = coverage_stats[f"overall_within_{mult}std"]
        print(f"  Overall coverage within {mult} std: {overall_cov * 100:.2f}%")
    if print_per_time_stats:
        print("\nPer-Time Coverage Stats:")
        for mult in [1, 2, 3, 4]:
            print(f"\n  Overall coverage within {mult} std:")
            per_frame_within = coverage_stats[f"per_frame_within_{mult}std"]
            for i, percent_within in enumerate(per_frame_within):
                print(f"    Frame {i}: {percent_within * 100:.2f}%")
    if print_per_joint_stats:
        print("\nPer-Joint Coverage Stats:")
        for mult in [1, 2, 3, 4]:
            print(f"\n  Overall coverage within {mult} std:")
            per_joint_within = coverage_stats[f"per_joint_within_{mult}std"]
            for i, percent_within in enumerate(per_joint_within):
                print(f"    Joint {i}: {percent_within * 100:.2f}%")


def save_mpjpe_results(
    mpjpe,
    per_time_errors,
    per_joint_errors,
    split="test",
    output_dir="results/motion_prediction"
):
    """Save MPJPE evaluation results to CSV files.

    Args:
        mpjpe: Overall MPJPE value
        per_time_errors: Per-time MPJPE errors, shape [T]
        per_joint_errors: Per-joint MPJPE errors, shape [J]
        split: Split name (e.g., 'train', 'val', 'test')
        output_dir: Output directory for CSV files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save overall MPJPE results
    overall_file = os.path.join(output_dir, f"mpjpe_results_{split}.csv")
    with open(overall_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['overall_mpjpe_mm', f'{mpjpe:.2f}'])
    print(f"Saved overall MPJPE results to {overall_file}")

    # Save per-time MPJPE results
    per_time_file = os.path.join(output_dir, f"per_time_mpjpe_results_{split}.csv")
    with open(per_time_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_point', 'mpjpe_mm'])
        for i, error in enumerate(per_time_errors):
            writer.writerow([i + 1, f'{error:.2f}'])
    print(f"Saved per-time MPJPE results to {per_time_file}")

    # Save per-joint MPJPE results
    per_joint_file = os.path.join(output_dir, f"per_joint_mpjpe_results_{split}.csv")
    with open(per_joint_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['joint', 'mpjpe_mm'])
        for i, error in enumerate(per_joint_errors):
            writer.writerow([i + 1, f'{error:.2f}'])
    print(f"Saved per-joint MPJPE results to {per_joint_file}")


def save_coverage_stats(
    coverage_stats,
    split="test",
    output_dir="results/motion_prediction"
):
    """Save uncertainty coverage statistics to CSV files.

    Args:
        coverage_stats: Dictionary containing coverage statistics with keys:
            - 'overall_within_{i}std': Overall coverage for i standard deviations
            - 'per_frame_within_{i}std': Per-frame coverage, shape [T]
            - 'per_joint_within_{i}std': Per-joint coverage, shape [J]
        split: Split name (e.g., 'train', 'val', 'test')
        output_dir: Output directory for CSV files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save overall coverage stats
    overall_file = os.path.join(output_dir, f"coverage_results_{split}.csv")
    with open(overall_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['std_multiplier', 'coverage_percent'])
        for mult in [1, 2, 3, 4]:
            overall_cov = coverage_stats[f"overall_within_{mult}std"]
            writer.writerow([mult, f'{overall_cov * 100:.2f}'])
    print(f"Saved overall coverage results to {overall_file}")

    # Save per-time coverage stats
    per_time_file = os.path.join(output_dir, f"per_time_coverage_results_{split}.csv")
    with open(per_time_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: time_point, coverage_1std, coverage_2std, coverage_3std, coverage_4std
        writer.writerow(['time_point', 'coverage_1std_percent', 'coverage_2std_percent',
                        'coverage_3std_percent', 'coverage_4std_percent'])

        # Get the length of per-frame arrays
        n_frames = len(coverage_stats['per_frame_within_1std'])

        for i in range(n_frames):
            row = [i + 1]
            for mult in [1, 2, 3, 4]:
                per_frame_within = coverage_stats[f"per_frame_within_{mult}std"]
                row.append(f'{per_frame_within[i] * 100:.2f}')
            writer.writerow(row)
    print(f"Saved per-time coverage results to {per_time_file}")

    # Save per-joint coverage stats
    per_joint_file = os.path.join(output_dir, f"per_joint_coverage_results_{split}.csv")
    with open(per_joint_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: joint, coverage_1std, coverage_2std, coverage_3std, coverage_4std
        writer.writerow(['joint', 'coverage_1std_percent', 'coverage_2std_percent',
                        'coverage_3std_percent', 'coverage_4std_percent'])

        # Get the length of per-joint arrays
        n_joints = len(coverage_stats['per_joint_within_1std'])

        for i in range(n_joints):
            row = [i + 1]
            for mult in [1, 2, 3, 4]:
                per_joint_within = coverage_stats[f"per_joint_within_{mult}std"]
                row.append(f'{per_joint_within[i] * 100:.2f}')
            writer.writerow(row)
    print(f"Saved per-joint coverage results to {per_joint_file}")


def compute_sara_predictions(
    last_input_poses: np.ndarray,
    prediction_horizon_times: list[float],
    v_human: float = 1.6,
    measurement_uncertainty: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the reachable set with the constant velocity model of SARA.

    radius = measurement_uncertainty + time * v_human

    Args:
        last_input_poses: pose to start at. Shape: [N, J, 3]
        prediction_horizon_times: time horizons for the predictions. Shape: [T]
        v_human: Maximal velocity of the human in m/s.
        measurement_uncertainty: Uncertainty in the human pose estimate in m.
    Returns:
        - Prediction: last_input_poses repeated for all T. Shape: [N, T, J, 3]
        - Radius: Radius of reachable set spheres in mm with shape: [N, T, J]
    """
    radius = np.array([measurement_uncertainty * 1000 + time * (v_human * 1000) for time in prediction_horizon_times])
    radius = np.repeat(radius[np.newaxis, ...], last_input_poses.shape[0], axis=0)
    radius = np.repeat(radius[:, :, np.newaxis], last_input_poses.shape[1], axis=2)
    predictions = np.repeat(last_input_poses[:, np.newaxis, ...], len(prediction_horizon_times), axis=1)
    return predictions, radius


def simple_coverage_stats_sara(
    predictions: np.ndarray,
    radius: np.ndarray,
    targets: np.ndarray
):
    """Compute the simple coverage statistics for a spherical reachable set.

    Args:
        predictions: predicted poses. Shape: [N, T, J, 3]
        radius: radius of the reachable set sphere. Shape: [N, T, J]
        targets: target poses. Shape: [N, T, J, 3]
    Returns:
        - coverage_stats dict with keys:
            "overall_within_set", "per_joint_within_set", "per_frame_within_set"
        - within set object
    """
    mask_predictions = np.all(predictions == 0.0, axis=(2, 3))  # [N, T], True = invalid
    mask_targets = np.all(targets == 0.0, axis=(2, 3))  # [N, T], True = invalid
    mask = np.logical_or(mask_predictions, mask_targets)
    full_mask = np.repeat(mask[:, :, np.newaxis], predictions.shape[2], axis=-1)  # [N, T, J]
    distances = np.linalg.norm(predictions - targets, axis=-1)  # Shape: [N, T, J]
    within_set = distances <= radius
    masked_within_set = np.ma.array(within_set, mask=full_mask)
    masked_radius = np.ma.array(radius / 1000.0, mask=full_mask)
    coverage_stats = {
        "overall_within_set": float(masked_within_set.mean()),
        "per_joint_within_set": np.array(masked_within_set.mean(axis=(0, 1))),
        "per_frame_within_set": np.array(masked_within_set.mean(axis=(0, 2))),
        "overall_volume": 4.0 / 3.0 * np.pi * np.power(float(masked_radius.mean()), 3.0),
        "per_joint_volume": 4.0 / 3.0 * np.pi * np.power(np.array(masked_radius.mean(axis=(0, 1))), 3.0),
        "per_frame_volume": 4.0 / 3.0 * np.pi * np.power(np.array(masked_radius.mean(axis=(0, 2))), 3.0),
    }
    return coverage_stats, within_set


def save_coverage_stats_sara(
    coverage_stats,
    filename,
    output_dir="results/motion_prediction"
):
    """Save spherical reachable set coverage statistics to CSV files.

    Args:
        coverage_stats: Dictionary with keys:
            - 'overall_within_set': scalar coverage
            - 'per_frame_within_set': per-frame coverage, shape [T]
            - 'per_joint_within_set': per-joint coverage, shape [J]
            - 'overall_volume': scalar mean sphere volume in m^3
            - 'per_frame_volume': per-frame mean volume, shape [T]
            - 'per_joint_volume': per-joint mean volume, shape [J]
        filename: Base filename (without extension, e.g. 'sara_coverage_predictions_test')
        output_dir: Output directory for CSV files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_file = os.path.join(output_dir, f"{filename}.csv")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['overall_coverage_percent', f'{coverage_stats["overall_within_set"] * 100:.2f}'])
        writer.writerow(['overall_volume_m3', f'{coverage_stats["overall_volume"]:.6f}'])

        per_frame_within = coverage_stats["per_frame_within_set"]
        per_frame_volume = coverage_stats["per_frame_volume"]
        for i in range(len(per_frame_within)):
            writer.writerow([f'frame_{i + 1}_coverage_percent', f'{per_frame_within[i] * 100:.2f}'])
            writer.writerow([f'frame_{i + 1}_volume_m3', f'{per_frame_volume[i]:.6f}'])

        per_joint_within = coverage_stats["per_joint_within_set"]
        per_joint_volume = coverage_stats["per_joint_volume"]
        for i in range(len(per_joint_within)):
            writer.writerow([f'joint_{i + 1}_coverage_percent', f'{per_joint_within[i] * 100:.2f}'])
            writer.writerow([f'joint_{i + 1}_volume_m3', f'{per_joint_volume[i]:.6f}'])

    print(f"Saved SARA coverage results to {output_file}")


def print_simple_coverage_stats_sara(
    coverage_stats,
    print_per_time_stats=True,
    print_per_joint_stats=True
):
    """Print coverage statistics."""
    overall_cov = coverage_stats["overall_within_set"]
    print(f"Overall coverage within set: {overall_cov * 100:.2f}%")
    print(f"Mean volume = {coverage_stats['overall_volume']:.4f} m^3")
    if print_per_time_stats:
        print("\nPer-Time Coverage Stats:")
        per_frame_within = coverage_stats["per_frame_within_set"]
        for i, percent_within in enumerate(per_frame_within):
            print(f"    Frame {i}: {percent_within * 100:.2f}%")
        print("\nPer-Time Volume [m^3]:")
        per_frame_volume = coverage_stats["per_frame_volume"]
        for i, volume in enumerate(per_frame_volume):
            print(f"    Frame {i}: {volume:.4f}")
    if print_per_joint_stats:
        print("\nPer-Joint Coverage Stats:")
        per_joint_within = coverage_stats["per_joint_within_set"]
        for i, percent_within in enumerate(per_joint_within):
            print(f"    Joint {i}: {percent_within * 100:.2f}%")
        print("\nPer-Joint Volume [m^3]:")
        per_joint_volume = coverage_stats["per_joint_volume"]
        for i, volume in enumerate(per_joint_volume):
            print(f"    Joint {i}: {volume:.4f}")


def convert_covariance_matrices_to_set(
    covariance_matrices: Union[np.ndarray, jnp.ndarray],
    likelihood: float
) -> np.ndarray:
    """Convert the covariance matrices to a spherical conformal prediction set X of likelihood confidence level.
    P(x \in X) >= likelihood.

    Args:
        covariance_matrices: Cov. matrices. Shape: [N, T, J, 3, 3]
        likelihood: Likelihood of points being in the set.
    Returns:
        Radius of the spherical reachable sets. Shape: [N, T, J]
    """
    from scipy.stats import chi2
    # largest eigenvalue
    # chi-square threshold for number of standard deviations in 3D
    chi_squared_val = chi2.ppf(likelihood, df=3)
    if isinstance(covariance_matrices, np.ndarray):
        lambda_max = np.max(np.linalg.eigvalsh(covariance_matrices), axis=-1)
        # sphere radius
        radius = np.sqrt(lambda_max * chi_squared_val)
    else:
        lambda_max = jnp.max(jnp.linalg.eigvalsh(covariance_matrices), axis=-1)
        # sphere radius
        radius = np.sqrt(lambda_max * chi_squared_val)

    return radius


OOD_SCORE_PERCENTILES = [0.01, 0.1, 0.5, 1, 3, 5, 10, 25, 50, 75, 90, 95, 97, 99, 99.5, 99.9, 99.99]


def print_ood_score_percentiles(scores: np.ndarray, label: str = "OOD scores") -> None:
    """Print percentiles of OOD scores.

    Args:
        scores: 1-D array of OOD scores.
        label: Description printed in the header.
    """
    scores = np.asarray(scores).ravel()
    print(f"OOD score percentiles — {label} (n={len(scores)}):")
    for p in OOD_SCORE_PERCENTILES:
        print(f"  p{p:6.2f}: {np.percentile(scores, p):.6f}")


def save_ood_score_percentiles(
    scores: np.ndarray,
    label: str = "ood_scores",
    output_dir: str = "results",
) -> None:
    """Save percentiles of OOD scores to a CSV file.

    Args:
        scores: 1-D array of OOD scores.
        label: Used as the filename stem (spaces replaced with underscores).
        output_dir: Directory in which to write the CSV.
    """
    scores = np.asarray(scores).ravel()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = label.replace(" ", "_") + "_percentiles.csv"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["percentile", "value"])
        for p in OOD_SCORE_PERCENTILES:
            writer.writerow([p, f"{np.percentile(scores, p):.6f}"])
    print(f"Saved OOD score percentiles to {filepath}")


def print_motion_validity_stats(
    motions_is_valid: np.ndarray,
    motions_is_ood: np.ndarray,
    pose_buffers_good: np.ndarray,
) -> None:
    """Print motion prediction validity, OOD rate, and pose buffer statistics.

    Args:
        motions_is_valid: Boolean array, one entry per attempted motion prediction.
        motions_is_ood: Boolean array, one entry per attempted motion prediction.
        pose_buffers_good: Boolean array, one entry per processed frame.
            Frames where pose_buffer_good=False produced no motion output at all.
    """
    motions_is_valid = np.asarray(motions_is_valid)
    motions_is_ood = np.asarray(motions_is_ood)
    pose_buffers_good = np.asarray(pose_buffers_good)

    n_pose_good = int(np.sum(pose_buffers_good))
    n_pose_bad = len(pose_buffers_good) - n_pose_good
    n_invalid = int(np.sum(~motions_is_valid))
    n_no_output = n_pose_bad + n_invalid

    print(f"Motion validity rate:  {np.mean(motions_is_valid):.4f} "
          f"({int(np.sum(motions_is_valid))}/{len(motions_is_valid)})")
    print(f"Motion OOD rate:       {np.mean(motions_is_ood):.4f} "
          f"({int(np.sum(motions_is_ood))}/{len(motions_is_ood)})")
    print(f"Pose buffer good/bad:  {n_pose_good}/{n_pose_bad} "
          f"(ratio good/all = {n_pose_good / len(pose_buffers_good):.4f})")
    print(f"No motion output:      {n_no_output} "
          f"(bad pose buffer: {n_pose_bad}, invalid motion: {n_invalid})")


def save_motion_validity_stats(
    motions_is_valid: np.ndarray,
    motions_is_ood: np.ndarray,
    pose_buffers_good: np.ndarray,
    output_dir: str = "results",
) -> None:
    """Save motion prediction validity, OOD rate, and pose buffer statistics to CSV.

    Args:
        motions_is_valid: Boolean array, one entry per attempted motion prediction.
        motions_is_ood: Boolean array, one entry per attempted motion prediction.
        pose_buffers_good: Boolean array, one entry per processed frame.
        output_dir: Directory in which to write the CSV.
    """
    motions_is_valid = np.asarray(motions_is_valid)
    motions_is_ood = np.asarray(motions_is_ood)
    pose_buffers_good = np.asarray(pose_buffers_good)

    n_pose_good = int(np.sum(pose_buffers_good))
    n_pose_bad = len(pose_buffers_good) - n_pose_good
    n_invalid = int(np.sum(~motions_is_valid))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(output_dir, "motion_validity_stats.csv")
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "rate", "count", "total"])
        writer.writerow([
            "validity_rate",
            f"{np.mean(motions_is_valid):.6f}",
            int(np.sum(motions_is_valid)),
            len(motions_is_valid),
        ])
        writer.writerow([
            "ood_rate",
            f"{np.mean(motions_is_ood):.6f}",
            int(np.sum(motions_is_ood)),
            len(motions_is_ood),
        ])
        writer.writerow([
            "pose_buffer_good_rate",
            f"{n_pose_good / len(pose_buffers_good):.6f}",
            n_pose_good,
            len(pose_buffers_good),
        ])
        writer.writerow([
            "no_motion_output_bad_pose_buffer",
            "",
            n_pose_bad,
            len(pose_buffers_good),
        ])
        writer.writerow([
            "no_motion_output_invalid_motion",
            "",
            n_invalid,
            len(motions_is_valid),
        ])
        writer.writerow([
            "no_motion_output_total",
            "",
            n_pose_bad + n_invalid,
            len(pose_buffers_good),
        ])
    print(f"Saved motion validity stats to {filepath}")
