"""Helper functions for motion prediction inference."""
from typing import Sequence, Union
from sympy import ShapeError
from tqdm import tqdm
from time import time
import jax.numpy as jnp
import numpy as np


def predict_poses(
    motion_prediction_jit_fn,
    params,
    batch_stats,
    dataset_loader,
    motion_ood_score_fn=None,
    ood_threshold=np.inf,
    max_batches=np.inf,
    device="cuda"
):
    """Evaluate the motion prediction model.

    Args:
        motion_prediction_jit_fn: JIT-compiled JAX function for motion prediction.
        params: Model parameters.
        batch_stats: Batch statistics for the model (if any).
        dataset_loader: DataLoader for the dataset.
        motion_ood_score_fn: Optional scoring function to detect OOD inputs.
        ood_threshold: Threshold for OOD detection.
        max_batches: Maximum number of batches to process.
        device: Device to run the computations on.
    Returns:
        predictions: Predicted poses. Shape: (num_samples, pred_horizon, n_joints * 3)
        targets: Ground truth poses. Shape: (num_samples, pred_horizon, n_joints * 3)
        covariance_matrices: Covariance matrices of the predictions. Shape: (num_samples, pred_horizon, n_joints * 3, n_joints * 3)
        ood_scores: OOD scores. Shape: (num_samples)
        is_oods: OOD detected. Shape: (num_samples)
        last_input_poses: Last input pose. Shape (num_samples, n_joints, 3)
    """
    predictions = []
    targets = []
    covariance_matrices = []
    ood_scores = []
    is_oods = []
    last_input_poses = []

    print("\nRunning model inference...")

    for i, batch in tqdm(enumerate(dataset_loader)):
        if i >= max_batches:
            break

        input_pose = batch[0]
        target_pose = batch[1]

        # To JAX arrays
        input_pose = jnp.array(input_pose, dtype=jnp.float32)
        target_pose = jnp.array(target_pose, dtype=jnp.float32)

        # To batch dimension
        if len(input_pose.shape) == 2:
            input_pose = jnp.expand_dims(input_pose, axis=0)
            target_pose = jnp.expand_dims(target_pose, axis=0)

        # Model inference
        t0 = time()
        if batch_stats is not None:
            pred_poses, (cov, L) = motion_prediction_jit_fn(params, batch_stats, input_pose)
        else:
            pred_poses, (cov, L) = motion_prediction_jit_fn(params, input_pose)
        t1 = time()
        # print(f"  Processed batch {i + 1} in {(t1 - t0) * 1000:.2f} ms")
        if motion_ood_score_fn is not None:
            motion_ood_score = motion_ood_score_fn(input_pose)
        else:
            motion_ood_score = jnp.zeros(input_pose.shape[0], dtype=jnp.float32)
        motion_is_ood = motion_ood_score > ood_threshold
        predictions.append(pred_poses)
        targets.append(target_pose)
        covariance_matrices.append(cov)
        ood_scores.append(motion_ood_score)
        is_oods.append(motion_is_ood)
        last_input_poses.append(input_pose[:, -1, ...])

    predictions = jnp.concatenate(predictions, axis=0)
    targets = jnp.concatenate(targets, axis=0)
    covariance_matrices = jnp.concatenate(covariance_matrices, axis=0)
    ood_scores = jnp.concatenate(ood_scores, axis=0)
    is_oods = jnp.concatenate(is_oods, axis=0)
    last_input_poses = jnp.concatenate(last_input_poses, axis=0)
    return predictions, targets, covariance_matrices, ood_scores, is_oods, last_input_poses


def compute_covariance_matrices(log_var, raw_cov):
    """Compute covariance matrices from predicted log-variances and raw covariance factors.

    Args:
        log_var: Log variances [B, T, J, 3]
        raw_cov: Raw covariance factors [B, T, J, 3]
    Returns:
        cov_matrix: Covariance matrices [B, T, J, 3, 3]
    """
    B, T, J, C = log_var.shape

    # Compute variances
    variance = jnp.exp(log_var)

    var_x, var_y, var_z = variance[..., 0], variance[..., 1], variance[..., 2]

    # Construct Cholesky factors for each frame separately
    L = jnp.zeros((B, T, J, C, C))
    eps = 0

    # Lower triangular Cholesky factor (using JAX's immutable array updates)
    L = L.at[..., 0, 0].set(jnp.sqrt(var_x + eps) * 1000)
    L = L.at[..., 1, 0].set(raw_cov[..., 0] * jnp.sqrt(var_x + eps) * 1000)
    L = L.at[..., 1, 1].set(jnp.sqrt(var_y + eps) * 1000)
    L = L.at[..., 2, 0].set(raw_cov[..., 1] * jnp.sqrt(var_x + eps) * 1000)
    L = L.at[..., 2, 1].set(raw_cov[..., 2] * jnp.sqrt(var_y + eps) * 1000)
    L = L.at[..., 2, 2].set(jnp.sqrt(var_z + eps) * 1000)

    # Compute full covariance matrix from Cholesky factors
    cov_matrix = jnp.matmul(L, jnp.matrix_transpose(L))
    return cov_matrix


def calibrate_covariance_matrices(
    covariance_matrices: Union[jnp.ndarray, np.ndarray],
    constant_time_factor: float = 1.2,
    increase_time_factor: float = 0.4,
    hand_factor: float = 1.7,
    feet_factor: float = 1.5,
    hand_indices: Sequence[int] = [5, 6],
    feet_indices: Sequence[int] = [11, 12]
) -> Union[jnp.ndarray, np.ndarray]:
    if len(covariance_matrices.shape) == 5:
        T = covariance_matrices.shape[1]
        J = covariance_matrices.shape[2]
        scaling_factors_joints = np.ones(J)
        scaling_factors_joints[hand_indices] = hand_factor
        scaling_factors_joints[feet_indices] = feet_factor
        scaling_factors_times = (constant_time_factor + increase_time_factor * np.arange(T))[None, :, None, None, None]
        scaling_factors_joints = scaling_factors_joints[None, None, :, None, None]
    elif len(covariance_matrices.shape) == 4:
        T = covariance_matrices.shape[0]
        J = covariance_matrices.shape[1]
        scaling_factors_joints = np.ones(J)
        scaling_factors_joints[hand_indices] = hand_factor
        scaling_factors_joints[feet_indices] = feet_factor
        scaling_factors_times = (constant_time_factor + increase_time_factor * np.arange(T))[:, None, None, None]
        scaling_factors_joints = scaling_factors_joints[None, :, None, None]
    else:
        raise ShapeError(f"Covaraince matrices have incorrect shape: {covariance_matrices.shape}.")

    covariance_matrices = covariance_matrices * scaling_factors_times
    covariance_matrices = covariance_matrices * scaling_factors_joints
    return covariance_matrices
