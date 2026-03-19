"""Helper functions for motion prediction inference."""
from typing import Optional, Sequence, Tuple, Union
from sympy import ShapeError
from tqdm import tqdm
from time import time
import jax.numpy as jnp
import numpy as np

from human_pose_pipeline.pose_estimation.inference_helper_batched import update_motion_prediction_buffer
from human_pose_pipeline.utils.eval_utils import convert_covariance_matrices_to_set


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


def run_motion_prediction(
    points_3d_buffer: jnp.ndarray,
    covariance_buffer: jnp.ndarray,
    pose_valid_buffer: jnp.ndarray,
    motion_prediction_buffer: jnp.ndarray,
    motion_uncertainty_buffer: jnp.ndarray,
    motion_prediction_jit_fn,
    motion_prediction_params,
    motion_prediction_batch_stats,
    motion_ood_score_fn,
    n_joints: int,
    input_horizon_length: int,
    prediction_horizon_length: int,
    ood_threshold: float,
    calibration_ct: float,
    calibration_it: float,
    calibration_factors: Optional[Sequence[float]],
    n_correct_poses_required: int,
    set_likelihood: float,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, bool, bool, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one step of motion prediction: inference, OOD scoring, calibration, buffer update.

    Args:
        points_3d_buffer: Rolling pose buffer [T, J, 3]
        covariance_buffer: Rolling covariance buffer [T, J, 3, 3]
        pose_valid_buffer: Rolling validity buffer [T]
        motion_prediction_buffer: Current motion prediction buffer [P, J, 3]
        motion_uncertainty_buffer: Current motion uncertainty buffer [P, J, 3, 3]
        motion_prediction_jit_fn: JIT-compiled JAX motion prediction function
        motion_prediction_params: Model parameters
        motion_prediction_batch_stats: Batch statistics (or None)
        motion_ood_score_fn: OOD scoring function (or None)
        n_joints: Number of skeleton joints
        input_horizon_length: Length of the input pose buffer T
        prediction_horizon_length: Length of the prediction horizon P
        ood_threshold: Threshold for classifying motion as OOD
        calibration_ct: Constant time calibration factor for covariance
        calibration_it: Increasing time calibration factor for covariance
        calibration_factors: Per-joint calibration factors (or None)
        n_correct_poses_required: Consecutive valid poses needed before using predicted motion
        set_likelihood: Likelihood level for converting covariance to set radius

    Returns:
        - Updated motion_prediction_buffer [P, J, 3]
        - Updated motion_uncertainty_buffer [P, J, 3, 3]
        - motion_set_radius [P, J]
        - motion_ood_score: float
        - motion_is_ood: bool
        - valid_motion: bool
        - motion_predicted [P, J, 3]: Raw model position prediction (before buffer update)
        - motion_cov_calibrated [P, J, 3, 3]: Calibrated covariance (before buffer update)
        - motion_cov_uncalibrated [P, J, 3, 3]: Raw model covariance before calibration
    """
    pose_input = points_3d_buffer.reshape([1, input_horizon_length, n_joints * 3])
    motion_prediction_input = jnp.concatenate([
        pose_input,
        covariance_buffer.reshape([1, input_horizon_length, n_joints * 3 * 3])
    ], axis=-1)

    # Model inference
    if motion_prediction_batch_stats is not None:
        motion_predicted, (motion_cov_predicted, _) = motion_prediction_jit_fn(
            motion_prediction_params, motion_prediction_batch_stats, motion_prediction_input
        )
    else:
        motion_predicted, (motion_cov_predicted, _) = motion_prediction_jit_fn(
            motion_prediction_params, motion_prediction_input
        )

    # OOD score
    motion_ood_score = motion_ood_score_fn(pose_input) if motion_ood_score_fn is not None else 0.0

    motion_predicted = motion_predicted.reshape(-1, prediction_horizon_length, n_joints, 3)[0]
    motion_cov_predicted = motion_cov_predicted[0]
    motion_cov_uncalibrated = motion_cov_predicted

    # Calibrate covariance
    motion_cov_predicted = calibrate_covariance_matrices(
        covariance_matrices=motion_cov_predicted,
        constant_time_factor=calibration_ct,
        increase_time_factor=calibration_it,
        joint_calibration_factors=calibration_factors,
    )
    if isinstance(motion_cov_predicted, np.ndarray):
        motion_cov_predicted = jnp.array(motion_cov_predicted)

    motion_is_ood = bool(motion_ood_score > ood_threshold)

    motion_prediction_buffer, motion_uncertainty_buffer, valid_motion = update_motion_prediction_buffer(
        motion_prediction_buffer=motion_prediction_buffer,
        motion_uncertainty_buffer=motion_uncertainty_buffer,
        predicted_motion=motion_predicted,
        predicted_motion_uncertainty=motion_cov_predicted,
        is_ood=motion_is_ood,
        pose_valid_buffer=pose_valid_buffer,
        n_correct_poses_required=n_correct_poses_required,
    )

    motion_set_radius = convert_covariance_matrices_to_set(
        motion_cov_predicted, likelihood=set_likelihood
    )

    return (
        motion_prediction_buffer,
        motion_uncertainty_buffer,
        motion_set_radius,
        motion_ood_score,
        motion_is_ood,
        valid_motion,
        motion_predicted,
        motion_cov_predicted,
        motion_cov_uncalibrated,
    )


def calibrate_covariance_matrices(
    covariance_matrices: Union[jnp.ndarray, np.ndarray],
    constant_time_factor: float = 1.2,
    increase_time_factor: float = 0.4,
    joint_calibration_factors: Optional[Sequence[float]] = None
) -> Union[jnp.ndarray, np.ndarray]:
    if len(covariance_matrices.shape) == 5:
        T = covariance_matrices.shape[1]
        J = covariance_matrices.shape[2]
        if not joint_calibration_factors:
            scaling_factors_joints = np.ones(J)
        else:
            assert len(joint_calibration_factors) == J
            scaling_factors_joints = np.array(joint_calibration_factors)
        scaling_factors_times = (constant_time_factor + increase_time_factor * np.arange(T))[None, :, None, None, None]
        scaling_factors_joints = scaling_factors_joints[None, None, :, None, None]
    elif len(covariance_matrices.shape) == 4:
        T = covariance_matrices.shape[0]
        J = covariance_matrices.shape[1]
        if not joint_calibration_factors:
            scaling_factors_joints = np.ones(J)
        else:
            assert len(joint_calibration_factors) == J
            scaling_factors_joints = np.array(joint_calibration_factors)
        scaling_factors_times = (constant_time_factor + increase_time_factor * np.arange(T))[:, None, None, None]
        scaling_factors_joints = scaling_factors_joints[None, :, None, None]
    else:
        raise ShapeError(f"Covaraince matrices have incorrect shape: {covariance_matrices.shape}.")

    covariance_matrices = covariance_matrices * scaling_factors_times
    covariance_matrices = covariance_matrices * scaling_factors_joints
    return covariance_matrices
