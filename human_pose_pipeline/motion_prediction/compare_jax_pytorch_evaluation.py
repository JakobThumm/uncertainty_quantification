"""Unified evaluation script to compare JAX and PyTorch models on exactly the same data.

This script evaluates both the JAX and PyTorch motion prediction models on the exact same
validation and test data to ensure fair comparison.
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import jax.numpy as jnp
from tqdm import tqdm

# JAX model imports
from human_pose_pipeline.pose_estimation.inference_helper import initialize_jax_models

# Add marian_code directory to path for PyTorch model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../marian_code/Experiment1/13_Joints"))

# PyTorch model imports
from model_prediction_transformer import DCTPoseTransformer

# Dataset imports
from src.datasets import dataloader_from_string

# Evaluation utilities
from human_pose_pipeline.utils.eval_utils import (
    evaluate_uncertainty_coverage_with_covariance,
    evaluate_pose_prediction_scores_np as evaluate_scores
)
from human_pose_pipeline.motion_prediction.h36m_settings import (
    INPUT_HORIZON_LENGTH,
    PREDICTION_HORIZON_LENGTH,
    N_JOINTS
)

# Get the root directory of the uncertainty_quantification project
# This script should be run from the uncertainty_quantification root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

BATCH_SIZE = 64  # Using the same batch size as PyTorch script for consistency


def get_dct_matrix(N):
    """Get DCT and inverse DCT matrices."""
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def evaluate_jax_model(jax_fn, params, batch_stats, dataset_loader, device):
    """Evaluate JAX model on dataset.

    Args:
        jax_fn: JIT-compiled JAX function
        params: Model parameters
        batch_stats: Batch statistics
        dataset_loader: DataLoader
        device: Device (not used for JAX, but kept for API consistency)

    Returns:
        predictions: [N, T, J*3]
        targets: [N, T, J*3]
        covariances: [N, T, J, 3, 3]
    """
    predictions = []
    targets = []
    covariance_matrices = []

    print("\nRunning JAX model inference...")
    for batch in tqdm(dataset_loader):
        # Dataset returns [input_pose, target_pose] format
        input_pose = batch[0]
        target_pose = batch[1]

        # Convert to JAX arrays
        input_pose = jnp.array(input_pose, dtype=jnp.float32)
        target_pose = jnp.array(target_pose, dtype=jnp.float32)

        # Model inference
        if batch_stats is not None:
            pred_poses, (cov, L) = jax_fn(params, batch_stats, input_pose)
        else:
            pred_poses, (cov, L) = jax_fn(params, input_pose)

        predictions.append(pred_poses)
        targets.append(target_pose)
        covariance_matrices.append(cov)

    predictions = jnp.concatenate(predictions, axis=0)
    targets = jnp.concatenate(targets, axis=0)
    covariance_matrices = jnp.concatenate(covariance_matrices, axis=0)

    return np.array(predictions), np.array(targets), np.array(covariance_matrices)


def evaluate_pytorch_model(model, dataset_loader, device, dct_m, idct_m):
    """Evaluate PyTorch model on dataset.

    Args:
        model: PyTorch model
        dataset_loader: DataLoader
        device: PyTorch device
        dct_m: DCT matrix
        idct_m: Inverse DCT matrix

    Returns:
        predictions: [N, T, J*3]
        targets: [N, T, J*3]
        covariances: [N, T, J, 3, 3]
    """
    model.eval()
    predictions = []
    targets = []
    covariance_matrices = []

    print("\nRunning PyTorch model inference...")
    with torch.no_grad():
        for batch in tqdm(dataset_loader):
            # Dataset returns [input_pose, target_pose] format
            input_pose = batch[0].to(device)
            target_pose = batch[1].to(device)

            # Apply DCT and normalize (PyTorch preprocessing)
            input_pose_dct = torch.matmul(input_pose.transpose(1, 2), dct_m.transpose(0, 1)).transpose(1, 2)
            input_pose_dct = input_pose_dct / 1000

            # Forward pass
            pred_poses, (log_vars, raw_covs) = model(input_pose_dct)

            # Denormalize and apply IDCT
            pred_poses = pred_poses * 1000
            pred_poses = torch.matmul(pred_poses.transpose(1, 2), idct_m.transpose(0, 1)).transpose(1, 2)

            # Add offset from last input frame
            offset = input_pose[:, -1:, :]
            pred_poses = pred_poses[:, :PREDICTION_HORIZON_LENGTH, :] + offset

            # Build covariance matrices from Cholesky decomposition
            variance = torch.exp(log_vars)
            var_x, var_y, var_z = variance[..., 0], variance[..., 1], variance[..., 2]

            B, T, J_flat = pred_poses.shape
            J = J_flat // 3

            L = torch.zeros(B, T, J, 3, 3, device=device)
            eps = 0

            # Cholesky lower triangular matrix
            L[..., 0, 0] = torch.sqrt(var_x + eps) * 1000
            L[..., 1, 0] = raw_covs[..., 0] * torch.sqrt(var_x + eps) * 1000
            L[..., 1, 1] = torch.sqrt(var_y + eps) * 1000
            L[..., 2, 0] = raw_covs[..., 1] * torch.sqrt(var_x + eps) * 1000
            L[..., 2, 1] = raw_covs[..., 2] * torch.sqrt(var_y + eps) * 1000
            L[..., 2, 2] = torch.sqrt(var_z + eps) * 1000

            # Compute full covariance matrix: cov = L @ L^T
            cov_matrices = torch.matmul(L, L.transpose(-1, -2))

            predictions.append(pred_poses.cpu().numpy())
            targets.append(target_pose.cpu().numpy())
            covariance_matrices.append(cov_matrices.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    covariance_matrices = np.concatenate(covariance_matrices, axis=0)

    return predictions, targets, covariance_matrices


def print_results(predictions, targets, covariances, model_name, split_name):
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print(f"{model_name} - {split_name.upper()} RESULTS")
    print("=" * 80)

    # Reshape for evaluation
    predictions_reshaped = predictions.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)
    targets_reshaped = targets.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)

    # Compute MPJPE scores
    mpjpe, std_score, per_time_errors, per_time_stds, per_joint_errors, per_joint_std = evaluate_scores(
        predictions_reshaped, targets_reshaped
    )

    print(f"\nOverall MPJPE: {mpjpe:.2f} mm, Std: {std_score:.2f} mm")

    # Per-time errors
    print("\nPer-Time Errors:")
    for i, error in enumerate(per_time_errors):
        print(f"  Time point {i + 1} error = {error:7.2f} mm")

    # Per-joint errors
    print("\nPer-Joint Errors:")
    for i, error in enumerate(per_joint_errors):
        print(f"  Joint {i + 1} error = {error:7.2f} mm")

    # Uncertainty coverage
    coverage_stats = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=predictions, true_poses=targets, cov_matrices=covariances
    )

    print("\nUncertainty Coverage Stats:")
    for mult in [1, 2, 3, 4]:
        overall_cov = coverage_stats[f"overall_within_{mult}std"]
        print(f"  Overall coverage within {mult} std: {overall_cov * 100:.2f}%")

    return {
        'mpjpe': mpjpe,
        'std': std_score,
        'per_time_errors': per_time_errors,
        'per_joint_errors': per_joint_errors,
        'coverage_stats': coverage_stats
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare JAX and PyTorch motion prediction models on identical data"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/",
        help="Path to datasets"
    )
    parser.add_argument(
        "--jax_model_path",
        type=str,
        default="human_pose_pipeline/models/motion_prediction/004qx4td/checkpoints/stage_3/dct_pose_transformer.pickle",
        help="Path to JAX model checkpoint",
    )
    parser.add_argument(
        "--pytorch_model_path",
        type=str,
        default="marian_code/Experiment1/13_Joints/checkpoints/model_checkpoint_prediction_transformer_end_to_end.pth",
        help="Path to PyTorch model checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="model_comparison_results.txt",
        help="File to save comparison results"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("MOTION PREDICTION MODEL COMPARISON")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")

    # ============================================================================
    # Load Dataset (SAME for both models)
    # ============================================================================
    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    # Handle both relative and absolute paths
    if os.path.isabs(args.data_path):
        data_path = args.data_path
    else:
        data_path = os.path.join(root_dir, args.data_path)

    print(f"Root directory: {root_dir}")
    print(f"Data path: {data_path}")

    # Check if dataset exists
    expected_data_dir = os.path.join(data_path, "H36M", "extracted")
    if not os.path.exists(expected_data_dir):
        raise FileNotFoundError(
            f"Dataset directory not found at {expected_data_dir}\n"
            f"Please ensure the H36M dataset is properly extracted."
        )

    dataset_name = "Human36mMotionDataset3D"

    train_loader, valid_loader, test_loader = dataloader_from_string(
        dataset_name,
        batch_size=args.batch_size,
        shuffle=False,  # IMPORTANT: No shuffling to ensure identical data order
        seed=420,
        download=False,
        data_path=data_path,
    )
    print(f"Validation batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # ============================================================================
    # Load JAX Model
    # ============================================================================
    print("\n" + "=" * 80)
    print("LOADING JAX MODEL")
    print("=" * 80)

    # Handle both relative and absolute paths
    if os.path.isabs(args.jax_model_path):
        jax_model_path = args.jax_model_path
    else:
        jax_model_path = os.path.join(root_dir, args.jax_model_path)

    print(f"Loading from: {jax_model_path}")
    if not os.path.exists(jax_model_path):
        raise FileNotFoundError(f"JAX model not found at {jax_model_path}")

    jax_fn, jax_params, jax_batch_stats = initialize_jax_models(checkpoint_path_jax=jax_model_path)
    print("JAX model loaded successfully!")

    # ============================================================================
    # Load PyTorch Model
    # ============================================================================
    print("\n" + "=" * 80)
    print("LOADING PYTORCH MODEL")
    print("=" * 80)

    # Handle both relative and absolute paths
    if os.path.isabs(args.pytorch_model_path):
        pytorch_model_path = args.pytorch_model_path
    else:
        pytorch_model_path = os.path.join(root_dir, args.pytorch_model_path)

    print(f"Loading from: {pytorch_model_path}")
    if not os.path.exists(pytorch_model_path):
        raise FileNotFoundError(f"PyTorch model not found at {pytorch_model_path}")

    pytorch_model = DCTPoseTransformer(input_dim=39, seq_len=50)
    pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    pytorch_model.to(device)
    pytorch_model.eval()
    print("PyTorch model loaded successfully!")

    # Get DCT matrices for PyTorch model
    N = INPUT_HORIZON_LENGTH
    dct_m, idct_m = get_dct_matrix(N)
    dct_m = torch.from_numpy(dct_m).float().to(device)
    idct_m = torch.from_numpy(idct_m).float().to(device)

    # ============================================================================
    # Evaluate on Validation Set
    # ============================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SET EVALUATION")
    print("=" * 80)

    # Evaluate JAX model
    jax_val_preds, jax_val_targets, jax_val_covs = evaluate_jax_model(
        jax_fn, jax_params, jax_batch_stats, valid_loader, device
    )
    jax_val_results = print_results(
        jax_val_preds, jax_val_targets, jax_val_covs, "JAX Model", "Validation"
    )

    # Evaluate PyTorch model
    pytorch_val_preds, pytorch_val_targets, pytorch_val_covs = evaluate_pytorch_model(
        pytorch_model, valid_loader, device, dct_m, idct_m
    )
    pytorch_val_results = print_results(
        pytorch_val_preds, pytorch_val_targets, pytorch_val_covs, "PyTorch Model", "Validation"
    )

    # ============================================================================
    # Evaluate on Test Set
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)

    # Evaluate JAX model
    jax_test_preds, jax_test_targets, jax_test_covs = evaluate_jax_model(
        jax_fn, jax_params, jax_batch_stats, test_loader, device
    )
    jax_test_results = print_results(
        jax_test_preds, jax_test_targets, jax_test_covs, "JAX Model", "Test"
    )

    # Evaluate PyTorch model
    pytorch_test_preds, pytorch_test_targets, pytorch_test_covs = evaluate_pytorch_model(
        pytorch_model, test_loader, device, dct_m, idct_m
    )
    pytorch_test_results = print_results(
        pytorch_test_preds, pytorch_test_targets, pytorch_test_covs, "PyTorch Model", "Test"
    )

    # ============================================================================
    # Comparison Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Validation set comparison
    print("\nValidation Set:")
    print(f"  JAX Model MPJPE:     {jax_val_results['mpjpe']:.2f} mm")
    print(f"  PyTorch Model MPJPE: {pytorch_val_results['mpjpe']:.2f} mm")
    print(f"  Difference:          {abs(jax_val_results['mpjpe'] - pytorch_val_results['mpjpe']):.2f} mm")

    # Test set comparison
    print("\nTest Set:")
    print(f"  JAX Model MPJPE:     {jax_test_results['mpjpe']:.2f} mm")
    print(f"  PyTorch Model MPJPE: {pytorch_test_results['mpjpe']:.2f} mm")
    print(f"  Difference:          {abs(jax_test_results['mpjpe'] - pytorch_test_results['mpjpe']):.2f} mm")

    # Coverage comparison
    print("\nValidation Set Coverage (1 std):")
    jax_val_cov = jax_val_results['coverage_stats']['overall_within_1std'] * 100
    pytorch_val_cov = pytorch_val_results['coverage_stats']['overall_within_1std'] * 100
    print(f"  JAX Model:     {jax_val_cov:.2f}%")
    print(f"  PyTorch Model: {pytorch_val_cov:.2f}%")
    print(f"  Difference:    {abs(jax_val_cov - pytorch_val_cov):.2f}%")

    print("\nTest Set Coverage (1 std):")
    jax_test_cov = jax_test_results['coverage_stats']['overall_within_1std'] * 100
    pytorch_test_cov = pytorch_test_results['coverage_stats']['overall_within_1std'] * 100
    print(f"  JAX Model:     {jax_test_cov:.2f}%")
    print(f"  PyTorch Model: {pytorch_test_cov:.2f}%")
    print(f"  Difference:    {abs(jax_test_cov - pytorch_test_cov):.2f}%")

    # ============================================================================
    # Sanity Check: Are the targets identical?
    # ============================================================================
    print("\n" + "=" * 80)
    print("SANITY CHECK")
    print("=" * 80)

    val_target_diff = np.abs(jax_val_targets - pytorch_val_targets).max()
    test_target_diff = np.abs(jax_test_targets - pytorch_test_targets).max()

    print(f"\nValidation targets max difference: {val_target_diff:.6f}")
    print(f"Test targets max difference: {test_target_diff:.6f}")

    if val_target_diff < 1e-5 and test_target_diff < 1e-5:
        print("✓ Targets are identical - models were evaluated on the same data!")
    else:
        print("✗ WARNING: Targets differ - models may have been evaluated on different data!")

    # ============================================================================
    # Save results to file
    # ============================================================================
    output_path = os.path.join(root_dir, args.output_file)
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MOTION PREDICTION MODEL COMPARISON RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("VALIDATION SET\n")
        f.write("-" * 80 + "\n")
        f.write(f"JAX Model MPJPE:     {jax_val_results['mpjpe']:.2f} mm\n")
        f.write(f"PyTorch Model MPJPE: {pytorch_val_results['mpjpe']:.2f} mm\n")
        f.write(f"Difference:          {abs(jax_val_results['mpjpe'] - pytorch_val_results['mpjpe']):.2f} mm\n\n")

        f.write("TEST SET\n")
        f.write("-" * 80 + "\n")
        f.write(f"JAX Model MPJPE:     {jax_test_results['mpjpe']:.2f} mm\n")
        f.write(f"PyTorch Model MPJPE: {pytorch_test_results['mpjpe']:.2f} mm\n")
        f.write(f"Difference:          {abs(jax_test_results['mpjpe'] - pytorch_test_results['mpjpe']):.2f} mm\n\n")

        f.write("COVERAGE (1 STD)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Validation - JAX:     {jax_val_cov:.2f}%\n")
        f.write(f"Validation - PyTorch: {pytorch_val_cov:.2f}%\n")
        f.write(f"Test - JAX:           {jax_test_cov:.2f}%\n")
        f.write(f"Test - PyTorch:       {pytorch_test_cov:.2f}%\n")

    print(f"\nResults saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
