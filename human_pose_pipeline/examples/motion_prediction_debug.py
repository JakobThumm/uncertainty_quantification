"""This script evaluates a motion prediction model on the Human3.6M dataset."""

import os
from time import time
import argparse
import numpy as np
from sympy import per
import torch
from torch.utils.data import DataLoader
import jax
import jax.numpy as jnp
from tqdm import tqdm
from human_pose_pipeline.pose_estimation.inference_helper import initialize_jax_models
from human_pose_pipeline.motion_prediction.inference_helper import compute_covariance_matrices, predict_poses
from human_pose_pipeline.utils.eval_utils import evaluate_uncertainty_coverage_with_covariance
from src.datasets import dataloader_from_string
from src.models.dct_pose_transformer import DCTPoseTransformer
from src.datasets.h36m_motion_prediction import Human36mMotionDataset3D
from human_pose_pipeline.utils.visualization import visualize_motion_prediction
from human_pose_pipeline.pose_estimation.h36m_settings import CONNECTIONS_13
from human_pose_pipeline.motion_prediction.h36m_settings import (
    INPUT_HORIZON_LENGTH,
    PREDICTION_HORIZON_LENGTH,
    N_JOINTS
)

jax.config.update("jax_enable_x64", True)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

BATCH_SIZE = 128


def evaluate_scores(predictions, targets):
    """Evaluate MPJPE scores."""
    errors = np.linalg.norm(predictions - targets, axis=-1)
    mpjpe = np.mean(errors)
    std = np.std(errors)
    per_time_errors = np.mean(np.mean(errors, axis=-1), axis=0)
    per_time_std = np.std(np.mean(errors, axis=-1), axis=0)
    per_joint_errors = np.mean(np.mean(errors, axis=1), axis=0)
    per_joint_std = np.std(np.mean(errors, axis=1), axis=0)
    return mpjpe, std, per_time_errors, per_time_std, per_joint_errors, per_joint_std


def main():
    parser = argparse.ArgumentParser(description="3D Pose Estimation with OOD Detection")
    # parser.add_argument('--cache_dir', type=str, default='cache/', help='Cache directory with score functions')
    # parser.add_argument('--base_key', type=str, default=None, help='Base key for loading the OOD score functions')
    parser.add_argument("--data_path", type=str, default="datasets/", help="Path to datasets")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="human_pose_pipeline/models/motion_prediction",
        help="Path to saved models",
    )
    # parser.add_argument('--run_name', type=str, default='finetuned_h36m_regressflow_with_unc', help='Model run name')
    # parser.add_argument('--ood_threshold', type=float, default=OOD_THRESHOLD, help='OOD threshold')
    # parser.add_argument('--subject', type=str, default='S1', help='Subject ID (e.g., S1, S6)')
    # parser.add_argument('--action', type=str, default='WalkingDog', help='Action to visualize')
    # parser.add_argument('--camera_ids', type=str, nargs=2, default=['55011271', '60457274'], help='Camera IDs')
    # parser.add_argument('--max_frames', type=int, default=100, help='Maximum number of frames to process')
    # parser.add_argument('--enable_ood', action='store_true', help='Enable OOD detection on left camera')
    parser.add_argument(
        "--output_dir", type=str, default="results/motion_prediction", help="Output directory for results"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 10)
    print("Evaluate Motion Prediction Model")
    print("=" * 10)
    print(f"Device: {device}")

    # Load model
    model_path = os.path.join(root_dir, args.model_save_path)
    motion_prediction_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path_jax=model_path)

    # Load data point
    input_pose = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/input_poses_tensor.pt"))
    input_uncertainties = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/input_uncertainties_tensor.pt"))
    target_pose = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/input_poses_tensor_10.pt"))
    target_pose = target_pose[:, -PREDICTION_HORIZON_LENGTH : , ...]
    marians_model_prediction = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/pred_poses.pt"))
    # Intermediate debug outputs
    poses_flat = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/poses_flat.pt"))
    poses_dct = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/poses_dct.pt"))
    input_embed_output = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/input_embed_output.pt"))
    freq_pos_embed = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/freq_pos_embed.pt"))
    uncertainty_features = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/uncertainty_features.pt"))
    transformer_input = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/transformer_input.pt"))
    transformer_output = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/transformer_output.pt"))
    high_freq_output = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/high_freq_output.pt"))
    low_freq_output = torch.load(os.path.join(root_dir, "human_pose_pipeline/examples/test_data/low_freq_output.pt"))

    torch.set_printoptions(precision=8)
    print("Pytorch poses_flat[0] = ", poses_flat[0, 0, :])
    print("Pytorch poses_dct[0] = ", poses_dct[0, 0, :])
    print("Pytorch input_embed_output[0] = ", input_embed_output[0, 0, :])
    print("Pytorch freq_pos_embed[0] = ", freq_pos_embed[0, 0, :])
    print("Pytorch uncertainty_features[0] = ", uncertainty_features[0, 0, :])
    print("Pytorch Transformer input torch x[0] = ", transformer_input[0, 0, :])
    print("Pytorch Transformer output torch x[0] = ", transformer_output[0, 0, :])
    print("Pytorch high_freq_output[0] = ", high_freq_output[0, 0, :])
    print("Pytorch low_freq_output[0] = ", low_freq_output[0, 0, :])

    # To JAX arrays
    # IMPORTANT: Reshape first, then concatenate to avoid interleaving pose and covariance values
    input_pose_flat = input_pose.reshape(input_pose.shape[0], input_pose.shape[1], -1)  # (batch, seq, 39)
    input_uncertainties_flat = input_uncertainties.reshape(input_uncertainties.shape[0], input_uncertainties.shape[1], -1)  # (batch, seq, 117)
    input_combined = torch.cat([input_pose_flat, input_uncertainties_flat], dim=-1)  # (batch, seq, 156)
    input_pose = jnp.array(input_combined, dtype=jnp.float64)
    target_pose = jnp.array(target_pose, dtype=jnp.float64)

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
    predictions = np.array([pred_poses])
    targets = np.array([target_pose])
    covariance_matrices = np.array([(cov)])

    print(f"Model prediction: {pred_poses[0, 0, :].reshape(13, 3)}")
    print(f"Marian prediction: {marians_model_prediction[0, 0, :].reshape(13, 3)}")
    print(f"Target pose: {target_pose[0, 0, :].reshape(13, 3)}")

    predictions = predictions.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)
    targets = targets.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)
    coverage_stats, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=predictions, true_poses=targets, cov_matrices=covariance_matrices, std_multipliers=[1, 2, 3, 4]
    )
    mpjpe, std_score, per_time_errors, per_time_stds, per_joint_errors, per_joint_std = evaluate_scores(
        predictions, targets
    )

    # Debug outputs
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall MPJPE: {mpjpe:.2f} mm, Std: {std_score:.2f} mm")

    # Per-joint errors
    print("\nPer-Time Errors:")
    for i, error in enumerate(per_time_errors):
        print(f"Time point {i + 1} error = {error:7.2f} mm")

    print("\nPer-Joint Errors:")
    for i, error in enumerate(per_joint_errors):
        print(f"Joint {i + 1} error = {error:7.2f} mm")

    # Visualize a few samples
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    os.makedirs("eval_fixed", exist_ok=True)

    # Visualize best and worst predictions
    all_scores = np.linalg.norm(predictions - targets, axis=-1)
    per_sample_errors = np.mean(all_scores, axis=(1, 2))

    best_idx = np.argmin(per_sample_errors)
    worst_idx = np.argmax(per_sample_errors)
    median_idx = np.argsort(per_sample_errors)[len(per_sample_errors) // 2]

    for label, idx in [("best", best_idx), ("median", median_idx), ("worst", worst_idx)]:
        frame_idx = 4  # Middle frame
        pred_pose = predictions[idx, frame_idx].reshape(13, 3)
        targ_pose = targets[idx, frame_idx].reshape(13, 3)
        visualize_motion_prediction(
            pred_pose=np.array(pred_pose),
            target_pose=np.array(targ_pose),
            skeleton=CONNECTIONS_13,
            label=label,
            idx=idx,
            output_path=args.output_dir,
        )

        print(f"  Saved {label} prediction visualization")


if __name__ == "__main__":
    main()
