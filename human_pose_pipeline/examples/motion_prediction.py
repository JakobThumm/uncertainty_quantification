"""This script evaluates a motion prediction model on the Human3.6M dataset."""

import os
from time import time
import argparse
import numpy as np
from sympy import per
import torch
from torch.utils.data import DataLoader
import jax.numpy as jnp
from tqdm import tqdm
from human_pose_pipeline.pose_estimation.inference_helper import initialize_jax_models
from human_pose_pipeline.motion_prediction.inference_helper import predict_poses
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
from human_pose_pipeline.utils.eval_utils import evaluate_pose_prediction_scores_np as evaluate_scores

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

BATCH_SIZE = 128


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

    # Load dataset
    print("\nLoading H36M dataset...")
    data_path = os.path.join(root_dir, args.data_path)  # , "H36M", "extracted")
    # dataset_name = "Human36mMotionDataset3DWithInputUncertainty"
    dataset_name = "Human36mMotionDataset3D"
    train_loader, valid_loader, test_loader = dataloader_from_string(
        dataset_name,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=420,
        download=False,  # False
        data_path=data_path,
    )
    # print(f"Loaded {len(dataset)} sequences.")

    # >>> Validation set <<<
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    predictions, targets, covariance_matrices = predict_poses(
        motion_prediction_jit_fn=motion_prediction_jit_fn,
        params=params,
        batch_stats=batch_stats,
        dataset_loader=valid_loader,
        device=device,
    )
    coverage_stats = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=predictions, true_poses=targets, cov_matrices=covariance_matrices, std_multipliers=[1, 2, 3, 4]
    )

    predictions = predictions.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)
    targets = targets.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)

    mpjpe, std_score, per_time_errors, per_time_stds, per_joint_errors, per_joint_std = evaluate_scores(
        predictions, targets
    )

    print(f"\nOverall MPJPE: {mpjpe:.2f} mm, Std: {std_score:.2f} mm")

    # Per-joint errors
    print("\nPer-Time Errors:")
    for i, error in enumerate(per_time_errors):
        print(f"Time point {i + 1} error = {error:7.2f} mm")

    print("\nPer-Joint Errors:")
    for i, error in enumerate(per_joint_errors):
        print(f"Joint {i + 1} error = {error:7.2f} mm")

    # >>> Test set <<<
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    predictions, targets, covariance_matrices = predict_poses(
        motion_prediction_jit_fn=motion_prediction_jit_fn,
        params=params,
        batch_stats=batch_stats,
        dataset_loader=test_loader,
        device=device,
    )
    coverage_stats = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=predictions, true_poses=targets, cov_matrices=covariance_matrices, std_multipliers=[1, 2, 3, 4]
    )

    predictions = predictions.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)
    targets = targets.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)

    mpjpe, std_score, per_time_errors, per_time_stds, per_joint_errors, per_joint_std = evaluate_scores(
        predictions, targets
    )

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
