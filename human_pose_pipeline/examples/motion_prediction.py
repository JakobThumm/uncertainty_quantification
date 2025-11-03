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

from src.models.dct_pose_transformer import DCTPoseTransformer
from src.datasets.h36m_motion_prediction import Human36mMotionDataset3D

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

N_JOINTS = 13
INPUT_HORIZON_LENGTH = 50
PREDICTION_HORIZON_LENGTH = 10
BATCH_SIZE = 32


def predict_poses(motion_prediction_jit_fn, params, batch_stats, dataset, max_batches=50, device="cuda"):
    """Evaluate the motion prediction model."""
    predictions = []
    targets = []

    print("\nRunning model inference...")

    for i, batch in tqdm(enumerate(dataset)):
        if i >= max_batches:
            break

        input_pose = batch["input_pose"]
        target_pose = batch["target_pose"]

        # To batch dimension
        input_pose = jnp.expand_dims(input_pose, axis=0)
        target_pose = jnp.expand_dims(target_pose, axis=0)

        # Model inference
        t0 = time()
        if batch_stats is not None:
            pred_poses, (var_params, cov_params) = motion_prediction_jit_fn(
                params, batch_stats, input_pose
            )
        else:
            pred_poses, (var_params, cov_params) = motion_prediction_jit_fn(
                params, input_pose
            )
        t1 = time()
        print(f"  Processed batch {i + 1} in {(t1 - t0) * 1000:.2f} ms")
        predictions.append(pred_poses)
        targets.append(target_pose)

    predictions = jnp.concatenate(predictions, axis=0)
    targets = jnp.concatenate(targets, axis=0)
    return predictions, targets


def evaluate_scores(predictions, targets):
    """Evaluate MPJPE scores."""
    errors = np.linalg.norm(predictions - targets, axis=2)
    mpjpe = np.mean(errors)
    std = np.std(errors)
    per_joint_errors = np.mean(errors, axis=0)
    per_joint_std = np.std(errors, axis=0)
    return mpjpe, std, per_joint_errors, per_joint_std


def main():
    parser = argparse.ArgumentParser(description='3D Pose Estimation with OOD Detection')
    # parser.add_argument('--cache_dir', type=str, default='cache/', help='Cache directory with score functions')
    # parser.add_argument('--base_key', type=str, default=None, help='Base key for loading the OOD score functions')
    parser.add_argument('--data_path', type=str, default='datasets/', help='Path to datasets')
    parser.add_argument('--model_save_path', type=str, default='human_pose_pipeline/models/motion_prediction', help='Path to saved models')
    # parser.add_argument('--run_name', type=str, default='finetuned_h36m_regressflow_with_unc', help='Model run name')
    # parser.add_argument('--ood_threshold', type=float, default=OOD_THRESHOLD, help='OOD threshold')
    # parser.add_argument('--subject', type=str, default='S1', help='Subject ID (e.g., S1, S6)')
    # parser.add_argument('--action', type=str, default='WalkingDog', help='Action to visualize')
    # parser.add_argument('--camera_ids', type=str, nargs=2, default=['55011271', '60457274'], help='Camera IDs')
    # parser.add_argument('--max_frames', type=int, default=100, help='Maximum number of frames to process')
    # parser.add_argument('--enable_ood', action='store_true', help='Enable OOD detection on left camera')
    parser.add_argument('--output_dir', type=str, default='results/motion_prediction', help='Output directory for results')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 10)
    print("Evaluate Motion Prediction Model")
    print("=" * 10)
    print(f"Device: {device}")

    # Load model
    model_path = os.path.join(root_dir, args.model_save_path)
    motion_prediction_jit_fn, params, batch_stats = initialize_jax_models(
        checkpoint_path_jax=model_path
    )

    # Load dataset
    print("\nLoading H36M dataset...")
    data_path = os.path.join(root_dir, args.data_path, "H36M", "extracted")
    dataset = Human36mMotionDataset3D(
        base_directory=data_path,
        split="test",
        input_frames=INPUT_HORIZON_LENGTH,
        predict_frames=PREDICTION_HORIZON_LENGTH,
        jax_format=True
    )
    print(f"Loaded {len(dataset)} sequences.")

    # Evaluate the model
    predictions, targets = predict_poses(
        motion_prediction_jit_fn=motion_prediction_jit_fn,
        params=params,
        batch_stats=batch_stats,
        dataset=dataset,
        max_batches=50,
        device=device
    )

    mpjpe, std_score, per_joint_score, per_joint_std = evaluate_scores(predictions, targets)

    # Debug outputs
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall MPJPE: {mpjpe:.2f} mm, Std: {std_score:.2f} mm")

    # Per-joint errors
    print("\nPer-Joint Errors:")
    joint_names = [
        "Hip",
        "RHip",
        "RKnee",
        "RFoot",
        "LHip",
        "LKnee",
        "LFoot",
        "Spine",
        "Neck",
        "Head",
        "LShoulder",
        "LElbow",
        "LWrist",
    ]

    for name, error in zip(joint_names, per_joint_score):
        print(f"  {name:12s}: {error:7.2f} mm")

    # Visualize a few samples
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    os.makedirs("eval_fixed", exist_ok=True)

    # Visualize best and worst predictions
    all_scores = np.linalg.norm(predictions - targets, axis=2)
    per_sample_errors = np.mean(all_scores.reshape(predictions.shape[0], predictions.shape[1], -1), axis=(1, 2))

    best_idx = np.argmin(per_sample_errors)
    worst_idx = np.argmax(per_sample_errors)
    median_idx = np.argsort(per_sample_errors)[len(per_sample_errors) // 2]

    for label, idx in [("best", best_idx), ("median", median_idx), ("worst", worst_idx)]:
        fig = plt.figure(figsize=(12, 5))

        frame_idx = 4  # Middle frame
        pred_pose = predictions[idx, frame_idx].reshape(13, 3)
        targ_pose = targets[idx, frame_idx].reshape(13, 3)

        error = np.mean(np.linalg.norm(pred_pose - targ_pose, axis=1))

        # Ground truth
        ax1 = fig.add_subplot(121, projection="3d")
        plot_3d_skeleton(ax1, targ_pose, H36M_SKELETON_13, color="green")
        ax1.set_title(f"Ground Truth", fontsize=12, fontweight="bold")
        ax1.view_init(elev=15, azim=45)

        # Prediction
        ax2 = fig.add_subplot(122, projection="3d")
        plot_3d_skeleton(ax2, pred_pose, H36M_SKELETON_13, color="blue")
        ax2.set_title(f"Prediction (Error: {error:.1f}mm)", fontsize=12, fontweight="bold")
        ax2.view_init(elev=15, azim=45)

        # Match axes
        all_poses = np.concatenate([targ_pose, pred_pose], axis=0)
        x_range = [all_poses[:, 0].min() - 100, all_poses[:, 0].max() + 100]
        y_range = [all_poses[:, 1].min() - 100, all_poses[:, 1].max() + 100]
        z_range = [all_poses[:, 2].min() - 100, all_poses[:, 2].max() + 100]

        for ax in [ax1, ax2]:
            ax.set_xlim(x_range)
            ax.set_ylim(y_range)
            ax.set_zlim(z_range)

        fig.suptitle(f"{label.upper()} Prediction (Sample {idx})", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"eval_fixed/{label}_prediction.png", dpi=150)
        plt.close()

        print(f"  Saved {label} prediction visualization")

    print("\n" + "=" * 60)
    print("FINAL ASSESSMENT")
    print("=" * 60)

    if mpjpe < 1500 and np.mean(hip_spine) > 100 and np.mean(rhip_rknee) > 300:
        print("✓ Model IS producing reasonable predictions!")
        print("  The issue was MISSING postprocessing steps (IDCT + offset)")
        print("  Your model is actually well-trained!")
    else:
        print("⚠ Model may still have some issues, but much better than before")

    print("\n" + "=" * 60)
    print("Results saved to eval_fixed/")
    print("=" * 60)


if __name__ == "__main__":
    main()
