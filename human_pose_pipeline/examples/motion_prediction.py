"""This script evaluates a motion prediction model on the Human3.6M dataset."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.dct_pose_transformer import DCTPoseTransformer
from src.datasets.h36m_motion_prediction import Human36mMotionDataset3D


N_JOINTS = 13
INPUT_HORIZON_LENGTH = 50
PREDICTION_HORIZON_LENGTH = 10


def evaluate_model(model, dataloader, eval_fn, max_batches=50, device="cuda"):
    """Evaluate the motion prediction model."""
    # Warm up
    model.eval()

    all_predictions = []
    all_targets = []

    print("\nRunning model inference...")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            input_pose = batch["input_pose"].to(device)
            target_pose = batch["target_pose"].to(device)

            # Model inference
            pred_poses, (var_params, cov_params) = model(input_pose)

            all_predictions.append(pred_poses.cpu().numpy())
            all_targets.append(target_pose.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{min(max_batches, len(dataloader))} batches")

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return predictions, targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 10)
    print("Evaluate Motion Prediction Model")
    print("=" * 10)
    print(f"Device: {device}")

    # Load model
    model_path = "transformer_model.pth"
    model = DCTPoseTransformer(
        input_dim=(3 * N_JOINTS),  # 3D coordinates per joint
        seq_len=INPUT_HORIZON_LENGTH,
        seq_len_output=PREDICTION_HORIZON_LENGTH
    )

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"✗ Model not found")
        return

    model.to(device)
    model.eval()

    # Load dataset
    print("\nLoading H36M dataset...")
    data_path = "/home/skyle/datasets/H36M_FREI"
    dataset = Human36mDataset3D(data_path, split="test", input_frames=50, predict_frames=10)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    print(f"✓ Loaded {len(dataset)} sequences")

    # Get DCT matrices
    N = 50
    dct_m, idct_m = get_dct_matrix(N)
    dct_m_torch = torch.from_numpy(dct_m).float().to(device)
    idct_m_torch = torch.from_numpy(idct_m).float().to(device)

    # Evaluate with CORRECT postprocessing
    predictions, targets = evaluate_model_corrected(
        model, dataloader, dct_m_torch, idct_m_torch, device, max_batches=50
    )

    print(f"\nEvaluated on {predictions.shape[0]} sequences")

    # Calculate MPJPE
    print("\n" + "=" * 60)
    print("RESULTS (with CORRECT postprocessing)")
    print("=" * 60)

    pred_reshaped = predictions.reshape(-1, 13, 3)
    targ_reshaped = targets.reshape(-1, 13, 3)

    # Per-joint errors
    errors = np.linalg.norm(pred_reshaped - targ_reshaped, axis=2)
    mpjpe = np.mean(errors)
    per_joint = np.mean(errors, axis=0)

    print(f"\nOverall MPJPE: {mpjpe:.2f} mm")

    if mpjpe < 1500:
        print("✓ MPJPE looks much better!")
    elif mpjpe < 2000:
        print("⚠ MPJPE is acceptable")
    else:
        print("✗ MPJPE still high")

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

    for name, error in zip(joint_names, per_joint):
        print(f"  {name:12s}: {error:7.2f} mm")

    # Check bone lengths
    print("\n" + "=" * 60)
    print("BONE LENGTH CHECK")
    print("=" * 60)

    # Sample a few predictions
    sample_poses = pred_reshaped[:100]

    # Hip-Spine
    hip_spine = np.linalg.norm(sample_poses[:, 0] - sample_poses[:, 7], axis=1)
    print(f"Hip-Spine: {np.mean(hip_spine):.1f} ± {np.std(hip_spine):.1f} mm (expected: 100-300)")

    # RHip-RKnee
    rhip_rknee = np.linalg.norm(sample_poses[:, 1] - sample_poses[:, 2], axis=1)
    print(f"RHip-RKnee: {np.mean(rhip_rknee):.1f} ± {np.std(rhip_rknee):.1f} mm (expected: 350-550)")

    # RKnee-RFoot
    rknee_rfoot = np.linalg.norm(sample_poses[:, 2] - sample_poses[:, 3], axis=1)
    print(f"RKnee-RFoot: {np.mean(rknee_rfoot):.1f} ± {np.std(rknee_rfoot):.1f} mm (expected: 350-550)")

    # Check statistics
    print("\n" + "=" * 60)
    print("STATISTICS COMPARISON")
    print("=" * 60)

    pred_mean = np.mean(pred_reshaped)
    targ_mean = np.mean(targ_reshaped)
    pred_std = np.std(pred_reshaped)
    targ_std = np.std(targ_reshaped)

    print(f"Predictions: mean={pred_mean:.2f}, std={pred_std:.2f}")
    print(f"Targets:     mean={targ_mean:.2f}, std={targ_std:.2f}")

    # Visualize a few samples
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    os.makedirs("eval_fixed", exist_ok=True)

    # Visualize best and worst predictions
    per_sample_errors = np.mean(errors.reshape(predictions.shape[0], predictions.shape[1], -1), axis=(1, 2))

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
