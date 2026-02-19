#!/usr/bin/env python3
"""
2D Pose Estimation - YOLOv26 Implementation

YOLOv26 version of pose_estimation_2D.py:
- Load H36M dataset with pose sequences and video frames
- Perform 2D pose estimation using YOLO v26 pose model
- The Pose26 head outputs per-keypoint sigma_x / sigma_y uncertainties
- Evaluate pose estimation accuracy using MPJPE and Mahalanobis distance
- Visualize results with ground truth, estimated poses, and uncertainty ellipses
"""

import os
import sys
from time import time
import numpy as np
from scipy.stats import chi2
import torch

from ultralytics import YOLO
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_2d_yolo,
    reset_yolo_tracking,
)
from human_pose_pipeline.utils.eval_utils import evaluate_pose_prediction_scores_np
from human_pose_pipeline.utils.visualization import visualize_pose_sequence
from src.datasets.h36m import Human36mDatasetSequence
from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
)

# Coverage levels for both evaluation methods
COVERAGES = [0.6800, 0.9500, 0.9973, 0.9999]
N_RLE_SAMPLES = 100000

# Add parent directory to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, root_dir)
# The ultralytics/ subfolder is a namespace package (no __init__.py at repo root level),
# which shadows the editable install when running from this directory.
# Insert the fork's repo root explicitly so the real package is found first.
# sys.path.insert(0, os.path.join(root_dir, 'ultralytics'))


def evaluate_pose_estimation_full(ground_truth, estimated_pose, estimated_uncertainty, estimated_covariance):
    """
    Evaluate pose estimation accuracy using Mahalanobis distance and confidence intervals.

    Args:
        ground_truth: (num_joints, 2)
        estimated_pose: (num_joints, 2)
        estimated_uncertainty: (num_joints, 2) — sigma_x, sigma_y per joint
        estimated_covariance: (num_joints,) — x-y covariance per joint (zeros for YOLO)

    Returns:
        dict with mpjpe, counts within N std, and num_joints
    """
    delta = ground_truth - estimated_pose  # (num_joints, 2)
    mpjpe = np.mean(np.linalg.norm(delta, axis=-1))

    std_x = estimated_uncertainty[:, 0]
    std_y = estimated_uncertainty[:, 1]
    cov_xy = estimated_covariance

    det_sigma = (std_x ** 2) * (std_y ** 2) - (cov_xy ** 2)
    det_sigma += 1e-6  # numerical stability

    inv_sigma_xx = (std_y ** 2) / det_sigma
    inv_sigma_yy = (std_x ** 2) / det_sigma
    inv_sigma_xy = (-cov_xy) / det_sigma

    mahalanobis = (
        inv_sigma_xx * (delta[:, 0] ** 2)
        + inv_sigma_yy * (delta[:, 1] ** 2)
        + 2 * inv_sigma_xy * (delta[:, 0] * delta[:, 1])
    )

    thresholds = [
        chi2.ppf(0.68, df=2),     # 1 std
        chi2.ppf(0.95, df=2),     # 2 std
        chi2.ppf(0.9973, df=2),   # 3 std
        chi2.ppf(0.99994, df=2),  # 4 std
    ]
    within_std = [mahalanobis <= t for t in thresholds]
    counts = {f'within_{i+1}std': int(np.sum(w)) for i, w in enumerate(within_std)}

    return {
        'mpjpe': mpjpe,
        'counts': counts,
        'num_joints': len(ground_truth),
    }


# ---------------------------------------------------------------------------
# RLE sampling-based coverage evaluation
# ---------------------------------------------------------------------------

def _get_flow_model_from_yolo(yolo_model):
    """Extract the RealNVP flow model from a YOLO Pose26 head, or None if unavailable."""
    try:
        head = yolo_model.model.model[-1]
        if hasattr(head, 'flow_model') and head.flow_model is not None:
            return head.flow_model
    except Exception:
        pass
    return None


def _flow_forward_p(flow_model, z):
    """Map N(0,I) latent samples to the normalised-error data space via the RealNVP inverse.

    Inverts flow_model.backward_p(x):
      backward:  z = (1-mask)*(x-t)*exp(-s) + mask*x    (iterating layers reversed)
      forward:   x = (1-mask)*z*exp(s) + t + mask*z      (iterating layers forward)

    Args:
        flow_model: RealNVP instance.
        z: (N, 2) tensor of standard-normal samples.

    Returns:
        (N, 2) tensor of samples in the normalised error space.
    """
    x = z.clone()
    for i in range(len(flow_model.t)):
        x_ = flow_model.mask[i] * x
        s = flow_model.s[i](x_) * (1 - flow_model.mask[i])
        t = flow_model.t[i](x_) * (1 - flow_model.mask[i])
        x = (1 - flow_model.mask[i]) * x * torch.exp(s) + t + x_
    return x


def precompute_rle_coverage_thresholds(flow_model, n_samples=N_RLE_SAMPLES, coverages=COVERAGES):
    """Precompute log-prob thresholds for RLE sampling-based coverage.

    Draws n_samples points from the RealNVP distribution and finds the
    log-prob values that separate each coverage level (HPD regions).

    Args:
        flow_model: trained RealNVP normalising flow from the Pose26 head.
        n_samples: number of Monte Carlo samples (default 10 000).
        coverages: sequence of coverage fractions, e.g. [0.68, 0.95, ...].

    Returns:
        List of log-prob thresholds — one per entry in coverages.
    """
    flow_device = flow_model.loc.device
    epsilon = torch.randn(n_samples, 2, device=flow_device)
    with torch.no_grad():
        error_samples = _flow_forward_p(flow_model, epsilon)       # (N, 2) data space
        log_p_samples = flow_model.log_prob(error_samples)         # (N,)
    log_p_np = log_p_samples.cpu().numpy()
    return [float(np.percentile(log_p_np, (1.0 - c) * 100.0)) for c in coverages]


def evaluate_pose_estimation_rle_sampling(
    ground_truth,
    estimated_pose,
    sigma,
    flow_model,
    rle_thresholds,
):
    """Evaluate pose coverage using RLE sampling-based confidence regions.

    For each joint the normalised GT error is evaluated under the learned
    RealNVP distribution.  The GT is considered 'covered' at level X% when
    its log-probability is at least as high as the precomputed threshold that
    corresponds to that level (i.e. it lies in the X% HPD region).

    Args:
        ground_truth:   (num_joints, 2) pixel coordinates.
        estimated_pose: (num_joints, 2) predicted pixel coordinates.
        sigma:          (num_joints, 2) per-joint (sigma_x, sigma_y) from RLE.
        flow_model:     trained RealNVP normalising flow.
        rle_thresholds: log-prob thresholds for [68%, 95%, 99.73%, 99.99%]
                        as returned by precompute_rle_coverage_thresholds().

    Returns:
        dict with 'counts' (joints within each level) and 'num_joints'.
    """
    delta = ground_truth - estimated_pose          # (num_joints, 2)

    # Normalise error by sigma — same normalisation used during RLE training
    error_gt = delta / (sigma + 1e-9)             # (num_joints, 2)

    flow_device = flow_model.loc.device
    error_gt_t = torch.tensor(error_gt, dtype=torch.float32, device=flow_device)
    with torch.no_grad():
        log_p_gt = flow_model.log_prob(error_gt_t).cpu().numpy()   # (num_joints,)

    label_map = ['within_1std', 'within_2std', 'within_3std', 'within_4std']
    counts = {
        label_map[i]: int(np.sum(log_p_gt >= thresh))
        for i, thresh in enumerate(rle_thresholds)
    }
    return {'counts': counts, 'num_joints': len(ground_truth)}


def main():
    base_directory = os.path.join(root_dir, "datasets", "H36M", "extracted")

    # ============ CONFIGURATION ============
    QUICK_TEST = True          # Set to False to run full evaluation
    splits = ['validation']
    yolo_model_name = "yolo26n-pose.pt"  # Options: yolo26n/s/m/l/x-pose.pt (auto-downloaded)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    visualize = False          # Set to True to save per-sequence GIF visualizations
    use_flow_model = True
    # =======================================

    if QUICK_TEST:
        max_files = 3
        sequence_length = 500
        print("\n" + "=" * 50)
        print("QUICK TEST MODE ENABLED")
        print(f"  Files:    {max_files}")
        print(f"  Frames:   {sequence_length} per sequence")
        print("=" * 50 + "\n")
    else:
        max_files = None
        sequence_length = 500

    # Initialize YOLO pose model (auto-downloads on first run)
    print(f"Initializing YOLO pose model: {yolo_model_name} on {device.upper()}")
    yolo_model = YOLO(yolo_model_name)
    yolo_model.to(device)
    print("Model ready.\n")

    # ---- RLE sampling setup ----
    if use_flow_model:
        flow_model = _get_flow_model_from_yolo(yolo_model)
    else:
        flow_model = None
    rle_thresholds = None
    if flow_model is not None:
        print(f"RLE flow model found. Precomputing coverage thresholds (N={N_RLE_SAMPLES})...")
        rle_thresholds = precompute_rle_coverage_thresholds(flow_model)
        print(f"  Log-prob thresholds: " +
              ", ".join(f"{c*100:.2f}%→{t:.3f}" for c, t in zip(COVERAGES, rle_thresholds)))
        print()
    else:
        print("RLE flow model not accessible (model may be fused). "
              "RLE sampling evaluation will be skipped.\n")

    # Build dataset splits
    datasets = {}
    for split in splits:
        datasets[split] = Human36mDatasetSequence(
            base_directory,
            split=split,
            sequence_length=sequence_length,
            max_files=max_files,
        )
        print(f"{split.capitalize()} dataset: {len(datasets[split])} sequences")

    # Process each split
    for split in splits:
        dataset = datasets[split]
        print(f"\n{'='*60}")
        print(f"Processing split: {split}")
        print('='*60)

        total_mpjpe = 0.0
        total_frames = 0
        total_joints = 0
        total_within_1std = 0
        total_within_2std = 0
        total_within_3std = 0
        total_within_4std = 0

        # RLE sampling counters
        total_within_1std_rle = 0
        total_within_2std_rle = 0
        total_within_3std_rle = 0
        total_within_4std_rle = 0

        for idx, sample in enumerate(dataset):
            full_sequence = np.array(sample['pose_sequence'])  # (T, 13, 2)
            frames = sample['frames']                          # list of PIL images

            print(f"\nSequence {idx}: {full_sequence.shape[0]} GT frames, {len(frames)} video frames")

            # Reset tracker between sequences so IDs don't carry over
            reset_yolo_tracking(yolo_model)

            gt_poses = []
            estimated_poses = []
            estimated_uncertainties = []
            estimated_covariances = []

            for frame_idx in range(len(frames)):
                frame_image = frames[frame_idx]

                pose_predictions = process_frame_2d_yolo(
                    frames=frame_image,
                    yolo_pose_model=yolo_model,
                    mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                    enable_tracking=False,
                    confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
                    verbose=False,
                    device=device,
                )

                if not pose_predictions['mask'][0].item():
                    print(f"  Frame {frame_idx}: no human detected, skipping.")
                    continue

                mapped_pose = pose_predictions['keypoints'][0].cpu().numpy()       # (13, 2)
                mapped_uncertainty = pose_predictions['uncertainties'][0].cpu().numpy()  # (13, 2)
                mapped_covariance = pose_predictions['covariance'][0].cpu().numpy()     # (13,)
                ground_truth = full_sequence[frame_idx]                                 # (13, 2)

                gt_poses.append(ground_truth)
                estimated_poses.append(mapped_pose)
                estimated_uncertainties.append(mapped_uncertainty)
                estimated_covariances.append(mapped_covariance)

                evaluation = evaluate_pose_estimation_full(
                    ground_truth=ground_truth,
                    estimated_pose=mapped_pose,
                    estimated_uncertainty=mapped_uncertainty,
                    estimated_covariance=mapped_covariance,
                )

                total_frames += 1
                total_joints += evaluation['num_joints']
                total_mpjpe += evaluation['mpjpe']
                total_within_1std += evaluation['counts']['within_1std']
                total_within_2std += evaluation['counts']['within_2std']
                total_within_3std += evaluation['counts']['within_3std']
                total_within_4std += evaluation['counts']['within_4std']

                # RLE sampling evaluation
                if flow_model is not None and rle_thresholds is not None:
                    eval_rle = evaluate_pose_estimation_rle_sampling(
                        ground_truth=ground_truth,
                        estimated_pose=mapped_pose,
                        sigma=mapped_uncertainty,
                        flow_model=flow_model,
                        rle_thresholds=rle_thresholds,
                    )
                    total_within_1std_rle += eval_rle['counts']['within_1std']
                    total_within_2std_rle += eval_rle['counts']['within_2std']
                    total_within_3std_rle += eval_rle['counts']['within_3std']
                    total_within_4std_rle += eval_rle['counts']['within_4std']

            if len(estimated_poses) == 0:
                print("  No valid frames in this sequence.")
                continue

            mpjpe, _, _, _, per_joint_errors, _ = evaluate_pose_prediction_scores_np(
                predictions=np.array(estimated_poses)[np.newaxis, :],
                targets=np.array(gt_poses)[np.newaxis, :],
            )
            print(f"  Sequence MPJPE:      {mpjpe:.2f} px")
            print(f"  Per-joint errors:    {np.round(per_joint_errors, 1)}")
            mean_sigma = np.mean([u for u in estimated_uncertainties], axis=(0, 1))
            print(f"  Mean sigma_x/y:      {mean_sigma[0]:.2f} / {mean_sigma[1]:.2f} px")

            if visualize:
                os.makedirs("visualizations", exist_ok=True)
                output_file = f"visualizations/pose_sequence_yolo_{split}_{idx}.gif"
                visualize_pose_sequence(
                    pose_sequence=np.array(gt_poses),
                    images=frames[:len(gt_poses)],
                    output_file=output_file,
                    estimated_poses=estimated_poses,
                    estimated_uncertainties=estimated_uncertainties,
                    estimated_covariances=estimated_covariances,
                    show_uncertainty=True,
                )
                print(f"  Visualization saved: {output_file}")

        # Overall summary for this split
        if total_frames > 0:
            avg_within_1std = (total_within_1std / total_joints) * 100
            avg_within_2std = (total_within_2std / total_joints) * 100
            avg_within_3std = (total_within_3std / total_joints) * 100
            avg_within_4std = (total_within_4std / total_joints) * 100

            print(f"\n{'='*60}")
            print(f"Overall Results — {split}")
            print('='*60)
            print(f"  Frames processed:        {total_frames}")
            print(f"  Joints evaluated:        {total_joints}")
            print(f"  Average MPJPE:           {total_mpjpe / total_frames:.2f} px")
            print()
            print(f"  --- Gaussian (sigma_x/sigma_y, Mahalanobis) ---")
            print(f"  Within 68.00% (1-std):   {avg_within_1std:.2f}%")
            print(f"  Within 95.00% (2-std):   {avg_within_2std:.2f}%")
            print(f"  Within 99.73% (3-std):   {avg_within_3std:.2f}%")
            print(f"  Within 99.99% (4-std):   {avg_within_4std:.2f}%")

            if flow_model is not None and rle_thresholds is not None:
                avg_within_1std_rle = (total_within_1std_rle / total_joints) * 100
                avg_within_2std_rle = (total_within_2std_rle / total_joints) * 100
                avg_within_3std_rle = (total_within_3std_rle / total_joints) * 100
                avg_within_4std_rle = (total_within_4std_rle / total_joints) * 100
                print()
                print(f"  --- RLE sampling (N={N_RLE_SAMPLES}, HPD regions via RealNVP flow) ---")
                print(f"  Within 68.00% HPD:       {avg_within_1std_rle:.2f}%")
                print(f"  Within 95.00% HPD:       {avg_within_2std_rle:.2f}%")
                print(f"  Within 99.73% HPD:       {avg_within_3std_rle:.2f}%")
                print(f"  Within 99.99% HPD:       {avg_within_4std_rle:.2f}%")


if __name__ == "__main__":
    main()
