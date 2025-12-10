#!/usr/bin/env python3
"""
Debug script to compare preprocessed motion data with ground truth H36M poses.

This script:
1. Loads preprocessed 3D poses with covariances from pre_processed_motion directory
2. Loads ground truth 3D poses from H36M extracted dataset
3. Aligns the sequences and computes MPJPE (Mean Per Joint Position Error)
4. Reports statistics on pose estimation quality
"""

import os
import argparse
import numpy as np
from spacepy import pycdf
from tqdm import tqdm
import matplotlib.pyplot as plt

from human_pose_pipeline.pose_estimation.h36m_settings import JOINT_IDX_17, JOINT_IDX_13

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Dataset splits
SPLIT = {
    'train': ['S1', 'S6', 'S7', 'S8', 'S9'],
    'validation': ['S11'],
    'test': ['S5']
}


def load_ground_truth_poses(extracted_dir, subject, action):
    """Load ground truth 3D poses from H36M extracted dataset.

    Args:
        extracted_dir: Path to H36M extracted directory
        subject: Subject ID (e.g., 'S1')
        action: Action name (e.g., 'Directions.55011271')

    Returns:
        poses_3d: Ground truth 3D poses (num_frames, 13, 3) in mm
    """
    poses_dir = os.path.join(extracted_dir, subject, 'Poses_D3_Positions')

    # Find matching CDF file
    cdf_file = None
    for filename in os.listdir(poses_dir):
        if filename.endswith('.cdf') and action in filename:
            cdf_file = os.path.join(poses_dir, filename)
            break

    if cdf_file is None:
        raise FileNotFoundError(f"No CDF file found for {subject}/{action}")

    # Load poses from CDF
    with pycdf.CDF(cdf_file) as cdf:
        poses = cdf["Pose"][:]
        poses = poses.reshape(-1, 32, 3)  # (num_frames, 32, 3)

        # Convert from 32 joints to 17 joints, then to 13 joints
        poses_17 = poses[:, JOINT_IDX_17, :]
        poses_13 = poses_17[:, JOINT_IDX_13, :]  # (num_frames, 13, 3)

    return poses_13


def compute_mpjpe(pred_poses, gt_poses):
    """Compute Mean Per Joint Position Error.

    Args:
        pred_poses: Predicted 3D poses (num_frames, 13, 3) in mm
        gt_poses: Ground truth 3D poses (num_frames, 13, 3) in mm

    Returns:
        mpjpe: Mean per joint position error in mm
        per_joint_errors: Error for each joint (13,)
        per_frame_errors: Error for each frame (num_frames,)
    """
    errors = np.linalg.norm(pred_poses - gt_poses, axis=-1)  # Shape = [B, T, J]
    mpjpe = np.mean(errors)  # Shape = [1]
    std = np.std(errors)  # Shape = [1]
    per_joint_errors = np.mean(errors, axis=0)  # Shape = [J]
    per_joint_std = np.std(errors, axis=0)  # Shape = [J]

    return mpjpe, per_joint_errors


def align_sequences(preprocessed_poses, gt_poses, valid_mask, downsample_rate=2):
    """Align preprocessed poses with ground truth poses.

    The preprocessed poses are extracted from videos which are downsampled by 2.
    We need to match them with the corresponding ground truth frames.

    Args:
        preprocessed_poses: Preprocessed 3D poses (num_frames, 13, 3)
        gt_poses: Ground truth 3D poses (num_gt_frames, 13, 3)
        valid_mask: Valid frame mask (num_frames,)
        downsample_rate: Downsampling rate for video extraction

    Returns:
        aligned_pred: Aligned predicted poses
        aligned_gt: Aligned ground truth poses
    """
    num_frames = len(preprocessed_poses)

    # The preprocessed frames correspond to downsampled GT frames
    # We need to handle potential offsets (0 or 1)
    aligned_pred = []
    aligned_gt = []

    for offset in [0, 1]:
        pred_list = []
        gt_list = []

        for i, is_valid in enumerate(valid_mask):
            if not is_valid:
                continue

            gt_idx = offset + i * downsample_rate

            if gt_idx >= len(gt_poses):
                break

            pred_list.append(preprocessed_poses[i])
            gt_list.append(gt_poses[gt_idx])

        if len(pred_list) > len(aligned_pred):
            aligned_pred = pred_list
            aligned_gt = gt_list

    return np.array(aligned_pred), np.array(aligned_gt)


def debug_sequence(preprocessed_path, extracted_dir, subject, action):
    """Debug a single sequence.

    Args:
        preprocessed_path: Path to preprocessed NPZ file
        extracted_dir: Path to H36M extracted directory
        subject: Subject ID
        action: Action name (without camera ID)

    Returns:
        dict with statistics
    """
    # Load preprocessed data
    data = np.load(preprocessed_path)
    pred_poses = data['poses_3d']  # (num_frames, 13, 3)
    covariances = data['covariances_3d']  # (num_frames, 13, 3, 3)
    valid_mask = data['valid_mask']  # (num_frames,)

    # Load ground truth
    try:
        gt_poses = load_ground_truth_poses(extracted_dir, subject, action)
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        return None

    # Align sequences
    min_length = min(len(pred_poses), len(gt_poses))
    aligned_pred = pred_poses[:min_length]
    aligned_gt = gt_poses[:min_length]
    # aligned_pred, aligned_gt = align_sequences(pred_poses, gt_poses, valid_mask)

    if len(aligned_pred) == 0:
        print(f"Warning: No valid frames to compare for {subject}/{action}")
        return None

    # Compute MPJPE
    mpjpe, per_joint_errors = compute_mpjpe(aligned_pred, aligned_gt)

    # Compute uncertainty statistics
    # Average uncertainty (standard deviation) for each joint
    valid_covariances = covariances[valid_mask][:len(aligned_pred)]
    avg_uncertainties = []
    for j in range(13):
        # Get diagonal elements (variances) for this joint
        variances = np.diagonal(valid_covariances[:, j, :, :], axis1=1, axis2=2)  # (num_frames, 3)
        std_devs = np.sqrt(variances)  # (num_frames, 3)
        avg_uncertainties.append(np.mean(std_devs))
    avg_uncertainties = np.array(avg_uncertainties)

    return {
        'subject': subject,
        'action': action,
        'num_frames': len(aligned_pred),
        'num_valid': valid_mask.sum(),
        'mpjpe': mpjpe,
        'per_joint_errors': per_joint_errors,
        'avg_uncertainties': avg_uncertainties,
    }


def debug_subject(preprocessed_dir, extracted_dir, subject):
    """Debug all sequences for a subject.

    Args:
        preprocessed_dir: Path to preprocessed motion data
        extracted_dir: Path to H36M extracted directory
        subject: Subject ID

    Returns:
        list of statistics dicts
    """
    subject_dir = os.path.join(preprocessed_dir, subject)

    if not os.path.exists(subject_dir):
        print(f"Warning: Preprocessed directory not found for {subject}")
        return []

    results = []
    npz_files = sorted([f for f in os.listdir(subject_dir) if f.endswith('.npz')])

    print(f"\nProcessing {subject}: {len(npz_files)} sequences")

    for npz_file in tqdm(npz_files, desc=f"Debugging {subject}"):
        action = npz_file.replace('.npz', '')
        npz_path = os.path.join(subject_dir, npz_file)

        result = debug_sequence(npz_path, extracted_dir, subject, action)
        if result is not None:
            results.append(result)

    return results


def print_statistics(results):
    """Print statistics from all sequences.

    Args:
        results: List of statistics dicts
    """
    if not results:
        print("No results to display")
        return

    # Aggregate statistics
    all_mpjpe = [r['mpjpe'] for r in results]
    all_per_joint = np.array([r['per_joint_errors'] for r in results])
    all_uncertainties = np.array([r['avg_uncertainties'] for r in results])

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Number of sequences: {len(results)}")
    print(f"Total valid frames: {sum(r['num_frames'] for r in results)}")
    print(f"\nMPJPE (Mean Per Joint Position Error):")
    print(f"  Mean:   {np.mean(all_mpjpe):.2f} mm")
    print(f"  Median: {np.median(all_mpjpe):.2f} mm")
    print(f"  Std:    {np.std(all_mpjpe):.2f} mm")
    print(f"  Min:    {np.min(all_mpjpe):.2f} mm")
    print(f"  Max:    {np.max(all_mpjpe):.2f} mm")

    print(f"\nPer-Joint MPJPE (averaged across all sequences):")
    joint_names = [
        'Head', 'Neck', 'Right Shoulder', 'Right Elbow', 'Right Wrist',
        'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Right Hip',
        'Right Knee', 'Right Ankle', 'Left Hip', 'Left Knee'
    ]
    avg_per_joint = np.mean(all_per_joint, axis=0)
    for i, (name, error) in enumerate(zip(joint_names, avg_per_joint)):
        print(f"  Joint {i:2d} ({name:20s}): {error:6.2f} mm")

    print(f"\nAverage Uncertainty (standard deviation):")
    avg_uncertainty_per_joint = np.mean(all_uncertainties, axis=0)
    for i, (name, unc) in enumerate(zip(joint_names, avg_uncertainty_per_joint)):
        print(f"  Joint {i:2d} ({name:20s}): {unc:6.2f} mm")

    print("\n" + "=" * 80)
    print("PER-SEQUENCE STATISTICS")
    print("=" * 80)
    print(f"{'Subject':<10} {'Action':<30} {'Frames':<10} {'MPJPE (mm)':<15}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x['mpjpe']):
        print(f"{r['subject']:<10} {r['action']:<30} {r['num_frames']:<10} {r['mpjpe']:<15.2f}")

    return all_mpjpe, avg_per_joint, avg_uncertainty_per_joint


def plot_statistics(results, output_dir):
    """Create visualization plots.

    Args:
        results: List of statistics dicts
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: MPJPE distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # MPJPE histogram
    all_mpjpe = [r['mpjpe'] for r in results]
    axes[0, 0].hist(all_mpjpe, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('MPJPE (mm)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of MPJPE across sequences')
    axes[0, 0].axvline(np.mean(all_mpjpe), color='r', linestyle='--',
                       label=f'Mean: {np.mean(all_mpjpe):.2f} mm')
    axes[0, 0].legend()

    # Per-joint errors
    all_per_joint = np.array([r['per_joint_errors'] for r in results])
    avg_per_joint = np.mean(all_per_joint, axis=0)
    std_per_joint = np.std(all_per_joint, axis=0)

    joint_indices = np.arange(13)
    axes[0, 1].bar(joint_indices, avg_per_joint, yerr=std_per_joint,
                   capsize=5, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Joint Index')
    axes[0, 1].set_ylabel('MPJPE (mm)')
    axes[0, 1].set_title('Average Error per Joint')
    axes[0, 1].set_xticks(joint_indices)

    # Per-joint uncertainty
    all_uncertainties = np.array([r['avg_uncertainties'] for r in results])
    avg_uncertainty = np.mean(all_uncertainties, axis=0)
    std_uncertainty = np.std(all_uncertainties, axis=0)

    axes[1, 0].bar(joint_indices, avg_uncertainty, yerr=std_uncertainty,
                   capsize=5, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel('Joint Index')
    axes[1, 0].set_ylabel('Avg Uncertainty (mm)')
    axes[1, 0].set_title('Average Uncertainty per Joint')
    axes[1, 0].set_xticks(joint_indices)

    # Error vs Uncertainty correlation
    axes[1, 1].scatter(avg_uncertainty, avg_per_joint, alpha=0.7, s=100)
    axes[1, 1].set_xlabel('Average Uncertainty (mm)')
    axes[1, 1].set_ylabel('Average Error (mm)')
    axes[1, 1].set_title('Error vs Uncertainty Correlation (per joint)')

    # Add correlation coefficient
    corr = np.corrcoef(avg_uncertainty, avg_per_joint)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                    transform=axes[1, 1].transAxes, verticalalignment='top')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'debug_statistics.png'), dpi=150)
    print(f"\nPlot saved to: {os.path.join(output_dir, 'debug_statistics.png')}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Debug preprocessed motion data by comparing with ground truth'
    )
    parser.add_argument(
        '--preprocessed_dir',
        type=str,
        default='datasets/H36M/pre_processed_motion',
        help='Path to preprocessed motion data'
    )
    parser.add_argument(
        '--extracted_dir',
        type=str,
        default='datasets/H36M/extracted',
        help='Path to H36M extracted directory with ground truth'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'validation', 'test', 'all'],
        help='Which split to debug'
    )
    parser.add_argument(
        '--subject',
        type=str,
        default=None,
        help='Debug only specific subject (e.g., S1)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/debug_motion_preprocessing',
        help='Directory to save debug plots and statistics'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Debug Preprocessed Motion Data")
    print("=" * 80)
    print(f"Preprocessed data: {args.preprocessed_dir}")
    print(f"Ground truth data: {args.extracted_dir}")
    print("=" * 80)

    # Determine which subjects to debug
    if args.subject:
        subjects_to_debug = [args.subject]
    elif args.split == 'all':
        subjects_to_debug = []
        for split_subjects in SPLIT.values():
            subjects_to_debug.extend(split_subjects)
    else:
        subjects_to_debug = SPLIT[args.split]

    print(f"\nSubjects to debug: {subjects_to_debug}")

    # Debug each subject
    all_results = []
    for subject in subjects_to_debug:
        results = debug_subject(args.preprocessed_dir, args.extracted_dir, subject)
        all_results.extend(results)

    # Print and save statistics
    if all_results:
        print_statistics(all_results)
        plot_statistics(all_results, args.output_dir)

        # Save detailed results to CSV
        import csv
        csv_path = os.path.join(args.output_dir, 'debug_results.csv')
        os.makedirs(args.output_dir, exist_ok=True)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['subject', 'action', 'num_frames', 'mpjpe'])
            writer.writeheader()
            for r in all_results:
                writer.writerow({
                    'subject': r['subject'],
                    'action': r['action'],
                    'num_frames': r['num_frames'],
                    'mpjpe': r['mpjpe']
                })

        print(f"\nDetailed results saved to: {csv_path}")
    else:
        print("\nNo valid results to report")

    print("\n" + "=" * 80)
    print("Debug complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
