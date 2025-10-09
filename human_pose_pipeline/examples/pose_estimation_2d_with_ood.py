#!/usr/bin/env python3
"""
2D Pose Estimation with OOD Detection - Evaluation Script

This script performs comprehensive evaluation of pose estimation with OOD detection:
1. Loads pre-computed OOD score functions from cache
2. Evaluates pose estimation on H36M data
3. Generates scatter plots of:
   - Pose prediction accuracy vs OOD score
   - Mean predicted uncertainty vs OOD score
4. Evaluates calibration: percentage of datapoints within 1, 2, 3, 4 sigma
   for ID vs OOD classifications

Based on pose_estimation_2D.py but extended with OOD detection capabilities.
"""

import os
import argparse
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax.numpy as jnp
from time import time

from src.datasets.h36m import Human36mDatasetSequence
from src.ood_scores.lm_lanczos import load_score_functions, _get_cache_base_key
from src.models import compute_num_params
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    process_frame_2d
)
from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def compute_mahalanobis_distance(ground_truth, estimated_pose, estimated_uncertainty, estimated_covariance):
    """
    Compute Mahalanobis distance for pose estimation evaluation.

    Args:
        ground_truth: (num_joints, 2) ground truth keypoints
        estimated_pose: (num_joints, 2) estimated keypoints
        estimated_uncertainty: (num_joints, 2) standard deviations
        estimated_covariance: (num_joints,) covariance between x and y

    Returns:
        mahalanobis: (num_joints,) Mahalanobis distances for each joint
    """
    delta = ground_truth - estimated_pose
    std_x = estimated_uncertainty[:, 0]
    std_y = estimated_uncertainty[:, 1]
    cov_xy = estimated_covariance

    # Compute determinant and inverse of covariance matrix
    det_sigma = (std_x ** 2) * (std_y ** 2) - (cov_xy ** 2)
    epsilon = 1e-6
    det_sigma += epsilon

    inv_sigma_xx = (std_y ** 2) / det_sigma
    inv_sigma_yy = (std_x ** 2) / det_sigma
    inv_sigma_xy = (-cov_xy) / det_sigma

    # Compute Mahalanobis distance
    mahalanobis = (
        inv_sigma_xx * (delta[:, 0] ** 2) +
        inv_sigma_yy * (delta[:, 1] ** 2) +
        2 * inv_sigma_xy * (delta[:, 0] * delta[:, 1])
    )

    return mahalanobis


def evaluate_within_sigma(mahalanobis_distances):
    """
    Evaluate how many joints fall within 1, 2, 3, and 4 sigma confidence intervals.

    Args:
        mahalanobis_distances: (num_samples, num_joints) Mahalanobis distances

    Returns:
        dict: Counts and percentages for each sigma level
    """
    # Chi-squared thresholds for 2 degrees of freedom
    thresholds = {
        '1sigma': chi2.ppf(0.68, df=2),
        '2sigma': chi2.ppf(0.95, df=2),
        '3sigma': chi2.ppf(0.9973, df=2),
        '4sigma': chi2.ppf(0.99994, df=2)
    }

    total_joints = mahalanobis_distances.size
    results = {}

    for sigma_name, threshold in thresholds.items():
        within = np.sum(mahalanobis_distances <= threshold)
        results[sigma_name] = {
            'count': within,
            'percentage': 100.0 * within / total_joints if total_joints > 0 else 0.0
        }

    return results


def compute_pose_accuracy(ground_truth, estimated_pose):
    """
    Compute mean per-joint position error (MPJPE).

    Args:
        ground_truth: (num_joints, 2) ground truth keypoints
        estimated_pose: (num_joints, 2) estimated keypoints

    Returns:
        float: Mean per-joint position error in pixels
    """
    return np.mean(np.linalg.norm(ground_truth - estimated_pose, axis=1))


def evaluate_h36m_with_ood(
    dataset, model, params, batch_stats,
    human_detector, device_torch, score_fn, ood_threshold,
    max_samples=None
):
    """
    Evaluate H36M dataset with pose estimation and OOD detection.

    Args:
        dataset: H36M dataset to evaluate
        model: JAX pose estimation model
        params: Model parameters
        batch_stats: Batch statistics
        human_detector: YOLO detector
        device_torch: PyTorch device
        score_fn: OOD scoring function
        ood_threshold: Threshold for OOD classification
        max_samples: Maximum number of samples to process

    Returns:
        dict: Evaluation results
    """
    results = {
        'ood_scores': [],
        'pose_accuracies': [],  # MPJPE for each sample
        'mean_uncertainties': [],  # Mean predicted uncertainty per sample
        'mahalanobis_distances': [],  # All Mahalanobis distances
        'is_ood': [],  # OOD classification
        'ground_truth': [],
        'predictions': [],
        'uncertainties': []
    }

    samples_processed = 0

    # Warm-up run
    print("Warming up the model...")
    warmup_frame = dataset[0]['frames'][0]
    _ = process_frame_2d(
        frame=warmup_frame,
        model=model,
        params=params,
        batch_stats=batch_stats,
        human_detector=human_detector,
        device_torch=device_torch,
        mirror_map=MIRROR_13_JOINT_MODEL_MAP,
        score_fn=score_fn,
        human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
        ood_threshold=ood_threshold
    )
    print("Warm-up complete!")

    print("\nEvaluating H36M dataset...")
    # H36M dataset processing
    for idx, sample in enumerate(tqdm(dataset, desc="Processing H36M")):
        t0 = time()
        if max_samples is not None and samples_processed >= max_samples:
            break

        full_sequence = np.array(sample['pose_sequence'])
        t0a = time()
        frames = sample['frames']
        t0b = time()

        # max_frames = len(frames)
        # for frame_idx in range(max_frames):

        # Test one random frame instead of all frames
        frame_idx = np.random.randint(len(frames))
        if True:
            if max_samples is not None and samples_processed >= max_samples:
                break

            frame_image_pil = frames[frame_idx]

            # Get pose estimation with OOD scoring
            t1 = time()
            print(f"Data loading time (pose_sequence): {t0a - t0:.3f} seconds")
            print(f"Data loading time (frames): {t0b - t0a:.3f} seconds")
            print(f"Frame indexing time: {t1 - t0b:.3f} seconds")
            pose_predictions = process_frame_2d(
                frame=frame_image_pil,
                model=model,
                params=params,
                batch_stats=batch_stats,
                human_detector=human_detector,
                device_torch=device_torch,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                score_fn=score_fn,
                human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
                ood_threshold=ood_threshold
            )
            t2 = time()
            print(f"Frame processing time (including OOD scoring): {t2 - t1:.3f} seconds")

            if not pose_predictions:
                continue

            pred = pose_predictions[0]
            estimated_pose = np.array(pred['keypoints'])
            uncertainties = np.array(pred['uncertainties'])
            covariance = np.array(pred['covariance'])
            ood_score = pred['ood_score']
            is_ood_sample = pred['is_ood']
            ground_truth = full_sequence[frame_idx]

            # Compute metrics
            pose_accuracy = compute_pose_accuracy(ground_truth, estimated_pose)
            mean_uncertainty = np.mean(np.linalg.norm(uncertainties, axis=1))
            mahalanobis = compute_mahalanobis_distance(ground_truth, estimated_pose, uncertainties, covariance)

            # Store results
            results['ood_scores'].append(ood_score)
            results['pose_accuracies'].append(pose_accuracy)
            results['mean_uncertainties'].append(mean_uncertainty)
            results['mahalanobis_distances'].append(mahalanobis)
            results['is_ood'].append(is_ood_sample)
            results['ground_truth'].append(ground_truth)
            results['predictions'].append(estimated_pose)
            results['uncertainties'].append(uncertainties)
            t3 = time()
            print(f"Metric computation and storage time: {t3 - t2:.3f} seconds")
            samples_processed += 1

    # Convert to numpy arrays
    for key in ['ood_scores', 'pose_accuracies', 'mean_uncertainties', 'is_ood']:
        results[key] = np.array(results[key])
    results['mahalanobis_distances'] = np.array(results['mahalanobis_distances'])

    print(f"Processed {samples_processed} samples from H36M")
    ood_scores_arr = results['ood_scores']
    is_ood_arr = results['is_ood']
    print(f"  Mean OOD score: {ood_scores_arr.mean():.4f}")
    print(f"  Classified as OOD: {is_ood_arr.sum()} / {len(is_ood_arr)}")

    return results


def create_evaluation_plots(h36m_results, ood_threshold, save_dir):
    """
    Create comprehensive evaluation plots.

    Args:
        h36m_results: Results from H36M dataset
        ood_threshold: OOD threshold used
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Scatter plot: Pose accuracy vs OOD score
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Color by ID/OOD classification
    id_mask = ~h36m_results['is_ood']
    ood_mask = h36m_results['is_ood']

    ax.scatter(h36m_results['ood_scores'][id_mask], h36m_results['pose_accuracies'][id_mask],
               alpha=0.6, c='blue', s=50, label=f'Classified as ID (n={id_mask.sum()})')
    ax.scatter(h36m_results['ood_scores'][ood_mask], h36m_results['pose_accuracies'][ood_mask],
               alpha=0.6, c='red', s=50, label=f'Classified as OOD (n={ood_mask.sum()})')
    ax.axvline(ood_threshold, color='black', linestyle='--', linewidth=2,
               label=f'OOD Threshold ({ood_threshold:.4f})')
    ax.set_xlabel('OOD Score', fontsize=14)
    ax.set_ylabel('Pose Accuracy (MPJPE in pixels)', fontsize=14)
    ax.set_title('H36M: Pose Accuracy vs OOD Score', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pose_accuracy_vs_ood_score.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'pose_accuracy_vs_ood_score.png')}")
    plt.close()

    # 2. Scatter plot: Mean uncertainty vs OOD score
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.scatter(h36m_results['ood_scores'][id_mask], h36m_results['mean_uncertainties'][id_mask],
               alpha=0.6, c='blue', s=50, label=f'Classified as ID (n={id_mask.sum()})')
    ax.scatter(h36m_results['ood_scores'][ood_mask], h36m_results['mean_uncertainties'][ood_mask],
               alpha=0.6, c='red', s=50, label=f'Classified as OOD (n={ood_mask.sum()})')
    ax.axvline(ood_threshold, color='black', linestyle='--', linewidth=2,
               label=f'OOD Threshold ({ood_threshold:.4f})')
    ax.set_xlabel('OOD Score', fontsize=14)
    ax.set_ylabel('Mean Predicted Uncertainty (pixels)', fontsize=14)
    ax.set_title('H36M: Mean Uncertainty vs OOD Score', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mean_uncertainty_vs_ood_score.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'mean_uncertainty_vs_ood_score.png')}")
    plt.close()

    # 3. Sigma evaluation: ID vs OOD classification
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Evaluate for ID-classified samples
    h36m_id_mask = ~h36m_results['is_ood']

    if h36m_id_mask.sum() > 0:
        h36m_id_mahal = h36m_results['mahalanobis_distances'][h36m_id_mask]
        h36m_id_sigma = evaluate_within_sigma(h36m_id_mahal)
    else:
        h36m_id_sigma = {f'{i}sigma': {'count': 0, 'percentage': 0.0} for i in range(1, 5)}

    # Evaluate for OOD-classified samples
    h36m_ood_mask = h36m_results['is_ood']

    if h36m_ood_mask.sum() > 0:
        h36m_ood_mahal = h36m_results['mahalanobis_distances'][h36m_ood_mask]
        h36m_ood_sigma = evaluate_within_sigma(h36m_ood_mahal)
    else:
        h36m_ood_sigma = {f'{i}sigma': {'count': 0, 'percentage': 0.0} for i in range(1, 5)}

    # Plot 1: H36M ID-classified
    sigma_labels = ['1σ', '2σ', '3σ', '4σ']
    h36m_id_percentages = [h36m_id_sigma[f'{i}sigma']['percentage'] for i in range(1, 5)]
    axes[0].bar(sigma_labels, h36m_id_percentages, color='blue', alpha=0.7)
    axes[0].set_title(f'H36M Classified as ID (n={h36m_id_mask.sum()})', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Percentage Within Confidence Interval (%)', fontsize=12)
    axes[0].set_xlabel('Confidence Interval', fontsize=12)
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(h36m_id_percentages):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

    # Plot 2: H36M OOD-classified
    h36m_ood_percentages = [h36m_ood_sigma[f'{i}sigma']['percentage'] for i in range(1, 5)]
    axes[1].bar(sigma_labels, h36m_ood_percentages, color='red', alpha=0.7)
    axes[1].set_title(f'H36M Classified as OOD (n={h36m_ood_mask.sum()})', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Percentage Within Confidence Interval (%)', fontsize=12)
    axes[1].set_xlabel('Confidence Interval', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(h36m_ood_percentages):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)

    plt.suptitle('H36M Uncertainty Calibration: Percentage of Joints Within Confidence Intervals',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sigma_evaluation_id_vs_ood.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(save_dir, 'sigma_evaluation_id_vs_ood.png')}")
    plt.close()

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SIGMA EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nH36M - Classified as ID (n={h36m_id_mask.sum()}):")
    for i in range(1, 5):
        print(f"  Within {i}σ: {h36m_id_sigma[f'{i}sigma']['percentage']:.1f}%")

    print(f"\nH36M - Classified as OOD (n={h36m_ood_mask.sum()}):")
    for i in range(1, 5):
        print(f"  Within {i}σ: {h36m_ood_sigma[f'{i}sigma']['percentage']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='2D Pose Estimation with OOD Detection')
    parser.add_argument('--cache_dir', type=str, default='cache/', help='Cache directory with score functions')
    parser.add_argument('--base_key', type=str, default='H36M_RegressFlow_n9000_f3c4d885', help='Base key for loading the score functions')
    parser.add_argument('--data_path', type=str, default='datasets/', help='Path to datasets')
    parser.add_argument('--model_save_path', type=str, default='models_tianle', help='Path to saved models')
    parser.add_argument('--run_name', type=str, default='finetuned_h36m_regressflow_pred', help='Model run name')
    parser.add_argument('--ood_threshold', type=float, default=0.3, help='OOD threshold')
    parser.add_argument('--max_samples', type=int, default=50, help='Max samples from H36M')
    parser.add_argument('--output_dir', type=str, default='results/pose_ood_evaluation', help='Output directory')

    args = parser.parse_args()

    print("=" * 80)
    print("2D POSE ESTIMATION WITH OOD DETECTION")
    print("=" * 80)

    # Load models
    print("\n1. Loading models...")
    models_dir = os.path.join(root_dir, args.model_save_path, "H36M", "RegressFlow", "seed_420")
    checkpoint_path = os.path.join(models_dir, args.run_name)
    model, params, batch_stats = initialize_jax_models(checkpoint_path)

    human_detector, device_torch = initialize_human_detector('cuda')

    # Load score functions
    print("\n2. Loading OOD score functions...")
    base_key = args.base_key

    print(f"Loading score functions with cache key: {base_key}")
    score_fn, _, _, _ = load_score_functions(args.cache_dir, base_key)
    print("Score functions loaded successfully!")
    print(f"Using OOD threshold: {args.ood_threshold:.6f}")

    # Load H36M dataset
    print("\n3. Loading H36M dataset...")
    h36m_dataset = Human36mDatasetSequence(
        base_directory=os.path.join(root_dir, args.data_path, "H36M", "extracted"),
        split='train',
        sequence_length=500
    )

    print(f"H36M dataset: {len(h36m_dataset)} sequences")

    # Evaluate H36M dataset
    print("\n4. Evaluating H36M dataset...")
    h36m_results = evaluate_h36m_with_ood(
        h36m_dataset, model, params, batch_stats,
        human_detector, device_torch, score_fn, args.ood_threshold,
        max_samples=args.max_samples
    )

    # Create evaluation plots
    print("\n5. Creating evaluation plots...")
    create_evaluation_plots(h36m_results, args.ood_threshold, args.output_dir)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
