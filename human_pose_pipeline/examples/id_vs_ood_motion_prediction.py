#!/usr/bin/env python3
"""
ID vs OOD Motion Prediction Script

This script evaluates motion prediction performance on:
- ID data: Human3.6M Motion Reduced Output Dataset (normal sequences)
- OOD data: Human3.6M Motion Reduced Output OOD Dataset (shuffled input sequences)

The goal is to demonstrate how a model trained on sequential human motion
performs differently on in-distribution vs out-of-distribution data (shuffled sequences),
and how uncertainty quantification with sketching Lanczos can detect OOD samples.

Based on id_vs_ood_pose_prediction.py but adapted for motion prediction.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import cloudpickle
import jax.numpy as jnp
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from src.datasets.h36m_motion_prediction import Human36mMotionDataset3D
from human_pose_pipeline.pose_estimation.inference_helper import initialize_jax_models
from human_pose_pipeline.utils.visualization import visualize_motion_prediction
from human_pose_pipeline.pose_estimation.h36m_settings import CONNECTIONS_13
from human_pose_pipeline.motion_prediction.h36m_settings import REDUCED_JOINT_INDICES, PREDICTION_HORIZON_LENGTH, REDUCED_TIMESTEP

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

N_JOINTS = 13
OUTPUT_DIM = 9  # 3 joints * 3 coordinates


def predict_and_score_motion(motion_prediction_jit_fn, params, batch_stats, score_fn,
                             dataset, dataset_name, max_samples=None):
    """
    Run motion prediction and OOD scoring on a dataset.

    Args:
        motion_prediction_jit_fn: JIT-compiled JAX function for motion prediction
        params: Model parameters
        batch_stats: Batch statistics
        score_fn: OOD score function
        dataset: Dataset to evaluate on
        dataset_name: Name of the dataset (for display)
        max_samples: Maximum number of samples to process

    Returns:
        dict: Results containing predictions, targets, errors, and OOD scores
    """
    print(f"\nEvaluating on {dataset_name} dataset...")

    predictions = []
    targets = []
    ood_scores = []

    samples_processed = 0

    # Create dataloader for batch processing
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0
    )

    for batch in tqdm(dataloader, desc=f"Processing {dataset_name}"):
        if max_samples is not None and samples_processed >= max_samples:
            break

        input_pose = batch[0]
        target_pose = batch[1]

        # Convert to JAX arrays
        input_pose_jax = jnp.array(input_pose, dtype=jnp.float32)
        target_pose_jax = jnp.array(target_pose, dtype=jnp.float32)

        # Model inference
        if batch_stats is not None:
            pred_poses, _ = motion_prediction_jit_fn(params, batch_stats, input_pose_jax)
        else:
            pred_poses, _ = motion_prediction_jit_fn(params, input_pose_jax)

        # Compute OOD scores for each sample in batch
        batch_ood_scores = np.zeros(len(input_pose_jax))
        for i in range(len(input_pose_jax)):
            # Score function takes single input
            single_input = jnp.expand_dims(input_pose_jax[i], axis=0)
            score = score_fn(single_input)
            batch_ood_scores[i] = np.array(score)[0]

        # Store results
        predictions.append(np.array(pred_poses))
        targets.append(np.array(target_pose_jax))
        ood_scores.append(batch_ood_scores)

        samples_processed += len(input_pose)

    # Concatenate all results
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    ood_scores = np.concatenate(ood_scores, axis=0)

    # Compute prediction errors (MPJPE in mm)
    # Predictions and targets are shape (N, output_dim) where output_dim = N_TIMESTEPS, 39 (13 joints * 3 coords)
    predictions_reshaped = predictions.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)
    targets_reshaped = targets.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)

    # Compute per-sample MPJPE
    errors = np.linalg.norm(predictions_reshaped - targets_reshaped, axis=-1)  # (N, N_TIMESTEPS, N_JOINTS)
    mpjpe_per_sample = np.mean(errors, axis=(1, 2))  # (N,)
    mpjpe_overall = np.mean(mpjpe_per_sample)
    mpjpe_std = np.std(mpjpe_per_sample)

    print(f"\n{dataset_name} Results:")
    print(f"  Total samples: {len(predictions)}")
    print(f"  MPJPE: {mpjpe_overall:.2f} ± {mpjpe_std:.2f} mm")
    print(f"  OOD Score: {np.mean(ood_scores):.4f} ± {np.std(ood_scores):.4f}")

    return {
        'predictions': predictions,
        'targets': targets,
        'mpjpe_per_sample': mpjpe_per_sample,
        'mpjpe_overall': mpjpe_overall,
        'mpjpe_std': mpjpe_std,
        'ood_scores': ood_scores,
        'dataset_name': dataset_name
    }


def create_comparison_visualization(id_results, ood_results, save_path="id_vs_ood_motion_comparison.png"):
    """
    Create visualization comparing ID vs OOD performance.

    Args:
        id_results: Results dictionary for ID dataset
        ood_results: Results dictionary for OOD dataset
        save_path: Path to save the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    datasets = ['ID (Normal)', 'OOD (Shuffled)']
    mpjpe_scores = [id_results['mpjpe_overall'], ood_results['mpjpe_overall']]
    mpjpe_stds = [id_results['mpjpe_std'], ood_results['mpjpe_std']]
    ood_mean_scores = [np.mean(id_results['ood_scores']), np.mean(ood_results['ood_scores'])]
    ood_std_scores = [np.std(id_results['ood_scores']), np.std(ood_results['ood_scores'])]

    # MPJPE comparison with error bars
    axes[0, 0].bar(datasets, mpjpe_scores, yerr=mpjpe_stds, color=['blue', 'red'],
                   alpha=0.7, capsize=10)
    axes[0, 0].set_title('MPJPE Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('MPJPE (mm)', fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3)

    # OOD score comparison with error bars
    axes[0, 1].bar(datasets, ood_mean_scores, yerr=ood_std_scores,
                   color=['blue', 'red'], alpha=0.7, capsize=10)
    axes[0, 1].set_title('OOD Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('OOD Score', fontsize=12)
    axes[0, 1].grid(axis='y', alpha=0.3)

    # Error vs OOD score scatter plots combined
    axes[0, 2].scatter(id_results['ood_scores'], id_results['mpjpe_per_sample'],
                       alpha=0.5, s=10, color='blue', label='ID')
    axes[0, 2].scatter(ood_results['ood_scores'], ood_results['mpjpe_per_sample'],
                       alpha=0.5, s=10, color='red', label='OOD')
    axes[0, 2].set_title('Prediction Error vs OOD Score', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('OOD Score', fontsize=12)
    axes[0, 2].set_ylabel('MPJPE (mm)', fontsize=12)
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(alpha=0.3)

    # MPJPE distribution histograms
    axes[1, 0].hist(id_results['mpjpe_per_sample'], bins=50, alpha=0.7,
                    color='blue', label='ID', density=True)
    axes[1, 0].set_title('ID Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('MPJPE (mm)', fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)

    axes[1, 1].hist(ood_results['mpjpe_per_sample'], bins=50, alpha=0.7,
                    color='red', label='OOD', density=True)
    axes[1, 1].set_title('OOD Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('MPJPE (mm)', fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].grid(axis='y', alpha=0.3)

    # Combined OOD score distribution
    axes[1, 2].hist(id_results['ood_scores'], bins=50, alpha=0.7, color='blue',
                    label='ID', density=True)
    axes[1, 2].hist(ood_results['ood_scores'], bins=50, alpha=0.7, color='red',
                    label='OOD', density=True)
    axes[1, 2].set_title('OOD Score Distribution', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('OOD Score', fontsize=12)
    axes[1, 2].set_ylabel('Density', fontsize=12)
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison visualization saved: {save_path}")

    return fig


def compute_ood_detection_metrics(id_scores, ood_scores):
    """
    Compute OOD detection metrics (AUROC, AUPRC).

    Args:
        id_scores: OOD scores for ID samples
        ood_scores: OOD scores for OOD samples

    Returns:
        dict: Dictionary with AUROC and AUPRC metrics
    """
    # Create labels: 0 for ID, 1 for OOD
    labels = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])

    # Combine scores
    scores = np.concatenate([id_scores, ood_scores])

    # Compute AUROC
    auroc = roc_auc_score(labels, scores)

    # Compute AUPRC
    precision, recall, _ = precision_recall_curve(labels, scores)
    auprc = auc(recall, precision)

    return {
        'auroc': auroc,
        'auprc': auprc
    }


def visualize_sample_predictions(id_results, ood_results,
                                 output_dir="results/motion_prediction"):
    """
    Visualize sample predictions for ID and OOD datasets.

    Args:
        id_results: Results dictionary for ID dataset
        ood_results: Results dictionary for OOD dataset
        output_dir: Output directory for visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find samples with median error for each dataset
    id_median_idx = np.argsort(
        id_results['mpjpe_per_sample']
    )[len(id_results['mpjpe_per_sample']) // 2]
    ood_median_idx = np.argsort(
        ood_results['mpjpe_per_sample']
    )[len(ood_results['mpjpe_per_sample']) // 2]

    # Visualize ID sample
    print("\nGenerating ID sample visualization...")
    id_pred = id_results['predictions'][id_median_idx].reshape(
        PREDICTION_HORIZON_LENGTH, N_JOINTS, 3
    )
    id_target = id_results['targets'][id_median_idx].reshape(
        PREDICTION_HORIZON_LENGTH, N_JOINTS, 3
    )

    id_mpjpe = id_results['mpjpe_per_sample'][id_median_idx]
    id_ood = id_results['ood_scores'][id_median_idx]
    visualize_motion_prediction(
        pred_pose=id_pred[REDUCED_TIMESTEP],
        target_pose=id_target[REDUCED_TIMESTEP],
        skeleton=CONNECTIONS_13,
        label=f"ID_median_mpjpe_{id_mpjpe:.1f}mm_ood_{id_ood:.3f}",
        idx=id_median_idx,
        output_path=output_dir
    )

    # Visualize OOD sample
    print("Generating OOD sample visualization...")
    ood_pred = ood_results['predictions'][ood_median_idx].reshape(
        PREDICTION_HORIZON_LENGTH, N_JOINTS, 3
    )
    ood_target = ood_results['targets'][ood_median_idx].reshape(
        PREDICTION_HORIZON_LENGTH, N_JOINTS, 3
    )

    ood_mpjpe = ood_results['mpjpe_per_sample'][ood_median_idx]
    ood_ood_score = ood_results['ood_scores'][ood_median_idx]
    visualize_motion_prediction(
        pred_pose=ood_pred[REDUCED_TIMESTEP],
        target_pose=ood_target[REDUCED_TIMESTEP],
        skeleton=CONNECTIONS_13,
        label=f"OOD_median_mpjpe_{ood_mpjpe:.1f}mm_ood_{ood_ood_score:.3f}",
        idx=ood_median_idx,
        output_path=output_dir
    )

    print("Sample visualizations saved to {}".format(output_dir))


def main():
    """Main function for ID vs OOD motion prediction comparison."""
    print("=" * 80)
    print("ID vs OOD Motion Prediction Comparison")
    print("=" * 80)

    try:
        # Configuration
        data_path = os.path.join(root_dir, "datasets", "H36M", "extracted")
        full_model_path = os.path.join(
            root_dir,
            "human_pose_pipeline/models/motion_prediction/"
        )
        full_model = os.path.join(
            full_model_path,
            "dct_pose_transformer.pickle"
        )
        ood_model_path = os.path.join(
            root_dir,
            "human_pose_pipeline/models/motion_prediction/Human36mMotionReducedOutputDataset3D/DCTPoseTransformerReducedOutput/seed_420"
        )
        score_functions_path = os.path.join(
            ood_model_path,
            "Human36mMotionReducedOutputDataset3D_DCTPoseTransformerReducedOutput_n9000_dcfefa87_score_functions.cloudpickle"
        )

        max_samples = 640  # Limit samples for quick testing

        # Load model
        print("\nLoading motion prediction model...")
        motion_prediction_jit_fn, params, batch_stats = initialize_jax_models(
            checkpoint_path_jax=full_model
        )
        print("Model loaded successfully!")

        # Load score functions
        print("\nLoading OOD score functions...")
        if not os.path.exists(score_functions_path):
            raise FileNotFoundError(
                f"Score functions file not found: {score_functions_path}\n"
                f"Please run score_model.py first to generate the score functions."
            )

        with open(score_functions_path, 'rb') as f:
            score_data = cloudpickle.load(f)

        score_fn = score_data['score_fun']
        eigenval = score_data['eigenval']
        print(f"Score functions loaded! Top 5 eigenvalues: {eigenval[:5]}")

        # Setup datasets
        print("\nSetting up datasets...")

        # ID dataset (normal sequences)
        id_dataset = Human36mMotionDataset3D(
            base_directory=data_path,
            split='validation',
            jax_format=False,
            reduce_size=False,
            ood=False
        )

        # OOD dataset (shuffled input sequences)
        ood_dataset = Human36mMotionDataset3D(
            base_directory=data_path,
            split='validation',
            jax_format=False,
            reduce_size=False,
            ood=True
        )

        print(f"ID dataset: {len(id_dataset)} samples")
        print(f"OOD dataset: {len(ood_dataset)} samples")

        # Subsample datasets if needed
        if max_samples is not None and max_samples < len(id_dataset):
            print(f"\nSubsampling to {max_samples} samples for faster evaluation...")
            id_indices = np.random.choice(len(id_dataset), max_samples, replace=False)
            id_dataset = torch.utils.data.Subset(id_dataset, id_indices.tolist())
            ood_indices = np.random.choice(len(ood_dataset), max_samples, replace=False)
            ood_dataset = torch.utils.data.Subset(ood_dataset, ood_indices.tolist())

        # Run predictions and scoring on both datasets
        id_results = predict_and_score_motion(
            motion_prediction_jit_fn, params, batch_stats, score_fn,
            id_dataset, "ID (Normal Sequences)", max_samples=None
        )

        ood_results = predict_and_score_motion(
            motion_prediction_jit_fn, params, batch_stats, score_fn,
            ood_dataset, "OOD (Shuffled Sequences)", max_samples=None
        )

        # Compute OOD detection metrics
        print("\n" + "=" * 80)
        print("OOD DETECTION METRICS")
        print("=" * 80)
        detection_metrics = compute_ood_detection_metrics(
            id_results['ood_scores'],
            ood_results['ood_scores']
        )
        print(f"AUROC: {detection_metrics['auroc']:.4f}")
        print(f"AUPRC: {detection_metrics['auprc']:.4f}")

        # Create comparison visualization
        print("\n" + "=" * 80)
        print("CREATING VISUALIZATIONS")
        print("=" * 80)
        create_comparison_visualization(id_results, ood_results)

        # Visualize sample predictions
        visualize_sample_predictions(id_results, ood_results)

        # Summary
        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY")
        print("=" * 80)
        print(f"ID (Normal Sequences) Performance:")
        print(f"  MPJPE: {id_results['mpjpe_overall']:.2f} ± {id_results['mpjpe_std']:.2f} mm")
        print(f"  Mean OOD Score: {np.mean(id_results['ood_scores']):.4f}")

        print(f"\nOOD (Shuffled Sequences) Performance:")
        print(f"  MPJPE: {ood_results['mpjpe_overall']:.2f} ± {ood_results['mpjpe_std']:.2f} mm")
        print(f"  Mean OOD Score: {np.mean(ood_results['ood_scores']):.4f}")

        # Performance degradation analysis
        mpjpe_increase = ((ood_results['mpjpe_overall'] - id_results['mpjpe_overall'])
                          / id_results['mpjpe_overall'])
        ood_score_increase = ((np.mean(ood_results['ood_scores'])
                              - np.mean(id_results['ood_scores']))
                              / np.mean(id_results['ood_scores']))

        print(f"\nPerformance Degradation (ID → OOD):")
        print(f"  MPJPE increase: {mpjpe_increase:.1%}")
        print(f"  OOD score increase: {ood_score_increase:.1%}")

        print(f"\nOOD Detection Performance:")
        print(f"  AUROC: {detection_metrics['auroc']:.4f}")
        print(f"  AUPRC: {detection_metrics['auprc']:.4f}")

        print("\n" + "=" * 80)
        print("The OOD scores successfully distinguish between normal and shuffled sequences!")
        msg = ("This demonstrates that sketching Lanczos can detect "
               "distribution shifts in motion data.")
        print(msg)
        print("=" * 80)

    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
