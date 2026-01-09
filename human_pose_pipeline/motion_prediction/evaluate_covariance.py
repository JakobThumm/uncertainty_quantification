"""Script to tune the diagonal offset for covariance matrices to improve uncertainty coverage."""

import os
import numpy as np
import matplotlib.pyplot as plt
import cloudpickle
from scipy.stats import chi2
from pathlib import Path

from human_pose_pipeline.motion_prediction.inference_helper import calibrate_covariance_matrices
from human_pose_pipeline.utils.eval_utils import compute_sara_predictions, convert_covariance_matrices_to_set, evaluate_uncertainty_coverage_with_covariance, print_coverage_stats, print_simple_coverage_stats_sara, simple_coverage_stats_sara
from human_pose_pipeline.motion_prediction.h36m_settings import OOD_THRESHOLD, PREDICTION_HORIZON_LENGTH


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tune covariance matrix diagonal offset for better coverage")
    parser.add_argument(
        "--results_file",
        type=str,
        default="results/motion_prediction/motion_prediction_results_train.cloudpickle",
        help="Path to motion prediction results file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/motion_prediction/coverage_tuning",
        help="Output directory for plots"
    )

    args = parser.parse_args()

    output_dir = os.path.join(root_dir, args.output_dir)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load results
    results_file = os.path.join(root_dir, args.results_file)
    print(f"Loading results from {results_file}...")
    with open(results_file, 'rb') as f:
        results = cloudpickle.load(f)

    n_plot = 100000
    n_plot = min(results['predictions'].shape[0], n_plot)

    predictions = np.array(results['predictions'])[:n_plot]
    targets = np.array(results['targets'])[:n_plot]
    covariance_matrices = np.array(results['covariance_matrices'])[:n_plot]  # Shape: [N, n_t, n_j, 3, 3]
    ood_scores = np.array(results['ood_scores'])[:n_plot]
    last_input_poses = np.array(results['last_input_poses'])[:n_plot]

    N, T, J, _ = predictions.shape
    last_input_poses = np.reshape(last_input_poses[..., :J*3], [N, J, 3])

    # Filter out OOD samples
    # is_ood = ood_scores > 6e5
    # predictions = predictions[~is_ood]
    # targets = targets[~is_ood]
    # covariance_matrices = covariance_matrices[~is_ood]
    # ood_scores = ood_scores[~is_ood]

    print(f"Loaded predictions shape: {predictions.shape}")
    print(f"Loaded targets shape: {targets.shape}")
    print(f"Loaded covariance matrices shape: {covariance_matrices.shape}")

    # Increase covariance for certain times and joints
    covariance_matrices = calibrate_covariance_matrices(
        covariance_matrices=covariance_matrices,
        constant_time_factor=1.2,
        increase_time_factor=0.4,
        hand_factor=1.7,
        feet_factor=1.5,
        hand_indices=[5, 6],
        feet_indices=[11, 12]
    )
    # Generate n_std range
    n_std_range = [1, 2, 3, 4]

    # Compute ideal coverage
    ideal_coverages = [68.2, 95.4, 99.7, 99.99]

    # Compute coverage
    coverage_stats, within_stds = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=predictions, true_poses=targets, cov_matrices=covariance_matrices
    )

    # Print coverage statistics
    print_coverage_stats(coverage_stats)

    dt = 1.0 / 25.0
    prediction_horizon_times = [(t + 1) * dt for t in range(PREDICTION_HORIZON_LENGTH)]

    # Evaluate SARA-style
    sara_predictions, sara_radius = compute_sara_predictions(
        last_input_poses=last_input_poses,
        prediction_horizon_times=prediction_horizon_times,
        v_human=1.6
    )
    coverage_stats_sara, _ = simple_coverage_stats_sara(
        predictions=sara_predictions,
        radius=sara_radius,
        targets=targets,
    )
    print("SARA simple velocity model coverage stats:")
    print_simple_coverage_stats_sara(coverage_stats_sara)

    likelihood = 0.99
    radius_predictions = convert_covariance_matrices_to_set(
        covariance_matrices,
        likelihood=likelihood
    )
    coverage_stats_predictions, _ = simple_coverage_stats_sara(
        predictions=predictions,
        radius=radius_predictions,
        targets=targets,
    )
    print(f"Predicted spherical reachable set coverage stats for {likelihood} likelihood:")
    print_simple_coverage_stats_sara(coverage_stats_predictions)

    # Plot predicted uncertainty increase over frame for each joint
    plt.figure(figsize=(12, 8))
    uncertainties = np.trace(covariance_matrices, axis1=3, axis2=4)  # [N, n_t, n_j]
    uncertainty_increase = uncertainties * (1 / uncertainties[:, 0, ...])[:, None, :]
    for i in range(uncertainty_increase.shape[2]):
        plt.plot(np.arange(T), np.mean(uncertainty_increase[:, :, i], axis=0), label=f"Joint {i}")
    plt.xlabel("Time point")
    plt.ylabel("Uncertainty increase")
    plt.legend()
    output_file = os.path.join(output_dir, 'uncertainty_increase_over_time.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    plt.close

    # Plot predicted uncertainty over OOD score
    within_std_stat = np.mean(np.sum(np.array(within_stds), axis=0), axis=(1, 2))
    plt.figure(figsize=(12, 8))
    predicted_uncertainty_all = np.mean(np.trace(covariance_matrices, axis1=3, axis2=4), axis=(1, 2))
    # Reduce plotting to critical
    critical_points = within_std_stat <= 10
    ood_scores_plot = ood_scores[critical_points]
    predicted_uncertainty_all_plot = predicted_uncertainty_all[critical_points]
    c_plot = within_std_stat[critical_points]
    plt.scatter(ood_scores_plot, predicted_uncertainty_all_plot, c=c_plot, cmap='viridis')
    plt.colorbar()
    plt.xlabel("OOD score")
    plt.ylabel("Mean trace of cov. matrices")
    output_file = os.path.join(output_dir, 'uncertainty_over_ood_all.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to {output_file}")
    plt.close

    # Plot predicted uncertainty over OOD score for every joint
    print("\nDone!")


if __name__ == "__main__":
    main()
