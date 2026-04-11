"""This script evaluates a motion prediction model on the Human3.6M dataset."""

import os
from time import time
import argparse
import numpy as np
from sympy import per
import torch
import cloudpickle
from torch.utils.data import DataLoader
import jax.numpy as jnp
from tqdm import tqdm
from human_pose_pipeline.pose_estimation.inference_helper import initialize_jax_models
from human_pose_pipeline.motion_prediction.inference_helper import calibrate_covariance_matrices, predict_poses
from human_pose_pipeline.utils.eval_utils import (
    compute_sara_predictions,
    convert_covariance_matrices_to_set,
    evaluate_uncertainty_coverage_with_covariance,
    print_coverage_stats,
    print_mpjpe_results,
    print_simple_coverage_stats_sara,
    save_coverage_stats,
    save_coverage_stats_sara,
    save_mpjpe_results,
    simple_coverage_stats_sara,
)
from src.datasets import dataloader_from_string
from src.models.dct_pose_transformer import DCTPoseTransformer
from src.datasets.h36m_motion_prediction import Human36mMotionDataset3D
from human_pose_pipeline.utils.visualization import visualize_motion_prediction
from human_pose_pipeline.pose_estimation.h36m_settings import CONNECTIONS_13
from human_pose_pipeline.motion_prediction.h36m_settings import (
    PREDICTION_HORIZON_LENGTH,
    N_JOINTS,
    OOD_THRESHOLD,
    COV_CALIBRATION_FACTORS,
    COV_CALIBRATION_CT,
    COV_CALIBRATION_IT,
    SARA_MEASUREMENT_UNCERTAINTY,
    SET_LIKELIHOOD,
)
from human_pose_pipeline.utils.eval_utils import evaluate_pose_prediction_scores_np as evaluate_scores

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

BATCH_SIZE = 16
FPS = 25.0


def main():
    parser = argparse.ArgumentParser(description="3D Pose Estimation with OOD Detection")
    # parser.add_argument('--cache_dir', type=str, default='cache/', help='Cache directory with score functions')
    # parser.add_argument('--base_key', type=str, default=None, help='Base key for loading the OOD score functions')
    parser.add_argument("--data_path", type=str, default="datasets/", help="Path to datasets")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle",
        help="Path to saved models",
    )
    parser.add_argument("--split", type=str, default="validation", help="train, validation, or test")
    # parser.add_argument('--run_name', type=str, default='finetuned_h36m_regressflow_with_unc', help='Model run name')
    # parser.add_argument('--ood_threshold', type=float, default=OOD_THRESHOLD, help='OOD threshold')
    # parser.add_argument('--subject', type=str, default='S1', help='Subject ID (e.g., S1, S6)')
    # parser.add_argument('--action', type=str, default='WalkingDog', help='Action to visualize')
    # parser.add_argument('--camera_ids', type=str, nargs=2, default=['55011271', '60457274'], help='Camera IDs')
    # parser.add_argument('--max_frames', type=int, default=100, help='Maximum number of frames to process')
    parser.add_argument("--enable_ood", action="store_true", help="Enable OOD detection on left camera")
    parser.add_argument(
        "--motion_score_fn_path",
        type=str,
        default="human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle",
        help="Path to the OOD score function for the motion prediction.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/motion_prediction", help="Output directory for results"
    )
    parser.add_argument(
        "--dataset_name",
        default="Human36mMotionDataset3DWithInputUncertainty",
        help="Dataset name to\
        validate on. Choose from: Human36mMotionDataset3D and Human36mMotionDataset3DWithInputUncertainty (Default).",
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

    print("\nLoading OOD score functions...")
    pose_ood_score_fn = None
    motion_ood_score_fn = None
    if args.enable_ood:
        if not os.path.exists(args.motion_score_fn_path):
            raise FileNotFoundError(
                f"Motion model score functions file not found: {args.motion_score_fn_path}\n"
                f"Please run score_model.py first to generate the score functions."
            )
        with open(args.motion_score_fn_path, "rb") as f:
            motion_score_data = cloudpickle.load(f)
            motion_ood_score_fn = motion_score_data["score_fun"]

    # Load dataset
    print("\nLoading H36M dataset...")
    data_path = os.path.join(root_dir, args.data_path)  # , "H36M", "extracted")
    dataset_name = args.dataset_name
    train_loader, valid_loader, test_loader = dataloader_from_string(
        dataset_name,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=420,
        download=False,
        data_path=data_path,
    )
    if args.split == "train":
        data_loader = train_loader
    elif args.split == "validation":
        data_loader = valid_loader
    elif args.split == "test":
        data_loader = test_loader
    else:
        raise NotImplementedError(f"Split {args.split} unknown.")
    # print(f"Loaded {len(dataset)} sequences.")

    # >>> Test dataset <<<
    print("\n" + "=" * 60)
    print(f"RESULTS for split {args.split}")
    print("=" * 60)
    predictions, targets, covariance_matrices, ood_scores, is_oods, last_input_poses = predict_poses(
        motion_prediction_jit_fn=motion_prediction_jit_fn,
        params=params,
        batch_stats=batch_stats,
        dataset_loader=data_loader,
        motion_ood_score_fn=motion_ood_score_fn,
        ood_threshold=OOD_THRESHOLD,
        device=device,
    )
    predictions = predictions.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)
    targets = targets.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)

    # Save all data
    os.makedirs(args.output_dir, exist_ok=True)
    results_cloudpickle_file = os.path.join(args.output_dir, f"motion_prediction_results_{args.split}.cloudpickle")

    motion_prediction_results = {
        "predictions": predictions,
        "targets": targets,
        "covariance_matrices": covariance_matrices,
        "ood_scores": ood_scores,
        "is_oods": is_oods,
        "last_input_poses": last_input_poses,
    }

    with open(results_cloudpickle_file, "wb") as f:
        cloudpickle.dump(motion_prediction_results, f)
        print(f"Saved results to {results_cloudpickle_file}")

    print("================================")
    print("Evaluating motion prediction.")
    print("================================")
    mpjpe, std_score, per_time_errors, per_time_stds, per_joint_errors, per_joint_std = evaluate_scores(
        predictions, targets
    )
    print("================================")
    print("Evaluating motion uncertainty prediction.")
    print("================================")
    coverage_stats, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=predictions, true_poses=targets, cov_matrices=covariance_matrices
    )
    print_mpjpe_results(mpjpe, per_time_errors, per_joint_errors)
    save_mpjpe_results(mpjpe, per_time_errors, per_joint_errors, split=args.split, output_dir=args.output_dir)
    print_coverage_stats(coverage_stats)
    save_coverage_stats(coverage_stats, split=args.split, output_dir=args.output_dir)

    print("================================")
    print("Evaluating conformal prediction sets.")
    print("================================")
    predictions = np.array(predictions)
    targets = np.array(targets)
    last_input_poses = np.array(last_input_poses)[..., :N_JOINTS * 3].reshape(-1, N_JOINTS, 3)
    # Increase covariance for certain times and joints
    covariance_matrices_calibrated = calibrate_covariance_matrices(
        covariance_matrices=covariance_matrices,
        constant_time_factor=COV_CALIBRATION_CT,
        increase_time_factor=COV_CALIBRATION_IT,
        joint_calibration_factors=COV_CALIBRATION_FACTORS,
    )
    print("Coverage Stats After Calibration")
    # Compute coverage
    coverage_stats_calibrated, within_stds_calibrated = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=predictions, true_poses=targets, cov_matrices=covariance_matrices_calibrated
    )
    print_coverage_stats(coverage_stats_calibrated)
    radius_conformal_prediction_sets = convert_covariance_matrices_to_set(
        np.array(covariance_matrices_calibrated), likelihood=SET_LIKELIHOOD
    )
    coverage_stats_conformal_prediction_sets, _ = simple_coverage_stats_sara(
        predictions=predictions,
        radius=radius_conformal_prediction_sets,
        targets=targets,
    )
    print(f"Predicted spherical reachable set coverage stats for {SET_LIKELIHOOD} likelihood:")
    print_simple_coverage_stats_sara(coverage_stats_conformal_prediction_sets)
    save_coverage_stats_sara(
        coverage_stats_conformal_prediction_sets,
        filename="coverage_stats_conformal_prediction_sets",
        output_dir=args.output_dir,
    )
    print("====================================")
    print("SARA Coverage Stats")
    print("====================================")

    dt = 1.0 / FPS
    prediction_horizon_times = [(t + 1) * dt for t in range(PREDICTION_HORIZON_LENGTH)]

    # Evaluate SARA-style
    sara_predictions, sara_radius = compute_sara_predictions(
        last_input_poses=last_input_poses,
        prediction_horizon_times=prediction_horizon_times,
        v_human=1.6,
        measurement_uncertainty=SARA_MEASUREMENT_UNCERTAINTY,
    )
    coverage_stats_sara, _ = simple_coverage_stats_sara(
        predictions=sara_predictions,
        radius=sara_radius,
        targets=targets,
    )
    print("SARA simple velocity model coverage stats:")
    print_simple_coverage_stats_sara(coverage_stats_sara)
    save_coverage_stats_sara(coverage_stats_sara, filename="coverage_stats_sara", output_dir=args.output_dir)

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
