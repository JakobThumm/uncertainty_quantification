#!/usr/bin/env python3
"""
Full Pipeline

Performs:
  - 2D pose estimation on left and right image + OOD detection
  - 3D triangulation
  - 3D motion prediction from estimated poses + OOD detection
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import cloudpickle
import jax.numpy as jnp

from human_pose_pipeline.motion_prediction.inference_helper import calibrate_covariance_matrices
from human_pose_pipeline.utils.visualization import plot_ood_score_histogram
from human_pose_pipeline.utils.eval_utils import (
    compute_sara_predictions,
    convert_covariance_matrices_to_set,
    evaluate_pose_prediction_scores_np,
    evaluate_uncertainty_coverage_with_covariance,
    print_coverage_stats,
    print_mpjpe_results,
    print_simple_coverage_stats_sara,
    save_coverage_stats,
    save_mpjpe_results,
    simple_coverage_stats_sara
)
from src.datasets.h36m import SPLIT, Human36mDatasetTwoCameras
from src.ood_scores.lm_lanczos import load_score_functions
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
)
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d,
    fill_pose_buffer,
    update_motion_prediction_buffer
)
from human_pose_pipeline.pose_estimation.triangulation_helper import (
    load_camera_parameters
)

from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
    OOD_THRESHOLD as POSE_OOD_THRESHOLD,
)
from human_pose_pipeline.motion_prediction.h36m_settings import (
    INPUT_HORIZON_LENGTH,
    PREDICTION_HORIZON_LENGTH,
    N_JOINTS,
    OOD_THRESHOLD as MOTION_OOD_THRESHOLD,
    N_CORRECT_POSES_REQUIRED,
    COV_CALIBRATION_CT,
    COV_CALIBRATION_IT,
    COV_CALIBRATION_FACTORS,
    SARA_MEASUREMENT_UNCERTAINTY,
    SET_LIKELIHOOD
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def main():
    """
    Main function for running 3D pose estimation on the Human3.6M dataset.
    JAX version of Marian's main function.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='3D Pose Estimation with OOD Detection')
    parser.add_argument('--cache_dir', type=str, default='cache/', help='Cache directory with score functions')
    parser.add_argument('--data_path', type=str, default='datasets/', help='Path to datasets')
    parser.add_argument('--pose_model_save_path', type=str, default='human_pose_pipeline/models/pose_estimation', help='Path to saved pose model')
    parser.add_argument('--pose_run_name', type=str, default='jax_resnet50_regressflow', help='Pose model run name')
    parser.add_argument('--pose_base_key', type=str, default=None, help='Base key for loading the pose estimation OOD score functions')
    parser.add_argument('--motion_model_save_path', type=str, default='human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle', help='Path to saved motion model')
    parser.add_argument('--motion_score_fn_path', type=str, default='human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle', help="Path to the OOD score function for the motion prediction.")
    parser.add_argument('--subsample', type=int, default=2, help='Subsampling of frames to match training camera frequency. 1 = no subsampling.')
    parser.add_argument('--split', type=str, default='validation', help='train, validation, or test')
    parser.add_argument('--action', type=str, default=None, help='Action to evaluate. Evaluate all actions if None.')
    parser.add_argument('--camera_ids', type=str, nargs=2, default=['55011271', '60457274'], help='Camera IDs')
    parser.add_argument('--max_sequences', type=int, default=10000000000, help='Maximum number of sequences to process')
    parser.add_argument('--enable_ood', action='store_true', help='Enable OOD detection')
    parser.add_argument('--output_dir', type=str, default='results/eval_full_pipeline', help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    print("=" * 60)
    print("Full Pipeline - JAX Implementation")
    print("=" * 60)

    # Configuration
    base_directory = os.path.join(root_dir, args.data_path, "H36M", "extracted")
    split = args.split
    eval_action = args.action
    camera_ids = args.camera_ids
    device = args.device
    subsample = args.subsample

    # Initialize models
    print("\nInitializing models...")

    # Initialize JAX pose estimation model with uncertainty estimation
    pose_models_dir = os.path.join(root_dir, args.pose_model_save_path, "H36M", "RegressFlow", "seed_420")
    pose_checkpoint_path_jax = os.path.join(pose_models_dir, args.pose_run_name)
    pose_estimation_jit_fn, pose_estimation_params, pose_estimation_batch_stats = initialize_jax_models(pose_checkpoint_path_jax)

    # Initialize JAX motion prediction model
    motion_model_path = os.path.join(root_dir, args.motion_model_save_path)
    motion_prediction_jit_fn, motion_prediction_params, motion_prediction_batch_stats = initialize_jax_models(motion_model_path)

    # Initialize YOLO human detector
    human_detector, device_torch = initialize_human_detector('cuda')

    print("Models initialized successfully!")

    # Load score functions
    print("\nLoading OOD score functions...")
    pose_ood_score_fn = None
    motion_ood_score_fn = None
    if args.enable_ood:
        if args.pose_base_key is None:
            print("\nWARNING: OOD detection enabled but no base_key provided. Skipping OOD detection.")
            print("Use --base_key to specify the cache key for OOD score functions.")
        else:
            print(f"\nLoading OOD score functions with cache key: {args.pose_base_key}")
            pose_ood_score_fn, _, _, _ = load_score_functions(args.cache_dir, args.pose_base_key)
            print("OOD score functions loaded successfully!")
            print(f"Using OOD threshold: {POSE_OOD_THRESHOLD:.6f}")

        if not os.path.exists(args.motion_score_fn_path):
            raise FileNotFoundError(
                f"Motion model score functions file not found: {args.motion_score_fn_path}\n"
                f"Please run score_model.py first to generate the score functions."
            )
        with open(args.motion_score_fn_path, 'rb') as f:
            motion_score_data = cloudpickle.load(f)
            motion_ood_score_fn = motion_score_data['score_fun']

    # Load camera parameters
    camera_parameters_path = os.path.join(pose_models_dir, 'camera-parameters.json')
    if not os.path.exists(camera_parameters_path):
        print(f"Warning: Camera parameters file not found at {camera_parameters_path}")
        print("Please ensure the camera-parameters.json file is available in the models directory")
        return

    # Create dataset
    dataset = Human36mDatasetTwoCameras(
        base_directory=base_directory,
        split=split,
        camera_ids=camera_ids
    )

    if len(dataset) == 0:
        print("No data found. Please check the dataset path and camera IDs.")
        return

    print(f"Dataset loaded with {len(dataset)} samples")
    counter = 0

    # Get a sample from the dataset
    poses_3d_estimated = []
    poses_3d_cov_estimated = []
    poses_3d_gt = []
    poses_3d_ood_scores = []
    poses_3d_is_ood = []
    poses_3d_human_detected = []
    motions_predicted = []
    motions_cov_predicted = []
    motions_set_radius = []
    motions_gt = []
    motions_ood_scores = []
    motions_is_ood = []
    motions_is_valid = []
    pose_buffers_good = []
    n_sequences = min(len(dataset), args.max_sequences)
    if eval_action is not None:
        eval_id = np.where(np.array([dataset.data[i]['action'] == eval_action for i in range(len(dataset.data))]))[0]
    for sample_id in range(len(dataset.data)):
        if eval_action is not None and sample_id != eval_id:
            continue
        sample = dataset[sample_id]
        all_camera_frames = sample['all_camera_frames']
        pose_sequence = sample['pose_sequence']  # Ground truth poses
        subject = sample['subject']
        action = sample['action']
        if counter >= n_sequences:
            break
        counter += 1
        intrinsics, extrinsics, projection_matrices = load_camera_parameters(camera_parameters_path, subject, camera_ids)
        # Compute projection matrices
        P1 = projection_matrices[camera_ids[0]]
        P2 = projection_matrices[camera_ids[1]]
        P1 = torch.from_numpy(P1).to(device)
        P2 = torch.from_numpy(P2).to(device)
        projection_matrices = [P1, P2]

        # Process a limited number of frames for testing
        frames_to_process = min(len(all_camera_frames[0]), len(pose_sequence))
        frames_to_process -= subsample * PREDICTION_HORIZON_LENGTH

        points_3d_buffer = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3])
        covariance_buffer = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3, 3])
        pose_valid_buffer = jnp.zeros([INPUT_HORIZON_LENGTH])

        motion_prediction_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3])
        motion_uncertainty_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3, 3])

        # Iterate through frames in a batched manner
        frame_counter = 0
        # Subsample every second frame to match motion prediction frequency.
        for frame_idx in tqdm(range(0, frames_to_process, subsample), f"Evaluating sequence {counter}/{n_sequences} of split {split}."):
            # Interleave left and right frames
            interleaved_frames = [
                all_camera_frames[0][frame_idx],
                all_camera_frames[1][frame_idx]
            ]

            points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected, _, _, _ = process_frame_3d(
                frames=interleaved_frames,
                projection_matrices=projection_matrices,
                pose_estimation_jit_fn=pose_estimation_jit_fn,
                params=pose_estimation_params,
                batch_stats=pose_estimation_batch_stats,
                human_detector=human_detector,
                device_torch=device_torch,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                score_fn=pose_ood_score_fn,
                human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
                ood_threshold=POSE_OOD_THRESHOLD,
                verbose=False,
                device=device
            )
            # process frame 3D has a batch size of 1, remove first dimension.
            # print(f"Frame {frame_counter} = {points_3d[0]}")
            points_3d = points_3d[0]
            C_3d_all = C_3d_all[0]
            # Valid prediction if not OOD and human detected
            pose_is_ood = bool(pose_is_ood)
            human_detected = bool(human_detected)
            is_valid = (not pose_is_ood) and human_detected

            points_3d_buffer, covariance_buffer, pose_valid_buffer, pose_buffer_good = fill_pose_buffer(
                points_3d_buffer=points_3d_buffer,
                covariance_buffer=covariance_buffer,
                pose_valid_buffer=pose_valid_buffer,
                points_3d=jnp.array(points_3d),
                covariance=jnp.array(C_3d_all),
                is_valid=is_valid,
                motion_prediction_buffer=motion_prediction_buffer,
                motion_uncertainty_buffer=motion_uncertainty_buffer,
            )

            # Store pose estimations
            poses_3d_estimated.append(points_3d)
            poses_3d_cov_estimated.append(C_3d_all)
            poses_3d_gt.append(pose_sequence[frame_idx])
            poses_3d_ood_scores.append(pose_ood_score)
            poses_3d_is_ood.append(pose_is_ood)
            poses_3d_human_detected.append(human_detected)

            # If enough datapoints, predict motion
            if frame_counter >= INPUT_HORIZON_LENGTH - 1 and pose_buffer_good:
                pose_input = points_3d_buffer.reshape([1, INPUT_HORIZON_LENGTH, N_JOINTS * 3])
                motion_prediction_input = jnp.concatenate([
                    pose_input,
                    covariance_buffer.reshape([1, INPUT_HORIZON_LENGTH, N_JOINTS * 3 * 3])
                ], axis=-1)
                # Model inference
                if motion_prediction_batch_stats is not None:
                    motion_predicted, (motion_cov_predicted, L) = motion_prediction_jit_fn(
                        motion_prediction_params,
                        motion_prediction_batch_stats,
                        motion_prediction_input
                    )
                else:
                    motion_predicted, (motion_cov_predicted, L) = motion_prediction_jit_fn(
                        motion_prediction_params,
                        motion_prediction_input
                    )
                if motion_ood_score_fn is not None:
                    motion_ood_score = motion_ood_score_fn(pose_input)
                else:
                    motion_ood_score = jnp.zeros([1])
                motion_predicted = motion_predicted.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)[0]
                motion_cov_predicted = motion_cov_predicted[0]
                motion_cov_predicted = calibrate_covariance_matrices(
                    covariance_matrices=motion_cov_predicted,
                    constant_time_factor=COV_CALIBRATION_CT,
                    increase_time_factor=COV_CALIBRATION_IT,
                    joint_calibration_factors=COV_CALIBRATION_FACTORS
                )
                if isinstance(motion_cov_predicted, np.ndarray):
                    motion_cov_predicted = jnp.array(motion_cov_predicted)
                motion_is_ood = bool(motion_ood_score > MOTION_OOD_THRESHOLD)
                # Update motion prediction buffer
                # TODO: Check if it makes a big difference if the saved uncertainties are the calibrated ones or not.
                motion_prediction_buffer, motion_uncertainty_buffer, valid_motion = update_motion_prediction_buffer(
                    motion_prediction_buffer=motion_prediction_buffer,
                    motion_uncertainty_buffer=motion_uncertainty_buffer,
                    predicted_motion=motion_predicted,
                    predicted_motion_uncertainty=motion_cov_predicted,
                    is_ood=motion_is_ood,
                    pose_valid_buffer=pose_valid_buffer,
                    n_correct_poses_required=N_CORRECT_POSES_REQUIRED
                )
                motion_prediction_set_radius = convert_covariance_matrices_to_set(
                    motion_uncertainty_buffer,
                    likelihood=SET_LIKELIHOOD
                )
                # Store motion predictions
                motions_predicted.append(motion_predicted)
                motions_cov_predicted.append(motion_cov_predicted)
                motions_set_radius.append(motion_prediction_set_radius)
                # Incorporate subsampling!
                motions_gt.append(pose_sequence[frame_idx + subsample : frame_idx + subsample * (PREDICTION_HORIZON_LENGTH + 1) : subsample])
                motions_ood_scores.append(motion_ood_score)
                motions_is_ood.append(motion_is_ood)
                motions_is_valid.append(valid_motion)
                pose_buffers_good.append(pose_buffer_good)
            else:
                motion_predicted = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3])
                motion_cov_predicted = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3, 3])
                motion_ood_score = jnp.zeros([1])
                valid_motion = False
                motion_is_ood = False
                motion_prediction_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3])
                motion_uncertainty_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3, 3])
                motion_prediction_set_radius = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS])

            frame_counter += 1

            # Remove GPU tensors to free memory
            # del points_3d, C_3d_all, ood_score, is_ood
    # Convert to numpy arrays
    num_frames = sum(poses_3d_gt)
    print("Full pipeline completed!")
    print(f"Processed {num_frames} frames")

    poses_3d_estimated = torch.stack(poses_3d_estimated, dim=0)
    poses_3d_cov_estimated = torch.stack(poses_3d_cov_estimated, dim=0)
    poses_3d_gt = jnp.stack(poses_3d_gt, axis=0)
    poses_3d_ood_scores = torch.stack(poses_3d_ood_scores, dim=0)
    motions_predicted = jnp.stack(motions_predicted, axis=0)
    motions_set_radius = jnp.stack(motions_set_radius, axis=0)
    motions_cov_predicted = jnp.stack(motions_cov_predicted, axis=0)
    motions_gt = jnp.array(motions_gt)

    # Move to cpu and numpy
    poses_3d_estimated_np = poses_3d_estimated.cpu().numpy()
    poses_3d_cov_estimated_np = poses_3d_cov_estimated.cpu().numpy()
    poses_3d_gt_np = np.array(poses_3d_gt)
    poses_3d_ood_scores_np = poses_3d_ood_scores.cpu().numpy()
    poses_3d_is_ood = np.array(poses_3d_is_ood)
    poses_3d_human_detected = np.array(poses_3d_human_detected)
    motions_predicted_np = np.array(motions_predicted)
    motions_set_radius_np = np.array(motions_set_radius)
    motions_cov_predicted_np = np.array(motions_cov_predicted)
    motions_gt_np = np.array(motions_gt)
    motions_ood_scores = np.array(motions_ood_scores)
    motions_is_ood = np.array(motions_is_ood)
    motions_is_valid = np.array(motions_is_valid)
    pose_buffers_good = np.array(pose_buffers_good)

    # Save raw results to pickle for further analysis
    os.makedirs(args.output_dir, exist_ok=True)
    results_pickle_file = os.path.join(args.output_dir, "full_pipeline_results.cloudpickle")
    full_pipeline_results = {
        'poses_3d_estimated': poses_3d_estimated_np,
        'poses_3d_cov_estimated': poses_3d_cov_estimated_np,
        'poses_3d_gt': poses_3d_gt_np,
        'poses_3d_ood_scores': poses_3d_ood_scores_np,
        'poses_3d_is_ood': poses_3d_is_ood,
        'poses_3d_human_detected': poses_3d_human_detected,
        'motions_predicted': motions_predicted_np,
        'motions_set_radius': motions_set_radius_np,
        'motions_cov_predicted': motions_cov_predicted_np,
        'motions_gt': motions_gt_np,
        'motions_ood_scores': motions_ood_scores,
        'motions_is_ood': motions_is_ood,
        'motions_is_valid': motions_is_valid,
        'pose_buffers_good': pose_buffers_good,
    }
    with open(results_pickle_file, 'wb') as f:
        cloudpickle.dump(full_pipeline_results, f)
    print(f"Saved full pipeline results to {results_pickle_file}")

    # Evaluate 3D pose estimation MPJPE and coverage
    print("================================")
    print("Evaluating 3D pose estimation.")
    print("================================")
    N = poses_3d_estimated_np.shape[0]
    # Convert to [B, T, J, 3] for eval
    poses_3d_estimated_np = poses_3d_estimated_np.reshape([N, 1, N_JOINTS, 3])
    poses_3d_cov_estimated_np = poses_3d_cov_estimated_np.reshape([N, 1, N_JOINTS, 3, 3])
    poses_3d_gt_np = poses_3d_gt_np.reshape([N, 1, N_JOINTS, 3])
    mpjpe, std, per_time_errors, per_time_std, per_joint_errors, per_joint_std = evaluate_pose_prediction_scores_np(
        predictions=poses_3d_estimated_np,
        targets=poses_3d_gt_np,
    )
    coverage_stats, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=poses_3d_estimated_np,
        true_poses=poses_3d_gt_np,
        cov_matrices=poses_3d_cov_estimated_np
    )
    print_mpjpe_results(mpjpe, per_time_errors, per_joint_errors)
    save_mpjpe_results(mpjpe, per_time_errors, per_joint_errors, split=split)
    print_coverage_stats(coverage_stats)
    save_coverage_stats(coverage_stats, split=split)

    # Evalute motion prediction MPJPE and coverage
    print("================================")
    print("Evaluating motion prediction.")
    print("================================")
    mpjpe, std, per_time_errors, per_time_std, per_joint_errors, per_joint_std = evaluate_pose_prediction_scores_np(
        predictions=motions_predicted_np,
        targets=motions_gt_np,
    )
    print("================================")
    print("Evaluating motion uncertainty prediction.")
    print("================================")
    coverage_stats, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=motions_predicted_np,
        true_poses=motions_gt_np,
        cov_matrices=motions_cov_predicted_np
    )
    print_mpjpe_results(mpjpe, per_time_errors, per_joint_errors)
    save_mpjpe_results(mpjpe, per_time_errors, per_joint_errors, split=split)
    print_coverage_stats(coverage_stats)
    save_coverage_stats(coverage_stats, split=split)

    coverage_stats_predictions, _ = simple_coverage_stats_sara(
        predictions=motions_predicted_np,
        radius=motions_set_radius_np,
        targets=motions_gt_np,
    )
    print(f"Predicted spherical reachable set coverage stats for {SET_LIKELIHOOD} likelihood:")
    print_simple_coverage_stats_sara(coverage_stats_predictions)

    print("================================")
    print("Evaluating motion SARA uncertainty.")
    print("================================")
    dt = 1.0 / 25.0
    prediction_horizon_times = [(t + 1) * dt for t in range(PREDICTION_HORIZON_LENGTH)]

    # Evaluate SARA-style
    sara_predictions, sara_radius = compute_sara_predictions(
        last_input_poses=poses_3d_estimated_np[INPUT_HORIZON_LENGTH - 1:, 0, ...],
        prediction_horizon_times=prediction_horizon_times,
        v_human=1.6,
        measurement_uncertainty=SARA_MEASUREMENT_UNCERTAINTY
    )
    coverage_stats_sara, _ = simple_coverage_stats_sara(
        predictions=sara_predictions,
        radius=sara_radius,
        targets=motions_gt_np,
    )
    print("SARA simple velocity model coverage stats:")
    print_simple_coverage_stats_sara(coverage_stats_sara)

    # Save OOD score histograms
    if args.enable_ood:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_ood_score_histogram(
            scores=poses_3d_ood_scores_np,
            threshold=POSE_OOD_THRESHOLD,
            title='2D Pose Prediction OOD Score Distribution',
            xlabel='OOD Score',
            save_path=os.path.join(args.output_dir, 'ood_histogram_pose_prediction.png'),
        )
        plot_ood_score_histogram(
            scores=motions_ood_scores,
            threshold=MOTION_OOD_THRESHOLD,
            title='Motion Prediction OOD Score Distribution',
            xlabel='OOD Score',
            save_path=os.path.join(args.output_dir, 'ood_histogram_motion_prediction.png'),
        )


if __name__ == "__main__":
    main()
