#!/usr/bin/env python3
"""
Full Pipeline

Performs:
  - 2D pose estimation on left and right image + OOD detection
  - 3D triangulation
  - 3D motion prediction from estimated poses + OOD detection
"""

import os
import argparse
from time import time
import numpy as np
import torch
from tqdm import tqdm
import cloudpickle
import jax.numpy as jnp
from PIL import Image

from ultralytics import YOLO
from human_pose_pipeline.motion_prediction.inference_helper import calibrate_covariance_matrices
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
from src.datasets.human_rgbd import HumanRGBDDataset
from src.ood_scores.lm_lanczos import load_score_functions
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
)
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d,
    fill_pose_buffer,
    process_frame_3d_from_rgbd_yolo,
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
    COV_CALIBRATION_HF,
    COV_CALIBRATION_FF,
    COV_CALIBRATION_HI,
    COV_CALIBRATION_FI,
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
    parser.add_argument('--data_path', type=str, default='datasets/rgbd_test/', help='Path to datasets')
    parser.add_argument("--yolo_model", type=str, default="yolo26n-pose.pt",
                        help="YOLO model name (e.g., yolo11n-pose.pt, yolo26n-pose.pt)")
    parser.add_argument('--motion_model_save_path', type=str, default='human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle', help='Path to saved motion model')
    parser.add_argument('--pose_run_name', type=str, default='jax_resnet50_regressflow', help='Pose model run name')
    parser.add_argument('--pose_base_key', type=str, default=None, help='Base key for loading the pose estimation OOD score functions')
    parser.add_argument('--motion_score_fn_path', type=str, default='human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle', help="Path to the OOD score function for the motion prediction.")
    parser.add_argument('--max_frames', type=int, default=10000000000, help='Maximum number of frames to process')
    parser.add_argument('--subsample', type=int, default=1, help='Subsampling of frames to match training camera frequency. Default 1 = no subsampling.')
    parser.add_argument('--enable_ood', action='store_true', help='Enable OOD detection')
    parser.add_argument('--enable_tracking', action='store_true', help='Enable YOLO tracking')
    parser.add_argument('--depth_uncertainty', type=float, default=0.002,
                        help='Assumed depth std-dev in metres for uncertainty propagation')
    parser.add_argument('--output_dir', type=str, default='results/pose_3d', help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    print("=" * 60)
    print("Full Pipeline - JAX Implementation")
    print("=" * 60)

    # Configuration
    device = args.device

    # Initialize models
    print("\nInitializing models...")

    # Initialize YOLO pose estimation model with uncertainty estimation
    print("\nInitializing YOLO pose model...")
    yolo_model = YOLO(args.yolo_model)

    if device == 'cuda':
        yolo_model.to('cuda')
        print(f"Model loaded on CUDA (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print("Model loaded on CPU")

    # Initialize JAX motion prediction model
    motion_model_path = os.path.join(root_dir, args.motion_model_save_path)
    motion_prediction_jit_fn, motion_prediction_params, motion_prediction_batch_stats = \
        initialize_jax_models(motion_model_path)

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
            print(f"Using OOD threshold: {args.ood_threshold:.6f}")

        if not os.path.exists(args.motion_score_fn_path):
            raise FileNotFoundError(
                f"Motion model score functions file not found: {args.motion_score_fn_path}\n"
                f"Please run score_model.py first to generate the score functions."
            )
        with open(args.motion_score_fn_path, 'rb') as f:
            motion_score_data = cloudpickle.load(f)
            motion_ood_score_fn = motion_score_data['score_fun']

    # Create dataset
    print("\nLoading RGB-D dataset...")
    data_path = args.data_path or os.path.join(root_dir, "datasets", "rgbd_test")

    dataset = HumanRGBDDataset(
        base_directory=data_path,
    )

    if len(dataset) == 0:
        print("No data found. Please check the dataset path and camera IDs.")
        return

    max_frames = args.max_frames
    # Process a limited number of frames for testing
    frames_to_process = min(len(dataset), max_frames)
    subsample = args.subsample
    print(f"Dataset loaded with {len(dataset)} frames, from which we are using {frames_to_process} frames \
            with {subsample} subsampling.")

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
    motions_frame_ids = []
    pose_buffers_good = []

    points_3d_buffer = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3])
    covariance_buffer = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3, 3])
    pose_valid_buffer = jnp.zeros([INPUT_HORIZON_LENGTH])

    motion_prediction_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3])
    motion_uncertainty_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3, 3])

    # Iterate through frames in a batched manner
    start_at = 30
    frame_counter = 0
    # Subsample every second frame to match motion prediction frequency.
    for frame_idx in tqdm(range(start_at, frames_to_process, subsample), "Evaluating sequence:"):
        sample = dataset[frame_idx]
        image_pil = sample['color_raw']
        depth_img = sample['depth_raw']
        # Could be moved outside the frame loop
        subject = sample['filename']
        camera_intrinsics = sample['camera_intrinsics']

        # Create a batch out of single instances
        rgb_batch = [image_pil]
        depth_batch = [depth_img]
        R_rect_to_world = sample['R_rect_to_world']  # (3, 3) numpy array
        t_rect_to_world = sample['t_rect_to_world']  # (3,) numpy array

        t3 = time()
        points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected, _, _, _ = \
            process_frame_3d_from_rgbd_yolo(
                rgb_frames=rgb_batch,
                depth_frames=depth_batch,
                camera_intrinsics=camera_intrinsics,
                yolo_pose_model=yolo_model,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                enable_tracking=args.enable_tracking,
                confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
                verbose=False,
                device=device,
                depth_uncertainty=args.depth_uncertainty,
                R_rect_to_world=R_rect_to_world,
                t_rect_to_world=t_rect_to_world,
            )
        t4 = time()
        # print(f"Time for batch processing: {t4 - t3:.3f}s")

        # process frame 3D has a batch size of 1, remove first dimension.
        points_3d = points_3d[0]
        C_3d_all = C_3d_all[0]
        # We don't have GT poses so we use the predictions for the motion prediction.
        gt_pose = points_3d
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
        poses_3d_gt.append(gt_pose)
        poses_3d_ood_scores.append(pose_ood_score)
        poses_3d_is_ood.append(pose_is_ood)
        poses_3d_human_detected.append(human_detected)

        # If enough datapoints, predict motion
        if frame_counter >= INPUT_HORIZON_LENGTH - 1 and \
           frame_counter < frames_to_process - start_at - PREDICTION_HORIZON_LENGTH and \
           pose_buffer_good:
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
                motion_ood_score = 0.0
            motion_predicted = motion_predicted.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)[0]
            motion_cov_predicted = motion_cov_predicted[0]
            motion_cov_predicted = calibrate_covariance_matrices(
                covariance_matrices=motion_cov_predicted,
                constant_time_factor=COV_CALIBRATION_CT,
                increase_time_factor=COV_CALIBRATION_IT,
                hand_factor=COV_CALIBRATION_HF,
                feet_factor=COV_CALIBRATION_FF,
                hand_indices=COV_CALIBRATION_HI,
                feet_indices=COV_CALIBRATION_FI
            )
            if isinstance(motion_cov_predicted, np.ndarray):
                motion_cov_predicted = jnp.array(motion_cov_predicted)
            motion_is_ood = bool(motion_ood_score > MOTION_OOD_THRESHOLD)
            # Update motion prediction buffer
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
            motions_frame_ids.append(frame_counter)
            # Store motion predictions
            motions_predicted.append(motion_predicted)
            motions_cov_predicted.append(motion_cov_predicted)
            motions_set_radius.append(motion_prediction_set_radius)
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

    # Fill motions GT
    for frame_id in motions_frame_ids:
        motions_gt.append(torch.stack(poses_3d_gt[frame_id + 1 : frame_id + PREDICTION_HORIZON_LENGTH + 1], dim=0))

    # Convert to numpy arrays
    num_frames = sum(poses_3d_gt)
    print("Full pipeline completed!")
    print(f"Processed {num_frames} frames")

    poses_3d_estimated = torch.stack(poses_3d_estimated, dim=0)
    poses_3d_cov_estimated = torch.stack(poses_3d_cov_estimated, dim=0)
    poses_3d_gt = torch.stack(poses_3d_gt, dim=0)
    poses_3d_ood_scores = torch.stack(poses_3d_ood_scores, dim=0)
    motions_predicted = jnp.stack(motions_predicted, axis=0)
    motions_set_radius = jnp.stack(motions_set_radius, axis=0)
    motions_cov_predicted = jnp.stack(motions_cov_predicted, axis=0)
    motions_gt = torch.stack(motions_gt, dim=0)

    # Move to cpu and numpy
    poses_3d_estimated_np = poses_3d_estimated.cpu().numpy()
    poses_3d_cov_estimated_np = poses_3d_cov_estimated.cpu().numpy()
    poses_3d_gt_np = poses_3d_gt.cpu().numpy()
    poses_3d_ood_scores_np = poses_3d_ood_scores.cpu().numpy()
    poses_3d_is_ood = np.array(poses_3d_is_ood)
    poses_3d_human_detected = np.array(poses_3d_human_detected)
    motions_predicted_np = np.array(motions_predicted)
    motions_set_radius_np = np.array(motions_set_radius)
    motions_cov_predicted_np = np.array(motions_cov_predicted)
    motions_gt_np = motions_gt.cpu().numpy()
    motions_ood_scores = np.array(motions_ood_scores)
    motions_is_ood = np.array(motions_is_ood)
    motions_is_valid = np.array(motions_is_valid)
    pose_buffers_good = np.array(pose_buffers_good)

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
    save_mpjpe_results(mpjpe, per_time_errors, per_joint_errors)
    print_coverage_stats(coverage_stats)
    save_coverage_stats(coverage_stats)

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
    save_mpjpe_results(mpjpe, per_time_errors, per_joint_errors)
    print_coverage_stats(coverage_stats)
    save_coverage_stats(coverage_stats)

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
        last_input_poses=poses_3d_estimated_np[INPUT_HORIZON_LENGTH - 1:-PREDICTION_HORIZON_LENGTH, 0, ...],
        prediction_horizon_times=prediction_horizon_times,
        v_human=1.6
    )
    coverage_stats_sara, _ = simple_coverage_stats_sara(
        predictions=sara_predictions,
        radius=sara_radius,
        targets=motions_gt_np,
    )
    print("SARA simple velocity model coverage stats:")
    print_simple_coverage_stats_sara(coverage_stats_sara)

    # Print OOD statistics if enabled
    # TODO


if __name__ == "__main__":
    main()
