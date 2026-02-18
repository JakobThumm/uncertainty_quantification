#!/usr/bin/env python3
"""
3D Pose Estimation - Stereo + Constant Depth Uncertainty (JAX)

Variant of pose_estimation_3D_full_eval.py that replaces the propagated depth
covariance from stereo triangulation with a user-specified constant uncertainty.
This is useful for the H36M wide-baseline setup where the lateral (X, Y)
position from triangulation is reliable but the stereo depth covariance estimate
is poor.

Pipeline per frame
  1. Run YOLO detection + JAX RegressFlow on both cameras and triangulate.
  2. Rotate the per-joint covariance to the primary camera frame.
  3. Replace the depth (Z) variance with sigma_depth² and zero cross-terms.
  4. Rotate the covariance back to world frame.
  5. Evaluate MPJPE and uncertainty coverage.
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

from human_pose_pipeline.utils.eval_utils import (
    evaluate_pose_prediction_scores_np,
    evaluate_uncertainty_coverage_with_covariance,
    print_coverage_stats,
    print_mpjpe_results,
    save_coverage_stats,
    save_mpjpe_results,
)
from src.datasets.h36m import Human36mDatasetTwoCameras
from src.ood_scores.lm_lanczos import load_score_functions
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
)
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d,
    set_depth_uncertainty_to_constant,
)
from human_pose_pipeline.pose_estimation.triangulation_helper import load_camera_parameters
from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
    OOD_THRESHOLD,
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def main():
    parser = argparse.ArgumentParser(
        description='3D Pose Estimation — Stereo + Constant Depth Uncertainty (JAX)'
    )
    parser.add_argument('--data_path', type=str, default='datasets/',
                        help='Root path containing H36M/extracted/')
    parser.add_argument('--model_save_path', type=str,
                        default='human_pose_pipeline/models/pose_estimation',
                        help='Path to saved models directory')
    parser.add_argument('--run_name', type=str,
                        default='finetuned_h36m_regressflow_with_unc',
                        help='Model checkpoint run name')
    parser.add_argument('--split', type=str, default='validation',
                        help='Dataset split: train / validation / test')
    parser.add_argument('--camera_ids', type=str, nargs=2,
                        default=['55011271', '60457274'],
                        help='Two camera IDs for stereo (primary first)')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum number of sequences to process')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Frames per inference batch')
    parser.add_argument('--sigma_depth', type=float, default=2.0,
                        help='Constant depth std dev in mm (replaces stereo covariance)')
    parser.add_argument('--enable_ood', action='store_true',
                        help='Enable OOD detection; requires --base_key')
    parser.add_argument('--base_key', type=str, default=None,
                        help='Cache key for OOD score functions')
    parser.add_argument('--cache_dir', type=str, default='cache/',
                        help='Cache directory for OOD score functions')
    parser.add_argument('--ood_threshold', type=float, default=OOD_THRESHOLD)
    parser.add_argument('--output_dir', type=str,
                        default='results/pose_3d_stereo_const_depth',
                        help='Directory for result files')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("=" * 60)
    print("3D Pose Estimation — Stereo + Constant Depth Uncertainty (JAX)")
    print(f"sigma_depth={args.sigma_depth} mm")
    print("=" * 60)

    base_directory = os.path.join(root_dir, args.data_path, 'H36M', 'extracted')
    models_dir = os.path.join(
        root_dir, args.model_save_path, 'H36M', 'RegressFlow', 'seed_420'
    )
    camera_params_path = os.path.join(models_dir, 'camera-parameters.json')

    if not os.path.exists(camera_params_path):
        print(f"Camera parameters not found: {camera_params_path}")
        return

    # ------------------------------------------------------------------
    # Initialise models
    # ------------------------------------------------------------------
    print("\nInitialising models...")
    checkpoint_path = os.path.join(models_dir, args.run_name)
    pose_estimation_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path)
    human_detector, device_torch = initialize_human_detector(args.device)
    print("Models initialised.")

    score_fn = None
    if args.enable_ood:
        if args.base_key is None:
            print("WARNING: --enable_ood set but --base_key not provided; OOD disabled.")
        else:
            print(f"Loading OOD score functions (key={args.base_key}) ...")
            score_fn, _, _, _ = load_score_functions(args.cache_dir, args.base_key)
            print(f"OOD threshold: {args.ood_threshold:.6f}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = Human36mDatasetTwoCameras(
        base_directory=base_directory,
        split=args.split,
        camera_ids=args.camera_ids,
    )
    if len(dataset) == 0:
        print("No data found.")
        return
    print(f"\nDataset: {len(dataset)} sequences")

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    all_3d_points_list = []
    all_3d_cov_list = []
    all_gt_list = []
    all_ood_scores_list = []
    all_is_ood_list = []
    all_batch_sizes = []

    counter = 0
    for sample in tqdm(dataset, desc="Processing sequences"):
        if args.max_sequences is not None and counter >= args.max_sequences:
            break
        counter += 1

        all_camera_frames = sample['all_camera_frames']
        pose_sequence = sample['pose_sequence']
        subject = sample['subject']

        intrinsics, extrinsics, projection_matrices = load_camera_parameters(
            camera_params_path, subject, args.camera_ids
        )
        R_world_to_cam = extrinsics[args.camera_ids[0]][:3, :3]  # (3,3) numpy

        P1 = torch.from_numpy(projection_matrices[args.camera_ids[0]]).to(args.device)
        P2 = torch.from_numpy(projection_matrices[args.camera_ids[1]]).to(args.device)

        frames_to_process = min(len(all_camera_frames[0]), len(pose_sequence))

        for frame_idx in range(0, frames_to_process, args.batch_size):
            B_req = min(args.batch_size, frames_to_process - frame_idx)
            left_frames = all_camera_frames[0][frame_idx:frame_idx + B_req]
            right_frames = all_camera_frames[1][frame_idx:frame_idx + B_req]
            interleaved = [x for pair in zip(left_frames, right_frames) for x in pair]

            points_3d, C_3d_all, ood_score, is_ood, _, _, _, _ = process_frame_3d(
                frames=interleaved,
                projection_matrices=[P1, P2],
                pose_estimation_jit_fn=pose_estimation_jit_fn,
                params=params,
                batch_stats=batch_stats,
                human_detector=human_detector,
                device_torch=device_torch,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                score_fn=score_fn,
                human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
                ood_threshold=args.ood_threshold,
                verbose=False,
                device=args.device,
            )

            C_3d_all = set_depth_uncertainty_to_constant(
                C_3d_all, R_world_to_cam, args.sigma_depth, device=args.device
            )

            all_batch_sizes.append(B_req)
            all_3d_points_list.append(points_3d.cpu().numpy())
            all_3d_cov_list.append(C_3d_all.cpu().numpy())
            all_gt_list.append(pose_sequence[frame_idx:frame_idx + B_req])
            all_ood_scores_list.append(ood_score.cpu().numpy())
            all_is_ood_list.append(is_ood.cpu().numpy())

            del points_3d, C_3d_all, ood_score, is_ood

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    num_frames = sum(all_batch_sizes)
    print(f"\nProcessed {num_frames} frames from {counter} sequences")

    all_3d_points = np.zeros((num_frames, 13, 3))
    all_3d_cov = np.zeros((num_frames, 13, 3, 3))
    all_gt_points = np.zeros((num_frames, 13, 3))
    all_ood_scores = np.zeros(num_frames)
    all_is_ood = np.zeros(num_frames, dtype=bool)
    idx = 0
    for i, bs in enumerate(all_batch_sizes):
        all_3d_points[idx:idx + bs] = all_3d_points_list[i]
        all_3d_cov[idx:idx + bs] = all_3d_cov_list[i]
        all_gt_points[idx:idx + bs] = all_gt_list[i]
        all_ood_scores[idx:idx + bs] = all_ood_scores_list[i]
        all_is_ood[idx:idx + bs] = all_is_ood_list[i]
        idx += bs

    if args.enable_ood and score_fn is not None:
        print(f"\nOOD: mean={all_ood_scores.mean():.4f}  "
              f"OOD frames={all_is_ood.sum()}/{len(all_is_ood)} "
              f"({100 * all_is_ood.mean():.1f}%)")

    good = ~all_is_ood
    all_3d_points = all_3d_points[good]
    all_3d_cov = all_3d_cov[good]
    all_gt_points = all_gt_points[good]
    num_frames = all_3d_points.shape[0]
    print(f"Valid frames: {num_frames}")

    preds = all_3d_points.reshape(1, num_frames, 13, 3)
    gts = all_gt_points.reshape(1, num_frames, 13, 3)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    mpjpe, _, per_time_errors, _, per_joint_errors, _ = \
        evaluate_pose_prediction_scores_np(predictions=preds, targets=gts)

    coverage_stats, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=preds, true_poses=gts, cov_matrices=all_3d_cov
    )

    output_dir = os.path.join(args.output_dir, f"sigma{args.sigma_depth:.1f}mm")
    os.makedirs(output_dir, exist_ok=True)

    print_mpjpe_results(mpjpe, per_time_errors, per_joint_errors,
                        print_per_time_errors=False)
    save_mpjpe_results(mpjpe, per_time_errors, per_joint_errors,
                       split=args.split, output_dir=output_dir)
    print_coverage_stats(coverage_stats, print_per_time_stats=False)
    save_coverage_stats(coverage_stats, split=args.split, output_dir=output_dir)
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
