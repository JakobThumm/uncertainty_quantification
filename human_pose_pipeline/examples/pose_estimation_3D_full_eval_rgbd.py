#!/usr/bin/env python3
"""
3D Pose Estimation from Emulated RGB-D — Full Evaluation

Uses Human36mDatasetEmulatedRGBD to obtain a depth map for each camera by
stereo-matching with its paired neighbour camera, then lifts 2D pose predictions
to 3D with process_frame_3d_from_rgbd (depth lifting + uncertainty propagation).

Pipeline per frame
  1. Load primary camera frame + paired frame → rectified RGB + depth map.
  2. Run YOLO + RegressFlow to get 2D pose + aleatoric uncertainty.
  3. Lift to 3D via depth and propagate uncertainty.
  4. Compare against ground-truth joints (pre-transformed to the rectified camera
     frame inside Human36mDatasetEmulatedRGBD.__getitem__).

Evaluation output: MPJPE (mm) and uncertainty coverage statistics.
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from human_pose_pipeline.utils.eval_utils import (
    evaluate_pose_prediction_scores_np,
    evaluate_uncertainty_coverage_with_covariance,
    print_coverage_stats,
    print_mpjpe_results,
    save_coverage_stats,
    save_mpjpe_results,
)
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
)
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d_from_rgbd,
)
from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
    OOD_THRESHOLD,
)
from src.ood_scores.lm_lanczos import load_score_functions
from src.datasets.h36m import Human36mDatasetEmulatedRGBD
from human_pose_pipeline.pose_estimation.triangulation_helper import load_camera_parameters


def main():
    parser = argparse.ArgumentParser(
        description='3D Pose Estimation Evaluation — Emulated RGB-D'
    )
    parser.add_argument('--data_path', type=str, default='datasets/',
                        help='Root path containing H36M/extracted/')
    parser.add_argument('--model_save_path', type=str,
                        default='human_pose_pipeline/models/pose_estimation',
                        help='Path to saved models directory')
    parser.add_argument('--run_name', type=str,
                        default='finetuned_h36m_regressflow_with_unc',
                        help='Model checkpoint run name')
    parser.add_argument('--cache_dir', type=str, default='cache/',
                        help='Cache directory for OOD score functions')
    parser.add_argument('--base_key', type=str, default=None,
                        help='Cache key for OOD score functions')
    parser.add_argument('--split', type=str, default='validation',
                        help='Dataset split: train / validation / test')
    parser.add_argument('--camera_ids', type=str, nargs='+',
                        default=['55011271', '60457274', '54138969', '58860488'],
                        help='Primary camera IDs to evaluate (all four by default)')
    parser.add_argument('--num_frames_per_video', type=int, default=None,
                        help='Number of frames sampled per action video (default: all frames)')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum number of (subject, action, camera) sequences to load '
                             '(default: all); useful for quick debug runs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of frames per inference batch')
    parser.add_argument('--sgbm_num_disparities', type=int, default=128,
                        help='StereoSGBM numDisparities (multiple of 16)')
    parser.add_argument('--sgbm_block_size', type=int, default=11,
                        help='StereoSGBM blockSize (odd number)')
    parser.add_argument('--depth_uncertainty', type=float, default=0.05,
                        help='Assumed depth std-dev in metres for uncertainty propagation')
    parser.add_argument('--enable_ood', action='store_true',
                        help='Enable OOD detection; requires --base_key')
    parser.add_argument('--ood_threshold', type=float, default=OOD_THRESHOLD)
    parser.add_argument('--output_dir', type=str, default='results/pose_3d_rgbd',
                        help='Directory for result files')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("=" * 60)
    print("3D Pose Estimation Evaluation — Emulated RGB-D")
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
    # Build emulated-RGBD dataset
    # ------------------------------------------------------------------
    print("\nBuilding emulated RGB-D dataset...")
    dataset = Human36mDatasetEmulatedRGBD(
        base_directory=base_directory,
        split=args.split,
        camera_ids=args.camera_ids,
        num_frames_per_video=args.num_frames_per_video,
        max_sequences=args.max_sequences,
        camera_params_path=camera_params_path,
        sgbm_num_disparities=args.sgbm_num_disparities,
        sgbm_block_size=args.sgbm_block_size,
    )

    if len(dataset) == 0:
        print("No samples found — check data path and camera IDs.")
        return

    # ------------------------------------------------------------------
    # Inference loop (manual batching; StereoSGBM is not multiprocess-safe)
    # ------------------------------------------------------------------
    all_3d_points_list = []
    all_3d_cov_list = []
    all_gt_list = []
    all_ood_scores_list = []
    all_is_ood_list = []
    all_batch_sizes = []

    # Cache (subject, cam_id) → camera_intrinsics dict to avoid repeated JSON reads.
    _intrinsics_cache: dict = {}

    n = len(dataset)
    for start in tqdm(range(0, n, args.batch_size), desc="Processing frames"):
        end = min(start + args.batch_size, n)

        rgb_batch, depth_batch, gt_batch = [], [], []
        R_batch, t_batch = [], []
        for i in range(start, end):
            sample = dataset[i]
            rgb_batch.append(sample['rgb_raw'])                  # (H, W, 3) uint8
            depth_batch.append(sample['depth_raw'])              # (H, W) float32 mm
            gt_batch.append(sample['gt_pose'].numpy())           # (13, 3) m, world
            R_batch.append(sample['R_rect_to_world'].numpy())    # (3, 3)
            t_batch.append(sample['t_rect_to_world'].numpy())    # (3,) m

        # Load camera intrinsics for the primary camera of this batch.
        # For clean per-camera evaluation, set --camera_ids to a single camera.
        first_meta = dataset.data[start]
        subject = first_meta['subject']
        primary_cam = first_meta['primary_cam']
        cache_key = (subject, primary_cam)
        if cache_key not in _intrinsics_cache:
            intrinsics, _, _ = load_camera_parameters(
                camera_params_path, subject, [primary_cam]
            )
            K = intrinsics[primary_cam]
            _intrinsics_cache[cache_key] = {
                'fx': float(K[0, 0]),
                'fy': float(K[1, 1]),
                'cx': float(K[0, 2]),
                'cy': float(K[1, 2]),
            }
        camera_intrinsics = _intrinsics_cache[cache_key]

        points_3d, C_3d_all, ood_score, is_ood, _, _, _, _ = \
            process_frame_3d_from_rgbd(
                rgb_frames=rgb_batch,
                depth_frames=depth_batch,
                camera_intrinsics=camera_intrinsics,
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
                depth_uncertainty=args.depth_uncertainty,
                R_rect_to_world=np.stack(R_batch),
                t_rect_to_world=np.stack(t_batch),
            )

        B = points_3d.shape[0]
        # ood_score is [B]; is_ood returned as a scalar bool for only the first
        # frame, so derive per-frame flags from ood_score directly.
        is_ood_batch = (ood_score > args.ood_threshold).cpu().numpy()

        all_batch_sizes.append(B)
        all_3d_points_list.append(points_3d.cpu().numpy())
        all_3d_cov_list.append(C_3d_all.cpu().numpy())
        all_gt_list.append(np.stack(gt_batch[:B]))
        all_ood_scores_list.append(ood_score.cpu().numpy())
        all_is_ood_list.append(is_ood_batch)

        del points_3d, C_3d_all, ood_score, is_ood_batch

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    if not all_batch_sizes:
        print("No frames processed successfully.")
        return

    num_frames = sum(all_batch_sizes)
    print(f"\n3D pose estimation completed. Frames processed: {num_frames}")

    all_3d_points = np.concatenate(all_3d_points_list, axis=0)   # (N, 13, 3) m
    all_3d_cov = np.concatenate(all_3d_cov_list, axis=0)         # (N, 13, 3, 3)
    all_gt_points = np.concatenate(all_gt_list, axis=0)          # (N, 13, 3) m
    all_ood_scores = np.concatenate(all_ood_scores_list, axis=0)
    all_is_ood = np.concatenate(all_is_ood_list, axis=0)

    if args.enable_ood and score_fn is not None:
        print("\nOOD Detection Statistics:")
        print(f"  Mean score: {all_ood_scores.mean():.4f}  Std: {all_ood_scores.std():.4f}")
        print(f"  Classified OOD: {all_is_ood.sum()} / {len(all_is_ood)} "
              f"({100 * all_is_ood.mean():.1f}%)")

    good = ~all_is_ood.astype(bool)
    all_3d_points = all_3d_points[good]
    all_3d_cov = all_3d_cov[good]
    all_gt_points = all_gt_points[good]
    num_frames = all_3d_points.shape[0]

    # Reshape to [1, T, J, 3] as expected by eval utilities
    preds = all_3d_points.reshape(1, num_frames, 13, 3)
    gts = all_gt_points.reshape(1, num_frames, 13, 3)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    mpjpe, _, per_time_errors, _, per_joint_errors, _ = \
        evaluate_pose_prediction_scores_np(predictions=preds, targets=gts)

    coverage_stats, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_poses=preds,
        true_poses=gts,
        cov_matrices=all_3d_cov,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print_mpjpe_results(mpjpe, per_time_errors, per_joint_errors, print_per_time_errors=False)
    save_mpjpe_results(mpjpe, per_time_errors, per_joint_errors,
                       split=args.split, output_dir=args.output_dir)
    print_coverage_stats(coverage_stats, print_per_time_stats=False)
    save_coverage_stats(coverage_stats, split=args.split, output_dir=args.output_dir)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
