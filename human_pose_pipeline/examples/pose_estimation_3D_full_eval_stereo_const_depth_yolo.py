#!/usr/bin/env python3
"""
3D Pose Estimation - Stereo + Constant Depth Uncertainty (YOLO)

Variant of pose_estimation_3D_full_eval_yolo.py that replaces the
propagated depth covariance from stereo triangulation with a user-specified
constant uncertainty.  This is useful for the H36M wide-baseline setup where
the lateral (X, Y) position from triangulation is reliable but the stereo
depth covariance estimate is poor.

Pipeline per frame
  1. Run YOLO on both cameras and triangulate to get 3D pose.
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

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ultralytics import YOLO

from human_pose_pipeline.utils.eval_utils import (
    evaluate_pose_prediction_scores_np,
    evaluate_uncertainty_coverage_with_covariance,
    print_coverage_stats,
    print_mpjpe_results,
    save_coverage_stats,
    save_mpjpe_results,
)
from src.datasets.h36m import Human36mDatasetTwoCameras
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d_yolo,
    reset_yolo_tracking,
    set_depth_uncertainty_to_constant,
)
from human_pose_pipeline.pose_estimation.triangulation_helper import load_camera_parameters
from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
)


def main():
    parser = argparse.ArgumentParser(
        description='3D Pose Estimation — Stereo + Constant Depth Uncertainty (YOLO)'
    )
    parser.add_argument('--data_path', type=str, default='datasets/',
                        help='Root path containing H36M/extracted/')
    parser.add_argument('--yolo_model', type=str, default='yolo11n-pose.pt',
                        help='YOLO pose model (e.g. yolo11n/s/m/l/x-pose.pt)')
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
    parser.add_argument('--enable_tracking', action='store_true',
                        help='Enable YOLO tracking (resets per sequence)')
    parser.add_argument('--confidence_threshold', type=float,
                        default=YOLO_CONFIDENCE_THRESHOLD,
                        help='YOLO keypoint confidence threshold')
    parser.add_argument('--camera_params_path', type=str, default=None,
                        help='Path to camera-parameters.json')
    parser.add_argument('--output_dir', type=str,
                        default='results/pose_3d_stereo_const_depth_yolo',
                        help='Directory for result files')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("=" * 60)
    print("3D Pose Estimation — Stereo + Constant Depth Uncertainty (YOLO)")
    print(f"Model: {args.yolo_model}  sigma_depth={args.sigma_depth} mm")
    print("=" * 60)

    base_directory = os.path.join(root_dir, args.data_path, 'H36M', 'extracted')

    if args.camera_params_path is not None:
        camera_params_path = args.camera_params_path
    else:
        models_dir = os.path.join(
            root_dir,
            'human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420'
        )
        camera_params_path = os.path.join(models_dir, 'camera-parameters.json')

    if not os.path.exists(camera_params_path):
        print(f"Camera parameters not found: {camera_params_path}")
        return

    # ------------------------------------------------------------------
    # Initialise YOLO model
    # ------------------------------------------------------------------
    print("\nInitialising YOLO pose model...")
    yolo_model = YOLO(args.yolo_model)
    device = args.device
    if device == 'cuda' and torch.cuda.is_available():
        yolo_model.to('cuda')
        print(f"Model on CUDA (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = 'cpu'
        print("Model on CPU")

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
    all_batch_sizes = []

    counter = 0
    for sample in tqdm(dataset, desc="Processing sequences"):
        if args.max_sequences is not None and counter >= args.max_sequences:
            break
        counter += 1

        all_camera_frames = sample['all_camera_frames']
        pose_sequence = sample['pose_sequence']
        subject = sample['subject']

        if args.enable_tracking:
            reset_yolo_tracking(yolo_model)

        intrinsics, extrinsics, projection_matrices = load_camera_parameters(
            camera_params_path, subject, args.camera_ids
        )
        R_world_to_cam = extrinsics[args.camera_ids[0]][:3, :3]  # (3,3) numpy

        P1 = torch.from_numpy(projection_matrices[args.camera_ids[0]]).to(device)
        P2 = torch.from_numpy(projection_matrices[args.camera_ids[1]]).to(device)

        frames_to_process = min(len(all_camera_frames[0]), len(pose_sequence))

        for frame_idx in range(0, frames_to_process, args.batch_size):
            B_req = min(args.batch_size, frames_to_process - frame_idx)
            left_frames = all_camera_frames[0][frame_idx:frame_idx + B_req]
            right_frames = all_camera_frames[1][frame_idx:frame_idx + B_req]
            interleaved = [x for pair in zip(left_frames, right_frames) for x in pair]

            points_3d, C_3d_all, _, _, _, _, _, _ = process_frame_3d_yolo(
                frames=interleaved,
                projection_matrices=[P1, P2],
                yolo_pose_model=yolo_model,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                enable_tracking=args.enable_tracking,
                confidence_threshold=args.confidence_threshold,
                verbose=False,
                device=device,
            )

            C_3d_all = set_depth_uncertainty_to_constant(
                C_3d_all, R_world_to_cam, args.sigma_depth, device=device
            )

            all_batch_sizes.append(B_req)
            all_3d_points_list.append(points_3d.cpu().numpy())
            all_3d_cov_list.append(C_3d_all.cpu().numpy())
            all_gt_list.append(pose_sequence[frame_idx:frame_idx + B_req])

            del points_3d, C_3d_all

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    num_frames = sum(all_batch_sizes)
    print(f"\nProcessed {num_frames} frames from {counter} sequences")

    all_3d_points = np.zeros((num_frames, 13, 3))
    all_3d_cov = np.zeros((num_frames, 13, 3, 3))
    all_gt_points = np.zeros((num_frames, 13, 3))
    idx = 0
    for i, bs in enumerate(all_batch_sizes):
        all_3d_points[idx:idx + bs] = all_3d_points_list[i]
        all_3d_cov[idx:idx + bs] = all_3d_cov_list[i]
        all_gt_points[idx:idx + bs] = all_gt_list[i]
        idx += bs

    # Filter frames with no human detected
    valid = ~np.all(all_3d_points == 0, axis=(1, 2))
    num_invalid = np.sum(~valid)
    if num_invalid > 0:
        print(f"Filtering {num_invalid} frames with no detection "
              f"({100 * num_invalid / num_frames:.1f}%)")
    all_3d_points = all_3d_points[valid]
    all_3d_cov = all_3d_cov[valid]
    all_gt_points = all_gt_points[valid]
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

    model_name = args.yolo_model.replace('.pt', '')
    tracking_suffix = '_tracking' if args.enable_tracking else '_notracking'
    output_dir = os.path.join(
        args.output_dir, f"{model_name}{tracking_suffix}_sigma{args.sigma_depth:.1f}mm"
    )
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
