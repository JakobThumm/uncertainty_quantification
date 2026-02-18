#!/usr/bin/env python3
"""
3D Pose Estimation from Emulated RGB-D — Full Evaluation (YOLO)

Uses Human36mDatasetGTPoseRGBD to obtain a depth map for each camera by
stereo-matching with its paired neighbour camera, then lifts YOLO 2D pose
predictions to 3D with process_frame_3d_from_rgbd_yolo (depth lifting +
uncertainty propagation).

Pipeline per frame
  1. Load primary camera frame + paired frame → rectified RGB + depth map.
  2. Run YOLO to get 2D pose keypoints.
  3. Lift to 3D via depth and propagate uncertainty.
  4. Transform from rectified camera frame to world frame.
  5. Compare against ground-truth joints (pre-transformed to the rectified
     camera frame inside Human36mDatasetGTPoseRGBD.__getitem__).

Evaluation output: MPJPE (mm) and uncertainty coverage statistics.
"""

import os
import sys
import argparse
from time import time
import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO

from human_pose_pipeline.utils.eval_utils import (
    evaluate_pose_prediction_scores_np,
    evaluate_uncertainty_coverage_with_covariance,
    print_coverage_stats,
    print_mpjpe_results,
    save_coverage_stats,
    save_mpjpe_results,
)
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d_from_rgbd_yolo,
    reset_yolo_tracking,
)
from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
)
from src.datasets.h36m import Human36mDatasetGTPoseRGBD, Human36mDatasetEmulatedRGBD
from human_pose_pipeline.pose_estimation.triangulation_helper import load_camera_parameters


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def apply_world_transform(points_3d, C_3d_all, R_batch, t_batch, device):
    """
    Rotate points and covariances from the rectified camera frame to the world
    frame and convert units from metres to millimetres.

    Args:
        points_3d: [B, N_joints, 3] in metres (camera frame)
        C_3d_all:  [B, N_joints, 3, 3] in m² (camera frame)
        R_batch:   (B, 3, 3) numpy array — rotation rect→world
        t_batch:   (B, 3) numpy array — translation rect→world in metres
        device:    torch device

    Returns:
        points_3d: [B, N_joints, 3] in mm (world frame)
        C_3d_all:  [B, N_joints, 3, 3] in mm² (world frame)
    """
    R = torch.tensor(R_batch, dtype=torch.float32, device=device)  # (B, 3, 3)
    t = torch.tensor(t_batch, dtype=torch.float32, device=device)  # (B, 3)

    # X_world[b, k] = R[b] @ X_cam[b, k] + t[b]
    points_3d = torch.einsum('bij,bkj->bki', R, points_3d)  # (B, J, 3)
    points_3d = points_3d + t.unsqueeze(1)                  # broadcast over joints

    # C_world[b, k] = R[b] @ C_cam[b, k] @ R[b]^T
    R_exp = R.unsqueeze(1)                                   # (B, 1, 3, 3)
    C_3d_all = R_exp @ C_3d_all @ R_exp.transpose(-1, -2)

    # Convert metres → millimetres
    points_3d = points_3d * 1000.0
    C_3d_all  = C_3d_all  * 1e6

    return points_3d, C_3d_all


def main():
    parser = argparse.ArgumentParser(
        description='3D Pose Estimation Evaluation — Emulated RGB-D (YOLO)'
    )
    parser.add_argument('--data_path', type=str, default='datasets/',
                        help='Root path containing H36M/extracted/')
    parser.add_argument('--model_save_path', type=str,
                        default='human_pose_pipeline/models/pose_estimation',
                        help='Path to saved models directory (for camera parameters)')
    parser.add_argument('--camera_params_path', type=str, default=None,
                        help='Direct path to camera-parameters.json '
                             '(overrides --model_save_path)')
    parser.add_argument('--yolo_model', type=str, default='yolo26n-pose.pt',
                        help='YOLO pose model to use '
                             '(e.g. yolo11n/s/m/l/x-pose.pt)')
    parser.add_argument('--split', type=str, default='validation',
                        help='Dataset split: train / validation / test')
    parser.add_argument('--camera_ids', type=str, nargs='+',
                        default=['55011271', '60457274', '54138969', '58860488'],
                        help='Primary camera IDs to evaluate (all four by default)')
    parser.add_argument('--num_frames_per_video', type=int, default=None,
                        help='Number of frames sampled per action video (default: all)')
    parser.add_argument('--max_sequences', type=int, default=None,
                        help='Maximum number of (subject, action, camera) sequences '
                             'to load (default: all); useful for quick debug runs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of frames per inference batch')
    parser.add_argument('--depth_mode', type=str, default='stereo',
                        choices=['gt', 'stereo'],
                        help='Depth source: "gt" uses ground-truth pose to paint depth '
                             '(Human36mDatasetGTPoseRGBD), "stereo" runs StereoSGBM '
                             '(Human36mDatasetEmulatedRGBD)')
    parser.add_argument('--sgbm_num_disparities', type=int, default=128,
                        help='[stereo] StereoSGBM numDisparities (multiple of 16)')
    parser.add_argument('--sgbm_block_size', type=int, default=11,
                        help='[stereo] StereoSGBM blockSize (odd number)')
    parser.add_argument('--depth_uncertainty', type=float, default=0.05,
                        help='Assumed depth std-dev in metres for uncertainty propagation')
    parser.add_argument('--enable_tracking', action='store_true',
                        help='Enable YOLO tracking (resets per sequence)')
    parser.add_argument('--confidence_threshold', type=float,
                        default=YOLO_CONFIDENCE_THRESHOLD,
                        help='YOLO keypoint confidence threshold')
    parser.add_argument('--output_dir', type=str,
                        default='results/pose_3d_rgbd_yolo',
                        help='Directory for result files')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print("=" * 60)
    print("3D Pose Estimation Evaluation — Emulated RGB-D (YOLO)")
    print(f"Model: {args.yolo_model}")
    if args.enable_tracking:
        print("WITH TRACKING (resets per sequence)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Camera parameters path
    # ------------------------------------------------------------------
    if args.camera_params_path is not None:
        camera_params_path = args.camera_params_path
    else:
        models_dir = os.path.join(
            root_dir, args.model_save_path, 'H36M', 'RegressFlow', 'seed_420'
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
        print(f"Model loaded on CUDA (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = 'cpu'
        print("Model loaded on CPU")
    print("YOLO model initialised.")

    # ------------------------------------------------------------------
    # Build emulated-RGBD dataset
    # ------------------------------------------------------------------
    base_directory = os.path.join(root_dir, args.data_path, 'H36M', 'extracted')
    print(f"\nBuilding RGB-D dataset (depth_mode={args.depth_mode})...")
    if args.depth_mode == 'gt':
        dataset = Human36mDatasetGTPoseRGBD(
            base_directory=base_directory,
            split=args.split,
            camera_ids=args.camera_ids,
            num_frames_per_video=args.num_frames_per_video,
            max_sequences=args.max_sequences,
            camera_params_path=camera_params_path,
            depth_radius_px=20,
            far_depth_m=20.0,
        )
    else:  # stereo
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
    # Inference loop
    # ------------------------------------------------------------------
    all_3d_points_list = []
    all_3d_cov_list = []
    all_gt_list = []
    all_batch_sizes = []

    # Cache (subject, cam_id) → camera_intrinsics to avoid repeated JSON reads.
    _intrinsics_cache: dict = {}

    for seq_idx in tqdm(range(len(dataset)), desc="Processing sequences"):
        t1 = time()
        sample = dataset[seq_idx]
        t2 = time()
        print(f"Time for sequence loading: {t2 - t1:.3f}s  "
              f"({len(sample['rgb_raw'])} frames)")

        seq_meta = dataset.data[seq_idx]
        subject = seq_meta['subject']
        primary_cam = seq_meta.get('cam_id') or seq_meta.get('primary_cam')

        # Camera intrinsics are constant across all frames in this sequence.
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

        rgb_seq = sample['rgb_raw']      # list[T] of (H, W, 3) uint8
        depth_seq = sample['depth_raw']    # list[T] of (H, W) float32
        gt_seq = sample['gt_pose']      # (T, 13, 3) tensor, world frame m
        R = sample['R_rect_to_world'].numpy()   # (3, 3)
        t = sample['t_rect_to_world'].numpy()   # (3,)

        T = len(rgb_seq)

        # Reset tracking at the start of each new sequence.
        if args.enable_tracking:
            reset_yolo_tracking(yolo_model)

        for start in range(0, T, args.batch_size):
            end = min(start + args.batch_size, T)
            B_req = end - start

            rgb_batch = rgb_seq[start:end]
            depth_batch = depth_seq[start:end]
            gt_batch = gt_seq[start:end].numpy()   # (B_req, 13, 3) m
            R_batch = np.stack([R] * B_req)       # (B_req, 3, 3)
            t_batch = np.stack([t] * B_req)       # (B_req, 3)

            t3 = time()
            points_3d, C_3d_all, _, is_ood, human_detected, _, _, _ = \
                process_frame_3d_from_rgbd_yolo(
                    rgb_frames=rgb_batch,
                    depth_frames=depth_batch,
                    camera_intrinsics=camera_intrinsics,
                    yolo_pose_model=yolo_model,
                    mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                    enable_tracking=args.enable_tracking,
                    confidence_threshold=args.confidence_threshold,
                    verbose=False,
                    device=device,
                    depth_uncertainty=args.depth_uncertainty,
                )
            t4 = time()
            print(f"Time for batch processing: {t4 - t3:.3f}s")

            # Transform from rectified camera frame to world frame (metres → mm).
            points_3d, C_3d_all = apply_world_transform(
                points_3d, C_3d_all, R_batch, t_batch, device=device
            )

            B = points_3d.shape[0]
            good = (~is_ood & human_detected).cpu().numpy()  # [B] bool
            pts = points_3d.cpu().numpy()[good]
            cov = C_3d_all.cpu().numpy()[good]
            gt = gt_batch[:B][good]
            if pts.shape[0] > 0:
                all_batch_sizes.append(pts.shape[0])
                all_3d_points_list.append(pts)
                all_3d_cov_list.append(cov)
                all_gt_list.append(gt)

            del points_3d, C_3d_all

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    if not all_batch_sizes:
        print("No frames processed successfully.")
        return

    num_frames = sum(all_batch_sizes)
    print(f"\n3D pose estimation completed. Frames processed: {num_frames}")

    all_3d_points = np.concatenate(all_3d_points_list, axis=0)   # (N, 13, 3) mm
    all_3d_cov = np.concatenate(all_3d_cov_list, axis=0)      # (N, 13, 3, 3) mm²
    all_gt_points = np.concatenate(all_gt_list, axis=0)          # (N, 13, 3) m

    num_frames = all_3d_points.shape[0]
    print(f"Valid frames for evaluation: {num_frames}")

    # Reshape to [1, T, J, 3] as expected by eval utilities.
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

    model_name = args.yolo_model.replace('.pt', '')
    tracking_suffix = "_tracking" if args.enable_tracking else "_notracking"
    output_dir = os.path.join(args.output_dir, f"{model_name}{tracking_suffix}")
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
