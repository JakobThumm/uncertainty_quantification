#!/usr/bin/env python3
"""
Debug script: visualize motion prediction from a saved pipeline pickle.

For a given index `idx` the script shows:
  - poses_3d_estimated[idx : idx + INPUT_HORIZON_LENGTH]   (input history)
  - motions_predicted[idx, :PREDICTION_HORIZON_LENGTH]     (predicted future)
  - motions_gt[idx]                                        (GT future, if present)

Usage example:
    python human_pose_pipeline/examples/debug_visualize_motion_prediction.py \
        --pickle_file results/eval_full_pipeline/full_pipeline_results.cloudpickle \
        --idx 100 \
        --input_horizon 50 \
        --output_path results/debug/motion_viz_100.mp4
"""

import argparse
import os
import cloudpickle
import numpy as np

from human_pose_pipeline.utils.visualization import (
    render_motion_prediction_video,
    CONNECTIONS_13,
)


def main():
    parser = argparse.ArgumentParser(description='Visualize motion prediction from pipeline pickle')
    parser.add_argument('--pickle_file', type=str, required=True,
                        help='Path to full_pipeline_results.cloudpickle')
    parser.add_argument('--idx', type=int, default=0,
                        help='Index into poses_3d_estimated / motions_predicted')
    parser.add_argument('--input_horizon', type=int, default=50,
                        help='Number of input poses to show (INPUT_HORIZON_LENGTH)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output MP4 path. Defaults to <pickle_dir>/motion_viz_<idx>.mp4')
    parser.add_argument('--fps', type=int, default=10,
                        help='Video frames per second')
    parser.add_argument('--no_gt', action='store_true',
                        help='Skip ground-truth future poses even if present in pickle')
    parser.add_argument('--no_cov', action='store_true',
                        help='Skip covariance uncertainty visualisation')
    parser.add_argument('--cov_scale', type=float, default=3.0,
                        help='Principal-axis length = cov_scale * std-dev')
    parser.add_argument('--elev', type=float, default=20.0,
                        help='3D view elevation angle in degrees')
    parser.add_argument('--azim', type=float, default=45.0,
                        help='3D view azimuth angle in degrees')
    args = parser.parse_args()

    # --- Load pickle ---
    print(f"Loading {args.pickle_file} ...")
    with open(args.pickle_file, 'rb') as f:
        data = cloudpickle.load(f)

    poses_3d_estimated = data['poses_3d_estimated']       # [N, J, 3]
    motions_predicted = data['motions_predicted']          # [M, T_pred, J, 3]
    motions_gt = data.get('motions_gt')                    # [M, T_pred, J, 3] or None
    motions_cov_predicted = data.get('motions_cov_predicted')  # [M, T_pred, J, 3, 3] or None

    N = poses_3d_estimated.shape[0]
    M = motions_predicted.shape[0]
    T_pred = motions_predicted.shape[1]
    idx = args.idx
    T_in = args.input_horizon

    print(f"poses_3d_estimated shape : {poses_3d_estimated.shape}")
    print(f"motions_predicted shape  : {motions_predicted.shape}")
    if motions_gt is not None:
        print(f"motions_gt shape         : {np.array(motions_gt).shape}")

    # --- Validate index ---
    if not (0 <= idx < min(N - T_in + 1, M)):
        raise ValueError(
            f"idx={idx} out of valid range [0, {min(N - T_in + 1, M) - 1}]. "
            f"Need idx+input_horizon <= N ({N}) and idx < M ({M})."
        )

    # --- Slice data ---
    input_poses = poses_3d_estimated[idx: idx + T_in]          # [T_in, J, 3]
    predicted_poses = motions_predicted[idx, :T_pred]           # [T_pred, J, 3]

    gt_poses = None
    if motions_gt is not None and not args.no_gt:
        gt_arr = np.array(motions_gt)
        gt_poses = gt_arr[idx, :T_pred]                         # [T_pred, J, 3]

    predicted_covs = None
    if motions_cov_predicted is not None and not args.no_cov:
        predicted_covs = motions_cov_predicted[idx, :T_pred]    # [T_pred, J, 3, 3]

    print(f"\nVisualising idx={idx}:")
    print(f"  Input poses  : [{idx} : {idx + T_in}]  → shape {input_poses.shape}")
    print(f"  Predictions  : {T_pred} steps → shape {predicted_poses.shape}")
    if gt_poses is not None:
        print(f"  GT future    : {T_pred} steps → shape {gt_poses.shape}")
    if predicted_covs is not None:
        print(f"  Covariances  : shape {predicted_covs.shape}")

    # --- Output path ---
    if args.output_path is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.pickle_file)), 'debug_viz')
        output_path = os.path.join(out_dir, f'motion_viz_{idx}.mp4')
    else:
        output_path = args.output_path

    # --- Render ---
    render_motion_prediction_video(
        input_poses=input_poses,
        predicted_poses=predicted_poses,
        connections=CONNECTIONS_13,
        output_path=output_path,
        gt_poses=gt_poses,
        predicted_covs=predicted_covs,
        fps=args.fps,
        elev=args.elev,
        azim=args.azim,
        cov_scale=args.cov_scale,
    )


if __name__ == '__main__':
    main()
