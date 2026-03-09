#!/usr/bin/env python3
"""
Debug RGBD 3D Pose Visualization Script - JAX Implementation

This script provides a 3D pose estimation pipeline for RGBD data:
- Loads RGB and depth images from the HumanRGBD dataset
- Detects humans using YOLO
- Performs 2D pose estimation using JAX RegressFlow model with uncertainty
- Lifts 2D poses to 3D using depth information with uncertainty propagation
- Visualizes the results with 2D overlay and 3D pose view

Based on debug_3d_pose_visualization.py but adapted for RGBD data (depth lifting instead of triangulation).
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
)
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d_from_rgbd
)
from human_pose_pipeline.utils.visualization import (
    visualize_poses_matplotlib,
    draw_3d_pose_with_covariance
)
from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_NAMES_13,
    CONNECTIONS_13,
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
    OOD_THRESHOLD
)
from src.datasets.human_rgbd import HumanRGBDDataset, DEFAULT_CAMERA_INTRINSICS


def print_3d_pose_summary(points_3d, covariances_3d, valid_depth):
    """Print detailed summary of 3D pose estimation."""
    print("\n" + "=" * 60)
    print("3D POSE ESTIMATION SUMMARY")
    print("=" * 60)

    # Convert to numpy if needed
    if isinstance(points_3d, torch.Tensor):
        points_3d = points_3d.cpu().numpy()
    if isinstance(covariances_3d, torch.Tensor):
        covariances_3d = covariances_3d.cpu().numpy()
    if isinstance(valid_depth, torch.Tensor):
        valid_depth = valid_depth.cpu().numpy()

    print(f"3D Pose Shape: {points_3d.shape}")
    print(f"Covariances Shape: {covariances_3d.shape}")
    print(f"Valid joints: {np.sum(valid_depth)} / {len(valid_depth)}")

    # Per-joint 3D positions
    print(f"\nPer-Joint 3D Positions (meters):")
    print("-" * 60)
    for i, joint_name in enumerate(JOINT_NAMES_13):
        x, y, z = points_3d[i]
        valid = "Y" if valid_depth[i] else "N"
        # Compute uncertainty magnitude from covariance
        cov = covariances_3d[i]
        eigenvals = np.linalg.eigvals(cov)
        unc_mag = np.sqrt(np.mean(np.abs(eigenvals)))
        print(f"{i:2d}. {joint_name:15s}: ({x:7.3f}, {y:7.3f}, {z:7.3f}) unc={unc_mag:.4f} valid={valid}")

    # Coordinate range analysis
    valid_mask = valid_depth.astype(bool)
    if np.any(valid_mask):
        valid_points = points_3d[valid_mask]
        print(f"\n3D Coordinate Statistics (valid joints only):")
        print(f"  X range: [{np.min(valid_points[:, 0]):.3f}, {np.max(valid_points[:, 0]):.3f}] m")
        print(f"  Y range: [{np.min(valid_points[:, 1]):.3f}, {np.max(valid_points[:, 1]):.3f}] m")
        print(f"  Z range: [{np.min(valid_points[:, 2]):.3f}, {np.max(valid_points[:, 2]):.3f}] m")

    # Uncertainty statistics
    print(f"\nUncertainty Statistics:")
    uncertainty_mags = []
    for i, cov in enumerate(covariances_3d):
        if valid_depth[i]:
            eigenvals = np.linalg.eigvals(cov)
            uncertainty_mag = np.sqrt(np.mean(np.abs(eigenvals)))
            uncertainty_mags.append(uncertainty_mag)

    if uncertainty_mags:
        uncertainty_mags = np.array(uncertainty_mags)
        print(f"  Mean uncertainty magnitude: {np.mean(uncertainty_mags):.4f} m")
        print(f"  Min uncertainty magnitude:  {np.min(uncertainty_mags):.4f} m")
        print(f"  Max uncertainty magnitude:  {np.max(uncertainty_mags):.4f} m")

    print("=" * 60)


def main():
    """Main debug function for 3D pose estimation on RGBD data."""
    parser = argparse.ArgumentParser(description="Debug 3D pose estimation on RGBD data")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index to process")
    parser.add_argument("--data_path", type=str, default=None, help="Path to RGBD dataset")
    parser.add_argument("--save_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference (cuda/cpu)")
    parser.add_argument("--depth_uncertainty", type=float, default=0.01, help="Depth uncertainty in meters")
    args = parser.parse_args()

    print("=" * 60)
    print("Debug RGBD 3D Pose Visualization - JAX Implementation")
    print("=" * 60)

    try:
        # Initialize models
        print("Initializing models...")

        # Use full RegressFlow model with uncertainty
        models_dir = os.path.join(root_dir, "human_pose_pipeline/models/pose_estimation", "H36M", "RegressFlow", "seed_420")
        checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_with_unc")
        print("Using RegressFlowWithAleatoric model for uncertainty estimation")

        pose_estimation_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path_jax)

        # Initialize YOLO human detector
        human_detector, device_torch = initialize_human_detector(args.device)

        print("Models initialized successfully!")

        # Load RGBD dataset
        print("\nLoading RGBD dataset...")
        data_path = args.data_path or os.path.join(root_dir, "datasets", "rgbd_test")

        dataset = HumanRGBDDataset(
            base_directory=data_path,
        )

        if args.frame_idx >= len(dataset):
            print(f"Frame index {args.frame_idx} exceeds dataset length {len(dataset)}, using frame 0")
            args.frame_idx = 0

        # Get sample
        sample = dataset[args.frame_idx]
        print(f"Loaded sample: {sample['filename']}")
        print(f"  - Color shape: {sample['color_raw'].shape}")
        print(f"  - Depth shape: {sample['depth_raw'].shape}")

        # Get camera intrinsics from sample
        camera_intrinsics = sample['camera_intrinsics']
        print(f"  - Camera intrinsics: fx={camera_intrinsics['fx']:.2f}, fy={camera_intrinsics['fy']:.2f}, "
              f"cx={camera_intrinsics['cx']:.2f}, cy={camera_intrinsics['cy']:.2f}")

        # Prepare input frames as lists (expected format for process_frame_3d_from_rgbd)
        rgb_frames = [sample['color_raw']]  # List of numpy arrays [H, W, C]
        depth_frames = [sample['depth_raw']]  # List of numpy arrays [H, W] in mm

        # Run 3D pose estimation from RGBD
        print("\nRunning 3D pose estimation from RGBD...")

        points_3d, C_3d_all, ood_score, is_ood, human_detected, keypoints_2d, uncertainties_2d, covariance_xy = \
            process_frame_3d_from_rgbd(
                rgb_frames=rgb_frames,
                depth_frames=depth_frames,
                camera_intrinsics=camera_intrinsics,
                pose_estimation_jit_fn=pose_estimation_jit_fn,
                params=params,
                batch_stats=batch_stats,
                human_detector=human_detector,
                device_torch=device_torch,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                score_fn=None,  # No OOD scoring
                human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
                ood_threshold=OOD_THRESHOLD,
                num_output_joints=17,
                verbose=True,
                device=args.device,
                depth_uncertainty=args.depth_uncertainty,
                R_rect_to_world=sample["R_rect_to_world"],
                t_rect_to_world=sample["t_rect_to_world"]
            )
        human_detected = human_detected.cpu()[0]
        is_ood = is_ood.cpu()[0]
        if not human_detected:
            print("No human detected in frame!")
            return

        print(f"3D pose estimation completed!")
        print(f"  - Human detected: {human_detected}")
        print(f"  - OOD score: {ood_score}")
        print(f"  - Is OOD: {is_ood}")

        # Remove batch dimension for single frame
        points_3d = points_3d[0]
        C_3d_all = C_3d_all[0]
        keypoints_2d = keypoints_2d[0]
        uncertainties_2d = uncertainties_2d[0]
        covariance_xy = covariance_xy[0]

        # Determine valid depth joints (non-zero 3D positions)
        if isinstance(points_3d, torch.Tensor):
            valid_depth = (torch.abs(points_3d).sum(dim=-1) > 0).cpu().numpy()
        else:
            valid_depth = np.abs(points_3d).sum(axis=-1) > 0

        # Print summary
        print_3d_pose_summary(points_3d, C_3d_all, valid_depth)

        # Create visualizations
        print("\nCreating visualizations...")
        os.makedirs(args.save_dir, exist_ok=True)

        # Convert tensors to numpy for visualization
        if isinstance(keypoints_2d, torch.Tensor):
            keypoints_2d_np = keypoints_2d.cpu().numpy()
            uncertainties_2d_np = uncertainties_2d.cpu().numpy()
            covariance_xy_np = covariance_xy.cpu().numpy()
            points_3d_np = points_3d.cpu().numpy()
            C_3d_all_np = C_3d_all.cpu().numpy()
        else:
            keypoints_2d_np = np.array(keypoints_2d)
            uncertainties_2d_np = np.array(uncertainties_2d)
            covariance_xy_np = np.array(covariance_xy)
            points_3d_np = np.array(points_3d)
            C_3d_all_np = np.array(C_3d_all)

        # 1. Save 2D pose visualization with uncertainty
        image_pil = Image.fromarray(sample['color_raw'])
        save_path_2d = os.path.join(args.save_dir, f"debug_rgbd_3d_2d_overlay_frame_{args.frame_idx}.png")
        visualize_poses_matplotlib(
            image=image_pil,
            gt_pose=None,  # No ground truth
            pred_pose=keypoints_2d_np,
            pred_uncertainties=uncertainties_2d_np,
            pred_covariances=covariance_xy_np,
            save_path=save_path_2d,
            show_uncertainty=True
        )
        print(f"2D visualization saved: {save_path_2d}")

        # 2. Create comprehensive figure with 2D and 3D views
        fig = plt.figure(figsize=(18, 8))

        # 2D RGB image with pose overlay
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(sample['color_raw'])
        ax1.set_title('RGB Image')
        ax1.axis('off')

        # 2D Depth image visualization
        ax2 = fig.add_subplot(1, 3, 2)
        depth_vis = sample['depth_raw'].astype(np.float32)
        depth_vis[depth_vis == 0] = np.nan  # Mark invalid depth as NaN for visualization
        im = ax2.imshow(depth_vis, cmap='viridis')
        plt.colorbar(im, ax=ax2, label='Depth (mm)')
        ax2.set_title('Depth Image')
        ax2.axis('off')

        # 3D pose visualization
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        draw_3d_pose_with_covariance(ax3, points_3d_np, C_3d_all_np, CONNECTIONS_13)
        ax3.set_title('3D Pose with Uncertainty')

        plt.tight_layout()

        # Save comprehensive visualization
        save_path_comprehensive = os.path.join(args.save_dir, f"debug_rgbd_3d_comprehensive_frame_{args.frame_idx}.png")
        plt.savefig(save_path_comprehensive, dpi=150, bbox_inches='tight')
        print(f"Comprehensive visualization saved: {save_path_comprehensive}")

        # 3. Save standalone 3D pose visualization
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        draw_3d_pose_with_covariance(ax_3d, points_3d_np, C_3d_all_np, CONNECTIONS_13)
        ax_3d.set_title(f'3D Pose from RGBD - Frame {args.frame_idx}')

        save_path_3d = os.path.join(args.save_dir, f"debug_rgbd_3d_pose_frame_{args.frame_idx}.png")
        plt.savefig(save_path_3d, dpi=150, bbox_inches='tight')
        print(f"3D pose visualization saved: {save_path_3d}")

        plt.show()

        print("\n" + "=" * 60)
        print("DEBUG 3D POSE ESTIMATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Frame processed: {args.frame_idx}")
        print(f"Filename: {sample['filename']}")
        print(f"Human detected: {human_detected}")
        print(f"Valid depth joints: {np.sum(valid_depth)} / 13")

    except Exception as e:
        print(f"\nError during debug 3D pose estimation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
