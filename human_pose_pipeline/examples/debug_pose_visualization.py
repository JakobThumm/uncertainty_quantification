#!/usr/bin/env python3
"""
Debug Pose Visualization Script

This script loads a single H36M sample, runs pose estimation, and visualizes:
1. Original image with ground truth pose
2. Original image with predicted pose
3. Side-by-side comparison
4. Detailed coordinate analysis

This helps debug coordinate system alignment and understand MPJPE values.
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

# Add root directory to path to access src
sys.path.append('../..')

from src.models.wrapper import model_from_string
from src.datasets.h36m import Human36mDataset
from human_pose_pipeline.evaluation.pose_metrics import (
    mpjpe_jax,
    JOINT_NAMES_13,
    JOINT_IDX_13_MODEL
)
from human_pose_pipeline.pose_estimation.coordinate_alignment import (
    align_pose_coordinates,
    compute_alignment_metrics
)

# Define joint connections for visualization
JOINT_CONNECTIONS_13 = [
    (0, 1), (0, 2),  # Nose to shoulders
    (1, 3), (3, 5),  # Left arm
    (2, 4), (4, 6),  # Right arm
    (1, 2), (1, 7), (2, 8),  # Shoulders to hips
    (7, 8),  # Connect hips
    (7, 9), (9, 11),  # Left leg
    (8, 10), (10, 12)  # Right leg
]

def load_regressflow_model():
    """Load the pre-trained RegressFlow model"""
    print("Loading RegressFlow model...")

    model_path = "../../models_tianle"
    dataset_name = "H36M"
    model_name = "RegressFlow"
    run_name = "finetuned_h36m_regressflow_pred"
    seed = 420

    # Load model arguments and parameters
    args_file = f"{model_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_args.json"
    with open(args_file, 'r') as f:
        args_dict = json.load(f)

    params_file = f"{model_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_params.pickle"
    with open(params_file, 'rb') as f:
        params_dict = pickle.load(f)

    model = model_from_string(
        model_name=args_dict["model"],
        output_dim=args_dict["output_dim"]
    )

    params = params_dict["params"]
    batch_stats = params_dict.get("batch_stats", None)

    print("[PASS] Model loaded successfully")
    return model, params, batch_stats

def load_single_sample(sample_idx=0):
    """Load a single H36M sample with detailed information"""
    print(f"Loading H36M sample {sample_idx}...")

    h36m_path = "../../datasets/H36M/extracted"

    # Create transform that preserves original coordinates for visualization
    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = Human36mDataset(
        base_directory=h36m_path,
        split='validation',
        num_frames_per_video=10,
        transform=transform,
        image_size=(256, 192)
    )

    # Get the sample
    sample = dataset[sample_idx]
    frame, gt_pose = sample

    # Also get the original image without preprocessing for visualization
    # We'll need to access the raw data
    raw_sample = dataset.data[sample_idx]
    video_path = raw_sample['video_path']
    frame_idx = raw_sample['frame_idx']
    original_pose = raw_sample['pose_13']

    # Load original frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, original_frame = cap.read()
    cap.release()

    if ret:
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    else:
        original_frame = None

    print(f"[PASS] Sample loaded successfully")
    print(f"  - Video: {os.path.basename(video_path)}")
    print(f"  - Frame: {frame_idx}")
    print(f"  - Processed frame shape: {frame.shape}")
    print(f"  - GT pose shape: {gt_pose.shape}")
    if original_frame is not None:
        print(f"  - Original frame shape: {original_frame.shape}")

    return {
        'processed_frame': frame,
        'original_frame': original_frame,
        'gt_pose': gt_pose,
        'original_pose': original_pose,
        'video_path': video_path,
        'frame_idx': frame_idx
    }

def predict_pose(model, params, batch_stats, frame):
    """Run pose prediction on a single frame"""
    print("Running pose prediction...")

    # Ensure correct shape for model input
    if hasattr(frame, 'numpy'):
        frame_np = frame.numpy()
    else:
        frame_np = np.array(frame)

    if frame_np.ndim == 3:
        frame_np = frame_np[None, ...]  # Add batch dimension

    frame_jax = jnp.array(frame_np, dtype=jnp.float32)

    # Forward pass
    if batch_stats is not None:
        pred_pose = model.apply_test(params, batch_stats, frame_jax)
    else:
        pred_pose = model.apply_test(params, frame_jax)

    # Extract predictions (remove batch dimension)
    pred_pose_np = np.array(pred_pose[0])  # Shape: (34,)
    pred_pose_reshaped = pred_pose_np.reshape(17, 2)  # 17 joints × 2 coords

    print(f"[PASS] Pose prediction completed")
    print(f"  - Prediction shape: {pred_pose_reshaped.shape}")
    print(f"  - Coordinate range: [{np.min(pred_pose_np):.3f}, {np.max(pred_pose_np):.3f}]")

    return pred_pose_reshaped

def analyze_coordinates(pred_pose, gt_pose, sample_info):
    """Analyze coordinate systems and compute detailed metrics with alignment"""
    print("\n" + "=" * 50)
    print("Coordinate System Analysis & Alignment")
    print("=" * 50)

    # Convert ground truth to proper shape
    if hasattr(gt_pose, 'numpy'):
        gt_pose_np = gt_pose.numpy()
    else:
        gt_pose_np = np.array(gt_pose)

    if gt_pose_np.shape[0] == 26:  # Flattened 13 joints
        gt_pose_reshaped = gt_pose_np.reshape(13, 2)
    else:
        gt_pose_reshaped = gt_pose_np

    # Map 17-joint prediction to 13-joint ground truth
    pred_pose_13 = pred_pose[JOINT_IDX_13_MODEL[:13], :]

    print(f"BEFORE alignment:")
    print(f"  Prediction coordinates (17 joints → 13 joints):")
    print(f"    - Shape: {pred_pose.shape} → {pred_pose_13.shape}")
    print(f"    - Range: [{np.min(pred_pose_13):.3f}, {np.max(pred_pose_13):.3f}]")
    print(f"    - Mean: [{np.mean(pred_pose_13[:, 0]):.3f}, {np.mean(pred_pose_13[:, 1]):.3f}]")

    print(f"  Ground truth coordinates:")
    print(f"    - Shape: {gt_pose_reshaped.shape}")
    print(f"    - Range: [{np.min(gt_pose_reshaped):.3f}, {np.max(gt_pose_reshaped):.3f}]")
    print(f"    - Mean: [{np.mean(gt_pose_reshaped[:, 0]):.3f}, {np.mean(gt_pose_reshaped[:, 1]):.3f}]")

    # Compute MPJPE before alignment
    mpjpe_before = mpjpe_jax(jnp.array(pred_pose_13[None, ...]), jnp.array(gt_pose_reshaped[None, ...]))
    print(f"  MPJPE (before alignment): {mpjpe_before:.3f}")

    # Apply coordinate alignment
    print(f"\nApplying coordinate alignment...")
    aligned_pred, aligned_gt = align_pose_coordinates(
        pred_pose_13[None, ...],  # Add batch dimension
        gt_pose_reshaped[None, ...],  # Add batch dimension
        pred_image_size=(256, 192),
        gt_image_size=(256, 192),  # Assuming GT is already scaled to model input size
        alignment_method='auto'
    )

    # Remove batch dimension
    aligned_pred = aligned_pred[0]
    aligned_gt = aligned_gt[0]

    # Compute alignment quality metrics
    alignment_metrics = compute_alignment_metrics(
        aligned_pred[None, ...], aligned_gt[None, ...]
    )

    print(f"\nAlignment quality assessment:")
    print(f"  Scale ratio: {alignment_metrics['scale_ratio']:.3f}")
    print(f"  Center offset: {alignment_metrics['center_offset']:.3f}")
    print(f"  Quality: {alignment_metrics['alignment_quality']}")

    # Compute MPJPE after alignment
    mpjpe_after = mpjpe_jax(jnp.array(aligned_pred[None, ...]), jnp.array(aligned_gt[None, ...]))
    print(f"\nMPJPE (after alignment): {mpjpe_after:.3f}")
    print(f"Improvement: {mpjpe_before - mpjpe_after:.3f} ({((mpjpe_before - mpjpe_after) / mpjpe_before * 100):.1f}%)")

    # Per-joint analysis (using aligned coordinates)
    joint_errors = np.sqrt(np.sum((aligned_pred - aligned_gt) ** 2, axis=1))
    print(f"\nPer-joint errors (aligned):")
    for i, (joint_name, error) in enumerate(zip(JOINT_NAMES_13, joint_errors)):
        print(f"  {joint_name:12s}: {error:6.2f}")

    return aligned_pred, aligned_gt, mpjpe_after, {'before': mpjpe_before, 'after': mpjpe_after, 'alignment_metrics': alignment_metrics}

def draw_pose_on_image(image, pose, connections, color=(255, 0, 0), point_size=3, line_width=2):
    """Draw pose keypoints and connections on image"""
    image_copy = image.copy()

    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(pose) and end_idx < len(pose):
            start_point = tuple(map(int, pose[start_idx]))
            end_point = tuple(map(int, pose[end_idx]))
            cv2.line(image_copy, start_point, end_point, color, line_width)

    # Draw keypoints
    for i, point in enumerate(pose):
        center = tuple(map(int, point))
        cv2.circle(image_copy, center, point_size, color, -1)
        # Add joint index as text
        cv2.putText(image_copy, str(i), (center[0]+5, center[1]-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return image_copy

def visualize_poses(sample_info, pred_pose_13, gt_pose):
    """Create comprehensive visualization of poses"""
    print("\nCreating pose visualization...")

    # Prepare the visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Pose Estimation Debug - Frame {sample_info["frame_idx"]}', fontsize=16)

    # Original processed frame
    if hasattr(sample_info['processed_frame'], 'numpy'):
        processed_frame = sample_info['processed_frame'].numpy()
    else:
        processed_frame = np.array(sample_info['processed_frame'])

    # Denormalize the processed frame for visualization
    processed_frame = processed_frame * 0.5 + 0.5  # Reverse normalization
    processed_frame = np.transpose(processed_frame, (1, 2, 0))  # CHW → HWC
    processed_frame = np.clip(processed_frame, 0, 1)

    axes[0, 0].imshow(processed_frame)
    axes[0, 0].set_title('Processed Frame (Model Input)')
    axes[0, 0].axis('off')

    # Ground truth pose on processed frame
    axes[0, 1].imshow(processed_frame)

    # Scale GT pose to processed frame size (256, 192)
    # GT pose is likely in original image coordinates, need to scale
    if sample_info['original_frame'] is not None:
        orig_h, orig_w = sample_info['original_frame'].shape[:2]
        proc_h, proc_w = 256, 192
        scale_x = proc_w / orig_w
        scale_y = proc_h / orig_h
        gt_pose_scaled = gt_pose.copy()
        gt_pose_scaled[:, 0] *= scale_x
        gt_pose_scaled[:, 1] *= scale_y
    else:
        gt_pose_scaled = gt_pose

    # Draw GT pose
    for i, point in enumerate(gt_pose_scaled):
        if 0 <= point[0] < proc_w and 0 <= point[1] < proc_h:
            axes[0, 1].scatter(point[0], point[1], c='green', s=30)
            axes[0, 1].text(point[0]+2, point[1]-2, str(i), fontsize=8, color='green')

    # Draw connections for GT
    for connection in JOINT_CONNECTIONS_13:
        start_idx, end_idx = connection
        if start_idx < len(gt_pose_scaled) and end_idx < len(gt_pose_scaled):
            start_point = gt_pose_scaled[start_idx]
            end_point = gt_pose_scaled[end_idx]
            if (0 <= start_point[0] < proc_w and 0 <= start_point[1] < proc_h and
                0 <= end_point[0] < proc_w and 0 <= end_point[1] < proc_h):
                axes[0, 1].plot([start_point[0], end_point[0]],
                               [start_point[1], end_point[1]], 'g-', linewidth=2)

    axes[0, 1].set_title('Ground Truth Pose')
    axes[0, 1].axis('off')

    # Predicted pose on processed frame
    axes[0, 2].imshow(processed_frame)

    # Predicted poses might be in normalized coordinates [-1, 1] or [0, 1]
    # Try different scaling approaches
    pred_pose_scaled = pred_pose_13.copy()

    # Check if predictions are in normalized coordinates
    if np.all(np.abs(pred_pose_13) <= 1.0):
        # Likely normalized to [-1, 1] or [0, 1]
        if np.any(pred_pose_13 < 0):
            # [-1, 1] → [0, image_size]
            pred_pose_scaled[:, 0] = (pred_pose_13[:, 0] + 1) * proc_w / 2
            pred_pose_scaled[:, 1] = (pred_pose_13[:, 1] + 1) * proc_h / 2
        else:
            # [0, 1] → [0, image_size]
            pred_pose_scaled[:, 0] = pred_pose_13[:, 0] * proc_w
            pred_pose_scaled[:, 1] = pred_pose_13[:, 1] * proc_h

    # Draw predicted pose
    for i, point in enumerate(pred_pose_scaled):
        if 0 <= point[0] < proc_w and 0 <= point[1] < proc_h:
            axes[0, 2].scatter(point[0], point[1], c='red', s=30)
            axes[0, 2].text(point[0]+2, point[1]-2, str(i), fontsize=8, color='red')

    # Draw connections for predictions
    for connection in JOINT_CONNECTIONS_13:
        start_idx, end_idx = connection
        if start_idx < len(pred_pose_scaled) and end_idx < len(pred_pose_scaled):
            start_point = pred_pose_scaled[start_idx]
            end_point = pred_pose_scaled[end_idx]
            if (0 <= start_point[0] < proc_w and 0 <= start_point[1] < proc_h and
                0 <= end_point[0] < proc_w and 0 <= end_point[1] < proc_h):
                axes[0, 2].plot([start_point[0], end_point[0]],
                               [start_point[1], end_point[1]], 'r-', linewidth=2)

    axes[0, 2].set_title('Predicted Pose')
    axes[0, 2].axis('off')

    # Coordinate scatter plots
    axes[1, 0].scatter(gt_pose[:, 0], gt_pose[:, 1], c='green', label='Ground Truth', alpha=0.7)
    axes[1, 0].scatter(pred_pose_13[:, 0], pred_pose_13[:, 1], c='red', label='Predicted', alpha=0.7)
    axes[1, 0].set_title('Coordinate Comparison (Raw)')
    axes[1, 0].set_xlabel('X coordinate')
    axes[1, 0].set_ylabel('Y coordinate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Error per joint
    joint_errors = np.sqrt(np.sum((pred_pose_13 - gt_pose) ** 2, axis=1))
    joint_indices = range(len(joint_errors))
    axes[1, 1].bar(joint_indices, joint_errors, color='orange', alpha=0.7)
    axes[1, 1].set_title('Per-Joint Error')
    axes[1, 1].set_xlabel('Joint Index')
    axes[1, 1].set_ylabel('Euclidean Error')
    axes[1, 1].set_xticks(joint_indices)
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Joint names on error plot
    joint_names_short = [name[:4] for name in JOINT_NAMES_13]
    axes[1, 2].bar(range(len(joint_errors)), joint_errors, color='orange', alpha=0.7)
    axes[1, 2].set_title('Per-Joint Error (Named)')
    axes[1, 2].set_xlabel('Joint')
    axes[1, 2].set_ylabel('Euclidean Error')
    axes[1, 2].set_xticks(range(len(joint_names_short)))
    axes[1, 2].set_xticklabels(joint_names_short, rotation=45, ha='right')

    plt.tight_layout()

    # Save the plot
    output_path = f"debug_pose_visualization_frame_{sample_info['frame_idx']}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[PASS] Visualization saved as: {output_path}")

    plt.show()

def main():
    """Main debugging function"""
    print("=" * 60)
    print("Pose Estimation Debug & Visualization")
    print("=" * 60)

    try:
        # Load model
        model, params, batch_stats = load_regressflow_model()

        # Load a single sample
        sample_info = load_single_sample(sample_idx=0)

        # Predict pose
        pred_pose = predict_pose(model, params, batch_stats, sample_info['processed_frame'])

        # Analyze coordinates
        aligned_pred, aligned_gt, mpjpe_aligned, metrics = analyze_coordinates(
            pred_pose, sample_info['gt_pose'], sample_info
        )

        # Create visualization
        visualize_poses(sample_info, aligned_pred, aligned_gt)

        print("\n" + "=" * 60)
        print("Debug Summary")
        print("=" * 60)
        print(f"[PASS] Successfully processed sample {sample_info['frame_idx']}")
        print(f"[PASS] MPJPE before alignment: {metrics['before']:.3f}")
        print(f"[PASS] MPJPE after alignment: {metrics['after']:.3f}")
        print(f"[PASS] Improvement: {metrics['before'] - metrics['after']:.3f} ({((metrics['before'] - metrics['after']) / metrics['before'] * 100):.1f}%)")
        print(f"[PASS] Alignment quality: {metrics['alignment_metrics']['alignment_quality']}")
        print(f"[PASS] Visualization created and saved")

        print("\nCoordinate System Analysis Results:")
        print(f"  - Coordinate alignment successfully applied")
        print(f"  - Scale ratio: {metrics['alignment_metrics']['scale_ratio']:.3f}")
        print(f"  - Center offset: {metrics['alignment_metrics']['center_offset']:.3f}")
        if metrics['after'] < 50:
            print(f"  - MPJPE now in reasonable range for pose estimation")
        else:
            print(f"  - MPJPE still high - may need further refinement")

    except Exception as e:
        print(f"[FAIL] Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()