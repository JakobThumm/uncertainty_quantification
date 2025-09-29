#!/usr/bin/env python3
"""
Debug Tiger Pose Script

This script loads a single tiger image, applies all transformations,
and visualizes the pose prediction step by step to debug transformation issues.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Add root directory to path to access src
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from src.datasets.tiger_pose import TigerPoseDataset, tiger_pose_to_h36m_format
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    resize_image,
    get_pose_estimations_jax,
    joint_mapping
)
from human_pose_pipeline.evaluation.pose_metrics import MIRROR_13_JOINT_MODEL_MAP


def transform_tiger_image_for_human_detection(image_pil, target_size=(512, 640)):
    """
    Transform tiger image to make it more suitable for human detection:
    1. Rotate by 90 degrees (tigers are wider than tall, humans are taller than wide)
    2. Resize to target dimensions matching human image format
    """
    # Step 1: Rotate by 90 degrees clockwise to make tiger taller than wide
    rotated_image = image_pil.rotate(-90, expand=True)

    # Step 2: Resize to target dimensions
    original_width, original_height = rotated_image.size
    transformed_image = rotated_image.resize(target_size, Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.Resampling.LANCZOS)

    # Calculate transformation parameters for keypoint conversion
    transform_info = {
        'rotation_angle': -90,  # Degrees
        'original_size': image_pil.size,  # Before rotation
        'rotated_size': rotated_image.size,  # After rotation
        'final_size': target_size,  # After resize
        'scale_x': target_size[0] / original_width,
        'scale_y': target_size[1] / original_height
    }

    return transformed_image, transform_info


def transform_tiger_keypoints_for_human_detection(keypoints, transform_info):
    """
    Transform tiger keypoints to match the image transformations:
    1. Apply rotation transformation to keypoints (opposite of image rotation)
    2. Apply scaling transformation
    """
    transformed_kpts = keypoints.copy()

    # Original image dimensions
    orig_w, orig_h = transform_info['original_size']

    # Step 1: Apply 90-degree counterclockwise rotation (opposite of image rotation)
    # Image is rotated -90° (clockwise), so keypoints need +90° (counterclockwise)
    # For 90-degree counterclockwise rotation: (x, y) -> (orig_h - y, x)
    rotated_kpts = np.zeros_like(transformed_kpts)
    rotated_kpts[:, 0] = orig_h - transformed_kpts[:, 1]  # new_x = orig_h - old_y
    rotated_kpts[:, 1] = transformed_kpts[:, 0]  # new_y = old_x

    # Step 2: Apply scaling to match final image size
    final_kpts = rotated_kpts.copy()
    final_kpts[:, 0] *= transform_info['scale_x']
    final_kpts[:, 1] *= transform_info['scale_y']

    return final_kpts


def visualize_step_by_step(original_image, original_keypoints_12, rotated_image, rotated_keypoints_12,
                          final_image, final_keypoints_13, predicted_pose, valid_mask_12, valid_mask_13):
    """
    Create step-by-step visualization of the transformation process.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Define joint connections for tiger skeleton drawing (12 joints)
    # Tiger connections: 0-1, 1-2, 2-3, 3-4, 4-5, 3-7, 7-6, 2-8, 8-9, 2-10, 10-11
    tiger_connections_12 = [
        (0, 1),   # nose to left_eye
        (1, 2),   # left_eye to right_eye
        (2, 3),   # right_eye to left_ear
        (3, 4),   # left_ear to right_ear
        (4, 5),   # right_ear to front_left_paw
        (3, 7),   # left_ear to back_left_paw  # CORRECTED: joint 7 is back_left_paw
        (7, 6),   # back_left_paw to front_right_paw  # CORRECTED: joint 6 is front_right_paw
        (2, 8),   # right_eye to back_right_paw
        (8, 9),   # back_right_paw to tail_start
        (2, 10),  # right_eye to tail_middle
        (10, 11)  # tail_middle to tail_end
    ]

    # Define connections for H36M format (13 joints) - simplified human skeleton
    h36m_connections_13 = [
        (0, 1), (0, 2),  # Nose to shoulders
        (1, 3), (3, 5),  # Left arm
        (2, 4), (4, 6),  # Right arm
        (1, 2), (1, 7), (2, 8),  # Shoulders to hips
        (7, 8),  # Connect hips
        (7, 9), (9, 11),  # Left leg
        (8, 10), (10, 12)  # Right leg
    ]

    def draw_skeleton_12(ax, image, keypoints, valid_mask, title, color='green', label='Ground Truth'):
        ax.imshow(image)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        # Draw skeleton connections for tiger (12 joints)
        for connection in tiger_connections_12:
            start_idx, end_idx = connection
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                valid_mask[start_idx] and valid_mask[end_idx]):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                       color=color, linewidth=2, alpha=0.8, label=label if connection == tiger_connections_12[0] else "")

        # Draw keypoints
        for joint_idx, (keypoint, is_valid) in enumerate(zip(keypoints, valid_mask)):
            if is_valid:
                ax.plot(keypoint[0], keypoint[1], 'o', color=color, markersize=6, alpha=0.8)
                ax.text(keypoint[0]+3, keypoint[1]-3, str(joint_idx),
                       fontsize=8, color=color, fontweight='bold')

    def draw_skeleton_13(ax, image, keypoints, valid_mask, title, color='green', label='Ground Truth'):
        ax.imshow(image)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        # Draw skeleton connections for H36M (13 joints)
        for connection in h36m_connections_13:
            start_idx, end_idx = connection
            if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                valid_mask[start_idx] and valid_mask[end_idx]):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                       color=color, linewidth=2, alpha=0.8, label=label if connection == h36m_connections_13[0] else "")

        # Draw keypoints
        for joint_idx, (keypoint, is_valid) in enumerate(zip(keypoints, valid_mask)):
            if is_valid:
                ax.plot(keypoint[0], keypoint[1], 'o', color=color, markersize=6, alpha=0.8)
                ax.text(keypoint[0]+3, keypoint[1]-3, str(joint_idx),
                       fontsize=8, color=color, fontweight='bold')

    def draw_skeleton_tiger_from_h36m(ax, image, keypoints_h36m, valid_mask_h36m, title, color='green', label='Ground Truth'):
        """
        Draw tiger skeleton using H36M-format keypoints (13 joints) but with tiger connections.
        Maps back from H36M indices to tiger joint connections.
        """
        ax.imshow(image)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        # Tiger connections mapped to H36M indices:
        # Based on tiger_pose_to_h36m_format mapping
        # Correct tiger connections: 0-1, 1-2, 2-3, 3-4, 4-5, 3-7, 7-6, 2-8, 8-9, 2-10, 10-11
        tiger_connections_h36m = [
            (0, 1),   # nose(0) to left_eye->left_shoulder(1)
            (1, 2),   # left_eye->left_shoulder(1) to right_eye->right_shoulder(2)
            (2, 3),   # right_eye->right_shoulder(2) to left_ear->left_elbow(3)
            (3, 4),   # left_ear->left_elbow(3) to right_ear->right_elbow(4)
            (4, 5),   # right_ear->right_elbow(4) to front_left_paw->left_wrist(5)
            (3, 7),   # left_ear->left_elbow(3) to back_left_paw->left_hip(7)  # CORRECTED
            (7, 6),   # back_left_paw->left_hip(7) to front_right_paw->right_wrist(6)  # CORRECTED
            (2, 8),   # right_eye->right_shoulder(2) to back_right_paw->right_hip(8)
            (8, 9),   # back_right_paw->right_hip(8) to tail_start->left_knee(9)
            (2, 10),  # right_eye->right_shoulder(2) to tail_middle->right_knee(10)
            (10, 11)  # tail_middle->right_knee(10) to tail_end->left_ankle(11)
        ]

        # Draw skeleton connections for tiger using H36M keypoints
        for connection in tiger_connections_h36m:
            start_idx, end_idx = connection
            if (start_idx < len(keypoints_h36m) and end_idx < len(keypoints_h36m) and
                valid_mask_h36m[start_idx] and valid_mask_h36m[end_idx]):
                start_point = keypoints_h36m[start_idx]
                end_point = keypoints_h36m[end_idx]
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                       color=color, linewidth=2, alpha=0.8, label=label if connection == tiger_connections_h36m[0] else "")

        # Draw keypoints
        for joint_idx, (keypoint, is_valid) in enumerate(zip(keypoints_h36m, valid_mask_h36m)):
            if is_valid:
                ax.plot(keypoint[0], keypoint[1], 'o', color=color, markersize=6, alpha=0.8)
                ax.text(keypoint[0]+3, keypoint[1]-3, str(joint_idx),
                       fontsize=8, color=color, fontweight='bold')

    # Original tiger image with original keypoints (12 joints)
    draw_skeleton_12(axes[0, 0], original_image, original_keypoints_12, valid_mask_12,
                    "1. Original Tiger Image\n(12 tiger joints)", 'blue', 'Original')

    # Rotated image with rotated keypoints (12 joints)
    draw_skeleton_12(axes[0, 1], rotated_image, rotated_keypoints_12, valid_mask_12,
                    "2. Rotated Image (-90°)\n(12 tiger joints rotated)", 'orange', 'Rotated')

    # Final transformed image with final keypoints (13 joints, but still tiger structure)
    # We'll use a custom drawing function that maps the 13 H36M joints back to tiger connections
    draw_skeleton_tiger_from_h36m(axes[1, 0], final_image, final_keypoints_13, valid_mask_13,
                                 "3. Resized Image (512x640)\n(tiger mapped to 13 H36M joints)", 'green', 'Transformed GT')

    # Final image with both ground truth and predicted pose
    axes[1, 1].imshow(final_image)
    axes[1, 1].set_title("4. Final Result\nGreen=GT, Red=Predicted", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    # Tiger connections mapped to H36M indices (same as above)
    # Correct tiger connections: 0-1, 1-2, 2-3, 3-4, 4-5, 3-7, 7-6, 2-8, 8-9, 2-10, 10-11
    tiger_connections_h36m = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 7), (7, 6), (2, 8), (8, 9), (2, 10), (10, 11)
    ]

    # Draw ground truth skeleton (green) using tiger connections on H36M keypoints
    for connection in tiger_connections_h36m:
        start_idx, end_idx = connection
        if (start_idx < len(final_keypoints_13) and end_idx < len(final_keypoints_13) and
            valid_mask_13[start_idx] and valid_mask_13[end_idx]):
            start_point = final_keypoints_13[start_idx]
            end_point = final_keypoints_13[end_idx]
            axes[1, 1].plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                           'g-', linewidth=2, alpha=0.8, label='Ground Truth' if connection == tiger_connections_h36m[0] else "")

    # Draw predicted skeleton (red) if we have a prediction
    if np.sum(np.abs(predicted_pose)) > 0:
        for connection in tiger_connections_h36m:
            start_idx, end_idx = connection
            if start_idx < len(predicted_pose) and end_idx < len(predicted_pose):
                start_point = predicted_pose[start_idx]
                end_point = predicted_pose[end_idx]
                axes[1, 1].plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                               'r-', linewidth=2, alpha=0.8, label='Predicted' if connection == tiger_connections_h36m[0] else "")

    # Draw keypoints
    for joint_idx, (keypoint, is_valid) in enumerate(zip(final_keypoints_13, valid_mask_13)):
        if is_valid:
            axes[1, 1].plot(keypoint[0], keypoint[1], 'go', markersize=6, alpha=0.8)
            axes[1, 1].text(keypoint[0]+3, keypoint[1]-3, str(joint_idx),
                           fontsize=8, color='green', fontweight='bold')

    if np.sum(np.abs(predicted_pose)) > 0:
        for joint_idx, keypoint in enumerate(predicted_pose):
            axes[1, 1].plot(keypoint[0], keypoint[1], 'ro', markersize=6, alpha=0.8)
            axes[1, 1].text(keypoint[0]+3, keypoint[1]+15, str(joint_idx),
                           fontsize=8, color='red', fontweight='bold')

    axes[1, 1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("debug_tiger_transformation_steps.png", dpi=150, bbox_inches='tight')
    plt.close()

    print("Step-by-step visualization saved: debug_tiger_transformation_steps.png")


def main():
    """Main debug function for tiger pose transformations."""
    print("=== Debug Tiger Pose Transformations ===")

    try:
        # Initialize models
        print("Initializing models...")

        # Initialize JAX pose estimation model
        models_dir = os.path.join(root_dir, "models_tianle", "H36M", "RegressFlow", "seed_420")
        checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_with_unc")
        model, params, batch_stats = initialize_jax_models(checkpoint_path_jax)

        # Initialize YOLO human detector (not used but required for consistency)
        human_detector, device_torch = initialize_human_detector('cuda')

        print("Models initialized successfully!")

        # Load tiger dataset
        print("\nLoading tiger dataset...")
        tiger_dataset = TigerPoseDataset(
            root_dir=os.path.join(root_dir, "datasets", "tiger-pose"),
            split='val',
            image_size=(256, 256)
        )

        # Get first sample
        sample = tiger_dataset[0]
        print(f"Loaded tiger sample from: {sample.get('image_path', 'unknown')}")

        # Extract data
        image_tensor = sample['image']
        tiger_keypoints_12 = sample['keypoints']  # (12, 2)
        valid_keypoints_12 = sample['valid_keypoints']  # (12,)

        print(f"Original tiger keypoints (12 joints):")
        for i, (kpt, valid) in enumerate(zip(tiger_keypoints_12, valid_keypoints_12)):
            if valid:
                print(f"  Joint {i}: ({kpt[0]:.1f}, {kpt[1]:.1f})")

        # Convert tiger keypoints to H36M format (12 -> 13)
        tiger_keypoints_13, valid_keypoints_13 = tiger_pose_to_h36m_format(
            tiger_keypoints_12[np.newaxis, :, :],
            valid_keypoints_12[np.newaxis, :]
        )
        tiger_keypoints_13 = tiger_keypoints_13[0]  # Remove batch dim
        valid_keypoints_13 = valid_keypoints_13[0]  # Remove batch dim

        # Convert tensor to PIL Image
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image_tensor_denorm = image_tensor * std[:, None, None] + mean[:, None, None]
        image_tensor_denorm = torch.clamp(image_tensor_denorm, 0, 1)
        image_np = (image_tensor_denorm.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
        original_image = Image.fromarray(image_np)

        print(f"Original image size: {original_image.size}")
        print(f"Tiger keypoints (13): {tiger_keypoints_13.shape}")
        print(f"Valid keypoints: {np.sum(valid_keypoints_13)}/{len(valid_keypoints_13)}")

        # Step 1: Transform image and keypoints
        print("\nApplying transformations...")

        # Transform image
        transformed_image, transform_info = transform_tiger_image_for_human_detection(
            original_image, target_size=(512, 640)
        )

        # Create intermediate rotated image for visualization
        rotated_image = original_image.rotate(-90, expand=True)

        # Transform keypoints step by step for visualization
        # Use original 12 tiger keypoints for the first steps
        original_keypoints_12 = tiger_keypoints_12.copy()

        # Rotated keypoints (after rotation but before scaling) - still 12 joints
        orig_w, orig_h = transform_info['original_size']
        rotated_keypoints_12 = np.zeros_like(original_keypoints_12)
        rotated_keypoints_12[:, 0] = orig_h - original_keypoints_12[:, 1]  # new_x = orig_h - old_y
        rotated_keypoints_12[:, 1] = original_keypoints_12[:, 0]  # new_y = old_x

        print(f"Sample keypoint transformation:")
        print(f"  Original keypoint 0: ({original_keypoints_12[0, 0]:.1f}, {original_keypoints_12[0, 1]:.1f})")
        print(f"  Rotated keypoint 0: ({rotated_keypoints_12[0, 0]:.1f}, {rotated_keypoints_12[0, 1]:.1f})")
        print(f"  Original size: {orig_w} x {orig_h}")

        # Final transformed keypoints (convert to H36M format and transform)
        final_keypoints = transform_tiger_keypoints_for_human_detection(
            tiger_keypoints_13, transform_info
        )

        print(f"Transformation info: {transform_info}")
        print(f"Original keypoints range: x[{np.min(original_keypoints_12[:, 0]):.1f}, {np.max(original_keypoints_12[:, 0]):.1f}], y[{np.min(original_keypoints_12[:, 1]):.1f}, {np.max(original_keypoints_12[:, 1]):.1f}]")
        print(f"Final keypoints range: x[{np.min(final_keypoints[:, 0]):.1f}, {np.max(final_keypoints[:, 0]):.1f}], y[{np.min(final_keypoints[:, 1]):.1f}, {np.max(final_keypoints[:, 1]):.1f}]")

        # Step 2: Run pose estimation
        print("\nRunning pose estimation...")

        # Skip human detection and use full image as bounding box
        resized_image, original_dimensions, scale_factors = resize_image(transformed_image)
        person_boxes = [[0.0, 0.0, 512, 640]]  # Full image

        # Perform pose estimation
        pose_estimations = get_pose_estimations_jax(
            transformed_image, original_dimensions, scale_factors, person_boxes,
            model, params, batch_stats, False
        )

        if pose_estimations:
            first_pose = np.array(pose_estimations[0]['keypoints'])
            first_uncertainty = np.array(pose_estimations[0]['uncertainties'])
            first_covariance = np.array(pose_estimations[0]['covariance'])

            # Apply mirror mapping
            mapped_pose = joint_mapping(first_pose, MIRROR_13_JOINT_MODEL_MAP)

            print("Pose estimation successful!")
            print(f"Predicted pose range: x[{np.min(mapped_pose[:, 0]):.1f}, {np.max(mapped_pose[:, 0]):.1f}], y[{np.min(mapped_pose[:, 1]):.1f}, {np.max(mapped_pose[:, 1]):.1f}]")
        else:
            mapped_pose = np.zeros((13, 2))
            print("No pose detected!")

        # Step 3: Create visualization
        print("\nCreating step-by-step visualization...")

        visualize_step_by_step(
            original_image=original_image,
            original_keypoints_12=original_keypoints_12,
            rotated_image=rotated_image,
            rotated_keypoints_12=rotated_keypoints_12,
            final_image=transformed_image,
            final_keypoints_13=final_keypoints,
            predicted_pose=mapped_pose,
            valid_mask_12=valid_keypoints_12,
            valid_mask_13=valid_keypoints_13
        )

        print("\n=== Debug Complete ===")
        print("Check debug_tiger_transformation_steps.png to see the step-by-step process")

    except Exception as e:
        print(f"\nError during debug: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()