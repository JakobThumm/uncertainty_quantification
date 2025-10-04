#!/usr/bin/env python3
"""
GPU-Accelerated Tiger Pose Dataset Preprocessing

This script preprocesses the tiger pose dataset by:
1. Loading tiger images in batches
2. Rotating images by -90 degrees (to make tigers taller than wide, like humans)
3. Resizing to TRANSFORM_IMAGE_SIZE
4. Transforming ground truth keypoints accordingly
5. Converting 12 tiger keypoints to 13 H36M format
6. Applying RegressFlow normalization

This is much simpler than H36M preprocessing as tigers don't require bbox detection.
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from src.datasets.tiger_pose import TigerPoseDataset, tiger_pose_to_h36m_format
from human_pose_pipeline.pose_estimation.h36m_settings import (
    TRANSFORM_IMAGE_SIZE,
    NORMALIZATION_OFFSET,
    MIRROR_13_JOINT_MODEL_MAP
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def rotate_image_batch_gpu(images: torch.Tensor, angle: float = -90) -> torch.Tensor:
    """
    Rotate a batch of images by a given angle on GPU

    Args:
        images: (B, C, H, W) tensor of images
        angle: Rotation angle in degrees (default -90 for clockwise rotation)

    Returns:
        rotated: (B, C, H', W') rotated images
    """
    # For -90 degree rotation, we can use torch.rot90
    if angle == -90:
        # rot90 with k=1 rotates 90 degrees counter-clockwise
        # For -90 (clockwise), we use k=3 (or k=-1)
        return torch.rot90(images, k=-1, dims=[2, 3])
    elif angle == 90:
        return torch.rot90(images, k=1, dims=[2, 3])
    elif angle == 180:
        return torch.rot90(images, k=2, dims=[2, 3])
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}. Use -90, 90, or 180.")


def rotate_keypoints_batch(keypoints: torch.Tensor, original_size: tuple, angle: float = -90) -> torch.Tensor:
    """
    Rotate keypoints to match image rotation

    Args:
        keypoints: (B, N, 2) tensor of keypoints in original image space
        original_size: (width, height) of original image
        angle: Rotation angle in degrees (should match image rotation)

    Returns:
        rotated_keypoints: (B, N, 2) tensor of rotated keypoints
    """
    orig_w, orig_h = original_size

    # For -90 degree (clockwise) image rotation, keypoints need +90 degree (counter-clockwise) rotation
    # Transformation: (x, y) -> (orig_h - y, x)
    if angle == -90:
        rotated_kpts = keypoints.clone()
        rotated_kpts[:, :, 0] = orig_h - keypoints[:, :, 1]  # new_x = orig_h - old_y
        rotated_kpts[:, :, 1] = keypoints[:, :, 0]  # new_y = old_x
        return rotated_kpts
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")


def normalize_images_regressflow(images: torch.Tensor) -> torch.Tensor:
    """
    Apply RegressFlow normalization to images

    Args:
        images: (B, C, H, W) tensor of images in range [0, 1]

    Returns:
        normalized: (B, C, H, W) normalized images
    """
    offset = torch.tensor(NORMALIZATION_OFFSET, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return images + offset


def preprocess_tiger_batch_gpu(
    images_pil: list,
    keypoints_12: np.ndarray,
    valid_masks: np.ndarray,
    original_sizes: list,
    target_size: tuple = None,
    rotation_angle: float = -90,
    device: str = 'cuda'
) -> tuple:
    """
    Preprocess a batch of tiger images and keypoints on GPU

    Args:
        images_pil: List of PIL Images
        keypoints_12: (B, 12, 2) numpy array of 12 tiger keypoints
        valid_masks: (B, 12) numpy array of valid keypoint masks
        original_sizes: List of (width, height) tuples for original image sizes
        target_size: (width, height) for output (default: TRANSFORM_IMAGE_SIZE)
        rotation_angle: Rotation angle in degrees (default: -90)
        device: Device to run on

    Returns:
        images_preprocessed: (B, 3, H, W) preprocessed images
        poses_normalized: (B, 13, 2) normalized poses in H36M format
        metadata: Dictionary with transformation info
    """
    if target_size is None:
        target_size = tuple(TRANSFORM_IMAGE_SIZE)  # (width, height)

    batch_size = len(images_pil)
    target_w, target_h = target_size

    # Step 1: Convert PIL images to tensor
    images_np = []
    for img_pil in images_pil:
        img_array = np.array(img_pil)  # (H, W, 3)
        images_np.append(img_array)

    images_np = np.stack(images_np, axis=0)  # (B, H, W, 3)

    # Normalize to [0, 1] if needed
    if images_np.max() > 1.0:
        images_tensor = torch.from_numpy(images_np).to(device).float() / 255.0
    else:
        images_tensor = torch.from_numpy(images_np).to(device).float()

    images_tensor = images_tensor.permute(0, 3, 1, 2)  # (B, 3, H, W)

    # Step 2: Rotate images
    images_rotated = rotate_image_batch_gpu(images_tensor, angle=rotation_angle)

    # Step 3: Resize to target size
    images_resized = F.interpolate(
        images_rotated,
        size=(target_h, target_w),  # (height, width) for F.interpolate
        mode='bilinear',
        align_corners=False
    )

    # Step 4: Apply RegressFlow normalization
    images_preprocessed = normalize_images_regressflow(images_resized)

    # Step 5: Transform keypoints
    keypoints_tensor = torch.from_numpy(keypoints_12).to(device).float()  # (B, 12, 2)

    # Rotate keypoints (for each sample individually since original sizes may differ)
    rotated_keypoints = []
    for i in range(batch_size):
        orig_w, orig_h = original_sizes[i]
        kpts = keypoints_tensor[i:i+1]  # (1, 12, 2)
        kpts_rotated = rotate_keypoints_batch(kpts, (orig_w, orig_h), angle=rotation_angle)
        rotated_keypoints.append(kpts_rotated)

    rotated_keypoints = torch.cat(rotated_keypoints, dim=0)  # (B, 12, 2)

    # Compute scale factors after rotation
    # After -90 degree rotation, dimensions swap: (orig_w, orig_h) -> (orig_h, orig_w)
    scale_factors = []
    for i in range(batch_size):
        orig_w, orig_h = original_sizes[i]
        # After rotation: rotated_w = orig_h, rotated_h = orig_w
        scale_x = target_w / orig_h
        scale_y = target_h / orig_w
        scale_factors.append([scale_x, scale_y])

    scale_factors = torch.tensor(scale_factors, device=device, dtype=torch.float32)  # (B, 2)

    # Apply scaling to rotated keypoints
    scaled_keypoints = rotated_keypoints.clone()
    scaled_keypoints[:, :, 0] *= scale_factors[:, 0:1]  # x coordinates
    scaled_keypoints[:, :, 1] *= scale_factors[:, 1:2]  # y coordinates

    # Step 6: Convert 12 tiger keypoints to 13 H36M format
    scaled_keypoints_np = scaled_keypoints.cpu().numpy()  # (B, 12, 2)
    h36m_keypoints, h36m_valid = tiger_pose_to_h36m_format(scaled_keypoints_np, valid_masks)
    h36m_keypoints_tensor = torch.from_numpy(h36m_keypoints).to(device).float()  # (B, 13, 2)

    # Step 7: Normalize poses to [-0.5, 0.5] range (RegressFlow format)
    poses_normalized = h36m_keypoints_tensor.clone()
    poses_normalized[:, :, 0] = (h36m_keypoints_tensor[:, :, 0] / target_w) - 0.5
    poses_normalized[:, :, 1] = (h36m_keypoints_tensor[:, :, 1] / target_h) - 0.5

    # Collect metadata
    metadata = {
        'scale_factors': scale_factors.cpu().numpy(),
        'rotation_angle': rotation_angle,
        'target_size': target_size,
        'original_sizes': original_sizes
    }

    return images_preprocessed, poses_normalized, h36m_keypoints_tensor, h36m_valid, metadata


def preprocess_tiger_dataset_gpu(
    dataset_dir,
    output_dir,
    batch_size=32,
    device='cuda'
):
    """
    Preprocess tiger pose dataset with GPU acceleration

    Args:
        dataset_dir: Path to tiger-pose dataset directory
        output_dir: Path to save preprocessed data
        batch_size: Number of images to process at once
        device: Device for PyTorch operations
    """
    print("=" * 80)
    print("Tiger Pose Dataset Preprocessing")
    print("=" * 80)

    # Process each split
    for split_name in ['train', 'val']:
        print(f"\nProcessing {split_name} split...")

        # Load dataset (without default transforms)
        dataset = TigerPoseDataset(
            root_dir=dataset_dir,
            split=split_name,
            transform=None,  # We'll handle transforms manually
            image_size=(256, 256)  # Temporary size, will be overridden
        )

        if len(dataset) == 0:
            print(f"No samples found in {split_name} split, skipping...")
            continue

        # Create output directories
        images_output_dir = os.path.join(output_dir, split_name, 'PreprocessedImages')
        poses_output_dir = os.path.join(output_dir, split_name, 'PreprocessedPoses')
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(poses_output_dir, exist_ok=True)

        # Process in batches
        all_preprocessed_images = []
        all_preprocessed_poses = []
        all_preprocessed_poses_pixel = []
        all_valid_masks = []
        all_metadata = []
        all_image_paths = []

        num_batches = (len(dataset) + batch_size - 1) // batch_size

        total_time = 0

        for batch_idx in tqdm(range(num_batches), desc=f'{split_name} batches'):
            batch_start_time = time.time()

            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))

            # Collect batch data
            batch_images_pil = []
            batch_keypoints = []
            batch_valid_masks = []
            batch_original_sizes = []
            batch_image_paths = []

            for idx in range(start_idx, end_idx):
                img_path, label_path = dataset.valid_samples[idx]

                # Load image
                img_pil = Image.open(img_path).convert('RGB')
                orig_w, orig_h = img_pil.size

                # Load keypoints from label file
                with open(label_path, 'r') as f:
                    line = f.readline().strip()

                if not line:
                    continue

                values = list(map(float, line.split()))
                if len(values) < 5 + 24:
                    continue

                # Extract keypoints (normalized coordinates)
                keypoints_norm = np.array(values[5:29]).reshape(12, 2)

                # Convert to pixel coordinates
                keypoints_pixel = keypoints_norm.copy()
                keypoints_pixel[:, 0] *= orig_w
                keypoints_pixel[:, 1] *= orig_h

                # Valid mask
                valid_mask = ~((keypoints_norm[:, 0] == 0) & (keypoints_norm[:, 1] == 0))

                batch_images_pil.append(img_pil)
                batch_keypoints.append(keypoints_pixel)
                batch_valid_masks.append(valid_mask)
                batch_original_sizes.append((orig_w, orig_h))
                batch_image_paths.append(img_path)

            if len(batch_images_pil) == 0:
                continue

            # Convert to numpy arrays
            batch_keypoints = np.stack(batch_keypoints, axis=0)  # (B, 12, 2)
            batch_valid_masks = np.stack(batch_valid_masks, axis=0)  # (B, 12)

            # Preprocess batch on GPU
            images_prep, poses_norm, poses_pixel, valid_h36m, metadata = preprocess_tiger_batch_gpu(
                images_pil=batch_images_pil,
                keypoints_12=batch_keypoints,
                valid_masks=batch_valid_masks,
                original_sizes=batch_original_sizes,
                target_size=tuple(TRANSFORM_IMAGE_SIZE),
                rotation_angle=-90,
                device=device
            )

            # Apply mirror mapping to match H36M convention
            poses_norm_mirrored = poses_norm[:, MIRROR_13_JOINT_MODEL_MAP, :].cpu().numpy()
            poses_pixel_mirrored = poses_pixel[:, MIRROR_13_JOINT_MODEL_MAP, :].cpu().numpy()
            valid_h36m_mirrored = valid_h36m[:, MIRROR_13_JOINT_MODEL_MAP]

            # Accumulate results
            all_preprocessed_images.append(images_prep.cpu().numpy())
            all_preprocessed_poses.append(poses_norm_mirrored)
            all_preprocessed_poses_pixel.append(poses_pixel_mirrored)
            all_valid_masks.append(valid_h36m_mirrored)
            all_metadata.append(metadata)
            all_image_paths.extend(batch_image_paths)

            batch_time = time.time() - batch_start_time
            total_time += batch_time

        # Concatenate all batches
        if len(all_preprocessed_images) == 0:
            print(f"No valid samples in {split_name} split")
            continue

        all_images = np.concatenate(all_preprocessed_images, axis=0)  # (N, 3, 256, 192)
        all_poses = np.concatenate(all_preprocessed_poses, axis=0)  # (N, 13, 2)
        all_poses_pix = np.concatenate(all_preprocessed_poses_pixel, axis=0)  # (N, 13, 2)
        all_valid = np.concatenate(all_valid_masks, axis=0)  # (N, 13)

        # Save preprocessed data
        print(f"\nSaving preprocessed data for {split_name}...")

        # Save images
        images_output_path = os.path.join(images_output_dir, f"tiger_pose_{split_name}.npy")
        np.save(images_output_path, all_images)

        # Save poses and metadata
        poses_output_path = os.path.join(poses_output_dir, f"tiger_pose_{split_name}.npz")
        np.savez_compressed(
            poses_output_path,
            poses_normalized=all_poses,
            poses_pixel=all_poses_pix,
            valid_keypoints=all_valid,
            image_paths=all_image_paths,
            target_size=TRANSFORM_IMAGE_SIZE,
            rotation_angle=-90
        )

        print(f"  Saved {len(all_images)} preprocessed samples")
        print(f"  Images: {images_output_path}")
        print(f"  Poses: {poses_output_path}")
        print(f"  Total time: {total_time:.1f}s ({total_time / len(all_images) * 1000:.1f}ms/sample)")

    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Preprocess tiger pose dataset with GPU acceleration')
    parser.add_argument('--dataset_dir', type=str,
                       default='datasets/tiger-pose',
                       help='Path to tiger-pose dataset')
    parser.add_argument('--output_dir', type=str,
                       default='datasets/tiger-pose/preprocessed',
                       help='Path to save preprocessed data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for PyTorch operations')

    args = parser.parse_args()

    preprocess_tiger_dataset_gpu(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device
    )


if __name__ == '__main__':
    main()
