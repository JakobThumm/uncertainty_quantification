"""
Batched GPU-accelerated transformations for pose estimation preprocessing

This module provides batched PyTorch operations on GPU to speed up preprocessing
by processing entire batches of frames at once instead of one-by-one.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional

from human_pose_pipeline.pose_estimation.h36m_settings import NORMALIZATION_OFFSET


def box_to_center_scale_batch(bboxes: torch.Tensor, aspect_ratio: float, scale_mult: float = 1.25) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert bounding boxes to center and scale format (batched)

    Args:
        bboxes: (B, 4) tensor of [xmin, ymin, xmax, ymax]
        aspect_ratio: Target aspect ratio (width / height)
        scale_mult: Scale multiplier (default 1.25)

    Returns:
        centers: (B, 2) tensor of [center_x, center_y]
        scales: (B, 2) tensor of [scale_x, scale_y]
    """
    pixel_std = 1.0
    xmin, ymin, xmax, ymax = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    w = xmax - xmin
    h = ymax - ymin

    center_x = xmin + w * 0.5
    center_y = ymin + h * 0.5
    centers = torch.stack([center_x, center_y], dim=1)

    # Adjust size based on aspect ratio (element-wise operations)
    # If w > aspect_ratio * h, adjust h; otherwise adjust w
    w_adjusted = torch.where(w > aspect_ratio * h, w, h * aspect_ratio)
    h_adjusted = torch.where(w > aspect_ratio * h, w / aspect_ratio, h)

    # Compute scale as [w / pixel_std, h / pixel_std]
    scales = torch.stack([w_adjusted / pixel_std, h_adjusted / pixel_std], dim=1)

    # Apply scale multiplier only if center[0] != -1
    scale_mult_mask = (centers[:, 0] != -1).unsqueeze(1)  # (B, 1)
    scales = torch.where(scale_mult_mask, scales * scale_mult, scales)

    return centers, scales


def get_affine_transform_batch(centers: torch.Tensor, scales: torch.Tensor,
                               output_size: Tuple[int, int], rot: float = 0, device: str = 'cuda') -> torch.Tensor:
    """
    Get affine transformation matrices for batch of centers and scales
    Replicates the behavior of transform_utils.get_affine_transform()

    Args:
        centers: (B, 2) tensor of [center_x, center_y]
        scales: (B, 2) tensor of [scale_x, scale_y]
        output_size: (width, height) of output image
        rot: Rotation angle in degrees (default 0)
        device: Device to create tensors on

    Returns:
        transforms: (B, 2, 3) affine transformation matrices
    """
    batch_size = centers.shape[0]
    output_w, output_h = output_size

    # Extract scale components - use only src_w like the original
    src_w = scales[:, 0]  # (B,)

    # Compute rotation
    rot_rad = np.pi * rot / 180
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    # Source direction vector: [0, src_w * -0.5] rotated by rot_rad
    src_dir_x = 0 * cs - (src_w * -0.5) * sn  # = src_w * 0.5 * sn
    src_dir_y = 0 * sn + (src_w * -0.5) * cs  # = src_w * -0.5 * cs

    # Destination direction vector: [0, dst_w * -0.5]
    dst_dir_x = torch.zeros(batch_size, device=device)
    dst_dir_y = torch.full((batch_size,), output_w * -0.5, device=device)

    # Define 3 source points
    src = torch.zeros((batch_size, 3, 2), device=device)
    src[:, 0, 0] = centers[:, 0]  # src[0] = center
    src[:, 0, 1] = centers[:, 1]
    src[:, 1, 0] = centers[:, 0] + src_dir_x  # src[1] = center + src_dir
    src[:, 1, 1] = centers[:, 1] + src_dir_y
    # src[2] = get_3rd_point(src[0], src[1]) = src[1] + perpendicular to (src[0] - src[1])
    direct_x = src[:, 0, 0] - src[:, 1, 0]
    direct_y = src[:, 0, 1] - src[:, 1, 1]
    src[:, 2, 0] = src[:, 1, 0] - direct_y  # perpendicular: [-direct_y, direct_x]
    src[:, 2, 1] = src[:, 1, 1] + direct_x

    # Define 3 destination points
    dst = torch.zeros((batch_size, 3, 2), device=device)
    dst[:, 0, 0] = output_w * 0.5  # dst[0] = [dst_w/2, dst_h/2]
    dst[:, 0, 1] = output_h * 0.5
    dst[:, 1, 0] = output_w * 0.5 + dst_dir_x  # dst[1] = [dst_w/2, dst_h/2] + dst_dir
    dst[:, 1, 1] = output_h * 0.5 + dst_dir_y
    # dst[2] = get_3rd_point(dst[0], dst[1])
    direct_x = dst[:, 0, 0] - dst[:, 1, 0]
    direct_y = dst[:, 0, 1] - dst[:, 1, 1]
    dst[:, 2, 0] = dst[:, 1, 0] - direct_y
    dst[:, 2, 1] = dst[:, 1, 1] + direct_x

    # Compute affine transformation for each batch element
    # Using the formula: M = dst * src^(-1) for affine transform
    # where src and dst are 2x3 matrices with homogeneous coordinates
    transforms = torch.zeros((batch_size, 2, 3), device=device)

    for i in range(batch_size):
        # Use opencv-style getAffineTransform logic
        # Convert to numpy for cv2.getAffineTransform
        src_np = src[i].cpu().numpy().astype(np.float32)
        dst_np = dst[i].cpu().numpy().astype(np.float32)
        trans_np = cv2.getAffineTransform(src_np, dst_np)
        transforms[i] = torch.from_numpy(trans_np).to(device)

    return transforms


def batched_affine_transform_images(
    images: torch.Tensor,
    transforms: torch.Tensor,
    output_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Apply affine transformations to batch of images

    Args:
        images: (B, C, H, W) tensor of images
        transforms: (B, 2, 3) affine transformation matrices
        output_size: (width, height) of output images

    Returns:
        transformed: (B, C, output_h, output_w) transformed images
    """
    output_w, output_h = output_size

    # Create sampling grid
    grid = F.affine_grid(transforms, [images.shape[0], images.shape[1], output_h, output_w],
                         align_corners=False)

    # Apply transformation
    transformed = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros',
                                align_corners=False)

    return transformed


def batched_affine_transform_points(
    points: torch.Tensor,
    transforms: torch.Tensor
) -> torch.Tensor:
    """
    Apply affine transformations to batch of point sets

    Args:
        points: (B, N, 2) tensor of points
        transforms: (B, 2, 3) affine transformation matrices

    Returns:
        transformed_points: (B, N, 2) transformed points
    """
    batch_size, num_points, _ = points.shape

    # Add homogeneous coordinate
    ones = torch.ones((batch_size, num_points, 1), device=points.device, dtype=points.dtype)
    points_hom = torch.cat([points, ones], dim=2)  # (B, N, 3)

    # Apply transformation: (B, 2, 3) @ (B, 3, N) = (B, 2, N)
    transformed = torch.bmm(transforms, points_hom.transpose(1, 2))  # (B, 2, N)

    # Transpose back to (B, N, 2)
    transformed_points = transformed.transpose(1, 2)

    return transformed_points


def normalize_images_regressflow(images: torch.Tensor) -> torch.Tensor:
    """
    Apply RegressFlow normalization to images

    Args:
        images: (B, C, H, W) tensor of images in range [0, 1]

    Returns:
        normalized: (B, C, H, W) normalized images
    """
    # RegressFlow uses ImageNet mean subtraction
    mean = torch.tensor(NORMALIZATION_OFFSET, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return images - mean


def batched_preprocess_frames_gpu(
    frames: List[np.ndarray],
    bboxes: List[Optional[List[float]]],
    poses: np.ndarray,
    scale_factors: List[Tuple[float, float]],
    output_image_size: Tuple[int, int] = (192, 256),  # (width, height)
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Preprocess a batch of frames on GPU using PyTorch

    Args:
        frames: List of B numpy arrays (H, W, 3) in range [0, 255], already resized for YOLO
        bboxes: List of B bounding boxes [xmin, ymin, xmax, ymax] or None
        poses: (B, 13, 2) numpy array of ground truth poses in original image space
        scale_factors: List of B tuples (scale_x, scale_y) from resize operation
        output_image_size: (width, height) for output preprocessed images
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        images_preprocessed: (B, 3, H, W) preprocessed images
        poses_normalized: (B, 13, 2) normalized poses
        metadata: Dictionary with transformation metadata
    """
    batch_size = len(frames)
    valid_indices = [i for i, bbox in enumerate(bboxes) if bbox is not None]

    if not valid_indices:
        # Return empty tensors if no valid bboxes
        return (torch.empty(0, 3, output_image_size[1], output_image_size[0], device=device),
                torch.empty(0, 13, 2, device=device),
                {'valid_indices': [], 'transforms': [], 'centers': [], 'scales': []})

    # Filter to valid frames
    valid_frames = [frames[i] for i in valid_indices]
    valid_bboxes = [bboxes[i] for i in valid_indices]
    valid_poses = poses[valid_indices]
    valid_scale_factors = [scale_factors[i] for i in valid_indices]

    # Convert frames to torch tensor (B, H, W, 3) -> (B, 3, H, W)
    frames_np = np.stack(valid_frames, axis=0)  # (B, H, W, 3)
    frames_tensor = torch.from_numpy(frames_np).to(device).float() / 255.0  # Normalize to [0, 1]
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (B, 3, H, W)

    # Convert bboxes to tensor
    bboxes_tensor = torch.tensor(valid_bboxes, device=device, dtype=torch.float32)  # (B, 4)

    # Convert poses to tensor and apply resize scaling
    poses_tensor = torch.from_numpy(valid_poses).to(device).float()  # (B, 13, 2)
    scale_factors_tensor = torch.tensor(valid_scale_factors, device=device, dtype=torch.float32)  # (B, 2)
    poses_resized = poses_tensor / scale_factors_tensor.unsqueeze(1)  # (B, 13, 2)

    # Compute centers and scales from bboxes
    aspect_ratio = output_image_size[0] / output_image_size[1]  # width / height
    centers, scales = box_to_center_scale_batch(bboxes_tensor, aspect_ratio)
    scales = scales * 1.0  # Additional scale multiplier (same as SimpleTransform)

    # Get affine transformation matrices
    transforms = get_affine_transform_batch(centers, scales, output_image_size, rot=0, device=device)

    # Apply affine transformations to images
    images_preprocessed = batched_affine_transform_images(frames_tensor, transforms, output_image_size)

    # Apply RegressFlow normalization
    images_preprocessed = normalize_images_regressflow(images_preprocessed)

    # Apply affine transformations to poses
    poses_transformed = batched_affine_transform_points(poses_resized, transforms)

    # Normalize poses to [-0.5, 0.5] range (RegressFlow format)
    output_w, output_h = output_image_size
    poses_normalized = poses_transformed.clone()
    poses_normalized[:, :, 0] = (poses_transformed[:, :, 0] / output_w) - 0.5
    poses_normalized[:, :, 1] = (poses_transformed[:, :, 1] / output_h) - 0.5

    # Collect metadata
    metadata = {
        'valid_indices': valid_indices,
        'transforms': transforms.cpu().numpy(),
        'centers': centers.cpu().numpy(),
        'scales': scales.cpu().numpy(),
        'bboxes': bboxes_tensor.cpu().numpy(),
        'scale_factors': scale_factors_tensor.cpu().numpy()
    }

    return images_preprocessed, poses_normalized, metadata


def batched_read_video_frames_cv2(
    video_path: str,
    frame_indices: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """
    Read multiple frames from video file and optionally resize them

    Args:
        video_path: Path to video file
        frame_indices: Array of frame indices to read
        target_size: Optional (width, height) to resize frames to

    Returns:
        frames: List of numpy arrays (H, W, 3) in RGB format
        scale_factors: List of (scale_x, scale_y) tuples if resized, else [(1.0, 1.0), ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames = []
    scale_factors = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Failed to read frame {frame_idx}")
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize if needed
        if target_size is not None:
            orig_h, orig_w = frame_rgb.shape[:2]
            target_w, target_h = target_size

            resized = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            scale_x = orig_w / target_w
            scale_y = orig_h / target_h

            frames.append(resized)
            scale_factors.append((scale_x, scale_y))
        else:
            frames.append(frame_rgb)
            scale_factors.append((1.0, 1.0))

    cap.release()

    return frames, scale_factors
