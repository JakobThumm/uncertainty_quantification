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
    transforms_cv: torch.Tensor,
    output_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Apply affine transformations to batch of images

    Converts OpenCV-style affine matrices to PyTorch format and applies them.

    Args:
        images: (B, C, H, W) tensor of images
        transforms_cv: (B, 2, 3) OpenCV-style affine transformation matrices
        output_size: (width, height) of output images

    Returns:
        transformed: (B, C, output_h, output_w) transformed images
    """
    batch_size = images.shape[0]
    input_h, input_w = images.shape[2], images.shape[3]
    output_w, output_h = output_size

    # Convert OpenCV affine matrices to PyTorch format
    # OpenCV: pixel coords -> pixel coords
    # PyTorch: normalized coords [-1,1] -> normalized coords [-1,1]

    # Create normalized coordinate transformation matrices
    transforms_pt = torch.zeros_like(transforms_cv)

    for i in range(batch_size):
        M = transforms_cv[i]  # (2, 3) OpenCV matrix

        # Use cv2.invertAffineTransform to get inverse
        M_inv = cv2.invertAffineTransform(M.cpu().numpy())
        M_inv_torch = torch.from_numpy(M_inv).to(transforms_cv.device)

        # Convert to 3x3 homogeneous form for easier composition
        M_inv_hom = torch.eye(3, device=transforms_cv.device, dtype=torch.float32)
        M_inv_hom[:2, :] = M_inv_torch

        # Scale transformations for normalized coordinates
        # Output: norm -> pixel: x_pix = (x_norm + 1) * output_w / 2
        scale_out_hom = torch.tensor([[output_w/2.0, 0, output_w/2.0],
                                       [0, output_h/2.0, output_h/2.0],
                                       [0, 0, 1]],
                                      device=transforms_cv.device, dtype=torch.float32)

        # Input: pixel -> norm: x_norm = 2 * x_pix / input_w - 1
        scale_in_hom = torch.tensor([[2.0/input_w, 0, -1],
                                      [0, 2.0/input_h, -1],
                                      [0, 0, 1]],
                                     device=transforms_cv.device, dtype=torch.float32)

        # Combine: output_norm -> output_pix -> input_pix -> input_norm
        # M_pt = scale_in @ M_inv @ scale_out
        temp = torch.mm(M_inv_hom, scale_out_hom)
        M_pt_hom = torch.mm(scale_in_hom, temp)

        # Extract 2x3 affine matrix
        transforms_pt[i] = M_pt_hom[:2, :]

    # Create sampling grid
    grid = F.affine_grid(transforms_pt, [batch_size, images.shape[1], output_h, output_w],
                         align_corners=False)

    # Apply transformation
    transformed = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros',
                                align_corners=False)

    return transformed


def batched_crop_and_resize(
    images: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
    output_size: Tuple[int, int]
) -> torch.Tensor:
    """
    Crop regions from images and resize to output size (batched, simplified for rotation=0)

    Args:
        images: (B, C, H, W) tensor of images
        centers: (B, 2) tensor of [center_x, center_y]
        scales: (B, 2) tensor of [scale_x, scale_y] - size of region to extract
        output_size: (width, height) of output images

    Returns:
        transformed: (B, C, output_h, output_w) transformed images
    """
    batch_size = images.shape[0]
    output_w, output_h = output_size
    device = images.device

    # Extract crops for each image in the batch
    crops = []
    for i in range(batch_size):
        cx, cy = centers[i, 0].item(), centers[i, 1].item()
        scale_x, scale_y = scales[i, 0].item(), scales[i, 1].item()

        # Compute crop box (half-width and half-height)
        half_w = scale_x / 2.0
        half_h = scale_y / 2.0

        # Crop coordinates
        x1 = int(cx - half_w)
        y1 = int(cy - half_h)
        x2 = int(cx + half_w)
        y2 = int(cy + half_h)

        # Clamp to image bounds
        img_h, img_w = images.shape[2], images.shape[3]
        x1_clamp = max(0, x1)
        y1_clamp = max(0, y1)
        x2_clamp = min(img_w, x2)
        y2_clamp = min(img_h, y2)

        # Crop the region
        crop = images[i:i+1, :, y1_clamp:y2_clamp, x1_clamp:x2_clamp]

        # If crop is empty due to out-of-bounds, create zeros
        if crop.shape[2] == 0 or crop.shape[3] == 0:
            crop = torch.zeros((1, images.shape[1], 1, 1), device=device, dtype=images.dtype)

        # Resize to output size
        crop_resized = F.interpolate(crop, size=(output_h, output_w), mode='bilinear', align_corners=False)
        crops.append(crop_resized)

    # Stack all crops
    result = torch.cat(crops, dim=0)
    return result


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
    # NORMALIZATION_OFFSET contains negative values [-0.406, -0.457, -0.480]
    # We add these negative values (equivalent to subtracting positive mean)
    offset = torch.tensor(NORMALIZATION_OFFSET, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return images + offset


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

    # Normalize to [0, 1] if needed
    if frames_np.max() > 1.0:
        frames_tensor = torch.from_numpy(frames_np).to(device).float() / 255.0
    else:
        frames_tensor = torch.from_numpy(frames_np).to(device).float()

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

    # Get affine transformation matrices (for pose transformation)
    transforms = get_affine_transform_batch(centers, scales, output_image_size, rot=0, device=device)

    # Crop and resize images (simplified approach for rotation=0)
    images_preprocessed = batched_affine_transform_images(frames_tensor, transforms, output_image_size)
    # images_preprocessed = batched_crop_and_resize(frames_tensor, centers, scales, output_image_size)

    # DEBUG: Visualize first preprocessed image (after affine transform, before normalization)
    # import matplotlib.pyplot as plt
    # debug_img0 = frames_tensor[0].cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
    # debug_img_viz0 = np.clip(debug_img0, 0, 1)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(debug_img_viz0)
    # plt.title('Preprocessed Image [0] (before affine transform)')
    # plt.axis('off')
    # plt.savefig('visualizations/debug_preprocessed_img_0_before.png', dpi=150, bbox_inches='tight')
    # plt.close()
    # debug_img = images_preprocessed[0].cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
    # debug_img_viz = np.clip(debug_img, 0, 1)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(debug_img_viz)
    # plt.title('Preprocessed Image [0] (after affine transform, before normalization)')
    # plt.axis('off')
    # plt.savefig('visualizations/debug_preprocessed_img_0_after.png', dpi=150, bbox_inches='tight')
    # plt.close()
    # print(f"[DEBUG] Saved preprocessed images to visualizations/debug_preprocessed_img_0_before.png and visualizations/debug_preprocessed_img_0_after.png")

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


def triangulate_points_torch(
    P1: torch.Tensor,
    P2: torch.Tensor,
    pts1: torch.Tensor,
    pts2: torch.Tensor
) -> torch.Tensor:
    """
    Triangulate 3D points from two camera views using PyTorch (DLT method).
    This is a PyTorch implementation of cv2.triangulatePoints.

    Args:
        P1: Projection matrix for camera 1, shape (..., 3, 4)
        P2: Projection matrix for camera 2, shape (..., 3, 4)
        pts1: 2D points from camera 1, shape (..., 2)
        pts2: 2D points from camera 2, shape (..., 2)

    Returns:
        points_3d: Triangulated 3D points, shape (..., 3)
    """
    # Get batch dimensions
    batch_shape = pts1.shape[:-1]
    device = pts1.device
    dtype = pts1.dtype

    # Reshape for batch processing
    P1_flat = P1.reshape(-1, 3, 4)
    P2_flat = P2.reshape(-1, 3, 4)
    pts1_flat = pts1.reshape(-1, 2)
    pts2_flat = pts2.reshape(-1, 2)

    batch_size = pts1_flat.shape[0]

    # Build the system of equations A @ X = 0 for each point
    # For each 2D point observation, we get 2 equations:
    # x * P[2, :] - P[0, :] = 0
    # y * P[2, :] - P[1, :] = 0

    A = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)

    # Camera 1 constraints
    A[:, 0, :] = pts1_flat[:, 0:1] * P1_flat[:, 2, :] - P1_flat[:, 0, :]
    A[:, 1, :] = pts1_flat[:, 1:2] * P1_flat[:, 2, :] - P1_flat[:, 1, :]

    # Camera 2 constraints
    A[:, 2, :] = pts2_flat[:, 0:1] * P2_flat[:, 2, :] - P2_flat[:, 0, :]
    A[:, 3, :] = pts2_flat[:, 1:2] * P2_flat[:, 2, :] - P2_flat[:, 1, :]

    # Solve using SVD: A @ X = 0, solution is last column of V
    # A = U @ S @ V^T, the solution is the last row of V^T (last column of V)
    U, S, Vt = torch.linalg.svd(A)

    # Last row of Vt (corresponding to smallest singular value)
    X_hom = Vt[:, -1, :]  # (batch_size, 4)

    # Convert from homogeneous to 3D coordinates
    # Prevent division by zero
    w = X_hom[:, 3:4]
    w = torch.where(torch.abs(w) < 1e-8, torch.ones_like(w) * 1e-8, w)

    points_3d = X_hom[:, :3] / w

    # Reshape back to original batch shape
    points_3d = points_3d.reshape(*batch_shape, 3)

    return points_3d


def triangulate_point_with_covariance_torch(
    pose_cam1: torch.Tensor,
    pose_cam2: torch.Tensor,
    P1: torch.Tensor,
    P2: torch.Tensor,
    C_joint: torch.Tensor,
    epsilon: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triangulate a single 3D point and propagate covariance (PyTorch version).

    Args:
        pose_cam1: 2D point from camera 1, shape (2,)
        pose_cam2: 2D point from camera 2, shape (2,)
        P1: Projection matrix for camera 1, shape (3, 4)
        P2: Projection matrix for camera 2, shape (3, 4)
        C_joint: 4x4 covariance matrix for the joint's 2D observations
        epsilon: Finite difference step size for Jacobian computation

    Returns:
        tuple: (point_3d, C_3d) - 3D point (3,) and 3x3 covariance matrix
    """
    device = pose_cam1.device
    dtype = pose_cam1.dtype

    # Triangulate the original point
    points_3d = triangulate_points_torch(P1, P2, pose_cam1, pose_cam2)  # (3,)

    # Compute Jacobian using numerical differentiation
    # J[i, j] = d(points_3d[i]) / d(input[j])
    # input = [x1, y1, x2, y2]

    J = torch.zeros(3, 4, device=device, dtype=dtype)

    # Perturb each input dimension
    for i in range(4):
        # Create perturbation vector
        delta = torch.zeros(4, device=device, dtype=dtype)
        delta[i] = epsilon

        # Apply perturbation
        pose_cam1_perturbed = pose_cam1 + delta[:2]
        pose_cam2_perturbed = pose_cam2 + delta[2:]

        # Triangulate with perturbed points
        points_3d_perturbed = triangulate_points_torch(
            P1, P2, pose_cam1_perturbed, pose_cam2_perturbed
        )

        # Compute partial derivative
        J[:, i] = (points_3d_perturbed - points_3d) / epsilon

    # Propagate covariance: C_3d = J @ C_joint @ J^T
    C_3d = J @ C_joint @ J.T  # (3, 3)

    return points_3d, C_3d


def triangulate_points_with_covariance_batched(
    poses_cam1: torch.Tensor,
    poses_cam2: torch.Tensor,
    P1: torch.Tensor,
    P2: torch.Tensor,
    C_joint_list: torch.Tensor,
    epsilon: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triangulate 3D points from two camera views with covariance propagation (batched).

    Args:
        poses_cam1: 2D keypoints from camera 1, shape (B, N, 2)
        poses_cam2: 2D keypoints from camera 2, shape (B, N, 2)
        P1: Projection matrix for camera 1, shape (3, 4) or (B, 3, 4)
        P2: Projection matrix for camera 2, shape (3, 4) or (B, 3, 4)
        C_joint_list: Covariance matrices for each joint, shape (B, N, 4, 4)
        epsilon: Finite difference step size for Jacobian computation

    Returns:
        tuple: (points_3d, C_3d_all)
            - points_3d: 3D points, shape (B, N, 3)
            - C_3d_all: 3D covariance matrices, shape (B, N, 3, 3)
    """
    device = poses_cam1.device
    dtype = poses_cam1.dtype
    batch_size, num_joints = poses_cam1.shape[:2]

    # Expand projection matrices if needed
    if P1.ndim == 2:
        P1 = P1.unsqueeze(0).expand(batch_size, -1, -1)
    if P2.ndim == 2:
        P2 = P2.unsqueeze(0).expand(batch_size, -1, -1)

    # Triangulate all points at once
    # Reshape to (B*N, 2) for batch processing
    poses_cam1_flat = poses_cam1.reshape(batch_size * num_joints, 2)
    poses_cam2_flat = poses_cam2.reshape(batch_size * num_joints, 2)
    P1_flat = P1.unsqueeze(1).expand(-1, num_joints, -1, -1).reshape(batch_size * num_joints, 3, 4)
    P2_flat = P2.unsqueeze(1).expand(-1, num_joints, -1, -1).reshape(batch_size * num_joints, 3, 4)

    # Triangulate original points
    points_3d_flat = triangulate_points_torch(P1_flat, P2_flat, poses_cam1_flat, poses_cam2_flat)  # (B*N, 3)
    points_3d = points_3d_flat.reshape(batch_size, num_joints, 3)

    # Compute Jacobians using numerical differentiation
    # For efficiency, compute all perturbations at once
    J = torch.zeros(batch_size, num_joints, 3, 4, device=device, dtype=dtype)

    # Perturb each of the 4 input dimensions
    for i in range(4):
        # Create perturbation: add epsilon to dimension i
        delta = torch.zeros(batch_size, num_joints, 4, device=device, dtype=dtype)
        delta[:, :, i] = epsilon

        # Apply perturbation to appropriate camera
        poses_cam1_perturbed = poses_cam1 + delta[:, :, :2]
        poses_cam2_perturbed = poses_cam2 + delta[:, :, 2:]

        # Flatten for triangulation
        poses_cam1_perturbed_flat = poses_cam1_perturbed.reshape(batch_size * num_joints, 2)
        poses_cam2_perturbed_flat = poses_cam2_perturbed.reshape(batch_size * num_joints, 2)

        # Triangulate with perturbed points
        points_3d_perturbed_flat = triangulate_points_torch(
            P1_flat, P2_flat, poses_cam1_perturbed_flat, poses_cam2_perturbed_flat
        )
        points_3d_perturbed = points_3d_perturbed_flat.reshape(batch_size, num_joints, 3)

        # Compute partial derivatives
        J[:, :, :, i] = (points_3d_perturbed - points_3d) / epsilon

    # Propagate covariance: C_3d = J @ C_joint @ J^T for each point
    # J: (B, N, 3, 4), C_joint_list: (B, N, 4, 4)
    # Result: (B, N, 3, 3)

    # Compute J @ C_joint: (B, N, 3, 4) @ (B, N, 4, 4) = (B, N, 3, 4)
    J_C = torch.matmul(J, C_joint_list)  # (B, N, 3, 4)

    # Compute J @ C_joint @ J^T: (B, N, 3, 4) @ (B, N, 4, 3) = (B, N, 3, 3)
    C_3d_all = torch.matmul(J_C, J.transpose(-2, -1))  # (B, N, 3, 3)

    return points_3d, C_3d_all
