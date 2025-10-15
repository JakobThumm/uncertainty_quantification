"""
GPU-accelerated utilities for single-frame pose estimation inference

This module provides PyTorch-based GPU operations to speed up image preprocessing
for real-time pose estimation, replacing slow CPU-based PIL and NumPy operations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import jax.numpy as jnp
import cv2
from PIL import Image
from typing import Tuple, List, Optional

from human_pose_pipeline.pose_estimation.h36m_settings import (
    YOLO_IMAGE_SIZE,
    TRANSFORM_IMAGE_SIZE,
    NORMALIZATION_OFFSET
)


def resize_image_gpu(pil_image: Image.Image, target_size: Tuple[int, int] = YOLO_IMAGE_SIZE,
                     device: str = 'cuda') -> Tuple[Image.Image, Tuple[int, int], Tuple[float, float]]:
    """
    GPU-accelerated image resizing using PyTorch.

    Args:
        pil_image: Input PIL image
        target_size: Target size (width, height)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        tuple: (resized_image_pil, original_dimensions, scale_factors)
    """
    # Get original dimensions
    original_width, original_height = pil_image.size
    target_width, target_height = target_size

    # Convert PIL to tensor: (H, W, C) -> (1, C, H, W)
    img_np = np.array(pil_image)
    img_tensor = torch.from_numpy(img_np).to(device).float()

    # Permute to (C, H, W) and add batch dimension
    if len(img_tensor.shape) == 2:  # Grayscale
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:  # RGB
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # Resize using bilinear interpolation (similar to LANCZOS but faster on GPU)
    resized_tensor = F.interpolate(
        img_tensor,
        size=(target_height, target_width),
        mode='bilinear',
        align_corners=False
    )

    # Convert back to PIL Image
    resized_np = resized_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    resized_image = Image.fromarray(resized_np)

    # Calculate scale factors
    scale_x = original_width / target_width
    scale_y = original_height / target_height

    return resized_image, (original_width, original_height), (scale_x, scale_y)


def preprocess_bbox_image_gpu(
    resized_image_np: np.ndarray,
    bbox: List[float],
    output_size: Tuple[int, int] = (TRANSFORM_IMAGE_SIZE[0], TRANSFORM_IMAGE_SIZE[1]),
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated bounding box preprocessing using PyTorch.

    This function replicates the behavior of transform_utils.preprocess_image_with_bbox()
    but uses GPU-accelerated operations.

    Args:
        resized_image_np: Resized image as numpy array (H, W, 3)
        bbox: Bounding box [xmin, ymin, xmax, ymax]
        output_size: Output image size (width, height)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        tuple: (preprocessed_image, center, scale, trans, processed_bbox)
            - preprocessed_image: (1, 3, H, W) normalized image ready for model
            - center: (2,) center coordinates
            - scale: (2,) scale values
            - trans: (2, 3) affine transformation matrix
            - processed_bbox: (4,) processed bounding box
    """
    # Convert image to torch tensor
    img_tensor = torch.from_numpy(resized_image_np).to(device).float()

    # Normalize to [0, 1] if needed
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0

    # Permute to (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # Compute center and scale from bbox
    xmin, ymin, xmax, ymax = bbox
    bbox_tensor = torch.tensor([xmin, ymin, xmax, ymax], device=device, dtype=torch.float32)

    aspect_ratio = output_size[0] / output_size[1]  # width / height
    pixel_std = 1.0
    scale_mult = 1.25

    w = bbox_tensor[2] - bbox_tensor[0]
    h = bbox_tensor[3] - bbox_tensor[1]

    center_x = bbox_tensor[0] + w * 0.5
    center_y = bbox_tensor[1] + h * 0.5
    center = torch.stack([center_x, center_y])

    # Adjust size based on aspect ratio
    w_adjusted = torch.where(w > aspect_ratio * h, w, h * aspect_ratio)
    h_adjusted = torch.where(w > aspect_ratio * h, w / aspect_ratio, h)

    scale = torch.stack([w_adjusted / pixel_std, h_adjusted / pixel_std])

    # Apply scale multiplier
    if center[0] != -1:
        scale = scale * scale_mult

    # Get affine transformation matrix (need to use CPU for cv2)
    center_np = center.cpu().numpy()
    scale_np = scale.cpu().numpy()

    # Compute affine transform using cv2
    trans = _get_affine_transform_cv2(center_np, scale_np, output_size)
    trans_tensor = torch.from_numpy(trans).to(device).float()

    # Apply affine transformation to image
    img_preprocessed = _apply_affine_transform_gpu(img_tensor, trans_tensor, output_size)

    # Apply RegressFlow normalization
    normalization_offset = torch.tensor(NORMALIZATION_OFFSET, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    img_preprocessed = img_preprocessed + normalization_offset

    # Compute processed bbox
    processed_bbox = _center_scale_to_box(center_np, scale_np)

    # Convert to JAX array for compatibility with existing code
    img_preprocessed_np = img_preprocessed.cpu().numpy()
    img_preprocessed_jax = jnp.array(img_preprocessed_np, dtype=jnp.float32)

    return img_preprocessed_jax, center_np, scale_np, trans, processed_bbox


def _get_affine_transform_cv2(center: np.ndarray, scale: np.ndarray,
                               output_size: Tuple[int, int], rot: float = 0) -> np.ndarray:
    """
    Get affine transformation matrix (CPU implementation using cv2).

    This replicates transform_utils.get_affine_transform() behavior.
    """
    if not isinstance(scale, np.ndarray):
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w, dst_h = output_size

    rot_rad = np.pi * rot / 180
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    # Source direction
    src_dir = np.array([0, src_w * -0.5], dtype=np.float32)
    src_dir_rotated = np.array([
        src_dir[0] * cs - src_dir[1] * sn,
        src_dir[0] * sn + src_dir[1] * cs
    ], dtype=np.float32)

    # Destination direction
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    # Define 3 source and destination points
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir_rotated

    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = [dst_w * 0.5, dst_h * 0.5] + dst_dir

    # Get third point (perpendicular)
    direct = src[0, :] - src[1, :]
    src[2, :] = src[1, :] + np.array([-direct[1], direct[0]], dtype=np.float32)

    direct = dst[0, :] - dst[1, :]
    dst[2, :] = dst[1, :] + np.array([-direct[1], direct[0]], dtype=np.float32)

    # Compute affine transformation
    trans = cv2.getAffineTransform(src, dst)

    return trans


def _apply_affine_transform_gpu(img_tensor: torch.Tensor, trans: torch.Tensor,
                                 output_size: Tuple[int, int]) -> torch.Tensor:
    """
    Apply affine transformation to image tensor on GPU.

    Args:
        img_tensor: (1, C, H, W) image tensor
        trans: (2, 3) affine transformation matrix (OpenCV format)
        output_size: (width, height) output size

    Returns:
        transformed: (1, C, output_h, output_w) transformed image
    """
    device = img_tensor.device
    input_h, input_w = img_tensor.shape[2], img_tensor.shape[3]
    output_w, output_h = output_size

    # Convert OpenCV affine matrix to PyTorch format
    # Invert the transformation
    trans_np = trans.cpu().numpy()
    trans_inv_np = cv2.invertAffineTransform(trans_np)
    trans_inv = torch.from_numpy(trans_inv_np).to(device).float()

    # Convert to homogeneous form
    trans_inv_hom = torch.eye(3, device=device, dtype=torch.float32)
    trans_inv_hom[:2, :] = trans_inv

    # Scale transformations for normalized coordinates
    scale_out = torch.tensor([[output_w/2.0, 0, output_w/2.0],
                              [0, output_h/2.0, output_h/2.0],
                              [0, 0, 1]], device=device, dtype=torch.float32)

    scale_in = torch.tensor([[2.0/input_w, 0, -1],
                             [0, 2.0/input_h, -1],
                             [0, 0, 1]], device=device, dtype=torch.float32)

    # Combine transformations
    trans_pt_hom = torch.mm(torch.mm(scale_in, trans_inv_hom), scale_out)
    trans_pt = trans_pt_hom[:2, :].unsqueeze(0)  # (1, 2, 3)

    # Create sampling grid and apply transformation
    grid = F.affine_grid(trans_pt, [1, img_tensor.shape[1], output_h, output_w],
                         align_corners=False)
    transformed = F.grid_sample(img_tensor, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)

    return transformed


def _center_scale_to_box(center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Convert center and scale to bounding box coordinates."""
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    return np.array([xmin, ymin, xmax, ymax])


def extract_bounding_box_images_gpu(
    full_image: Image.Image,
    person_boxes: List[List[float]],
    scale_factors: Tuple[float, float],
    resized_image_np: np.ndarray,
    device: str = 'cuda'
) -> List[dict]:
    """
    GPU-accelerated bounding box extraction for all detected persons.

    Args:
        full_image: Original PIL image (not used, kept for compatibility)
        person_boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        scale_factors: (scale_x, scale_y) from resize operation
        resized_image_np: Resized image as numpy array (H, W, 3)
        device: Device to use ('cuda' or 'cpu')

    Returns:
        List of dictionaries with preprocessed bounding box data
    """
    if not person_boxes:
        return []

    bounding_box_images = []
    scale_x, scale_y = scale_factors

    for bbox in person_boxes:
        # Preprocess this bounding box on GPU
        img_preprocessed, center, scale, trans, processed_bbox = preprocess_bbox_image_gpu(
            resized_image_np, bbox, device=device
        )

        bbox_struct = {
            'scale_factors_yolo': scale_factors,
            'bbox': bbox,
            'image': img_preprocessed,  # Already in JAX-compatible format (1, 3, H, W)
            'center': center,
            'scale': scale,
            'trans': trans
        }
        bounding_box_images.append(bbox_struct)

    return bounding_box_images
