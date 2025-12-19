"""
GPU-accelerated utilities for single-frame pose estimation inference using JAX

This module provides JAX-based GPU operations to speed up image preprocessing
for real-time pose estimation, using JIT compilation for maximum performance.
"""

import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Optional, Dict

from human_pose_pipeline.pose_estimation.h36m_settings import (
    YOLO_IMAGE_SIZE,
    TRANSFORM_IMAGE_SIZE,
    NORMALIZATION_OFFSET,
)


@jax.jit
def get_affine_transform_batch(src: jnp.ndarray, dst: jnp.ndarray) -> jnp.ndarray:
    """
    Compute affine transformation matrices for batched source and destination points.

    Args:
        src: Source points, shape (B, 3, 2)
        dst: Destination points, shape (B, 3, 2)

    Returns:
        Affine transformation matrices, shape (B, 2, 3)
    """
    src = src.astype(jnp.float32)
    dst = dst.astype(jnp.float32)

    B = src.shape[0]

    # Extract x, y
    x = src[:, :, 0]  # (B, 3)
    y = src[:, :, 1]  # (B, 3)

    # Build A matrices (B, 6, 6)
    A = jnp.zeros((B, 6, 6), dtype=jnp.float32)

    # Fill first rows: [x y 1 0 0 0]
    A = A.at[:, 0::2, 0].set(x)
    A = A.at[:, 0::2, 1].set(y)
    A = A.at[:, 0::2, 2].set(1)

    # Fill second rows: [0 0 0 x y 1]
    A = A.at[:, 1::2, 3].set(x)
    A = A.at[:, 1::2, 4].set(y)
    A = A.at[:, 1::2, 5].set(1)

    # Build b (target)
    b = dst.reshape(B, 6).astype(jnp.float32)

    # Solve A x = b  (B, 6)
    X = jnp.linalg.solve(A, b[:, :, None]).squeeze(-1)

    # Convert to affine matrix (B, 2, 3)
    M = X.reshape(B, 2, 3)

    return M


@jax.jit
def invert_affine_transform_batch(M: jnp.ndarray) -> jnp.ndarray:
    """
    Invert batched affine transformation matrices.

    Args:
        M: Affine transformation matrices, shape (B, 2, 3)

    Returns:
        Inverted affine transformation matrices, shape (B, 2, 3)
    """
    M = M.astype(jnp.float32)

    a = M[:, 0, 0]
    b = M[:, 0, 1]
    tx = M[:, 0, 2]
    c = M[:, 1, 0]
    d = M[:, 1, 1]
    ty = M[:, 1, 2]

    det = a * d - b * c
    inv_det = 1.0 / det

    A11 = d * inv_det
    A12 = -b * inv_det
    A21 = -c * inv_det
    A22 = a * inv_det

    b1 = -(A11 * tx + A12 * ty)
    b2 = -(A21 * tx + A22 * ty)

    iM = jnp.stack(
        [
            jnp.stack([A11, A12, b1], axis=-1),
            jnp.stack([A21, A22, b2], axis=-1),
        ],
        axis=1,
    )

    return iM


@jax.jit
def cv2_transform(src: jnp.ndarray, M: jnp.ndarray, shift: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    JAX implementation of cv2.transform for batched vector transformation.

    Args:
        src: Input vectors, shape (B, N, scn)
        M: Transformation matrix, shape (B, dcn, scn+1) or (dcn, scn+1) or (B, dcn, scn) or (dcn, scn)
             If last dimension is scn+1, it's treated as [linear | shift] like cv2.transform
             If last dimension is scn, it's just the linear transformation
        shift: Optional shift vector, shape (B, dcn) or (dcn,)

    Returns:
        Transformed vectors, shape (B, N, dcn)
    """
    src = src.astype(jnp.float32)

    # Ensure batching
    if M.ndim == 2:  # (dcn, scn) or (dcn, scn+1)
        M = M[None, :, :]  # (1, dcn, scn) or (1, dcn, scn+1)

    B, N, scn = src.shape
    Bm, dcn, last_dim = M.shape

    # Check if M includes shift (like cv2.transform format: 2x3 matrix)
    if last_dim == scn + 1:
        # M is (B, dcn, scn+1) - split into linear part and shift
        M_linear = M[:, :, :scn]  # (B, dcn, scn)
        shift = M[:, :, scn]  # (B, dcn)
    else:
        # M is (B, dcn, scn) - just linear transformation
        M_linear = M

    # Broadcast if needed
    if Bm == 1 and B > 1:
        M_linear = jnp.broadcast_to(M_linear, (B, dcn, scn))

    # Linear transform: dst = src @ M^T
    dst = jnp.matmul(src, M_linear.transpose(0, 2, 1))  # (B, N, dcn)

    if shift is not None:
        if shift.ndim == 1:
            shift = shift[None, :]  # (1, dcn)

        Bs, dcn_s = shift.shape

        if Bs == 1 and B > 1:
            shift = jnp.broadcast_to(shift, (B, dcn))

        dst = dst + shift[:, None, :]  # broadcast to (B, N, dcn)

    return dst


@jax.jit
def transform_predictions_to_original_space_batched(
    pred_joints_normalized: jnp.ndarray,
    trans: jnp.ndarray,
    scale_x: float,
    scale_y: float,
    uncertainties: Optional[jnp.ndarray] = None,
    covariance: Optional[jnp.ndarray] = None
) -> Dict[str, jnp.ndarray]:
    """
    Transform model predictions from normalized coordinates back to original image space.
    (Not JIT-compiled due to optional parameters and conditional logic)

    This function performs the full reverse transformation pipeline:
    1. Convert normalized coords (-0.5 to 0.5) to pixel coords in preprocessed image
    2. Apply inverse affine transformation to get coords in resized image
    3. Scale coords to original image dimensions
    4. Scale uncertainties appropriately if provided

    Args:
        pred_joints_normalized: Joint coordinates in normalized space (-0.5 to 0.5), shape (B, N, 2)
        trans: Affine transformation matrix used during preprocessing (B, 3, 2)
        scale_x: Scale factor from resized to original width
        scale_y: Scale factor from resized to original height
        uncertainties: Optional uncertainty values, shape (B, N, 2)
        covariance: Optional covariance values, shape (B, N,)

    Returns:
        dict: Dictionary containing:
            - 'keypoints': Joint coordinates in original image space
            - 'uncertainties': Scaled uncertainties (if provided)
            - 'covariance': Scaled covariance (if provided)
    """
    # Step 1: Convert normalized coordinates to pixel coordinates in preprocessed image
    img_height, img_width = TRANSFORM_IMAGE_SIZE[1], TRANSFORM_IMAGE_SIZE[0]
    pred_joints_pixel = jnp.zeros_like(pred_joints_normalized)
    pred_joints_pixel = pred_joints_pixel.at[:, :, 0].set((pred_joints_normalized[:, :, 0] + 0.5) * img_width)
    pred_joints_pixel = pred_joints_pixel.at[:, :, 1].set((pred_joints_normalized[:, :, 1] + 0.5) * img_height)

    # Step 2: Apply inverse affine transformation
    trans_inv = invert_affine_transform_batch(trans)

    pred_joints_resized = cv2_transform(pred_joints_pixel, trans_inv)

    # Step 3: Scale to original image dimensions
    pred_joints_original = pred_joints_resized
    pred_joints_original = pred_joints_original.at[:, :, 0].set(pred_joints_resized[:, :, 0] * scale_x)
    pred_joints_original = pred_joints_original.at[:, :, 1].set(pred_joints_resized[:, :, 1] * scale_y)

    result = {'keypoints': pred_joints_original}

    # Step 4: Transform uncertainties if provided
    if uncertainties is not None:
        # Scale uncertainties to preprocessed image dimensions
        uncertainties_pixel = jnp.zeros_like(uncertainties)
        uncertainties_pixel = uncertainties_pixel.at[:, :, 0].set(uncertainties[:, :, 0] * img_width)
        uncertainties_pixel = uncertainties_pixel.at[:, :, 1].set(uncertainties[:, :, 1] * img_height)

        # Scale to original image dimensions
        uncertainties_original = jnp.zeros_like(uncertainties_pixel)
        uncertainties_original = uncertainties_original.at[:, :, 0].set(uncertainties_pixel[:, :, 0] * scale_x)
        uncertainties_original = uncertainties_original.at[:, :, 1].set(uncertainties_pixel[:, :, 1] * scale_y)

        result['uncertainties'] = uncertainties_original

        # Transform covariance if provided
        if covariance is not None:
            covariance_scaled = covariance * img_width * img_height
            covariance_original = covariance_scaled * scale_x * scale_y
            result['covariance'] = covariance_original

    return result


@jax.jit
def _get_affine_transform_jax(
    center: jnp.ndarray, scale: jnp.ndarray, output_size: Tuple[int, int], rot: float = 0
) -> jnp.ndarray:
    """Get affine transformation matrices (GPU implementation using JAX).

    Args:
        center: bounding box centers [B, 2]
        scale: bounding box width, height [B, 2]
        output_size: width, height (int, int)
        rot: rotation angle in degrees (float)

    Returns:
        affine transformation matrix [B, 2, 3]
    """
    B = center.shape[0]
    src_w = scale[:, 0]
    dst_w, dst_h = output_size

    rot_rad = jnp.pi * rot / 180.0
    sn, cs = jnp.sin(rot_rad), jnp.cos(rot_rad)

    # Source direction
    src_dir = jnp.zeros([B, 2], dtype=jnp.float32)
    src_dir = src_dir.at[:, 1].set(src_w * -0.5)
    src_dir_rotated = jnp.stack(
        [src_dir[:, 0] * cs - src_dir[:, 1] * sn, src_dir[:, 0] * sn + src_dir[:, 1] * cs],
        axis=1
    )

    # Destination direction
    dst_dir = jnp.zeros([B, 2], dtype=jnp.float32)
    dst_dir = dst_dir.at[:, 1].set(dst_w * -0.5)

    # Define 3 source and destination points
    src = jnp.zeros([B, 3, 2], dtype=jnp.float32)
    dst = jnp.zeros([B, 3, 2], dtype=jnp.float32)

    src = src.at[:, 0, :].set(center)
    src = src.at[:, 1, :].set(center + src_dir_rotated)

    dst = dst.at[:, 0, 0].set(dst_w * 0.5)
    dst = dst.at[:, 0, 1].set(dst_h * 0.5)
    dst = dst.at[:, 1, 0].set(dst_dir[:, 0] + dst_w * 0.5)
    dst = dst.at[:, 1, 1].set(dst_dir[:, 1] + dst_h * 0.5)

    # Get third point (perpendicular)
    direct_src = src[:, 0, :] - src[:, 1, :]
    tangent_direct_src = jnp.stack([-direct_src[:, 1], direct_src[:, 0]], axis=1)
    src = src.at[:, 2, :].set(src[:, 1, :] + tangent_direct_src)

    direct_dst = dst[:, 0, :] - dst[:, 1, :]
    tangent_direct_dst = jnp.stack([-direct_dst[:, 1], direct_dst[:, 0]], axis=1)
    dst = dst.at[:, 2, :].set(dst[:, 1, :] + tangent_direct_dst)

    # Compute affine transformation
    trans = get_affine_transform_batch(src, dst)

    return trans


@jax.jit
def _apply_affine_transform_gpu(
    img_tensor: jnp.ndarray, trans: jnp.ndarray, output_size: Tuple[int, int]
) -> jnp.ndarray:
    """
    Apply affine transformation to image tensor on GPU using JAX.
    (Not JIT-compiled due to dynamic output_size)

    Args:
        img_tensor: (1, C, H, W) image tensor
        trans: (2, 3) affine transformation matrix (OpenCV format)
        output_size: (width, height) output size

    Returns:
        transformed: (1, C, output_h, output_w) transformed image
    """
    input_h, input_w = img_tensor.shape[2], img_tensor.shape[3]
    output_w, output_h = output_size

    # Invert transformation using JAX operations
    # Create homogeneous form
    trans_hom = jnp.eye(3, dtype=jnp.float32)
    trans_hom = trans_hom.at[:2, :].set(trans)

    # Invert
    trans_inv_hom = jnp.linalg.inv(trans_hom)
    trans_inv = trans_inv_hom[:2, :]

    # Create sampling grid
    y_coords = jnp.arange(output_h, dtype=jnp.float32)
    x_coords = jnp.arange(output_w, dtype=jnp.float32)
    x_grid, y_grid = jnp.meshgrid(x_coords, y_coords)

    # Flatten and stack coordinates
    coords = jnp.stack([x_grid.flatten(), y_grid.flatten(), jnp.ones(output_h * output_w, dtype=jnp.float32)], axis=0)

    # Apply inverse transformation
    src_coords = jnp.dot(trans_inv, coords)  # (2, H*W)
    src_x = src_coords[0, :].reshape(output_h, output_w)
    src_y = src_coords[1, :].reshape(output_h, output_w)

    # Bilinear interpolation
    transformed = _bilinear_sample(img_tensor, src_x, src_y)

    return transformed


@jax.jit
def _apply_affine_transform_batched_jax(
    img_tensor: jnp.ndarray, trans: jnp.ndarray, output_size: Tuple[int, int]
) -> jnp.ndarray:
    """
    Apply affine transformation to batched image tensors on GPU using JAX.
    (Not JIT-compiled due to dynamic output_size)

    Args:
        img_tensor: (B, C, H, W) image tensor
        trans: (B, 2, 3) affine transformation matrices (OpenCV format)
        output_size: (width, height) output size

    Returns:
        transformed: (B, C, output_h, output_w) transformed images
    """
    B, C, input_h, input_w = img_tensor.shape
    output_w, output_h = output_size

    # Invert transformations
    # Create homogeneous form
    trans_inv_hom = jnp.eye(3, dtype=jnp.float32)[None, :, :].repeat(B, axis=0)

    # Invert each transformation using the batch invert function
    trans_inv = invert_affine_transform_batch(trans)

    # Create sampling grid
    y_coords = jnp.arange(output_h, dtype=jnp.float32)
    x_coords = jnp.arange(output_w, dtype=jnp.float32)
    x_grid, y_grid = jnp.meshgrid(x_coords, y_coords)

    # Flatten and stack coordinates (add homogeneous coordinate)
    coords = jnp.stack([x_grid.flatten(), y_grid.flatten(), jnp.ones(output_h * output_w, dtype=jnp.float32)], axis=0)
    coords = coords[None, :, :].repeat(B, axis=0)  # (B, 3, H*W)

    # Apply inverse transformation for each batch
    # trans_inv is (B, 2, 3), coords is (B, 3, H*W)
    src_coords = jnp.matmul(trans_inv, coords)  # (B, 2, H*W)
    src_x = src_coords[:, 0, :].reshape(B, output_h, output_w)
    src_y = src_coords[:, 1, :].reshape(B, output_h, output_w)

    # Bilinear interpolation for each batch
    def sample_single(img, sx, sy):
        return _bilinear_sample(img[None, :, :, :], sx, sy)[0]

    transformed = jax.vmap(sample_single)(img_tensor, src_x, src_y)

    return transformed


@jax.jit
def _bilinear_sample(img: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Bilinear sampling for image transformation.
    (Not JIT-compiled due to dynamic shapes)

    Args:
        img: (1, C, H, W) image tensor
        x: (H_out, W_out) x-coordinates to sample
        y: (H_out, W_out) y-coordinates to sample

    Returns:
        sampled: (1, C, H_out, W_out) sampled image
    """
    C, H, W = img.shape[1], img.shape[2], img.shape[3]
    H_out, W_out = x.shape

    # Clamp coordinates to image bounds
    x = jnp.clip(x, 0, W - 1)
    y = jnp.clip(y, 0, H - 1)

    # Get floor coordinates
    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, W - 1)
    y1 = jnp.clip(y0 + 1, 0, H - 1)

    # Get fractional parts
    fx = x - x0
    fy = y - y0

    # Gather values at four corners
    img_flat = img[0]  # (C, H, W)

    # Compute linear indices
    idx_00 = y0 * W + x0
    idx_01 = y0 * W + x1
    idx_10 = y1 * W + x0
    idx_11 = y1 * W + x1

    # Reshape for gathering
    img_reshaped = img_flat.reshape(C, -1)  # (C, H*W)

    # Gather and interpolate - vectorized version
    def interpolate_channel(c):
        v00 = img_reshaped[c, idx_00]
        v01 = img_reshaped[c, idx_01]
        v10 = img_reshaped[c, idx_10]
        v11 = img_reshaped[c, idx_11]

        # Bilinear interpolation
        v0 = v00 * (1 - fx) + v01 * fx
        v1 = v10 * (1 - fx) + v11 * fx
        return v0 * (1 - fy) + v1 * fy

    result = jax.vmap(interpolate_channel)(jnp.arange(C))

    return result[None, :, :, :]


@jax.jit
def _center_scale_to_box(center: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Convert center and scale to bounding box coordinates."""
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    return jnp.stack([xmin, ymin, xmax, ymax], axis=1)


@jax.jit
def _center_scale_to_box_batched(center: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Convert center and scale to bounding box coordinates.

    Args:
        center: [B, 2]
        scale: [B, 2]
    Returns:
        box: [B, 4]
    """
    pixel_std = 1.0
    w = scale[:, 0] * pixel_std
    h = scale[:, 1] * pixel_std
    xmin = center[:, 0] - w * 0.5
    ymin = center[:, 1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    return jnp.stack([xmin, ymin, xmax, ymax], axis=1)


@jax.jit
def create_joint_covariance_batched(
    mapped_uncertainty_cam1: jnp.ndarray,
    mapped_covariance_cam1: jnp.ndarray,
    mapped_uncertainty_cam2: jnp.ndarray,
    mapped_covariance_cam2: jnp.ndarray,
    cross_covariance: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Create 4x4 joint covariance matrices from two camera observations (batched).

    Args:
        mapped_uncertainty_cam1: (std_x, std_y) for camera 1, shape (B, N, 2)
        mapped_covariance_cam1: covariance xy for camera 1, shape (B, N)
        mapped_uncertainty_cam2: (std_x, std_y) for camera 2, shape (B, N, 2)
        mapped_covariance_cam2: covariance xy for camera 2, shape (B, N)
        cross_covariance: 2x2 cross-covariance matrices between cameras (optional), shape (B, N, 2, 2)

    Returns:
        jnp.ndarray: 4x4 joint covariance matrices, shape (B, N, 4, 4)
    """
    dtype = mapped_uncertainty_cam1.dtype
    batch_size, num_joints = mapped_uncertainty_cam1.shape[:2]

    # Initialize C_joint as (B, N, 4, 4)
    C_joint = jnp.zeros((batch_size, num_joints, 4, 4), dtype=dtype)

    # Build C1 (2x2 covariance for camera 1)
    # C1 = [[var_x1, cov_xy1],
    #       [cov_xy1, var_y1]]
    var_x1 = mapped_uncertainty_cam1[:, :, 0] ** 2  # (B, N)
    var_y1 = mapped_uncertainty_cam1[:, :, 1] ** 2  # (B, N)
    cov_xy1 = mapped_covariance_cam1  # (B, N)

    C_joint = C_joint.at[:, :, 0, 0].set(var_x1)
    C_joint = C_joint.at[:, :, 0, 1].set(cov_xy1)
    C_joint = C_joint.at[:, :, 1, 0].set(cov_xy1)
    C_joint = C_joint.at[:, :, 1, 1].set(var_y1)

    # Build C2 (2x2 covariance for camera 2)
    # C2 = [[var_x2, cov_xy2],
    #       [cov_xy2, var_y2]]
    var_x2 = mapped_uncertainty_cam2[:, :, 0] ** 2  # (B, N)
    var_y2 = mapped_uncertainty_cam2[:, :, 1] ** 2  # (B, N)
    cov_xy2 = mapped_covariance_cam2  # (B, N)

    C_joint = C_joint.at[:, :, 2, 2].set(var_x2)
    C_joint = C_joint.at[:, :, 2, 3].set(cov_xy2)
    C_joint = C_joint.at[:, :, 3, 2].set(cov_xy2)
    C_joint = C_joint.at[:, :, 3, 3].set(var_y2)

    # Build C12 (2x2 cross-covariance between cameras)
    if cross_covariance is not None:
        # cross_covariance shape: (B, N, 2, 2)
        C_joint = C_joint.at[:, :, 0:2, 2:4].set(cross_covariance)
        C_joint = C_joint.at[:, :, 2:4, 0:2].set(jnp.swapaxes(cross_covariance, -2, -1))
    # else: C12 is already zeros

    return C_joint


@jax.jit
def triangulate_points(
    P1: jnp.ndarray,
    P2: jnp.ndarray,
    pts1: jnp.ndarray,
    pts2: jnp.ndarray
) -> jnp.ndarray:
    """
    Triangulate 3D points from two camera views using JAX (DLT method).
    This is a JAX implementation of cv2.triangulatePoints.

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

    A = jnp.zeros((batch_size, 4, 4), dtype=dtype)

    # Camera 1 constraints
    A = A.at[:, 0, :].set(pts1_flat[:, 0:1] * P1_flat[:, 2, :] - P1_flat[:, 0, :])
    A = A.at[:, 1, :].set(pts1_flat[:, 1:2] * P1_flat[:, 2, :] - P1_flat[:, 1, :])

    # Camera 2 constraints
    A = A.at[:, 2, :].set(pts2_flat[:, 0:1] * P2_flat[:, 2, :] - P2_flat[:, 0, :])
    A = A.at[:, 3, :].set(pts2_flat[:, 1:2] * P2_flat[:, 2, :] - P2_flat[:, 1, :])

    # Solve using SVD: A @ X = 0, solution is last column of V
    # A = U @ S @ V^T, the solution is the last row of V^T (last column of V)
    U, S, Vt = jnp.linalg.svd(A)

    # Last row of Vt (corresponding to smallest singular value)
    X_hom = Vt[:, -1, :]  # (batch_size, 4)

    # Convert from homogeneous to 3D coordinates
    # Prevent division by zero
    w = X_hom[:, 3:4]
    w = jnp.where(jnp.abs(w) < 1e-8, jnp.ones_like(w) * 1e-8, w)

    points_3d = X_hom[:, :3] / w

    # Reshape back to original batch shape
    points_3d = points_3d.reshape(*batch_shape, 3)

    return points_3d


def triangulate_points_with_covariance_batched(
    poses_cam1: jnp.ndarray,
    poses_cam2: jnp.ndarray,
    P1: jnp.ndarray,
    P2: jnp.ndarray,
    C_joint_list: jnp.ndarray,
    epsilon: float = 1e-3
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    dtype = poses_cam1.dtype
    batch_size, num_joints = poses_cam1.shape[:2]

    # Expand projection matrices if needed
    if P1.ndim == 2:
        P1 = jnp.broadcast_to(P1[None, :, :], (batch_size, 3, 4))
    if P2.ndim == 2:
        P2 = jnp.broadcast_to(P2[None, :, :], (batch_size, 3, 4))

    # Triangulate all points at once
    # Reshape to (B*N, 2) for batch processing
    poses_cam1_flat = poses_cam1.reshape(batch_size * num_joints, 2)
    poses_cam2_flat = poses_cam2.reshape(batch_size * num_joints, 2)
    P1_flat = jnp.broadcast_to(P1[:, None, :, :], (batch_size, num_joints, 3, 4)).reshape(batch_size * num_joints, 3, 4)
    P2_flat = jnp.broadcast_to(P2[:, None, :, :], (batch_size, num_joints, 3, 4)).reshape(batch_size * num_joints, 3, 4)

    # Triangulate original points
    points_3d_flat = triangulate_points(P1_flat, P2_flat, poses_cam1_flat, poses_cam2_flat)  # (B*N, 3)
    points_3d = points_3d_flat.reshape(batch_size, num_joints, 3)

    # Compute Jacobians using numerical differentiation
    # For efficiency, compute all perturbations at once
    J = jnp.zeros((batch_size, num_joints, 3, 4), dtype=dtype)

    # Perturb each of the 4 input dimensions
    for i in range(4):
        # Create perturbation: add epsilon to dimension i
        delta = jnp.zeros((batch_size, num_joints, 4), dtype=dtype)
        delta = delta.at[:, :, i].set(epsilon)

        # Apply perturbation to appropriate camera
        poses_cam1_perturbed = poses_cam1 + delta[:, :, :2]
        poses_cam2_perturbed = poses_cam2 + delta[:, :, 2:]

        # Flatten for triangulation
        poses_cam1_perturbed_flat = poses_cam1_perturbed.reshape(batch_size * num_joints, 2)
        poses_cam2_perturbed_flat = poses_cam2_perturbed.reshape(batch_size * num_joints, 2)

        # Triangulate with perturbed points
        points_3d_perturbed_flat = triangulate_points(
            P1_flat, P2_flat, poses_cam1_perturbed_flat, poses_cam2_perturbed_flat
        )
        points_3d_perturbed = points_3d_perturbed_flat.reshape(batch_size, num_joints, 3)

        # Compute partial derivatives
        J = J.at[:, :, :, i].set((points_3d_perturbed - points_3d) / epsilon)

    # Propagate covariance: C_3d = J @ C_joint @ J^T for each point
    # J: (B, N, 3, 4), C_joint_list: (B, N, 4, 4)
    # Result: (B, N, 3, 3)

    # Compute J @ C_joint: (B, N, 3, 4) @ (B, N, 4, 4) = (B, N, 3, 4)
    J_C = jnp.matmul(J, C_joint_list)  # (B, N, 3, 4)

    # Compute J @ C_joint @ J^T: (B, N, 3, 4) @ (B, N, 4, 3) = (B, N, 3, 3)
    C_3d_all = jnp.matmul(J_C, jnp.swapaxes(J, -2, -1))  # (B, N, 3, 3)

    return points_3d, C_3d_all
