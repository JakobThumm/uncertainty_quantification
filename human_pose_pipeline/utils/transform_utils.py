"""
Transform utilities for pose estimation preprocessing

This module implements the same preprocessing functions as Marian's Utils_From_Regress_Flow.py
but adapted for JAX and our pipeline. The goal is to exactly replicate RegressFlow's preprocessing.
"""

import math
import cv2
import numpy as np
import jax.numpy as jnp
from typing import Tuple, Union, Optional
from easydict import EasyDict

from human_pose_pipeline.pose_estimation.h36m_settings import (
    TRANSFORM_SIGMA,
    TRANSFORM_IMAGE_SIZE,
    TRANSFORM_HEATMAP_SIZE,
    NORMALIZATION_OFFSET
)

# RegressFlow configuration (matching Marian's CONFIG)
# CONFIG = EasyDict({
#     'DATA_PRESET': {
#         'TYPE': 'simple',
#         'SIGMA': 2,
#         'NUM_JOINTS': 17,
#         'IMAGE_SIZE': [256, 192],  # Height, Width
#         'HEATMAP_SIZE': [64, 48]
#     },
#     'MODEL': {
#         'TYPE': 'RegressFlow',
#         'NUM_LAYERS': 50,
#         'NUM_FC_FILTERS': [-1],
#         'HIDDEN_LIST': [-1],
#         'PRETRAINED': '',
#         'TRY_LOAD': ''
#     },
#     'TEST': {
#         'FLIP_TEST': True,
#         'HEATMAP2COORD': 'coord'
#     },
#     'LOSS': {
#         'TYPE': 'RLELoss'
#     }
# })

class SimpleTransform:
    """
    Simple transformation class that replicates Marian's SimpleTransform
    """
    def __init__(self, scale_factor=0, input_size=[256, 192], output_size=[64, 48],
                 rot=0, sigma=2, train=False):
        self._joint_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        self._scale_factor = scale_factor
        self._rot = rot
        self._input_size = input_size
        self._heatmap_size = output_size
        self._sigma = sigma
        self._train = train
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # width / height
        self._feat_stride = np.array(input_size) / np.array(output_size)
        self.pixel_std = 1
        self.align_coord = True

    def test_transform(self, src, bbox):
        """
        Transform image using bounding box for test/inference

        Args:
            src: Input image as numpy array (H, W, C)
            bbox: Bounding box [xmin, ymin, xmax, ymax] or None for full image

        Returns:
            tuple: (transformed_image, processed_bbox, center, scale, transformation_matrix)
        """
        if bbox is None:
            # Default to the whole image if no bbox is provided
            imgwidth, imght = src.shape[1], src.shape[0]
            bbox = [0, 0, imgwidth, imght]

        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio)
        scale = scale * 1.0

        input_size = self._input_size
        inp_h, inp_w = input_size

        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = _center_scale_to_box(center, scale)

        # Convert to tensor format and apply RegressFlow normalization
        img = im_to_jax(img)
        # Apply RegressFlow's specific normalization (subtract mean values)
        img = img.at[0].add(NORMALIZATION_OFFSET[0])
        img = img.at[1].add(NORMALIZATION_OFFSET[1])
        img = img.at[2].add(NORMALIZATION_OFFSET[2])

        return img, bbox, center, scale, trans

    def __call__(self, src, label=None):
        """
        Transform image for training/inference without specific bbox

        Args:
            src: Input image as numpy array (H, W, C)
            label: Not used in this implementation

        Returns:
            dict: Output dictionary with transformed data
        """
        imgwidth, imght = src.shape[1], src.shape[0]
        bbox = [0, 0, imgwidth, imght]
        center, scale = _box_to_center_scale(
            bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1], self._aspect_ratio)

        inp_h, inp_w = self._input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)

        # Convert to JAX array and apply normalization
        img = im_to_jax(img)
        img = img.at[0].add(NORMALIZATION_OFFSET[0])
        img = img.at[1].add(NORMALIZATION_OFFSET[1])
        img = img.at[2].add(NORMALIZATION_OFFSET[2])

        output = {
            'type': '2d_data',
            'image': img,
            'center': center,
            'scale': scale,
            'trans': trans,
            'bbox': bbox,
        }

        return output

def preprocess_image_with_bbox(image_array, bbox):
    """
    Preprocess image using the provided bounding box (JAX version of Marian's function)

    Args:
        image_array: Input image as numpy array (H, W, C)
        bbox: Bounding box [xmin, ymin, xmax, ymax]

    Returns:
        Tuple containing:
            - Preprocessed image tensor (JAX array)
            - Original image array
            - Center of the bounding box
            - Scale of the bounding box
            - Affine transformation matrix
            - Processed bounding box [xmin, ymin, xmax, ymax]
    """
    transformation = SimpleTransform(
        scale_factor=0,
        input_size=[TRANSFORM_IMAGE_SIZE[1], TRANSFORM_IMAGE_SIZE[0]],  # Height, Width
        output_size=[TRANSFORM_HEATMAP_SIZE[1], TRANSFORM_HEATMAP_SIZE[0]],  # Height, Width
        rot=0,
        sigma=TRANSFORM_SIGMA,
        train=False
    )

    # Apply transformation
    img, processed_bbox, center, scale, trans = transformation.test_transform(image_array, bbox)

    # Add batch dimension
    img = jnp.expand_dims(img, axis=0)

    return img, image_array, center, scale, trans, processed_bbox

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    """
    Get affine transformation matrix

    Args:
        center: Center point [x, y]
        scale: Scale factor [scale_x, scale_y]
        rot: Rotation angle in degrees
        output_size: Output size [width, height]
        shift: Translation shift
        inv: Whether to compute inverse transform

    Returns:
        np.ndarray: 2x3 affine transformation matrix
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_dir(src_point, rot_rad):
    """Rotate the point by rot_rad radians"""
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    """Return vector c that is perpendicular to (a - b)"""
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def im_to_jax(img):
    """
    Transform ndarray image to JAX tensor (equivalent to Marian's im_to_torch)

    Parameters
    ----------
    img: numpy.ndarray
        An ndarray with shape: (H, W, 3)

    Returns
    -------
    jax.numpy.ndarray
        A tensor with shape: (3, H, W)
    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = jnp.array(img, dtype=jnp.float32)
    if jnp.max(img) > 1:
        img = img / 255.0
    return img

def _box_to_center_scale(x, y, w, h, aspect_ratio=1.0, scale_mult=1.25):
    """Convert box coordinates to center and scale"""
    pixel_std = 1.0
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
    if center[0] != -1:
        scale = scale * scale_mult
    return center, scale

def _center_scale_to_box(center, scale):
    """Convert center and scale to box coordinates"""
    pixel_std = 1.0
    w = scale[0] * pixel_std
    h = scale[1] * pixel_std
    xmin = center[0] - w * 0.5
    ymin = center[1] - h * 0.5
    xmax = xmin + w
    ymax = ymin + h
    bbox = [xmin, ymin, xmax, ymax]
    return bbox

def convert_coordinates_regressflow_to_pixel(pred_joints, img_height=256, img_width=192):
    """
    Convert RegressFlow model predictions to pixel coordinates

    RegressFlow outputs coordinates in [-0.5, 0.5] range which need to be
    converted to pixel coordinates in the image.

    Args:
        pred_joints: Predicted joint coordinates in [-0.5, 0.5] range, shape (N, 2)
        img_height: Image height (default 256 for RegressFlow)
        img_width: Image width (default 192 for RegressFlow)

    Returns:
        np.ndarray: Joint coordinates in pixel space
    """
    pixel_joints = pred_joints.copy()
    pixel_joints[:, 0] = (pred_joints[:, 0] + 0.5) * img_width
    pixel_joints[:, 1] = (pred_joints[:, 1] + 0.5) * img_height
    return pixel_joints

def transform_coordinates_back_to_original(pred_joints_pixel, trans, scale_x=1.0, scale_y=1.0):
    """
    Transform coordinates back to original image space

    Args:
        pred_joints_pixel: Joint coordinates in processed image pixel space
        trans: Affine transformation matrix used during preprocessing
        scale_x: Scale factor for x coordinates (original_width / processed_width)
        scale_y: Scale factor for y coordinates (original_height / processed_height)

    Returns:
        np.ndarray: Joint coordinates in original image space
    """
    # Apply inverse affine transformation
    trans_inv = cv2.invertAffineTransform(trans)
    pred_joints_original = cv2.transform(np.expand_dims(pred_joints_pixel, axis=0), trans_inv)[0]

    # Scale to original image dimensions
    pred_joints_original[:, 0] *= scale_x
    pred_joints_original[:, 1] *= scale_y

    return pred_joints_original


def transform_predictions_to_original_space(pred_joints_normalized, trans, scale_x, scale_y,
                                           uncertainties=None, covariance=None):
    """
    Transform model predictions from normalized coordinates back to original image space.

    This function performs the full reverse transformation pipeline:
    1. Convert normalized coords (-0.5 to 0.5) to pixel coords in preprocessed image
    2. Apply inverse affine transformation to get coords in resized image
    3. Scale coords to original image dimensions
    4. Scale uncertainties appropriately if provided

    Args:
        pred_joints_normalized: Joint coordinates in normalized space (-0.5 to 0.5), shape (N, 2)
        trans: Affine transformation matrix used during preprocessing
        scale_x: Scale factor from resized to original width
        scale_y: Scale factor from resized to original height
        uncertainties: Optional uncertainty values, shape (N, 2)
        covariance: Optional covariance values, shape (N,)

    Returns:
        dict: Dictionary containing:
            - 'keypoints': Joint coordinates in original image space
            - 'uncertainties': Scaled uncertainties (if provided)
            - 'covariance': Scaled covariance (if provided)
    """
    # Step 1: Convert normalized coordinates to pixel coordinates in preprocessed image
    img_height, img_width = TRANSFORM_IMAGE_SIZE[1], TRANSFORM_IMAGE_SIZE[0]
    pred_joints_pixel = pred_joints_normalized.copy()
    pred_joints_pixel[:, 0] = (pred_joints_normalized[:, 0] + 0.5) * img_width
    pred_joints_pixel[:, 1] = (pred_joints_normalized[:, 1] + 0.5) * img_height

    # Step 2: Apply inverse affine transformation
    trans_inv = cv2.invertAffineTransform(trans)
    pred_joints_resized = cv2.transform(np.expand_dims(pred_joints_pixel, axis=0), trans_inv)[0]

    # Step 3: Scale to original image dimensions
    pred_joints_original = pred_joints_resized.copy()
    pred_joints_original[:, 0] *= scale_x
    pred_joints_original[:, 1] *= scale_y

    result = {'keypoints': pred_joints_original}

    # Step 4: Transform uncertainties if provided
    if uncertainties is not None:
        # Scale uncertainties to preprocessed image dimensions
        uncertainties_pixel = uncertainties.copy()
        uncertainties_pixel[:, 0] = uncertainties[:, 0] * img_width
        uncertainties_pixel[:, 1] = uncertainties[:, 1] * img_height

        # Scale to original image dimensions
        uncertainties_original = uncertainties_pixel.copy()
        uncertainties_original[:, 0] *= scale_x
        uncertainties_original[:, 1] *= scale_y

        result['uncertainties'] = uncertainties_original

        # Transform covariance if provided
        if covariance is not None:
            covariance_scaled = covariance.copy() * img_width * img_height
            covariance_original = covariance_scaled * scale_x * scale_y
            result['covariance'] = covariance_original

    return result