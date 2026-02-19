"""
Inference Helper for JAX-based Pose Estimation

This module provides inference functions for human pose estimation using JAX models.
Based on Marian's Inference_Helper.py but adapted for JAX instead of PyTorch.
"""

import json
import pickle
from time import time
from typing import Sequence, Tuple, Union
from matplotlib.pylab import f
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import threading

import torch

from human_pose_pipeline.utils.batched_transform_torch import create_joint_covariance_batched, triangulate_points_with_covariance_batched
from src.models.wrapper import model_from_string
from human_pose_pipeline.utils.gpu_accelerated_utils import (
    extract_bounding_box_images_batched,
    resize_image_batched_gpu,
    transform_predictions_to_original_space_batched,
    jax_to_torch
)
from human_pose_pipeline.pose_estimation.triangulation_helper import (
    create_joint_covariance,
    triangulate_points_with_covariance
)

from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_IDX_13_MODEL,
    YOLO_IMAGE_SIZE,
    YOLO_CONFIDENCE_THRESHOLD,
    OOD_THRESHOLD
)


# Global thread pool for parallel execution (reused across calls)
_thread_pool = None


def get_thread_pool():
    """Get or create a global thread pool with 2 workers."""
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=2)
    return _thread_pool


def joint_mapping(joints, mapping):
    """Apply joint mapping to reorder joints according to the provided mapping."""
    return joints[:, mapping, ...]


def resize_image(pil_image, target_size=YOLO_IMAGE_SIZE):
    """
    Resize image to network input dimensions.

    Args:
        pil_image (PIL.Image.Image): The input image
        target_size (tuple): Target size (width, height)

    Returns:
        tuple: (resized_image, original_dimensions, scale_factors)
    """
    # Get original image dimensions
    original_image_width, original_image_height = pil_image.size

    # Resize image to network input dimensions - same as Marian's approach
    resized_image = pil_image.resize(target_size, Image.LANCZOS)
    resized_width, resized_height = resized_image.size

    # Calculate scale factors for coordinate transformation
    scale_x = original_image_width / resized_width
    scale_y = original_image_height / resized_height

    return resized_image, (original_image_width, original_image_height), (scale_x, scale_y)


def pose_estimation_2d(
        input_images, pose_estimation_jit_fn, params, batch_stats, human_detector, device_torch, score_fn=None,
        human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
        ood_threshold=OOD_THRESHOLD,
        parallelize=False,
        num_output_joints=17,
        device='cpu'):
    """
    Complete 2D pose estimation pipeline: resize -> detect humans -> estimate poses.

    Args:
        input_images (torch.Tensor): The input high-resolution image
        pose_estimation_jit_fn: JIT-compiled pose estimation function
        params: JAX model parameters
        batch_stats: JAX model batch statistics (if available)
        human_detector: The pre-loaded YOLO human detection model (PyTorch)
        score_fn: Function to compute OOD score from model outputs. If None -> No OOD scoring.
        device_torch: PyTorch device for human detection
        human_detection_threshold (float, optional): Confidence threshold for human detection
        ood_threshold (float, optional): Threshold for OOD detection in pose estimation
        parallelize (bool, optional): Whether to run pose prediction and OOD scoring in parallel
        num_output_joints (int, optional): Number of joints the model outputs
        use_gpu_acceleration (bool, optional): Whether to use GPU-accelerated preprocessing

    Returns:
        List[Dict]: List of dictionaries containing for each detected person:
            - 'keypoints': Joint coordinates [[x1,y1], [x2,y2], ...]
            - 'uncertainties': Standard deviations
            - 'covariance': Covariance values
            - 'bbox': Bounding box in the YOLO image frame [x1, y1, x2, y2]
            - 'center': Center of the bounding box in the YOLO image frame [x, y]
            - 'scale': Width and height of the bounding box in the YOLO image frame [w, h]
            - 'ood_score': OOD score for the detected person (0 if no score_fn provided)
            - 'is_ood': Boolean indicating if the person is classified as OOD based on the threshold (False if no score_fn provided)
    """
    bounding_box_image_struct = extract_bounding_box_images(
        full_image=input_images,
        human_detector=human_detector,
        device_torch=device_torch,
        threshold=human_detection_threshold
    )
    scale_x, scale_y = bounding_box_image_struct['scale_factors_yolo']
    bbox = bounding_box_image_struct['bbox']
    bounding_box_image = bounding_box_image_struct['image']
    center = bounding_box_image_struct['center']
    scale = bounding_box_image_struct['scale']
    trans = bounding_box_image_struct['trans']
    mask = bounding_box_image_struct['mask']

    # Run pose prediction and OOD scoring in parallel
    if score_fn is None:
        # No OOD scoring - run pose prediction only
        pred_joints_13, uncertainties_13, covariance_13 = predict_pose(
            bounding_box_image, pose_estimation_jit_fn, params, batch_stats, num_output_joints, device=device
        )
        ood_score = torch.zeros(input_images.shape[0], device=device)
    elif score_fn is not None and parallelize:
        # Run pose prediction and OOD scoring in parallel using thread pool
        executor = get_thread_pool()

        # Submit both tasks to the thread pool simultaneously
        pose_future = executor.submit(predict_pose, bounding_box_image, pose_estimation_jit_fn, params, batch_stats, num_output_joints, device)
        ood_future = executor.submit(score_fn, bounding_box_image)

        # Wait for BOTH futures to complete simultaneously (more efficient than sequential .result() calls)
        wait([pose_future, ood_future], return_when=ALL_COMPLETED)

        # Get results (these are now instant since both are done)
        pred_joints_13, uncertainties_13, covariance_13 = pose_future.result()
        ood_score = float(np.asarray(ood_future.result()))
    else:
        pred_joints_13, uncertainties_13, covariance_13 = predict_pose(bounding_box_image, pose_estimation_jit_fn, params, batch_stats, num_output_joints, device=device)
        ood_score = torch.tensor(score_fn(bounding_box_image), device=device)
    is_ood = ood_score > ood_threshold

    # Transform back to original image space
    result = transform_predictions_to_original_space_batched(
        pred_joints_13, trans, scale_x, scale_y,
        uncertainties=uncertainties_13,
        covariance=covariance_13
    )
    # Fallback if no uncertainties are predicted
    if result.get('uncertainties') is None:
        result['uncertainties'] = torch.ones_like(result['keypoints']) * 10.0  # 10 pixel std dev
    if result.get('covariance') is None:
        result['covariance'] = torch.ones(len(result['keypoints']), device=device) * 0.1  # Small covariance

    # Store results for this person
    all_results = {
        'keypoints': result['keypoints'],
        'uncertainties': result['uncertainties'],
        'covariance': result['covariance'],
        'bbox': bbox,
        'center': center,
        'scale': scale,
        'ood_score': ood_score,
        'is_ood': is_ood,
        'mask': mask
    }
    return all_results


def extract_bounding_box_images(
        full_image: torch.Tensor,
        human_detector,
        device_torch,
        threshold=YOLO_CONFIDENCE_THRESHOLD
):
    """
    Extract bounding box images of detected humans from the full image.

    Args:
        full_image (PIL.Image.Image, batch of pytorch images [B, H, W, C]): The input high-resolution image
        human_detector: The pre-loaded YOLO human detection model (PyTorch)
        device_torch: PyTorch device for human detection
        threshold (float, optional): Confidence threshold for human detection
        use_gpu_acceleration (bool, optional): Whether to use GPU-accelerated preprocessing
    Returns:
        struct with keys (All of these are tensors with batch size B):
            - 'scale_factors_yolo': Scale factors (x, y) from original to YOLO input size
            - 'bbox': Bounding box coordinates [x1, y1, x2, y2]
            - 'image': Cropped bounding box image (PIL.Image)
            - 'center': Center of the bounding box in YOLO image [x, y]
            - 'scale': Width and height of the bounding box in YOLO image [w, h]
            - 'trans': Transformation matrix (2x3) from YOLO image to cropped bbox image
            - 'mask': Whether a human was found in the image or not (1 if found)
    """
    # Step 1: Resize image (YOLO needs image size divisible by 32)
    resized_image, original_dimensions, scale_factors = resize_image_batched_gpu(
        full_image, YOLO_IMAGE_SIZE, device=device_torch
    )
    # Step 2: Detect humans
    person_boxes, mask = detect_humans(human_detector, resized_image, device_torch, threshold=threshold)

    device_str = 'cuda' if str(device_torch).startswith('cuda') else 'cpu'
    bounding_box_images = extract_bounding_box_images_batched(
        resized_images=resized_image,
        person_boxes=person_boxes,
        scale_factors=scale_factors,
        device=device_str
    )
    bounding_box_images["mask"] = mask
    return bounding_box_images


def predict_pose(bounding_box_image, pose_estimation_jit_fn, params, batch_stats, num_output_joints=17, device='cpu'):
    """Predict pose for a single bounding box image using the JAX model.

    Args:
        bounding_box_image (torch.Tensor): Cropped image of the detected human [B, C, H, W]
        pose_estimation_jit_fn: JIT-compiled pose estimation function
        params: JAX model parameters
        batch_stats: JAX model batch statistics (if available)
        num_output_joints: Number of joints the model outputs (17 for full model, 3 for reduced model)
        device: Device to place output tensors on ('cpu' or 'cuda')
    Returns:
        tuple: (pred_joints_13, uncertainties_13, covariance_13)
    """
    # Convert to jax
    if isinstance(bounding_box_image, torch.Tensor):
        bounding_box_image = jnp.asarray(bounding_box_image)
    # Get model predictions using JIT-compiled function
    if batch_stats is not None:
        output = pose_estimation_jit_fn(params, batch_stats, bounding_box_image)
    else:
        output = pose_estimation_jit_fn(params, bounding_box_image)

    # Extract predictions - JAX model outputs (following Marian's approach)
    if isinstance(output, dict):
        # RegressFlowWithAleatoric returns dictionary with uncertainty outputs
        pred_joints = jax_to_torch(output['pred_jts'], device=device)  # Joint coordinates (num_output_joints, 2)
        log_variance = jax_to_torch(output.get('log_variance', output.get('pure_sigma', None)), device=device) if output.get('log_variance', output.get('pure_sigma', None)) is not None else None
        covariance_raw = jax_to_torch(output.get('covariance', None), device=device) if output.get('covariance', None) is not None else None
    else:
        # Regular RegressFlow returns tensor directly - reshape from flattened
        pred_joints_flat = jax_to_torch(output, device=device)  # Remove batch dimension
        pred_joints = pred_joints_flat.reshape(pred_joints_flat.shape[0], num_output_joints, 2)  # B, num_output_joints, 2 coords
        log_variance = None
        covariance_raw = None

    # Convert log variance to standard deviation (following Marian's approach)
    if log_variance is not None:
        uncertainties = torch.sqrt(torch.exp(log_variance))  # (num_output_joints, 2)
    else:
        uncertainties = None

    # Handle reduced 3-joint model (nose, left wrist, right wrist)
    if num_output_joints == 3:
        # TODO: Implement batched version if needed.
        # For 3-joint model: indices are [0=nose, 1=left_wrist, 2=right_wrist]
        # We need to expand to 13 joints by filling missing joints with nose position
        # pred_joints_13 = expand_3joints_to_13joints(pred_joints)
        # if uncertainties is not None:
        #     uncertainties_13 = expand_3joints_to_13joints(uncertainties)
        # else:
        #     uncertainties_13 = None
        # if covariance_raw is not None:
        #     # For 3-joint covariance, replicate nose covariance for missing joints
        #     covariance_13 = np.zeros(13)
        #     covariance_13[0] = covariance_raw[0]  # Nose
        #     covariance_13[5] = covariance_raw[1]  # LWrist
        #     covariance_13[6] = covariance_raw[2]  # RWrist
        #     covariance_13[1:5] = covariance_raw[0]  # Shoulders and elbows -> nose covariance
        #     covariance_13[7:] = covariance_raw[0]  # Hips, knees, ankles -> nose covariance
        # else:
        #     covariance_13 = None
        raise NotImplementedError("Batched version of 3 joint model not implemented yet.")
    # Select only the 13 joints of interest (same as Marian's approach)
    pred_joints_13 = pred_joints[:, JOINT_IDX_13_MODEL]  # (13, 2)
    if uncertainties is not None:
        uncertainties_13 = uncertainties[:, JOINT_IDX_13_MODEL]  # (13, 2)
    else:
        uncertainties_13 = None
    if covariance_raw is not None:
        covariance_13 = covariance_raw[:, JOINT_IDX_13_MODEL]  # (13,)
    else:
        covariance_13 = None
    return pred_joints_13, uncertainties_13, covariance_13


def expand_3joints_to_13joints(joints_3):
    """
    Expand 3-joint predictions (nose, left_wrist, right_wrist) to 13 joints.
    Missing joints are filled with nose position for debugging purposes.

    Args:
        joints_3: Array of shape (3, 2) with [nose, left_wrist, right_wrist]

    Returns:
        joints_13: Array of shape (13, 2) with all 13 joints
    """
    # 13-joint order: Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist,
    #                 LHip, RHip, LKnee, RKnee, LAnkle, RAnkle
    joints_13 = np.zeros((13, 2))

    nose = joints_3[0]
    left_wrist = joints_3[1]
    right_wrist = joints_3[2]

    # Set the 3 known joints
    joints_13[0] = nose         # Nose
    joints_13[5] = left_wrist   # LWrist
    joints_13[6] = right_wrist  # RWrist

    # Fill all other joints with nose position (for debugging)
    for i in [1, 2, 3, 4, 7, 8, 9, 10, 11, 12]:
        joints_13[i] = nose

    return joints_13


def process_frame_2d(frames, pose_estimation_jit_fn, params, batch_stats, human_detector, device_torch,
                     mirror_map, score_fn=None,
                     human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD, ood_threshold=OOD_THRESHOLD,
                     num_output_joints=17, verbose=False, device='cpu'):
    """
    Process a single frame to extract pose with uncertainty (JAX version).

    Args:
        frames: Input frame images, torch.Tensor
        pose_estimation_jit_fn: JIT-compiled pose estimation function
        params: JAX model parameters
        batch_stats: JAX model batch statistics
        human_detector: YOLO human detector
        device_torch: PyTorch device for YOLO
        mirror_map: Joint mapping to correct left/right swapping
        score_fn: Function to compute OOD score from model outputs. If None -> No OOD scoring.
        human_detection_threshold (float, optional): Confidence threshold for human detection
        ood_threshold (float, optional): Threshold for OOD detection in pose estimation
        num_output_joints (int, optional): Number of joints the model outputs

    Returns:
        List[Dict]: List of dictionaries containing for each detected person:
            - 'keypoints': Joint coordinates [[x1,y1], [x2,y2], ...]
            - 'uncertainties': Standard deviations
            - 'covariance': Covariance values
            - 'covariance_matrix': Per-joint 2x2 covariance matrices
            - 'bbox': Bounding box in the YOLO image frame [x1, y1, x2, y2]
            - 'center': Center of the bounding box in the YOLO image frame [x, y]
            - 'scale': Width and height of the bounding box in the YOLO image frame [w, h]
            - 'ood_score': OOD score for the detected person (0 if no score_fn provided)
            - 'is_ood': Boolean indicating if the person is classified as OOD based on the threshold (False if no score_fn provided)
    """
    if isinstance(frames, np.ndarray):
        frames = Image.fromarray(cv2.cvtColor(frames, cv2.COLOR_BGR2RGB))
    if isinstance(frames, torch.Tensor) and torch.mean(frames) > 2.0:
        frames = frames / 255.0
    t0 = time()
    pose_estimations = pose_estimation_2d(
        input_images=frames,
        pose_estimation_jit_fn=pose_estimation_jit_fn,
        params=params,
        batch_stats=batch_stats,
        human_detector=human_detector,
        device_torch=device_torch,
        score_fn=score_fn,
        human_detection_threshold=human_detection_threshold,
        ood_threshold=ood_threshold,
        num_output_joints=num_output_joints,
        device=device
    )
    pose_estimations['keypoints'] = joint_mapping(pose_estimations['keypoints'], mirror_map)
    pose_estimations['uncertainties'] = joint_mapping(pose_estimations['uncertainties'], mirror_map)
    pose_estimations['covariance'] = joint_mapping(pose_estimations['covariance'], mirror_map)
    # Construct per-joint 2x2 covariance matrices
    B, N, _ = pose_estimations['keypoints'].shape
    joint_covariances = torch.zeros((B, N, 2, 2), device=device)
    joint_covariances[:, :, 0, 0] = torch.pow(pose_estimations['uncertainties'][:, :, 0], 2)
    joint_covariances[:, :, 0, 1] = pose_estimations['covariance']
    joint_covariances[:, :, 1, 0] = pose_estimations['covariance']
    joint_covariances[:, :, 1, 1] = torch.pow(pose_estimations['uncertainties'][:, :, 1], 2)
    pose_estimations['covariance_matrix'] = joint_covariances
    if verbose:
        t1 = time()
        print(f"Total frame processing time (detection + pose estimation): {t1 - t0:.3f} seconds")
    return pose_estimations


def process_frame_3d(
    frames, projection_matrices, pose_estimation_jit_fn, params, batch_stats, human_detector, device_torch,
    mirror_map, score_fn=None,
    human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD, ood_threshold=OOD_THRESHOLD,
    num_output_joints=17, use_gpu_acceleration=True, verbose=True, device='cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
    """
    Process a single frame to extract pose with uncertainty (JAX version).

    Args:
        frames: Input frame images from the left and right camera. Shape: [2*B, H, W, C].
                The %2 = 0 elements correspond to the left camera,
                The %2 = 1 elements correspond to the right camera.
                Will be converted with jnp.reshape(B, 2, ...), so that
                  left_images = fames[:, 0] and
                  right_images = fames[:, 1]
        projection_matrices: The two camera projection matrices for triangulation
        pose_estimation_jit_fn: JIT-compiled pose estimation function
        params: JAX model parameters
        batch_stats: JAX model batch statistics
        human_detector: YOLO human detector
        device_torch: PyTorch device for YOLO
        mirror_map: Joint mapping to correct left/right swapping
        score_fn: Function to compute OOD score from model outputs. If None -> No OOD scoring.
        human_detection_threshold (float, optional): Confidence threshold for human detection
        ood_threshold (float, optional): Threshold for OOD detection in pose estimation
        num_output_joints (int, optional): Number of joints the model outputs
        use_gpu_acceleration (bool, optional): Whether to use GPU-accelerated preprocessing (default True)
        device: Device to place output tensors on ('cpu' or 'cuda')

    Returns:
        - points_3d: 3D joint coordinates [B, N_joints, 3]
        - C_3d_all: 3D covariance matrices [B, N_joints, 3, 3]
        - ood_score: OOD score for the detected person (0 if no score_fn provided)
        - is_ood: Boolean indicating if the person is classified as OOD based on the threshold (False if no score_fn provided)
        - human_detected: Boolean indicating if a human was detected in the frame
        - keypoints_2d: 2D joint coordinates from left camera [B, N_joints, 2]
        - uncertainties_2d: 2D uncertainties from left camera [B, N_joints, 2]
        - covariance_xy: 2D covariance (x-y) from left camera [B, N_joints]
    """
    assert len(frames) >= 2
    assert len(frames) % 2 == 0
    assert len(projection_matrices) == 2

    P1 = projection_matrices[0]
    P2 = projection_matrices[1]
    if isinstance(P1, np.ndarray):
        P1 = torch.from_numpy(P1).to(device)
        P2 = torch.from_numpy(P2).to(device)

    B = len(frames) // 2  # Number of frame pairs (left + right)
    first_frame = np.array(frames[0])
    np_frames = np.zeros([2 * B, first_frame.shape[0], first_frame.shape[1], first_frame.shape[2]], dtype=np.float32)
    for i in range(2 * B):
        new_frame = np.array(frames[i])
        if new_frame.shape != first_frame.shape:
            if verbose:
                print(f"New frame shape {new_frame.shape}, first frame shape {first_frame.shape}, adjusting new frame.")
            new_frame = new_frame[:first_frame.shape[0], :first_frame.shape[1]]
        np_frames[i] = new_frame
    frames = torch.from_numpy(np_frames).to(device_torch)

    batch_prediction = process_frame_2d(
        frames=frames,
        pose_estimation_jit_fn=pose_estimation_jit_fn,
        params=params,
        batch_stats=batch_stats,
        human_detector=human_detector,
        device_torch=device_torch,
        mirror_map=mirror_map,
        score_fn=score_fn,
        human_detection_threshold=human_detection_threshold,
        ood_threshold=ood_threshold,
        num_output_joints=num_output_joints,
        verbose=verbose,
        device=device
    )

    # Free GPU memory - frames are no longer needed
    del frames

    # Take the first detected person
    both_pose = batch_prediction['keypoints'].reshape(B, 2, 13, 2)
    both_uncertainty = batch_prediction['uncertainties'].reshape(B, 2, 13, 2)  # [B, 13, 2]
    both_covariance_matrix = batch_prediction['covariance_matrix'].reshape(B, 2, 13, 2, 2)  # [B, 13, 2, 2]
    both_ood_score = batch_prediction['ood_score'].reshape(B, 2)
    both_is_ood = batch_prediction['is_ood'].reshape(B, 2)
    both_human_detected = batch_prediction['mask'].reshape(B, 2)
    # Left
    left_pose = both_pose[:, 0]
    left_uncertainty = both_uncertainty[:, 0]
    left_covariance_matrix = both_covariance_matrix[:, 0]
    left_ood_score = both_ood_score[:, 0]
    left_is_ood = both_is_ood[:, 0]
    left_human_detected = both_human_detected[:, 0]
    # Right
    right_pose = both_pose[:, 1]
    right_uncertainty = both_uncertainty[:, 1]
    right_covariance_matrix = both_covariance_matrix[:, 1]
    right_ood_score = both_ood_score[:, 1]
    right_is_ood = both_is_ood[:, 1]
    right_human_detected = both_human_detected[:, 1]
    is_ood = torch.logical_or(left_is_ood, right_is_ood)
    ood_score = torch.max(left_ood_score, right_ood_score)
    human_detected = torch.logical_and(left_human_detected, right_human_detected)
    # is_ood = torch.logical_or(is_ood, ~human_detected)

    left_pose[human_detected == 0] = 0.0
    right_pose[human_detected == 0] = 0.0
    left_uncertainty[human_detected == 0] = 0.0
    right_uncertainty[human_detected == 0] = 0.0
    left_covariance_matrix[human_detected == 0] = 0.0
    right_covariance_matrix[human_detected == 0] = 0.0

    # Create joint covariance matrices
    C_2D = create_joint_covariance_batched(
        mapped_uncertainty_cam1=left_uncertainty,
        mapped_covariance_cam1=left_covariance_matrix[:, :, 0, 1],
        mapped_uncertainty_cam2=right_uncertainty,
        mapped_covariance_cam2=right_covariance_matrix[:, :, 0, 1],
        cross_covariance=torch.zeros((B, 13, 2, 2), device=device)  # Assume zero cross-covariance
    )
    points_3d, C_3d_all = triangulate_points_with_covariance_batched(
        left_pose, right_pose, P1, P2, C_2D
    )

    # Extract 2D keypoints from left camera for overlay visualization
    keypoints_2d = left_pose  # [B, 13, 2]
    uncertainties_2d = left_uncertainty  # [B, 13, 2]
    covariance_xy = left_covariance_matrix[:, :, 0, 1]  # [B, 13] (x-y covariance)

    return points_3d, C_3d_all, ood_score, is_ood, human_detected, keypoints_2d, uncertainties_2d, covariance_xy


def set_depth_uncertainty_to_constant(
    C_3d_all: torch.Tensor,
    R_world_to_cam,
    sigma_depth: float,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Replace the depth (camera-frame Z) component of 3D covariance matrices with
    a constant uncertainty, decoupled from the lateral (X, Y) components.

    This is useful when the lateral position is known accurately from stereo
    triangulation but the depth component from covariance propagation is
    unreliable (e.g. wide-baseline cameras).  The function:
      1. Rotates covariances to the primary camera frame:
             C_cam = R @ C_world @ R^T
      2. Replaces C_cam[..., 2, 2] with sigma_depth² and zeroes the
         cross-terms C_cam[..., :2, 2] and C_cam[..., 2, :2].
      3. Rotates back to world frame:
             C_world_new = R^T @ C_cam_new @ R

    Args:
        C_3d_all:        [B, J, 3, 3] covariances in world frame (any unit).
        R_world_to_cam:  (3, 3) or (B, 3, 3) rotation from world to camera frame.
                         Accepts numpy arrays or torch tensors.
        sigma_depth:     Depth std dev in the same unit as C_3d_all.
        device:          Torch device for intermediate tensors.

    Returns:
        [B, J, 3, 3] modified covariances in world frame.
    """
    B = C_3d_all.shape[0]

    if isinstance(R_world_to_cam, np.ndarray):
        R = torch.tensor(R_world_to_cam, dtype=torch.float32, device=device)
    else:
        R = R_world_to_cam.to(device=device, dtype=torch.float32)
    if R.ndim == 2:
        R = R.unsqueeze(0).expand(B, -1, -1)   # (B, 3, 3)

    R_exp = R.unsqueeze(1)                       # (B, 1, 3, 3)
    R_T_exp = R_exp.transpose(-1, -2)            # (B, 1, 3, 3)

    # Rotate to camera frame
    C_cam = R_exp @ C_3d_all @ R_T_exp           # (B, J, 3, 3)

    # Replace depth (Z) variance and zero cross-terms
    C_cam_new = C_cam.clone()
    C_cam_new[:, :, 2, 2] = sigma_depth ** 2
    C_cam_new[:, :, :2, 2] = 0.0
    C_cam_new[:, :, 2, :2] = 0.0

    # Rotate back to world frame: C_world = R^T @ C_cam @ R
    C_world_new = R_T_exp @ C_cam_new @ R_exp    # (B, J, 3, 3)
    return C_world_new


def lift_2d_to_3d_with_depth(keypoints_2d, depth_map, camera_intrinsics, device='cpu',
                             depth_outlier_threshold=1.5, search_radius=10,
                             border_clip=30):
    """
    Lift 2D keypoints to 3D using depth information (fully vectorized).

    For each joint, searches within `search_radius` pixels to handle zero-depth at
    object edges. The joint's (u, v) is moved to the minimum-depth pixel in that
    radius, and the final depth Z is set to the median of all valid depths in the
    radius (robust to noise).

    Args:
        keypoints_2d: 2D keypoint positions [B, N_joints, 2] in pixel coordinates
        depth_map: Depth image [B, H, W] in meters (or mm, will be handled)
        camera_intrinsics: Dict with keys 'fx', 'fy', 'cx', 'cy'
        device: Device for torch tensors
        depth_outlier_threshold: Max deviation from per-frame median depth (meters)
        search_radius: Pixel radius to search around each joint for valid depth
        border_clip: Pixel to remove from the border due to missing depth data at the edge.

    Returns:
        points_3d: 3D joint positions [B, N_joints, 3] in meters
        valid_depth: Boolean mask [B, N_joints] indicating valid depth readings
    """
    B, N_joints, _ = keypoints_2d.shape
    H, W = depth_map.shape[1], depth_map.shape[2]

    # Convert depth to meters if in millimeters (depth > 10 means mm)
    depth_map = torch.where(depth_map > 10, depth_map * 0.001, depth_map)

    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    # Extract u, v coordinates [B, N_joints]
    u = keypoints_2d[:, :, 0]  # [B, N_joints]
    v = keypoints_2d[:, :, 1]  # [B, N_joints]

    # Round to nearest integer for indexing
    u_int = torch.round(u).long()
    v_int = torch.round(v).long()

    # Check bounds - create validity mask [B, N_joints]
    valid_bounds = (u_int >= 0) & (u_int <= W) & (v_int >= 0) & (v_int <= H)

    # Clamp indices to valid range to prevent indexing errors
    u_clamped = torch.clamp(u_int, border_clip, W - (border_clip + 1))
    v_clamped = torch.clamp(v_int, border_clip, H - (border_clip + 1))

    # --- Radius-based depth search ---
    # Build offset grid [P] where P = (2*search_radius+1)^2
    dy, dx = torch.meshgrid(
        torch.arange(-search_radius, search_radius + 1, device=device),
        torch.arange(-search_radius, search_radius + 1, device=device),
        indexing='ij'
    )
    dy = dy.reshape(-1)  # [P]
    dx = dx.reshape(-1)  # [P]
    P = dy.shape[0]

    # Sample positions for every joint across the radius [B, N_joints, P]
    u_patch = torch.clamp(u_clamped.unsqueeze(-1) + dx.view(1, 1, P), 0, W - 1)
    v_patch = torch.clamp(v_clamped.unsqueeze(-1) + dy.view(1, 1, P), 0, H - 1)

    # Gather depth at all patch positions [B, N_joints, P]
    B_exp = torch.arange(B, device=device).view(B, 1, 1).expand(B, N_joints, P)
    depth_patch = depth_map[B_exp, v_patch, u_patch].float()

    # Step 1: Find the pixel with the minimum valid depth → update (u, v) only when
    # the original position has zero depth (edge of object).
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, N_joints)
    Z_orig = depth_map[batch_indices, v_clamped, u_clamped].float()
    needs_snap = Z_orig <= 0  # [B, N_joints]

    depth_for_argmin = depth_patch.masked_fill(depth_patch <= 0, float('inf'))
    min_idx = depth_for_argmin.argmin(dim=-1)  # [B, N_joints]

    u_snapped = torch.gather(u_patch, dim=-1, index=min_idx.unsqueeze(-1)).squeeze(-1)
    v_snapped = torch.gather(v_patch, dim=-1, index=min_idx.unsqueeze(-1)).squeeze(-1)

    u_final = torch.where(needs_snap, u_snapped, u_clamped)
    v_final = torch.where(needs_snap, v_snapped, v_clamped)

    Z = depth_map[batch_indices, v_final, u_final].float()

    # Cross-joint outlier rejection: discard joints whose depth deviates too much
    # from the per-frame median across all joints.
    Z_for_global = Z.masked_fill(Z <= 0, float('nan'))
    Z_global_median = torch.nanmedian(Z_for_global, dim=1).values  # [B]
    median_diff = torch.abs(Z - Z_global_median.unsqueeze(1))       # [B, N_joints]
    valid_depth_values = (Z > 0) & (median_diff <= depth_outlier_threshold)

    # Back-projection: use updated (u_final, v_final) for X/Y, median depth for Z
    X = (u_final - cx) * Z / fx
    Y = (v_final - cy) * Z / fy

    # Stack into 3D points [B, N_joints, 3]
    points_3d = torch.stack([X, Y, Z], dim=2)

    # Combine validity checks [B, N_joints]
    valid_depth = valid_bounds & valid_depth_values

    # Zero out invalid points
    points_3d = points_3d * valid_depth.unsqueeze(-1).float()

    return points_3d, valid_depth


def propagate_uncertainty_2d_to_3d(keypoints_2d, uncertainties_2d, covariance_2d,
                                   depth_map, camera_intrinsics,
                                   depth_uncertainty=0.01, device='cpu',
                                   depth_outlier_threshold=1.5):
    """
    Propagate 2D uncertainty to 3D using depth information and Jacobian (fully vectorized).

    Args:
        keypoints_2d: 2D keypoint positions [B, N_joints, 2]
        uncertainties_2d: 2D uncertainties [B, N_joints, 2] (std devs)
        covariance_2d: 2D covariance [B, N_joints] (covariance between x and y)
        depth_map: Depth image [B, H, W]
        camera_intrinsics: Dict with keys 'fx', 'fy', 'cx', 'cy'
        depth_uncertainty: Uncertainty in depth measurement (std dev in meters)
        device: Device for torch tensors

    Returns:
        C_3d: 3D covariance matrices [B, N_joints, 3, 3]
    """
    B, N_joints, _ = keypoints_2d.shape
    H, W = depth_map.shape[1], depth_map.shape[2]

    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    # Extract u, v coordinates [B, N_joints]
    u = keypoints_2d[:, :, 0]
    v = keypoints_2d[:, :, 1]

    # Round to nearest integer for indexing
    u_int = torch.round(u).long()
    v_int = torch.round(v).long()

    # Check bounds
    valid_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

    # Clamp indices
    u_clamped = torch.clamp(u_int, 0, W - 1)
    v_clamped = torch.clamp(v_int, 0, H - 1)

    # Gather depth values [B, N_joints]
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, N_joints)
    Z = depth_map[batch_indices, v_clamped, u_clamped]

    # Convert depth to meters if needed
    Z = torch.where(Z > 10, Z * 0.001, Z)

    # Median-based outlier rejection (same criterion as lift_2d_to_3d_with_depth)
    Z_for_median = Z.masked_fill(Z <= 0, float('nan'))
    Z_median = torch.nanmedian(Z_for_median, dim=1).values  # [B]
    median_diff = torch.abs(Z - Z_median.unsqueeze(1))      # [B, N_joints]
    valid_depth = (Z > 0) & (median_diff <= depth_outlier_threshold) & valid_bounds

    # Compute Jacobian matrices for all joints [B, N_joints, 3, 3]
    # J = [[Z/fx,     0,         (u-cx)/fx],
    #      [0,        Z/fy,      (v-cy)/fy],
    #      [0,        0,         1        ]]

    J = torch.zeros(B, N_joints, 3, 3, device=device)
    J[:, :, 0, 0] = Z / fx  # dX/du
    J[:, :, 0, 2] = (u - cx) / fx  # dX/dZ
    J[:, :, 1, 1] = Z / fy  # dY/dv
    J[:, :, 1, 2] = (v - cy) / fy  # dY/dZ
    J[:, :, 2, 2] = 1.0  # dZ/dZ

    # Construct input covariance matrices [B, N_joints, 3, 3]
    # C_input = [[σ_u²,    cov_uv,  0      ],
    #            [cov_uv,  σ_v²,    0      ],
    #            [0,       0,       σ_Z²   ]]

    sigma_u = uncertainties_2d[:, :, 0]  # [B, N_joints]
    sigma_v = uncertainties_2d[:, :, 1]  # [B, N_joints]
    cov_uv = covariance_2d  # [B, N_joints]

    C_input = torch.zeros(B, N_joints, 3, 3, device=device)
    C_input[:, :, 0, 0] = sigma_u ** 2
    C_input[:, :, 1, 1] = sigma_v ** 2
    C_input[:, :, 0, 1] = cov_uv
    C_input[:, :, 1, 0] = cov_uv
    C_input[:, :, 2, 2] = depth_uncertainty ** 2

    # Propagate uncertainty: C_3d = J @ C_input @ J^T
    # Using batched matrix multiplication
    # J: [B, N_joints, 3, 3]
    # C_input: [B, N_joints, 3, 3]
    # Result: [B, N_joints, 3, 3]

    # Reshape for batched matmul: [B*N_joints, 3, 3]
    J_flat = J.reshape(B * N_joints, 3, 3)
    C_input_flat = C_input.reshape(B * N_joints, 3, 3)

    # Compute J @ C_input
    temp = torch.bmm(J_flat, C_input_flat)  # [B*N_joints, 3, 3]

    # Compute (J @ C_input) @ J^T
    J_T_flat = J_flat.transpose(1, 2)  # [B*N_joints, 3, 3]
    C_3d_flat = torch.bmm(temp, J_T_flat)  # [B*N_joints, 3, 3]

    # Reshape back to [B, N_joints, 3, 3]
    C_3d = C_3d_flat.reshape(B, N_joints, 3, 3)

    # Zero out covariances for invalid joints
    C_3d = C_3d * valid_depth.unsqueeze(-1).unsqueeze(-1).float()

    return C_3d


def process_frame_3d_from_rgbd(
    rgb_frames, depth_frames, camera_intrinsics, pose_estimation_jit_fn, params, batch_stats,
    human_detector, device_torch, mirror_map, score_fn=None,
    human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD, ood_threshold=OOD_THRESHOLD,
    num_output_joints=17, use_gpu_acceleration=True, verbose=True, device='cpu',
    depth_uncertainty=0.01,
    R_rect_to_world=None,
    t_rect_to_world=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process RGB-D frame to extract 3D pose with uncertainty using depth lifting.

    Uncertainty propagation follows two steps:
      1. Pixel → rectified camera frame via back-projection Jacobian
         J = [[Z/fx, 0,    (u-cx)/fx],
              [0,    Z/fy, (v-cy)/fy],
              [0,    0,    1        ]]
         C_cam = J @ diag(σ_u², σ_v², σ_Z²) @ J^T
      2. Rectified camera → world frame (if R_rect_to_world is provided)
         X_world = R @ X_cam + t
         C_world = R @ C_cam @ R^T

    Args:
        rgb_frames: Input RGB images [B, H, W, C]
        depth_frames: Input aligned depth images [B, H, W] in mm or meters
        camera_intrinsics: Dict with keys 'fx', 'fy', 'cx', 'cy'
        pose_estimation_jit_fn: JIT-compiled pose estimation function
        params: JAX model parameters
        batch_stats: JAX model batch statistics
        human_detector: YOLO human detector
        device_torch: PyTorch device for YOLO
        mirror_map: Joint mapping to correct left/right swapping
        score_fn: Function to compute OOD score from model outputs
        human_detection_threshold: Confidence threshold for human detection
        ood_threshold: Threshold for OOD detection in pose estimation
        num_output_joints: Number of joints the model outputs
        use_gpu_acceleration: Whether to use GPU-accelerated preprocessing
        verbose: Print debug information
        device: Device to place output tensors on ('cpu' or 'cuda')
        depth_uncertainty: Std dev of depth measurement in meters (e.g. 0.002 for 2 mm)
        R_rect_to_world: Rotation from rectified camera to world frame.
            Shape (3, 3) broadcast to all frames, or (B, 3, 3) per-frame.
            numpy array or torch.Tensor. When None outputs remain in rectified camera frame.
        t_rect_to_world: Translation from rectified camera to world frame in meters.
            Shape (3,) or (B, 3). numpy array or torch.Tensor.

    Returns:
        - points_3d: 3D joint coordinates [B, N_joints, 3] (world frame if R given, else camera)
        - C_3d_all: 3D covariance matrices [B, N_joints, 3, 3] (same frame as points_3d)
        - ood_score: OOD scores [B]
        - is_ood: OOD flags [B]
        - human_detected: Detection flags [B]
        - keypoints_2d: 2D joint coordinates [B, N_joints, 2]
        - uncertainties_2d: 2D uncertainties [B, N_joints, 2]
        - covariance_xy: 2D covariance (x-y) [B, N_joints]
    """
    # Convert frames to tensor if needed
    if not isinstance(rgb_frames, torch.Tensor):
        np_frames = np.array(rgb_frames)
        rgb_frames = torch.from_numpy(np_frames).to(device_torch)
        rgb_frames = rgb_frames.float()

    if not isinstance(depth_frames, torch.Tensor):
        np_frames = np.array(depth_frames)
        depth_frames = torch.from_numpy(np_frames).to(device_torch)
        # Convert depth_frames to float32 for indexing and calculations (CUDA doesn't support UInt16 indexing)
        depth_frames = depth_frames.float()

    # Run 2D pose estimation (same as stereo mode)
    batch_prediction = process_frame_2d(
        frames=rgb_frames,
        pose_estimation_jit_fn=pose_estimation_jit_fn,
        params=params,
        batch_stats=batch_stats,
        human_detector=human_detector,
        device_torch=device_torch,
        mirror_map=mirror_map,
        score_fn=score_fn,
        human_detection_threshold=human_detection_threshold,
        ood_threshold=ood_threshold,
        num_output_joints=num_output_joints,
        verbose=verbose,
        device=device
    )

    # Extract 2D predictions
    keypoints_2d = batch_prediction['keypoints']  # [B, 13, 2]
    uncertainties_2d = batch_prediction['uncertainties']  # [B, 13, 2]
    covariance_2d = batch_prediction['covariance_matrix'][:, :, 0, 1]  # [B, 13] (x-y cov)
    ood_score = batch_prediction['ood_score']  # [B]
    is_ood = batch_prediction['is_ood']  # [B]
    human_detected = batch_prediction['mask']  # [B]

    # Lift 2D keypoints to 3D using depth
    points_3d, valid_depth = lift_2d_to_3d_with_depth(
        keypoints_2d, depth_frames, camera_intrinsics, device=device
    )

    # Propagate uncertainty to 3D
    C_3d_all = propagate_uncertainty_2d_to_3d(
        keypoints_2d, uncertainties_2d, covariance_2d,
        depth_frames, camera_intrinsics,
        depth_uncertainty=depth_uncertainty,
        device=device
    )

    # Mark invalid joints (no depth or no human detected) - Vectorized
    # Create combined validity mask: [B, N_joints]
    # If human not detected, all joints invalid
    # If human detected, use valid_depth mask
    human_detected_expanded = human_detected.unsqueeze(1)  # [B, 1]
    combined_valid = valid_depth & human_detected_expanded  # [B, N_joints]

    is_ood = torch.logical_or(is_ood, torch.any(torch.logical_not(combined_valid), dim=1))

    # Apply mask to zero out invalid joints
    # Use broadcasting: [B, N_joints, 1] for 3D coordinates
    points_3d = points_3d * combined_valid.unsqueeze(-1).float()

    # Use broadcasting: [B, N_joints, 1, 1] for 3×3 covariance matrices
    C_3d_all = C_3d_all * combined_valid.unsqueeze(-1).unsqueeze(-1).float()

    # Optionally rotate from rectified camera frame to world frame.
    if R_rect_to_world is not None:
        if isinstance(R_rect_to_world, np.ndarray):
            R = torch.tensor(R_rect_to_world, dtype=torch.float32, device=device)
        else:
            R = R_rect_to_world.to(device=device, dtype=torch.float32)
        if R.ndim == 2:
            R = R.unsqueeze(0).expand(points_3d.shape[0], -1, -1)  # (B, 3, 3)

        # Rotate 3D points: X_world[b,k] = R[b] @ X_cam[b,k]
        points_3d = torch.einsum('bij,bkj->bki', R, points_3d)

        if t_rect_to_world is not None:
            if isinstance(t_rect_to_world, np.ndarray):
                t = torch.tensor(t_rect_to_world, dtype=torch.float32, device=device)
            else:
                t = t_rect_to_world.to(device=device, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)              # (1, 3)
            points_3d = points_3d + t.unsqueeze(1)  # broadcast over joints

        # Rotate covariances: C_world[b,k] = R[b] @ C_cam[b,k] @ R[b]^T
        R_exp = R.unsqueeze(1)                  # (B, 1, 3, 3)
        C_3d_all = R_exp @ C_3d_all @ R_exp.transpose(-1, -2)
        # Convert to mm
        points_3d *= 1000
        C_3d_all *= 1000 * 1000

    # Return 2D keypoints for visualization overlay
    return points_3d, C_3d_all, ood_score, is_ood, human_detected, keypoints_2d, uncertainties_2d, covariance_2d


def detect_humans(
    model,
    images: torch.Tensor,
    device_torch: str,
    threshold: float = 0.8,
    verbose: bool = False
):
    """
    Detect human (Important!!! Takes first human per image.) in an image using YOLO (ultralytics).

    Can only predict one human per image, otherwise, batching doesn't work!

    Args:
        model: YOLO model from ultralytics
        images torch.Tensor [B, H, W, C] Input images
        device_torch: PyTorch device (for compatibility, not used with ultralytics)
        threshold (float): Detection confidence threshold

    Returns:
        person_boxes [B, 4]: Bounding boxes for detected humans, each box is [x1, y1, x2, y2] in image coordinates.
        mask [B]: indicates whether a human was found or not [1 if found, 0 if not]
    """
    # Run YOLO prediction
    images = images.permute(0, 3, 1, 2)  # [B, C, H, W]
    results = model.predict(images, conf=threshold, verbose=False)
    person_boxes = torch.zeros(images.shape[0], 4, device=device_torch)
    person_boxes[:, 2] = images.shape[3]
    person_boxes[:, 3] = images.shape[2]
    mask = torch.zeros(images.shape[0], dtype=torch.bool, device=device_torch)
    for idx, result in enumerate(results):
        if result is not None:
            # Get boxes, confidences, and classes
            boxes = result.boxes.xyxy
            confidences = result.boxes.conf
            classes = result.boxes.cls
            # Filter for person class (class 0 in COCO)
            for i, cls in enumerate(classes):
                if int(cls) == 0 and confidences[i] >= threshold:
                    person_boxes[idx] = boxes[i]
                    mask[idx] = 1
                    break
    return person_boxes, mask


def fill_pose_buffer(
    points_3d_buffer: jnp.ndarray,
    covariance_buffer: jnp.ndarray,
    pose_valid_buffer: jnp.ndarray,
    points_3d: jnp.ndarray,
    covariance: jnp.ndarray,
    is_valid: bool,
    motion_prediction_buffer: jnp.ndarray,
    motion_uncertainty_buffer: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
    """Fill the pose buffer with the latest 3D points and covariance, handling OOD cases.

    Args:
        points_3d_buffer: Buffer of 3D points [input_horizon_length, num_joints, 3]
        covariance_buffer: Buffer of covariance matrices [input_horizon_length, num_joints, 3, 3]
        pose_valid_buffer: Buffer indicating which poses are successful estimations from images (True) and
            which are taken from the motion prediction (False)
        points_3d: Latest 3D points [num_joints, 3]
        covariance: Latest covariance matrices [num_joints, 3, 3]
        is_valid: Boolean indicating if the latest pose estimation was valid
        motion_prediction_buffer: Buffer of motion predictions [prediction_horizon_length, num_joints, 3]
        motion_uncertainty_buffer: Buffer of motion uncertainties [prediction_horizon_length, num_joints, 3, 3]

    Returns:
        - Updated points_3d_buffer
        - Updated covariance_buffer
        - Updated pose_valid_buffer
        - Boolean indicating if a prediction is possible.
    """
    if is_valid:
        predicted_points = points_3d
        predicted_covariance = covariance
    else:
        # Use motion prediction from buffer instead of current OOD prediction
        predicted_points = motion_prediction_buffer[0]
        predicted_covariance = motion_uncertainty_buffer[0]

    # Shift buffers and add new prediction
    points_3d_buffer = jnp.roll(points_3d_buffer, shift=-1, axis=0)
    covariance_buffer = jnp.roll(covariance_buffer, shift=-1, axis=0)
    pose_valid_buffer = jnp.roll(pose_valid_buffer, shift=-1, axis=0)
    points_3d_buffer = points_3d_buffer.at[-1].set(predicted_points)
    covariance_buffer = covariance_buffer.at[-1].set(predicted_covariance)
    pose_valid_buffer = pose_valid_buffer.at[-1].set(is_valid)

    pose_buffer_good = bool(jnp.all(1 - jnp.all(points_3d_buffer == 0.0, axis=[1, 2])))

    return points_3d_buffer, covariance_buffer, pose_valid_buffer, pose_buffer_good


def update_motion_prediction_buffer(
    motion_prediction_buffer: jnp.ndarray,
    motion_uncertainty_buffer: jnp.ndarray,
    predicted_motion: jnp.ndarray,
    predicted_motion_uncertainty: jnp.ndarray,
    is_ood: bool,
    pose_valid_buffer: jnp.ndarray,
    n_correct_poses_required: int = 3
) -> Tuple[jnp.ndarray, jnp.ndarray, bool]:
    """Update the motion prediction buffer based on the latest 3D points and OOD status.

    If the latest prediction is OOD, shift the buffer and set the last motion to all zeros.
    Otherwise, use the predicted motion.

    Args:
        motion_prediction_buffer: Buffer of motion predictions [prediction_horizon_length, num_joints, 3]
        motion_uncertainty_buffer: Buffer of motion uncertainties [prediction_horizon_length, num_joints, 3, 3]
        predicted_motion: Predicted human motion [prediction_horizon_length, num_joints, 3]
        predicted_motion_uncertainty: Predicted human motion uncertainty [prediction_horizon_length, num_joints, 3, 3]
        is_ood: Boolean indicating if the latest prediction is OOD
        last_pose_valid: Boolean indicating if the most recent pose was valid
        pose_valid_buffer: Buffer indicating which poses are successful estimations from images (True) and
            which are taken from the motion prediction (False)
        n_correct_poses_required: Number of consecutive correct poses required to resume normal motion updates

    Returns:
        - Updated motion_prediction_buffer
        - Updated motion_uncertainty_buffer
        - Indicator if the predicted motion was used (True) or the motion prediction was rotated (False).
    """
    if not is_ood and jnp.all(pose_valid_buffer[-n_correct_poses_required:] == 1):
        # Use predicted motion
        return predicted_motion, predicted_motion_uncertainty, True
    else:
        # Shift buffers and set last motion to all zeros
        motion_prediction_buffer = jnp.roll(motion_prediction_buffer, shift=-1, axis=0)
        motion_uncertainty_buffer = jnp.roll(motion_uncertainty_buffer, shift=-1, axis=0)
        motion_prediction_buffer = motion_prediction_buffer.at[-1].set(jnp.zeros_like(motion_prediction_buffer[-1]))
        motion_uncertainty_buffer = motion_uncertainty_buffer.at[-1].set(jnp.zeros_like(motion_uncertainty_buffer[-1]))

    return motion_prediction_buffer, motion_uncertainty_buffer, False


def reset_yolo_tracking(yolo_pose_model):
    """
    Reset YOLO tracking state to start fresh tracking on a new sequence.

    Call this function between different video sequences or when you want to
    restart tracking with new IDs.

    Args:
        yolo_pose_model: YOLO pose estimation model

    Example:
        >>> from ultralytics import YOLO
        >>> yolo_model = YOLO("yolo11n-pose.pt")
        >>>
        >>> # Process first video sequence with tracking
        >>> for frame in video1_frames:
        >>>     results = process_frame_2d_yolo(frame, yolo_model, mirror_map,
        >>>                                      enable_tracking=True)
        >>>
        >>> # Reset tracking before processing a new sequence
        >>> reset_yolo_tracking(yolo_model)
        >>>
        >>> # Process second video sequence with fresh tracking IDs
        >>> for frame in video2_frames:
        >>>     results = process_frame_2d_yolo(frame, yolo_model, mirror_map,
        >>>                                      enable_tracking=True)
    """
    # Reset the predictor which contains the tracker state
    # This forces YOLO to reinitialize tracking on the next call
    if hasattr(yolo_pose_model, 'predictor') and yolo_pose_model.predictor is not None:
        # Try to reset trackers list if it exists
        if hasattr(yolo_pose_model.predictor, 'trackers'):
            yolo_pose_model.predictor.trackers = []
        # Alternatively, reset the entire predictor
        yolo_pose_model.predictor = None


def process_frame_2d_yolo(
    frames: Union[torch.Tensor, np.ndarray, Image.Image],
    yolo_pose_model,
    mirror_map,
    enable_tracking: bool = True,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    verbose: bool = False,
    device: str = 'cpu',
    tracker_config: str = 'botsort.yaml'
) -> dict:
    """
    Process frames using YOLO's built-in pose estimation and optional tracking.

    This function uses YOLO's end-to-end pose estimation without OOD detection.
    YOLO outputs 17 keypoints which are mapped to 13 joints (excluding eyes and ears).

    Args:
        frames: Input frame images (torch.Tensor [B, H, W, C], np.ndarray, or PIL Image)
        yolo_pose_model: YOLO pose estimation model (e.g., YOLO11-pose)
        mirror_map: Joint mapping to correct left/right swapping
        enable_tracking: Whether to enable multi-object tracking with persistent IDs
        confidence_threshold: Confidence threshold for pose detection
        verbose: Print debug information
        device: Device to place output tensors on ('cpu' or 'cuda')
        tracker_config: Tracker configuration file (e.g., 'botsort.yaml', 'bytetrack.yaml')

    Returns:
        Dict containing for each detected person:
            - 'keypoints': Joint coordinates [B, 13, 2]
            - 'uncertainties': Placeholder zeros (YOLO doesn't provide uncertainties) [B, 13, 2]
            - 'covariance': Placeholder zeros [B, 13]
            - 'covariance_matrix': Placeholder zeros [B, 13, 2, 2]
            - 'bbox': Bounding boxes [B, 4] in format [x1, y1, x2, y2]
            - 'center': Centers of bounding boxes [B, 2]
            - 'scale': Width and height of bounding boxes [B, 2]
            - 'confidence': Keypoint confidence scores [B, 13]
            - 'track_id': Track IDs if tracking enabled, otherwise -1 [B]
            - 'ood_score': Placeholder zeros [B]
            - 'is_ood': Placeholder False [B]
            - 'mask': Whether a human was detected [B]

    Note:
        - YOLO outputs 17 keypoints in COCO format
        - This function maps them to 13 joints by excluding eyes (indices 1-2) and ears (indices 3-4)
        - No uncertainty quantification or OOD detection is performed
        - When tracking is enabled and batch size > 1, frames are processed sequentially
          to maintain proper tracking state across frames
    """
    t0 = time()

    # YOLO can handle various input formats, but tensors must be in BCHW with specific sizes
    # For simplicity, convert everything to numpy arrays and let YOLO handle preprocessing
    original_format = None
    if isinstance(frames, Image.Image):
        frames_for_yolo = [np.array(frames)]
        B = 1
        original_format = 'pil'
    elif isinstance(frames, np.ndarray):
        if frames.ndim == 3:
            frames_for_yolo = [frames]
            B = 1
        else:
            # Split batch into list of numpy arrays
            frames_for_yolo = [frames[i] for i in range(frames.shape[0])]
            B = frames.shape[0]
        original_format = 'numpy'
    elif isinstance(frames, torch.Tensor):
        # Convert tensor to numpy arrays for YOLO
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)

        # Handle both BHWC and BCHW formats
        if frames.shape[1] == 3:  # Already in BCHW format
            frames_np = frames.permute(0, 2, 3, 1).cpu().numpy()
        else:  # BHWC format
            frames_np = frames.cpu().numpy()

        # Convert to uint8 if normalized
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)
        else:
            frames_np = frames_np.astype(np.uint8)

        frames_for_yolo = [frames_np[i] for i in range(frames_np.shape[0])]
        B = frames_np.shape[0]
        original_format = 'torch'
    else:
        raise ValueError(f"Unsupported frame type: {type(frames)}")

    # Run YOLO pose estimation with or without tracking
    # When tracking is enabled and B > 1, loop through frames sequentially
    if enable_tracking and B > 1:
        # Process frames one at a time to maintain tracking state
        results = []
        for i in range(B):
            frame_result = yolo_pose_model.track(
                frames_for_yolo[i],
                conf=confidence_threshold,
                verbose=verbose,
                persist=True,
                tracker=tracker_config,
                device=device
            )
            results.extend(frame_result)
    elif enable_tracking:
        # Single frame with tracking
        results = yolo_pose_model.track(
            frames_for_yolo[0] if B == 1 else frames_for_yolo,
            conf=confidence_threshold,
            verbose=verbose,
            persist=True,
            tracker=tracker_config,
            device=device
        )
    else:
        # Regular prediction mode (can process batch at once)
        if B == 1:
            results = yolo_pose_model.predict(
                frames_for_yolo[0],
                conf=confidence_threshold,
                verbose=verbose,
                device=device
            )
        else:
            # Process multiple frames
            results = []
            for frame in frames_for_yolo:
                frame_result = yolo_pose_model.predict(
                    frame,
                    conf=confidence_threshold,
                    verbose=verbose,
                    device=device
                )
                results.extend(frame_result)

    # Initialize output tensors
    keypoints_13 = torch.zeros((B, 13, 2), device=device)
    confidences_13 = torch.zeros((B, 13), device=device)
    sigmas_13 = torch.zeros((B, 13, 2), device=device)
    has_sigma = False
    bboxes = torch.zeros((B, 4), device=device)
    centers = torch.zeros((B, 2), device=device)
    scales = torch.zeros((B, 2), device=device)
    track_ids = torch.full((B,), -1, dtype=torch.long, device=device)
    mask = torch.zeros(B, dtype=torch.bool, device=device)

    # Process results for each frame in batch
    for idx, result in enumerate(results):
        if result.keypoints is not None and len(result.keypoints) > 0:
            # Take first detected person (for consistency with existing pipeline)
            # YOLO keypoints format: [N_persons, 17, 3] where last dim is [x, y, confidence]
            kpts = result.keypoints.data[0]  # [17, 3] - first person

            # Extract x, y coordinates [17, 2] and move to correct device
            kpts_xy = kpts[:, :2].to(device)

            # Extract confidence scores [17] and move to correct device
            kpts_conf = kpts[:, 2].to(device)

            # Map from 17 keypoints to 13 keypoints
            keypoints_13[idx] = kpts_xy[JOINT_IDX_13_MODEL]
            confidences_13[idx] = kpts_conf[JOINT_IDX_13_MODEL]

            # Extract sigma uncertainties if available (custom Pose26 model)
            if hasattr(result, 'kpts_sigma') and result.kpts_sigma is not None and len(result.kpts_sigma) > 0:
                sigma = result.kpts_sigma[0].to(device)  # [17, 2] - first person
                sigmas_13[idx] = sigma[JOINT_IDX_13_MODEL]
                has_sigma = True

            # Extract bounding box
            if result.boxes is not None and len(result.boxes) > 0:
                bbox = result.boxes.xyxy[0].to(device)  # [x1, y1, x2, y2]
                bboxes[idx] = bbox

                # Compute center and scale
                x1, y1, x2, y2 = bbox
                centers[idx, 0] = (x1 + x2) / 2
                centers[idx, 1] = (y1 + y2) / 2
                scales[idx, 0] = x2 - x1
                scales[idx, 1] = y2 - y1

                # Extract track ID if available
                if enable_tracking and hasattr(result.boxes, 'id') and result.boxes.id is not None:
                    track_ids[idx] = int(result.boxes.id[0].item())

                mask[idx] = True

    # Apply mirror mapping to keypoints
    keypoints_13 = joint_mapping(keypoints_13, mirror_map)
    confidences_13 = joint_mapping(confidences_13.unsqueeze(-1), mirror_map).squeeze(-1)

    # Use actual sigma values from the custom Pose26 model if available,
    # otherwise fall back to a placeholder (10 pixel std dev)
    if has_sigma:
        sigmas_13 = joint_mapping(sigmas_13, mirror_map)
        uncertainties = sigmas_13
    else:
        uncertainties = torch.ones((B, 13, 2), device=device) * 10.0

    covariance = torch.zeros((B, 13), device=device)  # No x-y covariance from sigma head

    # Construct per-joint 2x2 covariance matrices (diagonal only)
    joint_covariances = torch.zeros((B, 13, 2, 2), device=device)
    joint_covariances[:, :, 0, 0] = torch.pow(uncertainties[:, :, 0], 2)
    joint_covariances[:, :, 1, 1] = torch.pow(uncertainties[:, :, 1], 2)

    # Prepare output dictionary
    pose_estimations = {
        'keypoints': keypoints_13,
        'uncertainties': uncertainties,
        'covariance': covariance,
        'covariance_matrix': joint_covariances,
        'bbox': bboxes,
        'center': centers,
        'scale': scales,
        'confidence': confidences_13,
        'track_id': track_ids,
        'ood_score': torch.zeros(B, device=device),
        'is_ood': torch.zeros(B, dtype=torch.bool, device=device),
        'mask': mask
    }

    if verbose:
        t1 = time()
        print(f"YOLO pose estimation time: {t1 - t0:.3f} seconds")
        if enable_tracking:
            print(f"Detected persons with track IDs: {track_ids[mask].tolist()}")

    return pose_estimations


def process_frame_3d_yolo(
    frames: Union[torch.Tensor, np.ndarray],
    projection_matrices: list,
    yolo_pose_model,
    mirror_map,
    enable_tracking: bool = True,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    verbose: bool = False,
    device: str = 'cpu',
    tracker_config: str = 'botsort.yaml'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process stereo frames using YOLO pose estimation and triangulate to 3D.

    Args:
        frames: Input frame images from left and right cameras. Shape: [2*B, H, W, C]
                The %2 = 0 elements correspond to the left camera,
                The %2 = 1 elements correspond to the right camera.
        projection_matrices: The two camera projection matrices for triangulation [P1, P2]
        yolo_pose_model: YOLO pose estimation model
        mirror_map: Joint mapping to correct left/right swapping
        enable_tracking: Whether to enable multi-object tracking
        confidence_threshold: Confidence threshold for pose detection
        verbose: Print debug information
        device: Device to place output tensors on ('cpu' or 'cuda')
        tracker_config: Tracker configuration file

    Returns:
        - points_3d: 3D joint coordinates [B, 13, 3]
        - C_3d_all: 3D covariance matrices [B, 13, 3, 3] (minimal - no uncertainty from YOLO)
        - ood_score: Placeholder zeros [B]
        - is_ood: Placeholder False
        - human_detected: Boolean indicating if humans were detected
        - keypoints_2d: 2D joint coordinates from left camera [B, 13, 2]
        - uncertainties_2d: Placeholder uncertainties from left camera [B, 13, 2]
        - covariance_xy: Placeholder covariance from left camera [B, 13]

    Note:
        - When tracking is enabled and B > 1, stereo pairs are processed sequentially
          to maintain proper tracking state across frame pairs
    """
    assert len(frames) >= 2
    assert len(frames) % 2 == 0
    assert len(projection_matrices) == 2

    P1 = projection_matrices[0]
    P2 = projection_matrices[1]
    if isinstance(P1, np.ndarray):
        P1 = torch.from_numpy(P1).to(device)
        P2 = torch.from_numpy(P2).to(device)

    B = len(frames) // 2

    # Convert frames to tensor if needed
    if isinstance(frames, list):
        first_frame = np.array(frames[0])
        np_frames = np.zeros(
            [2 * B, first_frame.shape[0], first_frame.shape[1], first_frame.shape[2]],
            dtype=np.float32
        )
        for i in range(2 * B):
            new_frame = np.array(frames[i])
            if new_frame.shape != first_frame.shape:
                if verbose:
                    print(f"New frame shape {new_frame.shape}, first frame shape "
                          f"{first_frame.shape}, adjusting new frame.")
                new_frame = new_frame[:first_frame.shape[0], :first_frame.shape[1]]
            np_frames[i] = new_frame
        frames = torch.from_numpy(np_frames).to(device)

    # Run 2D pose estimation on both cameras
    # If tracking is enabled and B > 1, process stereo pairs sequentially
    if enable_tracking and B > 1:
        # Process each stereo pair sequentially
        all_predictions = []
        for i in range(B):
            # Extract stereo pair [left, right]
            stereo_pair = torch.stack([frames[2*i], frames[2*i+1]])
            pair_prediction = process_frame_2d_yolo(
                frames=stereo_pair,
                yolo_pose_model=yolo_pose_model,
                mirror_map=mirror_map,
                enable_tracking=enable_tracking,
                confidence_threshold=confidence_threshold,
                verbose=verbose,
                device=device,
                tracker_config=tracker_config
            )
            all_predictions.append(pair_prediction)

        # Combine predictions from all stereo pairs
        batch_prediction = {
            key: torch.cat([pred[key] for pred in all_predictions], dim=0)
            for key in all_predictions[0].keys()
        }
    else:
        # Process all frames at once (either single pair or tracking disabled)
        batch_prediction = process_frame_2d_yolo(
            frames=frames,
            yolo_pose_model=yolo_pose_model,
            mirror_map=mirror_map,
            enable_tracking=enable_tracking,
            confidence_threshold=confidence_threshold,
            verbose=verbose,
            device=device,
            tracker_config=tracker_config
        )

    # Reshape to separate left and right cameras
    both_pose = batch_prediction['keypoints'].reshape(B, 2, 13, 2)
    both_uncertainty = batch_prediction['uncertainties'].reshape(B, 2, 13, 2)
    both_covariance_matrix = batch_prediction['covariance_matrix'].reshape(B, 2, 13, 2, 2)
    both_mask = batch_prediction['mask'].reshape(B, 2)

    # Split left and right
    left_pose = both_pose[:, 0]
    left_uncertainty = both_uncertainty[:, 0]
    left_covariance_matrix = both_covariance_matrix[:, 0]
    left_human_detected = both_mask[:, 0]

    right_pose = both_pose[:, 1]
    right_uncertainty = both_uncertainty[:, 1]
    right_covariance_matrix = both_covariance_matrix[:, 1]
    right_human_detected = both_mask[:, 1]

    # Human detected only if both cameras detect
    human_detected = torch.logical_and(left_human_detected, right_human_detected)

    # Zero out invalid detections
    left_pose[human_detected == 0] = 0.0
    right_pose[human_detected == 0] = 0.0
    left_uncertainty[human_detected == 0] = 0.0
    right_uncertainty[human_detected == 0] = 0.0
    left_covariance_matrix[human_detected == 0] = 0.0
    right_covariance_matrix[human_detected == 0] = 0.0

    # Create joint covariance matrices for triangulation
    C_2D = create_joint_covariance_batched(
        mapped_uncertainty_cam1=left_uncertainty,
        mapped_covariance_cam1=left_covariance_matrix[:, :, 0, 1],
        mapped_uncertainty_cam2=right_uncertainty,
        mapped_covariance_cam2=right_covariance_matrix[:, :, 0, 1],
        cross_covariance=torch.zeros((B, 13, 2, 2), device=device)
    )

    # Triangulate to 3D
    points_3d, C_3d_all = triangulate_points_with_covariance_batched(
        left_pose, right_pose, P1, P2, C_2D
    )

    # Extract outputs
    keypoints_2d = left_pose
    uncertainties_2d = left_uncertainty
    covariance_xy = left_covariance_matrix[:, :, 0, 1]
    ood_score = torch.zeros(B, device=device)
    is_ood = torch.zeros(B, dtype=torch.bool, device=device)

    return points_3d, C_3d_all, ood_score, bool(is_ood[0]), bool(human_detected[0]), keypoints_2d, uncertainties_2d, covariance_xy


def process_frame_3d_from_rgbd_yolo(
    rgb_frames: Union[torch.Tensor, np.ndarray, Sequence],
    depth_frames: Union[torch.Tensor, np.ndarray, Sequence],
    camera_intrinsics: dict,
    yolo_pose_model,
    mirror_map,
    enable_tracking: bool = True,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    verbose: bool = False,
    device: str = 'cpu',
    depth_uncertainty: float = 0.01,
    tracker_config: str = 'botsort.yaml',
    R_rect_to_world=None,
    t_rect_to_world=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process RGB-D frames using YOLO pose estimation and lift to 3D using depth.

    Args:
        rgb_frames: Input RGB images [B, H, W, C]
        depth_frames: Input aligned depth images [B, H, W] in mm or meters
        camera_intrinsics: Dict with keys 'fx', 'fy', 'cx', 'cy'
        yolo_pose_model: YOLO pose estimation model
        mirror_map: Joint mapping to correct left/right swapping
        enable_tracking: Whether to enable multi-object tracking
        confidence_threshold: Confidence threshold for pose detection
        verbose: Print debug information
        device: Device to place output tensors on ('cpu' or 'cuda')
        depth_uncertainty: Uncertainty in depth measurement (std dev in meters)
        tracker_config: Tracker configuration file
        R_rect_to_world: Rotation from rectified camera to world frame.
            Shape (3, 3) broadcast to all frames, or (B, 3, 3) per-frame.
            numpy array or torch.Tensor. When None outputs remain in rectified camera frame.
        t_rect_to_world: Translation from rectified camera to world frame in meters.
            Shape (3,) or (B, 3). numpy array or torch.Tensor.

    Returns:
        - points_3d: 3D joint coordinates [B, 13, 3]
        - C_3d_all: 3D covariance matrices [B, 13, 3, 3]
        - ood_score: Placeholder zeros [B]
        - is_ood: Placeholder False
        - human_detected: Boolean indicating if humans were detected
        - keypoints_2d: 2D joint coordinates [B, 13, 2]
        - uncertainties_2d: 2D uncertainties [B, 13, 2]
        - covariance_xy: 2D covariance (x-y) [B, 13]

    Note:
        - When tracking is enabled and B > 1, frames are processed sequentially
          to maintain proper tracking state across frames
    """
    # Convert to tensors if needed
    if not isinstance(rgb_frames, torch.Tensor):
        rgb_frames = torch.from_numpy(np.array(rgb_frames)).to(device).float()

    if not isinstance(depth_frames, torch.Tensor):
        depth_frames = torch.from_numpy(np.array(depth_frames)).to(device).float()

    # Run 2D pose estimation (handles sequential processing internally if B > 1)
    batch_prediction = process_frame_2d_yolo(
        frames=rgb_frames,
        yolo_pose_model=yolo_pose_model,
        mirror_map=mirror_map,
        enable_tracking=enable_tracking,
        confidence_threshold=confidence_threshold,
        verbose=verbose,
        device=device,
        tracker_config=tracker_config
    )

    # Extract 2D predictions
    keypoints_2d = batch_prediction['keypoints']
    uncertainties_2d = batch_prediction['uncertainties']
    covariance_2d = batch_prediction['covariance_matrix'][:, :, 0, 1]
    human_detected = batch_prediction['mask']

    # Lift 2D to 3D using depth
    points_3d, valid_depth = lift_2d_to_3d_with_depth(
        keypoints_2d, depth_frames, camera_intrinsics, device=device
    )

    # Propagate uncertainty to 3D
    C_3d_all = propagate_uncertainty_2d_to_3d(
        keypoints_2d, uncertainties_2d, covariance_2d,
        depth_frames, camera_intrinsics,
        depth_uncertainty=depth_uncertainty,
        device=device
    )

    # Mark invalid joints
    human_detected_expanded = human_detected.unsqueeze(1)
    combined_valid = valid_depth & human_detected_expanded

    points_3d = points_3d * combined_valid.unsqueeze(-1).float()
    C_3d_all = C_3d_all * combined_valid.unsqueeze(-1).unsqueeze(-1).float()

    # Prepare outputs
    ood_score = torch.zeros(keypoints_2d.shape[0], device=device)
    is_ood = torch.zeros(keypoints_2d.shape[0], dtype=torch.bool, device=device)
    is_ood = torch.logical_or(is_ood, torch.any(combined_valid == 0, dim=1))

    # Optionally rotate from rectified camera frame to world frame.
    if R_rect_to_world is not None:
        if isinstance(R_rect_to_world, np.ndarray):
            R = torch.tensor(R_rect_to_world, dtype=torch.float32, device=device)
        else:
            R = R_rect_to_world.to(device=device, dtype=torch.float32)
        if R.ndim == 2:
            R = R.unsqueeze(0).expand(points_3d.shape[0], -1, -1)  # (B, 3, 3)

        # Rotate 3D points: X_world[b,k] = R[b] @ X_cam[b,k]
        points_3d = torch.einsum('bij,bkj->bki', R, points_3d)

        if t_rect_to_world is not None:
            if isinstance(t_rect_to_world, np.ndarray):
                t = torch.tensor(t_rect_to_world, dtype=torch.float32, device=device)
            else:
                t = t_rect_to_world.to(device=device, dtype=torch.float32)
            if t.ndim == 1:
                t = t.unsqueeze(0)              # (1, 3)
            points_3d = points_3d + t.unsqueeze(1)  # broadcast over joints

        # Rotate covariances: C_world[b,k] = R[b] @ C_cam[b,k] @ R[b]^T
        R_exp = R.unsqueeze(1)                  # (B, 1, 3, 3)
        C_3d_all = R_exp @ C_3d_all @ R_exp.transpose(-1, -2)

    return points_3d, C_3d_all, ood_score, is_ood, human_detected, keypoints_2d, uncertainties_2d, covariance_2d
