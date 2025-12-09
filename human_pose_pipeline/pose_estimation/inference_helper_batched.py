"""
Inference Helper for JAX-based Pose Estimation

This module provides inference functions for human pose estimation using JAX models.
Based on Marian's Inference_Helper.py but adapted for JAX instead of PyTorch.
"""

import json
import pickle
from time import time
from typing import Union
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
                     num_output_joints=17, verbose=True, device='cpu'):
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


def process_frame_3d(frames, projection_matrices, pose_estimation_jit_fn, params, batch_stats, human_detector, device_torch,
                     mirror_map, score_fn=None,
                     human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD, ood_threshold=OOD_THRESHOLD,
                     num_output_joints=17, use_gpu_acceleration=True, verbose=True, device='cpu'):
    """
    Process a single frame to extract pose with uncertainty (JAX version).

    Args:
        frames: Input frame images from the left and right camera. Shape: [2*B, H, W, C].
                The first B elements correspond to the left camera,
                the next B elements correspond to the right camera.
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
    left_pose = batch_prediction['keypoints'][:B]
    left_uncertainty = batch_prediction['uncertainties'][:B]  # [B, 13, 2]
    left_covariance_matrix = batch_prediction['covariance_matrix'][:B]  # [B, 13, 2, 2]
    left_ood_score = batch_prediction['ood_score'][:B]
    left_is_ood = batch_prediction['is_ood'][:B]
    left_human_detected = batch_prediction['mask'][:B]
    # Right
    right_pose = batch_prediction['keypoints'][B:]
    right_uncertainty = batch_prediction['uncertainties'][B:]
    right_covariance_matrix = batch_prediction['covariance_matrix'][B:]
    right_ood_score = batch_prediction['ood_score'][B:]
    right_is_ood = batch_prediction['is_ood'][B:]
    right_human_detected = batch_prediction['mask'][B:]
    is_ood = torch.logical_or(left_is_ood, right_is_ood)
    ood_score = torch.max(left_ood_score, right_ood_score)
    human_detected = torch.logical_and(left_human_detected, right_human_detected)
    is_ood = torch.logical_and(is_ood, human_detected)

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
    return points_3d, C_3d_all, ood_score, is_ood


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
