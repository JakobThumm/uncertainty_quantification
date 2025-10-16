"""
Inference Helper for JAX-based Pose Estimation

This module provides inference functions for human pose estimation using JAX models.
Based on Marian's Inference_Helper.py but adapted for JAX instead of PyTorch.
"""

import json
import pickle
from time import time
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

from human_pose_pipeline.utils.batched_transform_torch import (
    bboxes_np_to_torch_tensor,
    compute_transforms_from_bboxes,
    crop_and_normalize_frames_gpu,
    detect_humans_in_batch,
    frames_np_to_torch_tensor
)
from src.models.wrapper import model_from_string
from human_pose_pipeline.utils.transform_utils import (
    preprocess_image_with_bbox,
    transform_predictions_to_original_space
)

# TODO: Swtich to resize_images_batched and funtions from n_pose_pipeline.utils.batched_transform_torch 
from human_pose_pipeline.utils.gpu_accelerated_utils import (
    resize_image_gpu,
    extract_bounding_box_images_gpu
)

from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_IDX_13_MODEL,
    TRANSFORM_IMAGE_SIZE,
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
    return joints[mapping]


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


def resize_images_batched(frames, target_size=YOLO_IMAGE_SIZE):
    """
    Resize a batch of frames using OpenCV

    Args:
        frames: List or array of numpy arrays (H, W, 3)
        target_size: (width, height) for output

    Returns:
        resized_frames: List of PIL Images
        scale_factors: List of (scale_x, scale_y) tuples
    """
    resized_frames = []
    scale_factors = []

    for frame in frames:
        h, w = frame.shape[:2]
        target_w, target_h = target_size

        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        pil_image = Image.fromarray(resized)

        scale_x = w / target_w
        scale_y = h / target_h

        resized_frames.append(pil_image)
        scale_factors.append((scale_x, scale_y))

    return resized_frames, scale_factors


def pose_estimation_2d(
        pil_image, pose_estimation_jit_fn, params, batch_stats, human_detector, device_torch, score_fn=None,
        human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
        ood_threshold=OOD_THRESHOLD,
        parallelize=False,
        num_output_joints=17,
        use_gpu_acceleration=False):
    """
    Complete 2D pose estimation pipeline: resize -> detect humans -> estimate poses.

    Args:
        pil_image (PIL.Image.Image): The input high-resolution image
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
    bounding_box_images = extract_bounding_box_images(
        full_image=pil_image,
        human_detector=human_detector,
        device_torch=device_torch,
        threshold=human_detection_threshold,
        use_gpu_acceleration=use_gpu_acceleration
    )
    pose_estimations = []

    for i, bounding_box_image_struct in enumerate(bounding_box_images):
        scale_x, scale_y = bounding_box_image_struct['scale_factors_yolo']
        bbox = bounding_box_image_struct['bbox']
        bounding_box_image = bounding_box_image_struct['image']
        center = bounding_box_image_struct['center']
        scale = bounding_box_image_struct['scale']
        trans = bounding_box_image_struct['trans']

        # Run pose prediction and OOD scoring in parallel
        if score_fn is None:
            # No OOD scoring - run pose prediction only
            pred_joints_13, uncertainties_13, covariance_13 = predict_pose(bounding_box_image, pose_estimation_jit_fn, params, batch_stats, num_output_joints)
            ood_score = 0.0
        elif score_fn is not None and parallelize:
            # Run pose prediction and OOD scoring in parallel using thread pool
            executor = get_thread_pool()

            # Submit both tasks to the thread pool simultaneously
            pose_future = executor.submit(predict_pose, bounding_box_image, pose_estimation_jit_fn, params, batch_stats, num_output_joints)
            ood_future = executor.submit(score_fn, bounding_box_image)

            # Wait for BOTH futures to complete simultaneously (more efficient than sequential .result() calls)
            wait([pose_future, ood_future], return_when=ALL_COMPLETED)

            # Get results (these are now instant since both are done)
            pred_joints_13, uncertainties_13, covariance_13 = pose_future.result()
            ood_score = float(np.asarray(ood_future.result()))
        else:
            pred_joints_13, uncertainties_13, covariance_13 = predict_pose(bounding_box_image, pose_estimation_jit_fn, params, batch_stats, num_output_joints)
            ood_score = float(np.asarray(score_fn(bounding_box_image)))
        is_ood = ood_score > ood_threshold

        # Transform back to original image space
        result = transform_predictions_to_original_space(
            pred_joints_13, trans, scale_x, scale_y,
            uncertainties=uncertainties_13,
            covariance=covariance_13
        )
        # Fallback if no uncertainties are predicted
        if result.get('uncertainties') is None:
            result['uncertainties'] = np.ones_like(result['keypoints']) * 10.0  # 10 pixel std dev
        if result.get('covariance') is None:
            result['covariance'] = np.ones(len(result['keypoints'])) * 0.1  # Small covariance

        # Store results for this person
        pose = {
            'keypoints': result['keypoints'].tolist(),
            'uncertainties': result['uncertainties'].tolist(),
            'covariance': result['covariance'].tolist(),
            'bbox': bbox,
            'center': center.tolist(),
            'scale': scale.tolist(),
            'ood_score': ood_score,
            'is_ood': is_ood
        }
        pose_estimations.append(pose)

    return pose_estimations


def extract_bounding_box_images(
        full_image,
        human_detector,
        device_torch,
        threshold=YOLO_CONFIDENCE_THRESHOLD,
        use_gpu_acceleration=False
):
    """
    Extract bounding box images of detected humans from the full image.

    Args:
        full_image (PIL.Image.Image): The input high-resolution image
        human_detector: The pre-loaded YOLO human detection model (PyTorch)
        device_torch: PyTorch device for human detection
        threshold (float, optional): Confidence threshold for human detection
        use_gpu_acceleration (bool, optional): Whether to use GPU-accelerated preprocessing
    Returns:
        list of structs with keys:
            - 'scale_factors_yolo': Scale factors (x, y) from original to YOLO input size
            - 'bbox': Bounding box coordinates [x1, y1, x2, y2]
            - 'image': Cropped bounding box image (PIL.Image)
            - 'center': Center of the bounding box in YOLO image [x, y]
            - 'scale': Width and height of the bounding box in YOLO image [w, h]
            - 'trans': Transformation matrix (2x3) from YOLO image to cropped bbox image
    """
    # Step 1: Resize image
    if use_gpu_acceleration:
        device_str = 'cuda' if str(device_torch).startswith('cuda') else 'cpu'
        resized_image, original_dimensions, scale_factors = resize_image_gpu(full_image, device=device_str)
    else:
        resized_image, original_dimensions, scale_factors = resize_image(full_image)

    # Step 2: Detect humans
    person_boxes = detect_humans(human_detector, resized_image, device_torch, threshold=threshold)

    if not person_boxes:
        print("No humans detected with the specified threshold.")
        return []

    # Step 3: Extract bounding boxes
    if use_gpu_acceleration:
        # Use GPU-accelerated bounding box extraction
        device_str = 'cuda' if str(device_torch).startswith('cuda') else 'cpu'
        resized_image_np = np.array(resized_image)
        bounding_box_images = extract_bounding_box_images_gpu(
            full_image, person_boxes, scale_factors, resized_image_np, device=device_str
        )
    else:
        # Use CPU-based bounding box extraction
        scale_x, scale_y = scale_factors
        resized_image_np = np.array(resized_image)
        bounding_box_images = []

        for i, bbox in enumerate(person_boxes):
            # Transform image to model input dimension from bounding box
            bounding_box_image, _, center, scale, trans, processed_bbox = preprocess_image_with_bbox(resized_image_np, bbox)
            bbox_struct = {
                'scale_factors_yolo': scale_factors,
                'bbox': bbox,
                'image': bounding_box_image,
                'center': center,
                'scale': scale,
                'trans': trans
            }
            bounding_box_images.append(bbox_struct)
    return bounding_box_images


def predict_pose(bounding_box_image, pose_estimation_jit_fn, params, batch_stats, num_output_joints=17):
    """Predict pose for a single bounding box image using the JAX model.

    Args:
        bounding_box_image (np.ndarray): Cropped image of the detected human
        pose_estimation_jit_fn: JIT-compiled pose estimation function
        params: JAX model parameters
        batch_stats: JAX model batch statistics (if available)
        num_output_joints: Number of joints the model outputs (17 for full model, 3 for reduced model)
    Returns:
        tuple: (pred_joints_13, uncertainties_13, covariance_13)
    """
    # Get model predictions using JIT-compiled function
    if batch_stats is not None:
        output = pose_estimation_jit_fn(params, batch_stats, bounding_box_image)
    else:
        output = pose_estimation_jit_fn(params, bounding_box_image)

    # Extract predictions - JAX model outputs (following Marian's approach)
    if isinstance(output, dict):
        # RegressFlowWithAleatoric returns dictionary with uncertainty outputs
        pred_joints = np.array(output['pred_jts'][0])  # Joint coordinates (num_output_joints, 2)
        log_variance = np.array(output.get('log_variance', output.get('pure_sigma', None)))
        if log_variance is not None:
            log_variance = log_variance[0]  # Remove batch dimension (num_output_joints, 2)
        covariance_raw = np.array(output.get('covariance', None))
        if covariance_raw is not None:
            covariance_raw = covariance_raw[0]  # Remove batch dimension (num_output_joints,)
    else:
        # Regular RegressFlow returns tensor directly - reshape from flattened
        pred_joints_flat = np.array(output[0])  # Remove batch dimension
        pred_joints = pred_joints_flat.reshape(num_output_joints, 2)  # num_output_joints × 2 coords
        log_variance = None
        covariance_raw = None

    # Convert log variance to standard deviation (following Marian's approach)
    if log_variance is not None:
        uncertainties = np.sqrt(np.exp(log_variance))  # (num_output_joints, 2)
    else:
        uncertainties = None

    # Handle reduced 3-joint model (nose, left wrist, right wrist)
    if num_output_joints == 3:
        # For 3-joint model: indices are [0=nose, 1=left_wrist, 2=right_wrist]
        # We need to expand to 13 joints by filling missing joints with nose position
        pred_joints_13 = expand_3joints_to_13joints(pred_joints)
        if uncertainties is not None:
            uncertainties_13 = expand_3joints_to_13joints(uncertainties)
        else:
            uncertainties_13 = None
        if covariance_raw is not None:
            # For 3-joint covariance, replicate nose covariance for missing joints
            covariance_13 = np.zeros(13)
            covariance_13[0] = covariance_raw[0]  # Nose
            covariance_13[5] = covariance_raw[1]  # LWrist
            covariance_13[6] = covariance_raw[2]  # RWrist
            covariance_13[1:5] = covariance_raw[0]  # Shoulders and elbows -> nose covariance
            covariance_13[7:] = covariance_raw[0]  # Hips, knees, ankles -> nose covariance
        else:
            covariance_13 = None
    else:
        # Select only the 13 joints of interest (same as Marian's approach)
        pred_joints_13 = pred_joints[JOINT_IDX_13_MODEL]  # (13, 2)
        if uncertainties is not None:
            uncertainties_13 = uncertainties[JOINT_IDX_13_MODEL]  # (13, 2)
        else:
            uncertainties_13 = None
        if covariance_raw is not None:
            covariance_13 = covariance_raw[JOINT_IDX_13_MODEL]  # (13,)
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


def predict_pose_batch(bounding_box_images_batch, pose_estimation_jit_fn, params, batch_stats,
                       num_output_joints=17, score_fn=None):
    """Predict pose for a batch of bounding box images using the JAX model.

    Args:
        bounding_box_images_batch (np.ndarray): Batch of cropped images (B, 3, H, W)
        pose_estimation_jit_fn: JIT-compiled pose estimation function
        params: JAX model parameters
        batch_stats: JAX model batch statistics (if available)
        num_output_joints: Number of joints the model outputs (17 for full model, 3 for reduced model)
        score_fn: Optional OOD score function. If provided, computes OOD scores for the batch.

    Returns:
        tuple: (pred_joints_batch, uncertainties_batch, covariance_batch, ood_scores_batch)
            Each is a numpy array with batch dimension (B, ...)
            ood_scores_batch is None if score_fn is not provided
    """
    # Get model predictions using JIT-compiled function
    if batch_stats is not None:
        output = pose_estimation_jit_fn(params, batch_stats, bounding_box_images_batch)
    else:
        output = pose_estimation_jit_fn(params, bounding_box_images_batch)

    batch_size = bounding_box_images_batch.shape[0]

    # Extract predictions - JAX model outputs (following Marian's approach)
    if isinstance(output, dict):
        # RegressFlowWithAleatoric returns dictionary with uncertainty outputs
        pred_joints_batch = np.array(output['pred_jts'])  # (B, num_output_joints, 2)
        log_variance = np.array(output.get('log_variance', output.get('pure_sigma', None)))
        if log_variance is not None:
            uncertainties_batch = np.sqrt(np.exp(log_variance))  # (B, num_output_joints, 2)
        else:
            uncertainties_batch = None
        covariance_batch = np.array(output.get('covariance', None))
    else:
        # Regular RegressFlow returns tensor directly - reshape from flattened
        pred_joints_flat = np.array(output)  # (B, num_output_joints * 2)
        pred_joints_batch = pred_joints_flat.reshape(batch_size, num_output_joints, 2)
        uncertainties_batch = None
        covariance_batch = None

    # Handle reduced 3-joint model (nose, left wrist, right wrist)
    if num_output_joints == 3:
        raise NotImplementedError("Batch prediction for 3-joint model not implemented. The 3-joint model is used for the OOD detection only.")

    # Select only the 13 joints of interest (same as Marian's approach)
    pred_joints_13_batch = pred_joints_batch[:, JOINT_IDX_13_MODEL]  # (B, 13, 2)
    if uncertainties_batch is not None:
        uncertainties_13_batch = uncertainties_batch[:, JOINT_IDX_13_MODEL]  # (B, 13, 2)
    else:
        uncertainties_13_batch = None
    if covariance_batch is not None:
        covariance_13_batch = covariance_batch[:, JOINT_IDX_13_MODEL]  # (B, 13)
    else:
        covariance_13_batch = None

    # OOD detection: Compute OOD scores for the batch if score_fn is provided
    ood_scores_batch = None
    if score_fn is not None:
        ood_scores_batch = np.asarray(score_fn(bounding_box_images_batch))

    return pred_joints_13_batch, uncertainties_13_batch, covariance_13_batch, ood_scores_batch


def transform_predictions_to_original_space_batch(pred_joints_normalized_batch, transforms_batch,
                                                  scale_factors_batch, uncertainties_batch=None,
                                                  covariance_batch=None):
    """
    Transform batch of model predictions from normalized coordinates back to original image space.

    This function wraps the existing transform_predictions_to_original_space to avoid code duplication.

    Args:
        pred_joints_normalized_batch: Joint coordinates in normalized space, shape (B, N, 2)
        transforms_batch: Affine transformation matrices, shape (B, 2, 3)
        scale_factors_batch: Scale factors from resized to original, shape (B, 2) with (scale_x, scale_y)
        uncertainties_batch: Optional uncertainty values, shape (B, N, 2)
        covariance_batch: Optional covariance values, shape (B, N)

    Returns:
        dict: Dictionary containing:
            - 'keypoints': Joint coordinates in original image space (B, N, 2)
            - 'uncertainties': Scaled uncertainties if provided (B, N, 2)
            - 'covariance': Scaled covariance if provided (B, N)
    """
    batch_size = pred_joints_normalized_batch.shape[0]

    # Process each sample individually using the existing function
    keypoints_list = []
    uncertainties_list = []
    covariance_list = []

    for i in range(batch_size):
        scale_x, scale_y = scale_factors_batch[i]
        uncertainties_i = uncertainties_batch[i] if uncertainties_batch is not None else None
        covariance_i = covariance_batch[i] if covariance_batch is not None else None

        result = transform_predictions_to_original_space(
            pred_joints_normalized_batch[i],
            transforms_batch[i],
            scale_x,
            scale_y,
            uncertainties=uncertainties_i,
            covariance=covariance_i
        )

        keypoints_list.append(result['keypoints'])
        if 'uncertainties' in result:
            uncertainties_list.append(result['uncertainties'])
        if 'covariance' in result:
            covariance_list.append(result['covariance'])

    # Stack results back into batch format
    result_batch = {'keypoints': np.stack(keypoints_list, axis=0)}

    if uncertainties_list:
        result_batch['uncertainties'] = np.stack(uncertainties_list, axis=0)
    if covariance_list:
        result_batch['covariance'] = np.stack(covariance_list, axis=0)

    return result_batch


def match_bboxes_across_frames(batch_bboxes):
    """
    Match bounding boxes across two frames based on proximity of bbox centers.

    When batch_size = 2 (e.g., left and right camera), this ensures that the same person
    is consistently assigned to the same index across both frames.

    Args:
        batch_bboxes: List of bboxes for each frame. Each element can be:
                     - A list of bboxes [[x1, y1, x2, y2], ...]
                     - A single bbox [x1, y1, x2, y2]
                     - None if no detection
                     Format: [bbox_or_list_frame0, bbox_or_list_frame1]

    Returns:
        matched_bboxes: List of two bboxes (one from each frame), matched by proximity.
                       Returns None for frames where no match is found.
                       Format: [bbox_frame0, bbox_frame1] where each is [x1, y1, x2, y2] or None
    """
    if len(batch_bboxes) != 2:
        # If not exactly 2 frames, return the bboxes as-is
        result = []
        for item in batch_bboxes:
            if item is None:
                result.append(None)
            elif isinstance(item, list) and len(item) > 0:
                # Check if it's a list of bboxes or a single bbox
                if isinstance(item[0], (list, np.ndarray)):
                    result.append(item[0])  # Take first bbox
                else:
                    result.append(item)  # It's already a single bbox
            else:
                result.append(None)
        return result

    # Normalize inputs to lists of bboxes
    bboxes_frame0 = batch_bboxes[0]
    bboxes_frame1 = batch_bboxes[1]

    # Convert single bboxes to lists
    if bboxes_frame0 is not None and not isinstance(bboxes_frame0[0], (list, np.ndarray)):
        bboxes_frame0 = [bboxes_frame0]
    if bboxes_frame1 is not None and not isinstance(bboxes_frame1[0], (list, np.ndarray)):
        bboxes_frame1 = [bboxes_frame1]

    # Handle edge cases
    if not bboxes_frame0 and not bboxes_frame1:
        return [None, None]
    if not bboxes_frame0:
        return [None, bboxes_frame1[0] if bboxes_frame1 else None]
    if not bboxes_frame1:
        return [bboxes_frame0[0] if bboxes_frame0 else None, None]

    # Compute bbox centers for frame 0
    centers_frame0 = []
    for bbox in bboxes_frame0:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        centers_frame0.append((center_x, center_y))

    # Compute bbox centers for frame 1
    centers_frame1 = []
    for bbox in bboxes_frame1:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        centers_frame1.append((center_x, center_y))

    # Find the best match using nearest neighbor based on Euclidean distance
    best_match_idx0 = 0
    best_match_idx1 = 0
    min_distance = float('inf')

    for i, center0 in enumerate(centers_frame0):
        for j, center1 in enumerate(centers_frame1):
            dx = center0[0] - center1[0]
            dy = center0[1] - center1[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance < min_distance:
                min_distance = distance
                best_match_idx0 = i
                best_match_idx1 = j

    # Return matched bboxes
    return [bboxes_frame0[best_match_idx0], bboxes_frame1[best_match_idx1]]


def process_frame_2d(frame, pose_estimation_jit_fn, params, batch_stats, human_detector, device_torch,
                     mirror_map, score_fn=None,
                     human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD, ood_threshold=OOD_THRESHOLD,
                     num_output_joints=17, use_gpu_acceleration=True, verbose=True):
    """
    Process a single frame to extract pose with uncertainty (JAX version).

    Args:
        frame: Input frame image
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
    if not isinstance(frame, Image.Image):
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t0 = time()
    pose_estimations = pose_estimation_2d(
        pil_image=frame,
        pose_estimation_jit_fn=pose_estimation_jit_fn,
        params=params,
        batch_stats=batch_stats,
        human_detector=human_detector,
        device_torch=device_torch,
        score_fn=score_fn,
        human_detection_threshold=human_detection_threshold,
        ood_threshold=ood_threshold,
        num_output_joints=num_output_joints,
        use_gpu_acceleration=use_gpu_acceleration
    )
    for i in range(len(pose_estimations)):
        pose_estimations[i]['keypoints'] = joint_mapping(np.array(pose_estimations[i]['keypoints']), mirror_map)
        pose_estimations[i]['uncertainties'] = joint_mapping(np.array(pose_estimations[i]['uncertainties']), mirror_map)
        pose_estimations[i]['covariance'] = joint_mapping(np.array(pose_estimations[i]['covariance']), mirror_map)
        # Construct per-joint 2x2 covariance matrices
        joint_covariances = np.zeros((13, 2, 2))
        for j in range(13):
            joint_covariances[i] = [
                [float(pose_estimations[i]['uncertainties'][j, 0])**2, float(pose_estimations[i]['covariance'][j])],
                [float(pose_estimations[i]['covariance'][j]), float(pose_estimations[i]['uncertainties'][j, 1])**2]
            ]
        pose_estimations[i]['covariance_matrix'] = joint_covariances
    if verbose:
        t1 = time()
        print(f"Total frame processing time (detection + pose estimation): {t1 - t0:.3f} seconds")
    return pose_estimations


def process_frames_batch_2d(frames_np, pose_estimation_jit_fn, params, batch_stats, human_detector, device_torch,
                            mirror_map, score_fn=None,
                            human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD, ood_threshold=OOD_THRESHOLD,
                            num_output_joints=17, match_detected_bboxes=True, verbose=True):
    """
    Process multiple frames in batch using GPU-accelerated infrastructure.

    This function is designed to handle batch processing of frames, particularly for stereo camera setups
    where you have left and right camera images. When batch_size=2, it can match detected people across
    both frames to ensure consistent tracking.

    Args:
        frames_np: List of numpy arrays (H, W, 3) in RGB format
        pose_estimation_jit_fn: JIT-compiled pose estimation function
        params: JAX model parameters
        batch_stats: JAX model batch statistics
        human_detector: YOLO human detector
        device_torch: PyTorch device for YOLO
        mirror_map: Joint mapping to correct left/right swapping
        score_fn: OOD score function. If None -> No OOD scoring.
        human_detection_threshold: Confidence threshold for human detection
        ood_threshold: Threshold for OOD detection
        num_output_joints: Number of joints the model outputs
        match_detected_bboxes: Whether to match detected bboxes across frames (only for batch_size=2)
        verbose: Whether to print timing information

    Returns:
        List of pose estimation results, one per frame. Each element contains:
            - 'keypoints': Joint coordinates [[x1,y1], [x2,y2], ...]
            - 'uncertainties': Standard deviations
            - 'covariance': Covariance values
            - 'covariance_matrix': Per-joint 2x2 covariance matrices
            - 'bbox': Bounding box [x1, y1, x2, y2]
            - 'center': Center of the bounding box [x, y]
            - 'scale': Width and height of the bounding box [w, h]
            - 'ood_score': OOD score (0 if no score_fn provided)
            - 'is_ood': Boolean for OOD classification
    """
    device = 'cuda' if str(device_torch).startswith('cuda') else 'cpu'

    t_start = time()

    # Step 1 & 2: Resize frames to YOLO input size
    batch_frames_resized, batch_scale_factors = resize_images_batched(
        frames_np, target_size=YOLO_IMAGE_SIZE
    )

    # Step 3: Detect humans in batch
    batch_bboxes = detect_humans_in_batch(
        human_detector=human_detector,
        batch_frames_resized=batch_frames_resized,
        human_detection_threshold=human_detection_threshold,
        verbose=verbose
    )

    # Step 4: Match bounding boxes across frames if requested and batch_size=2
    if match_detected_bboxes and len(batch_bboxes) == 2:
        matched_bboxes = match_bboxes_across_frames(batch_bboxes)
        # Wrap in list to make it compatible with the rest of the pipeline
        batch_bboxes = [[bbox] if bbox is not None else [] for bbox in matched_bboxes]

    # Filter out frames with no detections
    valid_frame_indices = []
    valid_bboxes = []
    valid_frames_resized = []
    valid_scale_factors = []

    for i, bboxes_for_frame in enumerate(batch_bboxes):
        if bboxes_for_frame:  # If at least one bbox detected
            # For now, take only the first detection per frame
            valid_frame_indices.append(i)
            valid_bboxes.append(bboxes_for_frame[0])
            valid_frames_resized.append(np.array(batch_frames_resized[i]))
            valid_scale_factors.append(batch_scale_factors[i])

    if not valid_bboxes:
        if verbose:
            print("No humans detected in any frame.")
        # Return empty results for each frame
        return [[] for _ in range(len(frames_np))]

    # Step 5: Convert to torch tensors and compute transforms
    frames_tensor = frames_np_to_torch_tensor(valid_frames_resized, device=device)  # (B, 3, H, W)
    bboxes_tensor = bboxes_np_to_torch_tensor(valid_bboxes, device=device)  # (B, 4)
    output_image_size = (TRANSFORM_IMAGE_SIZE[0], TRANSFORM_IMAGE_SIZE[1])  # (W, H)

    centers, scales, transforms = compute_transforms_from_bboxes(
        bboxes_tensor, output_image_size, device=device)  # (B, 2), (B, 2), (B, 2, 3)

    images_preprocessed = crop_and_normalize_frames_gpu(
        frames_tensor, transforms, output_image_size)  # (B, 3, H, W)

    # Convert to numpy for JAX model (JAX expects numpy/jax arrays, not torch tensors)
    images_preprocessed_np = images_preprocessed.cpu().numpy()

    # Step 6: Run pose prediction and OOD detection on the batch of preprocessed images
    pred_joints_13_batch, uncertainties_13_batch, covariance_13_batch, ood_scores_batch = predict_pose_batch(
        images_preprocessed_np, pose_estimation_jit_fn, params, batch_stats, num_output_joints, score_fn
    )

    # Step 7: Transform predictions back to original image space
    transforms_np = transforms.cpu().numpy()
    scale_factors_np = np.array(valid_scale_factors)

    results_batch = transform_predictions_to_original_space_batch(
        pred_joints_13_batch, transforms_np, scale_factors_np,
        uncertainties_13_batch, covariance_13_batch
    )

    # Step 8: Apply mirror mapping to correct left/right swapping
    keypoints_batch = results_batch['keypoints']
    uncertainties_batch = results_batch.get('uncertainties')
    covariance_batch = results_batch.get('covariance')

    for i in range(len(valid_bboxes)):
        keypoints_batch[i] = joint_mapping(keypoints_batch[i], mirror_map)
        if uncertainties_batch is not None:
            uncertainties_batch[i] = joint_mapping(uncertainties_batch[i], mirror_map)
        if covariance_batch is not None:
            covariance_batch[i] = joint_mapping(covariance_batch[i], mirror_map)

    # Step 9: Construct per-joint 2x2 covariance matrices and format results
    pose_estimations_valid = []
    centers_np = centers.cpu().numpy()
    scales_np = scales.cpu().numpy()

    for i in range(len(valid_bboxes)):
        # Fallback if no uncertainties are predicted
        if uncertainties_batch is None:
            uncertainties_i = np.ones_like(keypoints_batch[i]) * 10.0  # 10 pixel std dev
        else:
            uncertainties_i = uncertainties_batch[i]

        if covariance_batch is None:
            covariance_i = np.ones(13) * 0.1  # Small covariance
        else:
            covariance_i = covariance_batch[i]

        # Construct per-joint 2x2 covariance matrices
        joint_covariances = np.zeros((13, 2, 2))
        for j in range(13):
            joint_covariances[j] = [
                [float(uncertainties_i[j, 0])**2, float(covariance_i[j])],
                [float(covariance_i[j]), float(uncertainties_i[j, 1])**2]
            ]

        # Get OOD score for this sample (already computed in step 6)
        ood_score = ood_scores_batch[i]
        is_ood = ood_score > ood_threshold

        pose = {
            'keypoints': keypoints_batch[i].tolist(),
            'uncertainties': uncertainties_i.tolist(),
            'covariance': covariance_i.tolist(),
            'covariance_matrix': joint_covariances.tolist(),
            'bbox': valid_bboxes[i],
            'center': centers_np[i].tolist(),
            'scale': scales_np[i].tolist(),
            'ood_score': ood_score,
            'is_ood': is_ood
        }
        pose_estimations_valid.append(pose)

    # Step 11: Organize results per original frame
    results_per_frame = [[] for _ in range(len(frames_np))]
    for i, frame_idx in enumerate(valid_frame_indices):
        results_per_frame[frame_idx] = [pose_estimations_valid[i]]

    if verbose:
        t_end = time()
        print(f"Batch processing time ({len(frames_np)} frames): {t_end - t_start:.3f} seconds")

    return results_per_frame 


def initialize_human_detector(device_torch=None):
    """
    Initialize the human detection model (YOLOv5).

    Args:
        device_torch (str or torch.device, optional): PyTorch device for human detection

    Returns:
        tuple: (human_detector, device_torch)
    """
    import torch  # Import torch only when needed for YOLOv5

    # Set up PyTorch device for human detection (GPU if available, else CPU)
    if device_torch is None:
        device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_torch = torch.device(device_torch)
    print(f"Using PyTorch device for human detection: {device_torch}")

    # Initialize YOLOv5 human detector
    human_detector = get_human_detector(device_torch)

    return human_detector, device_torch


def initialize_jax_models(checkpoint_path_jax, use_uncertainty=False):
    """
    Initialize and load the JAX pose estimation model.

    Args:
        checkpoint_path_jax (str): Path to the JAX pose estimation model parameters
        use_uncertainty (bool): Whether to use RegressFlowWithAleatoric for uncertainty estimation

    Returns:
        tuple: (pose_estimation_jit_fn, jax_params, jax_batch_stats)
    """

    # Load JAX pose estimation model
    print(f'Loading JAX pose estimation model from {checkpoint_path_jax}...')

    # Parse the model path to extract components
    # Expected format: models_tianle/H36M/RegressFlow/seed_420/finetuned_h36m_regressflow_pred_*
    path_parts = checkpoint_path_jax.split('/')

    # Find the directory containing the model files
    if 'models_tianle' in checkpoint_path_jax:
        base_dir = '/'.join(path_parts[:-1])  # Remove filename
        run_name = path_parts[-1].replace('_args.json', '').replace('_params.pickle', '')

        # Load model arguments
        args_file = f"{base_dir}/{run_name}_args.json"
        params_file = f"{base_dir}/{run_name}_params.pickle"
    else:
        # Direct file paths
        args_file = checkpoint_path_jax.replace('.pickle', '_args.json').replace('_params', '_args')
        params_file = checkpoint_path_jax if checkpoint_path_jax.endswith('.pickle') else f"{checkpoint_path_jax}_params.pickle"

    # Load model configuration
    with open(args_file, 'r') as f:
        args_dict = json.load(f)

    # Load model parameters
    with open(params_file, 'rb') as f:
        params_dict = pickle.load(f)

    # Create JAX model instance
    model = model_from_string(
        model_name=args_dict["model"],
        output_dim=args_dict["output_dim"]
    )

    # Extract parameters and batch statistics
    params = params_dict["params"]
    batch_stats = params_dict.get("batch_stats", None)

    print("JAX pose estimation model loaded successfully.")
    print(f"  - Model type: {args_dict['model']}")
    print(f"  - Output dim: {args_dict['output_dim']}")
    print(f"  - Has batch stats: {batch_stats is not None}")

    # Create JIT-compiled inference function for maximum performance
    print("Compiling JIT inference function...")
    if batch_stats is not None:
        pose_estimation_jit_fn = jax.jit(lambda p, bs, x: model.apply_test(p, bs, x))
    else:
        pose_estimation_jit_fn = jax.jit(lambda p, x: model.apply_test(p, x))
    print("JIT compilation complete!")

    return pose_estimation_jit_fn, params, batch_stats

def get_human_detector(device_torch):
    """
    Initialize the YOLO model for human detection using ultralytics.

    Args:
        device_torch: PyTorch device ('cuda' or 'cpu')

    Returns:
        YOLO: Loaded YOLO model
    """
    try:
        from ultralytics import YOLO
        import torch

        print("Loading YOLO human detector...")

        # Load YOLO model
        model = YOLO("yolo11n.pt")  # Fast and accurate

        # Move to appropriate device
        if device_torch == 'cuda' or str(device_torch).startswith('cuda'):
            model.to('cuda')
            print("YOLO human detector loaded successfully on GPU.")

            # Warmup inference for GPU
            import numpy as np
            from PIL import Image
            dummy_image = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
            _ = model.predict(dummy_image, verbose=False)
            torch.cuda.synchronize()
            print("GPU warmup completed.")

        else:
            print("YOLO human detector loaded successfully on CPU.")

        return model

    except Exception as e:
        print(f"Error loading YOLO: {e}")
        print("Falling back to CPU-only YOLO...")
        try:
            from ultralytics import YOLO
            model = YOLO("yolo11n.pt")
            print("YOLO loaded on CPU as fallback.")
            return model
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            raise e2

def detect_humans(model, image, device_torch, threshold=0.8, verbose=False):
    """
    Detect humans in an image using YOLO (ultralytics).

    Args:
        model: YOLO model from ultralytics
        image (PIL.Image): Input image
        device_torch: PyTorch device (for compatibility, not used with ultralytics)
        threshold (float): Detection confidence threshold

    Returns:
        list: List of bounding boxes for detected humans
            Each box is [x1, y1, x2, y2] in image coordinates
    """
    try:
        # Run YOLO prediction
        results = model.predict(image, conf=threshold, verbose=False)
        person_boxes = []

        # Extract detections from first result
        if len(results) > 0:
            detections = results[0].boxes
            if detections is not None:
                # Get boxes, confidences, and classes
                boxes = detections.xyxy.cpu().numpy()  # xyxy format
                confidences = detections.conf.cpu().numpy()
                classes = detections.cls.cpu().numpy()

                # Filter for person class (class 0 in COCO)
                for i, cls in enumerate(classes):
                    if int(cls) == 0 and confidences[i] >= threshold:
                        person_boxes.append(boxes[i].tolist())
        if verbose:
            print(f"Detected {len(person_boxes)} humans with confidence >= {threshold}")
        return person_boxes

    except Exception as e:
        print(f"Error in human detection: {e}")
        import traceback
        traceback.print_exc()
        return []

def visualize_pose_estimation_results(pil_image, pose_estimations, save_path=None):
    """
    Visualize pose estimation results on the original image

    Args:
        pil_image (PIL.Image): Original input image
        pose_estimations (list): List of pose estimation results
        save_path (str, optional): Path to save the visualization
    """
    # Convert PIL to numpy for OpenCV operations
    image_np = np.array(pil_image)

    # Define colors for different people
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    # Define joint connections for skeleton drawing
    joint_connections = [
        (0, 1), (0, 2),  # Head to shoulders
        (1, 3), (3, 5),  # Left arm
        (2, 4), (4, 6),  # Right arm
        (1, 7), (2, 8),  # Shoulders to hips
        (7, 8),  # Hip connection
        (7, 9), (9, 11),  # Left leg
        (8, 10), (10, 12)  # Right leg
    ]

    for person_idx, pose_data in enumerate(pose_estimations):
        color = colors[person_idx % len(colors)]
        keypoints = np.array(pose_data['keypoints'])

        # Draw skeleton connections
        for connection in joint_connections:
            start_idx, end_idx = connection
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = tuple(map(int, keypoints[start_idx]))
                end_point = tuple(map(int, keypoints[end_idx]))
                cv2.line(image_np, start_point, end_point, color, 2)

        # Draw keypoints
        for joint_idx, keypoint in enumerate(keypoints):
            center = tuple(map(int, keypoint))
            cv2.circle(image_np, center, 4, color, -1)
            cv2.putText(image_np, str(joint_idx), (center[0] + 5, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw bounding box
        if 'bbox' in pose_data:
            bbox = pose_data['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

    # Convert back to PIL for display/saving
    result_image = Image.fromarray(image_np)

    if save_path:
        result_image.save(save_path)
        print(f"Visualization saved to: {save_path}")

    return result_image
