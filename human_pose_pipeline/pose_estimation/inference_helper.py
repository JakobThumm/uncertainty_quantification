"""
Inference Helper for JAX-based Pose Estimation

This module provides inference functions for human pose estimation using JAX models.
Based on Marian's Inference_Helper.py but adapted for JAX instead of PyTorch.
"""

from operator import is_
import os
import logging
import json
import pickle
from time import time
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import matplotlib.lines as mlines

from src.models.wrapper import model_from_string
from human_pose_pipeline.utils.transform_utils import (
    preprocess_image_with_bbox,
    transform_predictions_to_original_space
)

from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_IDX_13_MODEL,
    YOLO_IMAGE_SIZE,
    YOLO_CONFIDENCE_THRESHOLD,
    OOD_THRESHOLD
)


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


def pose_estimation_2d(
        pil_image, model, params, batch_stats, human_detector, device_torch, score_fn=None, 
        human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
        ood_threshold=OOD_THRESHOLD):
    """
    Complete 2D pose estimation pipeline: resize -> detect humans -> estimate poses.

    Args:
        pil_image (PIL.Image.Image): The input high-resolution image
        model: The JAX pose estimation model
        params: JAX model parameters
        batch_stats: JAX model batch statistics (if available)
        human_detector: The pre-loaded YOLO human detection model (PyTorch)
        score_fn: Function to compute OOD score from model outputs. If None -> No OOD scoring.
        device_torch: PyTorch device for human detection
        human_detection_threshold (float, optional): Confidence threshold for human detection
        ood_threshold (float, optional): Threshold for OOD detection in pose estimation

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
        threshold=human_detection_threshold
    )
    pose_estimations = []

    for i, bounding_box_image_struct in enumerate(bounding_box_images):
        scale_x, scale_y = bounding_box_image_struct['scale_factors_yolo']
        bbox = bounding_box_image_struct['bbox']
        bounding_box_image = bounding_box_image_struct['image']
        center = bounding_box_image_struct['center']
        scale = bounding_box_image_struct['scale']
        trans = bounding_box_image_struct['trans']
        # Predict human pose
        t0 = time()
        pred_joints_13, uncertainties_13, covariance_13 = predict_pose(bounding_box_image, model, params, batch_stats)
        t1 = time()
        print(f"Pose prediction time: {t1 - t0:.3f} seconds")
        if score_fn is None:
            ood_score = 0.0
            is_ood = False
        else:
            ood_score = score_fn(bounding_box_image)
            ood_score = float(np.asarray(ood_score))
            is_ood = ood_score > ood_threshold
            t2 = time()
            print(f"OOD scoring time: {t2 - t1:.3f} seconds")
        # Transfrom back to original image space
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
        threshold=YOLO_CONFIDENCE_THRESHOLD
):
    """
    Extract bounding box images of detected humans from the full image.

    Args:
        full_image (PIL.Image.Image): The input high-resolution image
        human_detector: The pre-loaded YOLO human detection model (PyTorch)
        device_torch: PyTorch device for human detection
        threshold (float, optional): Confidence threshold for human detection
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
    resized_image, original_dimensions, scale_factors = resize_image(full_image)

    # Step 2: Detect humans
    person_boxes = detect_humans(human_detector, resized_image, device_torch, threshold=threshold)

    if not person_boxes:
        print("No humans detected with the specified threshold.")
        return []

    scale_x, scale_y = scale_factors

    # Convert PIL to numpy for processing
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


def predict_pose(bounding_box_image, model, params, batch_stats):
    """Predict pose for a single bounding box image using the JAX model.

    Args:
        bounding_box_image (np.ndarray): Cropped image of the detected human
        model: The JAX pose estimation model
        params: JAX model parameters
        batch_stats: JAX model batch statistics (if available)
    Returns:
        tuple: (pred_joints_13, uncertainties_13, covariance_13)
    """
    # Get model predictions using JAX
    with jax.disable_jit(False):  # Enable JIT for inference
        if batch_stats is not None:
            output = model.apply_test(params, batch_stats, bounding_box_image)
        else:
            output = model.apply_test(params, bounding_box_image)

    # Extract predictions - JAX model outputs (following Marian's approach)
    if isinstance(output, dict):
        # RegressFlowWithAleatoric returns dictionary with uncertainty outputs
        pred_joints = np.array(output['pred_jts'][0])  # Joint coordinates (17, 2)
        log_variance = np.array(output.get('log_variance', output.get('pure_sigma', None)))
        if log_variance is not None:
            log_variance = log_variance[0]  # Remove batch dimension (17, 2)
        covariance_raw = np.array(output.get('covariance', None))
        if covariance_raw is not None:
            covariance_raw = covariance_raw[0]  # Remove batch dimension (17,)
    else:
        # Regular RegressFlow returns tensor directly - reshape from flattened
        pred_joints_flat = np.array(output[0])  # Remove batch dimension
        pred_joints = pred_joints_flat.reshape(17, 2)  # 17 joints × 2 coords
        log_variance = None
        covariance_raw = None

    # Convert log variance to standard deviation (following Marian's approach)
    if log_variance is not None:
        uncertainties = np.sqrt(np.exp(log_variance))  # (17, 2)
    else:
        uncertainties = None

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


def process_frame_2d(frame, model, params, batch_stats, human_detector, device_torch,
                     mirror_map, score_fn=None,
                     human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD, ood_threshold=OOD_THRESHOLD):
    """
    Process a single frame to extract pose with uncertainty (JAX version).

    Args:
        frame: Input frame image
        model: JAX pose estimation model
        params: JAX model parameters
        batch_stats: JAX model batch statistics
        human_detector: YOLO human detector
        device_torch: PyTorch device for YOLO
        mirror_map: Joint mapping to correct left/right swapping
        score_fn: Function to compute OOD score from model outputs. If None -> No OOD scoring.
        human_detection_threshold (float, optional): Confidence threshold for human detection
        ood_threshold (float, optional): Threshold for OOD detection in pose estimation

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
    pose_estimations = pose_estimation_2d(
        pil_image=frame,
        model=model,
        params=params,
        batch_stats=batch_stats,
        human_detector=human_detector,
        device_torch=device_torch,
        score_fn=score_fn,
        human_detection_threshold=human_detection_threshold,
        ood_threshold=ood_threshold
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

    return pose_estimations


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
        tuple: (jax_model, jax_params, jax_batch_stats)
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

    return model, params, batch_stats

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
