"""
Inference Helper for JAX-based Pose Estimation

This module provides inference functions for human pose estimation using JAX models.
Based on Marian's Inference_Helper.py but adapted for JAX instead of PyTorch.
"""

import os
import sys
import logging
import json
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import matplotlib.patches as patches
import matplotlib.lines as mlines

# Add path to access our utilities and models
sys.path.append('../..')
from src.models.wrapper import model_from_string
from human_pose_pipeline.utils.transform_utils import (
    preprocess_image_with_bbox,
    CONFIG,
    convert_coordinates_regressflow_to_pixel,
    transform_coordinates_back_to_original
)

# Define indices for the 13 joints of interest in the human pose (same as Marian's)
JOINT_IDX_13 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def resize_image(pil_image, target_size=(512, 640)):
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


def pose_estimation_2d(pil_image, model, params, batch_stats, human_detector, device_torch, threshold=0.8, visualize=True):
    """
    Complete 2D pose estimation pipeline: resize -> detect humans -> estimate poses.

    Args:
        pil_image (PIL.Image.Image): The input high-resolution image
        model: The JAX pose estimation model
        params: JAX model parameters
        batch_stats: JAX model batch statistics (if available)
        human_detector: The pre-loaded YOLO human detection model (PyTorch)
        device_torch: PyTorch device for human detection
        threshold (float, optional): Confidence threshold for human detection
        visualize (bool, optional): Whether to visualize the results

    Returns:
        List[Dict]: List of dictionaries containing for each detected person:
            - 'keypoints': Joint coordinates [[x1,y1], [x2,y2], ...]
            - 'uncertainties': Standard deviations (placeholder for now)
            - 'covariance': Covariance values (placeholder for now)
    """
    # Step 1: Resize image
    resized_image, original_dimensions, scale_factors = resize_image(pil_image)

    # Step 2: Detect humans
    person_boxes = detect_humans(human_detector, resized_image, device_torch, threshold=0.4)

    if not person_boxes:
        print("No humans detected with the specified threshold.")
        return []

    # Step 3: Perform pose estimation
    return get_pose_estimations_jax(
        resized_image, original_dimensions, scale_factors, person_boxes,
        model, params, batch_stats, visualize
    )


def get_pose_estimations_jax(resized_image, original_dimensions, scale_factors, person_boxes, model, params, batch_stats, visualize=True):
    """
    Perform pose estimation on detected humans using JAX model and return keypoints in original image dimensions.

    Args:
        resized_image (PIL.Image.Image): The resized image
        original_dimensions (tuple): Original image dimensions (width, height)
        scale_factors (tuple): Scale factors for coordinate transformation (scale_x, scale_y)
        person_boxes (list): List of detected human bounding boxes
        model: The JAX pose estimation model
        params: JAX model parameters
        batch_stats: JAX model batch statistics (if available)
        visualize (bool, optional): Whether to visualize the results

    Returns:
        List[Dict]: List of dictionaries containing for each detected person:
            - 'keypoints': Joint coordinates [[x1,y1], [x2,y2], ...]
            - 'uncertainties': Standard deviations (placeholder for now)
            - 'covariance': Covariance values (placeholder for now)
    """
    # Unpack dimensions and scale factors
    original_image_width, original_image_height = original_dimensions
    scale_x, scale_y = scale_factors

    # Convert PIL to numpy for processing
    resized_image_np = np.array(resized_image)

    pose_estimations = []

    # Process each detected person
    for i, bbox in enumerate(person_boxes):
        # Preprocess image with detected bounding box using our JAX utilities
        input_tensor, _, center, scale, trans, processed_bbox = preprocess_image_with_bbox(resized_image_np, bbox)

        # Get model predictions using JAX
        with jax.disable_jit(False):  # Enable JIT for inference
            if batch_stats is not None:
                output = model.apply_test(params, batch_stats, input_tensor)
            else:
                output = model.apply_test(params, input_tensor)

        # Extract predictions - JAX model outputs flattened coordinates
        if isinstance(output, dict):
            # If model returns dictionary with multiple outputs
            pred_joints_flat = output.get('pred_jts', output.get('output', output))
        else:
            # If model returns tensor directly
            pred_joints_flat = output

        # Convert to numpy and reshape
        pred_joints_flat = np.array(pred_joints_flat[0])  # Remove batch dimension
        pred_joints_17 = pred_joints_flat.reshape(17, 2)  # 17 joints × 2 coords

        # Select only the 13 joints of interest (same as Marian's approach)
        pred_joints = pred_joints_17[JOINT_IDX_13]

        # Convert from RegressFlow normalized coordinates [-0.5, 0.5] to pixel coordinates
        img_height, img_width = CONFIG.DATA_PRESET.IMAGE_SIZE
        pred_joints_pixel = convert_coordinates_regressflow_to_pixel(
            pred_joints, img_height, img_width
        )

        # Transform coordinates back to original image space
        pred_joints_original = transform_coordinates_back_to_original(
            pred_joints_pixel, trans, scale_x, scale_y
        )

        # Placeholder uncertainty measures (to be implemented with proper uncertainty quantification)
        # For now, we'll create dummy values to match the expected interface
        uncertainties_original = np.ones_like(pred_joints_original) * 5.0  # 5 pixel std dev
        covariance_original = np.ones(len(pred_joints_original)) * 0.1  # Small covariance

        # Store results for this person
        pose = {
            'keypoints': pred_joints_original.tolist(),
            'uncertainties': uncertainties_original.tolist(),
            'covariance': covariance_original.tolist(),
            'bbox': bbox,
            'center': center.tolist(),
            'scale': scale.tolist()
        }
        pose_estimations.append(pose)

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

def initialize_jax_models(checkpoint_path_jax):
    """
    Initialize and load the JAX pose estimation model.

    Args:
        checkpoint_path_jax (str): Path to the JAX pose estimation model parameters

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

def detect_humans(model, image, device_torch, threshold=0.8):
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
            cv2.putText(image_np, str(joint_idx), (center[0]+5, center[1]-5),
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


def get_pose_estimations(pil_image, model, params, batch_stats, human_detector, device_torch, threshold=0.8, visualize=True):
    """
    Convenience wrapper that matches Marian's function signature
    """
    return pose_estimation_2d(pil_image, model, params, batch_stats, human_detector, device_torch, threshold, visualize)