#!/usr/bin/env python3
"""
Human Pose OOD Detection using Sketching Lanczos

This module integrates the human pose estimation pipeline with sketching Lanczos
OOD detection methods. It handles the preprocessing of pose data and applies
the low-memory Lanczos uncertainty quantification for OOD detection.
"""

import os
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Tuple
from PIL import Image
import torch

# Import sketching Lanczos functionality
from src.ood_scores.lm_lanczos import low_memory_lanczos_score_fun

# Import pose estimation functionality
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    process_frame_2d
)

from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IMAGE_SIZE
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


class PoseDataWrapper:
    """
    Wrapper class to make pose data compatible with the sketching Lanczos interface.

    This class handles the conversion between pose estimation data and the format
    expected by the OOD detection methods.
    """

    def __init__(self, pose_data_list, labels=None):
        """
        Args:
            pose_data_list: List of pose data (images, keypoints, etc.)
            labels: Optional labels for the data
        """
        self.data = pose_data_list
        self.labels = labels if labels is not None else [0] * len(pose_data_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return data in format expected by sketching Lanczos:
        (input_data, label) where input_data is what the model expects
        """
        return self.data[idx], self.labels[idx]


def process_pose_dataset_for_ood(
    dataset,
    model,
    params,
    batch_stats,
    human_detector,
    device_torch,
    max_samples=None,
    apply_transforms=None
):
    """
    Process a pose dataset and extract features for OOD detection.

    Args:
        dataset: Dataset containing images and poses
        model: JAX pose estimation model
        params: Model parameters
        batch_stats: Model batch statistics
        human_detector: YOLO human detector
        device_torch: PyTorch device
        max_samples: Maximum number of samples to process
        apply_transforms: Optional transformation function for images (e.g., for tigers)

    Returns:
        processed_data: List of processed pose data suitable for OOD detection
        metadata: Dictionary with processing metadata
    """
    processed_data = []
    successful_detections = 0
    total_samples = 0

    print(f"Processing dataset for OOD detection...")

    # Process dataset samples
    for idx, sample in enumerate(dataset):
        if max_samples is not None and total_samples >= max_samples:
            break

        try:
            # Handle different dataset formats
            if hasattr(sample, 'keys'):  # Dictionary format
                image_tensor = sample['image']
                # Convert tensor to PIL Image if needed
                if torch.is_tensor(image_tensor):
                    # Denormalize if needed
                    if image_tensor.max() <= 1.0:
                        mean = torch.tensor([0.485, 0.456, 0.406])
                        std = torch.tensor([0.229, 0.224, 0.225])
                        image_tensor = image_tensor * std[:, None, None] + mean[:, None, None]
                        image_tensor = torch.clamp(image_tensor, 0, 1)

                    image_np = (image_tensor.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
                    image_pil = Image.fromarray(image_np)
                else:
                    image_pil = sample['image']
            else:
                # Assume tuple format (image, label)
                image_pil = sample[0]

            # Apply transformations if provided (e.g., for tiger images)
            if apply_transforms is not None:
                image_pil = apply_transforms(image_pil)

            # Run pose estimation to get model input format
            pose_predictions = process_frame_2d(
                frame=image_pil,
                model=model,
                params=params,
                batch_stats=batch_stats,
                human_detector=human_detector,
                device_torch=device_torch,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                score_fn=None,  # No OOD scoring for now
                human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD
            )
            # Take the first detected person
            pose = pose_predictions[0]['keypoints']

            # Check if pose detection was successful
            if np.sum(np.abs(pose)) > 0:
                # Convert pose to format expected by model
                # For RegressFlow, we need to flatten and normalize the pose
                pose_flattened = pose.flatten()  # Shape: (26,) for 13 joints * 2 coords

                # Normalize to [-0.5, 0.5] range (as expected by RegressFlow)
                # Assuming pose coordinates are in pixel space, normalize by image size
                img_height, img_width = 256, 256  # Default image size
                if hasattr(image_pil, 'size'):
                    img_width, img_height = image_pil.size

                pose_normalized = np.zeros_like(pose_flattened)
                pose_normalized[0::2] = pose_flattened[0::2] / img_width - 0.5  # x coordinates
                pose_normalized[1::2] = pose_flattened[1::2] / img_height - 0.5  # y coordinates

                processed_data.append(pose_normalized)
                successful_detections += 1
            else:
                # No pose detected, skip this sample
                print(f"No pose detected in sample {idx}, skipping...")

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue

        total_samples += 1

        if idx % 10 == 0:
            print(f"Processed {idx+1} samples, {successful_detections} successful detections")

    metadata = {
        'total_samples': total_samples,
        'successful_detections': successful_detections,
        'detection_rate': successful_detections / total_samples if total_samples > 0 else 0.0
    }

    print(f"Dataset processing complete: {successful_detections}/{total_samples} successful detections")
    return processed_data, metadata


def create_pose_dataloader_for_ood(processed_data, batch_size=4, shuffle=False):
    """
    Create a dataloader compatible with sketching Lanczos from processed pose data.

    Args:
        processed_data: List of processed pose data
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data

    Returns:
        dataloader: PyTorch DataLoader compatible with sketching Lanczos
    """
    from torch.utils.data import DataLoader

    # Convert to JAX arrays
    data_arrays = [jnp.array(data) for data in processed_data]

    # Create wrapper dataset
    wrapper_dataset = PoseDataWrapper(data_arrays)

    # Create dataloader
    dataloader = DataLoader(
        wrapper_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Avoid multiprocessing issues with JAX
    )

    return dataloader


def setup_pose_ood_detection(
    model_path: str,
    args_dict: Dict[str, Any]
) -> Tuple[Any, Any, Any]:
    """
    Set up the pose estimation model and OOD detection framework.

    Args:
        model_path: Path to the trained pose estimation model
        args_dict: Arguments dictionary for OOD detection

    Returns:
        model: JAX pose estimation model
        params: Model parameters
        batch_stats: Model batch statistics
    """
    print("Setting up pose OOD detection...")

    # Initialize JAX pose estimation model
    model, params, batch_stats = initialize_jax_models(model_path)

    print(f"Pose model loaded: {model}")
    print(f"Model parameters loaded successfully")

    return model, params, batch_stats


def compute_pose_ood_scores(
    id_dataset,
    ood_dataset,
    model,
    params,
    batch_stats,
    args_dict: Dict[str, Any],
    apply_tiger_transforms=False
):
    """
    Compute OOD scores for pose estimation using sketching Lanczos.

    Args:
        id_dataset: In-distribution dataset (e.g., H36M)
        ood_dataset: Out-of-distribution dataset (e.g., Tiger poses)
        model: JAX pose estimation model
        params: Model parameters
        batch_stats: Model batch statistics
        args_dict: Arguments for OOD detection
        apply_tiger_transforms: Whether to apply tiger transformations to OOD data

    Returns:
        id_scores: OOD scores for ID data
        ood_scores: OOD scores for OOD data
        metadata: Processing metadata
    """
    print("Computing pose OOD scores using sketching Lanczos...")

    # Initialize human detector
    human_detector, device_torch = initialize_human_detector('cuda')

    # Set up transformation function for tigers if needed
    tiger_transform = None
    if apply_tiger_transforms:
        from human_pose_pipeline.examples.id_vs_ood_pose_prediction import transform_tiger_image_for_human_detection
        tiger_transform = lambda img: transform_tiger_image_for_human_detection(img, target_size=YOLO_IMAGE_SIZE)[0]

    # Process ID dataset
    print("Processing ID dataset...")
    id_data, id_metadata = process_pose_dataset_for_ood(
        id_dataset,
        model,
        params,
        batch_stats,
        human_detector,
        device_torch,
        max_samples=args_dict.get('subsample_trainset', 100),
        apply_transforms=None
    )

    # Process OOD dataset
    print("Processing OOD dataset...")
    ood_data, ood_metadata = process_pose_dataset_for_ood(
        ood_dataset,
        model,
        params,
        batch_stats,
        human_detector,
        device_torch,
        max_samples=args_dict.get('subsample_trainset', 100),
        apply_transforms=tiger_transform
    )

    # Create train dataloader from ID data
    train_loader = create_pose_dataloader_for_ood(
        id_data,
        batch_size=args_dict.get('train_batch_size', 4),
        shuffle=True
    )

    # Update params_dict format for low_memory_lanczos_score_fun
    params_dict = {"params": params}
    if batch_stats is not None:
        params_dict["batch_stats"] = batch_stats

    # Set up args_dict with proper likelihood for pose estimation
    lanczos_args = args_dict.copy()
    lanczos_args.update({
        'likelihood': 'regression',  # Pose estimation is a regression task
        'output_dim': 34,  # RegressFlow output dimension (17 joints * 2 coords)
        'use_hessian': False,
        'serialize_ggn_on_batches': False,
        'use_eigenvals': True
    })

    print("Computing sketching Lanczos score function...")
    # Compute the low-memory Lanczos score function
    score_fun, eigenval, approx_quadratic_form, quadratic_form = low_memory_lanczos_score_fun(
        model,
        params_dict,
        train_loader,
        lanczos_args,
        use_eigenvals=lanczos_args['use_eigenvals']
    )

    print("Computing OOD scores for ID data...")
    # Compute scores for ID data
    id_scores = []
    for data in id_data:
        score = score_fun(jnp.array(data))
        id_scores.append(float(score))

    print("Computing OOD scores for OOD data...")
    # Compute scores for OOD data
    ood_scores = []
    for data in ood_data:
        score = score_fun(jnp.array(data))
        ood_scores.append(float(score))

    metadata = {
        'id_metadata': id_metadata,
        'ood_metadata': ood_metadata,
        'eigenvalues': eigenval,
        'num_id_samples': len(id_scores),
        'num_ood_samples': len(ood_scores)
    }

    print(f"OOD scoring complete!")
    print(f"ID samples: {len(id_scores)}, OOD samples: {len(ood_scores)}")

    return np.array(id_scores), np.array(ood_scores), metadata


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to pose estimation model")
    parser.add_argument("--subsample_trainset", type=int, default=100, help="Number of samples for training")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--test_batch_size", type=int, default=4, help="Test batch size")
    parser.add_argument("--lanczos_lm_iter", type=int, default=81, help="Low-memory Lanczos iterations")
    parser.add_argument("--lanczos_seed", type=int, default=1, help="Lanczos random seed")
    parser.add_argument("--sketch", type=str, default="srft", help="Sketch type")
    parser.add_argument("--sketch_size", type=int, default=10000, help="Sketch size")

    args = parser.parse_args()

    # Convert to dictionary
    args_dict = vars(args)

    # Set up model
    model, params, batch_stats = setup_pose_ood_detection(args.model_path, args_dict)

    print("Pose OOD detection setup complete!")