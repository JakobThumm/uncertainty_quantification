#!/usr/bin/env python3
"""
ID vs OOD Pose Prediction Script

This script evaluates pose estimation performance on:
- ID data: Human3.6M dataset (humans)
- OOD data: Tiger-pose dataset (tigers)

The goal is to demonstrate how a model trained on human poses
performs differently on in-distribution vs out-of-distribution data,
setting up the foundation for OOD detection with sketching Lanczos.

Based on pose_estimation_2D.py but adapted for comparative evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torch

from human_pose_pipeline.pose_estimation.inference_helper_batched import predict_pose
from human_pose_pipeline.utils.gpu_accelerated_utils import extract_bounding_box_images_gpu
from human_pose_pipeline.utils.transform_utils import preprocess_image_with_bbox, transform_predictions_to_original_space
from src.datasets.h36m import Human36mDatasetSequence
from src.datasets.tiger_pose import TigerPoseDataset, tiger_pose_to_h36m_format
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    resize_image,
    process_frame_2d,
    joint_mapping
)
from human_pose_pipeline.evaluation.pose_metrics import (
    pck_jax,
    mpjpe_jax
)
from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_IMAGE_SIZE
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def transform_tiger_image_for_human_detection(image_pil, target_size=(256, 256)):
    """
    Transform tiger image to make it more suitable for human detection:
    1. Rotate by 90 degrees (tigers are wider than tall, humans are taller than wide)
    2. Resize to target dimensions matching human image format

    Args:
        image_pil: PIL Image of tiger
        target_size: Target dimensions (width, height)

    Returns:
        transformed_image: PIL Image transformed for human-like detection
        transform_info: Dictionary with transformation parameters for keypoint mapping
    """
    # Step 1: Rotate by 90 degrees clockwise to make tiger taller than wide
    rotated_image = image_pil.rotate(-90, expand=True)

    # Step 2: Resize to target dimensions
    original_width, original_height = rotated_image.size
    transformed_image = rotated_image.resize(target_size, Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.Resampling.LANCZOS)

    # Calculate transformation parameters for keypoint conversion
    transform_info = {
        'rotation_angle': -90,  # Degrees
        'original_size': image_pil.size,  # Before rotation
        'rotated_size': rotated_image.size,  # After rotation
        'final_size': target_size,  # After resize
        'scale_x': target_size[0] / original_width,
        'scale_y': target_size[1] / original_height
    }

    return transformed_image, transform_info


def transform_tiger_keypoints_for_human_detection(keypoints, transform_info):
    """
    Transform tiger keypoints to match the image transformations:
    1. Apply rotation transformation to keypoints (opposite of image rotation)
    2. Apply scaling transformation

    Args:
        keypoints: (N, 2) array of keypoints in original image coordinates
        transform_info: Dictionary from transform_tiger_image_for_human_detection

    Returns:
        transformed_keypoints: (N, 2) array of transformed keypoints
    """
    transformed_kpts = keypoints.copy()

    # Original image dimensions
    orig_w, orig_h = transform_info['original_size']

    # Step 1: Apply 90-degree counterclockwise rotation (opposite of image rotation)
    # Image is rotated -90° (clockwise), so keypoints need +90° (counterclockwise)
    # For 90-degree counterclockwise rotation: (x, y) -> (orig_h - y, x)
    rotated_kpts = np.zeros_like(transformed_kpts)
    rotated_kpts[:, 0] = orig_h - transformed_kpts[:, 1]  # new_x = orig_h - old_y
    rotated_kpts[:, 1] = transformed_kpts[:, 0]  # new_y = old_x

    # Step 2: Apply scaling to match final image size
    final_kpts = rotated_kpts.copy()
    final_kpts[:, 0] *= transform_info['scale_x']
    final_kpts[:, 1] *= transform_info['scale_y']

    return final_kpts


def visualize_pose_comparison(image_pil, predicted_pose, ground_truth_pose, valid_mask, title, save_path, use_tiger_connections=False):
    """
    Visualize predicted pose vs ground truth pose on the image.

    Args:
        image_pil: PIL Image
        predicted_pose: (13, 2) predicted keypoints
        ground_truth_pose: (13, 2) ground truth keypoints
        valid_mask: (13,) boolean mask for valid keypoints
        title: Title for the plot
        save_path: Path to save the visualization
        use_tiger_connections: If True, use tiger skeleton connections for H36M-mapped keypoints
    """
    # Define joint connections for skeleton drawing
    if use_tiger_connections:
        # Tiger connections mapped to H36M indices
        # Correct tiger connections: 0-1, 1-2, 2-3, 3-4, 4-5, 3-7, 7-6, 2-8, 8-9, 2-10, 10-11
        joint_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (3, 7), (7, 6), (2, 8), (8, 9), (2, 10), (10, 11)
        ]
    else:
        # Human skeleton connections (H36M format)
        joint_connections = [
            (0, 1), (0, 2),  # Nose to shoulders
            (1, 3), (3, 5),  # Left arm
            (2, 4), (4, 6),  # Right arm
            (1, 2), (1, 7), (2, 8),  # Shoulders to hips
            (7, 8),  # Connect hips
            (7, 9), (9, 11),  # Left leg
            (8, 10), (10, 12)  # Right leg
        ]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Display image
    ax.imshow(image_pil)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    # Draw ground truth skeleton (green)
    for connection in joint_connections:
        start_idx, end_idx = connection
        if (start_idx < len(ground_truth_pose) and end_idx < len(ground_truth_pose) and
            valid_mask[start_idx] and valid_mask[end_idx]):
            start_point = ground_truth_pose[start_idx]
            end_point = ground_truth_pose[end_idx]
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                   'g-', linewidth=3, alpha=0.8, label='Ground Truth' if connection == joint_connections[0] else "")

    # Draw predicted skeleton (red)
    if np.sum(np.abs(predicted_pose)) > 0:  # Only if we have a prediction
        for connection in joint_connections:
            start_idx, end_idx = connection
            if start_idx < len(predicted_pose) and end_idx < len(predicted_pose):
                start_point = predicted_pose[start_idx]
                end_point = predicted_pose[end_idx]
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]],
                       'r-', linewidth=3, alpha=0.8, label='Predicted' if connection == joint_connections[0] else "")

    # Draw ground truth keypoints (green circles)
    for joint_idx, (keypoint, is_valid) in enumerate(zip(ground_truth_pose, valid_mask)):
        if is_valid:
            ax.plot(keypoint[0], keypoint[1], 'go', markersize=8, alpha=0.8)
            ax.text(keypoint[0]+5, keypoint[1]-5, str(joint_idx),
                   fontsize=8, color='green', fontweight='bold')

    # Draw predicted keypoints (red circles)
    if np.sum(np.abs(predicted_pose)) > 0:
        for joint_idx, keypoint in enumerate(predicted_pose):
            ax.plot(keypoint[0], keypoint[1], 'ro', markersize=8, alpha=0.8)
            ax.text(keypoint[0]+5, keypoint[1]+15, str(joint_idx),
                   fontsize=8, color='red', fontweight='bold')

    # Add legend
    ax.legend(loc='upper right', fontsize=12)

    # Save visualization
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Pose comparison visualization saved: {save_path}")


def preprocess_tiger_data_for_h36m_model(tiger_batch):
    """
    Convert tiger pose data to format suitable for H36M-trained model.

    Args:
        tiger_batch: Batch from tiger pose dataset

    Returns:
        processed_batch: Batch in H36M-compatible format
    """
    # Convert tiger keypoints to H36M format
    tiger_keypoints = tiger_batch['keypoints'].numpy()  # (batch, 12, 2)
    valid_keypoints = tiger_batch['valid_keypoints'].numpy()  # (batch, 12)

    h36m_keypoints, h36m_valid = tiger_pose_to_h36m_format(tiger_keypoints, valid_keypoints)

    return {
        'image': tiger_batch['image'],
        'keypoints': torch.from_numpy(h36m_keypoints),  # (batch, 13, 2)
        'valid_keypoints': torch.from_numpy(h36m_valid),  # (batch, 13)
        'image_path': tiger_batch['image_path'],
        'original_size': tiger_batch.get('original_size', None),
        'dataset_type': 'tiger'
    }


def evaluate_pose_prediction_accuracy(predictions, ground_truth, valid_mask, threshold=0.05):
    """
    Evaluate pose prediction accuracy using PCK and MPJPE metrics.

    Args:
        predictions: (N, 13, 2) predicted keypoints
        ground_truth: (N, 13, 2) ground truth keypoints
        valid_mask: (N, 13) boolean mask for valid keypoints
        threshold: PCK threshold (relative to image size)

    Returns:
        dict: Dictionary with evaluation metrics
    """
    if len(predictions) == 0:
        return {
            'pck': 0.0,
            'mpjpe': float('inf'),
            'num_valid_samples': 0,
            'num_total_samples': 0
        }

    # Filter valid samples
    valid_samples = []
    valid_gt = []
    valid_pred = []

    for i in range(len(predictions)):
        if valid_mask[i].any():  # At least one valid keypoint
            valid_samples.append(i)
            valid_pred.append(predictions[i])
            valid_gt.append(ground_truth[i])

    if len(valid_samples) == 0:
        return {
            'pck': 0.0,
            'mpjpe': float('inf'),
            'num_valid_samples': 0,
            'num_total_samples': len(predictions)
        }

    valid_pred = np.array(valid_pred)
    valid_gt = np.array(valid_gt)
    valid_mask_filtered = valid_mask[valid_samples]

    # Compute PCK (Percentage of Correct Keypoints)
    # Normalize threshold by image size (assume 256x256)
    img_size = 256
    pck_threshold = threshold * img_size

    # Convert to JAX arrays for computation
    import jax.numpy as jnp
    valid_pred_jax = jnp.array(valid_pred)
    valid_gt_jax = jnp.array(valid_gt)

    # Compute PCK with normalized threshold
    pck_score = pck_jax(valid_pred_jax, valid_gt_jax, threshold=threshold, normalize=False)

    # Compute MPJPE (Mean Per Joint Position Error)
    mpjpe_score = mpjpe_jax(valid_pred_jax, valid_gt_jax)

    return {
        'pck': pck_score,
        'mpjpe': mpjpe_score,
        'num_valid_samples': len(valid_samples),
        'num_total_samples': len(predictions)
    }


def predict_poses_on_h36m_dataset(pose_estimation_jit_fn, params, batch_stats, human_detector, device_torch,
                                  dataset, dataset_name, max_samples=None):
    """
    Run pose prediction on H36M dataset using the same approach as pose_estimation_2D.py.
    """
    print(f"\\nEvaluating on {dataset_name} dataset...")

    all_predictions = []
    all_ground_truth = []
    all_valid_masks = []
    all_image_paths = []
    successful_predictions = 0
    total_samples = 0

    samples_processed = 0

    # Store first sample for visualization
    first_sample_data = None
    for idx, sample in enumerate(dataset):
        if max_samples is not None and samples_processed >= max_samples:
            break

        full_sequence = np.array(sample['pose_sequence'])  # (sequence_length, 13, 2)
        frames = sample['frames']  # List of PIL Images

        # Process only first few frames for efficiency
        max_frames = min(10, len(frames))

        for frame_idx in range(max_frames):
            if max_samples is not None and samples_processed >= max_samples:
                break

            frame_image_pil = frames[frame_idx]

            # Run pose estimation
            pose_predictions = process_frame_2d(
                frame=frame_image_pil,
                pose_estimation_jit_fn=pose_estimation_jit_fn,
                params=params,
                batch_stats=batch_stats,
                human_detector=human_detector,
                device_torch=device_torch,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                score_fn=None,  # No OOD scoring for now
                human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD
            )
            # Take the first detected person
            mapped_pose = pose_predictions[0]['keypoints']

            all_predictions.append(mapped_pose)

            # Get ground truth
            gt_keypoints = full_sequence[frame_idx]  # (13, 2)
            valid_mask = np.ones(13, dtype=bool)  # All joints are valid in H36M

            # Store first sample for visualization
            if first_sample_data is None and idx == 0 and frame_idx == 0:
                first_sample_data = {
                    'image': frame_image_pil,
                    'predicted_pose': mapped_pose.copy(),
                    'ground_truth': gt_keypoints.copy(),
                    'valid_mask': valid_mask.copy(),
                    'dataset_name': dataset_name
                }

            if np.sum(np.abs(mapped_pose)) > 0:
                successful_predictions += 1

            all_ground_truth.append(gt_keypoints)
            all_valid_masks.append(valid_mask)
            all_image_paths.append(f"h36m_sample_{idx}_frame_{frame_idx}")

            total_samples += 1
            samples_processed += 1

        # Break after first sequence for quick testing
        if idx == 0:
            break

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    all_valid_masks = np.array(all_valid_masks)

    # Compute evaluation metrics
    metrics = evaluate_pose_prediction_accuracy(
        all_predictions, all_ground_truth, all_valid_masks
    )

    print(f"\n {dataset_name} Results:")
    print(f"  Total samples: {total_samples}")
    print(f"  Successful predictions: {successful_predictions}")
    print(f"  Detection rate: {successful_predictions/total_samples:.2%}")
    print(f"  PCK@0.05: {metrics['pck']:.3f}")
    print(f"  MPJPE: {metrics['mpjpe']:.2f} pixels")

    return {
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'valid_masks': all_valid_masks,
        'image_paths': all_image_paths,
        'metrics': metrics,
        'detection_rate': successful_predictions / total_samples,
        'dataset_name': dataset_name,
        'first_sample': first_sample_data
    }


def predict_poses_on_tiger_dataset(pose_estimation_jit_fn, params, batch_stats, human_detector, device_torch,
                                  processed_batches, dataset_name, max_batches=None):
    """
    Run pose prediction on tiger dataset using processed batches.
    """
    print(f"\\nEvaluating on {dataset_name} dataset...")

    all_predictions = []
    all_ground_truth = []
    all_valid_masks = []
    all_image_paths = []
    successful_predictions = 0
    total_samples = 0

    # Store first sample for visualization
    first_sample_data = None

    for batch_idx, batch in enumerate(tqdm(processed_batches, desc=f"Processing {dataset_name}")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch_size = len(batch['image'])
        total_samples += batch_size

        # Process each image in the batch
        for i in range(batch_size):
            # Convert tensor to PIL Image for processing
            image_tensor = batch['image'][i]

            # Denormalize image tensor
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            image_tensor = image_tensor * std[:, None, None] + mean[:, None, None]
            image_tensor = torch.clamp(image_tensor, 0, 1)

            # Convert to PIL Image
            image_np = (image_tensor.permute(1, 2, 0) * 255).numpy().astype(np.uint8)
            image_pil = Image.fromarray(image_np)

            # Apply tiger transformations to make it more suitable for human detection
            transformed_image, transform_info = transform_tiger_image_for_human_detection(
                image_pil, target_size=YOLO_IMAGE_SIZE
            )

            # Run pose estimation on transformed tiger image
            # Step 1: Resize image
            resized_image, original_dimensions, scale_factors = resize_image(transformed_image)

            # Step 2: Detect humans (skipped, use full image as bounding box)
            # Full image as batch of 1
            person_boxes = [[0.0, 0.0, YOLO_IMAGE_SIZE[0], YOLO_IMAGE_SIZE[1]]]

            # Step 3: Perform pose estimation
            device_str = 'cuda' if str(device_torch).startswith('cuda') else 'cpu'
            resized_image_np = np.array(resized_image)
            bounding_box_images = extract_bounding_box_images_gpu(
                image_pil, person_boxes, scale_factors, resized_image_np, device=device_str
            )
            bounding_box_image, _, center, scale, trans, processed_bbox = preprocess_image_with_bbox(resized_image_np, person_boxes[0])
            pred_joints_13, uncertainties_13, covariance_13 = predict_pose(bounding_box_image, pose_estimation_jit_fn, params, batch_stats)
            result = transform_predictions_to_original_space(
                pred_joints_13, trans, scale[0], scale[1],
                uncertainties=uncertainties_13,
                covariance=covariance_13
            )
            pred_joints_13 = result['keypoints'].tolist()

            first_pose = np.array(pred_joints_13[0])
            first_uncertainty = np.array(result['uncertainties'][0])
            first_covariance = np.array(result['covariance'][0])
            # Apply mirror mapping to correct left/right joint swapping
            mapped_pose = joint_mapping(first_pose, MIRROR_13_JOINT_MODEL_MAP)
            mapped_uncertainty = joint_mapping(first_uncertainty, MIRROR_13_JOINT_MODEL_MAP)
            mapped_covariance = joint_mapping(first_covariance, MIRROR_13_JOINT_MODEL_MAP)
            successful_predictions += 1

            # Get ground truth and transform keypoints to match transformed image
            gt_keypoints_original = batch['keypoints'][i].numpy()  # (13, 2)
            valid_mask = batch['valid_keypoints'][i].numpy()  # (13,)

            # Transform ground truth keypoints to match the transformed image
            gt_keypoints = transform_tiger_keypoints_for_human_detection(
                gt_keypoints_original, transform_info
            )

            all_predictions.append(mapped_pose)

            # Store first sample for visualization
            if first_sample_data is None and batch_idx == 0 and i == 0:
                first_sample_data = {
                    'image': transformed_image,
                    'predicted_pose': mapped_pose.copy(),
                    'ground_truth': gt_keypoints.copy(),
                    'valid_mask': valid_mask.copy(),
                    'dataset_name': dataset_name
                }

            all_ground_truth.append(gt_keypoints)
            all_valid_masks.append(valid_mask)
            all_image_paths.append(batch['image_path'][i] if 'image_path' in batch else f"tiger_sample_{total_samples-batch_size+i}")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    all_valid_masks = np.array(all_valid_masks)

    # Compute evaluation metrics
    metrics = evaluate_pose_prediction_accuracy(
        all_predictions, all_ground_truth, all_valid_masks
    )

    print(f"\\n{dataset_name} Results:")
    print(f"  Total samples: {total_samples}")
    print(f"  Successful predictions: {successful_predictions}")
    print(f"  Detection rate: {successful_predictions/total_samples:.2%}")
    print(f"  PCK@0.05: {metrics['pck']:.3f}")
    print(f"  MPJPE: {metrics['mpjpe']:.2f} pixels")

    return {
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'valid_masks': all_valid_masks,
        'image_paths': all_image_paths,
        'metrics': metrics,
        'detection_rate': successful_predictions / total_samples,
        'dataset_name': dataset_name,
        'first_sample': first_sample_data
    }


def create_comparison_visualization(h36m_results, tiger_results, save_path="id_vs_ood_comparison.png"):
    """
    Create visualization comparing ID vs OOD performance.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Metrics comparison
    datasets = ['H36M (ID)', 'Tiger (OOD)']
    pck_scores = [h36m_results['metrics']['pck'], tiger_results['metrics']['pck']]
    mpjpe_scores = [h36m_results['metrics']['mpjpe'], tiger_results['metrics']['mpjpe']]
    detection_rates = [h36m_results['detection_rate'], tiger_results['detection_rate']]

    # PCK comparison
    axes[0, 0].bar(datasets, pck_scores, color=['blue', 'red'], alpha=0.7)
    axes[0, 0].set_title('PCK@0.05 Comparison')
    axes[0, 0].set_ylabel('PCK Score')
    axes[0, 0].set_ylim(0, 1)

    # MPJPE comparison
    axes[0, 1].bar(datasets, mpjpe_scores, color=['blue', 'red'], alpha=0.7)
    axes[0, 1].set_title('MPJPE Comparison')
    axes[0, 1].set_ylabel('MPJPE (pixels)')

    # Detection rate comparison
    axes[0, 2].bar(datasets, detection_rates, color=['blue', 'red'], alpha=0.7)
    axes[0, 2].set_title('Detection Rate Comparison')
    axes[0, 2].set_ylabel('Detection Rate')
    axes[0, 2].set_ylim(0, 1)

    # Error distribution histograms
    def compute_per_sample_errors(predictions, ground_truth, valid_masks):
        errors = []
        for i in range(len(predictions)):
            if valid_masks[i].any():
                valid_idx = valid_masks[i]
                pred_valid = predictions[i][valid_idx]
                gt_valid = ground_truth[i][valid_idx]
                sample_error = np.mean(np.linalg.norm(pred_valid - gt_valid, axis=1))
                errors.append(sample_error)
        return np.array(errors)

    h36m_errors = compute_per_sample_errors(
        h36m_results['predictions'], h36m_results['ground_truth'], h36m_results['valid_masks']
    )
    tiger_errors = compute_per_sample_errors(
        tiger_results['predictions'], tiger_results['ground_truth'], tiger_results['valid_masks']
    )

    axes[1, 0].hist(h36m_errors, bins=30, alpha=0.7, color='blue', label='H36M (ID)', density=True)
    axes[1, 0].set_title('H36M Error Distribution')
    axes[1, 0].set_xlabel('MPJPE (pixels)')
    axes[1, 0].set_ylabel('Density')

    axes[1, 1].hist(tiger_errors, bins=30, alpha=0.7, color='red', label='Tiger (OOD)', density=True)
    axes[1, 1].set_title('Tiger Error Distribution')
    axes[1, 1].set_xlabel('MPJPE (pixels)')
    axes[1, 1].set_ylabel('Density')

    # Combined error distribution
    axes[1, 2].hist(h36m_errors, bins=30, alpha=0.7, color='blue', label='H36M (ID)', density=True)
    axes[1, 2].hist(tiger_errors, bins=30, alpha=0.7, color='red', label='Tiger (OOD)', density=True)
    axes[1, 2].set_title('Combined Error Distribution')
    axes[1, 2].set_xlabel('MPJPE (pixels)')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison visualization saved: {save_path}")

    return fig


def main():
    """Main function for ID vs OOD pose prediction comparison."""
    print("=" * 60)
    print("ID vs OOD Pose Prediction Comparison")
    print("=" * 60)

    try:
        # Initialize models
        print("Initializing models...")

        # Use uncertainty-enabled model
        models_dir = os.path.join(root_dir, "human_pose_pipeline/models/pose_estimation", "H36M", "RegressFlow", "seed_420")
        checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_with_unc")
        pose_estimation_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path_jax)
        print("JAX RegressFlow model with uncertainty loaded successfully!")

        # Initialize YOLO human detector
        human_detector, device_torch = initialize_human_detector('cuda')
        print("YOLO human detector initialized!")

        # Setup datasets
        print("\\nSetting up datasets...")

        # H36M dataset (ID data) - using the same approach as pose_estimation_2D.py
        h36m_dataset = Human36mDatasetSequence(
            base_directory=os.path.join(root_dir, "datasets", "H36M", "extracted"),
            split='train',
            sequence_length=50  # Smaller sequence for faster processing
        )

        # Tiger pose dataset (OOD data)
        tiger_dataset = TigerPoseDataset(
            root_dir=os.path.join(root_dir, "datasets", "tiger-pose"),
            split='val',  # Use validation set
            image_size=(256, 256)
        )

        tiger_dataloader = torch.utils.data.DataLoader(
            tiger_dataset, batch_size=4, shuffle=False, num_workers=0  # Avoid multiprocessing issues with JAX
        )

        print(f"H36M dataset: {len(h36m_dataset)} samples")
        print(f"Tiger dataset: {len(tiger_dataset)} samples")

        # Process tiger batches to convert to H36M format
        processed_tiger_batches = []
        for batch in tiger_dataloader:
            processed_batch = preprocess_tiger_data_for_h36m_model(batch)
            processed_tiger_batches.append(processed_batch)

        # Create a new dataloader with processed data
        class ProcessedDataset:
            def __init__(self, processed_batches):
                self.batches = processed_batches

            def __iter__(self):
                return iter(self.batches)

            def __len__(self):
                return len(self.batches)

        processed_tiger_dataloader = ProcessedDataset(processed_tiger_batches)

        # Run predictions on both datasets
        max_samples = 20  # Limit for quick testing

        h36m_results = predict_poses_on_h36m_dataset(
            pose_estimation_jit_fn, params, batch_stats, human_detector, device_torch,
            h36m_dataset, "H36M", max_samples=max_samples
        )

        tiger_results = predict_poses_on_tiger_dataset(
            pose_estimation_jit_fn, params, batch_stats, human_detector, device_torch,
            processed_tiger_batches, "Tiger", max_batches=3
        )

        # Create comparison visualization
        print("\\nCreating comparison visualization...")
        create_comparison_visualization(h36m_results, tiger_results)

        # Summary
        print("\\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"H36M (ID) Performance:")
        print(f"  PCK@0.05: {h36m_results['metrics']['pck']:.3f}")
        print(f"  MPJPE: {h36m_results['metrics']['mpjpe']:.2f} pixels")
        print(f"  Detection Rate: {h36m_results['detection_rate']:.2%}")

        print(f"\\nTiger (OOD) Performance:")
        print(f"  PCK@0.05: {tiger_results['metrics']['pck']:.3f}")
        print(f"  MPJPE: {tiger_results['metrics']['mpjpe']:.2f} pixels")
        print(f"  Detection Rate: {tiger_results['detection_rate']:.2%}")

        # Performance degradation analysis
        pck_degradation = (h36m_results['metrics']['pck'] - tiger_results['metrics']['pck']) / h36m_results['metrics']['pck'] if h36m_results['metrics']['pck'] > 0 else float('inf')
        mpjpe_increase = (tiger_results['metrics']['mpjpe'] - h36m_results['metrics']['mpjpe']) / h36m_results['metrics']['mpjpe'] if h36m_results['metrics']['mpjpe'] > 0 else float('inf')
        detection_degradation = (h36m_results['detection_rate'] - tiger_results['detection_rate']) / h36m_results['detection_rate'] if h36m_results['detection_rate'] > 0 else float('inf')

        print(f"\\nPerformance Degradation (ID → OOD):")
        print(f"  PCK degradation: {pck_degradation:.1%}")
        print(f"  MPJPE increase: {mpjpe_increase:.1%}")
        print(f"  Detection rate degradation: {detection_degradation:.1%}")

        # Create individual pose visualizations for first samples
        print("\\nCreating individual pose visualizations...")

        if h36m_results['first_sample'] is not None:
            h36m_sample = h36m_results['first_sample']
            visualize_pose_comparison(
                image_pil=h36m_sample['image'],
                predicted_pose=h36m_sample['predicted_pose'],
                ground_truth_pose=h36m_sample['ground_truth'],
                valid_mask=h36m_sample['valid_mask'],
                title=f"H36M (ID) - First Sample\\nPredicted vs Ground Truth Pose",
                save_path="h36m_first_sample_pose_comparison.png",
                use_tiger_connections=False
            )

        if tiger_results['first_sample'] is not None:
            tiger_sample = tiger_results['first_sample']
            visualize_pose_comparison(
                image_pil=tiger_sample['image'],
                predicted_pose=tiger_sample['predicted_pose'],
                ground_truth_pose=tiger_sample['ground_truth'],
                valid_mask=tiger_sample['valid_mask'],
                title=f"Tiger (OOD) - First Sample\\nPredicted vs Ground Truth Pose",
                save_path="tiger_first_sample_pose_comparison.png",
                use_tiger_connections=True
            )

        print("\\nThis performance gap demonstrates the need for OOD detection!")
        print("Next step: Use sketching Lanczos to detect OOD samples.")

    except Exception as e:
        print(f"\\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()