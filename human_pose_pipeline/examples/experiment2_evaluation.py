#!/usr/bin/env python3
"""
Experiment 2 Evaluation Script - JAX Implementation
3D human pose estimation with uncertainty quantification

This implements the JAX version of Marian's Experiment 2:
- 2D pose estimation from images
- Uncertainty quantification using epistemic uncertainty
- 3D triangulation (future step)
- Evaluation metrics (MPJPE, PCK)
"""

import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import json
import cv2
from torchvision import transforms

# Add root directory to path to access src
sys.path.append('../..')

from src.models.wrapper import model_from_string
from src.datasets.h36m import Human36mDataset
from human_pose_pipeline.evaluation.pose_metrics import (
    comprehensive_pose_evaluation,
    print_evaluation_summary,
    joint_wise_analysis,
    JOINT_NAMES_13,
    JOINT_IDX_13_MODEL
)

def create_compatible_transform():
    """Create transform that produces the expected input size for RegressFlow"""
    return transforms.Compose([
        transforms.Resize((256, 192)),  # RegressFlow expects (256, 192)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

def load_regressflow_model():
    """Load the pre-trained RegressFlow model"""
    print("Loading RegressFlow model...")

    # Model parameters
    model_path = "../../models_tianle"
    dataset_name = "H36M"
    model_name = "RegressFlow"
    run_name = "finetuned_h36m_regressflow_pred"
    seed = 420

    # Load model arguments
    args_file = f"{model_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_args.json"
    with open(args_file, 'r') as f:
        args_dict = json.load(f)

    # Load model parameters
    params_file = f"{model_path}/{dataset_name}/{model_name}/seed_{seed}/{run_name}_params.pickle"
    with open(params_file, 'rb') as f:
        params_dict = pickle.load(f)

    # Create model instance
    model = model_from_string(
        model_name=args_dict["model"],
        output_dim=args_dict["output_dim"]
    )

    # Extract parameters
    params = params_dict["params"]
    batch_stats = params_dict.get("batch_stats", None)

    print(f"✓ Model loaded successfully")
    print(f"  - Output dim: {args_dict['output_dim']} (17 joints × 2 = 34)")
    print(f"  - Has batch stats: {batch_stats is not None}")

    return model, params, batch_stats, args_dict

def load_h36m_dataset(num_samples=50):
    """Load H36M dataset with compatible transform"""
    print(f"Loading H36M dataset (first {num_samples} samples)...")

    h36m_path = "../../datasets/H36M/extracted"

    # Create compatible transform
    transform = create_compatible_transform()

    # Load dataset
    dataset = Human36mDataset(
        base_directory=h36m_path,
        split='validation',  # Use validation for testing
        num_frames_per_video=10,  # More samples per video
        transform=transform,
        image_size=(256, 192)
    )

    print(f"✓ Dataset loaded successfully")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Using first {min(num_samples, len(dataset))} samples")

    return dataset

def evaluate_pose_estimation(model, params, batch_stats, dataset, num_samples=10):
    """Evaluate pose estimation on H36M samples"""
    print(f"Evaluating pose estimation on {num_samples} samples...")

    results = []
    errors = []

    for i in range(min(num_samples, len(dataset))):
        try:
            # Load sample
            sample = dataset[i]
            if isinstance(sample, tuple):
                frame, gt_pose = sample
            else:
                frame = sample['frame']
                gt_pose = sample['pose_13']

            # Convert torch tensor to numpy and reshape for JAX
            if hasattr(frame, 'numpy'):
                frame_np = frame.numpy()
            else:
                frame_np = np.array(frame)

            # Ensure correct shape: (1, 3, H, W) for batch processing
            if frame_np.ndim == 3:
                frame_np = frame_np[None, ...]  # Add batch dimension

            # Convert to JAX array
            frame_jax = jnp.array(frame_np, dtype=jnp.float32)

            print(f"  Sample {i+1}: Input shape {frame_jax.shape}")

            # Forward pass
            if batch_stats is not None:
                pred_pose = model.apply_test(params, batch_stats, frame_jax)
            else:
                pred_pose = model.apply_test(params, frame_jax)

            # Extract predictions (remove batch dimension)
            pred_pose_np = np.array(pred_pose[0])  # Shape: (34,) -> 17 joints × 2 coords

            # Reshape to (17, 2)
            pred_pose_reshaped = pred_pose_np.reshape(17, 2)

            # Convert ground truth to numpy if needed
            if hasattr(gt_pose, 'numpy'):
                gt_pose_np = gt_pose.numpy()
            else:
                gt_pose_np = np.array(gt_pose)

            # Reshape ground truth to (13, 2) or (17, 2) depending on format
            if gt_pose_np.shape[0] == 26:  # Flattened 13 joints
                gt_pose_reshaped = gt_pose_np.reshape(13, 2)
            elif gt_pose_np.shape[0] == 34:  # Flattened 17 joints
                gt_pose_reshaped = gt_pose_np.reshape(17, 2)
            else:
                gt_pose_reshaped = gt_pose_np

            # Store results
            result = {
                'sample_idx': i,
                'pred_pose': pred_pose_reshaped,
                'gt_pose': gt_pose_reshaped,
                'input_shape': frame_jax.shape,
                'pred_range': (float(np.min(pred_pose_np)), float(np.max(pred_pose_np)))
            }
            results.append(result)

            print(f"  ✓ Sample {i+1} processed successfully")
            print(f"    Pred pose range: [{result['pred_range'][0]:.3f}, {result['pred_range'][1]:.3f}]")

        except Exception as e:
            print(f"  ✗ Error processing sample {i+1}: {e}")
            errors.append((i, str(e)))
            continue

    print(f"\nEvaluation completed:")
    print(f"  - Successful samples: {len(results)}")
    print(f"  - Failed samples: {len(errors)}")

    return results, errors

def compute_comprehensive_metrics(results):
    """Compute comprehensive evaluation metrics using MPJPE, PCK, etc."""
    print("Computing comprehensive evaluation metrics...")

    if not results:
        print("No successful results to evaluate")
        return

    # Collect predictions and ground truth
    pred_poses_list = []
    gt_poses_list = []
    valid_results = []

    for result in results:
        pred_pose = result['pred_pose']  # (17, 2) or (13, 2)
        gt_pose = result['gt_pose']

        # Only use results where ground truth and prediction have compatible shapes
        if pred_pose.shape[0] == 17 and gt_pose.shape[0] == 13:
            # Map 17-joint prediction to 13-joint ground truth
            # Use the same mapping as in the dataset
            pred_pose_13 = pred_pose[JOINT_IDX_13_MODEL[:13], :]
            pred_poses_list.append(pred_pose_13)
            gt_poses_list.append(gt_pose)
            valid_results.append(result)
        elif pred_pose.shape[0] == gt_pose.shape[0]:
            # Same number of joints
            pred_poses_list.append(pred_pose)
            gt_poses_list.append(gt_pose)
            valid_results.append(result)

    if not pred_poses_list:
        print("No compatible prediction-ground truth pairs found")
        return

    # Stack into arrays
    pred_poses = np.stack(pred_poses_list)  # (N, num_joints, 2)
    gt_poses = np.stack(gt_poses_list)      # (N, num_joints, 2)

    print(f"Evaluating {len(pred_poses)} samples with {pred_poses.shape[1]} joints each")

    # Compute comprehensive evaluation
    eval_results = comprehensive_pose_evaluation(pred_poses, gt_poses)

    # Print summary
    print_evaluation_summary(eval_results, "Experiment 2 - JAX RegressFlow Evaluation")

    # Compute per-joint analysis
    print("\nPer-joint error analysis:")
    joint_stats = joint_wise_analysis(jnp.array(pred_poses), jnp.array(gt_poses))

    # Sort joints by error and show top 5 most/least accurate
    sorted_joints = sorted(joint_stats.items(), key=lambda x: x[1]['mean_error'])

    print("\nMost accurate joints:")
    for joint_name, stats in sorted_joints[:3]:
        print(f"  {joint_name}: {stats['mean_error']:.3f} ± {stats['std_error']:.3f}")

    print("\nLeast accurate joints:")
    for joint_name, stats in sorted_joints[-3:]:
        print(f"  {joint_name}: {stats['mean_error']:.3f} ± {stats['std_error']:.3f}")

    return eval_results

def main():
    """Main evaluation function"""
    print("=" * 60)
    print("Experiment 2 Evaluation - JAX Human Pose Estimation")
    print("=" * 60)

    try:
        # Load model
        model, params, batch_stats, args_dict = load_regressflow_model()

        # Load dataset
        dataset = load_h36m_dataset(num_samples=100)

        # Evaluate pose estimation
        results, errors = evaluate_pose_estimation(
            model, params, batch_stats, dataset, num_samples=5
        )

        # Compute metrics
        eval_results = compute_comprehensive_metrics(results)

        print("\n" + "=" * 60)
        print("Experiment 2 Evaluation Summary")
        print("=" * 60)

        if results:
            print("✓ JAX RegressFlow pose estimation working!")
            print("✓ H36M dataset loading successfully")
            print("✓ Model produces pose predictions")

            print("\nNext steps for complete Experiment 2:")
            print("  2. Add joint correspondence mapping")
            print("  3. Implement 3D triangulation")
            print("  4. Add uncertainty quantification")
            print("  5. Compare with Marian's PyTorch results")
        else:
            print("✗ No successful pose predictions")
            print("Please check the error messages above")

    except Exception as e:
        print(f"✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()