#!/usr/bin/env python3
"""
Script to reduce RegressFlow model output from 17 joints to 3 joints (head + 2 hands).

This significantly speeds up OOD detection by reducing output dimension t from 17 to 3,
which reduces deployment time from O(tp(log(p)+1+tsk)) to O(3p(log(p)+1+3sk)).

Joint indices in H36M 17-joint format:
- Index 0: Nose (head)
- Index 9: Left Wrist (left hand)
- Index 10: Right Wrist (right hand)

Usage:
    python reduce_regressflow_model.py --model_save_path human_pose_pipeline/models/pose_estimation \
                                       --run_name finetuned_h36m_regressflow_pred \
                                       --seed 420 \
                                       --output_run_name finetuned_h36m_regressflow_pred_3joints
"""

import argparse
import pickle
import json
from pathlib import Path
import jax
import jax.numpy as jnp

from src.models.wrapper import model_from_string


def load_original_model(model_save_path, run_name, seed):
    """Load the original 17-joint model."""
    dataset_name = "H36M"
    model_name = "RegressFlow"

    # Load args
    args_file = Path(model_save_path) / dataset_name / model_name / f"seed_{seed}" / f"{run_name}_args.json"
    with open(args_file, "r") as f:
        args_dict = json.load(f)

    # Load params
    params_file = Path(model_save_path) / dataset_name / model_name / f"seed_{seed}" / f"{run_name}_params.pickle"
    with open(params_file, "rb") as f:
        params_dict = pickle.load(f)

    # Remove 'model' key if present
    params_dict.pop("model", None)

    return params_dict, args_dict


def extract_joint_weights(original_params, joint_indices):
    """
    Extract weights for specific joints from the coordinate head.

    Args:
        original_params: Original model parameters (17 joints * 2 = 34 outputs)
        joint_indices: List of joint indices to keep (e.g., [0, 9, 10] for head and hands)

    Returns:
        Modified parameters with reduced output dimension
    """
    # The coordinate head is in params['params']['LinearNorm_0']
    # It has 'kernel' (weights) and 'bias'

    # Convert joint indices to coordinate indices (each joint has 2 coordinates: x, y)
    coord_indices = []
    for joint_idx in joint_indices:
        coord_indices.extend([joint_idx * 2, joint_idx * 2 + 1])

    coord_indices = jnp.array(coord_indices)

    # Create a copy of the parameters
    new_params = {
        "params": {},
        "batch_stats": original_params.get("batch_stats", None),
        "model": "regressflow",  # Add model identifier (required by pretrained_model_from_string)
    }

    # Copy all parameters except the final coordinate head
    for key, value in original_params["params"].items():
        if key == "LinearNorm_0":
            # This is the coordinate head - we need to slice it
            new_params["params"][key] = {
                "kernel": value["kernel"][coord_indices, :],  # Shape: (num_joints*2, feature_dim)
                "bias": value["bias"][coord_indices],  # Shape: (num_joints*2,)
            }
        else:
            # Copy all other parameters unchanged
            new_params["params"][key] = value

    return new_params


def main():
    parser = argparse.ArgumentParser(description="Reduce RegressFlow model from 17 to 3 joints")
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="human_pose_pipeline/models/pose_estimation",
        help="Path to saved models directory",
    )
    parser.add_argument(
        "--run_name", type=str, default="finetuned_h36m_regressflow_pred", help="Original model run name"
    )
    parser.add_argument("--seed", type=int, default=420, help="Random seed used for training")
    parser.add_argument(
        "--output_run_name",
        type=str,
        default="finetuned_h36m_regressflow_pred_3joints",
        help="Output run name for reduced model",
    )
    parser.add_argument(
        "--joint_indices",
        type=int,
        nargs="+",
        default=[0, 9, 10],
        help="Joint indices to keep (default: [0, 9, 10] for head, left hand, right hand)",
    )

    args = parser.parse_args()

    # Validate joint indices
    if len(args.joint_indices) != 3:
        print(f"Warning: Expected 3 joint indices, got {len(args.joint_indices)}")

    print(f"Loading original model from: {args.model_save_path}")
    print(f"  Run name: {args.run_name}")
    print(f"  Seed: {args.seed}")
    print(f"  Keeping joints: {args.joint_indices}")

    # Load original model
    original_params, original_args = load_original_model(args.model_save_path, args.run_name, args.seed)

    print(f"\nOriginal model:")
    print(f"  Output dimension: {original_args['output_dim']}")
    print(f"  Number of joints: {original_args['output_dim'] // 2}")

    # Extract weights for selected joints
    print(f"\nExtracting weights for joints {args.joint_indices}...")
    reduced_params = extract_joint_weights(original_params, args.joint_indices)

    # Update args for reduced model
    reduced_args = original_args.copy()
    reduced_args["output_dim"] = len(args.joint_indices) * 2
    reduced_args["run_name"] = args.output_run_name
    reduced_args["reduced_from"] = args.run_name
    reduced_args["joint_indices"] = args.joint_indices

    print(f"\nReduced model:")
    print(f"  Output dimension: {reduced_args['output_dim']}")
    print(f"  Number of joints: {len(args.joint_indices)}")

    # Create output directory
    dataset_name = "H36M"
    model_name = "RegressFlow"
    output_dir = Path(args.model_save_path) / dataset_name / model_name / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save reduced model
    args_output_file = output_dir / f"{args.output_run_name}_args.json"
    params_output_file = output_dir / f"{args.output_run_name}_params.pickle"

    print(f"\nSaving reduced model to:")
    print(f"  Args: {args_output_file}")
    print(f"  Params: {params_output_file}")

    with open(args_output_file, "w") as f:
        json.dump(reduced_args, f, indent=2)

    with open(params_output_file, "wb") as f:
        pickle.dump(reduced_params, f)

    print("\nDone! Reduced model saved successfully.")

    # Verify the model can be loaded
    print("\nVerifying reduced model...")
    reduced_model = model_from_string(
        model_name="RegressFlow", output_dim=reduced_args["output_dim"], architecture_str="resnet18"
    )

    # Test with a dummy input
    key = jax.random.PRNGKey(0)
    dummy_input = jax.random.normal(key, (1, 3, 2910, 192))  # NCHW format

    try:
        output = reduced_model.apply_test(reduced_params["params"], reduced_params["batch_stats"], dummy_input)
        print(f"  Model output shape: {output.shape}")
        print(f"  Expected shape: (1, {len(args.joint_indices) * 2})")

        if output.shape == (1, len(args.joint_indices) * 2):
            print("  ✓ Model verification successful!")
        else:
            print("  ✗ Model output shape mismatch!")
    except Exception as e:
        print(f"  ✗ Model verification failed: {e}")

    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Original: 17 joints (34 outputs)")
    print(f"  Reduced:  {len(args.joint_indices)} joints ({len(args.joint_indices) * 2} outputs)")
    print(f"  Speed improvement factor: ~{17 / len(args.joint_indices):.1f}x")
    print(f"  (Deployment time scales as O(tp), reduced from t=17 to t={len(args.joint_indices)})")
    print("=" * 80)


if __name__ == "__main__":
    main()
