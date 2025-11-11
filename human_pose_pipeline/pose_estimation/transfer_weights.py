"""
Transfer RegressFlow weights from PyTorch to JAX/Flax.
This script converts PyTorch checkpoint weights to JAX/Flax format for pose estimation models.
"""

import numpy as np
import pickle
import torch
import os
import argparse

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.regressflow import RegressFlowFlax


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _to_cpu_np(t):
    """Convert torch tensor to numpy."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _to_jax(t):
    """Convert torch tensor to JAX array (for Conv kernels)."""
    if isinstance(t, torch.Tensor):
        return jnp.array(t.detach().cpu().numpy())
    return jnp.array(t)


def _assign(dst_tree, path_list, array, desc):
    """Navigate dict path and assign; assert shape match."""
    node = dst_tree
    for k in path_list[:-1]:
        if k not in node:
            raise KeyError(f"Missing path segment '{k}' while setting {desc}")
        node = node[k]
    leaf = path_list[-1]
    if leaf not in node:
        raise KeyError(f"Missing leaf '{leaf}' while setting {desc}")
    if hasattr(node[leaf], "shape"):
        expect = tuple(node[leaf].shape)
        got = tuple(array.shape)
        if expect != got:
            raise ValueError(f"Shape mismatch for {desc}: expected {expect}, got {got}")
    node[leaf] = array


def transfer_conv(flax_params, path, torch_weight, desc):
    """
    Transfer PyTorch Conv2d layer to Flax Conv layer.
    PyTorch: (out_ch, in_ch, h, w)
    Flax: (h, w, in_ch, out_ch)

    NOTE: We use JAX arrays for Conv kernels to match the original model's
    format, which is important for XLA optimization performance.
    """
    # Transpose from OIHW to HWIO
    weight_np = _to_cpu_np(torch_weight)
    weight_np = np.transpose(weight_np, (2, 3, 1, 0))
    # Convert to JAX array for better XLA performance
    weight_jax = jnp.array(weight_np)
    _assign(flax_params, path + ["kernel"], weight_jax, f"{desc}.weight")


def transfer_batchnorm(flax_params, batch_stats, path, torch_state_dict, prefix, desc):
    """
    Transfer PyTorch BatchNorm2d to Flax BatchNorm.
    PyTorch: weight, bias, running_mean, running_var
    Flax: scale, bias (in params), mean, var (in batch_stats)
    """
    # Transfer parameters (weight -> scale, bias -> bias)
    _assign(flax_params, path + ["scale"], _to_cpu_np(torch_state_dict[f"{prefix}.weight"]), f"{desc}.weight")
    _assign(flax_params, path + ["bias"], _to_cpu_np(torch_state_dict[f"{prefix}.bias"]), f"{desc}.bias")

    # Transfer batch statistics
    _assign(
        batch_stats, path + ["mean"], _to_cpu_np(torch_state_dict[f"{prefix}.running_mean"]), f"{desc}.running_mean"
    )
    _assign(batch_stats, path + ["var"], _to_cpu_np(torch_state_dict[f"{prefix}.running_var"]), f"{desc}.running_var")


def transfer_linear_norm(flax_params, path, torch_weight, torch_bias, desc):
    """
    Transfer PyTorch Linear layer to Flax LinearNorm layer.
    PyTorch: (out, in)
    Flax: (out, in) for LinearNorm (note: different from Dense which is (in, out))
    """
    # LinearNorm in the Flax model uses (out, in) format, not transposed
    _assign(flax_params, path + ["kernel"], _to_cpu_np(torch_weight), f"{desc}.weight")
    _assign(flax_params, path + ["bias"], _to_cpu_np(torch_bias), f"{desc}.bias")


def transfer_basicblock(
    flax_params, batch_stats, torch_state_dict, stage_idx, block_idx, pytorch_prefix, flax_base_path
):
    """Transfer a BasicBlock (ResNet18/34)."""
    # Conv1
    transfer_conv(
        flax_params,
        flax_base_path + ["Conv_0"],
        torch_state_dict[f"{pytorch_prefix}.conv1.weight"],
        f"{pytorch_prefix}.conv1",
    )
    transfer_batchnorm(
        flax_params,
        batch_stats,
        flax_base_path + ["BatchNorm_0"],
        torch_state_dict,
        f"{pytorch_prefix}.bn1",
        f"{pytorch_prefix}.bn1",
    )

    # Conv2
    transfer_conv(
        flax_params,
        flax_base_path + ["Conv_1"],
        torch_state_dict[f"{pytorch_prefix}.conv2.weight"],
        f"{pytorch_prefix}.conv2",
    )
    transfer_batchnorm(
        flax_params,
        batch_stats,
        flax_base_path + ["BatchNorm_1"],
        torch_state_dict,
        f"{pytorch_prefix}.bn2",
        f"{pytorch_prefix}.bn2",
    )

    # Downsample if present (Conv2 and BatchNorm2 in Flax)
    if f"{pytorch_prefix}.downsample.0.weight" in torch_state_dict:
        transfer_conv(
            flax_params,
            flax_base_path + ["Conv_2"],
            torch_state_dict[f"{pytorch_prefix}.downsample.0.weight"],
            f"{pytorch_prefix}.downsample.0",
        )
        transfer_batchnorm(
            flax_params,
            batch_stats,
            flax_base_path + ["BatchNorm_2"],
            torch_state_dict,
            f"{pytorch_prefix}.downsample.1",
            f"{pytorch_prefix}.downsample.1",
        )


def transfer_bottleneck(
    flax_params, batch_stats, torch_state_dict, stage_idx, block_idx, pytorch_prefix, flax_base_path
):
    """Transfer a Bottleneck block (ResNet50/101/152)."""
    # Conv1 (1x1)
    transfer_conv(
        flax_params,
        flax_base_path + ["Conv_0"],
        torch_state_dict[f"{pytorch_prefix}.conv1.weight"],
        f"{pytorch_prefix}.conv1",
    )
    transfer_batchnorm(
        flax_params,
        batch_stats,
        flax_base_path + ["BatchNorm_0"],
        torch_state_dict,
        f"{pytorch_prefix}.bn1",
        f"{pytorch_prefix}.bn1",
    )

    # Conv2 (3x3)
    transfer_conv(
        flax_params,
        flax_base_path + ["Conv_1"],
        torch_state_dict[f"{pytorch_prefix}.conv2.weight"],
        f"{pytorch_prefix}.conv2",
    )
    transfer_batchnorm(
        flax_params,
        batch_stats,
        flax_base_path + ["BatchNorm_1"],
        torch_state_dict,
        f"{pytorch_prefix}.bn2",
        f"{pytorch_prefix}.bn2",
    )

    # Conv3 (1x1)
    transfer_conv(
        flax_params,
        flax_base_path + ["Conv_2"],
        torch_state_dict[f"{pytorch_prefix}.conv3.weight"],
        f"{pytorch_prefix}.conv3",
    )
    transfer_batchnorm(
        flax_params,
        batch_stats,
        flax_base_path + ["BatchNorm_2"],
        torch_state_dict,
        f"{pytorch_prefix}.bn3",
        f"{pytorch_prefix}.bn3",
    )

    # Downsample if present (Conv3 and BatchNorm3 in Flax)
    if f"{pytorch_prefix}.downsample.0.weight" in torch_state_dict:
        transfer_conv(
            flax_params,
            flax_base_path + ["Conv_3"],
            torch_state_dict[f"{pytorch_prefix}.downsample.0.weight"],
            f"{pytorch_prefix}.downsample.0",
        )
        transfer_batchnorm(
            flax_params,
            batch_stats,
            flax_base_path + ["BatchNorm_3"],
            torch_state_dict,
            f"{pytorch_prefix}.downsample.1",
            f"{pytorch_prefix}.downsample.1",
        )


def transfer_resnet_backbone(flax_params, batch_stats, torch_state_dict, architecture_str):
    """Transfer the ResNet backbone (conv1, bn1, and all layer stages)."""
    print("\n  Transferring ResNet backbone...")

    # Initial Conv1 and BN1
    transfer_conv(
        flax_params, ["ResNet50Backbone_0", "Conv_0"], torch_state_dict["preact.conv1.weight"], "preact.conv1"
    )
    transfer_batchnorm(
        flax_params, batch_stats, ["ResNet50Backbone_0", "BatchNorm_0"], torch_state_dict, "preact.bn1", "preact.bn1"
    )

    # Determine block counts based on architecture
    RESNET_LAYER_CONFIGS = {
        "resnet18": [(2, "BasicBlock"), (2, "BasicBlock"), (2, "BasicBlock"), (2, "BasicBlock")],
        "resnet34": [(3, "BasicBlock"), (4, "BasicBlock"), (6, "BasicBlock"), (3, "BasicBlock")],
        "resnet50": [(3, "Bottleneck"), (4, "Bottleneck"), (6, "Bottleneck"), (3, "Bottleneck")],
        "resnet101": [(3, "Bottleneck"), (4, "Bottleneck"), (23, "Bottleneck"), (3, "Bottleneck")],
        "resnet152": [(3, "Bottleneck"), (8, "Bottleneck"), (36, "Bottleneck"), (3, "Bottleneck")],
    }

    layer_config = RESNET_LAYER_CONFIGS[architecture_str]

    # Transfer each stage
    for stage_idx, (num_blocks, block_type) in enumerate(layer_config):
        print(f"    Transferring layer{stage_idx + 1} ({num_blocks} {block_type}s)...")

        for block_idx in range(num_blocks):
            pytorch_prefix = f"preact.layer{stage_idx + 1}.{block_idx}"
            flax_prefix = f"BottleneckStage_{stage_idx}"
            block_prefix = f"Bottleneck_{block_idx}" if block_type == "Bottleneck" else f"BasicBlock_{block_idx}"

            flax_base_path = ["ResNet50Backbone_0", flax_prefix, block_prefix]

            if block_type == "Bottleneck":
                transfer_bottleneck(
                    flax_params, batch_stats, torch_state_dict, stage_idx, block_idx, pytorch_prefix, flax_base_path
                )
            else:  # BasicBlock
                transfer_basicblock(
                    flax_params, batch_stats, torch_state_dict, stage_idx, block_idx, pytorch_prefix, flax_base_path
                )

    print("    ✓ ResNet backbone transferred")


def transfer_regressflow(torch_state_dict, flax_variables, architecture_str):
    """
    Transfer all weights from PyTorch RegressFlow to Flax version.
    """
    params = unfreeze(flax_variables["params"])
    batch_stats = unfreeze(flax_variables["batch_stats"])

    print("\n  Transferring RegressFlow components...")

    # Transfer ResNet backbone
    transfer_resnet_backbone(params, batch_stats, torch_state_dict, architecture_str)

    # Transfer head layers (fc_coord, fc_sigma, fc_sigma2)
    print("\n  Transferring prediction heads...")

    # fc_coord -> LinearNorm_0
    transfer_linear_norm(
        params,
        ["LinearNorm_0"],
        torch_state_dict["fc_coord.linear.weight"],
        torch_state_dict["fc_coord.linear.bias"],
        "fc_coord.linear",
    )

    # fc_sigma -> LinearNorm_1
    transfer_linear_norm(
        params,
        ["LinearNorm_1"],
        torch_state_dict["fc_sigma.linear.weight"],
        torch_state_dict["fc_sigma.linear.bias"],
        "fc_sigma.linear",
    )

    # fc_sigma2 -> LinearNorm_2
    transfer_linear_norm(
        params,
        ["LinearNorm_2"],
        torch_state_dict["fc_sigma2.linear.weight"],
        torch_state_dict["fc_sigma2.linear.bias"],
        "fc_sigma2.linear",
    )

    print("    ✓ Prediction heads transferred")

    return freeze(params), freeze(batch_stats)


def main():
    parser = argparse.ArgumentParser(description="Transfer RegressFlow weights from PyTorch to JAX/Flax")
    parser.add_argument(
        "--architecture", type=str, choices=["resnet18", "resnet50"], default="resnet50", help="ResNet architecture"
    )
    parser.add_argument("--seed", type=int, default=420, help="Random seed used in training")
    args = parser.parse_args()

    print("=" * 70)
    print(f"RegressFlow Weight Transfer: PyTorch → JAX/Flax ({args.architecture.upper()})")
    print("=" * 70)

    # Paths
    model_dir = os.path.join(
        root_dir, "human_pose_pipeline/models/pose_estimation/H36M/RegressFlow", f"seed_{args.seed}"
    )
    arch_name = args.architecture.capitalize()  # resnet18 -> Resnet18, resnet50 -> Resnet50
    # But we need ResNet18, ResNet50
    arch_name = "ResNet" + args.architecture[6:]  # Extract number part
    pytorch_model_path = os.path.join(model_dir, f"model_checkpoint_estimation_{arch_name}.pth")
    output_path = os.path.join(model_dir, f"jax_{args.architecture}_regressflow_params.pickle")

    # Model parameters
    num_joints = 17  # H36M has 17 joints
    fc_filters = [-1]  # Identity

    # Load PyTorch checkpoint
    print(f"\n1. Loading PyTorch checkpoint from: {pytorch_model_path}")
    torch_state_dict = torch.load(pytorch_model_path, map_location="cpu")

    print(f"   ✓ Loaded {len(torch_state_dict)} parameter tensors")

    # Initialize Flax model
    print(f"\n2. Initializing JAX/Flax model ({args.architecture})")
    flax_model = RegressFlowFlax(
        num_joints=num_joints,
        fc_filters=fc_filters,
        architecture_str=args.architecture,
        accept_nchw=True,
        predict_aleatoric_uncertainty=True,
    )

    rng = jax.random.PRNGKey(0)
    # Initialize with dummy input (batch_size=1, channels=3, height=256, width=256)
    dummy_x = jnp.zeros((1, 3, 256, 256), dtype=jnp.float32)
    flax_variables = flax_model.init(rng, dummy_x, train=False)

    print(f"   ✓ Initialized Flax model")
    print(f"   Top-level param keys: {list(flax_variables['params'].keys())}")
    print(f"   Top-level batch_stats keys: {list(flax_variables['batch_stats'].keys())}")

    # Transfer weights
    print(f"\n3. Transferring weights...")
    try:
        flax_params, flax_batch_stats = transfer_regressflow(torch_state_dict, flax_variables, args.architecture)
        print("\n   ✓ Weight transfer completed successfully!")
    except Exception as e:
        print(f"\n   ✗ Weight transfer failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Save model
    print(f"\n4. Saving JAX/Flax model to: {output_path}")

    model_dict = {
        "model": "regressflow",
        "params": flax_params,
        "batch_stats": flax_batch_stats,
        "config": {
            "num_joints": num_joints,
            "fc_filters": fc_filters,
            "architecture_str": args.architecture,
            "accept_nchw": True,
            "predict_aleatoric_uncertainty": True,
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(model_dict, f)

    print(f"   ✓ Saved successfully!")

    print("\n" + "=" * 70)
    print("Weight transfer completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
