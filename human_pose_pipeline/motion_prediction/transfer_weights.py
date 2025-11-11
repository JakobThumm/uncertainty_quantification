"""
Transfer DCTPoseTransformer weights from PyTorch to JAX/Flax.
This script ONLY performs weight transfer without any testing.
"""

import numpy as np
import pickle
import torch
import os

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from src.models.dct_pose_transformer import DCTPoseTransformer


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _to_cpu_np(t):
    """Convert torch tensor to numpy."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def _to_jax(t):
    """
    Convert torch tensor to JAX array.

    NOTE: Using JAX arrays instead of NumPy arrays is important for XLA
    optimization performance. This can make a significant difference in
    inference speed (2-3x faster).
    """
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


def transfer_linear(flax_params, path, torch_weight, torch_bias, desc):
    """
    Transfer PyTorch Linear layer to Flax Dense layer.

    NOTE: We use JAX arrays for Linear kernels to match the expected format
    and ensure optimal XLA compilation performance.
    """
    # PyTorch: (out, in), Flax: (in, out)
    _assign(flax_params, path + ['kernel'],
            _to_jax(torch_weight.t()), f"{desc}.weight")
    _assign(flax_params, path + ['bias'],
            _to_jax(torch_bias), f"{desc}.bias")


def transfer_layernorm(flax_params, path, torch_weight, torch_bias, desc):
    """Transfer PyTorch LayerNorm to Flax LayerNorm."""
    _assign(flax_params, path + ['scale'],
            _to_jax(torch_weight), f"{desc}.weight")
    _assign(flax_params, path + ['bias'],
            _to_jax(torch_bias), f"{desc}.bias")


def transfer_multihead_attention(flax_params, path, torch_mha_state_dict, desc, num_heads):
    """
    Transfer PyTorch MultiheadAttention to Flax MultiHeadDotProductAttention.

    PyTorch stores: in_proj_weight, in_proj_bias, out_proj.weight, out_proj.bias
    Flax stores: query.kernel, key.kernel, value.kernel, out.kernel (with multi-head structure)
    """
    # PyTorch in_proj combines Q, K, V projections
    # Shape: (3 * embed_dim, embed_dim)
    in_proj_weight = torch_mha_state_dict['in_proj_weight']
    in_proj_bias = torch_mha_state_dict['in_proj_bias']

    embed_dim = in_proj_weight.shape[1]
    head_dim = embed_dim // num_heads

    # Split into Q, K, V
    q_weight, k_weight, v_weight = torch.chunk(in_proj_weight, 3, dim=0)
    q_bias, k_bias, v_bias = torch.chunk(in_proj_bias, 3, dim=0)

    # Reshape for multi-head structure
    # PyTorch: (embed_dim, embed_dim) -> Flax: (embed_dim, num_heads, head_dim)
    q_weight_np = _to_cpu_np(q_weight.t())  # (embed_dim, embed_dim)
    q_weight_jax = jnp.array(q_weight_np.reshape(embed_dim, num_heads, head_dim))

    k_weight_np = _to_cpu_np(k_weight.t())
    k_weight_jax = jnp.array(k_weight_np.reshape(embed_dim, num_heads, head_dim))

    v_weight_np = _to_cpu_np(v_weight.t())
    v_weight_jax = jnp.array(v_weight_np.reshape(embed_dim, num_heads, head_dim))

    # Reshape biases: (embed_dim,) -> (num_heads, head_dim)
    q_bias_jax = jnp.array(_to_cpu_np(q_bias).reshape(num_heads, head_dim))
    k_bias_jax = jnp.array(_to_cpu_np(k_bias).reshape(num_heads, head_dim))
    v_bias_jax = jnp.array(_to_cpu_np(v_bias).reshape(num_heads, head_dim))

    # Transfer Q, K, V
    _assign(flax_params, path + ['query', 'kernel'], q_weight_jax, f"{desc}.query")
    _assign(flax_params, path + ['query', 'bias'], q_bias_jax, f"{desc}.query.bias")

    _assign(flax_params, path + ['key', 'kernel'], k_weight_jax, f"{desc}.key")
    _assign(flax_params, path + ['key', 'bias'], k_bias_jax, f"{desc}.key.bias")

    _assign(flax_params, path + ['value', 'kernel'], v_weight_jax, f"{desc}.value")
    _assign(flax_params, path + ['value', 'bias'], v_bias_jax, f"{desc}.value.bias")

    # Transfer output projection
    # PyTorch: (embed_dim, embed_dim) -> Flax: (num_heads, head_dim, embed_dim)
    out_proj_weight = torch_mha_state_dict['out_proj.weight']
    out_proj_bias = torch_mha_state_dict['out_proj.bias']

    out_weight_np = _to_cpu_np(out_proj_weight.t())  # (embed_dim, embed_dim)
    out_weight_jax = jnp.array(out_weight_np.reshape(num_heads, head_dim, embed_dim))

    _assign(flax_params, path + ['out', 'kernel'], out_weight_jax, f"{desc}.out")
    _assign(flax_params, path + ['out', 'bias'], _to_jax(out_proj_bias), f"{desc}.out.bias")


def transfer_uncertainty_embedding(flax_params, torch_state_dict):
    """Transfer UncertaintyEmbedding module weights (Experiment3 simple architecture)."""
    print("\n  Transferring UncertaintyEmbedding...")

    # Simple architecture: Linear -> LayerNorm -> GELU
    # uncertainty_embed.0: Linear(78, 128)
    transfer_linear(flax_params, ['uncertainty_embedding', 'uncertainty_embed_0'],
                    torch_state_dict['uncertainty_embedding.uncertainty_embed.0.weight'],
                    torch_state_dict['uncertainty_embedding.uncertainty_embed.0.bias'],
                    'uncertainty_embedding.uncertainty_embed.0')

    # uncertainty_embed.1: LayerNorm(128)
    transfer_layernorm(flax_params, ['uncertainty_embedding', 'uncertainty_embed_1'],
                      torch_state_dict['uncertainty_embedding.uncertainty_embed.1.weight'],
                      torch_state_dict['uncertainty_embedding.uncertainty_embed.1.bias'],
                      'uncertainty_embedding.uncertainty_embed.1')

    # uncertainty_scale parameter
    _assign(flax_params, ['uncertainty_embedding', 'uncertainty_scale'],
            _to_jax(torch_state_dict['uncertainty_embedding.uncertainty_scale']),
            'uncertainty_embedding.uncertainty_scale')

    print("    ✓ UncertaintyEmbedding transferred")


def transfer_uncertainty_head(flax_params, torch_state_dict):
    """Transfer UncertaintyHead module weights."""
    print("\n  Transferring UncertaintyHead...")

    # Main MLP path
    transfer_linear(flax_params, ['uncertainty_head', 'mlp_0'],
                   torch_state_dict['uncertainty_head.mlp.0.weight'],
                   torch_state_dict['uncertainty_head.mlp.0.bias'],
                   'uncertainty_head.mlp.0')
    transfer_linear(flax_params, ['uncertainty_head', 'mlp_1'],
                   torch_state_dict['uncertainty_head.mlp.2.weight'],
                   torch_state_dict['uncertainty_head.mlp.2.bias'],
                   'uncertainty_head.mlp.2')
    transfer_linear(flax_params, ['uncertainty_head', 'mlp_2'],
                   torch_state_dict['uncertainty_head.mlp.4.weight'],
                   torch_state_dict['uncertainty_head.mlp.4.bias'],
                   'uncertainty_head.mlp.4')

    # Uncertainty processor path
    transfer_linear(flax_params, ['uncertainty_head', 'unc_proc_0'],
                   torch_state_dict['uncertainty_head.uncertainty_processor.0.weight'],
                   torch_state_dict['uncertainty_head.uncertainty_processor.0.bias'],
                   'uncertainty_head.uncertainty_processor.0')
    transfer_linear(flax_params, ['uncertainty_head', 'unc_proc_1'],
                   torch_state_dict['uncertainty_head.uncertainty_processor.2.weight'],
                   torch_state_dict['uncertainty_head.uncertainty_processor.2.bias'],
                   'uncertainty_head.uncertainty_processor.2')

    # Uncertainty weight parameter
    _assign(flax_params, ['uncertainty_head', 'uncertainty_weight'],
            _to_jax(torch_state_dict['uncertainty_head.uncertainty_weight']),
            'uncertainty_head.uncertainty_weight')

    print("    ✓ UncertaintyHead transferred")


def transfer_dct_pose_transformer(torch_state_dict, flax_variables, nhead=4, num_layers=2):
    """
    Transfer all weights from PyTorch DCTPoseTransformer to Flax version.
    """
    params = unfreeze(flax_variables['params'])

    sd = torch_state_dict

    print("\n  Transferring main model components...")

    # Input embedding (Sequential: Linear, LayerNorm, GELU)
    transfer_linear(params, ['input_embed_0'],
                   sd['input_embed.0.weight'], sd['input_embed.0.bias'],
                   'input_embed.0')
    transfer_layernorm(params, ['input_embed_norm'],
                      sd['input_embed.1.weight'], sd['input_embed.1.bias'],
                      'input_embed.1')

    # Frequency positional embedding
    _assign(params, ['freq_pos_embed'],
            _to_jax(sd['freq_pos_embed']), 'freq_pos_embed')

    # Transformer blocks
    for i in range(num_layers):
        block_prefix = f'transformer_blocks.{i}'
        flax_block_prefix = f'transformer_block_{i}'

        print(f"    Transferring transformer block {i}...")

        # Frequency attention - freq_weights
        _assign(params, [flax_block_prefix, 'freq_attn', 'freq_weights'],
                _to_jax(sd[f'{block_prefix}.freq_attn.freq_weights']),
                f'{block_prefix}.freq_attn.freq_weights')

        # Multi-head attention
        mha_state = {
            'in_proj_weight': sd[f'{block_prefix}.freq_attn.mha.in_proj_weight'],
            'in_proj_bias': sd[f'{block_prefix}.freq_attn.mha.in_proj_bias'],
            'out_proj.weight': sd[f'{block_prefix}.freq_attn.mha.out_proj.weight'],
            'out_proj.bias': sd[f'{block_prefix}.freq_attn.mha.out_proj.bias'],
        }
        transfer_multihead_attention(params, [flax_block_prefix, 'freq_attn', 'mha'],
                                     mha_state, f'{block_prefix}.freq_attn.mha', nhead)

        # Layer norms
        transfer_layernorm(params, [flax_block_prefix, 'norm1'],
                          sd[f'{block_prefix}.norm1.weight'],
                          sd[f'{block_prefix}.norm1.bias'],
                          f'{block_prefix}.norm1')
        transfer_layernorm(params, [flax_block_prefix, 'norm2'],
                          sd[f'{block_prefix}.norm2.weight'],
                          sd[f'{block_prefix}.norm2.bias'],
                          f'{block_prefix}.norm2')

        # Low freq network
        transfer_linear(params, [flax_block_prefix, 'low_freq_0'],
                       sd[f'{block_prefix}.low_freq_net.0.weight'],
                       sd[f'{block_prefix}.low_freq_net.0.bias'],
                       f'{block_prefix}.low_freq_net.0')
        transfer_linear(params, [flax_block_prefix, 'low_freq_1'],
                       sd[f'{block_prefix}.low_freq_net.2.weight'],
                       sd[f'{block_prefix}.low_freq_net.2.bias'],
                       f'{block_prefix}.low_freq_net.2')

        # High freq network
        transfer_linear(params, [flax_block_prefix, 'high_freq_0'],
                       sd[f'{block_prefix}.high_freq_net.0.weight'],
                       sd[f'{block_prefix}.high_freq_net.0.bias'],
                       f'{block_prefix}.high_freq_net.0')
        transfer_linear(params, [flax_block_prefix, 'high_freq_1'],
                       sd[f'{block_prefix}.high_freq_net.2.weight'],
                       sd[f'{block_prefix}.high_freq_net.2.bias'],
                       f'{block_prefix}.high_freq_net.2')

    # Frequency decoders
    transfer_linear(params, ['low_freq_decoder'],
                   sd['low_freq_decoder.weight'],
                   sd['low_freq_decoder.bias'],
                   'low_freq_decoder')
    transfer_linear(params, ['high_freq_decoder'],
                   sd['high_freq_decoder.weight'],
                   sd['high_freq_decoder.bias'],
                   'high_freq_decoder')

    # Transfer uncertainty components
    transfer_uncertainty_embedding(params, sd)
    transfer_uncertainty_head(params, sd)

    return freeze(params)


def main():
    print("="*70)
    print("DCTPoseTransformer Weight Transfer: PyTorch → JAX/Flax")
    print("="*70)
    import os
    # Configuration
    # pytorch_model_path = os.path.join(root_dir, "marian_code/Experiment4/model_checkpoint_prediction_transformer.pth")
    pytorch_model_path = os.path.join(root_dir, "jax_hmp_files/transformer_model.pth")
    output_path = os.path.join(root_dir, "human_pose_pipeline/models/motion_prediction/dct_pose_transformer.pickle")

    # Model parameters
    input_dim = 39
    d_model = 128
    nhead = 4
    num_layers = 2
    seq_len = 50
    seq_len_output = 10

    # Load PyTorch checkpoint
    print(f"\n1. Loading PyTorch checkpoint from: {pytorch_model_path}")
    checkpoint = torch.load(pytorch_model_path, map_location='cpu')

    # Extract state_dict (handle both raw state_dict and checkpoint dict)
    if 'model_state_dict' in checkpoint:
        torch_state_dict = checkpoint['model_state_dict']
        print(f"   Found checkpoint with epoch {checkpoint.get('epoch', 'unknown')}")
    elif 'state_dict' in checkpoint:
        torch_state_dict = checkpoint['state_dict']
    else:
        torch_state_dict = checkpoint

    print(f"   ✓ Loaded {len(torch_state_dict)} parameter tensors")

    # Verify uncertainty_embedding exists
    unc_emb_keys = [k for k in torch_state_dict.keys() if 'uncertainty_embedding' in k]
    print(f"   ✓ Found {len(unc_emb_keys)} uncertainty_embedding parameters")

    # Initialize Flax model
    print(f"\n2. Initializing JAX/Flax model")
    flax_model = DCTPoseTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=seq_len,
        seq_len_output=seq_len_output,
        reduced_size=False  # Use full output for weight transfer
    )

    rng = jax.random.PRNGKey(0)
    # Initialize with dummy input INCLUDING uncertainty to ensure all params are created
    # Experiment3 architecture:
    # - Poses: [batch, seq_len, 39]
    # - Uncertainty: [batch, seq_len, 117] where 117 = 13 joints * 9 (3x3 covariance)
    # Total input: [batch, seq_len, 156]
    uncertainty_dim = 13 * 3 * 3  # 117
    dummy_x_with_unc = jnp.zeros((2, seq_len, input_dim + uncertainty_dim), dtype=jnp.float32)
    flax_variables = flax_model.init(rng, dummy_x_with_unc, train=False)

    print(f"   ✓ Initialized Flax model")
    print(f"   Top-level param keys: {list(flax_variables['params'].keys())}")

    # Transfer weights
    print(f"\n3. Transferring weights...")
    try:
        flax_params = transfer_dct_pose_transformer(
            torch_state_dict,
            flax_variables,
            nhead=nhead,
            num_layers=num_layers
        )
        print("\n   ✓ Weight transfer completed successfully!")
    except Exception as e:
        print(f"\n   ✗ Weight transfer failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save model
    print(f"\n4. Saving JAX/Flax model to: {output_path}")

    model_dict = {
        "model": "DCTPoseTransformer",
        "params": flax_params,
        "config": {
            "input_dim": input_dim,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "seq_len": seq_len,
            "seq_len_output": seq_len_output
        }
    }

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model_dict, f)

    print(f"   ✓ Saved successfully!")

    print("\n" + "="*70)
    print("Weight transfer completed!")
    print("="*70)


if __name__ == "__main__":
    main()
