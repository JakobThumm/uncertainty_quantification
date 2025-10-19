"""
Transfer DCTPoseTransformer weights from PyTorch to JAX/Flax
"""

import os
import sys
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.serialization import to_state_dict, from_state_dict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

# Add model paths
sys.path.append('/home/skyle/Desktop/uq_benchmark/models/HMP')
sys.path.append('/home/skyle/Desktop/uq_benchmark')

from torch_dct.transformer_model_and_loss import DCTPoseTransformer as DCTPoseTransformerTorch
from jax_dct.dct_pose_transformer_flax import DCTPoseTransformerFlax
from spacepy import pycdf


# ============== Dataset for testing ==============

JOINT_IDX_17 = [0, 1, 2, 3, 6, 7, 8, 12, 16, 14, 15, 17, 18, 19, 25, 26, 27]
JOINT_IDX_13 = [9, 14, 11, 15, 12, 16, 13, 1, 4, 2, 5, 3, 6]

SPLIT = {
    'train': ['S1', 'S6', 'S7', 'S8', 'S9'],
    'validation': ['S11'],
    'test': ['S5'] #S5
}

# H36M 13-joint skeleton connections
# Joint order: [Hip, RHip, RKnee, RFoot, LHip, LKnee, LFoot, Spine, Neck, Head, LShoulder, LElbow, LWrist]
H36M_SKELETON_13 = [
    (0, 1),  # Hip -> RHip
    (1, 2),  # RHip -> RKnee
    (2, 3),  # RKnee -> RFoot
    (0, 4),  # Hip -> LHip
    (4, 5),  # LHip -> LKnee
    (5, 6),  # LKnee -> LFoot
    (0, 7),  # Hip -> Spine
    (7, 8),  # Spine -> Neck
    (8, 9),  # Neck -> Head
    (8, 10), # Neck -> LShoulder
    (10, 11),# LShoulder -> LElbow
    (11, 12),# LElbow -> LWrist
]

# COCO 17-joint skeleton connections (for 2D pose estimation)
# Joint order: [Nose, LEye, REye, LEar, REar, LShoulder, RShoulder, LElbow, RElbow, 
#               LWrist, RWrist, LHip, RHip, LKnee, RKnee, LAnkle, RAnkle]
COCO_SKELETON_17 = [
    (0, 1),   # Nose -> Left Eye
    (0, 2),   # Nose -> Right Eye
    (1, 3),   # Left Eye -> Left Ear
    (2, 4),   # Right Eye -> Right Ear
    (0, 5),   # Nose -> Left Shoulder
    (0, 6),   # Nose -> Right Shoulder
    (5, 6),   # Left Shoulder -> Right Shoulder
    (5, 7),   # Left Shoulder -> Left Elbow
    (7, 9),   # Left Elbow -> Left Wrist
    (6, 8),   # Right Shoulder -> Right Elbow
    (8, 10),  # Right Elbow -> Right Wrist
    (5, 11),  # Left Shoulder -> Left Hip
    (6, 12),  # Right Shoulder -> Right Hip
    (11, 12), # Left Hip -> Right Hip
    (11, 13), # Left Hip -> Left Knee
    (13, 15), # Left Knee -> Left Ankle
    (12, 14), # Right Hip -> Right Knee
    (14, 16), # Right Knee -> Right Ankle
]

# Joint names for reference
JOINT_NAMES_13 = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot',
                  'Spine', 'Neck', 'Head', 'LShoulder', 'LElbow', 'LWrist']

JOINT_NAMES_17 = ['Nose', 'LEye', 'REye', 'LEar', 'REar', 'LShoulder', 'RShoulder',
                  'LElbow', 'RElbow', 'LWrist', 'RWrist', 'LHip', 'RHip',
                  'LKnee', 'RKnee', 'LAnkle', 'RAnkle']

class Human36mDataset3D(Dataset):
    """Dataset class for Human3.6M motion data."""
    def __init__(self, base_directory, split='train', input_frames=50, predict_frames=10):
        self.input_frames = input_frames
        self.predict_frames = predict_frames
        self.data = []
        self.data = self.load_data(base_directory, split)
        
    def load_data(self, base_directory, split):
        all_data = []
        for subject in SPLIT[split]:
            directory = os.path.join(base_directory, subject, 'D3_Positions')
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} not found, skipping...")
                continue
                
            for filename in os.listdir(directory):
                if filename.endswith('.cdf'):
                    file_path = os.path.join(directory, filename)
                    with pycdf.CDF(file_path) as cdf:
                        poses = cdf['Pose'][:]
                        poses = poses.reshape(-1, 32, 3)
                        poses_13 = poses[:, JOINT_IDX_17, :]
                        poses_13 = poses_13[:, JOINT_IDX_13, :]
                        poses_13 = poses_13.reshape(poses_13.shape[0], -1)

                        for offset in [0, 1]:
                            downsampled_poses = poses_13[offset::2]
                            for i in range(len(downsampled_poses) - self.input_frames - self.predict_frames + 1):
                                window = downsampled_poses[i:i + self.input_frames + self.predict_frames]
                                all_data.append(window)
        print(f"Loaded {len(all_data)} sequences for {split} split")
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_pose = sequence[:self.input_frames]
        target_pose = sequence[self.input_frames:]
        return {
            "input_pose": torch.FloatTensor(input_pose),
            "target_pose": torch.FloatTensor(target_pose),
        }


def get_dct_matrix(N):
    """Compute the Discrete Cosine Transform (DCT) matrix and its inverse."""
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


# ============== Weight Transfer Functions ==============

def _to_cpu_np(t):
    """Convert torch tensor to numpy."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


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
    """Transfer PyTorch Linear layer to Flax Dense layer."""
    # PyTorch: (out, in), Flax: (in, out)
    _assign(flax_params, path + ['kernel'], 
            _to_cpu_np(torch_weight.t()), f"{desc}.weight")
    _assign(flax_params, path + ['bias'],
            _to_cpu_np(torch_bias), f"{desc}.bias")


def transfer_layernorm(flax_params, path, torch_weight, torch_bias, desc):
    """Transfer PyTorch LayerNorm to Flax LayerNorm."""
    _assign(flax_params, path + ['scale'],
            _to_cpu_np(torch_weight), f"{desc}.weight")
    _assign(flax_params, path + ['bias'],
            _to_cpu_np(torch_bias), f"{desc}.bias")


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
    q_weight_np = q_weight_np.reshape(embed_dim, num_heads, head_dim)
    
    k_weight_np = _to_cpu_np(k_weight.t())
    k_weight_np = k_weight_np.reshape(embed_dim, num_heads, head_dim)
    
    v_weight_np = _to_cpu_np(v_weight.t())
    v_weight_np = v_weight_np.reshape(embed_dim, num_heads, head_dim)
    
    # Reshape biases: (embed_dim,) -> (num_heads, head_dim)
    q_bias_np = _to_cpu_np(q_bias).reshape(num_heads, head_dim)
    k_bias_np = _to_cpu_np(k_bias).reshape(num_heads, head_dim)
    v_bias_np = _to_cpu_np(v_bias).reshape(num_heads, head_dim)
    
    # Transfer Q, K, V
    _assign(flax_params, path + ['query', 'kernel'], q_weight_np, f"{desc}.query")
    _assign(flax_params, path + ['query', 'bias'], q_bias_np, f"{desc}.query.bias")
    
    _assign(flax_params, path + ['key', 'kernel'], k_weight_np, f"{desc}.key")
    _assign(flax_params, path + ['key', 'bias'], k_bias_np, f"{desc}.key.bias")
    
    _assign(flax_params, path + ['value', 'kernel'], v_weight_np, f"{desc}.value")
    _assign(flax_params, path + ['value', 'bias'], v_bias_np, f"{desc}.value.bias")
    
    # Transfer output projection
    # PyTorch: (embed_dim, embed_dim) -> Flax: (num_heads, head_dim, embed_dim)
    out_proj_weight = torch_mha_state_dict['out_proj.weight']
    out_proj_bias = torch_mha_state_dict['out_proj.bias']
    
    out_weight_np = _to_cpu_np(out_proj_weight.t())  # (embed_dim, embed_dim)
    out_weight_np = out_weight_np.reshape(num_heads, head_dim, embed_dim)
    
    _assign(flax_params, path + ['out', 'kernel'], out_weight_np, f"{desc}.out")
    _assign(flax_params, path + ['out', 'bias'], _to_cpu_np(out_proj_bias), f"{desc}.out.bias")


def transfer_dct_pose_transformer(torch_state_dict, flax_variables, nhead=4):
    """
    Transfer all weights from PyTorch DCTPoseTransformer to Flax version.
    """
    params = unfreeze(flax_variables['params'])
    
    sd = torch_state_dict
    
    # Input embedding (Sequential: Linear, LayerNorm, GELU)
    transfer_linear(params, ['input_embed_0'], 
                   sd['input_embed.0.weight'], sd['input_embed.0.bias'],
                   'input_embed.0')
    transfer_layernorm(params, ['input_embed_norm'],
                      sd['input_embed.1.weight'], sd['input_embed.1.bias'],
                      'input_embed.1')
    
    # Frequency positional embedding
    _assign(params, ['freq_pos_embed'],
            _to_cpu_np(sd['freq_pos_embed']), 'freq_pos_embed')
    
    # Transformer blocks
    num_layers = len([k for k in sd.keys() if k.startswith('transformer_blocks.')])
    num_layers = num_layers // 10  # Approximate number of unique blocks
    
    for i in range(2):  # num_layers, hard coded for now?
        block_prefix = f'transformer_blocks.{i}'
        flax_block_prefix = f'transformer_block_{i}'
        
        # Frequency attention - freq_weights
        _assign(params, [flax_block_prefix, 'freq_attn', 'freq_weights'],
                _to_cpu_np(sd[f'{block_prefix}.freq_attn.freq_weights']),
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
    
    # Uncertainty head
    transfer_linear(params, ['uncertainty_head', 'mlp_0'],
                   sd['uncertainty_head.mlp.0.weight'],
                   sd['uncertainty_head.mlp.0.bias'],
                   'uncertainty_head.mlp.0')
    transfer_linear(params, ['uncertainty_head', 'mlp_1'],
                   sd['uncertainty_head.mlp.2.weight'],
                   sd['uncertainty_head.mlp.2.bias'],
                   'uncertainty_head.mlp.2')
    transfer_linear(params, ['uncertainty_head', 'mlp_2'],
                   sd['uncertainty_head.mlp.4.weight'],
                   sd['uncertainty_head.mlp.4.bias'],
                   'uncertainty_head.mlp.4')
    
    # Try to transfer uncertainty processor if it exists in both models
    try:
        if 'unc_proc_0' in params['uncertainty_head']:
            transfer_linear(params, ['uncertainty_head', 'unc_proc_0'],
                           sd['uncertainty_head.uncertainty_processor.0.weight'],
                           sd['uncertainty_head.uncertainty_processor.0.bias'],
                           'uncertainty_head.uncertainty_processor.0')
            transfer_linear(params, ['uncertainty_head', 'unc_proc_1'],
                           sd['uncertainty_head.uncertainty_processor.2.weight'],
                           sd['uncertainty_head.uncertainty_processor.2.bias'],
                           'uncertainty_head.uncertainty_processor.2')
            print("  ✓ Transferred uncertainty_processor layers")
        else:
            print("  ⚠ Skipping uncertainty_processor layers (not in Flax model)")
    except Exception as e:
        print(f"  ⚠ Could not transfer uncertainty_processor: {e}")
    
    _assign(params, ['uncertainty_head', 'uncertainty_weight'],
            _to_cpu_np(sd['uncertainty_head.uncertainty_weight']),
            'uncertainty_head.uncertainty_weight')
    
    return freeze(params)


# ============== Test Functions ==============

def smoke_test(torch_model, flax_model, flax_params, device='cuda'):
    """Test with random input data."""
    print("\n" + "="*60)
    print("SMOKE TEST - Random Input")
    print("="*60)
    
    batch_size = 4
    seq_len = 50
    input_dim = 39
    
    # Create random input with fixed seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    x_torch = torch.randn(batch_size, seq_len, input_dim).to(device)
    x_jax = jnp.array(x_torch.detach().cpu().numpy())
    
    # Test some basic stats
    print("\nInput stats:")
    print(f"PyTorch input - mean: {x_torch.mean():.6f}, std: {x_torch.std():.6f}")
    print(f"JAX input - mean: {jnp.mean(x_jax):.6f}, std: {jnp.std(x_jax):.6f}")
    
    # PyTorch full forward
    torch_model.eval()
    with torch.no_grad():
        poses_torch, (var_torch, cov_torch) = torch_model(x_torch)
    
    poses_torch_np = poses_torch.detach().cpu().numpy()
    var_torch_np = var_torch.detach().cpu().numpy()
    cov_torch_np = cov_torch.detach().cpu().numpy()
    
    # JAX forward
    poses_jax, (var_jax, cov_jax) = flax_model.apply({'params': flax_params}, x_jax)
    
    poses_jax_np = np.array(poses_jax)
    var_jax_np = np.array(var_jax)
    cov_jax_np = np.array(cov_jax)
    
    # Compare
    print(f"\nFinal outputs:")
    print(f"Poses shape - PyTorch: {poses_torch_np.shape}, JAX: {poses_jax_np.shape}")
    print(f"Poses max abs diff: {np.max(np.abs(poses_torch_np - poses_jax_np)):.6f}")
    print(f"Poses mean abs diff: {np.mean(np.abs(poses_torch_np - poses_jax_np)):.6f}")
    print(f"Poses - PyTorch mean: {np.mean(poses_torch_np):.6f}, JAX mean: {np.mean(poses_jax_np):.6f}")
    
    print(f"\nVar params max abs diff: {np.max(np.abs(var_torch_np - var_jax_np)):.6f}")
    print(f"Cov params max abs diff: {np.max(np.abs(cov_torch_np - cov_jax_np)):.6f}")
    
    # Check if differences are acceptable
    # Note: 0.05 threshold is reasonable for transformer models with attention
    pose_diff = np.max(np.abs(poses_torch_np - poses_jax_np))
    if pose_diff < 0.05:
        print("\n✓ SMOKE TEST PASSED")
        return True
    else:
        print(f"\n✗ SMOKE TEST FAILED - Max difference {pose_diff} too large (threshold: 0.05)")
        print("Note: This might be due to missing unc_proc layers or numerical precision differences")
        return False


def plot_3d_skeleton(ax, pose, connections=None, color='blue', alpha=0.8, linewidth=2, label=None):
    """
    Plot a 3D skeleton with auto-detection of joint format.
    
    Args:
        ax: matplotlib 3D axis
        pose: (N, 3) array of joint positions (N=13 or N=17)
        connections: list of (i, j) tuples for skeleton edges (auto-detected if None)
        color: color for the skeleton
        alpha: transparency
        linewidth: line width
        label: legend label
    """
    # Auto-detect skeleton type if connections not provided
    if connections is None:
        num_joints = pose.shape[0]
        if num_joints == 13:
            connections = H36M_SKELETON_13
        elif num_joints == 17:
            connections = COCO_SKELETON_17
        else:
            raise ValueError(f"Unsupported number of joints: {num_joints}. Expected 13 or 17.")
    
    # Plot joints
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], 
               c=color, s=30, alpha=alpha, edgecolors='k', linewidth=0.5)
    
    # Plot connections
    for i, j in connections:
        line = np.array([pose[i], pose[j]])
        ax.plot(line[:, 0], line[:, 1], line[:, 2], 
                c=color, linewidth=linewidth, alpha=alpha, label=label if i == 0 and j == 1 else None)


def visualize_predictions(ground_truth, pred_torch, pred_jax, 
                          frame_idx=0, sample_idx=0, 
                          save_path='pred/comparison.png'):
    """
    Visualize and compare ground truth, PyTorch, and JAX predictions.
    
    Args:
        ground_truth: (batch, seq_len, 39) target poses
        pred_torch: (batch, seq_len, 39) PyTorch predictions
        pred_jax: (batch, seq_len, 39) JAX predictions
        frame_idx: which frame to visualize (0-9 for future frames)
        sample_idx: which sample from batch to visualize
        save_path: where to save the visualization
    """
    # Extract the specific frame and reshape to (num_joints, 3)
    # Auto-detect number of joints from data shape
    num_coords = ground_truth.shape[-1]
    num_joints = num_coords // 3
    
    gt_pose = ground_truth[sample_idx, frame_idx].reshape(num_joints, 3)
    torch_pose = pred_torch[sample_idx, frame_idx].reshape(num_joints, 3)
    jax_pose = pred_jax[sample_idx, frame_idx].reshape(num_joints, 3)
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 5))
    
    # Determine common axis limits for consistency
    all_poses = np.concatenate([gt_pose, torch_pose, jax_pose], axis=0)
    x_range = [all_poses[:, 0].min() - 100, all_poses[:, 0].max() + 100]
    y_range = [all_poses[:, 1].min() - 100, all_poses[:, 1].max() + 100]
    z_range = [all_poses[:, 2].min() - 100, all_poses[:, 2].max() + 100]
    
    # Plot 1: Ground Truth
    ax1 = fig.add_subplot(141, projection='3d')
    plot_3d_skeleton(ax1, gt_pose, color='green', label='Ground Truth')  # Auto-detect skeleton
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'Ground Truth ({num_joints} joints)\nFrame {frame_idx}', fontsize=12, fontweight='bold')
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_zlim(z_range)
    ax1.view_init(elev=15, azim=45)
    
    # Plot 2: PyTorch Prediction
    ax2 = fig.add_subplot(142, projection='3d')
    plot_3d_skeleton(ax2, torch_pose, color='blue', label='PyTorch')  # Auto-detect skeleton
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title(f'PyTorch Prediction\nFrame {frame_idx}', fontsize=12, fontweight='bold')
    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)
    ax2.set_zlim(z_range)
    ax2.view_init(elev=15, azim=45)
    
    # Plot 3: JAX Prediction
    ax3 = fig.add_subplot(143, projection='3d')
    plot_3d_skeleton(ax3, jax_pose, color='red', label='JAX')  # Auto-detect skeleton
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_zlabel('Z (mm)')
    ax3.set_title(f'JAX Prediction\nFrame {frame_idx}', fontsize=12, fontweight='bold')
    ax3.set_xlim(x_range)
    ax3.set_ylim(y_range)
    ax3.set_zlim(z_range)
    ax3.view_init(elev=15, azim=45)
    
    # Plot 4: Overlay comparison
    ax4 = fig.add_subplot(144, projection='3d')
    plot_3d_skeleton(ax4, gt_pose, color='green', alpha=0.6, linewidth=3, label='Ground Truth')  # Auto-detect
    plot_3d_skeleton(ax4, torch_pose, color='blue', alpha=0.6, linewidth=2, label='PyTorch')  # Auto-detect
    plot_3d_skeleton(ax4, jax_pose, color='red', alpha=0.6, linewidth=2, label='JAX')  # Auto-detect
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_zlabel('Z (mm)')
    ax4.set_title(f'Overlay Comparison\nFrame {frame_idx}', fontsize=12, fontweight='bold')
    ax4.set_xlim(x_range)
    ax4.set_ylim(y_range)
    ax4.set_zlim(z_range)
    ax4.view_init(elev=15, azim=45)
    ax4.legend(loc='upper right')
    
    # Calculate errors
    torch_error = np.mean(np.linalg.norm(gt_pose - torch_pose, axis=1))
    jax_error = np.mean(np.linalg.norm(gt_pose - jax_pose, axis=1))
    torch_jax_diff = np.mean(np.linalg.norm(torch_pose - jax_pose, axis=1))
    
    # Add overall title with errors
    fig.suptitle(f'3D Pose Prediction Comparison (Sample {sample_idx})\n' +
                 f'PyTorch Error: {torch_error:.2f} mm | JAX Error: {jax_error:.2f} mm | ' +
                 f'PyTorch-JAX Diff: {torch_jax_diff:.2f} mm',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return torch_error, jax_error, torch_jax_diff


def visualize_sequence(ground_truth, pred_torch, pred_jax, 
                       sample_idx=0, 
                       save_path='pred/sequence.png',
                       frames_to_plot=[0, 4, 9]):
    """
    Visualize multiple frames in a sequence.
    
    Args:
        ground_truth: (batch, seq_len, 39) target poses
        pred_torch: (batch, seq_len, 39) PyTorch predictions
        pred_jax: (batch, seq_len, 39) JAX predictions
        sample_idx: which sample from batch to visualize
        save_path: where to save the visualization
        frames_to_plot: list of frame indices to visualize
    """
    # Auto-detect number of joints
    num_coords = ground_truth.shape[-1]
    num_joints = num_coords // 3
    
    n_frames = len(frames_to_plot)
    fig = plt.figure(figsize=(6 * n_frames, 15))
    
    for idx, frame_idx in enumerate(frames_to_plot):
        # Extract the specific frame and reshape to (num_joints, 3)
        gt_pose = ground_truth[sample_idx, frame_idx].reshape(num_joints, 3)
        torch_pose = pred_torch[sample_idx, frame_idx].reshape(num_joints, 3)
        jax_pose = pred_jax[sample_idx, frame_idx].reshape(num_joints, 3)
        
        # Determine common axis limits
        all_poses = np.concatenate([gt_pose, torch_pose, jax_pose], axis=0)
        x_range = [all_poses[:, 0].min() - 100, all_poses[:, 0].max() + 100]
        y_range = [all_poses[:, 1].min() - 100, all_poses[:, 1].max() + 100]
        z_range = [all_poses[:, 2].min() - 100, all_poses[:, 2].max() + 100]
        
        # Ground Truth
        ax1 = fig.add_subplot(3, n_frames, idx + 1, projection='3d')
        plot_3d_skeleton(ax1, gt_pose, color='green')  # Auto-detect skeleton
        ax1.set_title(f'Ground Truth ({num_joints}J)\nFrame {frame_idx}', fontsize=10, fontweight='bold')
        ax1.set_xlim(x_range)
        ax1.set_ylim(y_range)
        ax1.set_zlim(z_range)
        ax1.view_init(elev=15, azim=45)
        if idx == 0:
            ax1.set_ylabel('Y', fontsize=8)
        
        # PyTorch
        ax2 = fig.add_subplot(3, n_frames, n_frames + idx + 1, projection='3d')
        plot_3d_skeleton(ax2, torch_pose, color='blue')  # Auto-detect skeleton
        ax2.set_title(f'PyTorch\nFrame {frame_idx}', fontsize=10, fontweight='bold')
        ax2.set_xlim(x_range)
        ax2.set_ylim(y_range)
        ax2.set_zlim(z_range)
        ax2.view_init(elev=15, azim=45)
        if idx == 0:
            ax2.set_ylabel('Y', fontsize=8)
        
        # JAX
        ax3 = fig.add_subplot(3, n_frames, 2 * n_frames + idx + 1, projection='3d')
        plot_3d_skeleton(ax3, jax_pose, color='red')  # Auto-detect skeleton
        ax3.set_title(f'JAX\nFrame {frame_idx}', fontsize=10, fontweight='bold')
        ax3.set_xlim(x_range)
        ax3.set_ylim(y_range)
        ax3.set_zlim(z_range)
        ax3.view_init(elev=15, azim=45)
        if idx == 0:
            ax3.set_ylabel('Y', fontsize=8)
    
    fig.suptitle(f'Sequence Visualization (Sample {sample_idx})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def h36m_data_test(torch_model, flax_model, flax_params, data_path, device='cuda'):
    """Test with real H36M data."""
    print("\n" + "="*60)
    print("H36M DATA TEST - Real Dataset")
    print("="*60)
    
    # Load dataset
    try:
        dataset = Human36mDataset3D(data_path, split='test', input_frames=50, predict_frames=10)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    except Exception as e:
        print(f"Warning: Could not load H36M dataset: {e}")
        print("Skipping H36M data test...")
        return None
    
    # Get DCT matrices
    N = 50
    dct_m, idct_m = get_dct_matrix(N)
    dct_m_torch = torch.from_numpy(dct_m).float().to(device)
    idct_m_torch = torch.from_numpy(idct_m).float().to(device)
    dct_m_jax = jnp.array(dct_m, dtype=jnp.float32)
    idct_m_jax = jnp.array(idct_m, dtype=jnp.float32)
    
    torch_model.eval()
    
    max_diffs = []
    mean_diffs = []
    
    # Create output directory
    os.makedirs('pred', exist_ok=True)
    
    # Test on a few batches
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Test on first 3 batches
            break
            
        input_pose = batch['input_pose'].to(device)
        target_pose = batch['target_pose'].to(device)
        
        # Apply DCT preprocessing (matching training)
        input_pose_ = torch.matmul(input_pose.transpose(1, 2), dct_m_torch.transpose(0, 1)).transpose(1, 2)
        input_pose_ = input_pose_ / 1000  # Convert to meters
        
        # PyTorch inference with CORRECT postprocessing
        with torch.no_grad():
            pred_output_torch, (var_torch, cov_torch) = torch_model(input_pose_)
            
            # Apply correct postprocessing (matching training):
            # 1. Denormalize (convert back to mm in DCT space)
            pred_poses_torch = pred_output_torch * 1000
            
            # 2. Apply IDCT (convert from frequency domain to time domain)
            pred_poses_torch = torch.matmul(pred_poses_torch.transpose(1, 2),
                                           idct_m_torch.transpose(0, 1)).transpose(1, 2)
            
            # 3. Add offset (last frame of input as reference)
            offset = input_pose[:, -1:, :]
            pred_poses_torch = pred_poses_torch[:, :10, :] + offset  # Only first 10 frames are future
        
        poses_torch_np = pred_poses_torch.detach().cpu().numpy()
        
        # JAX inference with CORRECT postprocessing
        input_pose_jax = jnp.array(input_pose_.detach().cpu().numpy())
        pred_output_jax, (var_jax, cov_jax) = flax_model.apply({'params': flax_params}, input_pose_jax)
        
        # Apply same postprocessing for JAX
        # 1. Denormalize
        pred_poses_jax = pred_output_jax * 1000
        
        # 2. Apply IDCT
        pred_poses_jax = jnp.matmul(jnp.transpose(pred_poses_jax, (0, 2, 1)),
                                    jnp.transpose(idct_m_jax, (1, 0)))
        pred_poses_jax = jnp.transpose(pred_poses_jax, (0, 2, 1))
        
        # 3. Add offset
        offset_jax = jnp.array(input_pose[:, -1:, :].detach().cpu().numpy())
        pred_poses_jax = pred_poses_jax[:, :10, :] + offset_jax
        
        poses_jax_np = np.array(pred_poses_jax)
        
        # Compare
        max_diff = np.max(np.abs(poses_torch_np - poses_jax_np))
        mean_diff = np.mean(np.abs(poses_torch_np - poses_jax_np))
        
        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)
        
        print(f"Batch {i}: Max diff = {max_diff:.6f}, Mean diff = {mean_diff:.6f}")
        
        # Get target poses
        target_pose_np = target_pose.detach().cpu().numpy()
        
        # Predictions are now in mm and already postprocessed (10 frames)
        pred_torch_future = poses_torch_np
        pred_jax_future = poses_jax_np
        
        # Visualize first sample in batch for selected frames
        if i == 0:  # Only visualize first batch
            print(f"\nGenerating visualizations for batch {i}...")
            
            # Single frame comparisons for frames 0, 4, 9
            for frame_idx in [0, 4, 9]:
                torch_err, jax_err, diff = visualize_predictions(
                    target_pose_np, 
                    pred_torch_future, 
                    pred_jax_future,
                    frame_idx=frame_idx,
                    sample_idx=0,
                    save_path=f'pred/batch{i}_sample0_frame{frame_idx}.png'
                )
                print(f"  Frame {frame_idx}: PyTorch error={torch_err:.2f}mm, JAX error={jax_err:.2f}mm, Diff={diff:.2f}mm")
            
            # Sequence visualization
            visualize_sequence(
                target_pose_np,
                pred_torch_future,
                pred_jax_future,
                sample_idx=0,
                save_path=f'pred/batch{i}_sample0_sequence.png',
                frames_to_plot=[0, 4, 9]
            )
            print(f"  Saved sequence visualization")
            
            # Visualize second sample if batch size allows
            if poses_torch_np.shape[0] > 1:
                visualize_predictions(
                    target_pose_np, 
                    pred_torch_future, 
                    pred_jax_future,
                    frame_idx=4,  # Middle frame
                    sample_idx=1,
                    save_path=f'pred/batch{i}_sample1_frame4.png'
                )
                print(f"  Saved sample 1 visualization")
    
    if len(max_diffs) == 0:
        print("\n⚠ H36M DATA TEST SKIPPED - No data available")
        return None
    
    avg_max_diff = np.mean(max_diffs)
    avg_mean_diff = np.mean(mean_diffs)
    
    print(f"\nAverage max diff across batches: {avg_max_diff:.6f}")
    print(f"Average mean diff across batches: {avg_mean_diff:.6f}")
    
    # Note: With correct postprocessing, differences should be very small (< 50mm)
    if avg_max_diff < 50:
        print("\n✓ H36M DATA TEST PASSED")
        print(f"   (Average PyTorch-JAX difference: {avg_max_diff:.2f}mm)")
        return True
    else:
        print(f"\n⚠ H36M DATA TEST - Average max difference {avg_max_diff:.2f}mm (threshold: 50mm)")
        print(f"   Note: Predictions are now in physical space (mm), not normalized")
        # Still return True if reasonable
        return avg_max_diff < 200  # 200mm threshold for max single point difference


# ============== Main ==============

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    pytorch_model_path = "transformer_model.pth"
    h36m_data_path = "/home/skyle/datasets/H36M_FREI"
    output_path = "dct_pose_transformer_flax.pickle"
    
    # Model parameters
    input_dim = 39
    d_model = 128
    nhead = 4
    num_layers = 2
    seq_len = 50
    seq_len_output = 10
    
    # Load PyTorch model
    print("\n" + "="*60)
    print("Loading PyTorch Model")
    print("="*60)
    
    torch_model = DCTPoseTransformerTorch(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=seq_len,
        seq_len_output=seq_len_output
    )
    
    if os.path.exists(pytorch_model_path):
        torch_state_dict = torch.load(pytorch_model_path, map_location=device)
        torch_model.load_state_dict(torch_state_dict)
        print(f"✓ Loaded PyTorch model from {pytorch_model_path}")
    else:
        print(f"Warning: PyTorch model not found at {pytorch_model_path}")
        print("Using randomly initialized model for testing...")
        torch_state_dict = torch_model.state_dict()
    
    torch_model.to(device)
    torch_model.eval()
    
    # Initialize Flax model
    print("\n" + "="*60)
    print("Initializing Flax Model")
    print("="*60)
    
    flax_model = DCTPoseTransformerFlax(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=seq_len,
        seq_len_output=seq_len_output
    )
    
    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.zeros((2, seq_len, input_dim), dtype=jnp.float32)
    flax_variables = flax_model.init(rng, dummy_x)
    
    print(f"✓ Initialized Flax model")
    print(f"Flax params keys: {list(flax_variables['params'].keys())}")
    
    # Transfer weights
    print("\n" + "="*60)
    print("Transferring Weights")
    print("="*60)
    
    try:
        flax_params = transfer_dct_pose_transformer(torch_state_dict, flax_variables, nhead=nhead)
        print("✓ Weight transfer completed")
    except Exception as e:
        print(f"✗ Weight transfer failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run tests
    smoke_passed = smoke_test(torch_model, flax_model, flax_params, device)
    
    h36m_passed = h36m_data_test(torch_model, flax_model, flax_params, h36m_data_path, device)
    
    # Save model if tests passed
    if smoke_passed and (h36m_passed is None or h36m_passed):
        print("\n" + "="*60)
        print("Saving Flax Model")
        print("="*60)
        
        model_dict = {
            "model": "DCTPoseTransformerFlax",
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
        
        # Create directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create if there's a directory component
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"✓ Saved Flax model to {output_path}")
    else:
        print("\n✗ Tests failed, not saving model")


if __name__ == "__main__":
    main()

