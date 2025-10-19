"""
Standalone script to visualize pose predictions from saved models.
Run this to generate more visualizations without re-running the full transfer.
"""

import os
import sys
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')

# Add model paths
sys.path.append('/home/skyle/Desktop/uq_benchmark/models/HMP')

from torch_dct.transformer_model_and_loss import DCTPoseTransformer as DCTPoseTransformerTorch
from jax_dct.dct_pose_transformer_flax import DCTPoseTransformerFlax
from spacepy import pycdf

# Import visualization functions from transfer script
from transfer_dct_pose_transformer import (
    Human36mDataset3D, get_dct_matrix,
    plot_3d_skeleton, visualize_predictions, visualize_sequence,
    H36M_SKELETON_13, COCO_SKELETON_17, JOINT_NAMES_13, JOINT_NAMES_17
)


def main():
    """Generate visualizations from saved models."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    pytorch_model_path = "transformer_model.pth"
    jax_model_path = "dct_pose_transformer_flax.pickle"
    h36m_data_path = "/home/skyle/datasets/H36M_FREI"
    output_dir = "pred"
    
    # Model parameters
    input_dim = 39
    d_model = 128
    nhead = 4
    num_layers = 2
    seq_len = 50
    seq_len_output = 10
    
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)
    
    # Load PyTorch model
    torch_model = DCTPoseTransformerTorch(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        seq_len=seq_len,
        seq_len_output=seq_len_output
    )
    torch_state_dict = torch.load(pytorch_model_path, map_location=device)
    torch_model.load_state_dict(torch_state_dict)
    torch_model.to(device)
    torch_model.eval()
    print("✓ Loaded PyTorch model")
    
    # Load JAX model
    with open(jax_model_path, 'rb') as f:
        jax_model_dict = pickle.load(f)
    flax_model = DCTPoseTransformerFlax(**jax_model_dict['config'])
    flax_params = jax_model_dict['params']
    print("✓ Loaded JAX model")
    
    # Load dataset
    print("\n" + "="*60)
    print("Loading H36M Dataset")
    print("="*60)
    
    dataset = Human36mDataset3D(h36m_data_path, split='test', 
                                input_frames=50, predict_frames=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    print(f"✓ Loaded {len(dataset)} sequences")
    
    # Get DCT matrices
    N = 50
    dct_m, idct_m = get_dct_matrix(N)
    dct_m_torch = torch.from_numpy(dct_m).float().to(device)
    idct_m_torch = torch.from_numpy(idct_m).float().to(device)
    idct_m_jax = jnp.array(idct_m, dtype=jnp.float32)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)
    
    # Process batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:  # Process first 2 batches
            break
        
        input_pose = batch['input_pose'].to(device)
        target_pose = batch['target_pose'].to(device)
        
        # Apply DCT preprocessing
        input_pose_ = torch.matmul(input_pose.transpose(1, 2), 
                                   dct_m_torch.transpose(0, 1)).transpose(1, 2)
        input_pose_ = input_pose_ / 1000  # Convert to meters
        
        # PyTorch inference with CORRECT postprocessing
        with torch.no_grad():
            pred_output_torch, _ = torch_model(input_pose_)
            
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
        pred_output_jax, _ = flax_model.apply({'params': flax_params}, input_pose_jax)
        
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
        
        # Get target poses
        target_pose_np = target_pose.detach().cpu().numpy()
        
        # Predictions are now in mm and already postprocessed (10 frames)
        pred_torch_future = poses_torch_np
        pred_jax_future = poses_jax_np
        
        print(f"\nBatch {batch_idx}:")
        
        # Visualize multiple samples
        for sample_idx in range(min(2, pred_torch_future.shape[0])):
            # Single frame
            torch_err, jax_err, diff = visualize_predictions(
                target_pose_np, 
                pred_torch_future, 
                pred_jax_future,
                frame_idx=4,  # Middle frame
                sample_idx=sample_idx,
                save_path=f'{output_dir}/vis_batch{batch_idx}_sample{sample_idx}_frame4.png'
            )
            print(f"  Sample {sample_idx}: PyTorch={torch_err:.1f}mm, "
                  f"JAX={jax_err:.1f}mm, Diff={diff:.1f}mm")
            
            # Sequence
            if sample_idx == 0:
                visualize_sequence(
                    target_pose_np,
                    pred_torch_future,
                    pred_jax_future,
                    sample_idx=sample_idx,
                    save_path=f'{output_dir}/vis_batch{batch_idx}_sample{sample_idx}_sequence.png',
                    frames_to_plot=[0, 4, 9]
                )
                print(f"  Saved sequence visualization")
    
    print("\n" + "="*60)
    print(f"✓ Visualizations saved to {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()

