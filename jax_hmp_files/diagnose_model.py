"""
Diagnostic script to identify why the model produces collapsed poses.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('/home/skyle/Desktop/uq_benchmark/models/HMP')

from torch_dct.transformer_model_and_loss import DCTPoseTransformer
from transfer_dct_pose_transformer import Human36mDataset3D, get_dct_matrix


def diagnose_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("DIAGNOSTIC: Model Output Analysis")
    print("="*60)
    
    # Load model
    model_path = "transformer_model.pth"
    model = DCTPoseTransformer(
        input_dim=39, d_model=128, nhead=4, num_layers=2,
        seq_len=50, seq_len_output=10
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load one batch
    data_path = "/home/skyle/datasets/H36M_FREI"
    dataset = Human36mDataset3D(data_path, split='test', input_frames=50, predict_frames=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))
    
    input_pose = batch['input_pose'].to(device)
    target_pose = batch['target_pose'].to(device)
    
    # Get DCT matrix
    N = 50
    dct_m, idct_m = get_dct_matrix(N)
    dct_m_torch = torch.from_numpy(dct_m).float().to(device)
    idct_m_torch = torch.from_numpy(idct_m).float().to(device)
    
    print("\n1. Input Data Statistics")
    print("-" * 60)
    print(f"Input pose (raw) shape: {input_pose.shape}")
    print(f"Input pose mean: {input_pose.mean():.2f}, std: {input_pose.std():.2f}")
    print(f"Input pose range: [{input_pose.min():.2f}, {input_pose.max():.2f}]")
    print(f"Target pose mean: {target_pose.mean():.2f}, std: {target_pose.std():.2f}")
    print(f"Target pose range: [{target_pose.min():.2f}, {target_pose.max():.2f}]")
    
    # Apply DCT (as in training)
    print("\n2. After DCT Transform")
    print("-" * 60)
    input_dct = torch.matmul(input_pose.transpose(1, 2), 
                             dct_m_torch.transpose(0, 1)).transpose(1, 2)
    print(f"After DCT mean: {input_dct.mean():.2f}, std: {input_dct.std():.2f}")
    print(f"After DCT range: [{input_dct.min():.2f}, {input_dct.max():.2f}]")
    
    # Normalize (divide by 1000)
    print("\n3. After Normalization (/1000)")
    print("-" * 60)
    input_norm = input_dct / 1000
    print(f"After norm mean: {input_norm.mean():.4f}, std: {input_norm.std():.4f}")
    print(f"After norm range: [{input_norm.min():.4f}, {input_norm.max():.4f}]")
    
    # Model forward pass
    print("\n4. Model Output (before denormalization)")
    print("-" * 60)
    with torch.no_grad():
        output, (var, cov) = model(input_norm)
    
    print(f"Model output shape: {output.shape}")
    print(f"Model output mean: {output.mean():.6f}, std: {output.std():.6f}")
    print(f"Model output range: [{output.min():.6f}, {output.max():.6f}]")
    
    # Check if output is in the same space as input
    print("\n5. Output Space Analysis")
    print("-" * 60)
    print(f"Is output in DCT space? (should be similar to input_dct)")
    print(f"  Input (DCT) mean: {input_dct.mean():.2f}")
    print(f"  Output mean: {output.mean():.6f}")
    print(f"  Match? {'YES' if abs(output.mean()) < 1.0 else 'NO - output should be ~1000x larger!'}")
    
    # Denormalize output
    print("\n6. After Denormalization (*1000)")
    print("-" * 60)
    output_denorm = output * 1000
    print(f"Denormalized mean: {output_denorm.mean():.2f}, std: {output_denorm.std():.2f}")
    print(f"Denormalized range: [{output_denorm.min():.2f}, {output_denorm.max():.2f}]")
    
    # Apply inverse DCT?
    print("\n7. Check if IDCT is needed")
    print("-" * 60)
    # Try applying IDCT
    output_reshaped = output_denorm.transpose(1, 2)
    output_idct = torch.matmul(output_reshaped, idct_m_torch.transpose(0, 1))
    output_idct = output_idct.transpose(1, 2)
    
    print(f"After IDCT mean: {output_idct.mean():.2f}, std: {output_idct.std():.2f}")
    print(f"After IDCT range: [{output_idct.min():.2f}, {output_idct.max():.2f}]")
    print(f"Compare to target: target mean = {target_pose.mean():.2f}")
    
    # Check what the model was trained to predict
    print("\n8. Training Target Analysis")
    print("-" * 60)
    # What should the model output be during training?
    target_dct = torch.matmul(target_pose.transpose(1, 2),
                              dct_m_torch.transpose(0, 1)).transpose(1, 2)
    target_norm = target_dct / 1000
    
    print("If model is trained to output normalized DCT space:")
    print(f"  Target (norm DCT) mean: {target_norm.mean():.4f}, std: {target_norm.std():.4f}")
    print(f"  Model output mean: {output.mean():.6f}, std: {output.std():.6f}")
    print(f"  Difference: {abs(target_norm.mean() - output.mean()):.4f}")
    
    # Compare future frames
    print("\n9. Future Prediction Comparison (last 10 frames)")
    print("-" * 60)
    output_future = output[:, -10:, :]
    target_future = target_pose[:, :10, :]  # First 10 frames of target
    
    print(f"Output future shape: {output_future.shape}")
    print(f"Target future shape: {target_future.shape}")
    
    # Denormalize and apply IDCT to predictions
    pred_denorm = output_future * 1000
    pred_idct = torch.matmul(pred_denorm.transpose(1, 2),
                             idct_m_torch.transpose(0, 1)).transpose(1, 2)
    
    print(f"\nPrediction (after denorm + IDCT):")
    print(f"  Mean: {pred_idct.mean():.2f}, Std: {pred_idct.std():.2f}")
    print(f"Target (ground truth):")
    print(f"  Mean: {target_future.mean():.2f}, Std: {target_future.std():.2f}")
    
    # Calculate error
    error = torch.abs(pred_idct - target_future).mean()
    print(f"\nMean Absolute Error: {error:.2f} mm")
    
    # Check bone lengths in predictions
    print("\n10. Bone Length Check in Predictions")
    print("-" * 60)
    pred_np = pred_idct[0, 0].cpu().numpy().reshape(13, 3)
    
    # Hip to Spine
    hip_spine_dist = np.linalg.norm(pred_np[0] - pred_np[7])
    print(f"Hip-Spine distance: {hip_spine_dist:.2f} mm (expected: 100-300 mm)")
    
    # RHip to RKnee
    rhip_rknee_dist = np.linalg.norm(pred_np[1] - pred_np[2])
    print(f"RHip-RKnee distance: {rhip_rknee_dist:.2f} mm (expected: 350-550 mm)")
    
    print("\n" + "="*60)
    print("DIAGNOSIS SUMMARY")
    print("="*60)
    
    issues = []
    
    # Check if model output is too small
    if abs(output.mean()) < 0.01:
        issues.append("Model output is near zero - model may not be trained properly")
    
    # Check if denormalization helps
    if output_denorm.std() < 10:
        issues.append("Even after denormalization, output std is very low")
    
    # Check if IDCT is needed
    if abs(output_idct.mean() - target_pose.mean()) > 500:
        issues.append("IDCT may not be correctly applied")
    
    # Check bone lengths
    if hip_spine_dist < 50 or rhip_rknee_dist < 100:
        issues.append("Predicted bone lengths are unrealistically small")
    
    if len(issues) == 0:
        print("✓ Model appears to be working correctly")
    else:
        print("⚠️  Issues detected:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\nLikely causes:")
        print("  - Model not trained long enough")
        print("  - Wrong loss function or training target")
        print("  - Data preprocessing mismatch between training and inference")
        print("  - Model architecture issue")
        
        print("\nNext steps:")
        print("  1. Check training script for correct loss computation")
        print("  2. Verify that training converged (check loss curves)")
        print("  3. Test with a simple input to see if model can predict anything")
        print("  4. Try loading a checkpoint from earlier in training")


if __name__ == "__main__":
    diagnose_model()

