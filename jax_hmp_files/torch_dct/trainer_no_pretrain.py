# Standard imports
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from spacepy import pycdf
from transformer_model_and_loss import DCTPoseTransformer, pose_prediction_loss, uncertainty_loss_with_covariance

# Define skeleton connections for visualization
# Each tuple represents a connection between two joints (e.g., (0,1) connects joint 0 to joint 1)
CONNECTIONS_13 = [
    (0, 1), (0, 2),  # Nose to shoulders
    (1, 3), (3, 5),  # Left arm
    (2, 4), (4, 6),  # Right arm
    (1, 2), (1, 7), (2, 8),  # Shoulders to hips
    (7, 8),  # Connect hips
    (7, 9), (9, 11),  # Left leg
    (8, 10), (10, 12)  # Right leg
]

# Joint indices for converting from H36M's 32 joints to our 17-joint subset
JOINT_IDX_17 = [0, 1, 2, 3, 6, 7, 8, 12, 16, 14, 15, 17, 18, 19, 25, 26, 27]

# Further reduce from 17 joints to our final 13-joint representation
JOINT_IDX_13 = [9, 14, 11, 15, 12, 16, 13, 1, 4, 2, 5, 3, 6]

# Define train/validation/test split using Human3.6M subject IDs

# Frei Version, standard h36m spilt way
# SPLIT = {
#     'train': ['S1', 'S6', 'S7', 'S8', 'S9'],
#     'validation': ['S11'],
#     'test': ['S5']
# }

SPLIT = {
    'train': ['S1', 'S6', 'S7', 'S8', 'S9'],
    'validation': ['S11'],
    'test': ['S5']
}

class Human36mDataset3D(Dataset):
    """
    Dataset class for Human3.6M motion data.
    Handles loading and preprocessing of 3D pose sequences.
    """
    def __init__(self, base_directory, split='train', input_frames=50, predict_frames=10):
        """
        Args:
            base_directory (str): Path to Human3.6M dataset
            split (str): One of 'train', 'validation', or 'test'
            input_frames (int): Number of input frames for prediction
            predict_frames (int): Number of frames to predict
        """
        self.input_frames = input_frames
        self.predict_frames = predict_frames
        self.data = []  # Initialize empty list
        self.data = self.load_data(base_directory, split)  # Load the data
        
    def load_data(self, base_directory, split):
        """
        Load and preprocess motion sequences from Human3.6M dataset.
        
        Args:
            base_directory (str): Path to Human3.6M dataset
            split (str): Dataset split to load
            
        Returns:
            list: Processed motion sequences
        """
        all_data = []
        for subject in SPLIT[split]:
            directory = os.path.join(base_directory, subject, 'D3_Positions') # Poses_D3_Positions in Frei version
            for filename in os.listdir(directory):
                if filename.endswith('.cdf'):
                    file_path = os.path.join(directory, filename)
                    with pycdf.CDF(file_path) as cdf:
                        # Load and reshape poses
                        poses = cdf['Pose'][:]
                        poses = poses.reshape(-1, 32, 3)
                        # Convert to 13-joint representation
                        poses_13 = poses[:, JOINT_IDX_17, :]
                        poses_13 = poses_13[:, JOINT_IDX_13, :]
                        poses_13 = poses_13.reshape(poses_13.shape[0], -1)

                        # Downsample to 25 fps
                        # Create sequences with 2 different offsets so that all data is used
                        for offset in [0, 1]:
                            downsampled_poses = poses_13[offset::2]  # Downsample by factor of 2
                            # Create overlapping windows of frames
                            for i in range(len(downsampled_poses) - self.input_frames - self.predict_frames + 1):
                                window = downsampled_poses[i:i + self.input_frames + self.predict_frames]
                                all_data.append(window)
        print(f"Loaded {len(all_data)} sequences for {split} split")
        return all_data

    def __len__(self):
        """Return the total number of sequences in the dataset"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sequence by index.
        
        Args:
            idx (int): Index of sequence to retrieve
            
        Returns:
            dict: Contains input and target poses as torch tensors
        """
        sequence = self.data[idx]
        input_pose = sequence[:self.input_frames]
        target_pose = sequence[self.input_frames:]
        return {
            "input_pose": torch.FloatTensor(input_pose),
            "target_pose": torch.FloatTensor(target_pose),
        }

def get_dct_matrix(N):
    """
    Compute the Discrete Cosine Transform (DCT) matrix and its inverse.
    
    Args:
        N (int): Size of the DCT matrix
        
    Returns:
        tuple: (dct_matrix, idct_matrix)
    """
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m

def plot_ellipsoid_with_covariance(ax, center, cov_matrix, color='g', alpha=0.1):
    """
    Plot uncertainty ellipsoid using the full covariance matrix.
    
    Args:
        ax: Matplotlib 3D axis
        center (array): Joint position [3]
        cov_matrix (tensor): 3x3 covariance matrix
        color (str): Color of ellipsoid
        alpha (float): Transparency of ellipsoid
    """
    # Compute eigenvalues and eigenvectors of covariance matrix
    eigenvals, eigenvecs = torch.linalg.eigh(cov_matrix)
    
    # Convert to numpy for visualization
    eigenvals = eigenvals.cpu().numpy()
    eigenvecs = eigenvecs.cpu().numpy()
    
    # Compute radii (sqrt of eigenvalues)
    radii = np.sqrt(np.maximum(eigenvals, 0))
    
    # Generate points for unit sphere
    u = np.linspace(0.0, 2.0 * np.pi, 20)
    v = np.linspace(0.0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Stack points
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    
    # Transform points
    transformed_points = (points * radii) @ eigenvecs.T + center
    
    # Reshape back to grid
    x = transformed_points[:, 0].reshape(20, 20)
    y = transformed_points[:, 1].reshape(20, 20)
    z = transformed_points[:, 2].reshape(20, 20)
    
    # Plot surface
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

from matplotlib import cm
def viz_3d_all_frames(output, target_pose, cov_matrices, epoch = 0):
    """
    Visualize predicted poses and ground truth with uncertainty ellipsoids.

    Args:
        output (tensor): Predicted poses [batch_size, frames, joints*3]
        target_pose (tensor): Ground truth poses [batch_size, frames, joints*3]
        cov_matrices (tensor): Covariance matrices [batch_size, frames, joints, 3, 3]
        epoch (int): current epoch
    """
    # Convert to numpy
    output_np = output.cpu().numpy()
    target_pose_np = target_pose.cpu().numpy()
    predict_frames = 10
    
    # Extract the sample and reshape
    pred_poses = output_np[0][:predict_frames].reshape(predict_frames, 13, 3)
    true_poses = target_pose_np[0][:predict_frames].reshape(predict_frames, 13, 3)
    # print("mean of std_dev: ", np.mean(std_dev))
    # print("max of std_dev: ", np.max(std_dev))
    # print("median of std_dev: ", np.median(std_dev))
    
    # Number of frames
    num_frames = pred_poses.shape[0]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot poses over time
    for t in range(0, num_frames, 1):
        pred_pose = pred_poses[t]  # [13, 3]
        true_pose = true_poses[t]  # [13, 3]
        
        # Plot predicted pose (red) and ground truth (green)
        ax.scatter(pred_pose[:, 0], pred_pose[:, 1], pred_pose[:, 2], c='r', marker='o', s=10)
        ax.scatter(true_pose[:, 0], true_pose[:, 1], true_pose[:, 2], c='g', marker='o', s=10)
        
        # Draw skeleton connections for the first frame
        # if t == 0:
        #     for connection in CONNECTIONS_13:
        #         ax.plot(pred_pose[connection, 0], pred_pose[connection, 1], pred_pose[connection, 2], 
        #                color='r', linestyle='-', linewidth=1, label="prediction")
        #         ax.plot(true_pose[connection, 0], true_pose[connection, 1], true_pose[connection, 2], 
        #                color='g', linestyle='-', linewidth=1, label="ground truth")
        # Draw skeleton connections for the last frame
        if t == num_frames-1:
            for connection in CONNECTIONS_13:
                ax.plot(pred_pose[connection, 0], pred_pose[connection, 1], pred_pose[connection, 2], 
                       color='r', linestyle='-', linewidth=1.5, label="prediction")
                ax.plot(true_pose[connection, 0], true_pose[connection, 1], true_pose[connection, 2], 
                       color='g', linestyle='-', linewidth=1.5, label="ground truth")
        
        # # Draw uncertainty ellipsoids for each joint
        # for t in range(num_frames):
        #     for j in range(13):
        #         plot_ellipsoid_with_covariance(
        #             ax,
        #             pred_poses[t, j],
        #             cov_matrices[0, t, j],  # Use first batch item
        #             color='r',
        #             alpha=0.1
        #         )
        
    # Set plot properties
    ax.set_title('Predicted Poses with Uncertainty Ellipsoids')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Auto-scale axes to fit all points
    all_points = np.vstack([pred_poses.reshape(-1, 3), true_poses.reshape(-1, 3)])
    max_range = np.array([all_points[:,0].ptp(), all_points[:,1].ptp(), all_points[:,2].ptp()]).max() / 2.0
    mid_x = (all_points[:,0].min() + all_points[:,0].max()) * 0.5
    mid_y = (all_points[:,1].min() + all_points[:,1].max()) * 0.5
    mid_z = (all_points[:,2].min() + all_points[:,2].max()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'visualization/visualization_example_last_3d(e={epoch}).png')
    plt.close()

def calculate_mpjpe_batch(pred_pose, true_pose):
    """
    Calculate Mean Per Joint Position Error (MPJPE) for batched data.
    
    Args:
        pred_pose (tensor): Predicted poses [batch_size, seq_len, joints*3]
        true_pose (tensor): Ground truth poses [batch_size, seq_len, joints*3]
        
    Returns:
        tuple: (average MPJPE across all frames, list of MPJPE for each frame)
    """
    batch_size, n_frames, _ = pred_pose.shape
    
    # Reshape to (batch_size, n_frames, n_joints, 3)
    pred_pose = pred_pose.reshape(batch_size, n_frames, -1, 3)
    true_pose = true_pose.reshape(batch_size, n_frames, -1, 3)
    
    # Calculate the Euclidean distance for each joint
    joint_errors = torch.sqrt(torch.sum((pred_pose - true_pose)**2, dim=3))
    
    # Calculate mean error across all joints and samples in batch
    avg_mpjpe = torch.mean(joint_errors).item()
    
    # Calculate MPJPE for each frame
    frame_mpjpes = [torch.mean(joint_errors[:, i, :]).item() for i in range(n_frames)]
    
    return avg_mpjpe, frame_mpjpes

def main():
    """Main training loop"""
    torch.cuda.set_per_process_memory_fraction(0.8 , device=0)  # Allocates 50% of GPU 0 memory
    # Setup hardware and paths
    # base_directory = r"D:\Human3.6m"
    base_directory = '../data/H36M_FREI'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and load pretrained weights
    model = DCTPoseTransformer(input_dim=39, seq_len=50)
    # print(model)
    # model_path = r"transformer_checkpoint_detached_head.pth"
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # print("Loaded state ", model_path)

    model.to(device)
    model.train()
    
    # Setup data loading
    train_dataset = Human36mDataset3D(base_directory, split='train', input_frames=50, predict_frames=10)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    
    # Initialize training components
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    num_epochs = 50
    
    # Get DCT matrices for motion encoding
    N = 50
    dct_m, idct_m = get_dct_matrix(N)
    dct_m = torch.from_numpy(dct_m).float().to(device)
    idct_m = torch.from_numpy(idct_m).float().to(device)
    
    # Training loop
    for epoch in range(num_epochs):
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device and preprocess
            input_pose = batch['input_pose'].to(device)
            target_pose = batch['target_pose'].to(device)
            
            # Apply DCT and normalize
            input_pose_ = torch.matmul(input_pose.transpose(1, 2), dct_m.transpose(0, 1)).transpose(1, 2)
            input_pose_ = input_pose_ / 1000  # Convert to meters
            
            # Forward pass
            pred_poses, (var_params, cov_params) = model(input_pose_)
            
            # Post-process predictions
            pred_poses = pred_poses * 1000  # Convert back to millimeters
            # TODO: Ask Marian, seems input_pose_.shape[1] - pred_poses.shape[1] = 0
            # padded_mean seems do nothing
            padded_mean = F.pad(pred_poses, (0, 0, 0, input_pose_.shape[1] - pred_poses.shape[1], 0, 0))
            pred_poses = torch.matmul(padded_mean.transpose(1, 2), idct_m.transpose(0, 1)).transpose(1, 2)
            offset = input_pose[:, -1:, :]
            pred_poses = pred_poses[:, :10, :] + offset
            
            # Calculate losses
            unc_loss, cov_matrix = uncertainty_loss_with_covariance(
                target_pose, pred_poses, var_params[:, :10, :], cov_params[:, :10, :],
                beta=0.5, lambda_cov=0.01
            )
            pose_loss = pose_prediction_loss(pred_poses, target_pose)
            total_loss = unc_loss + pose_loss #Loss for Uncertainty Head and Pose Transformer
            
            # Calculate MPJPE
            batch_mpjpe, frame_mpjpes = calculate_mpjpe_batch(pred_poses.detach(), target_pose)
            mpjpe_4 = frame_mpjpes[3]  # 4th frame MPJPE
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()
            
            # Periodic visualization
            
            if batch_idx % 1200 == 0 and batch_idx != 0:
                with torch.no_grad():
                    viz_3d_all_frames(pred_poses, target_pose, cov_matrix, epoch)
            
            # Update progress bar with additional metrics
            pbar.update(1)
            pbar.set_postfix({
                'Pose Loss': f"{pose_loss.item():.4f}",
                'Unc Loss': f"{unc_loss.item():.4f}",
                'Var_params': f"{torch.mean(var_params).item():.5f}",
                'MPJPE': f"{batch_mpjpe:.2f}",
                'MPJPE@4': f"{mpjpe_4:.2f}",
            })
        
        pbar.close()

     # After training is complete, save the model
    save_path = 'transformer_model(no_pretrain).pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()



