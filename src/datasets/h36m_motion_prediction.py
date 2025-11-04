"""Dataset for the motion prediction task on the Human3.6M dataset."""

import os
from torch.utils.data import Dataset
from spacepy import pycdf
import torch
import jax.numpy as jnp
import numpy as np

from human_pose_pipeline.pose_estimation.h36m_settings import JOINT_IDX_17, JOINT_IDX_13
from human_pose_pipeline.motion_prediction.h36m_settings import INPUT_HORIZON_LENGTH, PREDICTION_HORIZON_LENGTH
from src.datasets.utils import get_loader

# Dataset splits matching original H36M
SPLIT = {"train": ["S1", "S6", "S7", "S8"], "validation": ["S9"], "test": ["S11"]}


class Human36mMotionDataset3D(Dataset):
    """Dataset class for Human3.6M motion data."""

    def __init__(
        self,
        base_directory,
        split="train",
        input_frames=INPUT_HORIZON_LENGTH,
        predict_frames=PREDICTION_HORIZON_LENGTH,
        jax_format=False
    ):
        self.input_frames = input_frames
        self.predict_frames = predict_frames
        self.jax_format = jax_format
        self.data = []
        self.data = self.load_data(base_directory, split)

    def load_data(self, base_directory, split):
        all_data = []
        for subject in SPLIT[split]:
            directory = os.path.join(base_directory, subject, "Poses_D3_Positions")
            if not os.path.exists(directory):
                print(f"Warning: Directory {directory} not found, skipping...")
                continue

            for filename in os.listdir(directory):
                if filename.endswith(".cdf"):
                    file_path = os.path.join(directory, filename)
                    with pycdf.CDF(file_path) as cdf:
                        poses = cdf["Pose"][:]
                        poses = poses.reshape(-1, 32, 3)
                        poses_13 = poses[:, JOINT_IDX_17, :]
                        poses_13 = poses_13[:, JOINT_IDX_13, :]
                        poses_13 = poses_13.reshape(poses_13.shape[0], -1)

                        for offset in [0, 1]:
                            downsampled_poses = poses_13[offset::2]
                            for i in range(len(downsampled_poses) - self.input_frames - self.predict_frames + 1):
                                window = downsampled_poses[i : i + self.input_frames + self.predict_frames]
                                all_data.append(window)
        print(f"Loaded {len(all_data)} sequences for {split} split")
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx]
        input_pose = sequence[: self.input_frames]
        target_pose = sequence[self.input_frames :]
        if self.jax_format:
            # Convert to JAX arrays
            input_pose = jnp.array(input_pose, dtype=jnp.float32)
            target_pose = jnp.array(target_pose, dtype=jnp.float32)
        else:
            # Convert to PyTorch tensors
            input_pose = torch.FloatTensor(input_pose)
            target_pose = torch.FloatTensor(target_pose)
        return {
            "input_pose": input_pose,
            "target_pose": target_pose,
        }


def get_h36m_motion_dataset(
    base_directory,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    n_samples=None
):
    """
    Get data loaders for preprocessed H36M dataset

    Args:
        base_directory: Path to dataset directory
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        split_train_val_ratio: Ratio for splitting train set into train/val
        return_metadata: Whether to return metadata with samples
        n_samples: Number of samples to use from dataset (None = use all)

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # Create datasets
    train_dataset = Human36mMotionDataset3D(
        base_directory=base_directory,
        split='train',
        jax_format=False
    )

    test_dataset = Human36mMotionDataset3D(
        base_directory=base_directory,
        split='validation',
        jax_format=False
    )

    # Subsample if n_samples is specified
    if n_samples is not None:
        import torch.utils.data
        n_samples_train = min(n_samples, len(train_dataset))
        # Randomly select n_samples_train indices
        np.random.seed(seed)
        train_indices = np.random.choice(len(train_dataset), n_samples_train, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        n_samples_test = min(n_samples, len(test_dataset))
        test_indices = np.random.choice(len(test_dataset), n_samples_test, replace=False)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    # Split train dataset into train/val
    train_loader, valid_loader = get_loader(
        train_dataset,
        split_train_val_ratio=split_train_val_ratio,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )

    # Create test loader
    test_loader = get_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        seed=seed
    )

    return train_loader, valid_loader, test_loader
