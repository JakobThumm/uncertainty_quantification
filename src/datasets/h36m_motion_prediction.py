"""Dataset for the motion prediction task on the Human3.6M dataset."""

import os
from typing import Optional
import torch.utils.data
from torch.utils.data import Dataset
from spacepy import pycdf
import torch
import jax.numpy as jnp
import numpy as np

from human_pose_pipeline.pose_estimation.h36m_settings import JOINT_IDX_17, JOINT_IDX_13
from human_pose_pipeline.motion_prediction.h36m_settings import (
    INPUT_HORIZON_LENGTH,
    PREDICTION_HORIZON_LENGTH,
    REDUCED_TIMESTEP,
    REDUCED_JOINT_INDICES,
    FAKE_INPUT_UNCERTAINTY
)
from src.datasets.utils import get_loader

# Dataset splits matching original H36M
SPLIT = {"train": ["S1", "S6", "S7", "S8", "S9"], "validation": ["S11"], "test": ["S5"]}


class Human36mMotionDataset3D(Dataset):
    """Dataset class for Human3.6M motion data."""

    def __init__(
        self,
        base_directory,
        split="train",
        input_frames=INPUT_HORIZON_LENGTH,
        predict_frames=PREDICTION_HORIZON_LENGTH,
        jax_format=False,
        reduce_size=False,
        reduced_timestep=REDUCED_TIMESTEP,
        reduced_joints=REDUCED_JOINT_INDICES,
        ood=False,
        input_uncertainty=False,
        directory_uncertain=None,
    ):
        self.input_frames = input_frames
        self.predict_frames = predict_frames
        self.jax_format = jax_format
        self.pose_data = []
        if not input_uncertainty:
            self.pose_data = self.load_data(base_directory, split)
            self.covariance_data = None
        else:
            assert directory_uncertain is not None, "directory_uncertain must be provided when input_uncertainty is True"
            self.pose_data, self.covariance_data = self.load_data_preprocessed(
                base_directory_uncertain=directory_uncertain,
                base_directory_gt=base_directory,
                split=split
            )
        self.reduce_size = reduce_size
        self.reduced_timestep = reduced_timestep
        self.reduced_joints = reduced_joints
        self.ood = ood
        self.input_uncertainty = input_uncertainty

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

    def load_data_preprocessed(self, base_directory_uncertain, base_directory_gt, split):
        all_poses = []
        all_covariances = []
        for subject in SPLIT[split]:
            uncertain_directory = os.path.join(base_directory_uncertain, subject)
            if not os.path.exists(uncertain_directory):
                print(f"Warning: Directory {uncertain_directory} not found, skipping...")
                continue
            for filename in os.listdir(uncertain_directory):
                if not filename.endswith(".npz"):
                    continue
                action = os.path.splitext(filename)[0]
                file_path = os.path.join(uncertain_directory, filename)
                data = np.load(file_path)
                pred_poses = data['poses_3d']  # (num_frames, 13, 3)
                covariances = data['covariances_3d']  # (num_frames, 13, 3, 3)
                valid_mask = data['valid_mask']  # (num_frames,)
                pred_poses = pred_poses.reshape(pred_poses.shape[0], -1)  # (num_frames, 13*3)
                covariances = covariances.reshape(covariances.shape[0], -1)  # (num_frames, 13*3*3)
                valid_mask = valid_mask.astype(bool)

                gt_file = os.path.join(base_directory_gt, subject, "Poses_D3_Positions", f"{action}.cdf")
                if os.path.exists(gt_file):
                    with pycdf.CDF(gt_file) as cdf:
                        gt_poses = cdf["Pose"][:]
                        gt_poses = gt_poses.reshape(-1, 32, 3)
                        gt_poses_13 = gt_poses[:, JOINT_IDX_17, :]
                        gt_poses_13 = gt_poses_13[:, JOINT_IDX_13, :]
                        gt_poses_13 = gt_poses_13.reshape(gt_poses_13.shape[0], -1)
                else:
                    print(f"Warning: GT file {gt_file} not found, using predicted poses as GT.")
                    gt_poses_13 = pred_poses.copy()

                # Trim sequences to match lengths
                min_length = min(len(pred_poses), len(gt_poses_13), len(covariances), len(valid_mask))
                pred_poses = pred_poses[:min_length]
                gt_poses_13 = gt_poses_13[:min_length]
                covariances = covariances[:min_length]
                valid_mask = valid_mask[:min_length]

                for offset in [0, 1]:
                    downsampled_poses = pred_poses[offset::2]
                    downsampled_covariances = covariances[offset::2]
                    downsampled_valid_mask = valid_mask[offset::2]
                    downsampled_gt_poses = gt_poses_13[offset::2]
                    for i in range(len(downsampled_poses) - self.input_frames - self.predict_frames + 1):
                        poses_window = downsampled_poses[i : i + self.input_frames + self.predict_frames]
                        # Replace the target frames with ground truth poses
                        poses_window[-self.predict_frames:] = downsampled_gt_poses[i + self.input_frames : i + self.input_frames + self.predict_frames]
                        covariances_window = downsampled_covariances[i : i + self.input_frames + self.predict_frames]
                        valid_mask_window = downsampled_valid_mask[i : i + self.input_frames + self.predict_frames]
                        if sum(valid_mask_window) < self.input_frames + self.predict_frames:
                            continue
                        all_poses.append(poses_window)
                        all_covariances.append(covariances_window)
        all_poses = np.array(all_poses)
        all_covariances = np.array(all_covariances)
        print(f"Loaded {len(all_poses)} sequences for {split} split from preprocessed data")
        return all_poses, all_covariances

    def __len__(self):
        return len(self.pose_data)

    def __getitem__(self, idx):
        sequence = self.pose_data[idx]
        input_pose = sequence[: self.input_frames]
        if self.ood:
            # Randomly shuffle the input sequence in the first dimension for OOD testing
            input_pose = input_pose.copy()
            np.random.shuffle(input_pose)
        target_pose = sequence[self.input_frames :]
        if self.reduce_size:
            # Extract only the specified timestep and joints
            target_pose_timestep = target_pose[self.reduced_timestep]  # [num_joints*3]
            target_pose_timestep = target_pose_timestep.reshape(-1, 3)  # [num_joints, 3]
            reduced_target = target_pose_timestep[self.reduced_joints, :]  # [len(reduced_joints), 3]
            target_pose = reduced_target.reshape(-1)  # [len(reduced_joints)*3]
        if self.input_uncertainty and self.covariance_data is not None:
            covariance_sequence = self.covariance_data[idx]
            input_covariances = covariance_sequence[: self.input_frames]
            if self.ood:
                # Randomly shuffle the input covariances in the first dimension for OOD testing
                input_covariances = input_covariances.copy()
                np.random.shuffle(input_covariances)
            # Append to input_pose
            input_pose = np.concatenate([input_pose, input_covariances], axis=-1)
        if self.jax_format:
            # Convert to JAX arrays
            input_pose = jnp.array(input_pose, dtype=jnp.float32)
            target_pose = jnp.array(target_pose, dtype=jnp.float32)
        else:
            # Convert to PyTorch tensors
            input_pose = torch.FloatTensor(input_pose)
            target_pose = torch.FloatTensor(target_pose)
        return [input_pose, target_pose]


def subsample_dataset(dataset, n_samples, seed=0):
    """Subsample a dataset to a specified number of samples.

    Args:
        dataset: The original dataset (torch.utils.data.Dataset)
        n_samples: Number of samples to select
        seed: Random seed for reproducibility

    Returns:
        Subsampled dataset (torch.utils.data.Subset)
    """
    n_samples_train = min(n_samples, len(dataset))
    # Randomly select n_samples_train indices
    np.random.seed(seed)
    train_indices = np.random.choice(len(dataset), n_samples_train, replace=False)
    subsampled_dataset = torch.utils.data.Subset(dataset, train_indices)
    return subsampled_dataset


def get_h36m_motion_dataset_function(
    base_directory: str,
    batch_size: int = 128,
    shuffle: bool = False,
    seed: int = 0,
    split_train_val_ratio: float = 1.0,
    n_samples: Optional[int] = None,
    input_uncertainty: bool = False,
    reduce_size: bool = False,
    ood: bool = False,
    directory_uncertain: Optional[str] = None
):
    """
    Get data loaders for preprocessed H36M dataset

    Args:
        base_directory: Path to dataset directory
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        split_train_val_ratio: Ratio for splitting train set into train/val -> Not used for this dataset!
        return_metadata: Whether to return metadata with samples
        n_samples: Number of samples to use from dataset (None = use all),
        input_uncertainty: Add artificial input uncertainty of this amount, defaults to None.
        reduced_size: Reduced output size of only head and two hand poses.
        ood: Shuffle input poses in time dimension.
        directory_uncertain: Directory containing preprocessed uncertain data.

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    # Create datasets
    train_dataset = Human36mMotionDataset3D(
        base_directory=base_directory,
        split='train',
        jax_format=False,
        input_uncertainty=input_uncertainty,
        reduce_size=reduce_size,
        ood=ood,
        directory_uncertain=directory_uncertain
    )

    validation_dataset = Human36mMotionDataset3D(
        base_directory=base_directory,
        split='validation',
        jax_format=False,
        input_uncertainty=input_uncertainty,
        reduce_size=reduce_size,
        ood=ood,
        directory_uncertain=directory_uncertain
    )

    test_dataset = Human36mMotionDataset3D(
        base_directory=base_directory,
        split='test',
        jax_format=False,
        input_uncertainty=input_uncertainty,
        reduce_size=reduce_size,
        ood=ood,
        directory_uncertain=directory_uncertain
    )

    # Subsample if n_samples is specified
    if n_samples is not None:
        train_dataset = subsample_dataset(train_dataset, n_samples, seed)
        validation_dataset = subsample_dataset(validation_dataset, n_samples, seed)
        test_dataset = subsample_dataset(test_dataset, n_samples, seed)

    # Split train dataset into train/val
    train_loader = get_loader(
        train_dataset,
        split_train_val_ratio=1.0,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )

    validation_loader = get_loader(
        validation_dataset,
        split_train_val_ratio=1.0,
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

    return train_loader, validation_loader, test_loader


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
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
    )


def get_h36m_motion_dataset_with_uncertainty(
    base_directory,
    directory_uncertain,
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
        directory_uncertain: Directory containing preprocessed uncertain data.
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        split_train_val_ratio: Ratio for splitting train set into train/val
        return_metadata: Whether to return metadata with samples
        n_samples: Number of samples to use from dataset (None = use all)

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
        input_uncertainty=True,
        directory_uncertain=directory_uncertain
    )


def get_h36m_motion_reduced_output_dataset(
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
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
        reduce_size=True
    )


def get_h36m_motion_ood_dataset(
    base_directory,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    n_samples=None
):
    """Get data loaders for the H36M motion prediction OOD dataset.
    
    This dataset shuffles the input sequences to create out-of-distribution samples.

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
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
        ood=True
    )


def get_h36m_motion_ood_dataset_with_uncertainty(
    base_directory,
    directory_uncertain,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    n_samples=None
):
    """Get data loaders for the H36M motion prediction OOD dataset.
    
    This dataset shuffles the input sequences to create out-of-distribution samples.

    Args:
        base_directory: Path to dataset directory
        directory_uncertain: Directory containing preprocessed uncertain data.
        batch_size: Batch size for data loaders
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        split_train_val_ratio: Ratio for splitting train set into train/val
        return_metadata: Whether to return metadata with samples
        n_samples: Number of samples to use from dataset (None = use all)

    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
        ood=True,
        input_uncertainty=True,
        directory_uncertain=directory_uncertain
    )


def get_h36m_motion_reduced_output_ood_dataset(
    base_directory,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    n_samples=None
):
    """Get data loaders for the H36M motion prediction OOD dataset.

    This dataset shuffles the input sequences to create out-of-distribution samples.
    This dataset comes with reduced output size (only specific timestep and joints).

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
    return get_h36m_motion_dataset_function(
        base_directory=base_directory,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        n_samples=n_samples,
        reduce_size=True,
        ood=True
    )
