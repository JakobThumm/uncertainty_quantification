"""
Dataset class for preprocessed H36M data with bounding box extraction

This dataset loads preprocessed H36M images where human detection and
bounding box extraction have already been performed. This significantly
speeds up data loading during training and inference.

The preprocessed data is organized in H36M-like structure:
- preprocessed_dir/S1/PreprocessedImages/action.camera.npy
- preprocessed_dir/S1/PreprocessedPoses/action.camera.npz
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import jax.numpy as jnp
from src.datasets.utils import get_loader

# Dataset splits matching original H36M
SPLIT = {
    'train': ['S1', 'S6', 'S7', 'S8'],
    'validation': ['S9'],
    'test': ['S11']
}


class Human36mPreprocessedDataset(Dataset):
    """
    Dataset class for preprocessed H36M pose estimation data.

    This dataset loads images that have already been:
    - Resized for human detection
    - Had bounding boxes detected
    - Cropped to bounding box region
    - Preprocessed with RegressFlow transformations

    Args:
        preprocessed_dir: Path to preprocessed dataset directory
        split: One of 'train', 'validation', or 'test'
        return_metadata: Whether to return metadata dict along with image and pose
        jax_format: If True, return JAX arrays instead of PyTorch tensors
    """

    def __init__(self, preprocessed_dir, split='train', return_metadata=False, jax_format=False):
        self.preprocessed_dir = preprocessed_dir
        self.split = split
        self.return_metadata = return_metadata
        self.jax_format = jax_format

        # Build index of all samples
        self.samples = []

        for subject in SPLIT[split]:
            subject_dir = os.path.join(preprocessed_dir, subject)
            images_dir = os.path.join(subject_dir, 'PreprocessedImages')
            poses_dir = os.path.join(subject_dir, 'PreprocessedPoses')

            if not os.path.exists(images_dir) or not os.path.exists(poses_dir):
                print(f"Warning: Skipping {subject} - directories not found")
                continue

            # Find all preprocessed sequences
            for img_file in os.listdir(images_dir):
                if not img_file.endswith('.npy'):
                    continue

                base_name = os.path.splitext(img_file)[0]
                poses_file = f"{base_name}.npz"

                img_path = os.path.join(images_dir, img_file)
                poses_path = os.path.join(poses_dir, poses_file)

                if not os.path.exists(poses_path):
                    print(f"Warning: Pose file not found for {img_file}")
                    continue

                # Load sequence to count frames
                images = np.load(img_path, mmap_mode='r')  # Memory-mapped for efficiency
                num_frames = len(images)

                # Add each frame as a separate sample
                for frame_idx in range(num_frames):
                    self.samples.append({
                        'subject': subject,
                        'action': base_name,
                        'img_path': img_path,
                        'poses_path': poses_path,
                        'frame_idx': frame_idx
                    })

        print(f"Loaded {len(self.samples)} samples for {split} split from {preprocessed_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # Load image for this frame (using memory mapping for efficiency)
        images = np.load(sample_info['img_path'], mmap_mode='r')
        image = np.array(images[sample_info['frame_idx']])  # Copy to RAM: (3, 256, 192)

        # Load pose data for this frame
        poses_data = np.load(sample_info['poses_path'])
        frame_idx = sample_info['frame_idx']

        pose_normalized = poses_data['poses_normalized'][frame_idx]  # (13, 2)

        # Flatten pose for model output format
        pose_flat = pose_normalized.reshape(-1).astype(np.float32)  # Shape: (26,)

        if self.jax_format:
            # Convert to JAX arrays
            image = jnp.array(image, dtype=jnp.float32)
            pose_flat = jnp.array(pose_flat, dtype=jnp.float32)
        else:
            # Convert to PyTorch tensors
            image = torch.FloatTensor(image)
            pose_flat = torch.FloatTensor(pose_flat)

        if self.return_metadata:
            # Build metadata dict
            metadata = {
                'subject': sample_info['subject'],
                'action': sample_info['action'],
                'original_frame_idx': int(poses_data['valid_frame_indices'][frame_idx]),
                'pose_normalized': poses_data['poses_normalized'][frame_idx].tolist(),
                'pose_pixel': poses_data['poses_pixel'][frame_idx].tolist(),
                'bbox': poses_data['bboxes'][frame_idx].tolist(),
                'center': poses_data['centers'][frame_idx].tolist(),
                'scale': poses_data['scales'][frame_idx].tolist(),
                'trans': poses_data['trans'][frame_idx].tolist(),
                'scale_factors': tuple(poses_data['scale_factors'][frame_idx].tolist()),
                'original_dims': tuple(poses_data['original_dims'].tolist())
            }
            return image, pose_flat, metadata
        else:
            return image, pose_flat

    def get_sample_info(self, idx):
        """Get metadata for a specific sample without loading the image"""
        sample_info = self.samples[idx]
        poses_data = np.load(sample_info['poses_path'])
        frame_idx = sample_info['frame_idx']

        return {
            'subject': sample_info['subject'],
            'action': sample_info['action'],
            'original_frame_idx': int(poses_data['valid_frame_indices'][frame_idx]),
            'pose_normalized': poses_data['poses_normalized'][frame_idx].tolist(),
            'pose_pixel': poses_data['poses_pixel'][frame_idx].tolist(),
            'bbox': poses_data['bboxes'][frame_idx].tolist(),
            'center': poses_data['centers'][frame_idx].tolist(),
            'scale': poses_data['scales'][frame_idx].tolist(),
            'trans': poses_data['trans'][frame_idx].tolist(),
            'scale_factors': tuple(poses_data['scale_factors'][frame_idx].tolist()),
            'original_dims': tuple(poses_data['original_dims'].tolist())
        }


class Human36mPreprocessedDatasetJAX:
    """
    JAX-compatible wrapper for Human36mPreprocessedDataset

    This class provides a simpler interface for JAX training loops
    that expect data as numpy/JAX arrays rather than PyTorch tensors.
    """

    def __init__(self, preprocessed_dir, split='train', return_metadata=False):
        self.dataset = Human36mPreprocessedDataset(
            preprocessed_dir=preprocessed_dir,
            split=split,
            return_metadata=return_metadata,
            jax_format=True
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_sample_info(self, idx):
        return self.dataset.get_sample_info(idx)


def get_h36m_preprocessed(
    preprocessed_dir,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    return_metadata=False,
    n_samples=None
):
    """
    Get data loaders for preprocessed H36M dataset

    Args:
        preprocessed_dir: Path to preprocessed dataset directory
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
    train_dataset = Human36mPreprocessedDataset(
        preprocessed_dir=preprocessed_dir,
        split='train',
        return_metadata=return_metadata,
        jax_format=False
    )

    test_dataset = Human36mPreprocessedDataset(
        preprocessed_dir=preprocessed_dir,
        split='validation',  # Using validation split as test
        return_metadata=return_metadata,
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


def verify_preprocessed_dataset(preprocessed_dir, split='train', num_samples=5):
    """
    Verify preprocessed dataset by loading and displaying sample information

    Args:
        preprocessed_dir: Path to preprocessed dataset directory
        split: Which split to verify
        num_samples: Number of samples to check
    """
    print(f"\nVerifying preprocessed H36M dataset: {preprocessed_dir}")
    print(f"Split: {split}")
    print(f"{'='*80}\n")

    dataset = Human36mPreprocessedDataset(
        preprocessed_dir=preprocessed_dir,
        split=split,
        return_metadata=True
    )

    print(f"Total samples in {split}: {len(dataset)}")

    # Check a few samples
    for i in range(min(num_samples, len(dataset))):
        image, pose, metadata = dataset[i]

        print(f"\nSample {i}:")
        print(f"  Image shape: {image.shape}")
        print(f"  Pose shape: {pose.shape}")
        print(f"  Subject: {metadata['subject']}")
        print(f"  Action: {metadata['action']}")
        print(f"  Original frame: {metadata['original_frame_idx']}")
        print(f"  Bbox: {metadata['bbox']}")
        print(f"  Image value range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Pose normalized range: [{np.min(metadata['pose_normalized']):.3f}, {np.max(metadata['pose_normalized']):.3f}]")

    print(f"\n{'='*80}")
    print("Verification complete!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Verify preprocessed H36M dataset')
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                       help='Path to preprocessed dataset directory')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'validation', 'test'],
                       help='Which split to verify')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to check')

    args = parser.parse_args()

    verify_preprocessed_dataset(
        preprocessed_dir=args.preprocessed_dir,
        split=args.split,
        num_samples=args.num_samples
    )
