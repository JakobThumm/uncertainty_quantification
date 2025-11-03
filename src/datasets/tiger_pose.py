"""
Tiger Pose Dataset for Out-of-Distribution Detection

This dataset contains pose annotations for tigers with 12 keypoints.
It's used as OOD data for human pose estimation models trained on H36M.

Tiger keypoints (12 total):
- Nose, left eye, right eye, left ear, right ear
- Front left paw, front right paw, back left paw, back right paw
- Tail start, tail middle, tail end
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import jax.numpy as jnp
from src.datasets.utils import get_loader


class TigerPoseDataset(Dataset):
    """
    Tiger Pose Dataset for OOD detection.

    The dataset follows YOLO format:
    - Images in images/train/ and images/val/
    - Labels in labels/train/ and labels/val/
    - Each label file contains: class_id x_center y_center width height kpt1_x kpt1_y ... kpt12_x kpt12_y

    For pose estimation, we extract and normalize the 12 keypoints to match
    the input format expected by the H36M-trained models.
    """

    def __init__(self, root_dir, split='train', transform=None, image_size=(256, 256)):
        """
        Args:
            root_dir: Path to tiger-pose dataset directory
            split: 'train' or 'val'
            transform: Optional transform to be applied on images
            image_size: Target image size (width, height)
        """
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size
        self.transform = transform

        # Setup paths
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.labels_dir = os.path.join(root_dir, 'labels', split)

        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(self.images_dir, '*.jpg')))

        # Filter to only include images that have corresponding label files
        self.valid_samples = []
        for img_path in self.image_files:
            img_name = os.path.basename(img_path)
            label_name = img_name.replace('.jpg', '.txt')
            label_path = os.path.join(self.labels_dir, label_name)

            if os.path.exists(label_path):
                self.valid_samples.append((img_path, label_path))

        print(f"TigerPose {split}: Found {len(self.valid_samples)} valid samples")

        # Default transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_path, label_path = self.valid_samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        original_width, original_height = image.size

        # Load label (YOLO format)
        with open(label_path, 'r') as f:
            line = f.readline().strip()

        if not line:
            # No annotations, return dummy data
            return {
                'image': self.transform(image),
                'keypoints': np.zeros((12, 2), dtype=np.float32),
                'valid_keypoints': np.zeros(12, dtype=bool),
                'image_path': img_path
            }

        # Parse YOLO annotation
        values = list(map(float, line.split()))

        # YOLO format: class_id x_center y_center width height kpt1_x kpt1_y ... kpt12_x kpt12_y
        if len(values) < 5 + 24:  # class + bbox + 12 keypoints * 2
            print(f"Warning: Insufficient annotation data in {label_path}")
            return {
                'image': self.transform(image),
                'keypoints': np.zeros((12, 2), dtype=np.float32),
                'valid_keypoints': np.zeros(12, dtype=bool),
                'image_path': img_path
            }

        # Extract keypoints (normalized coordinates)
        keypoints_norm = np.array(values[5:29]).reshape(12, 2)  # 12 keypoints, 2 coords each

        # Convert normalized coordinates to pixel coordinates
        keypoints_pixel = keypoints_norm.copy()
        keypoints_pixel[:, 0] *= original_width  # x coordinates
        keypoints_pixel[:, 1] *= original_height  # y coordinates

        # Scale keypoints to target image size
        keypoints_scaled = keypoints_pixel.copy()
        keypoints_scaled[:, 0] *= (self.image_size[0] / original_width)  # scale x
        keypoints_scaled[:, 1] *= (self.image_size[1] / original_height)  # scale y

        # Check for valid keypoints (non-zero coordinates indicate valid keypoints)
        valid_keypoints = ~((keypoints_norm[:, 0] == 0) & (keypoints_norm[:, 1] == 0))

        # Apply image transform
        image_tensor = self.transform(image)

        return {
            'image': image_tensor,
            'keypoints': keypoints_scaled.astype(np.float32),
            'valid_keypoints': valid_keypoints,
            'image_path': img_path,
            'original_size': (original_width, original_height)
        }

    def get_stats(self):
        """Get dataset statistics."""
        total_keypoints = 0
        valid_keypoints = 0

        for i in range(len(self)):
            sample = self[i]
            total_keypoints += len(sample['valid_keypoints'])
            valid_keypoints += sample['valid_keypoints'].sum()

        return {
            'total_samples': len(self),
            'total_keypoints': total_keypoints,
            'valid_keypoints': valid_keypoints,
            'keypoint_visibility_ratio': valid_keypoints / total_keypoints if total_keypoints > 0 else 0.0
        }


def get_tiger_pose(batch_size=32, shuffle=True, seed=0, data_path="../datasets"):
    """
    Get tiger pose data loaders.

    Returns:
        train_loader, val_loader, test_loader (test_loader = val_loader for tiger pose)
    """
    torch.manual_seed(seed)

    tiger_pose_path = os.path.join(data_path, "tiger-pose")

    if not os.path.exists(tiger_pose_path):
        raise FileNotFoundError(f"Tiger pose dataset not found at {tiger_pose_path}")

    # Create datasets
    train_dataset = TigerPoseDataset(tiger_pose_path, split='train')
    val_dataset = TigerPoseDataset(tiger_pose_path, split='val')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # For OOD detection, we typically use the same data for test
    test_loader = val_loader

    return train_loader, val_loader, test_loader


def tiger_pose_to_h36m_format(tiger_keypoints, valid_mask=None):
    """
    Convert tiger pose keypoints to a format compatible with H36M pose estimation models.

    This is a mapping function to allow tiger pose data to be processed by
    models trained on human poses. The mapping is approximate and serves
    the purpose of OOD detection.

    Tiger keypoints (12): nose, left_eye, right_eye, left_ear, right_ear,
                         front_left_paw, front_right_paw, back_left_paw, back_right_paw,
                         tail_start, tail_middle, tail_end

    H36M format (13): nose, left_shoulder, right_shoulder, left_elbow, right_elbow,
                     left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee,
                     left_ankle, right_ankle

    Args:
        tiger_keypoints: (N, 12, 2) tiger keypoints
        valid_mask: (N, 12) boolean mask for valid keypoints

    Returns:
        h36m_keypoints: (N, 13, 2) keypoints in H36M-like format
        h36m_valid_mask: (N, 13) boolean mask for valid keypoints
    """
    batch_size = tiger_keypoints.shape[0]
    h36m_keypoints = np.zeros((batch_size, 13, 2), dtype=np.float32)
    h36m_valid_mask = np.zeros((batch_size, 13), dtype=bool)

    if valid_mask is None:
        valid_mask = np.ones((batch_size, 12), dtype=bool)

    # Mapping from tiger to human pose (approximate)
    mapping = {
        0: 0,   # nose -> nose
        1: 1,   # left_eye -> left_shoulder (approximate)
        2: 2,   # right_eye -> right_shoulder (approximate)
        3: 3,   # left_ear -> left_elbow (approximate)
        4: 4,   # right_ear -> right_elbow (approximate)
        5: 5,   # front_left_paw -> left_wrist
        6: 6,   # front_right_paw -> right_wrist
        7: 7,   # back_left_paw -> left_hip
        8: 8,   # back_right_paw -> right_hip
        9: 9,   # tail_start -> left_knee
        10: 10, # tail_middle -> right_knee
        11: 11, # tail_end -> left_ankle
        # Note: right_ankle (index 12) will remain zero
    }

    for tiger_idx, h36m_idx in mapping.items():
        h36m_keypoints[:, h36m_idx] = tiger_keypoints[:, tiger_idx]
        h36m_valid_mask[:, h36m_idx] = valid_mask[:, tiger_idx]

    return h36m_keypoints, h36m_valid_mask


class TigerPosePreprocessedDataset(Dataset):
    """
    Dataset class for preprocessed Tiger Pose data.

    This dataset loads images that have already been:
    - Rotated by 90° to have approximately 3/4 aspect ratio
    - Scaled to the input size of the pose estimation network (192, 256)
    - Normalized with RGB values divided by 255 and offset applied

    The preprocessed data structure:
    - preprocessed_dir/train/PreprocessedImages/tiger_pose_train.npy
    - preprocessed_dir/train/PreprocessedPoses/tiger_pose_train.npz
    - preprocessed_dir/val/PreprocessedImages/tiger_pose_val.npy
    - preprocessed_dir/val/PreprocessedPoses/tiger_pose_val.npz

    Args:
        preprocessed_dir: Path to preprocessed dataset directory
        split: One of 'train' or 'val'
        return_metadata: Whether to return metadata dict along with image and pose
        jax_format: If True, return JAX arrays instead of PyTorch tensors
    """

    def __init__(self, preprocessed_dir, split='train', return_metadata=False, jax_format=False):
        self.preprocessed_dir = preprocessed_dir
        self.split = split
        self.return_metadata = return_metadata
        self.jax_format = jax_format

        # Build paths
        split_dir = os.path.join(preprocessed_dir, split)
        images_path = os.path.join(split_dir, 'PreprocessedImages', f'tiger_pose_{split}.npy')
        poses_path = os.path.join(split_dir, 'PreprocessedPoses', f'tiger_pose_{split}.npz')

        if not os.path.exists(images_path):
            raise FileNotFoundError(f"Images file not found: {images_path}")
        if not os.path.exists(poses_path):
            raise FileNotFoundError(f"Poses file not found: {poses_path}")

        # Load preprocessed data (use memory mapping for images)
        self.images = np.load(images_path, mmap_mode='r')  # Shape: (N, 3, 256, 192)
        self.poses_data = np.load(poses_path)

        # Extract data from npz
        self.poses_normalized = self.poses_data['poses_normalized']  # (N, 13, 2)
        self.poses_pixel = self.poses_data['poses_pixel']  # (N, 13, 2)
        self.valid_keypoints = self.poses_data['valid_keypoints']  # (N, 13)
        self.image_paths = self.poses_data['image_paths']  # (N,)
        self.target_size = self.poses_data['target_size']  # (2,)
        self.rotation_angle = self.poses_data['rotation_angle']  # scalar

        assert len(self.images) == len(self.poses_normalized), \
            f"Mismatch between images ({len(self.images)}) and poses ({len(self.poses_normalized)})"

        print(f"Loaded {len(self.images)} preprocessed tiger pose samples for {split} split from {preprocessed_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image (copy from memory-mapped array)
        image = np.array(self.images[idx])  # (3, 256, 192)

        # Get pose data
        pose_normalized = self.poses_normalized[idx]  # (13, 2)

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
                'image_path': str(self.image_paths[idx]),
                'pose_normalized': self.poses_normalized[idx].tolist(),
                'pose_pixel': self.poses_pixel[idx].tolist(),
                'valid_keypoints': self.valid_keypoints[idx].tolist(),
                'target_size': tuple(self.target_size.tolist()),
                'rotation_angle': int(self.rotation_angle)
            }
            return image, pose_flat, metadata
        else:
            return image, pose_flat

    def get_sample_info(self, idx):
        """Get metadata for a specific sample without loading the image"""
        return {
            'image_path': str(self.image_paths[idx]),
            'pose_normalized': self.poses_normalized[idx].tolist(),
            'pose_pixel': self.poses_pixel[idx].tolist(),
            'valid_keypoints': self.valid_keypoints[idx].tolist(),
            'target_size': tuple(self.target_size.tolist()),
            'rotation_angle': int(self.rotation_angle)
        }


class TigerPosePreprocessedDatasetJAX:
    """
    JAX-compatible wrapper for TigerPosePreprocessedDataset

    This class provides a simpler interface for JAX training loops
    that expect data as numpy/JAX arrays rather than PyTorch tensors.
    """

    def __init__(self, preprocessed_dir, split='train', return_metadata=False):
        self.dataset = TigerPosePreprocessedDataset(
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


def get_tiger_pose_preprocessed(
    preprocessed_dir,
    batch_size=128,
    shuffle=False,
    seed=0,
    split_train_val_ratio=0.9,
    return_metadata=False,
    n_samples=None
):
    """
    Get data loaders for preprocessed Tiger Pose dataset

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
    train_dataset = TigerPosePreprocessedDataset(
        preprocessed_dir=preprocessed_dir,
        split='train',
        return_metadata=return_metadata,
        jax_format=False
    )

    test_dataset = TigerPosePreprocessedDataset(
        preprocessed_dir=preprocessed_dir,
        split='val',
        return_metadata=return_metadata,
        jax_format=False
    )

    # Subsample if n_samples is specified
    if n_samples is not None:
        import torch.utils.data
        indices = list(range(min(n_samples, len(train_dataset))))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

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


def verify_tiger_pose_preprocessed_dataset(preprocessed_dir, split='train', num_samples=5):
    """
    Verify preprocessed Tiger Pose dataset by loading and displaying sample information

    Args:
        preprocessed_dir: Path to preprocessed dataset directory
        split: Which split to verify
        num_samples: Number of samples to check
    """
    print(f"\nVerifying preprocessed Tiger Pose dataset: {preprocessed_dir}")
    print(f"Split: {split}")
    print(f"{'='*80}\n")

    dataset = TigerPosePreprocessedDataset(
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
        print(f"  Image path: {metadata['image_path']}")
        print(f"  Target size: {metadata['target_size']}")
        print(f"  Rotation angle: {metadata['rotation_angle']}°")
        print(f"  Valid keypoints: {sum(metadata['valid_keypoints'])}/13")
        print(f"  Image value range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Pose normalized range: [{np.min(metadata['pose_normalized']):.3f}, {np.max(metadata['pose_normalized']):.3f}]")

    print(f"\n{'='*80}")
    print("Verification complete!")


if __name__ == "__main__":
    # Test the dataset
    import matplotlib.pyplot as plt

    data_path = "datasets"
    train_loader, val_loader, test_loader = get_tiger_pose(batch_size=4, data_path=data_path)

    print("Tiger Pose Dataset Test")
    print("=" * 40)

    # Get dataset statistics
    train_stats = train_loader.dataset.get_stats()
    val_stats = val_loader.dataset.get_stats()

    print(f"Train set stats: {train_stats}")
    print(f"Val set stats: {val_stats}")

    # Test a few samples
    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i}:")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Keypoints shape: {batch['keypoints'].shape}")
        print(f"  Valid keypoints shape: {batch['valid_keypoints'].shape}")
        print(f"  Paths: {batch['image_path']}")

        # Test conversion to H36M format
        h36m_kpts, h36m_valid = tiger_pose_to_h36m_format(
            batch['keypoints'].numpy(),
            batch['valid_keypoints'].numpy()
        )
        print(f"  H36M format shape: {h36m_kpts.shape}")

        if i >= 2:  # Test only first few batches
            break