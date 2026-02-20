"""
Dataset class for loading RGBD data for human pose estimation.

This dataset loads color (RGB) and depth images from a directory structure:
    base_directory/
        color/
            <timestamp>.png
        depth/
            <timestamp>.png

The dataset is designed to work with the pose estimation pipeline in
human_pose_pipeline/pose_estimation/inference_helper_batched.py
"""

import os
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

from src.datasets.utils import get_loader


# Default camera intrinsics for RealSense D435i (640x480)
# These should be overridden with actual calibration data
DEFAULT_CAMERA_INTRINSICS = {
    'fx': 383.12,  # focal length x in pixels
    'fy': 383.12,  # focal length y in pixels
    'cx': 319.5,   # principal point x
    'cy': 239.5,   # principal point y
}

# Default camera placement: 0.8 m above ground, pitched 10° downward
DEFAULT_CAMERA_POSITION = np.array([0.0, -2.0, 0.7], dtype=np.float32)
DEFAULT_CAMERA_RPY_DEG = np.array([10.0, 0.0, 0.0], dtype=np.float32)
CAMERA_TO_WORLD_TRANSFORM = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, -1.0, 0.0]], dtype=np.float32
)


class HumanRGBDDataset(Dataset):
    """
    Dataset class for loading RGBD data for human pose estimation.

    Loads paired color and depth images from separate directories.
    Only returns samples where both color and depth images exist.

    Args:
        base_directory: Path to the dataset directory containing 'color' and 'depth' subdirs
        transform: Optional transform to apply to color images
        depth_scale: Scale factor to convert depth values to meters (default 0.001 for mm)
        camera_intrinsics: Dict with camera intrinsic parameters (fx, fy, cx, cy)
        camera_position: Camera position in world frame [x, y, z] in metres.
            Defaults to [0, 0, 0.8] (camera mounted 0.8 m above ground).
        camera_rpy_deg: Camera orientation as (roll, pitch, yaw) in degrees using
            intrinsic XYZ convention.  Defaults to [0, -10, 0] (pitched 10° down).
    """

    def __init__(
        self,
        base_directory: str,
        transform=None,
        depth_scale: float = 0.001,
        camera_intrinsics: dict = None,
        camera_position: np.ndarray = None,
        camera_rpy_deg: np.ndarray = None,
    ):
        self.base_directory = base_directory
        self.color_dir = os.path.join(base_directory, 'color')
        self.depth_dir = os.path.join(base_directory, 'depth')
        self.depth_scale = depth_scale
        self.camera_intrinsics = camera_intrinsics or DEFAULT_CAMERA_INTRINSICS

        # Compute rotation matrix (camera → world) from RPY angles
        rpy = camera_rpy_deg if camera_rpy_deg is not None else DEFAULT_CAMERA_RPY_DEG
        pos = camera_position if camera_position is not None else DEFAULT_CAMERA_POSITION
        rotation = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix().astype(np.float32)
        self.R_rect_to_world = CAMERA_TO_WORLD_TRANSFORM @ rotation
        self.t_rect_to_world = np.asarray(pos, dtype=np.float32)

        # Default transform: convert to tensor and normalize
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # Load paired samples (only files that exist in both color and depth)
        self.samples = self._load_paired_samples()
        print(f"Loaded {len(self.samples)} paired RGBD samples from {base_directory}")

    def _load_paired_samples(self):
        """Find all color-depth pairs that exist in both directories."""
        if not os.path.exists(self.color_dir):
            raise ValueError(f"Color directory not found: {self.color_dir}")
        if not os.path.exists(self.depth_dir):
            raise ValueError(f"Depth directory not found: {self.depth_dir}")

        # Get all color and depth filenames
        color_files = set(os.listdir(self.color_dir))
        depth_files = set(os.listdir(self.depth_dir))

        # Find common filenames (paired samples)
        paired_files = sorted(color_files & depth_files)

        # Filter to only include PNG files
        paired_files = [f for f in paired_files if f.endswith('.png')]

        samples = []
        for filename in paired_files:
            # Extract timestamp from filename (e.g., "1770838413514059326.png" -> 1770838413514059326)
            timestamp_str = os.path.splitext(filename)[0]
            try:
                timestamp = int(timestamp_str)
            except ValueError:
                timestamp = 0  # If filename is not a timestamp, use 0

            samples.append({
                'color_path': os.path.join(self.color_dir, filename),
                'depth_path': os.path.join(self.depth_dir, filename),
                'filename': filename,
                'timestamp': timestamp
            })

        # Sort by timestamp
        samples.sort(key=lambda x: x['timestamp'])

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single RGBD sample.

        Returns:
            dict containing:
                - 'color': RGB image as tensor [C, H, W]
                - 'depth': Depth image as tensor [H, W] in meters
                - 'color_raw': Raw RGB image as numpy array [H, W, C] (uint8)
                - 'depth_raw': Raw depth image as numpy array [H, W] (uint16 or float)
                - 'filename': Original filename
                - 'timestamp': Timestamp extracted from filename
                - 'camera_intrinsics': Camera intrinsic parameters
        """
        sample = self.samples[idx]

        # Load color image
        color_img = Image.open(sample['color_path']).convert('RGB')
        color_raw = np.array(color_img)

        # Load depth image (typically uint16 in millimeters)
        depth_raw = cv2.imread(sample['depth_path'], cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise RuntimeError(f"Failed to load depth image: {sample['depth_path']}")

        # Convert depth to float meters
        depth_meters = depth_raw.astype(np.float32) * self.depth_scale

        # Apply transform to color image
        color_tensor = self.transform(color_img)

        # Convert depth to tensor
        depth_tensor = torch.from_numpy(depth_meters)

        return {
            'color': color_tensor,
            'depth': depth_tensor,
            'color_raw': color_raw,
            'depth_raw': depth_raw,
            'filename': sample['filename'],
            'timestamp': sample['timestamp'],
            'camera_intrinsics': self.camera_intrinsics,
            'R_rect_to_world': self.R_rect_to_world,
            't_rect_to_world': self.t_rect_to_world,
        }


class HumanRGBDDatasetSequence(Dataset):
    """
    Dataset class for loading sequences of RGBD data.

    Similar to HumanRGBDDataset but returns sequences of consecutive frames
    for temporal processing or motion prediction.

    Args:
        base_directory: Path to the dataset directory
        sequence_length: Number of frames per sequence
        stride: Number of frames to skip between sequences (default=sequence_length for non-overlapping)
        transform: Optional transform to apply to color images
        depth_scale: Scale factor to convert depth values to meters
        camera_intrinsics: Dict with camera intrinsic parameters
    """

    def __init__(
        self,
        base_directory: str,
        sequence_length: int = 10,
        stride: int = None,
        transform=None,
        depth_scale: float = 0.001,
        camera_intrinsics: dict = None
    ):
        self.base_directory = base_directory
        self.sequence_length = sequence_length
        self.stride = stride if stride is not None else sequence_length
        self.depth_scale = depth_scale
        self.camera_intrinsics = camera_intrinsics or DEFAULT_CAMERA_INTRINSICS

        # Create base dataset to get all samples
        self._base_dataset = HumanRGBDDataset(
            base_directory=base_directory,
            transform=transform,
            depth_scale=depth_scale,
            camera_intrinsics=camera_intrinsics
        )

        # Calculate number of sequences
        self.num_sequences = max(0, (len(self._base_dataset) - sequence_length) // self.stride + 1)
        print(f"Created {self.num_sequences} sequences of length {sequence_length}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Get a sequence of RGBD samples.

        Returns:
            dict containing:
                - 'colors': RGB images as tensor [T, C, H, W]
                - 'depths': Depth images as tensor [T, H, W] in meters
                - 'colors_raw': Raw RGB images as numpy array [T, H, W, C]
                - 'depths_raw': Raw depth images as numpy array [T, H, W]
                - 'filenames': List of filenames
                - 'timestamps': List of timestamps
                - 'camera_intrinsics': Camera intrinsic parameters
        """
        start_idx = idx * self.stride

        colors = []
        depths = []
        colors_raw = []
        depths_raw = []
        filenames = []
        timestamps = []

        for i in range(self.sequence_length):
            sample = self._base_dataset[start_idx + i]
            colors.append(sample['color'])
            depths.append(sample['depth'])
            colors_raw.append(sample['color_raw'])
            depths_raw.append(sample['depth_raw'])
            filenames.append(sample['filename'])
            timestamps.append(sample['timestamp'])

        return {
            'colors': torch.stack(colors),
            'depths': torch.stack(depths),
            'colors_raw': np.stack(colors_raw),
            'depths_raw': np.stack(depths_raw),
            'filenames': filenames,
            'timestamps': timestamps,
            'camera_intrinsics': self.camera_intrinsics
        }


def get_human_rgbd(
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 0,
    data_path: str = "../datasets/rgbd_test",
    camera_intrinsics: dict = None,
    depth_scale: float = 0.001,
    num_workers: int = 0
):
    """
    Get DataLoaders for the HumanRGBD dataset.

    Since this is a test dataset without train/val/test splits,
    returns the same loader for all three (for API compatibility with other datasets).

    Args:
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        data_path: Path to the dataset directory
        camera_intrinsics: Dict with camera intrinsic parameters
        depth_scale: Scale factor to convert depth values to meters
        num_workers: Number of worker processes for data loading

    Returns:
        tuple: (train_loader, valid_loader, test_loader) - all pointing to the same data
    """
    torch.manual_seed(seed)

    dataset = HumanRGBDDataset(
        base_directory=data_path,
        depth_scale=depth_scale,
        camera_intrinsics=camera_intrinsics
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )

    # Return same loader for all splits (this is a test dataset)
    return loader, loader, loader


def get_human_rgbd_sequence(
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 0,
    data_path: str = "../datasets/rgbd_test",
    sequence_length: int = 10,
    stride: int = None,
    camera_intrinsics: dict = None,
    depth_scale: float = 0.001,
    num_workers: int = 0
):
    """
    Get DataLoaders for the HumanRGBDDatasetSequence.

    Args:
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        data_path: Path to the dataset directory
        sequence_length: Number of frames per sequence
        stride: Number of frames to skip between sequences
        camera_intrinsics: Dict with camera intrinsic parameters
        depth_scale: Scale factor to convert depth values to meters
        num_workers: Number of worker processes for data loading

    Returns:
        tuple: (train_loader, valid_loader, test_loader) - all pointing to the same data
    """
    torch.manual_seed(seed)

    dataset = HumanRGBDDatasetSequence(
        base_directory=data_path,
        sequence_length=sequence_length,
        stride=stride,
        depth_scale=depth_scale,
        camera_intrinsics=camera_intrinsics
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )

    # Return same loader for all splits (this is a test dataset)
    return loader, loader, loader
