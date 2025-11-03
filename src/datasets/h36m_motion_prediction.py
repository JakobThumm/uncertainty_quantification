"""Dataset for the motion prediction task on the Human3.6M dataset."""

import os
from torch.utils.data import Dataset
from spacepy import pycdf
import torch

from human_pose_pipeline.pose_estimation.h36m_settings import JOINT_IDX_17, JOINT_IDX_13

# Dataset splits matching original H36M
SPLIT = {"train": ["S1", "S6", "S7", "S8"], "validation": ["S9"], "test": ["S11"]}


class Human36mMotionDataset3D(Dataset):
    """Dataset class for Human3.6M motion data."""

    def __init__(self, base_directory, split="train", input_frames=50, predict_frames=10):
        self.input_frames = input_frames
        self.predict_frames = predict_frames
        self.data = []
        self.data = self.load_data(base_directory, split)

    def load_data(self, base_directory, split):
        all_data = []
        for subject in SPLIT[split]:
            directory = os.path.join(base_directory, subject, "D3_Positions")
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
        return {
            "input_pose": torch.FloatTensor(input_pose),
            "target_pose": torch.FloatTensor(target_pose),
        }
