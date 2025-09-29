#!/usr/bin/env python3
"""
2D Pose Estimation - JAX Implementation

This implements the JAX version of Marian's 2D_Pose_Estimation.py:
- Load H36M dataset with pose sequences and video frames
- Detect humans using YOLO
- Perform 2D pose estimation using JAX RegressFlow model
- Evaluate pose estimation accuracy using Mahalanobis distance
- Visualize results with ground truth and estimated poses

Based on marian_code/Experiment2/2D_Pose_Estimation.py but adapted for JAX.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from spacepy import pycdf
from spacepy.pycdf import CDF
from collections import defaultdict
import jax
import jax.numpy as jnp
import json
import pickle
import cv2
from PIL import Image
from torchvision import transforms
from scipy.stats import chi2

# Add root directory to path to access src
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(root_dir)

from src.models.wrapper import model_from_string
from src.datasets.h36m import Human36mDataset
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    pose_estimation_2d
)
from human_pose_pipeline.utils.transform_utils import (
    preprocess_image_with_bbox,
    convert_coordinates_regressflow_to_pixel,
    transform_coordinates_back_to_original
)
from human_pose_pipeline.evaluation.pose_metrics import (
    mpjpe_jax,
    JOINT_NAMES_13,
    JOINT_IDX_13_MODEL
)

# Define the 17 joints we want to keep from the original data (same as Marian's)
JOINT_IDX_17 = [0, 1, 2, 3, 6, 7, 8, 12, 16, 14, 15, 17, 18, 19, 25, 26, 27]

# Define the mapping from 17 joints to 13 joints (same as Marian's)
JOINT_IDX_13 = [10, 14, 11, 15, 12, 16, 13, 1, 4, 2, 5, 3, 6]

# Corrected mapping for the model to align left and right joints with ground truth (same as Marian's)
JOINT_IDX_13_MODEL = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]

# Define the connections for the 13-joint representation (same as Marian's)
CONNECTIONS_13 = [
    (0, 1), (0, 2),  # Nose to shoulders
    (1, 3), (3, 5),  # Left arm
    (2, 4), (4, 6),  # Right arm
    (1, 2), (1, 7), (2, 8),  # Shoulders to hips
    (7, 8),  # Connect hips
    (7, 9), (9, 11),  # Left leg
    (8, 10), (10, 12)  # Right leg
]

# Dataset splits (same as Marian's)
SPLIT = {
    'train': ['S1'],
    'validation': ['S11'],
    'test': ['S5']
}

class Human36mDatasetJAX:
    """
    Dataset class for loading Human3.6M data for pose estimation (JAX version).

    Handles loading of pose sequences and corresponding video frames from the Human3.6M dataset.
    Supports splitting data into train/validation/test sets and sequence-based sampling.
    """
    def __init__(self, base_directory, split='train', sequence_length=50, transform=None):
        self.sequence_length = sequence_length
        self.transform = transform if transform else transforms.ToTensor()
        self.data = self.load_data(base_directory, split)
        self.base_directory = base_directory
        self.split = split

    def load_data(self, base_directory, split):
        all_data = []
        for subject in SPLIT[split]:
            poses_dir = os.path.join(base_directory, subject, 'Poses_D2_Positions')
            videos_dir = os.path.join(base_directory, subject, 'Videos')
            print(f"Loading data from {poses_dir} and {videos_dir}")

            for filename in os.listdir(poses_dir):
                try:
                    if filename.endswith('.cdf'):
                        file_path = os.path.join(poses_dir, filename)
                        video_filename = self.get_corresponding_video_filename(filename, videos_dir)
                        if not video_filename:
                            print(f"No corresponding video found for {filename}")
                            continue
                        video_path = os.path.join(videos_dir, video_filename)

                        print(file_path)

                        with CDF(file_path) as cdf:
                            poses = cdf['Pose'][:]
                            poses = poses.reshape(-1, 32, 2)  # (frames, 32 joints, 2 coords)
                            poses_17 = poses[:, JOINT_IDX_17, :]
                            poses_13 = poses_17[:, JOINT_IDX_13, :]

                            # Create non-overlapping sequences
                            num_sequences = len(poses_13) // self.sequence_length
                            for i in range(num_sequences):
                                start_idx = i * self.sequence_length
                                end_idx = start_idx + self.sequence_length
                                sequence = poses_13[start_idx:end_idx]
                                frame_indices = range(start_idx, end_idx)
                                all_data.append({
                                    'pose_sequence': sequence,
                                    'video_path': video_path,
                                    'frame_indices': frame_indices,
                                })
                except Exception as e:
                    print(f"Error loading data: {str(e)}")

        print(f"Loaded {len(all_data)} sequences for {split} split")
        return all_data

    def get_corresponding_video_filename(self, pose_filename, videos_dir):
        base = os.path.splitext(pose_filename)[0]
        possible_video_names = [f"{base}.mp4", f"_{base}.mp4"]
        for video_name in possible_video_names:
            if os.path.exists(os.path.join(videos_dir, video_name)):
                return video_name
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        pose_sequence = sample['pose_sequence']
        video_path = sample['video_path']
        frame_indices = sample['frame_indices']

        # Load the necessary frames from the video
        frames = self.load_frames(video_path, frame_indices)

        return {
            'pose_sequence': jnp.array(pose_sequence, dtype=jnp.float32),
            'frames': frames  # List of PIL images
        }

    def load_frames(self, video_path, frame_indices):
        cap = cv2.VideoCapture(video_path)
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame)
                frames.append(frame_pil)
            else:
                print(f"Failed to read frame {frame_idx}")
                # Add dummy frame to maintain sequence length
                dummy_frame = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
                frames.append(dummy_frame)
        cap.release()
        return frames

def visualize_pose_sequence_with_images(pose_sequence, images, output_file, num_frames=None,
                                       estimated_poses=None, estimated_uncertainties=None,
                                       estimated_covariances=None):
    """
    Create an animated visualization of pose sequences overlaid on image frames.

    JAX version of Marian's visualization function.
    """
    if num_frames is None:
        num_frames = pose_sequence.shape[0]

    # Initialize a list to store individual frames for the GIF
    frames_for_gif = []

    for frame in range(num_frames):
        # Retrieve the image frame
        image_pil = images[frame]
        image = np.array(image_pil).copy()

        # Overlay the ground truth pose on the image
        gt_pose = pose_sequence[frame]

        # Ensure pose coordinates are within image bounds
        image_height, image_width, _ = image.shape
        gt_pose = np.clip(gt_pose, 0, [image_width - 1, image_height - 1])

        # Draw ground truth connections
        for connection in CONNECTIONS_13:
            start_idx, end_idx = connection
            start_point = tuple(gt_pose[start_idx].astype(int))
            end_point = tuple(gt_pose[end_idx].astype(int))
            cv2.line(image, start_point, end_point, color=(255, 0, 0), thickness=2)  # Red lines for GT

        # Draw ground truth keypoints
        for idx, (x, y) in enumerate(gt_pose):
            cv2.circle(image, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)  # Green dots for GT
            cv2.putText(image, f"GT {idx}", (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 0, 255), 1, cv2.LINE_AA)  # Red indices for GT

        # Overlay the estimated pose if provided
        if estimated_poses is not None and frame < len(estimated_poses):
            est_pose = estimated_poses[frame]
            est_pose = np.array(est_pose)
            est_uncertainty = estimated_uncertainties[frame] if estimated_uncertainties else None
            est_covariance = estimated_covariances[frame] if estimated_covariances else None

            # Draw estimated connections
            for connection in CONNECTIONS_13:
                start_idx, end_idx = connection
                start_point = tuple(est_pose[start_idx].astype(int))
                end_point = tuple(est_pose[end_idx].astype(int))
                cv2.line(image, start_point, end_point, color=(0, 0, 255), thickness=2)  # Blue lines for Estimation

            # Draw estimated keypoints
            for idx, (x, y) in enumerate(est_pose):
                cv2.circle(image, (int(x), int(y)), radius=3, color=(255, 0, 0), thickness=-1)  # Blue dots for Estimation
                cv2.putText(image, f"Est {idx}", (int(x)+5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (255, 0, 0), 1, cv2.LINE_AA)  # Blue indices for Estimation

                # Draw uncertainty ellipses if available
                if est_uncertainty is not None and est_covariance is not None:
                    std_x, std_y = est_uncertainty[idx]
                    cov_xy = est_covariance[idx]

                    if std_x > 0 and std_y > 0:
                        # Calculate the angle of the ellipse
                        angle = 0.5 * np.arctan2(2 * cov_xy, (std_x**2 - std_y**2)) * (180 / np.pi)

                        # Calculate the width and height of the ellipse based on standard deviations
                        width = int(2 * std_x)  # 2 standard deviations
                        height = int(2 * std_y)

                        # Draw the uncertainty ellipse
                        if width > 0 and height > 0:
                            cv2.ellipse(image, (int(x), int(y)), (width, height), angle, 0, 360, (0, 0, 255), 1)

        # Convert back to PIL Image for consistency
        image_with_pose = Image.fromarray(image)

        # Append to frames list for GIF creation
        frames_for_gif.append(image_with_pose)

    # Create an animated GIF with the overlaid poses
    if frames_for_gif:
        frames_for_gif[0].save(
            output_file,
            save_all=True,
            append_images=frames_for_gif[1:],
            duration=10,  # Duration between frames in milliseconds
            loop=0
        )
        print(f"Visualization with overlaid poses saved as {output_file}")

def map_17_to_13_joints(pose_17, mapping):
    """Convert a 17-joint pose representation to a 13-joint representation using a specified mapping."""
    return pose_17[mapping]

def evaluate_pose_estimation_full(ground_truth, estimated_pose, estimated_uncertainty, estimated_covariance):
    """
    Evaluate pose estimation accuracy using Mahalanobis distance and confidence intervals.
    JAX version of Marian's evaluation function.
    """
    # Calculate the difference between ground truth and estimated pose
    delta = ground_truth - estimated_pose  # Shape: (num_joints, 2)

    # Extract uncertainties and covariance
    std_x = estimated_uncertainty[:, 0]  # Shape: (num_joints,)
    std_y = estimated_uncertainty[:, 1]  # Shape: (num_joints,)
    cov_xy = estimated_covariance  # Shape: (num_joints,)

    # Compute the determinant of the covariance matrix
    det_sigma = (std_x ** 2) * (std_y ** 2) - (cov_xy ** 2)  # Shape: (num_joints,)

    # Add a small epsilon to determinant for numerical stability
    epsilon = 1e-6
    det_sigma += epsilon

    # Compute the inverse of the covariance matrix
    inv_sigma_xx = (std_y ** 2) / det_sigma  # Shape: (num_joints,)
    inv_sigma_yy = (std_x ** 2) / det_sigma  # Shape: (num_joints,)
    inv_sigma_xy = (-cov_xy) / det_sigma    # Shape: (num_joints,)

    # Compute the Mahalanobis distance for each joint
    mahalanobis = (inv_sigma_xx * (delta[:, 0] ** 2) +
                   inv_sigma_yy * (delta[:, 1] ** 2) +
                   2 * inv_sigma_xy * (delta[:, 0] * delta[:, 1]))  # Shape: (num_joints,)

    # Define chi-squared thresholds for 2 degrees of freedom
    thresholds = [chi2.ppf(0.68, df=2),   # 1 std
                  chi2.ppf(0.95, df=2),   # 2 std
                  chi2.ppf(0.9973, df=2), # 3 std
                  chi2.ppf(0.99994, df=2)]# 4 std

    # Determine which keypoints fall within each threshold
    within_std = [mahalanobis <= threshold for threshold in thresholds]

    # Count the number of keypoints within each threshold
    counts = {f'within_{i+1}std': np.sum(within) for i, within in enumerate(within_std)}

    # Prepare detailed results per joint
    joint_results = []
    for i, dist in enumerate(mahalanobis):
        joint_result = {
            'joint_index': i,
            'mahalanobis_distance': dist,
            'within_1std': dist <= thresholds[0],
            'within_2std': dist <= thresholds[1],
            'within_3std': dist <= thresholds[2],
            'within_4std': dist <= thresholds[3]
        }
        joint_results.append(joint_result)

    return {
        'counts': counts,
        'joint_results': joint_results,
        'num_joints': len(ground_truth)
    }


def main():
    """
    Main function for running pose estimation inference on the Human3.6M dataset.
    JAX version of Marian's main function.
    """
    base_directory = os.path.join(root_dir, "datasets", "H36M", "extracted")

    # Initialize models
    print("Initializing models...")

    # Initialize JAX pose estimation model
    models_dir = os.path.join(root_dir, "models_tianle", "H36M", "RegressFlow", "seed_420")
    checkpoint_path_jax = os.path.join(models_dir, "finetuned_h36m_regressflow_pred")
    model, params, batch_stats = initialize_jax_models(checkpoint_path_jax)

    # Initialize YOLO human detector
    human_detector, device_torch = initialize_human_detector('cuda')

    print("Models initialized successfully")

    # Create datasets and dataloaders for each split
    datasets = {}
    for split in ['train']:  # Reduced to just 'train' split
        datasets[split] = Human36mDatasetJAX(base_directory, split=split, sequence_length=500)
        print(f"{split.capitalize()} dataset size: {len(datasets[split])}")

    # Process each split
    for split in ['train']:
        dataset = datasets[split]
        print(f"\nProcessing {split} split...")

        for idx, sample in enumerate(dataset):
            full_sequence = np.array(sample['pose_sequence'])
            frames = sample['frames']

            print(f"\n{split.capitalize()} split:")
            print(f"Full sequence shape: {full_sequence.shape}")
            print(f"Number of frames: {len(frames)}")

            # Initialize statistics variables
            total_frames = 0
            total_joints = 0
            total_within_1std = 0
            total_within_2std = 0
            total_within_3std = 0
            total_within_4std = 0

            # Perform pose estimation on each frame
            estimated_poses = []
            estimated_uncertainties = []
            estimated_covariances = []

            for frame_idx in range(len(frames)):
                frame_image_pil = frames[frame_idx]

                # Get pose estimations using JAX model
                pose_estimations = pose_estimation_2d(
                    pil_image=frame_image_pil,
                    model=model,
                    params=params,
                    batch_stats=batch_stats,
                    human_detector=human_detector,
                    device_torch=device_torch,
                    threshold=0.8
                )

                if not pose_estimations:
                    estimated_poses.append(np.zeros((13, 2)))
                    estimated_uncertainties.append(np.ones((13, 2)) * 5.0)
                    estimated_covariances.append(np.ones(13) * 0.1)
                else:
                    first_pose = pose_estimations[0]['keypoints']
                    first_uncertainty = pose_estimations[0]['uncertainties']
                    first_covariance = pose_estimations[0]['covariance']

                    # Map from 17 joints to 13 joints
                    mapped_pose = map_17_to_13_joints(np.array(first_pose), JOINT_IDX_13_MODEL)
                    mapped_uncertainty = map_17_to_13_joints(np.array(first_uncertainty), JOINT_IDX_13_MODEL)
                    mapped_covariance = np.array(first_covariance)[JOINT_IDX_13_MODEL]

                    estimated_poses.append(mapped_pose)
                    estimated_uncertainties.append(mapped_uncertainty)
                    estimated_covariances.append(mapped_covariance)

                # Evaluate pose estimation
                ground_truth = full_sequence[frame_idx]
                estimated_pose = np.array(estimated_poses[frame_idx])
                estimated_uncertainty = np.array(estimated_uncertainties[frame_idx])
                estimated_covariance = np.array(estimated_covariances[frame_idx])

                evaluation = evaluate_pose_estimation_full(
                    ground_truth=ground_truth,
                    estimated_pose=estimated_pose,
                    estimated_uncertainty=estimated_uncertainty,
                    estimated_covariance=estimated_covariance
                )

                # Update counters
                total_frames += 1
                total_joints += evaluation['num_joints']
                total_within_1std += evaluation['counts']['within_1std']
                total_within_2std += evaluation['counts']['within_2std']
                total_within_3std += evaluation['counts']['within_3std']
                total_within_4std += evaluation['counts']['within_4std']

            # Print evaluation results
            if total_frames > 0:
                avg_within_1std = (total_within_1std / total_joints) * 100
                avg_within_2std = (total_within_2std / total_joints) * 100
                avg_within_3std = (total_within_3std / total_joints) * 100
                avg_within_4std = (total_within_4std / total_joints) * 100

                print(f"\nOverall Evaluation Results:")
                print(f"Total frames processed: {total_frames}")
                print(f"Total joints evaluated: {total_joints}")
                print(f"Average percentage of keypoints within 1 std: {avg_within_1std:.2f}%")
                print(f"Average percentage of keypoints within 2 std: {avg_within_2std:.2f}%")
                print(f"Average percentage of keypoints within 3 std: {avg_within_3std:.2f}%")
                print(f"Average percentage of keypoints within 4 std: {avg_within_4std:.2f}%")

            # Visualize the results
            output_file = f"sample_pose_sequence_with_images_{split}_{idx}.gif"
            visualize_pose_sequence_with_images(
                pose_sequence=full_sequence,
                images=frames,
                output_file=output_file,
                num_frames=len(frames),
                estimated_poses=estimated_poses,
                estimated_uncertainties=estimated_uncertainties,
                estimated_covariances=estimated_covariances
            )
            print(f"Visualization saved as {output_file}")

            # Break after first sample
            if idx == 0:
                break

if __name__ == "__main__":
    main()