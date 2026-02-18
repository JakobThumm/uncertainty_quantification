import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from spacepy import pycdf
from spacepy.pycdf import CDF
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
from scipy.stats import chi2
import jax.numpy as jnp
# Define the 17 joints we want to keep from the original data
JOINT_IDX_17 = [0, 1, 2, 3, 6, 7, 8, 12, 16, 14, 15, 17, 18, 19, 25, 26, 27]

# Define the mapping from 17 joints to 13 joints
JOINT_IDX_13 = [10, 14, 11, 15, 12, 16, 13, 1, 4, 2, 5, 3, 6]#9

# Define the mapping from 17 joints to 13 joints
JOINT_IDX_13_MODEL = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] # when mapping coco to 13 joint representation

# Corrected mapping for the model to align left and right joints with ground truth
# JOINT_IDX_13_MODEL = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]


from src.datasets.utils import get_loader, get_subset_data, RotationTransform

# Define the connections for the 13-joint representation
CONNECTIONS_13 = [
    (0, 1), (0, 2),  # Nose to shoulders
    (1, 3), (3, 5),  # Left arm
    (2, 4), (4, 6),  # Right arm
    (1, 2), (1, 7), (2, 8),  # Shoulders to hips
    (7, 8),  # Connect hips
    (7, 9), (9, 11),  # Left leg
    (8, 10), (10, 12)  # Right leg
]

SPLIT = {"train": ["S1", "S6", "S7", "S8", "S9"], "validation": ["S11"], "test": ["S5"]}

# Stereo pairs: front (55011271 ↔ 60457274) and back (54138969 ↔ 58860488)
CAMERA_PAIRS = {
    '55011271': '60457274',
    '60457274': '55011271',
    '54138969': '58860488',
    '58860488': '54138969',
}

# Native H36M image resolution (width, height) — matches cx/cy ≈ 508-519 / 501-515
H36M_IMAGE_SIZE = (1000, 1000)

_DEFAULT_CAMERA_PARAMS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420/camera-parameters.json',
)

transform = transforms.Compose([
    transforms.Resize((256, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


# Map string names to actual model attributes
def get_model_layer(model, layer_name):
    """Helper function to extract submodules dynamically."""
    parts = layer_name.split(".")
    sub_model = model
    for part in parts:
        if "[" in part and "]" in part:  # Handling list indices like layer4[2]
            name, idx = part.split("[")
            idx = int(idx[:-1])  # Convert '2]' -> 2
            sub_model = getattr(sub_model, name)[idx]
        else:
            sub_model = getattr(sub_model, part)
    return sub_model

def extract_frames(video_path, output_dir, start_frame=0, end_frame=None):
    """
    Extract frames from a video file and save them as individual image files.

    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory where extracted frames will be saved
        start_frame (int, optional): First frame to extract. Defaults to 0
        end_frame (int, optional): Last frame to extract. Defaults to None (extract all frames)

    Returns:
        None: Frames are saved directly to the output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_num = start_frame

    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_num += 1

    cap.release()
    print(f"Extracted frames {start_frame} to {end_frame} from {video_path} to {output_dir}")


class Human36mDataset(Dataset):
    """
    Dataset class for Human3.6M pose estimation.
    Returns cropped human image using bounding box around 13-joint keypoints,
    and adjusts keypoints to match the cropped image.
    """
    def __init__(self, base_directory, split='train', num_frames_per_video=5, transform=None, image_size=(256, 192)):
        self.num_frames_per_video = num_frames_per_video
        self.transform = transform if transform else transforms.ToTensor()
        self.data = self.load_data(base_directory, split)
        self.base_directory = base_directory
        self.split = split
        self.image_size = image_size

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

                        with CDF(file_path) as cdf:
                            poses = cdf['Pose'][:].reshape(-1, 32, 2)
                            poses_17 = poses[:, JOINT_IDX_17, :]
                            poses_13 = poses_17[:, JOINT_IDX_13, :]
                            total_frames = len(poses_13)

                            if total_frames < self.num_frames_per_video:
                                continue
                            indices = np.linspace(0, total_frames - 1, self.num_frames_per_video, dtype=int)

                            for idx in indices:
                                all_data.append({
                                    'pose_13': poses_13[idx],
                                    'video_path': video_path,
                                    'frame_idx': idx
                                })
                except Exception as e:
                    print(f"Error loading data: {str(e)}")

        print(f"Loaded {len(all_data)} samples for {split} split")
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
        keypoints = sample['pose_13']
        video_path = sample['video_path']
        frame_idx = sample['frame_idx']

        frame, adjusted_keypoints = self.load_cropped_frame(video_path, frame_idx, keypoints)
        assert frame.shape == (3, 256, 192), f"Image shape mismatch: {frame.shape}"
        assert adjusted_keypoints.shape == (13, 2), f"Keypoints shape mismatch: {keypoints.shape}"
        adjusted_keypoints = adjusted_keypoints.reshape(26)
        pose_13 = torch.FloatTensor(adjusted_keypoints)
        # return {
        #     'pose_13': torch.FloatTensor(adjusted_keypoints),
        #     'frame': frame,
        #     'video_path': video_path
        # }
        return frame, pose_13

    def load_cropped_frame(self, video_path, frame_idx, keypoints):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to load frame {frame_idx} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Compute tight bounding box around keypoints
        min_x, min_y = np.min(keypoints, axis=0)
        max_x, max_y = np.max(keypoints, axis=0)
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        scale = max(max_x - min_x, max_y - min_y) * 1.25  # Add padding

        crop_size = int(scale)
        top = int(center_y - crop_size // 2)
        left = int(center_x - crop_size // 2)

        # Ensure bounding box stays within image bounds
        top = max(0, top)
        left = max(0, left)
        bottom = min(frame.shape[0], top + crop_size)
        right = min(frame.shape[1], left + crop_size)

        cropped = frame[top:bottom, left:right]
        image = Image.fromarray(cropped)
        image = self.transform(image)

        # Adjust keypoints to cropped and resized image
        adjusted_keypoints = keypoints.copy()
        adjusted_keypoints[:, 0] -= left
        adjusted_keypoints[:, 1] -= top

        # after
        crop_w = right - left
        crop_h = bottom - top
        target_h, target_w = self.image_size  # note the swap

        adjusted_keypoints[:, 0] *= (target_w / crop_w)
        adjusted_keypoints[:, 1] *= (target_h / crop_h)


        return image, adjusted_keypoints


class Human36mDatasetSequence:
    """
    Dataset class for loading Human3.6M data for pose estimation (JAX version).

    Handles loading of pose sequences and corresponding video frames from the Human3.6M dataset.
    Supports splitting data into train/validation/test sets and sequence-based sampling.
    """
    def __init__(self, base_directory, split='train', sequence_length=50, transform=None, max_files=None):
        self.sequence_length = sequence_length
        self.transform = transform if transform else transforms.ToTensor()
        self.max_files = max_files
        self.base_directory = base_directory
        self.split = split
        self.data = self.load_data(base_directory, split)

    def load_data(self, base_directory, split):
        all_data = []
        file_counter = 0
        for subject in SPLIT[split]:
            poses_dir = os.path.join(base_directory, subject, 'Poses_D2_Positions')
            videos_dir = os.path.join(base_directory, subject, 'Videos')
            print(f"Loading data from {poses_dir} and {videos_dir}")

            for filename in os.listdir(poses_dir):
                if self.max_files and file_counter >= self.max_files:
                    break
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
                        file_counter += 1
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


class Human36mDatasetSequenceTwoCameras:
    """
    Dataset class for loading Human3.6M data for pose estimation (JAX version).

    Handles loading of pose sequences and corresponding video frames from the Human3.6M dataset.
    Supports splitting data into train/validation/test sets and sequence-based sampling.
    """
    def __init__(self, base_directory, split='train', sequence_length=50, transform=None, camera_ids=['55011271', '60457274']):
        self.sequence_length = sequence_length
        self.camera_ids = camera_ids
        self.transform = transform if transform else transforms.ToTensor()
        self.data = self.load_data(base_directory, split, camera_ids)
        self.base_directory = base_directory
        self.split = split

    def load_data(self, base_directory, split, camera_ids):
        all_data = []
        for subject in SPLIT[split]:
            poses_dir = os.path.join(base_directory, subject, 'Poses_D3_Positions')
            videos_dir = os.path.join(base_directory, subject, 'Videos')
            print(f"Loading data from {poses_dir} and {videos_dir}")
            pose_files = [f for f in os.listdir(poses_dir) if f.endswith('.cdf')]

            for pose_file in pose_files:
                pose_path = os.path.join(poses_dir, pose_file)
                action = os.path.splitext(pose_file)[0]

                # Look for corresponding video files
                video_files = [f"{action}.{camera_id}.mp4" for camera_id in camera_ids]
                video_paths = [os.path.join(videos_dir, vf) for vf in video_files
                              if os.path.exists(os.path.join(videos_dir, vf))]
                with CDF(pose_path) as cdf:
                    poses = cdf['Pose'][:]
                poses = np.squeeze(poses)
                poses = poses.reshape(-1, 32, 3)  # (frames, 32 joints, 3 coords)
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
                        'video_paths': video_paths,
                        'frame_indices': frame_indices,
                    })
        print(f"Loaded {len(all_data)} sequences for {split} split")
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        pose_sequence = sample['pose_sequence']
        video_paths = sample['video_paths']
        frame_indices = sample['frame_indices']

        # Load the necessary frames from the video
        all_camera_frames = self.load_frames(video_paths, frame_indices)

        return {
            'pose_sequence': jnp.array(pose_sequence),
            'all_camera_frames': all_camera_frames
        }

    def load_frames(self, video_paths, frame_indices):
        all_camera_frames = []
        for video_path in video_paths:
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
            all_camera_frames.append(frames)
        return all_camera_frames


class Human36mDatasetTwoCameras:
    """
    Dataset class for loading Human3.6M data for 3D pose estimation with two cameras.

    Handles loading of pose sequences and corresponding video frames from two camera views
    for stereo triangulation.
    """
    def __init__(self, base_directory, split='train', camera_ids=['55011271', '60457274'], max_files=None):
        self.camera_ids = camera_ids
        self.max_files = max_files
        self.data = self.load_data(base_directory, split, camera_ids)
        self.base_directory = base_directory

    def load_data(self, base_directory, split, camera_ids):
        all_data = []
        for subject in SPLIT[split]:
            poses_dir = os.path.join(base_directory, subject, 'Poses_D3_Positions')
            videos_dir = os.path.join(base_directory, subject, 'Videos')
            print(f"Loading data from {poses_dir} and {videos_dir}")
            pose_files = [f for f in os.listdir(poses_dir) if f.endswith('.cdf')]

            for pose_file in pose_files:
                if self.max_files and len(all_data) >= self.max_files:
                    break
                pose_path = os.path.join(poses_dir, pose_file)
                action = os.path.splitext(pose_file)[0]

                # Look for corresponding video files
                video_files = [f"{action}.{camera_id}.mp4" for camera_id in camera_ids]
                video_paths = [os.path.join(videos_dir, vf) for vf in video_files
                               if os.path.exists(os.path.join(videos_dir, vf))]
                with CDF(pose_path) as cdf:
                    poses = cdf['Pose'][:]
                    poses = poses.reshape(-1, 32, 3)
                    poses_17 = poses[:, JOINT_IDX_17, :]
                    poses_13 = poses_17[:, JOINT_IDX_13, :]

                all_data.append({
                    'pose_sequence': poses_13,
                    'video_paths': video_paths,
                    'subject': subject,
                    'action': action
                })

        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        pose_sequence = sample['pose_sequence']
        video_paths = sample['video_paths']
        all_camera_frames = self.load_frames(video_paths)

        return {
            'pose_sequence': jnp.array(pose_sequence, dtype=jnp.float32),
            'all_camera_frames': all_camera_frames,
            'video_paths': video_paths,
            'subject': sample['subject'],
            'action': sample['action']
        }

    def load_frames(self, video_paths):
        all_camera_frames = []
        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            for frame_idx in range(total_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame)
                    frames.append(frame_pil)
                else:
                    break
            cap.release()
            all_camera_frames.append(frames)
        min_number_of_frames = min(len(frames) for frames in all_camera_frames)
        # Trim all camera frames to the minimum number of frames
        all_camera_frames = [frames[:min_number_of_frames] for frames in all_camera_frames]
        return all_camera_frames


class Human36mDatasetEmulatedRGBD(Dataset):
    """
    Emulated RGB-D dataset built from H36M stereo camera pairs.

    For each of the four cameras, depth is computed by stereo matching with its
    neighboring camera using the known H36M intrinsics and extrinsics:
      front pair: 55011271 ↔ 60457274
      back  pair: 54138969 ↔ 58860488

    Stereo rectification maps are computed once per (subject, primary_cam, pair_cam)
    and cached.  Depth values are in millimetres (same units as the H36M translation
    vectors), with 0 marking pixels where disparity could not be estimated.

    Returns:
        (rgb_tensor, depth_tensor) where
          rgb_tensor   – shape (3, H, W), normalised to [-1, 1]
          depth_tensor – shape (1, H, W), depth in mm
    """

    def __init__(
        self,
        base_directory,
        split='train',
        camera_ids=None,
        num_frames_per_video=5,
        transform=None,
        image_size=(256, 192),
        camera_params_path=None,
        sgbm_num_disparities=128,
        sgbm_block_size=11,
    ):
        self.base_directory = base_directory
        self.split = split
        self.camera_ids = camera_ids or list(CAMERA_PAIRS.keys())
        self.num_frames_per_video = num_frames_per_video
        self.image_size = image_size  # (H, W)
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

        self._load_camera_params(camera_params_path or _DEFAULT_CAMERA_PARAMS_PATH)
        self._stereo_cache = {}  # (subject, primary_cam, pair_cam) -> stereo params
        self._stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=sgbm_num_disparities,
            blockSize=sgbm_block_size,
            P1=8 * 3 * sgbm_block_size ** 2,
            P2=32 * 3 * sgbm_block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.data = self._load_data()
        print(f"Loaded {len(self.data)} emulated-RGBD samples for {split} split")

    def _load_camera_params(self, json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)

        self.intrinsics = {}
        self.distortions = {}
        self.extrinsics = {}  # subject -> cam_id -> {'R': ndarray, 't': ndarray (3,1)}

        for cam_key, intr in params['intrinsics'].items():
            cam_id = cam_key.lstrip('.')
            self.intrinsics[cam_id] = np.array(intr['calibration_matrix'])
            self.distortions[cam_id] = np.array(intr['distortion'])

        for subject, subject_data in params['extrinsics'].items():
            self.extrinsics[subject] = {}
            for cam_key, ext in subject_data.items():
                cam_id = cam_key.lstrip('.')
                self.extrinsics[subject][cam_id] = {
                    'R': np.array(ext['R']),
                    't': np.array(ext['t']).reshape(3, 1),
                }

    def _get_stereo_params(self, subject, primary_cam, pair_cam):
        """Compute (and cache) stereo rectification maps for one camera pair."""
        key = (subject, primary_cam, pair_cam)
        if key in self._stereo_cache:
            return self._stereo_cache[key]

        K1 = self.intrinsics[primary_cam]
        d1 = self.distortions[primary_cam]
        K2 = self.intrinsics[pair_cam]
        d2 = self.distortions[pair_cam]

        R1 = self.extrinsics[subject][primary_cam]['R']
        t1 = self.extrinsics[subject][primary_cam]['t']
        R2 = self.extrinsics[subject][pair_cam]['R']
        t2 = self.extrinsics[subject][pair_cam]['t']

        # Relative pose: X_cam2 = R_rel @ X_cam1 + t_rel
        # (H36M convention: X_cam = R @ X_world + t)
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1  # shape (3, 1)

        img_size = H36M_IMAGE_SIZE  # (width, height)
        R1_rect, R2_rect, P1_rect, P2_rect, _, _, _ = cv2.stereoRectify(
            K1, d1, K2, d2, img_size, R_rel, t_rel,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )

        map1x, map1y = cv2.initUndistortRectifyMap(
            K1, d1, R1_rect, P1_rect, img_size, cv2.CV_32FC1,
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            K2, d2, R2_rect, P2_rect, img_size, cv2.CV_32FC1,
        )

        # Baseline from rectified projection matrix: P2[0,3] = -fx * baseline
        baseline_mm = abs(P2_rect[0, 3] / P2_rect[0, 0])

        stereo_params = {
            'map1x': map1x, 'map1y': map1y,
            'map2x': map2x, 'map2y': map2y,
            'fx_rect': P1_rect[0, 0],
            'baseline_mm': baseline_mm,
            # Kept for GT coordinate transformation and intrinsics construction:
            'R1_rect': R1_rect,   # (3,3) rotation from original cam1 frame to rectified frame
            'P1_rect': P1_rect,   # (3,4) projection matrix of rectified primary camera
        }
        self._stereo_cache[key] = stereo_params
        return stereo_params

    def _compute_depth(self, frame_primary, frame_pair, stereo_params):
        """
        Rectify stereo pair and compute a depth map.

        Args:
            frame_primary: (H, W, 3) RGB uint8 — primary camera
            frame_pair:    (H, W, 3) RGB uint8 — pair camera

        Returns:
            rgb_rect: (H, W, 3) uint8 — rectified primary frame
            depth:    (H, W) float32 — depth in mm, 0 where invalid
        """
        rect1 = cv2.remap(frame_primary, stereo_params['map1x'], stereo_params['map1y'],
                          cv2.INTER_LINEAR)
        rect2 = cv2.remap(frame_pair, stereo_params['map2x'], stereo_params['map2y'],
                          cv2.INTER_LINEAR)

        gray1 = cv2.cvtColor(rect1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(rect2, cv2.COLOR_RGB2GRAY)

        disparity = self._stereo_matcher.compute(gray1, gray2).astype(np.float32) / 16.0

        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = stereo_params['fx_rect'] * stereo_params['baseline_mm'] / disparity[valid]

        return rect1, depth

    def _load_data(self):
        all_data = []
        for subject in SPLIT[self.split]:
            if subject not in self.extrinsics:
                print(f"Warning: no extrinsics for {subject}, skipping")
                continue
            poses_dir = os.path.join(self.base_directory, subject, 'Poses_D3_Positions')
            videos_dir = os.path.join(self.base_directory, subject, 'Videos')
            if not os.path.exists(poses_dir) or not os.path.exists(videos_dir):
                continue

            for pose_file in sorted(os.listdir(poses_dir)):
                if not pose_file.endswith('.cdf'):
                    continue
                action = os.path.splitext(pose_file)[0]

                # Load 3D GT poses once per action (world coordinates, mm).
                pose_path = os.path.join(poses_dir, pose_file)
                try:
                    with CDF(pose_path) as cdf:
                        poses_raw = np.squeeze(cdf['Pose'][:]).reshape(-1, 32, 3)
                    gt_poses_13 = poses_raw[:, JOINT_IDX_17, :][:, JOINT_IDX_13, :]
                except Exception:
                    gt_poses_13 = None

                for cam_id in self.camera_ids:
                    pair_cam = CAMERA_PAIRS[cam_id]
                    primary_video = os.path.join(videos_dir, f"{action}.{cam_id}.mp4")
                    pair_video = os.path.join(videos_dir, f"{action}.{pair_cam}.mp4")
                    if not os.path.exists(primary_video) or not os.path.exists(pair_video):
                        continue

                    cap = cv2.VideoCapture(primary_video)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    if total_frames < self.num_frames_per_video:
                        continue

                    for frame_idx in np.linspace(0, total_frames - 1,
                                                 self.num_frames_per_video, dtype=int):
                        frame_idx = int(frame_idx)
                        gt = (gt_poses_13[frame_idx]
                              if gt_poses_13 is not None and frame_idx < len(gt_poses_13)
                              else None)
                        all_data.append({
                            'subject': subject,
                            'action': action,
                            'primary_cam': cam_id,
                            'pair_cam': pair_cam,
                            'primary_video': primary_video,
                            'pair_video': pair_video,
                            'frame_idx': frame_idx,
                            'gt_pose_world_mm': gt,  # (13, 3) float or None
                        })
        return all_data

    def _load_frame(self, video_path, frame_idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to load frame {frame_idx} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a dict:
            'rgb_raw'   – (H_full, W_full, 3) uint8, rectified primary frame
            'depth_raw' – (H_full, W_full) float32, depth in mm
            'rgb'       – (3, H, W) normalised tensor
            'depth'     – (1, H, W) float32 tensor, depth in mm
            'gt_pose'   – (13, 3) float32 tensor, GT joints in rectified camera
                          frame in metres; zeros when GT is unavailable
            'subject'   – str
            'action'    – str
            'camera_id' – str (primary camera ID)
        """
        sample = self.data[idx]
        subject = sample['subject']
        primary_cam = sample['primary_cam']

        frame_primary = self._load_frame(sample['primary_video'], sample['frame_idx'])
        frame_pair = self._load_frame(sample['pair_video'], sample['frame_idx'])

        stereo_params = self._get_stereo_params(subject, primary_cam, sample['pair_cam'])
        rgb_rect, depth = self._compute_depth(frame_primary, frame_pair, stereo_params)

        target_h, target_w = self.image_size
        rgb_resized = cv2.resize(rgb_rect, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb_tensor = self.transform(Image.fromarray(rgb_resized))

        depth_resized = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        depth_tensor = torch.FloatTensor(depth_resized).unsqueeze(0)  # (1, H, W)

        # GT: world mm → rectified camera frame (m)
        gt_raw = sample['gt_pose_world_mm']  # (13, 3) or None
        if gt_raw is not None:
            R_cam = self.extrinsics[subject][primary_cam]['R']   # (3, 3)
            t_cam = self.extrinsics[subject][primary_cam]['t']   # (3, 1)
            R1_rect = stereo_params['R1_rect']                   # (3, 3)
            X_cam = (R_cam @ gt_raw.T + t_cam).T                # (13, 3) mm
            gt_pose = torch.FloatTensor((R1_rect @ X_cam.T).T / 1000.0)  # m
        else:
            gt_pose = torch.zeros(13, 3)

        return {
            'rgb_raw': rgb_rect,       # (H_full, W_full, 3) uint8
            'depth_raw': depth,        # (H_full, W_full) float32 mm
            'rgb': rgb_tensor,         # (3, H, W)
            'depth': depth_tensor,     # (1, H, W) mm
            'gt_pose': gt_pose,        # (13, 3) m, rectified camera frame
            'subject': subject,
            'action': sample['action'],
            'camera_id': primary_cam,
        }


class Human36mDatasetSequenceEmulatedRGBD:
    """
    Sequence version of Human36mDatasetEmulatedRGBD.

    Returns non-overlapping sequences of RGBD frames for each camera, where depth
    is computed via stereo matching with the paired camera (same stereo pairs as
    Human36mDatasetEmulatedRGBD).

    Note: each __getitem__ call runs StereoSGBM for every frame in the sequence,
    which is CPU-intensive.  Consider precomputing and caching depth maps for
    large-scale training.

    Returns a dict with:
        'rgb_sequence'   – (T, 3, H, W) float tensor, normalised to [-1, 1]
        'depth_sequence' – (T, 1, H, W) float tensor, depth in mm
        'subject'        – str, e.g. 'S1'
        'action'         – str, e.g. 'Walking'
        'camera_id'      – str, primary camera ID
    """

    def __init__(
        self,
        base_directory,
        split='train',
        camera_ids=None,
        sequence_length=50,
        transform=None,
        image_size=(256, 192),
        camera_params_path=None,
        sgbm_num_disparities=128,
        sgbm_block_size=11,
    ):
        self.base_directory = base_directory
        self.split = split
        self.camera_ids = camera_ids or list(CAMERA_PAIRS.keys())
        self.sequence_length = sequence_length
        self.image_size = image_size  # (H, W)
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ])

        self._load_camera_params(camera_params_path or _DEFAULT_CAMERA_PARAMS_PATH)
        self._stereo_cache = {}
        self._stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=sgbm_num_disparities,
            blockSize=sgbm_block_size,
            P1=8 * 3 * sgbm_block_size ** 2,
            P2=32 * 3 * sgbm_block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.data = self._load_data()
        print(f"Loaded {len(self.data)} emulated-RGBD sequences for {split} split")

    # ---- camera-parameter helpers (identical to the single-frame class) ----

    def _load_camera_params(self, json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)

        self.intrinsics = {}
        self.distortions = {}
        self.extrinsics = {}

        for cam_key, intr in params['intrinsics'].items():
            cam_id = cam_key.lstrip('.')
            self.intrinsics[cam_id] = np.array(intr['calibration_matrix'])
            self.distortions[cam_id] = np.array(intr['distortion'])

        for subject, subject_data in params['extrinsics'].items():
            self.extrinsics[subject] = {}
            for cam_key, ext in subject_data.items():
                cam_id = cam_key.lstrip('.')
                self.extrinsics[subject][cam_id] = {
                    'R': np.array(ext['R']),
                    't': np.array(ext['t']).reshape(3, 1),
                }

    def _get_stereo_params(self, subject, primary_cam, pair_cam):
        key = (subject, primary_cam, pair_cam)
        if key in self._stereo_cache:
            return self._stereo_cache[key]

        K1 = self.intrinsics[primary_cam]
        d1 = self.distortions[primary_cam]
        K2 = self.intrinsics[pair_cam]
        d2 = self.distortions[pair_cam]

        R1 = self.extrinsics[subject][primary_cam]['R']
        t1 = self.extrinsics[subject][primary_cam]['t']
        R2 = self.extrinsics[subject][pair_cam]['R']
        t2 = self.extrinsics[subject][pair_cam]['t']

        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1

        img_size = H36M_IMAGE_SIZE
        R1_rect, R2_rect, P1_rect, P2_rect, _, _, _ = cv2.stereoRectify(
            K1, d1, K2, d2, img_size, R_rel, t_rel,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )

        map1x, map1y = cv2.initUndistortRectifyMap(
            K1, d1, R1_rect, P1_rect, img_size, cv2.CV_32FC1,
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            K2, d2, R2_rect, P2_rect, img_size, cv2.CV_32FC1,
        )

        stereo_params = {
            'map1x': map1x, 'map1y': map1y,
            'map2x': map2x, 'map2y': map2y,
            'fx_rect': P1_rect[0, 0],
            'baseline_mm': abs(P2_rect[0, 3] / P2_rect[0, 0]),
            'R1_rect': R1_rect,
            'P1_rect': P1_rect,
        }
        self._stereo_cache[key] = stereo_params
        return stereo_params

    def _compute_depth(self, frame_primary, frame_pair, stereo_params):
        rect1 = cv2.remap(frame_primary, stereo_params['map1x'], stereo_params['map1y'],
                          cv2.INTER_LINEAR)
        rect2 = cv2.remap(frame_pair, stereo_params['map2x'], stereo_params['map2y'],
                          cv2.INTER_LINEAR)

        gray1 = cv2.cvtColor(rect1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(rect2, cv2.COLOR_RGB2GRAY)

        disparity = self._stereo_matcher.compute(gray1, gray2).astype(np.float32) / 16.0

        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = stereo_params['fx_rect'] * stereo_params['baseline_mm'] / disparity[valid]

        return rect1, depth

    # ---- data loading ----

    def _load_data(self):
        all_data = []
        for subject in SPLIT[self.split]:
            if subject not in self.extrinsics:
                print(f"Warning: no extrinsics for {subject}, skipping")
                continue
            poses_dir = os.path.join(self.base_directory, subject, 'Poses_D3_Positions')
            videos_dir = os.path.join(self.base_directory, subject, 'Videos')
            if not os.path.exists(poses_dir) or not os.path.exists(videos_dir):
                continue

            for pose_file in sorted(os.listdir(poses_dir)):
                if not pose_file.endswith('.cdf'):
                    continue
                action = os.path.splitext(pose_file)[0]

                for cam_id in self.camera_ids:
                    pair_cam = CAMERA_PAIRS[cam_id]
                    primary_video = os.path.join(videos_dir, f"{action}.{cam_id}.mp4")
                    pair_video = os.path.join(videos_dir, f"{action}.{pair_cam}.mp4")
                    if not os.path.exists(primary_video) or not os.path.exists(pair_video):
                        continue

                    cap = cv2.VideoCapture(primary_video)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    for i in range(total_frames // self.sequence_length):
                        all_data.append({
                            'subject': subject,
                            'action': action,
                            'primary_cam': cam_id,
                            'pair_cam': pair_cam,
                            'primary_video': primary_video,
                            'pair_video': pair_video,
                            'start_frame': i * self.sequence_length,
                        })
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        subject = sample['subject']
        primary_cam = sample['primary_cam']
        pair_cam = sample['pair_cam']
        start_frame = sample['start_frame']
        target_h, target_w = self.image_size

        stereo_params = self._get_stereo_params(subject, primary_cam, pair_cam)

        rgb_frames = []
        depth_frames = []

        cap1 = cv2.VideoCapture(sample['primary_video'])
        cap2 = cv2.VideoCapture(sample['pair_video'])
        cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(self.sequence_length):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                # Pad with last valid frame, or zeros at the start
                if rgb_frames:
                    rgb_frames.append(rgb_frames[-1])
                    depth_frames.append(depth_frames[-1])
                else:
                    rgb_frames.append(torch.zeros(3, target_h, target_w))
                    depth_frames.append(torch.zeros(1, target_h, target_w))
                continue

            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            rgb_rect, depth = self._compute_depth(frame1, frame2, stereo_params)

            rgb_resized = cv2.resize(rgb_rect, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            rgb_tensor = self.transform(Image.fromarray(rgb_resized))

            depth_resized = cv2.resize(depth, (target_w, target_h),
                                       interpolation=cv2.INTER_NEAREST)
            depth_tensor = torch.FloatTensor(depth_resized).unsqueeze(0)

            rgb_frames.append(rgb_tensor)
            depth_frames.append(depth_tensor)

        cap1.release()
        cap2.release()

        return {
            'rgb_sequence': torch.stack(rgb_frames),      # (T, 3, H, W)
            'depth_sequence': torch.stack(depth_frames),  # (T, 1, H, W)
            'subject': subject,
            'action': sample['action'],
            'camera_id': primary_cam,
        }


# for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")):
#                 poses = batch['pose_13'].to(DEVICE)
#                 frames = batch['frame'].to(DEVICE)
# TODO: get h36m
IMG_SIZE = [256, 192]
NUM_FRAMES = 10
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


def get_h36m(
        batch_size = 128,
        shuffle = False,
        seed = 0,
        download: bool = True,
        data_path="../datasets/H36M_FREI", # "/home/skyle/datasets/H36M_FREI"
        num_frames = NUM_FRAMES, 
        image_size = IMG_SIZE, 
    ):

    dataset = Human36mDataset(
        base_directory=data_path, #  # "../data/H36M_FREI"
        split='train',
        num_frames_per_video=num_frames,
        transform=transform,
        image_size=image_size
    )

    # Abuse: just treat validation as test set
    dataset_test = Human36mDataset(
        base_directory=data_path,  # "../data/H36M_FREI",
        split='validation',
        num_frames_per_video=int(num_frames/4),
        transform=transform,
        image_size=image_size
    )
    # train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    # val_loader   = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # TODO: Debug remain to test whether use properly
    train_loader, valid_loader = get_loader(
        dataset,
        split_train_val_ratio = 0.9,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )

    test_loader = get_loader(
        dataset_test,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        seed=seed
    )
    return train_loader, valid_loader, test_loader



def map_17_to_13_joints(pose_17, mapping):
    """
    Convert a 17-joint pose representation to a 13-joint representation using a specified mapping.

    Args:
        pose_17 (np.ndarray): Input pose with 17 joints of shape (17, 2)
        mapping (List[int]): List of indices mapping 17 joints to 13 joints

    Returns:
        np.ndarray: Mapped pose with 13 joints of shape (13, 2)
    """
    return pose_17[mapping]


