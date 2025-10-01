import os
import numpy as np
from spacepy.pycdf import CDF
import torch

from PIL import Image
from torchvision import transforms
import cv2

from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_IDX_13,
    JOINT_IDX_17,
    CONNECTIONS_13
)

SPLIT = {
    'train': ['S1', 'S9'],#, 'S6', 'S7', 'S8', 'S9'],
    'validation': ['S11'],
    'test': ['S9']
}

# THIS SHOULD NOT BE USED.
# transform = transforms.Compose([
#     transforms.Resize((256, 192)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
# ])


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
            poses_dir = os.path.join(base_directory, subject, 'D2_Positions')
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

        return {
            'pose_13': torch.FloatTensor(adjusted_keypoints),
            'frame': frame,
            'video_path': video_path
        }

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



def visualize_pose_sequence_with_images(pose_sequence, images, output_file, num_frames=None, estimated_poses=None, estimated_uncertainties=None, estimated_covariances=None):
    """
    Create an animated visualization of pose sequences overlaid on image frames.

    Args:
        pose_sequence (np.ndarray): Ground truth pose sequence of shape (num_frames, num_joints, 2)
        images (torch.Tensor): Image frames of shape (num_frames, C, H, W)
        output_file (str): Path where the output GIF will be saved
        num_frames (int, optional): Number of frames to visualize. Defaults to None (all frames)
        estimated_poses (List[np.ndarray], optional): Estimated poses for each frame
        estimated_uncertainties (List[np.ndarray], optional): Uncertainty values for each joint
        estimated_covariances (List[np.ndarray], optional): Covariance values for each joint

    Returns:
        None: The visualization is saved directly to the output file
    """
    if num_frames is None:
        num_frames = pose_sequence.shape[0]
    
    # Initialize a list to store individual frames for the GIF
    frames_for_gif = []
    
    for frame in range(num_frames):
        # Retrieve the image frame and convert it to a NumPy array
        image_tensor = images[frame]
        image_pil = transforms.ToPILImage()(image_tensor)
        image = np.array(image_pil).copy()
        
        # Overlay the ground truth pose on the image
        gt_pose = pose_sequence[frame]
        
        # Ensure pose coordinates are within image bounds
        image_height, image_width, _ = image.shape
        gt_pose[:, 0] = np.clip(gt_pose[:, 0], 0, image_width - 1)
        gt_pose[:, 1] = np.clip(gt_pose[:, 1], 0, image_height - 1)
        
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
            est_uncertainty = estimated_uncertainties[frame]
            est_covariance = estimated_covariances[frame]
            #est_pose[:, 0] = np.clip(est_pose[:, 0], 0, image_width - 1)
            #est_pose[:, 1] = np.clip(est_pose[:, 1], 0, image_height - 1)
            
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

                std_x, std_y = est_uncertainty[idx]
                cov_xy = est_covariance[idx]

                if std_x > 0 and std_y > 0:
                    # Calculate the angle of the ellipse
                    angle = 0.5 * np.arctan2(2 * cov_xy, (std_x**2 - std_y**2)) * (180 / np.pi)

                    # Calculate the width and height of the ellipse based on standard deviations
                    width = int(2 * std_x)  # 2 standard deviations
                    height = int(2 * std_y)

                    #print("Width and Height: ", width, height)
                    #print("Stdx, stdy: ", std_x, std_y)
                    #print("Angle: ", angle)

                    # Ensure width and height are positive
                    #width = max(width, 1)
                    #height = max(height, 1)

                    # Draw the uncertainty ellipse
                    cv2.ellipse(image, (int(x), int(y)), (width, height), angle, 0, 360, (0, 0, 255), 1)  # Red ellipse for uncertainty

        
        # Convert back to PIL Image for consistency
        image_with_pose = Image.fromarray(image)
        
        # Append to frames list for GIF creation
        frames_for_gif.append(image_with_pose)
    
    # Create an animated GIF with the overlaid poses
    frames_for_gif[0].save(
        output_file,
        save_all=True,
        append_images=frames_for_gif[1:],
        duration=10,  # Duration between frames in milliseconds
        loop=0
    )
    print(f"Visualization with overlaid poses saved as {output_file}")


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


def evaluate_pose_estimation_full(ground_truth, estimated_pose, estimated_uncertainty, estimated_covariance):
    """
    Evaluate pose estimation accuracy using Mahalanobis distance and confidence intervals.

    Computes how many estimated joints fall within different standard deviation intervals
    of their corresponding ground truth positions, taking into account uncertainty and covariance.

    Args:
        ground_truth (np.ndarray): Ground truth pose of shape (num_joints, 2)
        estimated_pose (np.ndarray): Estimated pose of shape (num_joints, 2)
        estimated_uncertainty (np.ndarray): Standard deviations for each joint of shape (num_joints, 2)
        estimated_covariance (np.ndarray): Covariance values for each joint of shape (num_joints,)

    Returns:
        dict: Dictionary containing:
            - counts: Number of joints within each standard deviation interval
            - joint_results: Detailed results for each joint
            - num_joints: Total number of joints evaluated
    """
    from scipy.stats import chi2

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


def compute_oks(pred_keypoints, gt_keypoints, bbox_area, sigmas=None):
    """
    Compute the Object Keypoint Similarity (OKS) between predicted and ground truth keypoints.
    0: No overlap or similarity between predicted and ground truth keypoints.
    1: Perfect match between predicted and ground truth keypoints.

    Args:
    pred_keypoints (np.ndarray): Predicted keypoints of shape (K, 2).
    gt_keypoints (np.ndarray): Ground truth keypoints of shape (K, 3), where each keypoint is (x, y, v).
    bbox_area (float): Area of the ground truth bounding box.
    sigmas (np.ndarray): Per-keypoint sigmas. If None, uses COCO's default sigmas.

    Returns:
    float: OKS score.
    """

    if sigmas is None:
        # Default COCO sigmas for 17 keypoints
        sigmas = np.array([
            .26, .25, .25, .35, .35, .79, .79,
            .72, .72, .62, .62, 1.07, 1.07,
            .87, .87, .89, .89
        ]) / 10.0  # Convert to appropriate scale

    # Ensure inputs are NumPy arrays
    pred_keypoints = np.array(pred_keypoints)  # Shape: (K, 2)
    gt_keypoints = np.array(gt_keypoints)      # Shape: (K, 3)
    # pred_keypoints = np.squeeze(pred_keypoints)  # Now shape should be (K, 2)
    gt_keypoints = np.squeeze(gt_keypoints)            # Now shape should be (K, 2)

    # print(f"pred_keypoints shape: {pred_keypoints.shape}")
    # print(f"gt_keypoints shape: {gt_keypoints.shape}")
   
    sigmas = np.array(sigmas)

    # Extract ground truth coordinates and visibility
    gt_coords = gt_keypoints[:, :2]       # Shape: (K, 2)
    gt_visibility = gt_keypoints[:, 2]    # Shape: (K,)

    # Only consider annotated keypoints (v > 0)
    valid = gt_visibility > 0
    # print(f"valid shape: {valid.shape}")
    if not np.any(valid):
        return 0.0  # No valid keypoints to compare
    
    # Compute squared distances for valid keypoints
    dx = pred_keypoints[valid, 0] - gt_coords[valid, 0]
    dy = pred_keypoints[valid, 1] - gt_coords[valid, 1]
    squared_distances = dx ** 2 + dy ** 2

    # Object scale (s)
    s = bbox_area
    if s <= 0:
        s = 1.0  # Avoid division by zero or negative area

    # Per-keypoint variances
    vars = (sigmas[valid] * 2) ** 2

    # Compute OKS numerator
    oks_elements = np.exp(-squared_distances / (2 * s * vars))
    numerator = np.sum(oks_elements)

    # Compute OKS denominator
    denominator = len(oks_elements)  # Number of valid keypoints

    # Compute OKS
    oks = numerator / denominator

    return oks
