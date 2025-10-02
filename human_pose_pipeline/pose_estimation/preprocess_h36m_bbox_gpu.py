#!/usr/bin/env python3
"""
GPU-Accelerated H36M Dataset Preprocessing with Bounding Box Extraction

This script preprocesses the entire H36M dataset by:
1. Loading video frames in batches
2. Resizing frames for YOLO detection (batched on CPU with OpenCV)
3. Running YOLO human detection (batched on GPU)
4. Applying bbox transformations (batched on GPU with PyTorch)
5. Transforming ground truth poses (batched on GPU with PyTorch)

This version is optimized for speed using batched GPU operations.
"""

import os
import argparse
import time
import numpy as np
import cv2
from spacepy.pycdf import CDF
from tqdm import tqdm
from PIL import Image

from human_pose_pipeline.pose_estimation.inference_helper import initialize_human_detector
from human_pose_pipeline.utils.batched_transform_torch import batched_preprocess_frames_gpu

from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_IDX_17,
    JOINT_IDX_13,
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_IMAGE_SIZE,
    YOLO_CONFIDENCE_THRESHOLD,
    TRANSFORM_IMAGE_SIZE
)

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    print("=" * 80)
    print("WARNING: decord library not available, falling back to OpenCV (slower)")
    print("For faster video reading, install decord:")
    print("  pip install decord")
    print("=" * 80)


root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Dataset splits
SPLIT = {
    'train': ['S1', 'S6', 'S7', 'S8'],
    'validation': ['S9'],
    'test': ['S11']
}


def read_video_frames_decord(video_path, frame_indices):
    """
    Fast batch reading of video frames using decord library

    Args:
        video_path: Path to video file
        frame_indices: Array of frame indices to read

    Returns:
        frames: Numpy array (N, H, W, 3) in RGB format
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(frame_indices).asnumpy()  # (N, H, W, 3) RGB
    return frames


def read_video_frames_opencv(video_path, frame_indices):
    """
    Fallback video reading using OpenCV (slower)

    Args:
        video_path: Path to video file
        frame_indices: Array of frame indices to read

    Returns:
        frames: List of numpy arrays (H, W, 3) in RGB format
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


def resize_frames_batch(frames, target_size=YOLO_IMAGE_SIZE):
    """
    Resize a batch of frames using OpenCV

    Args:
        frames: List or array of numpy arrays (H, W, 3)
        target_size: (width, height) for output

    Returns:
        resized_frames: List of PIL Images
        scale_factors: List of (scale_x, scale_y) tuples
    """
    resized_frames = []
    scale_factors = []

    for frame in frames:
        h, w = frame.shape[:2]
        target_w, target_h = target_size

        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        pil_image = Image.fromarray(resized)

        scale_x = w / target_w
        scale_y = h / target_h

        resized_frames.append(pil_image)
        scale_factors.append((scale_x, scale_y))

    return resized_frames, scale_factors


def preprocess_h36m_dataset_gpu(
    dataset_dir,
    output_dir,
    batch_size=128,
    num_frames_per_video=None,
    device='cuda'
):
    """
    Preprocess H36M dataset with GPU acceleration

    Args:
        dataset_dir: Path to H36M extracted dataset
        output_dir: Path to save preprocessed data
        batch_size: Number of frames to process at once
        num_frames_per_video: Limit frames per video (None = all)
        device: Device for PyTorch operations
    """
    # Initialize YOLO detector
    print("Initializing YOLO detector...")
    human_detector, device_torch = initialize_human_detector(device)
    print(f"YOLO detector initialized on {device_torch}")

    # Process each split
    for split_name, subjects in SPLIT.items():
        print(f"\nProcessing {split_name} split...")

        for subject in subjects:
            print(f"\nProcessing subject {subject}...")
            subject_dir = os.path.join(dataset_dir, subject)

            poses_dir = os.path.join(subject_dir, 'Poses_D2_Positions')
            videos_dir = os.path.join(subject_dir, 'Videos')

            if not os.path.exists(poses_dir) or not os.path.exists(videos_dir):
                print(f"Skipping {subject} - directories not found")
                continue

            # Create output directories
            images_output_dir = os.path.join(output_dir, subject, 'PreprocessedImages')
            poses_output_dir = os.path.join(output_dir, subject, 'PreprocessedPoses')
            os.makedirs(images_output_dir, exist_ok=True)
            os.makedirs(poses_output_dir, exist_ok=True)

            # Process each pose file
            pose_files = sorted([f for f in os.listdir(poses_dir) if f.endswith('.cdf')])

            for filename in tqdm(pose_files, desc=f'Subject {subject}'):
                video_start_time = time.time()

                try:
                    file_path = os.path.join(poses_dir, filename)
                    base = os.path.splitext(filename)[0]

                    # Find corresponding video
                    video_filename = None
                    for video_ext in ['.mp4', '.avi']:
                        candidate = base + video_ext
                        if os.path.exists(os.path.join(videos_dir, candidate)):
                            video_filename = candidate
                            break

                    if not video_filename:
                        print(f"No video found for {filename}")
                        continue

                    video_path = os.path.join(videos_dir, video_filename)

                    # Load poses
                    with CDF(file_path) as cdf:
                        poses = cdf['Pose'][:].reshape(-1, 32, 2)
                        poses_17 = poses[:, JOINT_IDX_17, :]
                        poses_13 = poses_17[:, JOINT_IDX_13, :]
                        total_frames = len(poses_13)

                    # Determine which frames to process
                    if num_frames_per_video is not None and total_frames > num_frames_per_video:
                        frame_indices = np.linspace(0, total_frames - 1, num_frames_per_video, dtype=int)
                    else:
                        frame_indices = np.arange(total_frames)

                    # Collect all preprocessed data for this video
                    all_preprocessed_images = []
                    all_preprocessed_poses = []
                    all_metadata = {
                        'bboxes': [],
                        'centers': [],
                        'scales': [],
                        'trans': [],
                        'original_dims': None,
                        'scale_factors': [],
                        'valid_frame_indices': []
                    }

                    # Timing accumulators
                    time_read_frames = 0
                    time_resize = 0
                    time_yolo = 0
                    time_gpu_preprocess = 0

                    # Process frames in batches
                    for batch_start in range(0, len(frame_indices), batch_size):
                        batch_end = min(batch_start + batch_size, len(frame_indices))
                        batch_indices = frame_indices[batch_start:batch_end]

                        # ==================== STEP 1: Read frames ====================
                        t0 = time.time()

                        # Use decord for fast batch reading if available
                        if DECORD_AVAILABLE:
                            try:
                                batch_frames_raw = read_video_frames_decord(video_path, batch_indices)
                            except Exception as e:
                                print(f"Decord failed: {e}, falling back to OpenCV")
                                batch_frames_raw = read_video_frames_opencv(video_path, batch_indices)
                        else:
                            batch_frames_raw = read_video_frames_opencv(video_path, batch_indices)

                        batch_poses_raw = poses_13[batch_indices]

                        if len(batch_frames_raw) == 0:
                            continue

                        t1 = time.time()
                        time_read_frames += t1 - t0

                        # ==================== STEP 2: Resize frames ====================
                        t0 = time.time()
                        batch_frames_resized, batch_scale_factors = resize_frames_batch(
                            batch_frames_raw, target_size=YOLO_IMAGE_SIZE
                        )
                        t1 = time.time()
                        time_resize += t1 - t0

                        # ==================== STEP 3: YOLO detection ====================
                        t0 = time.time()
                        results = human_detector.predict(batch_frames_resized, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)

                        batch_bboxes = []
                        for result in results:
                            person_boxes = []
                            if result.boxes is not None:
                                boxes = result.boxes.xyxy.cpu().numpy()
                                confidences = result.boxes.conf.cpu().numpy()
                                classes = result.boxes.cls.cpu().numpy()

                                for i, cls in enumerate(classes):
                                    if int(cls) == 0 and confidences[i] >= YOLO_CONFIDENCE_THRESHOLD:
                                        person_boxes.append(boxes[i].tolist())

                            batch_bboxes.append(person_boxes[0] if person_boxes else None)

                        t1 = time.time()
                        time_yolo += t1 - t0

                        # ==================== STEP 4: GPU batch preprocessing ====================
                        t0 = time.time()
                        images_preprocessed, poses_normalized, metadata = batched_preprocess_frames_gpu(
                            frames=batch_frames_resized,
                            bboxes=batch_bboxes,
                            poses=batch_poses_raw,
                            scale_factors=batch_scale_factors,
                            output_image_size=(TRANSFORM_IMAGE_SIZE[0], TRANSFORM_IMAGE_SIZE[1]),
                            device=device
                        )
                        t1 = time.time()
                        time_gpu_preprocess += t1 - t0

                        # Skip if no valid frames
                        if len(metadata['valid_indices']) == 0:
                            continue

                        # Convert to numpy and apply mirror mapping
                        images_np = images_preprocessed.cpu().numpy()  # (B, 3, 256, 192)
                        poses_np = poses_normalized.cpu().numpy()  # (B, 13, 2)

                        # Apply mirror mapping to poses
                        poses_mirrored = poses_np[:, MIRROR_13_JOINT_MODEL_MAP, :]

                        # Get original image dimensions
                        if all_metadata['original_dims'] is None:
                            orig_h, orig_w = batch_frames_raw[0].shape[:2]
                            all_metadata['original_dims'] = (orig_w, orig_h)

                        # Accumulate results
                        all_preprocessed_images.append(images_np)
                        all_preprocessed_poses.append(poses_mirrored)

                        # Accumulate metadata
                        valid_batch_indices = metadata['valid_indices']
                        for i, valid_idx in enumerate(valid_batch_indices):
                            global_frame_idx = batch_indices[valid_idx]
                            all_metadata['bboxes'].append(metadata['bboxes'][i])
                            all_metadata['centers'].append(metadata['centers'][i])
                            all_metadata['scales'].append(metadata['scales'][i])
                            all_metadata['trans'].append(metadata['transforms'][i])
                            all_metadata['scale_factors'].append(metadata['scale_factors'][i])
                            all_metadata['valid_frame_indices'].append(int(global_frame_idx))

                    # Skip if no valid frames
                    if len(all_preprocessed_images) == 0:
                        print(f"  [{base}] No valid frames")
                        continue

                    # Concatenate all batches
                    t0 = time.time()
                    sequence_images = np.concatenate(all_preprocessed_images, axis=0)  # (N, 3, 256, 192)
                    sequence_poses = np.concatenate(all_preprocessed_poses, axis=0)  # (N, 13, 2)

                    # Compute pixel poses for metadata (reverse normalization)
                    sequence_poses_pixel = sequence_poses.copy()
                    sequence_poses_pixel[:, :, 0] = (sequence_poses[:, :, 0] + 0.5) * TRANSFORM_IMAGE_SIZE[0]
                    sequence_poses_pixel[:, :, 1] = (sequence_poses[:, :, 1] + 0.5) * TRANSFORM_IMAGE_SIZE[1]

                    # Save preprocessed images
                    images_output_path = os.path.join(images_output_dir, f"{base}.npy")
                    np.save(images_output_path, sequence_images)

                    # Save poses and metadata
                    poses_output_path = os.path.join(poses_output_dir, f"{base}.npz")
                    np.savez_compressed(
                        poses_output_path,
                        poses_normalized=sequence_poses,
                        poses_pixel=sequence_poses_pixel,
                        bboxes=np.array(all_metadata['bboxes']),
                        centers=np.array(all_metadata['centers']),
                        scales=np.array(all_metadata['scales']),
                        trans=np.array(all_metadata['trans']),
                        scale_factors=np.array(all_metadata['scale_factors']),
                        original_dims=np.array(all_metadata['original_dims']),
                        valid_frame_indices=np.array(all_metadata['valid_frame_indices']),
                        subject=subject,
                        action=base
                    )
                    time_save = time.time() - t0

                    total_time = time.time() - video_start_time
                    print(f"  [{base}] {len(sequence_images)} frames in {total_time:.1f}s "
                          f"({total_time / len(sequence_images) * 1000:.1f}ms/frame) - "
                          f"read:{time_read_frames:.1f}s yolo:{time_yolo:.1f}s gpu:{time_gpu_preprocess:.1f}s save:{time_save:.1f}s")

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Preprocess H36M dataset with GPU acceleration')
    parser.add_argument('--dataset_dir', type=str,
                       default='datasets/H36M/extracted',
                       help='Path to H36M extracted dataset')
    parser.add_argument('--output_dir', type=str,
                       default='datasets/H36M/pre_processed',
                       help='Path to save preprocessed data')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for processing')
    parser.add_argument('--num_frames', type=int, default=None,
                       help='Limit frames per video (None = all)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for PyTorch operations')

    args = parser.parse_args()

    preprocess_h36m_dataset_gpu(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_frames_per_video=args.num_frames,
        device=args.device
    )


if __name__ == '__main__':
    main()
