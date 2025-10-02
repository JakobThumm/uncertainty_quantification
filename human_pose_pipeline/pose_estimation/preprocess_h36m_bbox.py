"""
Preprocess H36M dataset by extracting bounding box images

This script processes the entire H36M dataset to:
1. Resize images for human detection
2. Detect human bounding boxes using YOLO
3. Extract bounding box only images
4. Transform ground truth poses to match the cropped images
5. Save preprocessed data in a new dataset format

The preprocessed dataset can be loaded much faster during training/inference
since human detection and preprocessing steps are already done.
"""

import os
import time
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from spacepy.pycdf import CDF
import jax.numpy as jnp

from human_pose_pipeline.utils.transform_utils import preprocess_image_with_bbox, SimpleTransform
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_human_detector,
    detect_humans,
    resize_image
)
from human_pose_pipeline.pose_estimation.h36m_settings import (
    JOINT_IDX_17,
    JOINT_IDX_13,
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_IMAGE_SIZE,
    YOLO_CONFIDENCE_THRESHOLD,
    TRANSFORM_SIGMA,
    TRANSFORM_IMAGE_SIZE,
    TRANSFORM_HEATMAP_SIZE,
    NORMALIZATION_OFFSET
)

# Dataset splits
SPLIT = {
    'train': ['S1', 'S6', 'S7', 'S8'],
    'validation': ['S9'],
    'test': ['S11']
}


def process_single_frame(frame_rgb, pose_13, human_detector, device_torch,
                         resized_target_size=YOLO_IMAGE_SIZE, verbose_timing=False,
                         transform_obj=None):
    """
    Process a single frame: resize, detect human, extract bbox image, transform pose

    Args:
        frame_rgb: Frame in RGB format (numpy array)
        pose_13: Ground truth 13-joint pose in original image coordinates
        human_detector: YOLO detector model
        device_torch: PyTorch device
        resized_target_size: Target size for human detection (width, height)
        verbose_timing: If True, print timing for each step
        transform_obj: Pre-created SimpleTransform object (for performance)

    Returns:
        dict or None: Dictionary with preprocessed data or None if processing failed
    """
    # Get original dimensions
    original_height, original_width = frame_rgb.shape[:2]

    # Step 1: Resize for human detection
    t0 = time.time() if verbose_timing else None
    pil_image = Image.fromarray(frame_rgb)
    resized_image, (orig_w, orig_h), (scale_x, scale_y) = resize_image(pil_image, resized_target_size)
    if verbose_timing:
        print(f"    Resize: {time.time()-t0:.4f}s")

    # Step 2: Detect humans
    t0 = time.time() if verbose_timing else None
    person_boxes = detect_humans(human_detector, resized_image, device_torch, threshold=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
    if verbose_timing:
        print(f"    YOLO detection: {time.time()-t0:.4f}s")

    if not person_boxes:
        return None

    # Use the first detected person (assumes single person in H36M)
    bbox = person_boxes[0]

    # Convert resized image back to numpy
    resized_image_np = np.array(resized_image)

    # Step 3: Preprocess with bounding box (this applies RegressFlow preprocessing)
    t0 = time.time() if verbose_timing else None
    if transform_obj is not None:
        # Use provided transform object (faster - no re-initialization)
        img, processed_bbox, center, scale, trans = transform_obj.test_transform(resized_image_np, bbox)
        input_tensor = jnp.expand_dims(img, axis=0)
    else:
        # Fallback: create transform object (slower)
        input_tensor, _, center, scale, trans, processed_bbox = preprocess_image_with_bbox(resized_image_np, bbox)
    if verbose_timing:
        print(f"    Bbox preprocessing: {time.time()-t0:.4f}s")

    # Extract the preprocessed image (remove batch dimension and convert to numpy)
    preprocessed_img = np.array(input_tensor[0])  # Shape: (3, 256, 192)

    # Step 4: Transform ground truth pose to match the preprocessed image
    t0 = time.time() if verbose_timing else None
    # First, scale pose from original image to resized image
    pose_resized = pose_13.copy()
    pose_resized[:, 0] = pose_resized[:, 0] / scale_x
    pose_resized[:, 1] = pose_resized[:, 1] / scale_y

    # Apply the affine transformation that was used for the image
    pose_transformed = cv2.transform(np.expand_dims(pose_resized, axis=0), trans)[0]

    # Convert to normalized coordinates (-0.5 to 0.5) as expected by RegressFlow
    img_height, img_width = CONFIG.DATA_PRESET.IMAGE_SIZE
    pose_normalized = pose_transformed.copy()
    pose_normalized[:, 0] = (pose_transformed[:, 0] / img_width) - 0.5
    pose_normalized[:, 1] = (pose_transformed[:, 1] / img_height) - 0.5

    # Apply mirror mapping to align left/right joints with model predictions
    pose_normalized_mirrored = pose_normalized[MIRROR_13_JOINT_MODEL_MAP]
    pose_pixel_mirrored = pose_13[MIRROR_13_JOINT_MODEL_MAP]
    if verbose_timing:
        print(f"    Pose transform: {time.time()-t0:.4f}s")

    return {
        'preprocessed_image': preprocessed_img,  # (3, 256, 192) normalized and ready for model
        'pose_normalized': pose_normalized_mirrored,  # (13, 2) in normalized coords, mirrored
        'pose_pixel': pose_pixel_mirrored,  # (13, 2) in pixel coords, mirrored
        'bbox': bbox,
        'center': center,
        'scale': scale,
        'trans': trans,
        'original_dims': (original_width, original_height),
        'scale_factors': (scale_x, scale_y)
    }


def preprocess_h36m_dataset(base_directory, output_directory, splits=['train', 'validation', 'test'],
                            num_frames_per_video=None):
    """
    Preprocess entire H36M dataset and save in H36M-like structure

    Output structure mirrors input:
    - output_dir/S1/PreprocessedImages/action.camera.npy (shape: N, 3, 256, 192)
    - output_dir/S1/PreprocessedPoses/action.camera.npz (normalized and pixel poses + metadata)

    Args:
        base_directory: Path to original H36M dataset
        output_directory: Path to save preprocessed dataset
        splits: Which splits to process
        num_frames_per_video: Number of frames to sample per video (None = all frames)
    """
    # Initialize human detector
    print("Initializing human detector...")
    human_detector, device_torch = initialize_human_detector()

    # Create output directory
    os.makedirs(output_directory, exist_ok=True)

    # Process each split
    for split in splits:
        print(f"\n{'='*80}")
        print(f"Processing {split} split...")
        print(f"{'='*80}\n")

        # Process each subject
        for subject in SPLIT[split]:
            print(f"\nProcessing subject {subject}...")
            poses_dir = os.path.join(base_directory, subject, 'Poses_D2_Positions')
            videos_dir = os.path.join(base_directory, subject, 'Videos')

            if not os.path.exists(poses_dir) or not os.path.exists(videos_dir):
                print(f"Skipping {subject}: directories not found")
                continue

            # Create output directories for this subject
            subject_output_dir = os.path.join(output_directory, subject)
            images_output_dir = os.path.join(subject_output_dir, 'PreprocessedImages')
            poses_output_dir = os.path.join(subject_output_dir, 'PreprocessedPoses')
            os.makedirs(images_output_dir, exist_ok=True)
            os.makedirs(poses_output_dir, exist_ok=True)

            # Process each action/video
            for filename in tqdm(os.listdir(poses_dir), desc=f"Subject {subject}"):
                if not filename.endswith('.cdf'):
                    continue

                try:
                    video_start_time = time.time()

                    # Load pose data
                    t0 = time.time()
                    file_path = os.path.join(poses_dir, filename)

                    # Find corresponding video
                    base = os.path.splitext(filename)[0]
                    video_filename = None
                    for possible_name in [f"{base}.mp4", f"_{base}.mp4"]:
                        if os.path.exists(os.path.join(videos_dir, possible_name)):
                            video_filename = possible_name
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
                    t1 = time.time()
                    print(f"  [{base}] Load poses: {t1-t0:.2f}s")

                    # Determine which frames to process
                    if num_frames_per_video is not None and total_frames > num_frames_per_video:
                        frame_indices = np.linspace(0, total_frames - 1, num_frames_per_video, dtype=int)
                    else:
                        frame_indices = np.arange(total_frames)

                    # Open video once for all frames (PERFORMANCE FIX)
                    t0 = time.time()
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Failed to open video: {video_path}")
                        continue
                    t1 = time.time()
                    print(f"  [{base}] Open video: {t1-t0:.2f}s")

                    # Create transform object once for all frames (PERFORMANCE OPTIMIZATION)
                    transform_obj = SimpleTransform(
                        scale_factor=0,
                        input_size=[TRANSFORM_IMAGE_SIZE[1], TRANSFORM_IMAGE_SIZE[0]],  # Height, Width
                        output_size=[TRANSFORM_HEATMAP_SIZE[1], TRANSFORM_HEATMAP_SIZE[0]],  # Height, Width
                        rot=0,
                        sigma=TRANSFORM_SIGMA,
                        train=False
                    )

                    # Collect all preprocessed data for this video
                    sequence_images = []
                    sequence_poses_normalized = []
                    sequence_poses_pixel = []
                    sequence_metadata = {
                        'bboxes': [],
                        'centers': [],
                        'scales': [],
                        'trans': [],
                        'original_dims': None,
                        'scale_factors': [],
                        'valid_frame_indices': [],
                        'subject': subject,
                        'action': base
                    }

                    # Timing accumulators
                    time_read_frames = 0
                    time_detect_human = 0
                    time_preprocess = 0
                    time_pose_transform = 0

                    # Process frames in batches for better YOLO performance
                    batch_size = 128  # Process 128 frames at a time
                    time_process_frames = 0
                    time_yolo_batch = 0

                    for batch_start in range(0, len(frame_indices), batch_size):
                        batch_end = min(batch_start + batch_size, len(frame_indices))
                        batch_indices = frame_indices[batch_start:batch_end]

                        # Read all frames in batch
                        t0_read = time.time()
                        batch_frames = []
                        batch_poses = []
                        batch_frame_ids = []

                        for frame_idx in batch_indices:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                            ret, frame = cap.read()

                            if not ret:
                                print(f"Failed to read frame {frame_idx} from {video_path}")
                                continue

                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            batch_frames.append(frame_rgb)
                            batch_poses.append(poses_13[frame_idx])
                            batch_frame_ids.append(frame_idx)

                        t1_read = time.time()
                        time_read_frames += t1_read - t0_read

                        if batch_start == 0 and len(batch_frames) > 0:
                            print(f"    First batch - Read {len(batch_frames)} frames: {t1_read - t0_read:.3f}s ({(t1_read - t0_read)/len(batch_frames)*1000:.1f}ms/frame)")

                        if not batch_frames:
                            continue

                        # Resize all frames
                        t0_resize = time.time()
                        batch_resized = []
                        batch_scale_factors = []

                        for frame_rgb in batch_frames:
                            pil_image = Image.fromarray(frame_rgb)
                            resized_image, (orig_w, orig_h), (scale_x, scale_y) = resize_image(
                                pil_image, target_size=YOLO_IMAGE_SIZE
                            )
                            batch_resized.append(resized_image)
                            batch_scale_factors.append((scale_x, scale_y))

                        t1_resize = time.time()
                        if batch_start == 0:
                            print(f"    First batch - Resize {len(batch_frames)} frames: {t1_resize-t0_resize:.3f}s ({(t1_resize-t0_resize)/len(batch_frames)*1000:.1f}ms/frame)")

                        # Batch YOLO detection (PERFORMANCE OPTIMIZATION)
                        t0_yolo = time.time()
                        batch_bboxes = []

                        # Run YOLO on entire batch at once
                        t0_predict = time.time()
                        results = human_detector.predict(batch_resized, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
                        t1_predict = time.time()

                        # Process YOLO results
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

                        t1_yolo = time.time()
                        time_yolo_batch += t1_yolo - t0_yolo

                        # Print detailed YOLO timing for first batch
                        if batch_start == 0:
                            print(f"    First batch - YOLO predict: {t1_predict - t0_predict:.3f}s ({(t1_predict - t0_predict)/len(batch_frames)*1000:.1f}ms/frame)")
                            print(f"    First batch - YOLO postprocess: {t1_yolo - t1_predict:.3f}s ({(t1_yolo - t1_predict)/len(batch_frames)*1000:.1f}ms/frame)")

                        # Process each frame in batch with detected bboxes
                        t0_batch_process = time.time()

                        # Timing for detailed breakdown (first batch only)
                        time_to_numpy = 0
                        time_bbox_transform = 0
                        time_pose_transform = 0

                        for idx, (frame_rgb, resized_image, pose_13, bbox, scale_factors, frame_idx) in enumerate(
                            zip(batch_frames, batch_resized, batch_poses, batch_bboxes, batch_scale_factors, batch_frame_ids)
                        ):
                            if bbox is None:
                                continue

                            # Get original dimensions
                            original_height, original_width = frame_rgb.shape[:2]
                            scale_x, scale_y = scale_factors

                            # Convert already-resized image to numpy (avoid re-resizing!)
                            t0 = time.time()
                            resized_image_np = np.array(resized_image)
                            if batch_start == 0:
                                time_to_numpy += time.time() - t0

                            # Preprocess with bounding box
                            t0 = time.time()
                            img, processed_bbox, center, scale, trans = transform_obj.test_transform(resized_image_np, bbox)
                            input_tensor = jnp.expand_dims(img, axis=0)
                            preprocessed_img = np.array(input_tensor[0])
                            if batch_start == 0:
                                time_bbox_transform += time.time() - t0

                            # Transform ground truth pose
                            t0 = time.time()
                            pose_resized = pose_13.copy()
                            pose_resized[:, 0] = pose_resized[:, 0] / scale_x
                            pose_resized[:, 1] = pose_resized[:, 1] / scale_y

                            pose_transformed = cv2.transform(np.expand_dims(pose_resized, axis=0), trans)[0]

                            img_height, img_width = [TRANSFORM_IMAGE_SIZE[1], TRANSFORM_IMAGE_SIZE[0]]
                            pose_normalized = pose_transformed.copy()
                            pose_normalized[:, 0] = (pose_transformed[:, 0] / img_width) - 0.5
                            pose_normalized[:, 1] = (pose_transformed[:, 1] / img_height) - 0.5

                            pose_normalized_mirrored = pose_normalized[MIRROR_13_JOINT_MODEL_MAP]
                            pose_pixel_mirrored = pose_13[MIRROR_13_JOINT_MODEL_MAP]
                            if batch_start == 0:
                                time_pose_transform += time.time() - t0

                            # Accumulate data for this sequence
                            sequence_images.append(preprocessed_img)
                            sequence_poses_normalized.append(pose_normalized_mirrored)
                            sequence_poses_pixel.append(pose_pixel_mirrored)
                            sequence_metadata['bboxes'].append(bbox)
                            sequence_metadata['centers'].append(center)
                            sequence_metadata['scales'].append(scale)
                            sequence_metadata['trans'].append(trans)
                            sequence_metadata['scale_factors'].append((scale_x, scale_y))
                            sequence_metadata['valid_frame_indices'].append(int(frame_idx))

                            if sequence_metadata['original_dims'] is None:
                                sequence_metadata['original_dims'] = (original_width, original_height)

                        t1_batch_process = time.time()
                        time_process_frames += t1_batch_process - t0_batch_process

                        # Print detailed breakdown for first batch
                        if batch_start == 0 and len(batch_frames) > 0:
                            print(f"    First batch - PIL to numpy: {time_to_numpy:.3f}s ({time_to_numpy/len(batch_frames)*1000:.1f}ms/frame)")
                            print(f"    First batch - Bbox transform: {time_bbox_transform:.3f}s ({time_bbox_transform/len(batch_frames)*1000:.1f}ms/frame)")
                            print(f"    First batch - Pose transform: {time_pose_transform:.3f}s ({time_pose_transform/len(batch_frames)*1000:.1f}ms/frame)")

                    # Close video after processing all frames
                    cap.release()

                    print(f"  [{base}] Frame reading: {time_read_frames:.2f}s total")
                    print(f"  [{base}] YOLO batch detection: {time_yolo_batch:.2f}s total ({time_yolo_batch/len(sequence_images)*1000:.1f}ms/frame)")
                    print(f"  [{base}] Frame processing: {time_process_frames:.2f}s total")

                    # Skip if no valid frames
                    if len(sequence_images) == 0:
                        print(f"No valid frames for {filename}")
                        continue

                    # Convert lists to arrays
                    t0 = time.time()
                    sequence_images = np.array(sequence_images)  # (N, 3, 256, 192)
                    sequence_poses_normalized = np.array(sequence_poses_normalized)  # (N, 13, 2)
                    sequence_poses_pixel = np.array(sequence_poses_pixel)  # (N, 13, 2)
                    t1 = time.time()
                    print(f"  [{base}] Convert to arrays: {t1-t0:.2f}s")

                    # Save preprocessed images as single array
                    t0 = time.time()
                    images_output_path = os.path.join(images_output_dir, f"{base}.npy")
                    np.save(images_output_path, sequence_images)
                    t1 = time.time()
                    print(f"  [{base}] Save images: {t1-t0:.2f}s")

                    # Save poses and metadata as compressed npz
                    t0 = time.time()
                    poses_output_path = os.path.join(poses_output_dir, f"{base}.npz")
                    np.savez_compressed(
                        poses_output_path,
                        poses_normalized=sequence_poses_normalized,
                        poses_pixel=sequence_poses_pixel,
                        bboxes=np.array(sequence_metadata['bboxes']),
                        centers=np.array(sequence_metadata['centers']),
                        scales=np.array(sequence_metadata['scales']),
                        trans=np.array(sequence_metadata['trans']),
                        scale_factors=np.array(sequence_metadata['scale_factors']),
                        original_dims=np.array(sequence_metadata['original_dims']),
                        valid_frame_indices=np.array(sequence_metadata['valid_frame_indices']),
                        subject=subject,
                        action=base
                    )
                    t1 = time.time()
                    print(f"  [{base}] Save poses: {t1-t0:.2f}s")

                    video_total_time = time.time() - video_start_time
                    print(f"  [{base}] TOTAL: {video_total_time:.2f}s for {len(sequence_images)} frames ({video_total_time/len(sequence_images):.3f}s/frame)")

                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        print(f"\n{split} split complete")

    print(f"\n{'='*80}")
    print("Preprocessing complete!")
    print(f"{'='*80}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess H36M dataset with bounding box extraction')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to original H36M dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to save preprocessed dataset')
    parser.add_argument('--splits', nargs='+', default=['train', 'validation', 'test'],
                       help='Which splits to process')
    parser.add_argument('--num_frames', type=int, default=None,
                       help='Number of frames to sample per video (None = all)')

    args = parser.parse_args()

    preprocess_h36m_dataset(
        base_directory=args.input_dir,
        output_directory=args.output_dir,
        splits=args.splits,
        num_frames_per_video=args.num_frames
    )


if __name__ == '__main__':
    main()
