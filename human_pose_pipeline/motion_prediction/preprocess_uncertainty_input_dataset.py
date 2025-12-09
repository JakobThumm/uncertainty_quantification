"""This script preprocesses the data to be used for the motion prediction with uncertainty input.

The model input is the predicted 3D human pose with covariance matrix.
The model output is the predicted 3D human pose in the next 10 timesteps.
The ground truth future human pose is available in datasets/H36M/extracted.

This script loads the raw extracted H36M data from datasets/H36M/extracted,
  predicts the human pose and uncertainty covariance matrices using batched processing,
  and saves the data to datasets/H36M/pre_processed_motion.
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm

from src.datasets.h36m import Human36mDatasetSequenceTwoCameras, SPLIT
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
)
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d
)
from human_pose_pipeline.pose_estimation.triangulation_helper import (
    load_camera_parameters
)
from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
    OOD_THRESHOLD
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Camera IDs for stereo setup
CAMERA_IDS = ['55011271', '60457274']


def process_sequence_batched(
    left_frames,
    right_frames,
    pose_estimation_jit_fn,
    params,
    batch_stats,
    human_detector,
    device_torch,
    projection_matrices,
    batch_size=32,
    device='cpu',
    score_fn=None,
    ood_threshold=OOD_THRESHOLD,
):
    """Process a sequence of image pairs in batches to extract 3D poses with covariances.

    Args:
        left_frames: List of PIL images from left camera
        right_frames: List of PIL images from right camera
        pose_estimation_jit_fn: JAX pose estimation function
        params: Model parameters
        batch_stats: Batch statistics
        human_detector: YOLO human detector
        device_torch: PyTorch device for YOLO
        projection_matrices: List with projection matrices for both cameras [P1, P2]
        batch_size: Number of frames to process in a batch
        device: Device to use for tensors
        score_fn: Optional OOD score function
        ood_threshold: OOD detection threshold

    Returns:
        poses_3d: Array of 3D poses (num_frames, 13, 3)
        covariances_3d: Array of 3D covariances (num_frames, 13, 3, 3)
        valid_mask: Boolean mask indicating which frames have valid detections
    """
    num_frames = len(left_frames)

    # Pre-allocate arrays for results
    all_3d_points_list = []
    all_3d_covariances_list = []
    all_ood_scores_list = []
    all_is_ood_list = []
    all_batch_sizes = []

    # Process frames in batches
    for frame_idx in range(0, num_frames, batch_size):
        current_batch_size = min(batch_size, num_frames - frame_idx)
        all_batch_sizes.append(current_batch_size)

        # Get batch of frames from both cameras
        left_batch = left_frames[frame_idx:frame_idx + current_batch_size]
        right_batch = right_frames[frame_idx:frame_idx + current_batch_size]

        # Combine frames for batched processing
        both_frames = left_batch + right_batch

        # Process the batch
        points_3d, C_3d_all, ood_score, is_ood = process_frame_3d(
            frames=both_frames,
            projection_matrices=projection_matrices,
            pose_estimation_jit_fn=pose_estimation_jit_fn,
            params=params,
            batch_stats=batch_stats,
            human_detector=human_detector,
            device_torch=device_torch,
            mirror_map=MIRROR_13_JOINT_MODEL_MAP,
            score_fn=score_fn,
            human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
            ood_threshold=ood_threshold,
            verbose=False,
            device=device
        )

        # Store results (move to CPU to free GPU memory)
        all_3d_points_list.append(points_3d.to('cpu').numpy())
        all_3d_covariances_list.append(C_3d_all.to('cpu').numpy())
        all_ood_scores_list.append(ood_score.to('cpu').numpy())
        all_is_ood_list.append(is_ood.to('cpu').numpy())

        # Free GPU memory
        del points_3d, C_3d_all, ood_score, is_ood

    # Concatenate results
    poses_3d = np.zeros((num_frames, 13, 3))
    covariances_3d = np.zeros((num_frames, 13, 3, 3))
    ood_scores = np.zeros((num_frames,))
    is_ood_all = np.zeros((num_frames,), dtype=bool)

    index = 0
    for i, batch_size_i in enumerate(all_batch_sizes):
        poses_3d[index:index + batch_size_i] = all_3d_points_list[i]
        covariances_3d[index:index + batch_size_i] = all_3d_covariances_list[i]
        ood_scores[index:index + batch_size_i] = all_ood_scores_list[i]
        is_ood_all[index:index + batch_size_i] = all_is_ood_list[i]
        index += batch_size_i

    # Create valid mask (frames where human was detected in both cameras)
    # Check if all values are zero (no detection)
    valid_mask = ~np.all(poses_3d == 0, axis=(1, 2))

    return poses_3d, covariances_3d, valid_mask, ood_scores, is_ood_all


def preprocess_subject(
    subject,
    dataset,
    output_dir,
    pose_estimation_jit_fn,
    params,
    batch_stats,
    human_detector,
    device_torch,
    projection_matrices,
    batch_size=32,
    device='cpu',
    score_fn=None,
    ood_threshold=OOD_THRESHOLD,
):
    """Preprocess all sequences for a given subject.

    Args:
        subject: Subject ID (e.g., 'S1')
        dataset: Human36mDatasetSequenceTwoCameras instance
        output_dir: Path to save processed motion data
        pose_estimation_jit_fn: JAX pose estimation function
        params: Model parameters
        batch_stats: Batch statistics
        human_detector: YOLO human detector
        device_torch: PyTorch device for YOLO
        projection_matrices: List with projection matrices [P1, P2]
        batch_size: Batch size for processing
        device: Device to use for tensors
        score_fn: Optional OOD score function
        ood_threshold: OOD detection threshold
    """
    subject_output_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_output_dir, exist_ok=True)

    print(f"\nProcessing subject {subject}: {len(dataset)} sequences")

    for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {subject}")):
        # Extract sequence information from video paths
        # video_paths format: /path/to/Subject/Videos/Action.CameraID.mp4
        video_paths = dataset.data[idx]['video_paths']
        if len(video_paths) == 0:
            continue

        # Extract subject and action from the video path
        video_path = video_paths[0]
        path_parts = video_path.split(os.sep)
        sample_subject = path_parts[-3]  # Subject directory
        action_file = os.path.basename(video_path)  # Action.CameraID.mp4
        action_name = '.'.join(action_file.split('.')[:-2])  # Remove .CameraID.mp4

        # Skip if this sample doesn't belong to the current subject
        if sample_subject != subject:
            continue

        left_frames = sample['all_camera_frames'][0]
        right_frames = sample['all_camera_frames'][1]

        # Process the sequence in batches
        poses_3d, covariances_3d, valid_mask, ood_scores, is_ood = process_sequence_batched(
            left_frames,
            right_frames,
            pose_estimation_jit_fn,
            params,
            batch_stats,
            human_detector,
            device_torch,
            projection_matrices,
            batch_size=batch_size,
            device=device,
            score_fn=score_fn,
            ood_threshold=ood_threshold,
        )

        # Save the processed data
        output_filename = f"{action_name}_seq{idx:04d}.npz"
        output_path = os.path.join(subject_output_dir, output_filename)
        np.savez_compressed(
            output_path,
            poses_3d=poses_3d,
            covariances_3d=covariances_3d,
            valid_mask=valid_mask,
            ood_scores=ood_scores,
            is_ood=is_ood,
        )

        valid_count = valid_mask.sum()
        ood_count = is_ood.sum() if score_fn is not None else 0
        print(f"  {action_name}_seq{idx:04d}: {valid_count}/{len(valid_mask)} valid frames, {ood_count} OOD detections")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess H36M data for motion prediction with uncertainty'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='datasets/',
        help='Path to datasets directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='datasets/H36M/pre_processed_motion',
        help='Path to save processed motion prediction data'
    )
    parser.add_argument(
        '--model_save_path',
        type=str,
        default='human_pose_pipeline/models/pose_estimation',
        help='Path to saved pose estimation models'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default='finetuned_h36m_regressflow_with_unc',
        help='Model run name'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='all',
        choices=['train', 'validation', 'test', 'all'],
        help='Which split to process'
    )
    parser.add_argument(
        '--subject',
        type=str,
        default=None,
        help='Process only specific subject (e.g., S1). If not specified, processes all subjects in split.'
    )
    parser.add_argument(
        '--camera_ids',
        type=str,
        nargs=2,
        default=CAMERA_IDS,
        help='Camera IDs to use'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--enable_ood',
        action='store_true',
        help='Enable OOD detection'
    )
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache/',
        help='Cache directory with score functions'
    )
    parser.add_argument(
        '--base_key',
        type=str,
        default=None,
        help='Base key for loading the OOD score functions'
    )
    parser.add_argument(
        '--ood_threshold',
        type=float,
        default=OOD_THRESHOLD,
        help='OOD threshold'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("H36M Motion Prediction Dataset Preprocessing")
    print("=" * 80)
    print(f"Data directory: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.run_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    if args.enable_ood:
        print(f"OOD detection: ENABLED (threshold={args.ood_threshold})")
    print("=" * 80)

    # Initialize models
    print("\nInitializing models...")
    models_dir = os.path.join(root_dir, args.model_save_path, "H36M", "RegressFlow", "seed_420")
    checkpoint_path_jax = os.path.join(models_dir, args.run_name)

    pose_estimation_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path_jax)
    print("Pose estimation model loaded")

    # Initialize YOLO human detector
    human_detector, device_torch = initialize_human_detector('cuda')
    print("YOLO human detector loaded")

    # Load OOD score functions if enabled
    score_fn = None
    if args.enable_ood:
        if args.base_key is None:
            print("\nWARNING: OOD detection enabled but no base_key provided. Skipping OOD detection.")
            print("Use --base_key to specify the cache key for OOD score functions.")
        else:
            from src.ood_scores.lm_lanczos import load_score_functions
            print(f"\nLoading OOD score functions with cache key: {args.base_key}")
            score_fn, _, _, _ = load_score_functions(args.cache_dir, args.base_key)
            print("OOD score functions loaded successfully!")

    # Load camera parameters
    camera_parameters_path = os.path.join(models_dir, 'camera-parameters.json')
    if not os.path.exists(camera_parameters_path):
        raise FileNotFoundError(
            f"Camera parameters file not found at {camera_parameters_path}. "
            "Please ensure the camera-parameters.json file is available in the models directory."
        )

    # Determine which subjects to process
    if args.subject:
        subjects_to_process = [args.subject]
    elif args.split == 'all':
        subjects_to_process = []
        for split_subjects in SPLIT.values():
            subjects_to_process.extend(split_subjects)
    else:
        subjects_to_process = SPLIT[args.split]

    print(f"\nSubjects to process: {subjects_to_process}")

    # Base directory for extracted H36M data
    base_directory = os.path.join(root_dir, args.data_path, "H36M", "extracted")

    # Process each subject
    for subject in subjects_to_process:
        # Load camera parameters for this subject
        _, _, projection_matrices_dict = load_camera_parameters(
            camera_parameters_path, subject, args.camera_ids
        )

        # Convert to torch tensors and move to device
        P1 = torch.from_numpy(projection_matrices_dict[args.camera_ids[0]]).to(args.device)
        P2 = torch.from_numpy(projection_matrices_dict[args.camera_ids[1]]).to(args.device)
        projection_matrices = [P1, P2]

        # Create dataset for this subject
        # Determine split for this subject
        subject_split = None
        for split_name, split_subjects in SPLIT.items():
            if subject in split_subjects:
                subject_split = split_name
                break

        if subject_split is None:
            print(f"Warning: Subject {subject} not found in any split. Skipping.")
            continue

        dataset = Human36mDatasetSequenceTwoCameras(
            base_directory=base_directory,
            split=subject_split,
            camera_ids=args.camera_ids
        )

        if len(dataset) == 0:
            print(f"Warning: No data found for subject {subject}. Skipping.")
            continue

        # Preprocess subject
        preprocess_subject(
            subject,
            dataset,
            args.output_dir,
            pose_estimation_jit_fn,
            params,
            batch_stats,
            human_detector,
            device_torch,
            projection_matrices,
            batch_size=args.batch_size,
            device=args.device,
            score_fn=score_fn,
            ood_threshold=args.ood_threshold,
        )

    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print(f"Processed data saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
