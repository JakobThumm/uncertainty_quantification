"""This script preprocesses the data to be used for the motion prediction with uncertainty input.

The model input is the predicted 3D human pose with covariance matrix.
The model output is the predicted 3D human pose in the next 10 timesteps.
The ground truth future human pose is available in datasets/H36M/pre_processed/{subject}/PreprocessedPoses.
However, as the model input should be uncertain, we have to use the 3D human pose estimation
  (see: human_pose_pipeline/examples/pose_estimation_3D.py) to create the model input.

This script loads the preprocessed images in datasets/H36M/pre_processed/{subject}/PreprocessedImages,
  predicts the human pose and uncertainty covariance matrices, and saves the data to datasets/H36M/pre_processed_motion.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import cv2

from human_pose_pipeline.utils.transform_utils import transform_predictions_to_original_space
from src.ood_scores.lm_lanczos import load_score_functions
from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
    joint_mapping,
    predict_pose,
    process_frame_2d
)
from human_pose_pipeline.pose_estimation.triangulation_helper import (
    load_camera_parameters,
    create_joint_covariance,
    triangulate_points_with_covariance,
)
from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
)
from human_pose_pipeline.motion_prediction.h36m_settings import (
    N_JOINTS,
    INPUT_HORIZON_LENGTH,
    PREDICTION_HORIZON_LENGTH,
)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# Dataset splits matching original H36M
SPLIT = {
    'train': ['S1', 'S6', 'S7', 'S8', 'S9'],
    'validation': ['S11'],
    'test': ['S5']
}

# Camera IDs for stereo setup
CAMERA_IDS = ['55011271', '60457274']


def process_sequence(
    images_cam1,
    images_cam2,
    information_cam1,
    information_cam2,
    pose_estimation_jit_fn,
    params,
    batch_stats,
    human_detector,
    device_torch,
    projection_matrices,
):
    """Process a sequence of image pairs to extract 3D poses with covariances.

    Args:
        images_cam1: Array of images from camera 1 (num_frames, H, W, C)
        images_cam2: Array of images from camera 2 (num_frames, H, W, C)
        information_cam1: includes all pose and transformation information from camera 1
        information_cam2: includes all pose and transformation information from camera 2
        pose_estimation_jit_fn: JAX pose estimation function
        params: Model parameters
        batch_stats: Batch statistics
        human_detector: YOLO detector
        device_torch: PyTorch device
        projection_matrices: Dict with projection matrices for both cameras

    Returns:
        poses_3d: Array of 3D poses (num_frames, 13, 3)
        covariances_3d: Array of 3D covariances (num_frames, 13, 3, 3)
        valid_mask: Boolean mask indicating which frames have valid detections
    """
    num_frames = len(images_cam1)
    poses_3d = []
    covariances_3d = []
    valid_mask = []

    for frame_idx in range(num_frames):
        frame_cam1 = images_cam1[frame_idx]
        frame_cam2 = images_cam2[frame_idx]

        # Process both camera views
        uncertainties_cam1 = None
        uncertainties_cam2 = None
        cov_cam1 = None
        cov_cam2 = None

        for cam_idx, frame in enumerate([frame_cam1, frame_cam2]):
            # Get pose estimations using JAX model
            if cam_idx == 0:
                information = information_cam1
            else:
                information = information_cam2
            pred_joints_13, uncertainties_13, covariance_13 = predict_pose(
                np.reshape(frame.copy(), [1, frame.shape[0], frame.shape[1], frame.shape[2]]),  # Batch size of 1
                pose_estimation_jit_fn,
                params,
                batch_stats,
                13
            )
            result = transform_predictions_to_original_space(
                pred_joints_13, information['trans'][frame_idx],
                information['scale_factors'][frame_idx][0],
                information['scale_factors'][frame_idx][1],
                uncertainties=uncertainties_13,
                covariance=covariance_13
            )
            pose = joint_mapping(np.array(result['keypoints']), MIRROR_13_JOINT_MODEL_MAP)
            uncertainties = joint_mapping(np.array(result['uncertainties']), MIRROR_13_JOINT_MODEL_MAP)
            covariance_factor = joint_mapping(np.array(result['covariance']), MIRROR_13_JOINT_MODEL_MAP)
            # Construct per-joint 2x2 covariance matrices
            covariance_matrix = np.zeros((13, 2, 2))
            for j in range(13):
                covariance_matrix[j] = [
                    [float(uncertainties[j, 0])**2, float(covariance_factor[j])],
                    [float(covariance_factor[j]), float(uncertainties[j, 1])**2]
                ]

            if cam_idx == 0:
                poses_cam1 = pose
                uncertainties_cam1 = uncertainties
                cov_cam1 = covariance_matrix
            else:
                poses_cam2 = pose
                uncertainties_cam2 = uncertainties
                cov_cam2 = covariance_matrix

        # Triangulate 3D points if both poses are available
        if poses_cam1 is not None and poses_cam2 is not None:
            # Create joint covariance matrices
            C_joint_list = []
            for i in range(N_JOINTS):
                C_joint = create_joint_covariance(
                    mapped_uncertainty_cam1=uncertainties_cam1[i],
                    mapped_covariance_cam1=cov_cam1[i, 0, 1],
                    mapped_uncertainty_cam2=uncertainties_cam2[i],
                    mapped_covariance_cam2=cov_cam2[i, 0, 1],
                    cross_covariance=np.zeros((2, 2))  # Assume zero cross-covariance
                )
                C_joint_list.append(C_joint)

            P1 = projection_matrices[CAMERA_IDS[0]]
            P2 = projection_matrices[CAMERA_IDS[1]]
            points_3d, C_3d_all = triangulate_points_with_covariance(
                poses_cam1, poses_cam2, P1, P2, C_joint_list
            )

            poses_3d.append(points_3d)
            covariances_3d.append(C_3d_all)
            valid_mask.append(True)
        else:
            # No valid detection - create zeros
            poses_3d.append(np.zeros((N_JOINTS, 3)))
            covariances_3d.append(np.zeros((N_JOINTS, 3, 3)))
            valid_mask.append(False)

    return np.array(poses_3d), np.array(covariances_3d), np.array(valid_mask)


def preprocess_subject(
    subject,
    preprocessed_dir,
    output_dir,
    pose_estimation_jit_fn,
    params,
    batch_stats,
    human_detector,
    device_torch,
    projection_matrices,
):
    """Preprocess all sequences for a given subject.

    Args:
        subject: Subject ID (e.g., 'S1')
        preprocessed_dir: Path to preprocessed images directory
        output_dir: Path to save processed motion data
        pose_estimation_jit_fn: JAX pose estimation function
        params: Model parameters
        batch_stats: Batch statistics
        human_detector: YOLO detector
        device_torch: PyTorch device
        projection_matrices: Dict with projection matrices
    """
    subject_image_dir = os.path.join(preprocessed_dir, subject, 'PreprocessedImages')
    subject_pose_dir = os.path.join(preprocessed_dir, subject, 'PreprocessedPoses')
    subject_output_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_output_dir, exist_ok=True)

    if not os.path.exists(subject_image_dir):
        print(f"Warning: Preprocessed directory not found for {subject}: {subject_image_dir}")
        return

    # Find all preprocessed image files
    image_files = sorted([f for f in os.listdir(subject_image_dir) if f.endswith('.npy')])

    print(f"\nProcessing subject {subject}: {len(image_files)} sequences")

    for img_file in tqdm(image_files, desc=f"Processing {subject}"):
        # Load images for both cameras
        # Expected format: action.camera1.npy and action.camera2.npy
        base_name = img_file.replace('.npy', '')
        parts = base_name.split('.')

        if len(parts) < 2:
            print(f"Warning: Unexpected filename format: {img_file}")
            continue

        # Determine which camera this file is from
        camera_id = parts[-1]
        action_name = '.'.join(parts[:-1])

        # We only process when we have both camera views
        if camera_id != CAMERA_IDS[0]:
            continue

        # Load images from both cameras
        img_path_cam1 = os.path.join(subject_image_dir, f"{action_name}.{CAMERA_IDS[0]}.npy")
        img_path_cam2 = os.path.join(subject_image_dir, f"{action_name}.{CAMERA_IDS[1]}.npy")
        poses_path_cam1 = os.path.join(subject_pose_dir, f"{action_name}.{CAMERA_IDS[0]}.npz")
        poses_path_cam2 = os.path.join(subject_pose_dir, f"{action_name}.{CAMERA_IDS[1]}.npz")

        if not os.path.exists(img_path_cam1) or not os.path.exists(img_path_cam2):
            print(f"Warning: Missing camera view for {action_name}")
            continue

        # Load images
        images_cam1 = np.load(img_path_cam1)  # Shape: (num_frames, H, W, C)
        images_cam2 = np.load(img_path_cam2)
        information_cam1 = np.load(poses_path_cam1)
        information_cam2 = np.load(poses_path_cam2)
        if len(images_cam1) != len(images_cam2):
            print(f"Warning: Frame count mismatch for {action_name}")
            continue

        # Convert images from (H, W, C) to RGB if needed
        if images_cam1.shape[-1] == 3:
            # Assuming images are in RGB format already
            pass

        # Process the sequence
        poses_3d, covariances_3d, valid_mask = process_sequence(
            images_cam1,
            images_cam2,
            information_cam1,
            information_cam2,
            pose_estimation_jit_fn,
            params,
            batch_stats,
            human_detector,
            device_torch,
            projection_matrices,
        )

        # Save the processed data
        output_path = os.path.join(subject_output_dir, f"{action_name}.npz")
        np.savez_compressed(
            output_path,
            poses_3d=poses_3d,
            covariances_3d=covariances_3d,
            valid_mask=valid_mask,
        )

        valid_count = valid_mask.sum()
        print(f"  {action_name}: {valid_count}/{len(valid_mask)} valid frames")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess H36M data for motion prediction with uncertainty'
    )
    parser.add_argument(
        '--preprocessed_dir',
        type=str,
        default='datasets/H36M/pre_processed',
        help='Path to preprocessed H36M images directory'
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

    args = parser.parse_args()

    print("=" * 80)
    print("H36M Motion Prediction Dataset Preprocessing")
    print("=" * 80)
    print(f"Preprocessed images directory: {args.preprocessed_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.run_name}")
    print("=" * 80)

    # Initialize models
    print("\nInitializing models...")
    models_dir = os.path.join(root_dir, args.model_save_path, "H36M", "RegressFlow", "seed_420")
    checkpoint_path_jax = os.path.join(models_dir, args.run_name)

    pose_estimation_jit_fn, params, batch_stats = initialize_jax_models(checkpoint_path_jax)
    print("Pose estimation model loaded")

    human_detector, device_torch = initialize_human_detector('cuda')
    print("Human detector loaded")

    # Load camera parameters
    camera_parameters_path = os.path.join(models_dir, 'camera-parameters.json')
    if not os.path.exists(camera_parameters_path):
        raise FileNotFoundError(
            f"Camera parameters file not found at {camera_parameters_path}. "
            "Please ensure the camera-parameters.json file is available in the models directory."
        )

    # We'll load camera parameters for each subject separately since they may differ
    # For now, we'll use a default subject to get the structure
    # In practice, you may need to load per-subject camera parameters

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

    # Process each subject
    for subject in subjects_to_process:
        # Load camera parameters for this subject
        intrinsics, extrinsics = load_camera_parameters(
            camera_parameters_path, subject, CAMERA_IDS
        )

        # Compute projection matrices
        projection_matrices = {}
        for cam_id in CAMERA_IDS:
            K = intrinsics[cam_id]
            RT = extrinsics[cam_id]
            projection_matrices[cam_id] = K @ RT  # P = K[R|t]

        # Preprocess subject
        preprocess_subject(
            subject,
            args.preprocessed_dir,
            args.output_dir,
            pose_estimation_jit_fn,
            params,
            batch_stats,
            human_detector,
            device_torch,
            projection_matrices,
        )

    print("\n" + "=" * 80)
    print("Preprocessing complete!")
    print(f"Processed data saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
