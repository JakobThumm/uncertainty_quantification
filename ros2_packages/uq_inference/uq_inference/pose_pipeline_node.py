#!/usr/bin/env python3
"""
ROS2 node for real-time human pose estimation and motion prediction.

This node:
1. Subscribes to RGB(-D) image topics from RealSense camera(s)
2. Performs 2D pose estimation with uncertainty quantification
3. Computes 3D poses via triangulation (stereo) or depth lifting (RGB-D)
4. Predicts future motion with uncertainty
5. Publishes estimated poses and predicted motions
"""

import os
import sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import torch
import jax.numpy as jnp
import cloudpickle
from message_filters import Subscriber, ApproximateTimeSynchronizer

from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
    initialize_human_detector,
)
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d,
    process_frame_3d_from_rgbd,
    fill_pose_buffer,
    update_motion_prediction_buffer
)
from human_pose_pipeline.pose_estimation.triangulation_helper import (
    load_camera_parameters
)
from human_pose_pipeline.motion_prediction.inference_helper import calibrate_covariance_matrices
from human_pose_pipeline.utils.eval_utils import convert_covariance_matrices_to_set
from src.ood_scores.lm_lanczos import load_score_functions

from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
    OOD_THRESHOLD as POSE_OOD_THRESHOLD,
)
from human_pose_pipeline.motion_prediction.h36m_settings import (
    INPUT_HORIZON_LENGTH,
    PREDICTION_HORIZON_LENGTH,
    N_JOINTS,
    OOD_THRESHOLD as MOTION_OOD_THRESHOLD,
    N_CORRECT_POSES_REQUIRED,
    COV_CALIBRATION_CT,
    COV_CALIBRATION_IT,
    COV_CALIBRATION_HF,
    COV_CALIBRATION_FF,
    COV_CALIBRATION_HI,
    COV_CALIBRATION_FI,
    SET_LIKELIHOOD
)

# Add the workspace root to the path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Try to import custom messages (will be available after building)
try:
    from uq_msgs.msg import Pose3D, MotionPrediction
except ImportError:
    print("WARNING: uq_msgs not found. Please build the uq_msgs package first.")
    Pose3D = None
    MotionPrediction = None


class PosePipelineNode(Node):
    """
    ROS2 node for real-time human pose estimation and motion prediction.

    Supports two modes:
    1. Stereo mode: Uses two RGB cameras for triangulation
    2. RGB-D mode: Uses single RGB-D camera with depth for 3D lifting
    """

    def __init__(self):
        super().__init__('pose_pipeline_node')

        # Declare parameters
        self.declare_parameter('mode', 'rgbd')  # 'stereo' or 'rgbd'
        self.declare_parameter('pose_model_path', 'human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420/jax_resnet50_regressflow')
        self.declare_parameter('motion_model_path', 'human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle')
        self.declare_parameter('camera_params_path', 'human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420/camera-parameters.json')
        self.declare_parameter('enable_ood', True)
        self.declare_parameter('pose_base_key', '')
        self.declare_parameter('motion_score_fn_path', 'human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle')
        self.declare_parameter('cache_dir', 'cache/')
        self.declare_parameter('device', 'cuda')

        # Camera topics (stereo mode)
        self.declare_parameter('camera_1_color_topic', '/realsense/camera_1/color/image_raw')
        self.declare_parameter('camera_2_color_topic', '/realsense/camera_2/color/image_raw')
        self.declare_parameter('camera_1_info_topic', '/realsense/camera_1/color/camera_info')
        self.declare_parameter('camera_2_info_topic', '/realsense/camera_2/color/camera_info')

        # Camera topics (RGB-D mode)
        self.declare_parameter('rgbd_color_topic', '/realsense/camera_1/color/image_raw')
        self.declare_parameter('rgbd_depth_topic', '/realsense/camera_1/aligned_depth_to_color/image_raw')
        self.declare_parameter('rgbd_info_topic', '/realsense/camera_1/color/camera_info')

        # Camera IDs for loading calibration
        self.declare_parameter('camera_1_id', '55011271')
        self.declare_parameter('camera_2_id', '60457274')
        self.declare_parameter('subject', 'S1')  # For H36M camera parameters

        # Output topics
        self.declare_parameter('pose_output_topic', '/uq/pose_3d')
        self.declare_parameter('motion_output_topic', '/uq/motion_prediction')

        # Get parameters
        self.mode = self.get_parameter('mode').value
        self.enable_ood = self.get_parameter('enable_ood').value
        self.device = self.get_parameter('device').value

        # Resolve paths relative to workspace root
        self.pose_model_path = os.path.join(workspace_root, self.get_parameter('pose_model_path').value)
        self.motion_model_path = os.path.join(workspace_root, self.get_parameter('motion_model_path').value)
        self.camera_params_path = os.path.join(workspace_root, self.get_parameter('camera_params_path').value)
        self.motion_score_fn_path = os.path.join(workspace_root, self.get_parameter('motion_score_fn_path').value)
        self.cache_dir = os.path.join(workspace_root, self.get_parameter('cache_dir').value)
        self.pose_base_key = self.get_parameter('pose_base_key').value

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Create SensorDataQoS profile with depth 1
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # Initialize models
        self.get_logger().info('Initializing models...')
        self._initialize_models()

        # Initialize pose and motion buffers
        self.points_3d_buffer = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3])
        self.covariance_buffer = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3, 3])
        self.pose_valid_buffer = jnp.zeros([INPUT_HORIZON_LENGTH])
        self.motion_prediction_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3])
        self.motion_uncertainty_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3, 3])
        self.frame_counter = 0

        # Camera intrinsics (for RGB-D mode)
        self.camera_intrinsics = None
        self.intrinsics_received = False

        # Setup subscribers based on mode
        if self.mode == 'stereo':
            self._setup_stereo_subscribers()
        elif self.mode == 'rgbd':
            self._setup_rgbd_subscribers()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Must be 'stereo' or 'rgbd'")

        # Setup publishers
        self._setup_publishers()

        # Statistics
        self.frames_processed = 0
        self.create_timer(10.0, self.print_statistics)

        self.get_logger().info(f'Pose pipeline node initialized in {self.mode} mode')

    def _initialize_models(self):
        """Initialize JAX models and human detector."""
        # Initialize pose estimation model
        self.get_logger().info(f'Loading pose model from: {self.pose_model_path}')
        self.pose_estimation_jit_fn, self.pose_estimation_params, self.pose_estimation_batch_stats = \
            initialize_jax_models(self.pose_model_path)

        # Initialize motion prediction model
        self.get_logger().info(f'Loading motion model from: {self.motion_model_path}')
        self.motion_prediction_jit_fn, self.motion_prediction_params, self.motion_prediction_batch_stats = \
            initialize_jax_models(self.motion_model_path)

        # Initialize YOLO human detector
        self.get_logger().info('Loading YOLO human detector...')
        self.human_detector, self.device_torch = initialize_human_detector('cuda' if self.device == 'cuda' else 'cpu')

        # Load OOD score functions
        self.pose_ood_score_fn = None
        self.motion_ood_score_fn = None

        if self.enable_ood:
            if self.pose_base_key:
                self.get_logger().info(f'Loading pose OOD score functions with key: {self.pose_base_key}')
                self.pose_ood_score_fn, _, _, _ = load_score_functions(self.cache_dir, self.pose_base_key)
            else:
                self.get_logger().warn('OOD enabled but no pose_base_key provided. Skipping pose OOD detection.')

            if os.path.exists(self.motion_score_fn_path):
                self.get_logger().info(f'Loading motion OOD score function from: {self.motion_score_fn_path}')
                with open(self.motion_score_fn_path, 'rb') as f:
                    motion_score_data = cloudpickle.load(f)
                    self.motion_ood_score_fn = motion_score_data['score_fun']
            else:
                self.get_logger().warn(f'Motion score function file not found: {self.motion_score_fn_path}')

        # Load camera parameters (for stereo mode)
        if self.mode == 'stereo':
            camera_1_id = self.get_parameter('camera_1_id').value
            camera_2_id = self.get_parameter('camera_2_id').value
            subject = self.get_parameter('subject').value
            self.camera_ids = [camera_1_id, camera_2_id]

            self.get_logger().info(f'Loading camera parameters for subject {subject}, cameras {camera_1_id}, {camera_2_id}')
            intrinsics, extrinsics, projection_matrices = load_camera_parameters(
                self.camera_params_path, subject, self.camera_ids
            )

            P1 = projection_matrices[camera_1_id]
            P2 = projection_matrices[camera_2_id]
            self.projection_matrices = [
                torch.from_numpy(P1).to(self.device),
                torch.from_numpy(P2).to(self.device)
            ]
            self.get_logger().info('Camera parameters loaded successfully')

        self.get_logger().info('All models initialized successfully!')

    def _setup_stereo_subscribers(self):
        """Setup subscribers for stereo camera mode."""
        camera_1_topic = self.get_parameter('camera_1_color_topic').value
        camera_2_topic = self.get_parameter('camera_2_color_topic').value

        self.get_logger().info(f'Setting up stereo subscribers:')
        self.get_logger().info(f'  Camera 1: {camera_1_topic}')
        self.get_logger().info(f'  Camera 2: {camera_2_topic}')

        # Create synchronized subscribers for both cameras
        self.camera_1_sub = Subscriber(self, Image, camera_1_topic, qos_profile=self.sensor_qos)
        self.camera_2_sub = Subscriber(self, Image, camera_2_topic, qos_profile=self.sensor_qos)

        # Synchronize messages with approximate time sync
        self.sync = ApproximateTimeSynchronizer(
            [self.camera_1_sub, self.camera_2_sub],
            queue_size=1,
            slop=0.05  # 50ms tolerance
        )
        self.sync.registerCallback(self.stereo_callback)

    def _setup_rgbd_subscribers(self):
        """Setup subscribers for RGB-D camera mode."""
        color_topic = self.get_parameter('rgbd_color_topic').value
        depth_topic = self.get_parameter('rgbd_depth_topic').value
        info_topic = self.get_parameter('rgbd_info_topic').value

        self.get_logger().info('Setting up RGB-D subscribers:')
        self.get_logger().info(f'  Color: {color_topic}')
        self.get_logger().info(f'  Depth: {depth_topic}')
        self.get_logger().info(f'  Camera Info: {info_topic}')

        # Subscribe to camera info to get intrinsics
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            info_topic,
            self.camera_info_callback,
            qos_profile=self.sensor_qos
        )

        # Create synchronized subscribers for color and depth
        self.color_sub = Subscriber(self, Image, color_topic, qos_profile=self.sensor_qos)
        self.depth_sub = Subscriber(self, Image, depth_topic, qos_profile=self.sensor_qos)

        # Synchronize messages
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=1,
            slop=0.05
        )
        self.sync.registerCallback(self.rgbd_callback)

    def _setup_publishers(self):
        """Setup publishers for pose and motion predictions."""
        pose_topic = self.get_parameter('pose_output_topic').value
        motion_topic = self.get_parameter('motion_output_topic').value

        self.get_logger().info('Setting up publishers:')
        self.get_logger().info(f'  Pose: {pose_topic}')
        self.get_logger().info(f'  Motion: {motion_topic}')

        # Reliable QoS for output topics
        # reliable_qos = QoSProfile(
        #     reliability=ReliabilityPolicy.RELIABLE,
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=10
        # )

        if Pose3D is not None and MotionPrediction is not None:
            self.pose_publisher = self.create_publisher(Pose3D, pose_topic, self.sensor_qos)
            self.motion_publisher = self.create_publisher(MotionPrediction, motion_topic, self.sensor_qos)
        else:
            self.get_logger().error('Custom messages not available. Cannot create publishers.')

    def stereo_callback(self, img1_msg, img2_msg):
        """Process synchronized stereo camera images."""
        try:
            # Convert ROS messages to OpenCV images
            img1 = self.bridge.imgmsg_to_cv2(img1_msg, desired_encoding='bgr8')
            img2 = self.bridge.imgmsg_to_cv2(img2_msg, desired_encoding='bgr8')

            # Process frames through the pipeline
            frames = [img1, img2]
            self._process_frames(frames, img1_msg.header)

        except Exception as e:
            self.get_logger().error(f'Error in stereo callback: {e}', throttle_duration_sec=1.0)

    def camera_info_callback(self, msg):
        """Callback to receive and store camera intrinsics."""
        if not self.intrinsics_received:
            # Extract intrinsics from CameraInfo message
            K = msg.k  # Intrinsic matrix (3x3) stored as 9-element array
            self.camera_intrinsics = {
                'fx': K[0],  # K[0, 0]
                'fy': K[4],  # K[1, 1]
                'cx': K[2],  # K[0, 2]
                'cy': K[5],  # K[1, 2]
            }
            self.intrinsics_received = True
            self.get_logger().info(
                f'Camera intrinsics received: fx={self.camera_intrinsics["fx"]:.2f}, '
                f'fy={self.camera_intrinsics["fy"]:.2f}, '
                f'cx={self.camera_intrinsics["cx"]:.2f}, '
                f'cy={self.camera_intrinsics["cy"]:.2f}'
            )

    def rgbd_callback(self, color_msg, depth_msg):
        """Process synchronized RGB-D camera images."""
        # Check if we have intrinsics
        if not self.intrinsics_received:
            self.get_logger().warn(
                'Waiting for camera intrinsics...',
                throttle_duration_sec=2.0
            )
            return

        try:
            # Convert ROS messages to OpenCV images
            color_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            # Depth is typically uint16 in millimeters
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            # Process frames through RGB-D pipeline
            frames = [color_img]
            depth_frames = [depth_img]
            self._process_frames_rgbd(frames, depth_frames, color_msg.header)

        except Exception as e:
            self.get_logger().error(f'Error in RGB-D callback: {e}', throttle_duration_sec=1.0)
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _process_frames_rgbd(self, rgb_frames, depth_frames, header):
        """
        Process RGB-D frames through the full pipeline.

        Args:
            rgb_frames: List of RGB images [img1] for RGB-D mode
            depth_frames: List of depth images [depth1] for RGB-D mode
            header: ROS message header for timestamp
        """
        # Process only every other frame to match motion prediction frequency
        if self.frame_counter % 2 != 0:
            self.frame_counter += 1
            return

        # Perform 2D pose estimation and 3D depth lifting
        points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected = process_frame_3d_from_rgbd(
            rgb_frames=rgb_frames,
            depth_frames=depth_frames,
            camera_intrinsics=self.camera_intrinsics,
            pose_estimation_jit_fn=self.pose_estimation_jit_fn,
            params=self.pose_estimation_params,
            batch_stats=self.pose_estimation_batch_stats,
            human_detector=self.human_detector,
            device_torch=self.device_torch,
            mirror_map=MIRROR_13_JOINT_MODEL_MAP,
            score_fn=self.pose_ood_score_fn,
            human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
            ood_threshold=POSE_OOD_THRESHOLD,
            verbose=False,
            device=self.device
        )

        # Process the results through common pipeline
        self._process_pose_results(points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected, header)

    def _process_frames(self, frames, header):
        """
        Process frames through the full pipeline.

        Args:
            frames: List of images [img1, img2] for stereo mode
            header: ROS message header for timestamp
        """
        # Process only every other frame to match motion prediction frequency
        if self.frame_counter % 2 != 0:
            self.frame_counter += 1
            return

        # Perform 2D pose estimation and 3D triangulation
        points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected = process_frame_3d(
            frames=frames,
            projection_matrices=self.projection_matrices,
            pose_estimation_jit_fn=self.pose_estimation_jit_fn,
            params=self.pose_estimation_params,
            batch_stats=self.pose_estimation_batch_stats,
            human_detector=self.human_detector,
            device_torch=self.device_torch,
            mirror_map=MIRROR_13_JOINT_MODEL_MAP,
            score_fn=self.pose_ood_score_fn,
            human_detection_threshold=YOLO_CONFIDENCE_THRESHOLD,
            ood_threshold=POSE_OOD_THRESHOLD,
            verbose=False,
            device=self.device
        )

        # Process the results through common pipeline
        self._process_pose_results(points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected, header)

    def _process_pose_results(self, points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected, header):
        """
        Common pipeline for processing pose estimation results.

        Args:
            points_3d: 3D joint positions (batched)
            C_3d_all: 3D covariance matrices (batched)
            pose_ood_score: OOD score for the pose
            pose_is_ood: Whether the pose is OOD
            human_detected: Whether a human was detected
            header: ROS message header for timestamp
        """
        # Remove batch dimension
        points_3d = points_3d[0]
        C_3d_all = C_3d_all[0]

        # Publish pose
        self._publish_pose(points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected, header)

        # Valid prediction if not OOD and human detected
        is_valid = (not pose_is_ood) and human_detected

        # Update pose buffer
        self.points_3d_buffer, self.covariance_buffer, self.pose_valid_buffer, pose_buffer_good = fill_pose_buffer(
            points_3d_buffer=self.points_3d_buffer,
            covariance_buffer=self.covariance_buffer,
            pose_valid_buffer=self.pose_valid_buffer,
            points_3d=jnp.array(points_3d.cpu().numpy()),
            covariance=jnp.array(C_3d_all.cpu().numpy()),
            is_valid=is_valid,
            motion_prediction_buffer=self.motion_prediction_buffer,
            motion_uncertainty_buffer=self.motion_uncertainty_buffer,
        )

        # Predict and publish motion if enough poses are in buffer
        self._predict_and_publish_motion(pose_buffer_good, header)

        self.frame_counter += 1
        self.frames_processed += 1

    def _predict_and_publish_motion(self, pose_buffer_good, header):
        """
        Predict motion and publish results if buffer has enough poses.

        Args:
            pose_buffer_good: Boolean indicating if pose buffer is ready
            header: ROS message header for timestamp
        """
        if self.frame_counter >= INPUT_HORIZON_LENGTH - 1 and pose_buffer_good:
            pose_input = self.points_3d_buffer.reshape([1, INPUT_HORIZON_LENGTH, N_JOINTS * 3])
            motion_prediction_input = jnp.concatenate([
                pose_input,
                self.covariance_buffer.reshape([1, INPUT_HORIZON_LENGTH, N_JOINTS * 3 * 3])
            ], axis=-1)

            # Model inference
            if self.motion_prediction_batch_stats is not None:
                motion_predicted, (motion_cov_predicted, L) = self.motion_prediction_jit_fn(
                    self.motion_prediction_params,
                    self.motion_prediction_batch_stats,
                    motion_prediction_input
                )
            else:
                motion_predicted, (motion_cov_predicted, L) = self.motion_prediction_jit_fn(
                    self.motion_prediction_params,
                    motion_prediction_input
                )

            # Compute OOD score for motion
            if self.motion_ood_score_fn is not None:
                motion_ood_score = self.motion_ood_score_fn(pose_input)
            else:
                motion_ood_score = 0.0

            motion_predicted = motion_predicted.reshape(-1, PREDICTION_HORIZON_LENGTH, N_JOINTS, 3)[0]
            motion_cov_predicted = motion_cov_predicted[0]

            # Calibrate covariance
            motion_cov_predicted = calibrate_covariance_matrices(
                covariance_matrices=motion_cov_predicted,
                constant_time_factor=COV_CALIBRATION_CT,
                increase_time_factor=COV_CALIBRATION_IT,
                hand_factor=COV_CALIBRATION_HF,
                feet_factor=COV_CALIBRATION_FF,
                hand_indices=COV_CALIBRATION_HI,
                feet_indices=COV_CALIBRATION_FI
            )

            motion_is_ood = bool(motion_ood_score > MOTION_OOD_THRESHOLD)

            # Update motion prediction buffer
            self.motion_prediction_buffer, self.motion_uncertainty_buffer, valid_motion = update_motion_prediction_buffer(
                motion_prediction_buffer=self.motion_prediction_buffer,
                motion_uncertainty_buffer=self.motion_uncertainty_buffer,
                predicted_motion=motion_predicted,
                predicted_motion_uncertainty=motion_cov_predicted,
                is_ood=motion_is_ood,
                pose_valid_buffer=self.pose_valid_buffer,
                n_correct_poses_required=N_CORRECT_POSES_REQUIRED
            )

            # Convert covariance to set radius
            motion_set_radius = convert_covariance_matrices_to_set(
                self.motion_uncertainty_buffer,
                likelihood=SET_LIKELIHOOD
            )

            # Publish motion prediction
            self._publish_motion(
                self.motion_prediction_buffer,
                self.motion_uncertainty_buffer,
                motion_set_radius,
                motion_ood_score,
                motion_is_ood,
                valid_motion,
                header
            )

    def _publish_pose(self, points_3d, covariance_3d, ood_score, is_ood, human_detected, header):
        """Publish 3D pose with uncertainty."""
        if Pose3D is None:
            return

        msg = Pose3D()
        msg.header = header
        msg.header.frame_id = 'world'

        # Convert tensors to numpy and flatten
        points_np = points_3d.cpu().numpy().flatten().tolist()
        cov_np = covariance_3d.cpu().numpy().flatten().tolist()

        msg.points_3d = points_np
        msg.covariance_3d = cov_np
        msg.n_joints = N_JOINTS
        msg.is_ood = is_ood
        msg.ood_score = float(ood_score)
        msg.human_detected = human_detected

        self.pose_publisher.publish(msg)

    def _publish_motion(self, motion_buffer, uncertainty_buffer, set_radius, ood_score, is_ood, is_valid, header):
        """Publish motion prediction with uncertainty."""
        if MotionPrediction is None:
            return

        msg = MotionPrediction()
        msg.header = header
        msg.header.frame_id = 'world'

        # Convert to numpy and flatten
        motion_np = np.array(motion_buffer).flatten().tolist()
        uncertainty_np = np.array(uncertainty_buffer).flatten().tolist()
        radius_np = np.array(set_radius).flatten().tolist()

        msg.motion_predicted = motion_np
        msg.motion_covariance = uncertainty_np
        msg.set_radius = radius_np
        msg.prediction_horizon_length = PREDICTION_HORIZON_LENGTH
        msg.n_joints = N_JOINTS
        msg.is_ood = is_ood
        msg.ood_score = float(ood_score) if isinstance(ood_score, (int, float)) else float(ood_score[0])
        msg.is_valid = is_valid

        self.motion_publisher.publish(msg)

    def print_statistics(self):
        """Print processing statistics."""
        self.get_logger().info(f'Frames processed: {self.frames_processed}')


def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)

    try:
        node = PosePipelineNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
