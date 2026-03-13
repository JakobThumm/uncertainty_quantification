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
import rclpy.duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import jax.numpy as jnp
import cloudpickle
import zstandard
import tf2_ros
from scipy.spatial.transform import Rotation
from message_filters import Subscriber, ApproximateTimeSynchronizer
from ultralytics import YOLO

from human_pose_pipeline.pose_estimation.inference_helper import (
    initialize_jax_models,
)
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    process_frame_3d_from_rgbd_yolo,
    process_pose_output,
)
from human_pose_pipeline.pose_estimation.triangulation_helper import (
    load_camera_parameters
)
from human_pose_pipeline.motion_prediction.inference_helper import run_motion_prediction
from human_pose_pipeline.pose_estimation.h36m_settings import (
    MIRROR_13_JOINT_MODEL_MAP,
    YOLO_CONFIDENCE_THRESHOLD,
)
from human_pose_pipeline.motion_prediction.rgbd_yolo_settings import (
    INPUT_HORIZON_LENGTH,
    PREDICTION_HORIZON_LENGTH,
    N_JOINTS,
    OOD_THRESHOLD as MOTION_OOD_THRESHOLD,
    N_CORRECT_POSES_REQUIRED,
    COV_CALIBRATION_CT,
    COV_CALIBRATION_IT,
    COV_CALIBRATION_FACTORS,
    SET_LIKELIHOOD
)

# Add the workspace root to the path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Try to import custom messages (will be available after building)
try:
    from uq_msgs.msg import Pose2D, Pose3D, MotionPrediction
except ImportError:
    print("WARNING: uq_msgs not found. Please build the uq_msgs package first.")
    Pose2D = None
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
        self.declare_parameter('yolo_model', 'yolo26n-pose.pt')
        self.declare_parameter('motion_model_path', 'human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle')
        self.declare_parameter('camera_params_path', 'human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420/camera-parameters.json')
        self.declare_parameter('enable_ood', True)
        self.declare_parameter('enable_tracking', False)
        self.declare_parameter('depth_uncertainty', 0.002)
        self.declare_parameter('motion_score_fn_path', 'human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle')
        self.declare_parameter('device', 'cuda')

        # Camera topics (stereo mode)
        self.declare_parameter('camera_1_color_topic', '/realsense/camera_1/color/image_raw')
        self.declare_parameter('camera_2_color_topic', '/realsense/camera_2/color/image_raw')
        self.declare_parameter('camera_1_info_topic', '/realsense/camera_1/color/camera_info')
        self.declare_parameter('camera_2_info_topic', '/realsense/camera_2/color/camera_info')

        # Camera topics (RGB-D mode)
        self.declare_parameter('rgbd_color_topic', 'rgbd_stream/rgb/compressed')
        self.declare_parameter('rgbd_depth_topic', 'rgbd_stream/depth/compressed')
        self.declare_parameter('rgbd_info_topic', '/camera/camera/color/camera_info')

        # TF frames for camera-to-world transform (RGB-D mode)
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('camera_optical_frame', 'camera_depth_optical_frame')

        # Camera IDs for loading calibration
        self.declare_parameter('camera_1_id', '55011271')
        self.declare_parameter('camera_2_id', '60457274')
        self.declare_parameter('subject', 'S1')  # For H36M camera parameters

        # Output topics
        self.declare_parameter('pose_2d_output_topic', '/uq/pose_2d')
        self.declare_parameter('pose_output_topic', '/uq/pose_3d')
        self.declare_parameter('motion_output_topic', '/uq/motion_prediction')

        # Get parameters
        self.mode = self.get_parameter('mode').value
        self.enable_ood = self.get_parameter('enable_ood').value
        self.enable_tracking = self.get_parameter('enable_tracking').value
        self.depth_uncertainty = self.get_parameter('depth_uncertainty').value
        self.device = self.get_parameter('device').value
        self.get_logger().info(f"Launching with Options: mode = {self.mode}, enable_odd = {self.enable_ood}, enable_tracking (YOLO only) = {self.enable_tracking}, depth_uncertainty (RGB-D only) = {self.depth_uncertainty}, device = {self.device}.")

        # Resolve paths relative to workspace root
        self.yolo_model_name = self.get_parameter('yolo_model').value
        self.motion_model_path = os.path.join(workspace_root, self.get_parameter('motion_model_path').value)
        self.camera_params_path = os.path.join(workspace_root, self.get_parameter('camera_params_path').value)
        self.motion_score_fn_path = os.path.join(workspace_root, self.get_parameter('motion_score_fn_path').value)
        self.get_logger().info(f"Using models: Yolo = {self.yolo_model_name}, Motion Model = {self.motion_model_path}, Motion Score Fn = {self.motion_score_fn_path}.")

        self.world_frame = self.get_parameter('world_frame').value
        self.camera_optical_frame = self.get_parameter('camera_optical_frame').value

        # TF2 buffer and listener for camera-to-world transform
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.R_rect_to_world = None
        self.t_rect_to_world = None
        self.tf_transform_received = False

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
        self.latencies = []  # Store latencies in milliseconds
        self.create_timer(10.0, self.print_statistics)

        self.get_logger().info(f'Pose pipeline node initialized in {self.mode} mode')

    def _initialize_models(self):
        """Initialize YOLO pose model and JAX motion prediction model."""
        # Initialize YOLO pose estimation model
        self.get_logger().info(f'Loading YOLO pose model: {self.yolo_model_name}')
        self.yolo_model = YOLO(self.yolo_model_name)
        if self.device == 'cuda':
            self.yolo_model.to('cuda')
            self.get_logger().info(f'YOLO model loaded on CUDA (GPU: {torch.cuda.get_device_name(0)})')
        else:
            self.get_logger().info('YOLO model loaded on CPU')

        # Initialize motion prediction model
        self.get_logger().info(f'Loading motion model from: {self.motion_model_path}')
        self.motion_prediction_jit_fn, self.motion_prediction_params, self.motion_prediction_batch_stats = \
            initialize_jax_models(self.motion_model_path)

        # Load OOD score functions
        self.motion_ood_score_fn = None

        if self.enable_ood:
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
        """Setup subscribers for RGB-D camera mode (compressed topics)."""
        color_topic = self.get_parameter('rgbd_color_topic').value
        depth_topic = self.get_parameter('rgbd_depth_topic').value
        info_topic = self.get_parameter('rgbd_info_topic').value

        self.get_logger().info('Setting up RGB-D subscribers:')
        self.get_logger().info(f'  Color (compressed): {color_topic}')
        self.get_logger().info(f'  Depth (compressed): {depth_topic}')
        self.get_logger().info(f'  Camera Info: {info_topic}')

        # Subscribe to camera info to get intrinsics
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            info_topic,
            self.camera_info_callback,
            qos_profile=self.sensor_qos
        )

        # Create synchronized subscribers for compressed color and depth
        self.color_sub = Subscriber(self, CompressedImage, color_topic, 10)  # , qos_profile=self.sensor_qos)
        self.depth_sub = Subscriber(self, CompressedImage, depth_topic, 10)  # , qos_profile=self.sensor_qos)

        # Synchronize messages
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=1,
            slop=0.05
        )
        self.sync.registerCallback(self.rgbd_callback)

    def _setup_publishers(self):
        """Setup publishers for pose and motion predictions."""
        pose_2d_topic = self.get_parameter('pose_2d_output_topic').value
        pose_topic = self.get_parameter('pose_output_topic').value
        motion_topic = self.get_parameter('motion_output_topic').value

        self.get_logger().info('Setting up publishers:')
        self.get_logger().info(f'  Pose 2D: {pose_2d_topic}')
        self.get_logger().info(f'  Pose 3D: {pose_topic}')
        self.get_logger().info(f'  Motion: {motion_topic}')

        # Reliable QoS for output topics
        # reliable_qos = QoSProfile(
        #     reliability=ReliabilityPolicy.RELIABLE,
        #     history=HistoryPolicy.KEEP_LAST,
        #     depth=10
        # )

        if Pose2D is not None and Pose3D is not None and MotionPrediction is not None:
            self.pose_2d_publisher = self.create_publisher(Pose2D, pose_2d_topic, 10)  # , self.sensor_qos)
            self.pose_publisher = self.create_publisher(Pose3D, pose_topic, 10)  # , self.sensor_qos)
            self.motion_publisher = self.create_publisher(MotionPrediction, motion_topic, 10)  # , self.sensor_qos)
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

    def _lookup_camera_transform(self):
        """Look up the stationary camera-to-world transform from TF2 (cached after first success)."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_optical_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            q = tf.transform.rotation
            t = tf.transform.translation
            self.R_rect_to_world = Rotation.from_quat(
                [q.x, q.y, q.z, q.w]
            ).as_matrix().astype(np.float32)
            self.t_rect_to_world = np.array(
                [t.x, t.y, t.z], dtype=np.float32
            )
            self.tf_transform_received = True
            self.get_logger().info(
                f'Camera transform {self.camera_optical_frame} → {self.world_frame}: '
                f't=[{t.x:.3f}, {t.y:.3f}, {t.z:.3f}] m'
            )
        except tf2_ros.TransformException as e:
            self.get_logger().warn(
                f'TF lookup {self.camera_optical_frame} → {self.world_frame} failed: {e}',
                throttle_duration_sec=2.0,
            )

    def rgbd_callback(self, color_msg, depth_msg):
        """Process synchronized compressed RGB-D images."""
        # Check if we have intrinsics
        if not self.intrinsics_received:
            self.get_logger().warn(
                'Waiting for camera intrinsics...',
                throttle_duration_sec=2.0
            )
            return

        # Lazily look up the stationary camera transform (cached after first success)
        if not self.tf_transform_received:
            self._lookup_camera_transform()
            if not self.tf_transform_received:
                self.get_logger().warn(
                    'Waiting for camera TF transform...',
                    throttle_duration_sec=2.0,
                )
                return

        try:
            # Decompress RGB: JPEG → BGR → RGB
            np_arr = np.frombuffer(color_msg.data, np.uint8)
            bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr_img is None:
                self.get_logger().error('Failed to decode compressed RGB image', throttle_duration_sec=1.0)
                return
            color_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # Decompress depth: zstd uint16 or PNG fallback
            if depth_msg.format.startswith('zstd_16UC1:'):
                dims = depth_msg.format.split(':')[1]
                h, w = map(int, dims.split('x'))
                raw = zstandard.ZstdDecompressor().decompress(bytes(depth_msg.data))
                depth_img = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
            else:
                np_arr = np.frombuffer(depth_msg.data, np.uint8)
                depth_img = cv2.imdecode(np_arr, cv2.IMREAD_ANYDEPTH)

            if depth_img is None:
                self.get_logger().error('Failed to decode compressed depth image', throttle_duration_sec=1.0)
                return

            self._process_frames_rgbd([color_img], [depth_img], color_msg.header)

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
        # Perform YOLO pose estimation and 3D depth lifting
        points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected, keypoints_2d, uncertainties_2d, covariance_xy = \
            process_frame_3d_from_rgbd_yolo(
                rgb_frames=rgb_frames,
                depth_frames=depth_frames,
                camera_intrinsics=self.camera_intrinsics,
                yolo_pose_model=self.yolo_model,
                mirror_map=MIRROR_13_JOINT_MODEL_MAP,
                enable_tracking=self.enable_tracking,
                confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
                verbose=False,
                device=self.device,
                depth_uncertainty=self.depth_uncertainty,
                R_rect_to_world=self.R_rect_to_world,
                t_rect_to_world=self.t_rect_to_world,
            )

        # Process the results through common pipeline
        self._process_pose_results(points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected, keypoints_2d, uncertainties_2d, covariance_xy, header)

    def _process_frames(self, frames, header):
        """
        Process stereo frames through the full pipeline.

        Args:
            frames: List of images [img1, img2] for stereo mode
            header: ROS message header for timestamp
        """
        raise NotImplementedError('Stereo mode is not supported with the YOLO pipeline. Use rgbd mode.')

    def _process_pose_results(self, points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected, keypoints_2d, uncertainties_2d, covariance_xy, header):
        """
        Common pipeline for processing pose estimation results.

        Args:
            points_3d: 3D joint positions (batched)
            C_3d_all: 3D covariance matrices (batched)
            pose_ood_score: OOD score for the pose
            pose_is_ood: Whether the pose is OOD
            human_detected: Whether a human was detected
            keypoints_2d: 2D joint positions (batched)
            uncertainties_2d: 2D uncertainties (batched)
            covariance_xy: 2D covariance (batched)
            header: ROS message header for timestamp
        """
        # Unbatch, compute validity, update pose buffers
        self.points_3d_buffer, self.covariance_buffer, self.pose_valid_buffer, \
            points_3d, C_3d_all, is_valid, pose_buffer_good = process_pose_output(
                points_3d=points_3d,
                C_3d_all=C_3d_all,
                pose_is_ood=pose_is_ood,
                human_detected=human_detected,
                points_3d_buffer=self.points_3d_buffer,
                covariance_buffer=self.covariance_buffer,
                pose_valid_buffer=self.pose_valid_buffer,
                motion_prediction_buffer=self.motion_prediction_buffer,
                motion_uncertainty_buffer=self.motion_uncertainty_buffer,
            )
        pose_is_ood = bool(pose_is_ood)
        human_detected = bool(human_detected)

        # Publish 2D and 3D poses (using unbatched keypoints)
        self._publish_pose_2d(keypoints_2d[0], uncertainties_2d[0], covariance_xy[0], pose_ood_score, pose_is_ood, human_detected, header)
        self._publish_pose(points_3d, C_3d_all, pose_ood_score, pose_is_ood, human_detected, header)

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
            self.motion_prediction_buffer, self.motion_uncertainty_buffer, motion_set_radius, \
                motion_ood_score, motion_is_ood, valid_motion, _, _, _ = run_motion_prediction(
                    points_3d_buffer=self.points_3d_buffer,
                    covariance_buffer=self.covariance_buffer,
                    pose_valid_buffer=self.pose_valid_buffer,
                    motion_prediction_buffer=self.motion_prediction_buffer,
                    motion_uncertainty_buffer=self.motion_uncertainty_buffer,
                    motion_prediction_jit_fn=self.motion_prediction_jit_fn,
                    motion_prediction_params=self.motion_prediction_params,
                    motion_prediction_batch_stats=self.motion_prediction_batch_stats,
                    motion_ood_score_fn=self.motion_ood_score_fn,
                    n_joints=N_JOINTS,
                    input_horizon_length=INPUT_HORIZON_LENGTH,
                    prediction_horizon_length=PREDICTION_HORIZON_LENGTH,
                    ood_threshold=MOTION_OOD_THRESHOLD,
                    calibration_ct=COV_CALIBRATION_CT,
                    calibration_it=COV_CALIBRATION_IT,
                    calibration_factors=COV_CALIBRATION_FACTORS,
                    n_correct_poses_required=N_CORRECT_POSES_REQUIRED,
                    set_likelihood=SET_LIKELIHOOD,
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

    def _publish_pose_2d(self, keypoints_2d, uncertainties_2d, covariance_xy, ood_score, is_ood, human_detected, header):
        """Publish 2D pose with uncertainty."""
        if Pose2D is None:
            return

        msg = Pose2D()
        msg.header = header
        msg.header.frame_id = 'camera_color_optical_frame'  # Image frame

        # Convert tensors to numpy and flatten
        keypoints_np = keypoints_2d.cpu().numpy().flatten().tolist()
        uncertainties_np = uncertainties_2d.cpu().numpy().flatten().tolist()
        covariance_np = covariance_xy.cpu().numpy().flatten().tolist()

        msg.keypoints_2d = keypoints_np
        msg.uncertainties_2d = uncertainties_np
        msg.covariance_xy = covariance_np
        msg.n_joints = N_JOINTS
        msg.is_ood = is_ood
        msg.ood_score = float(ood_score)
        msg.human_detected = human_detected

        self.pose_2d_publisher.publish(msg)

        # Calculate latency: time from image capture to pose publication
        current_time = self.get_clock().now()
        input_time = rclpy.time.Time.from_msg(header.stamp)
        latency_ns = (current_time - input_time).nanoseconds
        latency_ms = latency_ns / 1e6  # Convert to milliseconds

        # Store latency for statistics
        self.latencies.append(latency_ms)
        # Keep only last 1000 measurements to avoid memory growth
        if len(self.latencies) > 1000:
            self.latencies.pop(0)

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
        stats_msg = f'Frames processed: {self.frames_processed}'

        if self.latencies:
            avg_latency = sum(self.latencies) / len(self.latencies)
            min_latency = min(self.latencies)
            max_latency = max(self.latencies)
            stats_msg += f' | Latency (ms): avg={avg_latency:.1f}, min={min_latency:.1f}, max={max_latency:.1f}'

        self.get_logger().info(stats_msg)


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
