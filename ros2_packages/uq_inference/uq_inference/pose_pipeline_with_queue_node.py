"""
ROS2 node for real-time human pose estimation and motion prediction with queue-based
batch processing.

This node collects RGB-D frame pairs from an ApproximateTimeSynchronizer into a queue
while the pipeline is busy.  Once the previous inference finishes, the entire accumulated
batch is processed in one shot:
  1. YOLO + depth lifting is called once for all B frames (expensive, batched).
  2. The pose buffer is filled with all B poses in a vectorised operation, producing B
     intermediate buffer states.
  3. The motion prediction model is called once on B buffer states (batched JAX call).
  4. The motion prediction buffer is updated sequentially for each frame.
  5. Results are published one message per frame, preserving original timestamps.

Queue overflow behaviour (queue capped at PREDICTION_HORIZON_LENGTH):
  - At PREDICTION_HORIZON_LENGTH/2 frames:  log a WARNING.
  - At PREDICTION_HORIZON_LENGTH frames:    log an ERROR, reset all pose/motion buffers,
                                             and discard the queued frames.
"""

import os
import sys
import time
import threading
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
    process_pose_output_batched,
    fill_pose_buffer_batched,
)
from human_pose_pipeline.pose_estimation.triangulation_helper import (
    load_camera_parameters,
)
from human_pose_pipeline.motion_prediction.inference_helper import (
    run_motion_prediction_batched,
)
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
    SET_LIKELIHOOD,
)

# Add the workspace root to the path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

try:
    from uq_msgs.msg import Pose2D, Pose3D, MotionPrediction
except ImportError:
    print("WARNING: uq_msgs not found. Please build the uq_msgs package first.")
    Pose2D = None
    Pose3D = None
    MotionPrediction = None

# Maximum queue depth equals the motion prediction horizon so that even a fully-invalid
# batch can be covered by the motion prediction buffer.
_QUEUE_MAX = PREDICTION_HORIZON_LENGTH
_QUEUE_WARN = PREDICTION_HORIZON_LENGTH // 2


class PosePipelineWithQueueNode(Node):
    """
    ROS2 node for real-time human pose estimation and motion prediction.

    RGB-D only.  Uses a dedicated worker thread so that incoming frames are
    buffered while the pose pipeline is busy.
    """

    def __init__(self):
        super().__init__("pose_pipeline_with_queue_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("yolo_model", "yolo26n-pose.pt")
        self.declare_parameter(
            "motion_model_path", "human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle"
        )
        self.declare_parameter("enable_ood", True)
        self.declare_parameter("enable_tracking", False)
        self.declare_parameter("depth_uncertainty", 0.002)
        self.declare_parameter(
            "motion_score_fn_path",
            "human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle",
        )
        self.declare_parameter("device", "cuda")
        self.declare_parameter("stream_reliable", True)

        # Camera topics (RGB-D)
        self.declare_parameter("rgbd_color_topic", "rgbd_stream/rgb/compressed")
        self.declare_parameter("rgbd_depth_topic", "rgbd_stream/depth/compressed")
        self.declare_parameter("rgbd_info_topic", "/camera/camera/color/camera_info")

        # TF frames
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("camera_optical_frame", "camera_depth_optical_frame")

        # Output topics
        self.declare_parameter("pose_2d_output_topic", "/uq/pose_2d")
        self.declare_parameter("pose_output_topic", "/uq/pose_3d")
        self.declare_parameter("motion_output_topic", "/uq/motion_prediction")

        # ── Read parameters ────────────────────────────────────────────────────
        self.enable_ood = self.get_parameter("enable_ood").value
        self.enable_tracking = self.get_parameter("enable_tracking").value
        self.depth_uncertainty = self.get_parameter("depth_uncertainty").value
        self.device = self.get_parameter("device").value
        self.get_logger().info(
            f"Launching with Options: enable_ood={self.enable_ood}, "
            f"enable_tracking={self.enable_tracking}, "
            f"depth_uncertainty={self.depth_uncertainty}, device={self.device}."
        )

        self.yolo_model_name = self.get_parameter("yolo_model").value
        self.motion_model_path = os.path.join(workspace_root, self.get_parameter("motion_model_path").value)
        self.motion_score_fn_path = os.path.join(workspace_root, self.get_parameter("motion_score_fn_path").value)

        self.world_frame = self.get_parameter("world_frame").value
        self.camera_optical_frame = self.get_parameter("camera_optical_frame").value

        # ── TF2 ───────────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.R_rect_to_world = None
        self.t_rect_to_world = None
        self.tf_transform_received = False

        # ── Misc ──────────────────────────────────────────────────────────────
        self.bridge = CvBridge()

        # ── QoS ───────────────────────────────────────────────────────────────
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=PREDICTION_HORIZON_LENGTH,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=PREDICTION_HORIZON_LENGTH,
            durability=DurabilityPolicy.VOLATILE,
        )
        stream_reliable = self.get_parameter("stream_reliable").value
        self.stream_qos = self.reliable_qos if stream_reliable else self.best_effort_qos
        self.get_logger().info(f"Stream QoS: {'Reliable' if stream_reliable else 'Best Effort'} (keep_last={PREDICTION_HORIZON_LENGTH})")

        # ── Models ────────────────────────────────────────────────────────────
        self.get_logger().info("Initializing models...")
        self._initialize_models()
        self._warm_start_models()

        # ── Pose / motion state (worker-thread only after init) ────────────────
        self._reset_buffers()

        # ── Camera intrinsics ─────────────────────────────────────────────────
        self.camera_intrinsics = None
        self.intrinsics_received = False

        # ── Queue infrastructure ──────────────────────────────────────────────
        # Each entry: (rgb_img, depth_img, header, t_received_ms)
        self._process_queue: list = []
        self._queue_lock = threading.Lock()
        self._queue_event = threading.Event()
        self._shutdown = False

        # ── Subscribers / publishers ──────────────────────────────────────────
        self._setup_rgbd_subscribers()
        self._setup_publishers()

        # ── Worker thread ─────────────────────────────────────────────────────
        self._worker_thread = threading.Thread(target=self._pipeline_worker, daemon=True, name="pose_pipeline_worker")
        self._worker_thread.start()

        # ── Statistics ────────────────────────────────────────────────────────
        self.frames_processed = 0
        self.latencies: list = []
        self.create_timer(10.0, self.print_statistics)

        self.get_logger().info("PosePipelineWithQueueNode initialized.")

    # ── Initialisation helpers ─────────────────────────────────────────────────

    def _initialize_models(self):
        self.get_logger().info(f"Loading YOLO pose model: {self.yolo_model_name}")
        self.yolo_model = YOLO(self.yolo_model_name)
        if self.device == "cuda":
            self.yolo_model.to("cuda")
            self.get_logger().info(f"YOLO model loaded on CUDA (GPU: {torch.cuda.get_device_name(0)})")
        else:
            self.get_logger().info("YOLO model loaded on CPU")

        self.get_logger().info(f"Loading motion model from: {self.motion_model_path}")
        self.motion_prediction_jit_fn, self.motion_prediction_params, self.motion_prediction_batch_stats = (
            initialize_jax_models(self.motion_model_path)
        )

        self.motion_ood_score_fn = None
        if self.enable_ood:
            if os.path.exists(self.motion_score_fn_path):
                self.get_logger().info(f"Loading motion OOD score function from: {self.motion_score_fn_path}")
                with open(self.motion_score_fn_path, "rb") as f:
                    motion_score_data = cloudpickle.load(f)
                    self.motion_ood_score_fn = motion_score_data["score_fun"]
            else:
                self.get_logger().warn(f"Motion score function file not found: {self.motion_score_fn_path}")

        self.get_logger().info("All models initialized successfully!")

    def _warm_start_models(self):
        """Run one dummy inference through every model to trigger JIT compilation.

        JAX JIT-compiles on the first call for each unique input shape, which can
        take several seconds.  YOLO also has a slow first-pass initialisation.
        Doing this before the subscribers are created ensures the pipeline is
        already at full speed when real frames start arriving.

        We warm-start for batch sizes 1 and 2 because those cover the typical
        queue sizes at runtime.
        """
        self.get_logger().info("Warm-starting models (triggering JIT compilation)...")
        t0 = time.perf_counter()

        # ── YOLO: one forward pass on a blank image ────────────────────────────
        dummy_rgb = np.zeros((848, 480, 3), dtype=np.uint8)
        self.yolo_model.predict(dummy_rgb, verbose=False)
        self.get_logger().info(f"  YOLO warm-start done ({(time.perf_counter() - t0) * 1e3:.0f} ms)")

        # ── JAX motion model: compile for B=1 and B=2 ─────────────────────────
        # Input shape: [B, T, J*3 + J*9]  (pose + flattened covariance)
        motion_input_dim = N_JOINTS * 3 + N_JOINTS * 3 * 3
        for B in range(1, PREDICTION_HORIZON_LENGTH + 1):
            dummy_motion_input = jnp.zeros([B, INPUT_HORIZON_LENGTH, motion_input_dim], dtype=jnp.float32)
            if self.motion_prediction_batch_stats is not None:
                _ = self.motion_prediction_jit_fn(
                    self.motion_prediction_params,
                    self.motion_prediction_batch_stats,
                    dummy_motion_input,
                )
            else:
                _ = self.motion_prediction_jit_fn(
                    self.motion_prediction_params,
                    dummy_motion_input,
                )
            self.get_logger().info(f"  Motion model B={B} warm-start done ({(time.perf_counter() - t0) * 1e3:.0f} ms)")

        # ── fill_pose_buffer_batched: trigger JAX indexing ops for B=1, 2 ──────
        dummy_buf = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3])
        dummy_cov_buf = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3, 3])
        dummy_val_buf = jnp.zeros([INPUT_HORIZON_LENGTH])
        dummy_mot_buf = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3])
        dummy_mot_cov = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3, 3])
        for B in range(1, PREDICTION_HORIZON_LENGTH + 1):
            _ = fill_pose_buffer_batched(
                points_3d_buffer=dummy_buf,
                covariance_buffer=dummy_cov_buf,
                pose_valid_buffer=dummy_val_buf,
                points_3d_batch=jnp.zeros([B, N_JOINTS, 3]),
                covariance_batch=jnp.zeros([B, N_JOINTS, 3, 3]),
                is_valid_batch=jnp.ones([B], dtype=bool),
                motion_prediction_buffer=dummy_mot_buf,
                motion_uncertainty_buffer=dummy_mot_cov,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1e3
        self.get_logger().info(f"Warm-start complete in {elapsed_ms:.0f} ms.")

    def _reset_buffers(self):
        """Zero all pose/motion rolling buffers and reset the frame counter."""
        self.points_3d_buffer = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3])
        self.covariance_buffer = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3, 3])
        self.pose_valid_buffer = jnp.zeros([INPUT_HORIZON_LENGTH])
        self.motion_prediction_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3])
        self.motion_uncertainty_buffer = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3, 3])
        self.frame_counter = 0

    def _setup_rgbd_subscribers(self):
        color_topic = self.get_parameter("rgbd_color_topic").value
        depth_topic = self.get_parameter("rgbd_depth_topic").value
        info_topic = self.get_parameter("rgbd_info_topic").value

        self.get_logger().info("Setting up RGB-D subscribers:")
        self.get_logger().info(f"  Color (compressed): {color_topic}")
        self.get_logger().info(f"  Depth (compressed): {depth_topic}")
        self.get_logger().info(f"  Camera Info: {info_topic}")

        self.camera_info_sub = self.create_subscription(
            CameraInfo, info_topic, self.camera_info_callback, qos_profile=self.sensor_qos
        )
        self.color_sub = Subscriber(self, CompressedImage, color_topic, qos_profile=self.stream_qos)
        self.depth_sub = Subscriber(self, CompressedImage, depth_topic, qos_profile=self.stream_qos)
        self.sync = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=1, slop=0.05)
        self.sync.registerCallback(self.rgbd_callback)

    def _setup_publishers(self):
        pose_2d_topic = self.get_parameter("pose_2d_output_topic").value
        pose_topic = self.get_parameter("pose_output_topic").value
        motion_topic = self.get_parameter("motion_output_topic").value

        self.get_logger().info("Setting up publishers:")
        self.get_logger().info(f"  Pose 2D: {pose_2d_topic}")
        self.get_logger().info(f"  Pose 3D: {pose_topic}")
        self.get_logger().info(f"  Motion:  {motion_topic}")

        if Pose2D is not None and Pose3D is not None and MotionPrediction is not None:
            self.pose_2d_publisher = self.create_publisher(Pose2D, pose_2d_topic, self.best_effort_qos)
            self.pose_publisher = self.create_publisher(Pose3D, pose_topic, self.best_effort_qos)
            self.motion_publisher = self.create_publisher(MotionPrediction, motion_topic, self.best_effort_qos)
        else:
            self.get_logger().error("Custom messages not available. Cannot create publishers.")

    # ── Callbacks (ROS2 thread) ────────────────────────────────────────────────

    def camera_info_callback(self, msg):
        if not self.intrinsics_received:
            K = msg.k
            self.camera_intrinsics = {
                "fx": K[0],
                "fy": K[4],
                "cx": K[2],
                "cy": K[5],
            }
            self.intrinsics_received = True
            self.get_logger().info(
                f"Camera intrinsics received: fx={K[0]:.2f}, fy={K[4]:.2f}, cx={K[2]:.2f}, cy={K[5]:.2f}"
            )

    def rgbd_callback(self, color_msg, depth_msg):
        """Decode compressed RGB-D pair and push onto the process queue."""
        if not self.intrinsics_received:
            self.get_logger().warn("Waiting for camera intrinsics...", throttle_duration_sec=2.0)
            return

        if not self.tf_transform_received:
            self._lookup_camera_transform()
            if not self.tf_transform_received:
                self.get_logger().warn("Waiting for camera TF transform...", throttle_duration_sec=2.0)
                return

        try:
            t_received = int(time.perf_counter() * 1000)

            # Decompress RGB
            np_arr = np.frombuffer(color_msg.data, np.uint8)
            bgr_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr_img is None:
                self.get_logger().error("Failed to decode compressed RGB image", throttle_duration_sec=1.0)
                return
            color_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # Decompress depth
            if depth_msg.format.startswith("zstd_16UC1:"):
                dims = depth_msg.format.split(":")[1]
                h, w = map(int, dims.split("x"))
                raw = zstandard.ZstdDecompressor().decompress(bytes(depth_msg.data))
                depth_img = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
            else:
                np_arr2 = np.frombuffer(depth_msg.data, np.uint8)
                depth_img = cv2.imdecode(np_arr2, cv2.IMREAD_ANYDEPTH)

            if depth_img is None:
                self.get_logger().error("Failed to decode compressed depth image", throttle_duration_sec=1.0)
                return

            # ── Queue overflow protection ──────────────────────────────────────
            with self._queue_lock:
                qlen = len(self._process_queue)
                if qlen >= _QUEUE_MAX:
                    self.get_logger().error(
                        f"Frame queue full ({qlen}/{_QUEUE_MAX}). "
                        "Pipeline cannot keep up — resetting pose/motion buffers and "
                        "discarding queued frames."
                    )
                    self._process_queue.clear()
                    self._reset_buffers()
                    return
                if qlen >= _QUEUE_WARN:
                    self.get_logger().warn(
                        f"Frame queue growing ({qlen}/{_QUEUE_MAX}). Pipeline may be falling behind."
                    )
                self._process_queue.append((color_img, depth_img, color_msg.header, t_received))

            self._queue_event.set()

        except Exception as e:
            self.get_logger().error(f"Error in RGB-D callback: {e}", throttle_duration_sec=1.0)
            import traceback

            self.get_logger().error(traceback.format_exc())

    # ── Worker thread ──────────────────────────────────────────────────────────

    def _pipeline_worker(self):
        """Dedicated thread: wait for queued frames, process batch, publish."""
        while not self._shutdown:
            signalled = self._queue_event.wait(timeout=1.0)
            self._queue_event.clear()
            if not signalled:
                continue

            # Grab the entire current queue as one batch
            with self._queue_lock:
                if not self._process_queue:
                    continue
                batch = self._process_queue[:]
                self._process_queue.clear()

            try:
                self._process_batch(batch)
            except Exception as e:
                self.get_logger().error(f"Error in pipeline worker: {e}")
                import traceback

                self.get_logger().error(traceback.format_exc())

    def _process_batch(self, batch):
        """Process a list of (rgb, depth, header, t_received) tuples as one batch."""
        B = len(batch)
        rgb_frames = [item[0] for item in batch]
        depth_frames = [item[1] for item in batch]
        headers = [item[2] for item in batch]
        t_received_list = [item[3] for item in batch]

        # ── Step 1: Batched YOLO + depth lifting ──────────────────────────────
        t_pose_start = int(time.perf_counter() * 1000)
        (
            points_3d,
            C_3d_all,
            pose_ood_score,
            pose_is_ood,
            human_detected,
            keypoints_2d,
            uncertainties_2d,
            covariance_xy,
        ) = process_frame_3d_from_rgbd_yolo(
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
        t_pose_done = int(time.perf_counter() * 1000)

        # ── Step 2: Vectorised pose buffer fill → B intermediate states ────────
        (
            self.points_3d_buffer,
            self.covariance_buffer,
            self.pose_valid_buffer,
            intermediate_points_3d,  # [B, T, J, 3]
            intermediate_covariance,  # [B, T, J, 3, 3]
            intermediate_pose_valid,  # [B, T]
            pose_buffer_good_batch,  # [B]
        ) = process_pose_output_batched(
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

        # ── Step 3 & 4: Single batched motion prediction + sequential buffer update
        t_motion_start = int(time.perf_counter() * 1000)
        (
            self.motion_prediction_buffer,
            self.motion_uncertainty_buffer,
            motion_set_radii,  # [B, P, J]
            motion_ood_scores,  # [B]
            motion_is_oods,  # [B]
            valid_motions,  # List[bool] length B
            motion_predicted,  # [B, P, J, 3]
            motion_cov_calibrated,  # [B, P, J, 3, 3]
        ) = run_motion_prediction_batched(
            points_3d_buffers=intermediate_points_3d,
            covariance_buffers=intermediate_covariance,
            pose_valid_buffers=intermediate_pose_valid,
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
            pose_buffer_good_batch=pose_buffer_good_batch,
            frame_counter=self.frame_counter,
        )
        t_motion_done = int(time.perf_counter() * 1000)

        # ── Step 5: Publish one message per frame ─────────────────────────────
        for b in range(B):
            header = headers[b]
            t_received = t_received_list[b]

            is_ood_b = bool(pose_is_ood[b])
            detected_b = bool(human_detected[b])
            ood_score_b = float(pose_ood_score[b])

            self._publish_pose_2d(
                keypoints_2d[b],
                uncertainties_2d[b],
                covariance_xy[b],
                ood_score_b,
                is_ood_b,
                detected_b,
                header,
            )
            self._publish_pose(
                points_3d[b],
                C_3d_all[b],
                ood_score_b,
                is_ood_b,
                detected_b,
                header,
                t_received,
                t_pose_start,
                t_pose_done,
                t_motion_start,
                t_motion_done,
            )
            if valid_motions[b]:
                self._publish_motion(
                    motion_predicted[b],
                    motion_cov_calibrated[b],
                    motion_set_radii[b],
                    float(motion_ood_scores[b]),
                    bool(motion_is_oods[b]),
                    valid_motions[b],
                    header,
                    t_received,
                    t_pose_start,
                    t_pose_done,
                    t_motion_start,
                    t_motion_done,
                )

            self.frame_counter += 1
            self.frames_processed += 1

    # ── TF lookup ──────────────────────────────────────────────────────────────

    def _lookup_camera_transform(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_optical_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            q = tf.transform.rotation
            t = tf.transform.translation
            self.R_rect_to_world = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
            self.t_rect_to_world = np.array([t.x, t.y, t.z], dtype=np.float32)
            self.tf_transform_received = True
            self.get_logger().info(
                f"Camera transform {self.camera_optical_frame} → {self.world_frame}: "
                f"t=[{t.x:.3f}, {t.y:.3f}, {t.z:.3f}] m"
            )
        except tf2_ros.TransformException as e:
            self.get_logger().warn(
                f"TF lookup {self.camera_optical_frame} → {self.world_frame} failed: {e}",
                throttle_duration_sec=2.0,
            )

    # ── Publishers ─────────────────────────────────────────────────────────────

    def _publish_pose_2d(
        self, keypoints_2d, uncertainties_2d, covariance_xy, ood_score, is_ood, human_detected, header
    ):
        if Pose2D is None:
            return
        msg = Pose2D()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.t_image = header.stamp
        msg.keypoints_2d = keypoints_2d.cpu().numpy().flatten().tolist()
        msg.uncertainties_2d = uncertainties_2d.cpu().numpy().flatten().tolist()
        msg.covariance_xy = covariance_xy.cpu().numpy().flatten().tolist()
        msg.n_joints = N_JOINTS
        msg.is_ood = is_ood
        msg.ood_score = ood_score
        msg.human_detected = human_detected
        self.pose_2d_publisher.publish(msg)

        current_time = self.get_clock().now()
        input_time = rclpy.time.Time.from_msg(header.stamp)
        latency_ms = (current_time - input_time).nanoseconds / 1e6
        self.latencies.append(latency_ms)
        if len(self.latencies) > 1000:
            self.latencies.pop(0)

    def _publish_pose(
        self,
        points_3d,
        covariance_3d,
        ood_score,
        is_ood,
        human_detected,
        header,
        t_received,
        t_pose_start,
        t_pose_done,
        t_motion_start,
        t_motion_done,
    ):
        if Pose3D is None:
            return
        msg = Pose3D()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.t_image = header.stamp
        msg.points_3d = points_3d.cpu().numpy().flatten().tolist()
        msg.covariance_3d = covariance_3d.cpu().numpy().flatten().tolist()
        msg.n_joints = N_JOINTS
        msg.is_ood = is_ood
        msg.ood_score = ood_score
        msg.human_detected = human_detected
        msg.t_received_ms = t_received
        msg.t_pose_start_ms = t_pose_start
        msg.t_pose_done_ms = t_pose_done
        msg.t_motion_start_ms = t_motion_start
        msg.t_motion_done_ms = t_motion_done
        msg.t_sent_ms = int(time.perf_counter() * 1000)
        self.pose_publisher.publish(msg)

    def _publish_motion(
        self,
        motion_predicted,
        motion_cov,
        set_radius,
        ood_score,
        is_ood,
        is_valid,
        header,
        t_received,
        t_pose_start,
        t_pose_done,
        t_motion_start,
        t_motion_done,
    ):
        if MotionPrediction is None:
            return
        msg = MotionPrediction()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.t_image = header.stamp
        msg.motion_predicted = np.array(motion_predicted).flatten().tolist()
        msg.motion_covariance = np.array(motion_cov).flatten().tolist()
        msg.set_radius = np.array(set_radius).flatten().tolist()
        msg.prediction_horizon_length = PREDICTION_HORIZON_LENGTH
        msg.n_joints = N_JOINTS
        msg.is_ood = is_ood
        msg.ood_score = float(ood_score) if isinstance(ood_score, (int, float)) else float(ood_score[0])
        msg.is_valid = is_valid
        msg.t_received_ms = t_received
        msg.t_pose_start_ms = t_pose_start
        msg.t_pose_done_ms = t_pose_done
        msg.t_motion_start_ms = t_motion_start
        msg.t_motion_done_ms = t_motion_done
        msg.t_sent_ms = int(time.perf_counter() * 1000)
        self.motion_publisher.publish(msg)

    # ── Statistics ─────────────────────────────────────────────────────────────

    def print_statistics(self):
        stats_msg = f"Frames processed: {self.frames_processed}"
        if self.latencies:
            avg = sum(self.latencies) / len(self.latencies)
            stats_msg += f" | Latency (ms): avg={avg:.1f}, min={min(self.latencies):.1f}, max={max(self.latencies):.1f}"
        self.get_logger().info(stats_msg)

    def destroy_node(self):
        self._shutdown = True
        self._queue_event.set()
        self._worker_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PosePipelineWithQueueNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
