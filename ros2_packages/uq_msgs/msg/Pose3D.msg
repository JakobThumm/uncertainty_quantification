# 3D Pose with uncertainty
# Header with timestamp and frame_id
std_msgs/Header header

# 3D joint positions [N_JOINTS, 3]
# Flattened array: [x1, y1, z1, x2, y2, z2, ..., xN, yN, zN]
float64[] points_3d

# Covariance matrices for each joint [N_JOINTS, 3, 3]
# Flattened array in row-major order
float64[] covariance_3d

# Number of joints
int32 n_joints

# Whether this pose is out-of-distribution
bool is_ood

# OOD score
float64 ood_score

# Whether a human was detected in the frame
bool human_detected

# Inference-side timing (time.perf_counter() * 1000, ms, inference machine clock)
# Used by the streamer node to compute end-to-end latency breakdowns.
int64 t_received_ms       # image arrived at inference node
int64 t_pose_start_ms     # 3D pose estimation started
int64 t_pose_done_ms      # 3D pose estimation finished
int64 t_motion_start_ms   # motion prediction started (0 if not run this frame)
int64 t_motion_done_ms    # motion prediction finished (0 if not run this frame)
int64 t_sent_ms           # message published
