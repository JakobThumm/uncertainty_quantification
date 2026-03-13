# Motion prediction with uncertainty
# Header with timestamp and frame_id
std_msgs/Header header

# Predicted motion [PREDICTION_HORIZON_LENGTH, N_JOINTS, 3]
# Flattened array: [t1_j1_x, t1_j1_y, t1_j1_z, t1_j2_x, ..., tN_jM_z]
float64[] motion_predicted

# Covariance matrices [PREDICTION_HORIZON_LENGTH, N_JOINTS, 3, 3]
# Flattened array in row-major order
float64[] motion_covariance

# Set radius for uncertainty visualization [PREDICTION_HORIZON_LENGTH, N_JOINTS]
# Flattened array
float64[] set_radius

# Prediction horizon length
int32 prediction_horizon_length

# Number of joints
int32 n_joints

# Whether this prediction is out-of-distribution
bool is_ood

# OOD score
float64 ood_score

# Whether this prediction is valid (enough good poses in buffer)
bool is_valid

# Inference-side timing (time.perf_counter() * 1000, ms, inference machine clock)
# Used by the streamer node to compute end-to-end latency breakdowns.
int64 t_received_ms       # image arrived at inference node
int64 t_pose_start_ms     # 3D pose estimation started
int64 t_pose_done_ms      # 3D pose estimation finished
int64 t_motion_start_ms   # motion prediction started
int64 t_motion_done_ms    # motion prediction finished
int64 t_sent_ms           # message published
