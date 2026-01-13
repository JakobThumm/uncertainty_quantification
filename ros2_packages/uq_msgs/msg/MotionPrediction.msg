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
