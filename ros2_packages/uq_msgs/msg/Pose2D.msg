# 2D Pose with uncertainty
# Header with timestamp and frame_id
std_msgs/Header header

# 2D joint positions [N_JOINTS, 2]
# Flattened array: [x1, y1, x2, y2, ..., xN, yN]
float64[] keypoints_2d

# Uncertainties (standard deviations) for each joint [N_JOINTS, 2]
# Flattened array: [std_x1, std_y1, std_x2, std_y2, ..., std_xN, std_yN]
float64[] uncertainties_2d

# Covariance between x and y for each joint [N_JOINTS]
# Array: [cov_xy1, cov_xy2, ..., cov_xyN]
float64[] covariance_xy

# Number of joints
int32 n_joints

# Whether this pose is out-of-distribution
bool is_ood

# OOD score
float64 ood_score

# Whether a human was detected in the frame
bool human_detected
