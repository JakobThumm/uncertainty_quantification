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
