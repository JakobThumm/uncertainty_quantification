N_JOINTS = 13
INPUT_HORIZON_LENGTH = 50
PREDICTION_HORIZON_LENGTH = 10
REDUCED_TIMESTEP = 4  # Predict only timestep 4
REDUCED_JOINT_INDICES = [0, 5, 6]  # Predict only joints: Head, Left Hand, Right Hand
# Only used in (get_h36m_motion_dataset_with_uncertainty)
FAKE_INPUT_UNCERTAINTY = 0.01
OOD_THRESHOLD = 6e5
# Number of recent non-ood 3D poses required to accept the current motion prediction.
# This prevents an infinite feedback loop of predicted poses.
N_CORRECT_POSES_REQUIRED = 3

# Covariance calibration for motion prediction
# Constant time factor
COV_CALIBRATION_CT = 3.5
# Increase time factor
COV_CALIBRATION_IT = 0.7
# JOINT_NAMES_13 = [
#     'Nose', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
#     'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee',
#     'RKnee', 'LAnkle', 'RAnkle'
# ]
COV_CALIBRATION_FACTORS = [1.3, 1.2, 1.2, 1.4, 1.4,
                           2.2, 2.2, 1.0, 1.0, 1.0,
                           1.0, 1.1, 1.1]
# Likelihood boundary for the predicted set
SET_LIKELIHOOD = 0.99

SARA_MEASUREMENT_UNCERTAINTY = 0.20
