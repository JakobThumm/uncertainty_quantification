N_JOINTS = 13
INPUT_HORIZON_LENGTH = 50
PREDICTION_HORIZON_LENGTH = 10
REDUCED_TIMESTEP = 4  # Predict only timestep 4
REDUCED_JOINT_INDICES = [0, 5, 6]  # Predict only joints: Head, Left Hand, Right Hand
# Only used in (get_h36m_motion_dataset_with_uncertainty)
FAKE_INPUT_UNCERTAINTY = 0.01
OOD_THRESHOLD = 5e5
# Number of recent non-ood 3D poses required to accept the current motion prediction.
# This prevents an infinite feedback loop of predicted poses.
N_CORRECT_POSES_REQUIRED = 3

# Covariance calibration for motion prediction
COV_CALIBRATION_CT = 1.2
COV_CALIBRATION_IT = 0.4
COV_CALIBRATION_FACTORS = [1.0, 1.0, 1.0, 1.0, 1.0,
                           1.7, 1.7, 1.0, 1.0, 1.0,
                           1.0, 1.5, 1.5]
# Likelihood boundary for the predicted set
SET_LIKELIHOOD = 0.99

SARA_MEASUREMENT_UNCERTAINTY = 0.005
