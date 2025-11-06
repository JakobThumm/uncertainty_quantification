N_JOINTS = 13
INPUT_HORIZON_LENGTH = 50
PREDICTION_HORIZON_LENGTH = 10
REDUCED_TIMESTEP = 4  # Predict only timestep 4
REDUCED_JOINT_INDICES = [0, 5, 6]  # Predict only joints: Head, Left Hand, Right Hand
# Only used in (get_h36m_motion_dataset_with_uncertainty)
FAKE_INPUT_UNCERTAINTY = 0.01
