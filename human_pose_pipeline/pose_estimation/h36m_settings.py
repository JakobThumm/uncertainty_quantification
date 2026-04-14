# Standard joint mappings for H36M dataset
# Map from 17 joints (COCO format) to 13 joints (H36M subset)
JOINT_IDX_17 = [0, 1, 2, 3, 6, 7, 8, 12, 16, 14, 15, 17, 18, 19, 25, 26, 27]
JOINT_IDX_13 = [10, 14, 11, 15, 12, 16, 13, 1, 4, 2, 5, 3, 6]
JOINT_IDX_13_MODEL = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
MIRROR_13_JOINT_MODEL_MAP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]

# Joint names for interpretability
JOINT_NAMES_13 = [
    'Nose', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
    'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee',
    'LAnkle', 'RAnkle'
]

CONNECTIONS_13 = [
    (0, 1), (0, 2),  # Nose to shoulders
    (1, 3), (3, 5),  # Left arm
    (2, 4), (4, 6),  # Right arm
    (1, 2), (1, 7), (2, 8),  # Shoulders to hips
    (7, 8),  # Connect hips
    (7, 9), (9, 11),  # Left leg
    (8, 10), (10, 12)  # Right leg
]

YOLO_IMAGE_SIZE = (512, 640)
YOLO_CONFIDENCE_THRESHOLD = 0.3

TRANSFORM_SIGMA = 2.0,
PREDICTION_NUM_JOINTS = 17,
TRANSFORM_IMAGE_SIZE = [192, 256]  # Width, Height
TRANSFORM_HEATMAP_SIZE = [48, 64]  # Width, Height
NORMALIZATION_OFFSET = [-0.406, -0.457, -0.480]

OOD_THRESHOLD = 0.1  # Threshold for OOD detection in pose estimation
COVARIANCE_OOD_THRESHOLD = 1e5
