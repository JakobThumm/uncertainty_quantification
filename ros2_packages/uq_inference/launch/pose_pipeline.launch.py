"""
Launch file for the pose pipeline node.

Usage:
    ros2 launch uq_inference pose_pipeline.launch.py
    ros2 launch uq_inference pose_pipeline.launch.py mode:=rgbd
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='stereo',
        description='Mode: stereo or rgbd'
    )

    enable_ood_arg = DeclareLaunchArgument(
        'enable_ood',
        default_value='true',
        description='Enable OOD detection'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device: cuda or cpu'
    )

    # Create the node
    pose_pipeline_node = Node(
        package='uq_inference',
        executable='pose_pipeline',
        name='pose_pipeline_node',
        output='screen',
        parameters=[{
            'mode': LaunchConfiguration('mode'),
            'enable_ood': LaunchConfiguration('enable_ood'),
            'device': LaunchConfiguration('device'),
            # Model paths (relative to workspace root)
            'pose_model_path': 'human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420/jax_resnet50_regressflow',
            'motion_model_path': 'human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle',
            'camera_params_path': 'human_pose_pipeline/models/pose_estimation/H36M/RegressFlow/seed_420/camera-parameters.json',
            'motion_score_fn_path': 'human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle',
            'cache_dir': 'cache/',
            'pose_base_key': '',  # Set this if you have pose OOD detection
            # Stereo camera topics
            'camera_1_color_topic': '/realsense/camera_1/color/image_raw',
            'camera_2_color_topic': '/realsense/camera_2/color/image_raw',
            'camera_1_info_topic': '/realsense/camera_1/color/camera_info',
            'camera_2_info_topic': '/realsense/camera_2/color/camera_info',
            # RGB-D camera topics
            'rgbd_color_topic': '/realsense/camera_1/color/image_raw',
            'rgbd_depth_topic': '/realsense/camera_1/aligned_depth_to_color/image_raw',
            'rgbd_info_topic': '/realsense/camera_1/color/camera_info',
            # Camera IDs for calibration
            'camera_1_id': '55011271',
            'camera_2_id': '60457274',
            'subject': 'S1',
            # Output topics
            'pose_output_topic': '/uq/pose_3d',
            'motion_output_topic': '/uq/motion_prediction',
        }]
    )

    return LaunchDescription([
        mode_arg,
        enable_ood_arg,
        device_arg,
        pose_pipeline_node
    ])
