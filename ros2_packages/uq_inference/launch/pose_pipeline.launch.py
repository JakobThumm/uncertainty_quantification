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
        default_value='rgbd',
        description='Mode: stereo or rgbd'
    )

    enable_ood_arg = DeclareLaunchArgument(
        'enable_ood',
        default_value='true',
        description='Enable OOD detection'
    )

    enable_tracking_arg = DeclareLaunchArgument(
        'enable_tracking',
        default_value='false',
        description='Enable YOLO multi-object tracking'
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
            'enable_tracking': LaunchConfiguration('enable_tracking'),
            'device': LaunchConfiguration('device'),
            # Model paths (relative to workspace root)
            'yolo_model': 'yolo26n-pose.pt',
            'motion_model_path': 'human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle',
            'motion_score_fn_path': 'human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle',
            'depth_uncertainty': 0.002,
            # RGB-D camera topics
            'rgbd_color_topic': '/realsense/camera_1/color/image_raw',
            'rgbd_depth_topic': '/realsense/camera_1/aligned_depth_to_color/image_raw',
            'rgbd_info_topic': '/realsense/camera_1/color/camera_info',
            # Output topics
            'pose_2d_output_topic': '/uq/pose_2d',
            'pose_output_topic': '/uq/pose_3d',
            'motion_output_topic': '/uq/motion_prediction',
        }]
    )

    return LaunchDescription([
        mode_arg,
        enable_ood_arg,
        enable_tracking_arg,
        device_arg,
        pose_pipeline_node
    ])
