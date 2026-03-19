"""
Launch file for the queue-based pose pipeline node.

Usage:
    ros2 launch uq_inference pose_pipeline_with_queue.launch.py
    ros2 launch uq_inference pose_pipeline_with_queue.launch.py enable_ood:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    enable_ood_arg = DeclareLaunchArgument(
        'enable_ood',
        default_value='false',
        description='Enable OOD detection'
    )

    enable_tracking_arg = DeclareLaunchArgument(
        'enable_tracking',
        default_value='true',
        description='Enable YOLO multi-object tracking'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device: cuda or cpu'
    )

    stream_reliable_arg = DeclareLaunchArgument(
        'stream_reliable',
        default_value='true',
        description='QoS reliability for /rgbd_stream subscriptions: true=Reliable, false=Best Effort'
    )

    pose_pipeline_node = Node(
        package='uq_inference',
        executable='pose_pipeline_with_queue',
        name='pose_pipeline_with_queue_node',
        output='screen',
        parameters=[{
            'enable_ood': LaunchConfiguration('enable_ood'),
            'enable_tracking': LaunchConfiguration('enable_tracking'),
            'device': LaunchConfiguration('device'),
            'stream_reliable': LaunchConfiguration('stream_reliable'),
            # Model paths (relative to workspace root)
            'yolo_model': 'yolo26n-pose.pt',
            'motion_model_path': 'human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle',
            'motion_score_fn_path': 'human_pose_pipeline/models/motion_prediction/final_model_for_ood/dct_pose_transformer_scores_subsample10000_lanczos_seed0_size_HM0of0_LM1440of1600_sketch_srft_seed0_size20000.cloudpickle',
            'depth_uncertainty': 0.002,
            # RGB-D camera topics
            'rgbd_color_topic': 'rgbd_stream/rgb/compressed',
            'rgbd_depth_topic': 'rgbd_stream/depth/compressed',
            'rgbd_info_topic': '/camera/camera/color/camera_info',
            # TF frames
            'world_frame': 'world',
            'camera_optical_frame': 'camera_depth_optical_frame',
            # Output topics
            'pose_2d_output_topic': '/uq/pose_2d',
            'pose_output_topic': '/uq/pose_3d',
            'motion_output_topic': '/uq/motion_prediction',
        }]
    )

    return LaunchDescription([
        enable_ood_arg,
        enable_tracking_arg,
        device_arg,
        stream_reliable_arg,
        pose_pipeline_node,
    ])
