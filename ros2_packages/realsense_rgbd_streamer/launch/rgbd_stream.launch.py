from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='10.0',
        description='Publishing rate in Hz'
    )

    compress_rgb_arg = DeclareLaunchArgument(
        'compress_rgb',
        default_value='false',
        description='Compress RGB images (true/false)'
    )

    compress_depth_arg = DeclareLaunchArgument(
        'compress_depth',
        default_value='false',
        description='Compress depth images (true/false)'
    )

    rgb_quality_arg = DeclareLaunchArgument(
        'rgb_quality',
        default_value='90',
        description='JPEG quality for RGB compression (0-100)'
    )

    depth_compression_arg = DeclareLaunchArgument(
        'depth_png_compression',
        default_value='3',
        description='PNG compression level for depth (0-9)'
    )

    camera_namespace_arg = DeclareLaunchArgument(
        'camera_namespace',
        default_value='/camera/camera',
        description='RealSense camera namespace'
    )

    # RGBD Publisher node
    rgbd_publisher = Node(
        package='realsense_rgbd_streamer',
        executable='rgbd_publisher',
        name='rgbd_publisher',
        output='screen',
        parameters=[{
            'publish_rate': LaunchConfiguration('publish_rate'),
            'compress_rgb': LaunchConfiguration('compress_rgb'),
            'compress_depth': LaunchConfiguration('compress_depth'),
            'rgb_quality': LaunchConfiguration('rgb_quality'),
            'depth_png_compression': LaunchConfiguration('depth_png_compression'),
            'camera_namespace': LaunchConfiguration('camera_namespace'),
        }]
    )

    return LaunchDescription([
        publish_rate_arg,
        compress_rgb_arg,
        compress_depth_arg,
        rgb_quality_arg,
        depth_compression_arg,
        camera_namespace_arg,
        rgbd_publisher,
    ])
