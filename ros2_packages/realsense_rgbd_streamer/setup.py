from setuptools import setup
import os
from glob import glob

package_name = 'realsense_rgbd_streamer'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='RealSense RGBD camera streamer with compression support',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rgbd_publisher = realsense_rgbd_streamer.rgbd_publisher:main',
            'rgbd_subscriber = realsense_rgbd_streamer.rgbd_subscriber:main',
        ],
    },
)
