from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'uq_inference'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='ROS2 package for uncertainty quantification inference on streamed images',
    license='MIT',
    entry_points={
        'console_scripts': [
            'image_processor = uq_inference.image_processor_node:main',
            'pose_pipeline = uq_inference.pose_pipeline_node:main',
            'pose_pipeline_with_queue = uq_inference.pose_pipeline_with_queue_node:main',
        ],
    },
)
