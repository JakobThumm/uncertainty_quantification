#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading


class RGBDPublisher(Node):
    """
    Publishes RGBD images from RealSense camera with optional compression.

    Subscribes to RealSense camera topics and republishes RGB + Depth at
    a controlled frequency with optional compression.
    """

    def __init__(self):
        super().__init__('rgbd_publisher')

        # Declare parameters
        self.declare_parameter('publish_rate', 10.0)  # Hz
        self.declare_parameter('compress_rgb', False)
        self.declare_parameter('compress_depth', False)
        self.declare_parameter('rgb_quality', 90)  # JPEG quality 0-100
        self.declare_parameter('depth_png_compression', 3)  # PNG compression 0-9
        self.declare_parameter('camera_namespace', '/camera/camera')

        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.compress_rgb = self.get_parameter('compress_rgb').value
        self.compress_depth = self.get_parameter('compress_depth').value
        self.rgb_quality = self.get_parameter('rgb_quality').value
        self.depth_png_compression = self.get_parameter('depth_png_compression').value
        camera_ns = self.get_parameter('camera_namespace').value

        # CV Bridge
        self.bridge = CvBridge()

        # Latest images storage
        self.rgb_image = None
        self.depth_image = None
        self.rgb_timestamp = None
        self.depth_timestamp = None
        self.lock = threading.Lock()

        # Statistics
        self.rgb_received_count = 0
        self.depth_received_count = 0
        self.published_count = 0
        self.last_rgb_time = None
        self.last_depth_time = None
        self.compression_times = []

        # Subscribers to RealSense camera
        self.rgb_sub = self.create_subscription(
            Image,
            f'{camera_ns}/color/image_raw',
            self.rgb_callback,
            10)

        self.depth_sub = self.create_subscription(
            Image,
            f'{camera_ns}/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10)

        # Publishers
        if self.compress_rgb:
            self.rgb_pub = self.create_publisher(
                CompressedImage,
                'rgbd_stream/rgb/compressed',
                10)
        else:
            self.rgb_pub = self.create_publisher(
                Image,
                'rgbd_stream/rgb/raw',
                10)

        if self.compress_depth:
            self.depth_pub = self.create_publisher(
                CompressedImage,
                'rgbd_stream/depth/compressed',
                10)
        else:
            self.depth_pub = self.create_publisher(
                Image,
                'rgbd_stream/depth/raw',
                10)

        # Create timer for publishing at controlled rate
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_callback)

        # Status timer
        self.status_timer = self.create_timer(2.0, self.print_status)

        self.get_logger().info('=== RGBD Publisher Started ===')
        self.get_logger().info(f'Publishing rate: {self.publish_rate} Hz')
        self.get_logger().info(f'RGB compression: {self.compress_rgb}' +
                              (f' (quality={self.rgb_quality})' if self.compress_rgb else ''))
        self.get_logger().info(f'Depth compression: {self.compress_depth}' +
                              (f' (level={self.depth_png_compression})' if self.compress_depth else ''))
        self.get_logger().info(f'Subscribing to: {camera_ns}/color/image_raw')
        self.get_logger().info(f'Subscribing to: {camera_ns}/aligned_depth_to_color/image_raw')

    def rgb_callback(self, msg):
        """Callback for RGB images from RealSense."""
        with self.lock:
            self.rgb_image = msg
            self.rgb_timestamp = msg.header.stamp
            self.rgb_received_count += 1

            current_time = time.time()
            if self.last_rgb_time is not None:
                fps = 1.0 / (current_time - self.last_rgb_time)
                self.get_logger().debug(f'RGB FPS: {fps:.1f}')
            self.last_rgb_time = current_time

    def depth_callback(self, msg):
        """Callback for depth images from RealSense."""
        with self.lock:
            self.depth_image = msg
            self.depth_timestamp = msg.header.stamp
            self.depth_received_count += 1

            current_time = time.time()
            if self.last_depth_time is not None:
                fps = 1.0 / (current_time - self.last_depth_time)
                self.get_logger().debug(f'Depth FPS: {fps:.1f}')
            self.last_depth_time = current_time

    def compress_rgb_image(self, cv_image, timestamp):
        """Compress RGB image to JPEG."""
        compress_start = time.time()

        # Encode to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.rgb_quality]
        success, encoded = cv2.imencode('.jpg', cv_image, encode_param)

        if not success:
            self.get_logger().error('Failed to compress RGB image')
            return None

        # Create CompressedImage message
        msg = CompressedImage()
        msg.header.stamp = timestamp
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.format = 'jpeg'
        msg.data = encoded.tobytes()

        compress_time = (time.time() - compress_start) * 1000  # ms
        self.compression_times.append(compress_time)

        return msg

    def compress_depth_image(self, cv_image, timestamp):
        """Compress depth image to PNG."""
        compress_start = time.time()

        # Depth is typically uint16, encode to PNG with compression
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), self.depth_png_compression]
        success, encoded = cv2.imencode('.png', cv_image, encode_param)

        if not success:
            self.get_logger().error('Failed to compress depth image')
            return None

        # Create CompressedImage message
        msg = CompressedImage()
        msg.header.stamp = timestamp
        msg.header.frame_id = 'camera_depth_optical_frame'
        msg.format = '16UC1; png'  # Depth format
        msg.data = encoded.tobytes()

        compress_time = (time.time() - compress_start) * 1000  # ms
        self.compression_times.append(compress_time)

        return msg

    def publish_callback(self):
        """Timer callback to publish images at controlled rate."""
        with self.lock:
            if self.rgb_image is None or self.depth_image is None:
                self.get_logger().warn('Waiting for both RGB and depth images...',
                                      throttle_duration_sec=2.0)
                return

            # Convert to OpenCV format
            try:
                rgb_cv = self.bridge.imgmsg_to_cv2(self.rgb_image, desired_encoding='bgr8')
                depth_cv = self.bridge.imgmsg_to_cv2(self.depth_image, desired_encoding='passthrough')
            except Exception as e:
                self.get_logger().error(f'Error converting images: {e}')
                return

            # Publish RGB
            if self.compress_rgb:
                rgb_msg = self.compress_rgb_image(rgb_cv, self.rgb_timestamp)
                if rgb_msg is not None:
                    self.rgb_pub.publish(rgb_msg)
            else:
                self.rgb_pub.publish(self.rgb_image)

            # Publish Depth
            if self.compress_depth:
                depth_msg = self.compress_depth_image(depth_cv, self.depth_timestamp)
                if depth_msg is not None:
                    self.depth_pub.publish(depth_msg)
            else:
                self.depth_pub.publish(self.depth_image)

            self.published_count += 1

            self.get_logger().debug(f'Published frame {self.published_count}')

    def print_status(self):
        """Print status information."""
        with self.lock:
            avg_compression_time = 0.0
            if len(self.compression_times) > 0:
                avg_compression_time = sum(self.compression_times) / len(self.compression_times)
                self.compression_times = []  # Reset

            status = f'Status: RGB received={self.rgb_received_count}, ' \
                    f'Depth received={self.depth_received_count}, ' \
                    f'Published={self.published_count}'

            if avg_compression_time > 0:
                status += f', Avg compression time={avg_compression_time:.2f}ms'

            self.get_logger().info(status)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = RGBDPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
