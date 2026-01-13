"""
ROS2 node for processing images with uncertainty quantification.

This node:
1. Subscribes to image topics from remote PCs
2. Runs UQ inference on received images
3. Publishes results back
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String
from cv_bridge import CvBridge
import numpy as np

from .uq_wrapper import UQInferenceWrapper


class ImageProcessorNode(Node):
    """
    ROS2 node that processes images with uncertainty quantification.
    """

    def __init__(self):
        super().__init__('uq_image_processor')

        # Declare parameters
        self.declare_parameter('model_path', '../models/default_model.pkl')
        self.declare_parameter('model_type', 'ResNet')
        self.declare_parameter('dataset', 'CIFAR-10')
        self.declare_parameter('score_method', 'local_ensemble')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('input_topic', '/camera/image_raw')
        self.declare_parameter('output_uncertainty_topic', '/uq/uncertainty_score')
        self.declare_parameter('output_prediction_topic', '/uq/prediction')

        # Get parameters
        model_path = self.get_parameter('model_path').value
        model_type = self.get_parameter('model_type').value
        dataset = self.get_parameter('dataset').value
        score_method = self.get_parameter('score_method').value
        device = self.get_parameter('device').value
        input_topic = self.get_parameter('input_topic').value
        output_uncertainty_topic = self.get_parameter('output_uncertainty_topic').value
        output_prediction_topic = self.get_parameter('output_prediction_topic').value

        # Initialize CV Bridge for image conversion
        self.bridge = CvBridge()

        # Initialize UQ wrapper
        self.get_logger().info(f'Initializing UQ wrapper with model: {model_path}')
        self.uq_wrapper = UQInferenceWrapper(
            model_path=model_path,
            model_type=model_type,
            dataset=dataset,
            score_method=score_method,
            device=device
        )

        # Initialize model (do this here to fail fast if there are issues)
        try:
            self.uq_wrapper.initialize()
            self.get_logger().info('UQ model initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize UQ model: {e}')
            raise

        # Create subscribers
        self.image_subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10  # QoS depth
        )

        # Create publishers
        self.uncertainty_publisher = self.create_publisher(
            Float32,
            output_uncertainty_topic,
            10
        )
        self.prediction_publisher = self.create_publisher(
            String,
            output_prediction_topic,
            10
        )

        # Statistics
        self.images_processed = 0
        self.create_timer(10.0, self.print_statistics)  # Print stats every 10 seconds

        self.get_logger().info('Image processor node initialized')
        self.get_logger().info(f'Subscribing to: {input_topic}')
        self.get_logger().info(f'Publishing uncertainty to: {output_uncertainty_topic}')
        self.get_logger().info(f'Publishing predictions to: {output_prediction_topic}')

    def image_callback(self, msg):
        """
        Callback for receiving images.

        Args:
            msg: ROS Image message
        """
        try:
            # Convert ROS Image message to OpenCV image (numpy array)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Process image with UQ
            result = self.uq_wrapper.compute_uncertainty(cv_image)

            # Publish uncertainty score
            uncertainty_msg = Float32()
            uncertainty_msg.data = float(result['uncertainty_score'])
            self.uncertainty_publisher.publish(uncertainty_msg)

            # Publish prediction
            prediction_msg = String()
            prediction_msg.data = str(result['prediction'])
            self.prediction_publisher.publish(prediction_msg)

            # Update statistics
            self.images_processed += 1

            # Log occasionally
            if self.images_processed % 100 == 0:
                self.get_logger().info(
                    f'Processed {self.images_processed} images. '
                    f'Latest uncertainty: {result["uncertainty_score"]:.4f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def print_statistics(self):
        """Print processing statistics."""
        self.get_logger().info(f'Total images processed: {self.images_processed}')


def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)

    try:
        node = ImageProcessorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
