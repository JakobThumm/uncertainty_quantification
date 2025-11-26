"""
Unit tests for GPU-accelerated affine transformation utilities.

Tests verify that torch-based implementations match cv2 reference implementations.
"""

import unittest
import numpy as np
import torch
import cv2
from typing import Tuple

from gpu_accelerated_utils import (
    get_affine_transform_torch_batch,
    _get_affine_transform_cv2,
    _get_affine_transform_torch,
)


class TestAffineTransformFunctions(unittest.TestCase):
    """Test cases for affine transformation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_size = (192, 256)  # Common pose estimation size
        self.tolerance_rtol = 1e-4
        self.tolerance_atol = 1e-5

    def test_get_affine_transform_torch_batch_single(self):
        """Test that batched torch implementation matches cv2 for single example."""
        # Create source and destination points (matching cv2 format)
        src_np = np.array([[100.0, 100.0], [150.0, 100.0], [100.0, 150.0]], dtype=np.float32)
        dst_np = np.array([[50.0, 50.0], [100.0, 50.0], [50.0, 100.0]], dtype=np.float32)

        # Get cv2 transformation
        trans_cv2 = cv2.getAffineTransform(src_np, dst_np)

        # Convert to torch and add batch dimension
        src_torch = torch.from_numpy(src_np).unsqueeze(0).to(self.device)  # (1, 3, 2)
        dst_torch = torch.from_numpy(dst_np).unsqueeze(0).to(self.device)  # (1, 3, 2)

        # Get torch transformation
        trans_torch = get_affine_transform_torch_batch(src_torch, dst_torch)
        trans_torch_np = trans_torch.cpu().numpy().squeeze(0)  # Remove batch dimension

        # Compare results
        np.testing.assert_allclose(
            trans_torch_np,
            trans_cv2,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Torch batch implementation doesn't match cv2 for single example",
        )

    def test_get_affine_transform_torch_batch_multiple(self):
        """Test that batched torch implementation matches cv2 for multiple examples."""
        batch_size = 5

        for i in range(batch_size):
            # Create random source and destination points
            src_np = np.random.rand(3, 2).astype(np.float32) * 200
            dst_np = np.random.rand(3, 2).astype(np.float32) * 200

            # Get cv2 transformation
            trans_cv2 = cv2.getAffineTransform(src_np, dst_np)

            # Convert to torch
            src_torch = torch.from_numpy(src_np).unsqueeze(0).to(self.device)
            dst_torch = torch.from_numpy(dst_np).unsqueeze(0).to(self.device)

            # Get torch transformation
            trans_torch = get_affine_transform_torch_batch(src_torch, dst_torch)
            trans_torch_np = trans_torch.cpu().numpy().squeeze(0)

            # Compare results
            np.testing.assert_allclose(
                trans_torch_np,
                trans_cv2,
                rtol=self.tolerance_rtol,
                atol=self.tolerance_atol,
                err_msg=f"Torch batch implementation doesn't match cv2 for example {i}",
            )

    def test_get_affine_transform_torch_batch_batched(self):
        """Test batched processing with multiple examples in parallel."""
        batch_size = 8

        # Create batched source and destination points
        src_batch_np = np.random.rand(batch_size, 3, 2).astype(np.float32) * 200
        dst_batch_np = np.random.rand(batch_size, 3, 2).astype(np.float32) * 200

        # Get cv2 transformations for each example
        trans_cv2_list = []
        for i in range(batch_size):
            trans_cv2 = cv2.getAffineTransform(src_batch_np[i], dst_batch_np[i])
            trans_cv2_list.append(trans_cv2)
        trans_cv2_batch = np.stack(trans_cv2_list, axis=0)

        # Convert to torch and get batched transformation
        src_torch = torch.from_numpy(src_batch_np).to(self.device)
        dst_torch = torch.from_numpy(dst_batch_np).to(self.device)
        trans_torch = get_affine_transform_torch_batch(src_torch, dst_torch)
        trans_torch_np = trans_torch.cpu().numpy()

        # Compare results
        np.testing.assert_allclose(
            trans_torch_np,
            trans_cv2_batch,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Batched torch implementation doesn't match cv2",
        )

    def test_get_affine_transform_torch_identity(self):
        """Test identity transformation (src == dst)."""
        src_np = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        dst_np = src_np.copy()

        # Get cv2 transformation
        trans_cv2 = cv2.getAffineTransform(src_np, dst_np)

        # Convert to torch
        src_torch = torch.from_numpy(src_np).unsqueeze(0).to(self.device)
        dst_torch = torch.from_numpy(dst_np).unsqueeze(0).to(self.device)

        # Get torch transformation
        trans_torch = get_affine_transform_torch_batch(src_torch, dst_torch)
        trans_torch_np = trans_torch.cpu().numpy().squeeze(0)

        # Compare results
        np.testing.assert_allclose(
            trans_torch_np,
            trans_cv2,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Identity transformation doesn't match",
        )

        # Should be approximately identity matrix
        expected_identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(
            trans_torch_np,
            expected_identity,
            rtol=1e-3,
            atol=1e-3,
            err_msg="Identity transformation is not close to identity matrix",
        )

    def test_affine_transform_functions_no_rotation(self):
        """Test that _get_affine_transform_torch matches _get_affine_transform_cv2 with no rotation."""
        # Create center and scale
        center_np = np.array([128.0, 128.0], dtype=np.float32)
        scale_np = np.array([200.0, 200.0], dtype=np.float32)

        # Get cv2 transformation
        trans_cv2 = _get_affine_transform_cv2(center_np, scale_np, self.output_size, rot=0)

        # Convert to torch
        center_torch = torch.from_numpy(center_np).unsqueeze(0).to(self.device)  # (1, 2)
        scale_torch = torch.from_numpy(scale_np).unsqueeze(0).to(self.device)  # (1, 2)

        # Get torch transformation
        trans_torch = _get_affine_transform_torch(
            center_torch, scale_torch, self.output_size, rot=0, device=self.device
        )
        trans_torch_np = trans_torch.cpu().numpy().squeeze(0)  # Remove batch dimension

        # Compare results
        np.testing.assert_allclose(
            trans_torch_np,
            trans_cv2,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="_get_affine_transform_torch doesn't match _get_affine_transform_cv2 (no rotation)",
        )

    def test_affine_transform_functions_with_rotation(self):
        """Test that _get_affine_transform_torch matches _get_affine_transform_cv2 with rotation."""
        # Test various rotation angles
        rotation_angles = [0, 15, 30, 45, 90, -30, -45]

        for rot in rotation_angles:
            with self.subTest(rotation=rot):
                # Create center and scale
                center_np = np.array([150.0, 200.0], dtype=np.float32)
                scale_np = np.array([180.0, 180.0], dtype=np.float32)

                # Get cv2 transformation
                trans_cv2 = _get_affine_transform_cv2(center_np, scale_np, self.output_size, rot=rot)

                # Convert to torch
                center_torch = torch.from_numpy(center_np).unsqueeze(0).to(self.device)
                scale_torch = torch.from_numpy(scale_np).unsqueeze(0).to(self.device)

                # Get torch transformation
                trans_torch = _get_affine_transform_torch(
                    center_torch, scale_torch, self.output_size, rot=rot, device=self.device
                )
                trans_torch_np = trans_torch.cpu().numpy().squeeze(0)

                # Compare results
                np.testing.assert_allclose(
                    trans_torch_np,
                    trans_cv2,
                    rtol=self.tolerance_rtol,
                    atol=self.tolerance_atol,
                    err_msg=f"_get_affine_transform_torch doesn't match _get_affine_transform_cv2 (rotation={rot})",
                )

    def test_affine_transform_functions_batched(self):
        """Test batched version of _get_affine_transform_torch against multiple cv2 calls."""
        batch_size = 10

        # Create random centers and scales
        centers_np = np.random.rand(batch_size, 2).astype(np.float32) * 256
        scales_np = np.random.rand(batch_size, 2).astype(np.float32) * 100 + 100

        # Get cv2 transformations
        trans_cv2_list = []
        for i in range(batch_size):
            trans_cv2 = _get_affine_transform_cv2(centers_np[i], scales_np[i], self.output_size, rot=0)
            trans_cv2_list.append(trans_cv2)
        trans_cv2_batch = np.stack(trans_cv2_list, axis=0)

        # Convert to torch
        centers_torch = torch.from_numpy(centers_np).to(self.device)
        scales_torch = torch.from_numpy(scales_np).to(self.device)

        # Get torch transformation
        trans_torch = _get_affine_transform_torch(
            centers_torch, scales_torch, self.output_size, rot=0, device=self.device
        )
        trans_torch_np = trans_torch.cpu().numpy()

        # Compare results
        np.testing.assert_allclose(
            trans_torch_np,
            trans_cv2_batch,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Batched _get_affine_transform_torch doesn't match multiple cv2 calls",
        )

    def test_affine_transform_different_scales(self):
        """Test with different x and y scales."""
        center_np = np.array([100.0, 150.0], dtype=np.float32)
        scale_np = np.array([150.0, 200.0], dtype=np.float32)  # Different x and y scales

        # Get cv2 transformation
        trans_cv2 = _get_affine_transform_cv2(center_np, scale_np, self.output_size, rot=0)

        # Convert to torch
        center_torch = torch.from_numpy(center_np).unsqueeze(0).to(self.device)
        scale_torch = torch.from_numpy(scale_np).unsqueeze(0).to(self.device)

        # Get torch transformation
        trans_torch = _get_affine_transform_torch(
            center_torch, scale_torch, self.output_size, rot=0, device=self.device
        )
        trans_torch_np = trans_torch.cpu().numpy().squeeze(0)

        # Compare results
        np.testing.assert_allclose(
            trans_torch_np,
            trans_cv2,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Torch doesn't match cv2 with different x/y scales",
        )

    def test_affine_transform_edge_cases(self):
        """Test edge cases like very small or very large scales."""
        test_cases = [
            # (center, scale, description)
            (np.array([10.0, 10.0]), np.array([20.0, 20.0]), "small values"),
            (np.array([1000.0, 1000.0]), np.array([500.0, 500.0]), "large values"),
            (np.array([0.0, 0.0]), np.array([100.0, 100.0]), "zero center"),
        ]

        for center_np, scale_np, description in test_cases:
            with self.subTest(case=description):
                center_np = center_np.astype(np.float32)
                scale_np = scale_np.astype(np.float32)

                # Get cv2 transformation
                trans_cv2 = _get_affine_transform_cv2(center_np, scale_np, self.output_size, rot=0)

                # Convert to torch
                center_torch = torch.from_numpy(center_np).unsqueeze(0).to(self.device)
                scale_torch = torch.from_numpy(scale_np).unsqueeze(0).to(self.device)

                # Get torch transformation
                trans_torch = _get_affine_transform_torch(
                    center_torch, scale_torch, self.output_size, rot=0, device=self.device
                )
                trans_torch_np = trans_torch.cpu().numpy().squeeze(0)

                # Compare results
                np.testing.assert_allclose(
                    trans_torch_np,
                    trans_cv2,
                    rtol=self.tolerance_rtol,
                    atol=self.tolerance_atol,
                    err_msg=f"Torch doesn't match cv2 for {description}",
                )

    def test_numerical_precision(self):
        """Test that float64 precision is maintained in batch computation."""
        # Create points that might have precision issues with float32
        src_np = np.array(
            [[1000.123456, 2000.654321], [1001.234567, 2000.111111], [1000.999999, 2001.888888]],
            dtype=np.float32,
        )
        dst_np = np.array(
            [[500.111111, 600.222222], [501.333333, 600.444444], [500.555555, 601.666666]], dtype=np.float32
        )

        # Get cv2 transformation (uses float64 internally)
        trans_cv2 = cv2.getAffineTransform(src_np, dst_np)

        # Convert to torch
        src_torch = torch.from_numpy(src_np).unsqueeze(0).to(self.device)
        dst_torch = torch.from_numpy(dst_np).unsqueeze(0).to(self.device)

        # Get torch transformation
        trans_torch = get_affine_transform_torch_batch(src_torch, dst_torch)
        trans_torch_np = trans_torch.cpu().numpy().squeeze(0)

        # Compare results
        np.testing.assert_allclose(
            trans_torch_np,
            trans_cv2,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Precision issue detected in torch implementation",
        )


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAffineTransformFunctions)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys

    success = run_tests()
    sys.exit(0 if success else 1)
