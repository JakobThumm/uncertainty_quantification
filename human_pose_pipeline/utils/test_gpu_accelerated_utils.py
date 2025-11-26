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
    invert_affine_transform_torch_batch,
    _apply_affine_transform_gpu,
    _apply_affine_transform_batched,
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

    def test_invert_affine_transform_single(self):
        """Test that invert_affine_transform_torch_batch matches cv2.invertAffineTransform for single example."""
        # Create a random affine transformation
        src_np = np.array([[100.0, 100.0], [150.0, 100.0], [100.0, 150.0]], dtype=np.float32)
        dst_np = np.array([[50.0, 50.0], [100.0, 60.0], [45.0, 95.0]], dtype=np.float32)

        # Get affine transformation
        trans_cv2 = cv2.getAffineTransform(src_np, dst_np)

        # Invert using cv2
        trans_inv_cv2 = cv2.invertAffineTransform(trans_cv2)

        # Convert to torch and invert
        trans_torch = torch.from_numpy(trans_cv2).unsqueeze(0).to(self.device)  # (1, 2, 3)
        trans_inv_torch = invert_affine_transform_torch_batch(trans_torch)
        trans_inv_torch_np = trans_inv_torch.cpu().numpy().squeeze(0)

        # Compare results
        np.testing.assert_allclose(
            trans_inv_torch_np,
            trans_inv_cv2,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="invert_affine_transform_torch_batch doesn't match cv2.invertAffineTransform",
        )

    def test_invert_affine_transform_batched(self):
        """Test that invert_affine_transform_torch_batch matches cv2.invertAffineTransform for batched examples."""
        batch_size = 10

        # Create random affine transformations
        src_batch_np = np.random.rand(batch_size, 3, 2).astype(np.float32) * 200
        dst_batch_np = np.random.rand(batch_size, 3, 2).astype(np.float32) * 200

        # Get affine transformations and their inverses using cv2
        trans_cv2_list = []
        trans_inv_cv2_list = []
        for i in range(batch_size):
            trans_cv2 = cv2.getAffineTransform(src_batch_np[i], dst_batch_np[i])
            trans_inv_cv2 = cv2.invertAffineTransform(trans_cv2)
            trans_cv2_list.append(trans_cv2)
            trans_inv_cv2_list.append(trans_inv_cv2)

        trans_cv2_batch = np.stack(trans_cv2_list, axis=0)
        trans_inv_cv2_batch = np.stack(trans_inv_cv2_list, axis=0)

        # Convert to torch and invert
        trans_torch = torch.from_numpy(trans_cv2_batch).to(self.device)
        trans_inv_torch = invert_affine_transform_torch_batch(trans_torch)
        trans_inv_torch_np = trans_inv_torch.cpu().numpy()

        # Compare results
        np.testing.assert_allclose(
            trans_inv_torch_np,
            trans_inv_cv2_batch,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Batched invert_affine_transform_torch_batch doesn't match cv2.invertAffineTransform",
        )

    def test_invert_affine_transform_composition(self):
        """Test that M * inv(M) = Identity."""
        batch_size = 5

        # Create random affine transformations
        src_batch_np = np.random.rand(batch_size, 3, 2).astype(np.float32) * 200 + 50
        dst_batch_np = np.random.rand(batch_size, 3, 2).astype(np.float32) * 200 + 50

        # Get affine transformations
        trans_cv2_list = []
        for i in range(batch_size):
            trans_cv2 = cv2.getAffineTransform(src_batch_np[i], dst_batch_np[i])
            trans_cv2_list.append(trans_cv2)

        trans_cv2_batch = np.stack(trans_cv2_list, axis=0)

        # Convert to torch with float32
        trans_torch = torch.from_numpy(trans_cv2_batch).to(self.device).float()
        trans_inv_torch = invert_affine_transform_torch_batch(trans_torch).float()

        # Apply transformation then inverse to a point
        # Create test points
        test_points = torch.tensor([[100.0, 150.0]], dtype=torch.float32, device=self.device).expand(
            batch_size, 2
        )  # (B, 2)

        # Apply forward transformation: [x', y'] = M @ [x, y, 1]
        test_points_hom = torch.cat([test_points, torch.ones(batch_size, 1, dtype=torch.float32, device=self.device)], dim=1)  # (B, 3)
        transformed = torch.bmm(trans_torch, test_points_hom.unsqueeze(-1)).squeeze(-1)  # (B, 2)

        # Apply inverse transformation
        transformed_hom = torch.cat(
            [transformed, torch.ones(batch_size, 1, dtype=torch.float32, device=self.device)], dim=1
        )  # (B, 3)
        recovered = torch.bmm(trans_inv_torch, transformed_hom.unsqueeze(-1)).squeeze(-1)  # (B, 2)

        # Should recover original points
        np.testing.assert_allclose(
            recovered.cpu().numpy(),
            test_points.cpu().numpy(),
            rtol=1e-3,
            atol=1e-3,
            err_msg="Composition of transform and inverse doesn't yield identity",
        )

    def test_apply_affine_transform_consistency(self):
        """Test that _apply_affine_transform_gpu and _apply_affine_transform_batched produce the same output."""
        # Create a test image
        img_size = (64, 48)  # (H, W)
        img_np = np.random.rand(*img_size, 3).astype(np.float32)
        img_torch_single = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Create an affine transformation
        center_np = np.array([32.0, 24.0], dtype=np.float32)
        scale_np = np.array([40.0, 40.0], dtype=np.float32)
        output_size = (32, 32)

        trans_cv2 = _get_affine_transform_cv2(center_np, scale_np, output_size, rot=0)
        trans_torch_single = torch.from_numpy(trans_cv2).to(self.device)  # (2, 3)
        trans_torch_batch = trans_torch_single.unsqueeze(0)  # (1, 2, 3)

        # Apply transformation using single version
        result_single = _apply_affine_transform_gpu(img_torch_single, trans_torch_single, output_size)

        # Apply transformation using batched version
        result_batched = _apply_affine_transform_batched(img_torch_single, trans_torch_batch, output_size, device=self.device)

        # Compare results
        np.testing.assert_allclose(
            result_single.cpu().numpy(),
            result_batched.cpu().numpy(),
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="_apply_affine_transform_gpu and _apply_affine_transform_batched produce different outputs",
        )

    def test_apply_affine_transform_batched_multiple(self):
        """Test that _apply_affine_transform_batched produces consistent results for multiple images."""
        batch_size = 4
        img_size = (64, 48)  # (H, W)
        output_size = (32, 32)

        # Create test images
        imgs_np = np.random.rand(batch_size, *img_size, 3).astype(np.float32)
        imgs_torch = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).to(self.device)  # (B, 3, H, W)

        # Create affine transformations for each image
        centers_np = np.random.rand(batch_size, 2).astype(np.float32) * 40 + 12
        scales_np = np.random.rand(batch_size, 2).astype(np.float32) * 20 + 30

        trans_cv2_list = []
        result_single_list = []
        for i in range(batch_size):
            trans_cv2 = _get_affine_transform_cv2(centers_np[i], scales_np[i], output_size, rot=0)
            trans_cv2_list.append(trans_cv2)

            # Apply single transformation
            trans_torch = torch.from_numpy(trans_cv2).to(self.device)
            img_single = imgs_torch[i : i + 1]  # (1, 3, H, W)
            result_single = _apply_affine_transform_gpu(img_single, trans_torch, output_size)
            result_single_list.append(result_single)

        # Stack single results
        result_single_stacked = torch.cat(result_single_list, dim=0)

        # Apply batched transformation
        trans_torch_batch = torch.from_numpy(np.stack(trans_cv2_list, axis=0)).to(self.device)
        result_batched = _apply_affine_transform_batched(imgs_torch, trans_torch_batch, output_size, device=self.device)

        # Compare results
        np.testing.assert_allclose(
            result_single_stacked.cpu().numpy(),
            result_batched.cpu().numpy(),
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="_apply_affine_transform_batched doesn't match individual _apply_affine_transform_gpu calls",
        )

    def test_apply_affine_transform_output_shape(self):
        """Test that apply affine transform produces correct output shapes."""
        batch_size = 3
        input_size = (64, 48)  # (H, W)
        output_size = (32, 32)

        # Create test images
        imgs_torch = torch.rand(batch_size, 3, *input_size, device=self.device)

        # Create transformations
        centers = torch.rand(batch_size, 2, device=self.device) * 40 + 12
        scales = torch.rand(batch_size, 2, device=self.device) * 20 + 30
        trans_torch = _get_affine_transform_torch(centers, scales, output_size, rot=0, device=self.device)

        # Apply batched transformation
        result = _apply_affine_transform_batched(imgs_torch, trans_torch, output_size, device=self.device)

        # Check output shape
        expected_shape = (batch_size, 3, output_size[1], output_size[0])  # (B, C, H, W)
        self.assertEqual(
            result.shape,
            expected_shape,
            f"Output shape {result.shape} doesn't match expected {expected_shape}",
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
