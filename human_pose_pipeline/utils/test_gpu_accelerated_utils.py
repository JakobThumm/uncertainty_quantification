"""
Unit tests for GPU-accelerated affine transformation utilities.

Tests verify that torch-based implementations match cv2 reference implementations.
"""

import unittest
import numpy as np
import torch
import cv2
import sys
import os

# Add parent directory to path to import transform_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_accelerated_utils import (
    get_affine_transform_torch_batch,
    _get_affine_transform_cv2,
    _get_affine_transform_torch,
    invert_affine_transform_torch_batch,
    _apply_affine_transform_gpu,
    _apply_affine_transform_batched,
    preprocess_bbox_image_gpu,
    preprocess_bbox_image_batched_gpu,
    cv2_transform_torch,
    transform_predictions_to_original_space_batched,
)

from transform_utils import transform_predictions_to_original_space

# Import triangulation functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from human_pose_pipeline.pose_estimation.triangulation_helper import (
    triangulate_point_with_covariance,
    triangulate_points_with_covariance,
    create_joint_covariance,
)
from human_pose_pipeline.utils.batched_transform_torch import (
    triangulate_points_torch,
    triangulate_point_with_covariance_torch,
    triangulate_points_with_covariance_batched,
    create_joint_covariance_batched,
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

    def test_preprocess_bbox_batched_vs_single(self):
        """Test that preprocess_bbox_image_batched_gpu matches multiple preprocess_bbox_image_gpu calls."""
        batch_size = 4
        img_size = (640, 640, 3)  # (H, W, C)
        output_size = (192, 256)  # Common pose estimation size

        # Create test images (as numpy arrays for single version)
        imgs_np = (np.random.rand(batch_size, *img_size) * 255).astype(np.uint8)

        # Create random bounding boxes for each image
        bboxes_np = np.random.rand(batch_size, 4).astype(np.float32)
        bboxes_np[:, 0] *= 50  # xmin
        bboxes_np[:, 1] *= 50  # ymin
        bboxes_np[:, 2] = bboxes_np[:, 0] + 30 + np.random.rand(batch_size) * 30  # xmax
        bboxes_np[:, 3] = bboxes_np[:, 1] + 40 + np.random.rand(batch_size) * 40  # ymax

        # Process each image individually
        single_results = []
        for i in range(batch_size):
            img_prep, center, scale, trans, proc_bbox = preprocess_bbox_image_gpu(
                imgs_np[i], bboxes_np[i].tolist(), output_size=output_size, device=self.device
            )
            single_results.append({
                'image': img_prep,
                'center': center,
                'scale': scale,
                'trans': trans,
                'bbox': proc_bbox
            })

        # Process batch
        imgs_torch = torch.from_numpy(imgs_np).to(self.device)
        bboxes_torch = torch.from_numpy(bboxes_np).to(self.device)

        img_prep_batch, center_batch, scale_batch, trans_batch, bbox_batch = preprocess_bbox_image_batched_gpu(
            imgs_torch, bboxes_torch, output_size, self.device
        )

        # Compare results for each item in batch
        for i in range(batch_size):
            with self.subTest(batch_idx=i):
                # Compare images (convert from JAX to numpy for single result)
                # Single result has shape (1, 3, H, W), need to squeeze batch dimension
                single_img = np.array(single_results[i]['image']).squeeze(0)  # JAX to numpy, (3, H, W)
                batch_img = img_prep_batch[i].cpu().numpy()  # (3, H, W)

                # Use slightly relaxed tolerance for images due to cv2 vs torch differences
                np.testing.assert_allclose(
                    single_img,
                    batch_img,
                    rtol=1e-3,  # 0.1% relative tolerance
                    atol=1e-4,  # Absolute tolerance for near-zero values
                    err_msg=f"Preprocessed images don't match for batch index {i}",
                )

                # Compare centers
                np.testing.assert_allclose(
                    single_results[i]['center'],
                    center_batch[i].cpu().numpy(),
                    rtol=self.tolerance_rtol,
                    atol=self.tolerance_atol,
                    err_msg=f"Centers don't match for batch index {i}",
                )

                # Compare scales
                np.testing.assert_allclose(
                    single_results[i]['scale'],
                    scale_batch[i].cpu().numpy(),
                    rtol=self.tolerance_rtol,
                    atol=self.tolerance_atol,
                    err_msg=f"Scales don't match for batch index {i}",
                )

                # Compare transformations
                np.testing.assert_allclose(
                    single_results[i]['trans'],
                    trans_batch[i].cpu().numpy(),
                    rtol=self.tolerance_rtol,
                    atol=self.tolerance_atol,
                    err_msg=f"Transformations don't match for batch index {i}",
                )

                # Compare processed bboxes
                np.testing.assert_allclose(
                    single_results[i]['bbox'],
                    bbox_batch[i].cpu().numpy(),
                    rtol=self.tolerance_rtol,
                    atol=self.tolerance_atol,
                    err_msg=f"Processed bboxes don't match for batch index {i}",
                )

    def test_preprocess_bbox_batched_output_shapes(self):
        """Test that preprocess_bbox_image_batched_gpu produces correct output shapes."""
        batch_size = 5
        img_size = (128, 96)  # (H, W)
        output_size = (192, 256)

        # Create test data
        imgs_torch = torch.rand(batch_size, *img_size, 3, device=self.device) * 255
        bboxes_torch = torch.rand(batch_size, 4, device=self.device) * 80 + 10
        bboxes_torch[:, 2] += bboxes_torch[:, 0]  # Ensure xmax > xmin
        bboxes_torch[:, 3] += bboxes_torch[:, 1]  # Ensure ymax > ymin

        # Process batch
        img_prep_batch, center_batch, scale_batch, trans_batch, bbox_batch = preprocess_bbox_image_batched_gpu(
            imgs_torch, bboxes_torch, output_size, self.device
        )

        # Check output shapes
        self.assertEqual(
            img_prep_batch.shape,
            (batch_size, 3, output_size[1], output_size[0]),
            "Preprocessed image shape is incorrect",
        )
        self.assertEqual(center_batch.shape, (batch_size, 2), "Center shape is incorrect")
        self.assertEqual(scale_batch.shape, (batch_size, 2), "Scale shape is incorrect")
        self.assertEqual(trans_batch.shape, (batch_size, 2, 3), "Transform shape is incorrect")
        self.assertEqual(bbox_batch.shape, (batch_size, 4), "Bbox shape is incorrect")

    def test_cv2_transform_torch_vs_cv2_transform_single(self):
        """Test that cv2_transform_torch matches cv2.transform for single batch."""
        # Create test data: N points, 2D coordinates
        num_points = 17  # e.g., number of keypoints
        src_np = np.random.rand(num_points, 2).astype(np.float32) * 100

        # Create a 2x3 affine transformation matrix
        M_2x3 = np.random.rand(2, 3).astype(np.float64)

        # cv2.transform expects (1, N, 2) and (2, 3) matrix
        src_cv2 = np.expand_dims(src_np, axis=0)  # (1, N, 2)
        dst_cv2 = cv2.transform(src_cv2, M_2x3)[0]  # (N, 2)

        # cv2_transform_torch can accept full 2x3 matrix like cv2.transform
        src_torch = torch.from_numpy(src_np).unsqueeze(0).to(self.device)  # (1, N, 2)
        M_torch = torch.from_numpy(M_2x3).to(self.device)  # (2, 3)

        dst_torch = cv2_transform_torch(src_torch, M_torch)  # (1, N, 2)
        dst_torch_np = dst_torch.cpu().numpy().squeeze(0)  # (N, 2)

        # Compare results
        np.testing.assert_allclose(
            dst_torch_np,
            dst_cv2,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="cv2_transform_torch doesn't match cv2.transform",
        )

    def test_cv2_transform_torch_vs_cv2_transform_batched(self):
        """Test that cv2_transform_torch matches cv2.transform for batched data."""
        batch_size = 5
        num_points = 17
        src_batch_np = np.random.rand(batch_size, num_points, 2).astype(np.float32) * 100

        # Create different transformation matrices for each batch item
        M_batch_np = np.random.rand(batch_size, 2, 3).astype(np.float64)

        # Process each with cv2.transform
        dst_cv2_list = []
        for i in range(batch_size):
            src_cv2 = np.expand_dims(src_batch_np[i], axis=0)  # (1, N, 2)
            dst_cv2 = cv2.transform(src_cv2, M_batch_np[i])[0]  # (N, 2)
            dst_cv2_list.append(dst_cv2)
        dst_cv2_batch = np.stack(dst_cv2_list, axis=0)  # (B, N, 2)

        # Process with cv2_transform_torch using full 2x3 format
        src_torch = torch.from_numpy(src_batch_np).to(self.device)  # (B, N, 2)
        M_torch = torch.from_numpy(M_batch_np).to(self.device)  # (B, 2, 3)

        dst_torch = cv2_transform_torch(src_torch, M_torch)  # (B, N, 2)
        dst_torch_np = dst_torch.cpu().numpy()

        # Compare results
        np.testing.assert_allclose(
            dst_torch_np,
            dst_cv2_batch,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Batched cv2_transform_torch doesn't match cv2.transform",
        )

    def test_cv2_transform_torch_no_shift(self):
        """Test cv2_transform_torch without shift parameter."""
        batch_size = 3
        num_points = 10
        src_batch_np = np.random.rand(batch_size, num_points, 2).astype(np.float32) * 50

        # Create transformation matrices (linear part only)
        M_batch_np = np.random.rand(batch_size, 2, 2).astype(np.float64)

        # Process each with cv2.transform (add zero shift for cv2)
        dst_cv2_list = []
        for i in range(batch_size):
            M_2x3 = np.hstack([M_batch_np[i], np.zeros((2, 1))])  # Add zero shift
            src_cv2 = np.expand_dims(src_batch_np[i], axis=0)
            dst_cv2 = cv2.transform(src_cv2, M_2x3)[0]
            dst_cv2_list.append(dst_cv2)
        dst_cv2_batch = np.stack(dst_cv2_list, axis=0)

        # Process with cv2_transform_torch (no shift)
        src_torch = torch.from_numpy(src_batch_np).to(self.device)
        M_torch = torch.from_numpy(M_batch_np).to(self.device)

        dst_torch = cv2_transform_torch(src_torch, M_torch, shift=None)
        dst_torch_np = dst_torch.cpu().numpy()

        # Compare results
        np.testing.assert_allclose(
            dst_torch_np,
            dst_cv2_batch,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="cv2_transform_torch without shift doesn't match cv2.transform",
        )

    def test_transform_predictions_batched_vs_single(self):
        """Test that transform_predictions_to_original_space_batched matches single version."""
        batch_size = 4
        num_joints = 17
        scale_x, scale_y = 1.5, 1.2

        # Create test data
        pred_joints_np = (np.random.rand(batch_size, num_joints, 2).astype(np.float32) - 0.5)  # Normalized coords
        trans_batch_np = np.random.rand(batch_size, 2, 3).astype(np.float32)

        # Process each individually with the single version
        single_results = []
        for i in range(batch_size):
            result = transform_predictions_to_original_space(
                pred_joints_np[i],
                trans_batch_np[i],
                scale_x,
                scale_y,
                uncertainties=None,
                covariance=None
            )
            single_results.append(result)

        # Process batch with batched version
        pred_joints_torch = torch.from_numpy(pred_joints_np).to(self.device)
        trans_torch = torch.from_numpy(trans_batch_np).to(self.device)

        batch_result = transform_predictions_to_original_space_batched(
            pred_joints_torch,
            trans_torch,
            scale_x,
            scale_y,
            uncertainties=None,
            covariance=None
        )

        # Compare results
        for i in range(batch_size):
            with self.subTest(batch_idx=i):
                np.testing.assert_allclose(
                    single_results[i]['keypoints'],
                    batch_result['keypoints'][i].cpu().numpy(),
                    rtol=1e-3,
                    atol=1e-4,
                    err_msg=f"Keypoints don't match for batch index {i}",
                )

    def test_transform_predictions_batched_with_uncertainties(self):
        """Test transform_predictions_to_original_space_batched with uncertainties."""
        batch_size = 3
        num_joints = 17
        scale_x, scale_y = 2.0, 1.8

        # Create test data with uncertainties
        pred_joints_np = (np.random.rand(batch_size, num_joints, 2).astype(np.float32) - 0.5)
        uncertainties_np = np.random.rand(batch_size, num_joints, 2).astype(np.float32) * 0.1
        covariance_np = np.random.rand(batch_size, num_joints).astype(np.float32) * 0.01
        trans_batch_np = np.random.rand(batch_size, 2, 3).astype(np.float32)

        # Process each individually
        single_results = []
        for i in range(batch_size):
            result = transform_predictions_to_original_space(
                pred_joints_np[i],
                trans_batch_np[i],
                scale_x,
                scale_y,
                uncertainties=uncertainties_np[i],
                covariance=covariance_np[i]
            )
            single_results.append(result)

        # Process batch
        pred_joints_torch = torch.from_numpy(pred_joints_np).to(self.device)
        uncertainties_torch = torch.from_numpy(uncertainties_np).to(self.device)
        covariance_torch = torch.from_numpy(covariance_np).to(self.device)
        trans_torch = torch.from_numpy(trans_batch_np).to(self.device)

        batch_result = transform_predictions_to_original_space_batched(
            pred_joints_torch,
            trans_torch,
            scale_x,
            scale_y,
            uncertainties=uncertainties_torch,
            covariance=covariance_torch
        )

        # Compare results
        for i in range(batch_size):
            with self.subTest(batch_idx=i, field='keypoints'):
                np.testing.assert_allclose(
                    single_results[i]['keypoints'],
                    batch_result['keypoints'][i].cpu().numpy(),
                    rtol=1e-3,
                    atol=1e-4,
                    err_msg=f"Keypoints don't match for batch index {i}",
                )

            with self.subTest(batch_idx=i, field='uncertainties'):
                np.testing.assert_allclose(
                    single_results[i]['uncertainties'],
                    batch_result['uncertainties'][i].cpu().numpy(),
                    rtol=1e-3,
                    atol=1e-4,
                    err_msg=f"Uncertainties don't match for batch index {i}",
                )

            with self.subTest(batch_idx=i, field='covariance'):
                np.testing.assert_allclose(
                    single_results[i]['covariance'],
                    batch_result['covariance'][i].cpu().numpy(),
                    rtol=1e-3,
                    atol=1e-4,
                    err_msg=f"Covariance doesn't match for batch index {i}",
                )


class TestTriangulationFunctions(unittest.TestCase):
    """Test cases for triangulation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tolerance_rtol = 1e-3
        self.tolerance_atol = 1e-4
        # Covariance uses numerical differentiation which amplifies float32 precision errors
        # Covariance values can range from small to millions, float32 has ~7 significant digits
        self.covariance_rtol = 1e-2  # 1% relative tolerance for float32 numerical differentiation
        self.covariance_atol = 0.01  # Absolute tolerance for very large covariance values

        # Create sample projection matrices
        self.P1 = np.array([
            [866.14, 905.29, -104.57, 2.8059e+06],
            [-104.57, 243.94, -1226.6, 3.331e+06],
            [-0.38023, 0.9025, -0.2023, 5541.1]
        ], dtype=np.float32)
        self.P2 = np.array([
            [1254.5, 4.114, -60.179, 2.1064e+06],
            [146.39, 137.42, -1233.7, 2.7093e+06],
            [0.39505, 0.88055, -0.26185, 4435.4]
        ], dtype=np.float32)

    def test_triangulate_points_torch_vs_cv2_single(self):
        """Test that triangulate_points_torch matches cv2.triangulatePoints for single point."""
        # Create 2D points (simulated projections)
        pts1_np = np.array([[320.0, 240.0]], dtype=np.float32)
        pts2_np = np.array([[280.0, 240.0]], dtype=np.float32)

        # Triangulate with cv2
        pts_4d_cv2 = cv2.triangulatePoints(
            self.P1, self.P2,
            pts1_np.T,  # cv2 expects (2, N)
            pts2_np.T
        )
        pts_3d_cv2 = (pts_4d_cv2[:3, :] / pts_4d_cv2[3, :]).T  # (N, 3)

        # Triangulate with torch
        P1_torch = torch.from_numpy(self.P1).to(self.device)
        P2_torch = torch.from_numpy(self.P2).to(self.device)
        pts1_torch = torch.from_numpy(pts1_np).to(self.device)
        pts2_torch = torch.from_numpy(pts2_np).to(self.device)

        pts_3d_torch = triangulate_points_torch(P1_torch, P2_torch, pts1_torch, pts2_torch)
        pts_3d_torch_np = pts_3d_torch.cpu().numpy()

        # Compare results
        np.testing.assert_allclose(
            pts_3d_torch_np,
            pts_3d_cv2,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="triangulate_points_torch doesn't match cv2.triangulatePoints",
        )

    def test_triangulate_points_torch_vs_cv2_multiple(self):
        """Test that triangulate_points_torch matches cv2.triangulatePoints for multiple points."""
        num_points = 10
        # Create random 2D points
        pts1_np = np.random.rand(num_points, 2).astype(np.float32) * 640
        pts2_np = np.random.rand(num_points, 2).astype(np.float32) * 640

        # Triangulate with cv2 (one at a time)
        pts_3d_cv2_list = []
        for i in range(num_points):
            pts_4d = cv2.triangulatePoints(
                self.P1, self.P2,
                pts1_np[i:i+1].T,
                pts2_np[i:i+1].T
            )
            pts_3d = (pts_4d[:3, 0] / pts_4d[3, 0])
            pts_3d_cv2_list.append(pts_3d)
        pts_3d_cv2 = np.stack(pts_3d_cv2_list, axis=0)

        # Triangulate with torch (batched)
        P1_torch = torch.from_numpy(self.P1).to(self.device)
        P2_torch = torch.from_numpy(self.P2).to(self.device)
        pts1_torch = torch.from_numpy(pts1_np).to(self.device)
        pts2_torch = torch.from_numpy(pts2_np).to(self.device)

        pts_3d_torch = triangulate_points_torch(P1_torch, P2_torch, pts1_torch, pts2_torch)
        pts_3d_torch_np = pts_3d_torch.cpu().numpy()

        # Compare results
        np.testing.assert_allclose(
            pts_3d_torch_np,
            pts_3d_cv2,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="triangulate_points_torch doesn't match cv2.triangulatePoints for multiple points",
        )

    def test_triangulate_point_with_covariance_torch_vs_numpy(self):
        """Test that triangulate_point_with_covariance_torch matches numpy version."""
        # Create test data
        pose_cam1_np = np.array([535.0259, 245.7977], dtype=np.float64)
        pose_cam2_np = np.array([504.1812, 160.5679], dtype=np.float64)

        # Create covariance matrix
        C_joint_np = np.array([[0.68762, -0.0011856, 0, 0],
                               [-0.0011856, 1.0798, 0, 0],
                               [0, 0, 0.56644, 0.011395],
                               [0, 0, 0.011395, 0.89831]], dtype=np.float64)

        # Compute with numpy version
        pt_3d_np, C_3d_np = triangulate_point_with_covariance(
            pose_cam1_np, pose_cam2_np, self.P1, self.P2, C_joint_np
        )

        # Compute with torch version
        P1_torch = torch.from_numpy(self.P1).to(self.device)
        P2_torch = torch.from_numpy(self.P2).to(self.device)
        pose_cam1_torch = torch.from_numpy(pose_cam1_np).to(self.device)
        pose_cam2_torch = torch.from_numpy(pose_cam2_np).to(self.device)
        C_joint_torch = torch.from_numpy(C_joint_np).to(self.device)

        pt_3d_torch, C_3d_torch = triangulate_point_with_covariance_torch(
            pose_cam1_torch, pose_cam2_torch, P1_torch, P2_torch, C_joint_torch
        )

        pt_3d_torch_np = pt_3d_torch.cpu().numpy()
        C_3d_torch_np = C_3d_torch.cpu().numpy()

        # Compare 3D points
        np.testing.assert_allclose(
            pt_3d_torch_np,
            pt_3d_np,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="3D points don't match between torch and numpy versions",
        )

        # Compare covariance matrices (relaxed tolerance for float32)
        np.testing.assert_allclose(
            C_3d_torch_np,
            C_3d_np,
            rtol=self.covariance_rtol,
            atol=self.covariance_atol,
            err_msg="Covariance matrices don't match between torch and numpy versions",
        )

    def test_triangulate_points_with_covariance_batched_vs_single(self):
        """Test that batched version matches multiple single calls."""
        batch_size = 4
        num_joints = 13

        # Create test data
        poses_cam1_np = np.random.rand(batch_size, num_joints, 2).astype(np.float64) * 640
        poses_cam2_np = np.random.rand(batch_size, num_joints, 2).astype(np.float64) * 640

        # Create covariance matrices
        C_joint_list_np = np.zeros((batch_size, num_joints, 4, 4), dtype=np.float64)
        for b in range(batch_size):
            for j in range(num_joints):
                C_joint_list_np[b, j] = np.eye(4, dtype=np.float64) * 0.1

        # Compute with numpy version (single calls)
        pts_3d_np_list = []
        C_3d_np_list = []
        for b in range(batch_size):
            pts_3d_batch = []
            C_3d_batch = []
            for j in range(num_joints):
                pt_3d, C_3d = triangulate_point_with_covariance(
                    poses_cam1_np[b, j], poses_cam2_np[b, j],
                    self.P1, self.P2, C_joint_list_np[b, j]
                )
                pts_3d_batch.append(pt_3d)
                C_3d_batch.append(C_3d)
            pts_3d_np_list.append(np.stack(pts_3d_batch, axis=0))
            C_3d_np_list.append(np.stack(C_3d_batch, axis=0))

        pts_3d_np = np.stack(pts_3d_np_list, axis=0)
        C_3d_np = np.stack(C_3d_np_list, axis=0)

        # Compute with batched torch version
        P1_torch = torch.from_numpy(self.P1).to(self.device)
        P2_torch = torch.from_numpy(self.P2).to(self.device)
        poses_cam1_torch = torch.from_numpy(poses_cam1_np).to(self.device)
        poses_cam2_torch = torch.from_numpy(poses_cam2_np).to(self.device)
        C_joint_list_torch = torch.from_numpy(C_joint_list_np).to(self.device)

        pts_3d_torch, C_3d_torch = triangulate_points_with_covariance_batched(
            poses_cam1_torch, poses_cam2_torch, P1_torch, P2_torch, C_joint_list_torch
        )

        pts_3d_torch_np = pts_3d_torch.cpu().numpy()
        C_3d_torch_np = C_3d_torch.cpu().numpy()

        # Compare 3D points
        np.testing.assert_allclose(
            pts_3d_torch_np,
            pts_3d_np,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Batched 3D points don't match single calls",
        )

        # Compare covariance matrices (relaxed tolerance for float32)
        np.testing.assert_allclose(
            C_3d_torch_np,
            C_3d_np,
            rtol=self.covariance_rtol,
            atol=self.covariance_atol,
            err_msg="Batched covariance matrices don't match single calls",
        )

    def test_triangulate_points_with_covariance_batched_output_shape(self):
        """Test that batched version produces correct output shapes."""
        batch_size = 3
        num_joints = 13

        # Create test data
        poses_cam1 = torch.rand(batch_size, num_joints, 2, device=self.device) * 640
        poses_cam2 = torch.rand(batch_size, num_joints, 2, device=self.device) * 640
        C_joint_list = torch.eye(4, device=self.device).unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_joints, 4, 4
        ) * 0.1

        P1_torch = torch.from_numpy(self.P1).to(self.device)
        P2_torch = torch.from_numpy(self.P2).to(self.device)

        # Compute
        pts_3d, C_3d = triangulate_points_with_covariance_batched(
            poses_cam1, poses_cam2, P1_torch, P2_torch, C_joint_list
        )

        # Check shapes
        self.assertEqual(
            pts_3d.shape,
            (batch_size, num_joints, 3),
            f"3D points shape is incorrect: {pts_3d.shape}",
        )
        self.assertEqual(
            C_3d.shape,
            (batch_size, num_joints, 3, 3),
            f"Covariance shape is incorrect: {C_3d.shape}",
        )

    def test_triangulate_points_with_covariance_batched_vs_numpy_function(self):
        """Test batched torch version matches numpy triangulate_points_with_covariance."""
        batch_size = 2
        num_joints = 13

        # Create test data
        poses_cam1_np = np.random.rand(batch_size, num_joints, 2).astype(np.float64) * 640
        poses_cam2_np = np.random.rand(batch_size, num_joints, 2).astype(np.float64) * 640

        # Create covariance matrices
        C_joint_list_np = []
        for b in range(batch_size):
            batch_C = []
            for j in range(num_joints):
                batch_C.append(np.eye(4, dtype=np.float64) * 0.1)
            C_joint_list_np.append(batch_C)

        # Compute with numpy version (using the original function)
        pts_3d_np_list = []
        C_3d_np_list = []
        for b in range(batch_size):
            pts_3d, C_3d = triangulate_points_with_covariance(
                poses_cam1_np[b], poses_cam2_np[b],
                self.P1, self.P2, C_joint_list_np[b]
            )
            pts_3d_np_list.append(pts_3d)
            C_3d_np_list.append(C_3d)

        pts_3d_np = np.stack(pts_3d_np_list, axis=0)
        C_3d_np = np.stack(C_3d_np_list, axis=0)

        # Compute with batched torch version
        P1_torch = torch.from_numpy(self.P1).to(self.device)
        P2_torch = torch.from_numpy(self.P2).to(self.device)
        poses_cam1_torch = torch.from_numpy(poses_cam1_np).to(self.device)
        poses_cam2_torch = torch.from_numpy(poses_cam2_np).to(self.device)

        # Convert list of lists to tensor
        C_joint_tensor = np.array(C_joint_list_np, dtype=np.float64)
        C_joint_list_torch = torch.from_numpy(C_joint_tensor).to(self.device)

        pts_3d_torch, C_3d_torch = triangulate_points_with_covariance_batched(
            poses_cam1_torch, poses_cam2_torch, P1_torch, P2_torch, C_joint_list_torch
        )

        pts_3d_torch_np = pts_3d_torch.cpu().numpy()
        C_3d_torch_np = C_3d_torch.cpu().numpy()

        # Compare 3D points
        np.testing.assert_allclose(
            pts_3d_torch_np,
            pts_3d_np,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Batched torch version doesn't match numpy triangulate_points_with_covariance",
        )

        # Compare covariance matrices (relaxed tolerance for float32)
        np.testing.assert_allclose(
            C_3d_torch_np,
            C_3d_np,
            rtol=self.covariance_rtol,
            atol=self.covariance_atol,
            err_msg="Batched covariance doesn't match numpy triangulate_points_with_covariance",
        )

    def test_create_joint_covariance_batched_single(self):
        """Test that create_joint_covariance_batched matches numpy version for single example."""
        # Create test data
        unc_cam1 = np.array([0.8, 1.2], dtype=np.float32)
        cov_cam1 = np.float32(0.1)
        unc_cam2 = np.array([0.7, 1.0], dtype=np.float32)
        cov_cam2 = np.float32(-0.05)

        # Compute with numpy version
        C_joint_np = create_joint_covariance(
            unc_cam1, cov_cam1, unc_cam2, cov_cam2, cross_covariance=None
        )

        # Compute with torch batched version (add batch and joint dimensions)
        unc_cam1_torch = torch.from_numpy(unc_cam1).to(self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
        cov_cam1_torch = torch.tensor([[cov_cam1]], device=self.device)  # (1, 1)
        unc_cam2_torch = torch.from_numpy(unc_cam2).to(self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
        cov_cam2_torch = torch.tensor([[cov_cam2]], device=self.device)  # (1, 1)

        C_joint_torch = create_joint_covariance_batched(
            unc_cam1_torch, cov_cam1_torch, unc_cam2_torch, cov_cam2_torch, cross_covariance=None
        )

        C_joint_torch_np = C_joint_torch[0, 0].cpu().numpy()  # Remove batch and joint dimensions

        # Compare results
        np.testing.assert_allclose(
            C_joint_torch_np,
            C_joint_np,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="create_joint_covariance_batched doesn't match numpy version",
        )

    def test_create_joint_covariance_batched_multiple(self):
        """Test create_joint_covariance_batched with multiple joints and batches."""
        batch_size = 3
        num_joints = 13

        # Create random test data
        unc_cam1_np = np.random.rand(batch_size, num_joints, 2).astype(np.float32) * 2
        cov_cam1_np = (np.random.rand(batch_size, num_joints).astype(np.float32) - 0.5) * 0.5
        unc_cam2_np = np.random.rand(batch_size, num_joints, 2).astype(np.float32) * 2
        cov_cam2_np = (np.random.rand(batch_size, num_joints).astype(np.float32) - 0.5) * 0.5

        # Compute with numpy version (loop over batch and joints)
        C_joint_np_list = []
        for b in range(batch_size):
            batch_list = []
            for j in range(num_joints):
                C_joint = create_joint_covariance(
                    unc_cam1_np[b, j], cov_cam1_np[b, j],
                    unc_cam2_np[b, j], cov_cam2_np[b, j],
                    cross_covariance=None
                )
                batch_list.append(C_joint)
            C_joint_np_list.append(np.stack(batch_list, axis=0))
        C_joint_np = np.stack(C_joint_np_list, axis=0)

        # Compute with torch batched version
        unc_cam1_torch = torch.from_numpy(unc_cam1_np).to(self.device)
        cov_cam1_torch = torch.from_numpy(cov_cam1_np).to(self.device)
        unc_cam2_torch = torch.from_numpy(unc_cam2_np).to(self.device)
        cov_cam2_torch = torch.from_numpy(cov_cam2_np).to(self.device)

        C_joint_torch = create_joint_covariance_batched(
            unc_cam1_torch, cov_cam1_torch, unc_cam2_torch, cov_cam2_torch, cross_covariance=None
        )

        C_joint_torch_np = C_joint_torch.cpu().numpy()

        # Compare results
        np.testing.assert_allclose(
            C_joint_torch_np,
            C_joint_np,
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Batched create_joint_covariance doesn't match numpy version",
        )

    def test_create_joint_covariance_batched_with_cross_covariance(self):
        """Test create_joint_covariance_batched with cross-covariance."""
        batch_size = 2
        num_joints = 5

        # Create test data
        unc_cam1 = torch.rand(batch_size, num_joints, 2, device=self.device) * 2
        cov_cam1 = (torch.rand(batch_size, num_joints, device=self.device) - 0.5) * 0.5
        unc_cam2 = torch.rand(batch_size, num_joints, 2, device=self.device) * 2
        cov_cam2 = (torch.rand(batch_size, num_joints, device=self.device) - 0.5) * 0.5
        cross_cov = (torch.rand(batch_size, num_joints, 2, 2, device=self.device) - 0.5) * 0.3

        # Compute with batched version
        C_joint = create_joint_covariance_batched(
            unc_cam1, cov_cam1, unc_cam2, cov_cam2, cross_covariance=cross_cov
        )

        # Check shape
        self.assertEqual(
            C_joint.shape,
            (batch_size, num_joints, 4, 4),
            f"Output shape is incorrect: {C_joint.shape}",
        )

        # Verify structure: check that cross-covariance blocks are symmetric
        upper_right = C_joint[:, :, 0:2, 2:4]  # (B, N, 2, 2)
        lower_left = C_joint[:, :, 2:4, 0:2]  # (B, N, 2, 2)

        np.testing.assert_allclose(
            upper_right.cpu().numpy(),
            cross_cov.cpu().numpy(),
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Upper-right block doesn't match input cross-covariance",
        )

        np.testing.assert_allclose(
            lower_left.cpu().numpy(),
            cross_cov.transpose(-2, -1).cpu().numpy(),
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Lower-left block is not transpose of cross-covariance",
        )

    def test_create_joint_covariance_batched_output_structure(self):
        """Test that create_joint_covariance_batched produces correct structure."""
        batch_size = 2
        num_joints = 3

        # Create simple test data
        unc_cam1 = torch.ones(batch_size, num_joints, 2, device=self.device)  # std = 1
        cov_cam1 = torch.zeros(batch_size, num_joints, device=self.device)  # no covariance
        unc_cam2 = torch.ones(batch_size, num_joints, 2, device=self.device) * 2  # std = 2
        cov_cam2 = torch.zeros(batch_size, num_joints, device=self.device)

        C_joint = create_joint_covariance_batched(
            unc_cam1, cov_cam1, unc_cam2, cov_cam2, cross_covariance=None
        )

        # Check diagonal values (variances)
        expected_diag = torch.tensor([1.0, 1.0, 4.0, 4.0], device=self.device)
        actual_diag = torch.diagonal(C_joint[0, 0], dim1=0, dim2=1)

        np.testing.assert_allclose(
            actual_diag.cpu().numpy(),
            expected_diag.cpu().numpy(),
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Diagonal values (variances) are incorrect",
        )

        # Check that matrix is symmetric
        C_joint_T = C_joint.transpose(-2, -1)
        np.testing.assert_allclose(
            C_joint.cpu().numpy(),
            C_joint_T.cpu().numpy(),
            rtol=self.tolerance_rtol,
            atol=self.tolerance_atol,
            err_msg="Covariance matrix is not symmetric",
        )


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAffineTransformFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestTriangulationFunctions))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import sys

    success = run_tests()
    sys.exit(0 if success else 1)
