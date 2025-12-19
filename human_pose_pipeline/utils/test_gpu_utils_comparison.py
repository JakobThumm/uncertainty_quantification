"""
Test file to validate that PyTorch and JAX implementations produce identical results.

This test suite compares outputs from gpu_accelerated_utils.py (PyTorch) and
gpu_accelerated_utils_jax.py (JAX) to ensure correctness of the conversion.
"""

import pytest
import numpy as np
import torch
import jax.numpy as jnp
from PIL import Image

# Import PyTorch version
from human_pose_pipeline.utils import gpu_accelerated_utils as pytorch_utils

# Import JAX version
from human_pose_pipeline.utils import gpu_accelerated_utils_jax as jax_utils


def allclose(torch_tensor, jax_array, rtol=1e-5, atol=1e-5):
    """Helper function to compare PyTorch tensor and JAX array."""
    if isinstance(torch_tensor, torch.Tensor):
        torch_np = torch_tensor.cpu().detach().numpy()
    else:
        torch_np = torch_tensor

    if isinstance(jax_array, jnp.ndarray):
        jax_np = np.array(jax_array)
    else:
        jax_np = jax_array

    return np.allclose(torch_np, jax_np, rtol=rtol, atol=atol)


class TestAffineFunctions:
    """Test affine transformation functions."""

    def test_get_affine_transform_torch_batch(self):
        """Test batched affine transform computation."""
        B = 4
        src = np.random.randn(B, 3, 2).astype(np.float32)
        dst = np.random.randn(B, 3, 2).astype(np.float32)

        # PyTorch version
        src_torch = torch.from_numpy(src)
        dst_torch = torch.from_numpy(dst)
        result_torch = pytorch_utils.get_affine_transform_torch_batch(src_torch, dst_torch)

        # JAX version
        src_jax = jnp.array(src)
        dst_jax = jnp.array(dst)
        result_jax = jax_utils.get_affine_transform_batch(src_jax, dst_jax)

        assert allclose(result_torch, result_jax), "Affine transform results don't match"
        print("✓ get_affine_transform_torch_batch: PASSED")

    def test_invert_affine_transform_torch_batch(self):
        """Test batched affine transform inversion."""
        B = 4
        M = np.random.randn(B, 2, 3).astype(np.float32)
        # Make sure it's invertible by avoiding near-zero determinants
        M[:, 0, 0] += 2.0  # Ensure diagonal dominance
        M[:, 1, 1] += 2.0

        # PyTorch version
        M_torch = torch.from_numpy(M)
        result_torch = pytorch_utils.invert_affine_transform_torch_batch(M_torch)

        # JAX version
        M_jax = jnp.array(M)
        result_jax = jax_utils.invert_affine_transform_batch(M_jax)

        assert allclose(result_torch, result_jax), "Inverted affine transform results don't match"
        print("✓ invert_affine_transform_torch_batch: PASSED")

    def test_cv2_transform_torch(self):
        """Test cv2.transform equivalent."""
        B, N = 4, 17
        src = np.random.randn(B, N, 2).astype(np.float32)
        M = np.random.randn(B, 2, 3).astype(np.float32)

        # PyTorch version
        src_torch = torch.from_numpy(src)
        M_torch = torch.from_numpy(M)
        result_torch = pytorch_utils.cv2_transform_torch(src_torch, M_torch)

        # JAX version
        src_jax = jnp.array(src)
        M_jax = jnp.array(M)
        result_jax = jax_utils.cv2_transform(src_jax, M_jax)

        assert allclose(result_torch, result_jax), "cv2_transform results don't match"
        print("✓ cv2_transform_torch: PASSED")


class TestTransformPredictions:
    """Test prediction transformation functions."""

    def test_transform_predictions_to_original_space_batched(self):
        """Test transformation of predictions back to original space."""
        B, N = 4, 17
        pred_joints = np.random.randn(B, N, 2).astype(np.float32) * 0.3  # In [-0.5, 0.5] range
        trans = np.random.randn(B, 2, 3).astype(np.float32)
        scale_x, scale_y = 1.5, 1.8

        # PyTorch version
        pred_joints_torch = torch.from_numpy(pred_joints)
        trans_torch = torch.from_numpy(trans)
        result_torch = pytorch_utils.transform_predictions_to_original_space_batched(
            pred_joints_torch, trans_torch, scale_x, scale_y
        )

        # JAX version
        pred_joints_jax = jnp.array(pred_joints)
        trans_jax = jnp.array(trans)
        result_jax = jax_utils.transform_predictions_to_original_space_batched(
            pred_joints_jax, trans_jax, scale_x, scale_y
        )

        assert allclose(result_torch['keypoints'], result_jax['keypoints']), \
            "Transformed keypoints don't match"
        print("✓ transform_predictions_to_original_space_batched: PASSED")

    def test_transform_predictions_with_uncertainties(self):
        """Test transformation with uncertainties and covariance."""
        B, N = 4, 17
        pred_joints = np.random.randn(B, N, 2).astype(np.float32) * 0.3
        trans = np.random.randn(B, 2, 3).astype(np.float32)
        uncertainties = np.random.rand(B, N, 2).astype(np.float32) * 0.1
        covariance = np.random.rand(B, N).astype(np.float32) * 0.01
        scale_x, scale_y = 1.5, 1.8

        # PyTorch version (note: PyTorch modifies uncertainties in-place!)
        pred_joints_torch = torch.from_numpy(pred_joints.copy())
        trans_torch = torch.from_numpy(trans.copy())
        uncertainties_torch = torch.from_numpy(uncertainties.copy())
        covariance_torch = torch.from_numpy(covariance.copy())
        result_torch = pytorch_utils.transform_predictions_to_original_space_batched(
            pred_joints_torch, trans_torch, scale_x, scale_y,
            uncertainties=uncertainties_torch, covariance=covariance_torch
        )

        # JAX version
        pred_joints_jax = jnp.array(pred_joints.copy())
        trans_jax = jnp.array(trans.copy())
        uncertainties_jax = jnp.array(uncertainties.copy())
        covariance_jax = jnp.array(covariance.copy())
        result_jax = jax_utils.transform_predictions_to_original_space_batched(
            pred_joints_jax, trans_jax, scale_x, scale_y,
            uncertainties=uncertainties_jax, covariance=covariance_jax
        )

        assert allclose(result_torch['keypoints'], result_jax['keypoints']), \
            "Keypoints don't match"
        assert allclose(result_torch['uncertainties'], result_jax['uncertainties']), \
            "Uncertainties don't match"
        assert allclose(result_torch['covariance'], result_jax['covariance']), \
            "Covariance doesn't match"
        print("✓ transform_predictions_with_uncertainties: PASSED")


class TestBBoxPreprocessing:
    """Test bounding box preprocessing functions."""

    def test_get_affine_transform_jax_vs_torch(self):
        """Test JAX affine transform against PyTorch version."""
        B = 4
        center = np.random.randn(B, 2).astype(np.float32) * 100 + 200
        scale = np.random.rand(B, 2).astype(np.float32) * 50 + 100
        output_size = (256, 256)

        # PyTorch version
        center_torch = torch.from_numpy(center)
        scale_torch = torch.from_numpy(scale)
        trans_torch = pytorch_utils._get_affine_transform_torch(
            center_torch, scale_torch, output_size, device='cpu'
        )

        # JAX version
        center_jax = jnp.array(center)
        scale_jax = jnp.array(scale)
        trans_jax = jax_utils._get_affine_transform_jax(
            center_jax, scale_jax, output_size
        )

        assert allclose(trans_torch, trans_jax, rtol=1e-4, atol=1e-4), \
            "Affine transforms don't match"
        print("✓ _get_affine_transform (jax vs torch): PASSED")

    def test_center_scale_to_box_batched(self):
        """Test center/scale to bbox conversion."""
        B = 4
        center = np.random.randn(B, 2).astype(np.float32) * 100 + 200
        scale = np.random.rand(B, 2).astype(np.float32) * 50 + 100

        # PyTorch version
        center_torch = torch.from_numpy(center)
        scale_torch = torch.from_numpy(scale)
        bbox_torch = pytorch_utils._center_scale_to_box_batched(center_torch, scale_torch)

        # JAX version
        center_jax = jnp.array(center)
        scale_jax = jnp.array(scale)
        bbox_jax = jax_utils._center_scale_to_box_batched(center_jax, scale_jax)

        assert allclose(bbox_torch, bbox_jax), "Bounding boxes don't match"
        print("✓ _center_scale_to_box_batched: PASSED")

    # Commented out - preprocess_bbox_image_batched_gpu not yet implemented in JAX
    def _test_preprocess_bbox_image_batched_gpu(self):
        """Test batched bbox preprocessing."""
        B = 2
        H, W = 480, 640
        resized_images_np = np.random.randint(0, 255, (B, H, W, 3), dtype=np.uint8).astype(np.float32)
        bboxes_np = np.array([
            [100, 100, 300, 400],
            [150, 120, 350, 420]
        ], dtype=np.float32)
        output_size = (256, 256)

        # PyTorch version
        resized_torch = torch.from_numpy(resized_images_np)
        bboxes_torch = torch.from_numpy(bboxes_np)
        (img_prep_torch, center_torch, scale_torch,
         trans_torch, proc_bbox_torch) = pytorch_utils.preprocess_bbox_image_batched_gpu(
            resized_torch, bboxes_torch, output_size, device='cpu'
        )

        # JAX version
        resized_jax = jnp.array(resized_images_np)
        bboxes_jax = jnp.array(bboxes_np)
        (img_prep_jax, center_jax, scale_jax,
         trans_jax, proc_bbox_jax) = jax_utils.preprocess_bbox_image_batched_gpu(
            resized_jax, bboxes_jax, output_size, device='cpu'
        )

        # Compare all outputs
        assert allclose(center_torch, center_jax), "Centers don't match"
        assert allclose(scale_torch, scale_jax), "Scales don't match"
        assert allclose(trans_torch, trans_jax, rtol=1e-4, atol=1e-4), "Transforms don't match"
        assert allclose(proc_bbox_torch, proc_bbox_jax), "Processed bboxes don't match"

        # Compare preprocessed images with tolerance for interpolation
        diff = np.abs(img_prep_torch.cpu().numpy() - np.array(img_prep_jax))
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"  Preprocess bbox - Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}")
        # More lenient thresholds due to interpolation and normalization differences
        assert max_diff < 1.0, f"Max preprocessed image difference too large: {max_diff}"
        assert mean_diff < 0.3, f"Mean preprocessed image difference too large: {mean_diff}"
        print("✓ preprocess_bbox_image_batched_gpu: PASSED (within interpolation tolerance)")


class TestEndToEnd:
    """End-to-end integration tests."""

    # Commented out - depends on preprocess_bbox_image_batched_gpu not yet implemented in JAX
    def _test_full_pipeline_batched(self):
        """Test full pipeline from image to transformed predictions."""
        B = 2
        N = 17  # Number of joints
        H, W = 480, 640

        # Create test data
        images_np = np.random.randint(0, 255, (B, H, W, 3), dtype=np.uint8).astype(np.float32)
        bboxes_np = np.array([
            [100, 100, 300, 400],
            [150, 120, 350, 420]
        ], dtype=np.float32)
        pred_joints_np = np.random.randn(B, N, 2).astype(np.float32) * 0.3
        scale_x, scale_y = 1.5, 1.8
        output_size = (256, 256)

        # PyTorch pipeline
        images_torch = torch.from_numpy(images_np)
        bboxes_torch = torch.from_numpy(bboxes_np)
        pred_joints_torch = torch.from_numpy(pred_joints_np)

        (img_prep_torch, center_torch, scale_torch,
         trans_torch, proc_bbox_torch) = pytorch_utils.preprocess_bbox_image_batched_gpu(
            images_torch, bboxes_torch, output_size, device='cpu'
        )

        result_torch = pytorch_utils.transform_predictions_to_original_space_batched(
            pred_joints_torch, trans_torch, scale_x, scale_y
        )

        # JAX pipeline
        images_jax = jnp.array(images_np)
        bboxes_jax = jnp.array(bboxes_np)
        pred_joints_jax = jnp.array(pred_joints_np)

        (img_prep_jax, center_jax, scale_jax,
         trans_jax, proc_bbox_jax) = jax_utils.preprocess_bbox_image_batched_gpu(
            images_jax, bboxes_jax, output_size, device='cpu'
        )

        result_jax = jax_utils.transform_predictions_to_original_space_batched(
            pred_joints_jax, trans_jax, scale_x, scale_y
        )

        # Compare final outputs
        assert allclose(result_torch['keypoints'], result_jax['keypoints'], rtol=1e-4, atol=1e-3), \
            "End-to-end keypoints don't match"

        print("✓ Full pipeline (end-to-end): PASSED")


class TestTriangulationFunctions:
    """Test triangulation and covariance propagation functions."""

    def test_triangulate_points(self):
        """Test basic triangulation."""
        # Create simple projection matrices
        P1 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float32)

        P2 = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float32)

        # Create test 2D points
        B, N = 4, 10
        pts1 = np.random.randn(B, N, 2).astype(np.float32)
        pts2 = np.random.randn(B, N, 2).astype(np.float32)

        # PyTorch version
        from human_pose_pipeline.utils.batched_transform_torch import triangulate_points_torch
        P1_torch = torch.from_numpy(P1)
        P2_torch = torch.from_numpy(P2)
        pts1_torch = torch.from_numpy(pts1)
        pts2_torch = torch.from_numpy(pts2)

        result_torch = triangulate_points_torch(P1_torch, P2_torch, pts1_torch, pts2_torch)

        # JAX version
        P1_jax = jnp.array(P1)
        P2_jax = jnp.array(P2)
        pts1_jax = jnp.array(pts1)
        pts2_jax = jnp.array(pts2)

        result_jax = jax_utils.triangulate_points(P1_jax, P2_jax, pts1_jax, pts2_jax)

        assert allclose(result_torch, result_jax, rtol=1e-4, atol=1e-4), \
            "Triangulated points don't match"
        print("✓ triangulate_points: PASSED")

    def test_create_joint_covariance_batched(self):
        """Test joint covariance matrix creation."""
        B, N = 4, 10
        unc_cam1 = np.random.rand(B, N, 2).astype(np.float32) * 0.1
        cov_cam1 = np.random.rand(B, N).astype(np.float32) * 0.01
        unc_cam2 = np.random.rand(B, N, 2).astype(np.float32) * 0.1
        cov_cam2 = np.random.rand(B, N).astype(np.float32) * 0.01

        # PyTorch version
        from human_pose_pipeline.utils.batched_transform_torch import create_joint_covariance_batched
        unc_cam1_torch = torch.from_numpy(unc_cam1)
        cov_cam1_torch = torch.from_numpy(cov_cam1)
        unc_cam2_torch = torch.from_numpy(unc_cam2)
        cov_cam2_torch = torch.from_numpy(cov_cam2)

        result_torch = create_joint_covariance_batched(
            unc_cam1_torch, cov_cam1_torch, unc_cam2_torch, cov_cam2_torch
        )

        # JAX version
        unc_cam1_jax = jnp.array(unc_cam1)
        cov_cam1_jax = jnp.array(cov_cam1)
        unc_cam2_jax = jnp.array(unc_cam2)
        cov_cam2_jax = jnp.array(cov_cam2)

        result_jax = jax_utils.create_joint_covariance_batched(
            unc_cam1_jax, cov_cam1_jax, unc_cam2_jax, cov_cam2_jax
        )

        assert allclose(result_torch, result_jax), \
            "Joint covariance matrices don't match"
        print("✓ create_joint_covariance_batched: PASSED")

    def test_triangulate_points_with_covariance_batched(self):
        """Test triangulation with covariance propagation."""
        # Create projection matrices
        P1 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float32)

        P2 = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ], dtype=np.float32)

        # Create test data
        B, N = 2, 5
        poses_cam1 = np.random.randn(B, N, 2).astype(np.float32)
        poses_cam2 = np.random.randn(B, N, 2).astype(np.float32)

        # Create covariance matrices
        unc_cam1 = np.random.rand(B, N, 2).astype(np.float32) * 0.1
        cov_cam1 = np.random.rand(B, N).astype(np.float32) * 0.01
        unc_cam2 = np.random.rand(B, N, 2).astype(np.float32) * 0.1
        cov_cam2 = np.random.rand(B, N).astype(np.float32) * 0.01

        # PyTorch version
        from human_pose_pipeline.utils.batched_transform_torch import (
            create_joint_covariance_batched,
            triangulate_points_with_covariance_batched
        )

        poses_cam1_torch = torch.from_numpy(poses_cam1.copy())
        poses_cam2_torch = torch.from_numpy(poses_cam2.copy())
        P1_torch = torch.from_numpy(P1)
        P2_torch = torch.from_numpy(P2)

        C_joint_torch = create_joint_covariance_batched(
            torch.from_numpy(unc_cam1),
            torch.from_numpy(cov_cam1),
            torch.from_numpy(unc_cam2),
            torch.from_numpy(cov_cam2)
        )

        points_3d_torch, C_3d_torch = triangulate_points_with_covariance_batched(
            poses_cam1_torch, poses_cam2_torch, P1_torch, P2_torch, C_joint_torch
        )

        # JAX version
        poses_cam1_jax = jnp.array(poses_cam1.copy())
        poses_cam2_jax = jnp.array(poses_cam2.copy())
        P1_jax = jnp.array(P1)
        P2_jax = jnp.array(P2)

        C_joint_jax = jax_utils.create_joint_covariance_batched(
            jnp.array(unc_cam1),
            jnp.array(cov_cam1),
            jnp.array(unc_cam2),
            jnp.array(cov_cam2)
        )

        points_3d_jax, C_3d_jax = jax_utils.triangulate_points_with_covariance_batched(
            poses_cam1_jax, poses_cam2_jax, P1_jax, P2_jax, C_joint_jax
        )

        # Compare results
        assert allclose(points_3d_torch, points_3d_jax, rtol=1e-4, atol=1e-4), \
            "Triangulated 3D points don't match"
        # Slightly higher tolerance for covariance due to numerical differences in finite difference Jacobian
        assert allclose(C_3d_torch, C_3d_jax, rtol=1e-2, atol=1e-2), \
            "3D covariance matrices don't match"
        print("✓ triangulate_points_with_covariance_batched: PASSED")


def run_all_tests():
    """Run all test classes."""
    print("\n" + "="*70)
    print("Testing PyTorch vs JAX GPU Accelerated Utils")
    print("="*70 + "\n")

    test_classes = [
        TestAffineFunctions,
        TestTransformPredictions,
        TestBBoxPreprocessing,
        TestEndToEnd,
        TestTriangulationFunctions,
    ]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 70)

        test_instance = test_class()
        test_methods = [method for method in dir(test_instance)
                       if method.startswith('test_') and callable(getattr(test_instance, method))]

        for test_method_name in test_methods:
            total_tests += 1
            try:
                test_method = getattr(test_instance, test_method_name)
                test_method()
                passed_tests += 1
            except AssertionError as e:
                print(f"✗ {test_method_name}: FAILED - {e}")
            except Exception as e:
                print(f"✗ {test_method_name}: ERROR - {e}")

    print("\n" + "="*70)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print("="*70 + "\n")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
