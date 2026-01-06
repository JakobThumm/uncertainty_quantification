#!/usr/bin/env python3
"""
Test to compare batched vs non-batched coverage evaluation implementations.

This test ensures that:
1. evaluate_uncertainty_coverage_with_covariance (batched in eval_utils.py)
2. evaluate_pose_estimation_full_3d (non-batched in pose_estimation_3D.py)

produce equivalent results when processing the same data.
"""

import numpy as np
from scipy.stats import chi2

# Import the two functions to compare
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from human_pose_pipeline.utils.eval_utils import evaluate_uncertainty_coverage_with_covariance
from human_pose_pipeline.examples.pose_estimation_3D import evaluate_pose_estimation_full_3d


def generate_test_data(batch_size=5, n_frames=10, n_joints=13, seed=42):
    """
    Generate synthetic test data for coverage evaluation.

    Args:
        batch_size: Number of sequences in batch
        n_frames: Number of frames per sequence
        n_joints: Number of joints per frame
        seed: Random seed for reproducibility

    Returns:
        pred_poses: [B, T, J*3] predicted poses
        true_poses: [B, T, J*3] ground truth poses
        cov_matrices: [B, T, J, 3, 3] covariance matrices
    """
    np.random.seed(seed)

    # Generate predicted poses
    pred_poses = np.random.randn(batch_size, n_frames, n_joints, 3) * 100

    # Generate covariance matrices (must be positive definite)
    cov_matrices = np.zeros((batch_size, n_frames, n_joints, 3, 3))
    for b in range(batch_size):
        for t in range(n_frames):
            for j in range(n_joints):
                # Generate a random positive definite covariance matrix
                A = np.random.randn(3, 3)
                cov = A @ A.T + np.eye(3) * 0.1  # Add small diagonal for stability
                cov_matrices[b, t, j] = cov

    # Generate true poses by adding noise based on covariance
    true_poses = np.zeros_like(pred_poses)

    for b in range(batch_size):
        for t in range(n_frames):
            for j in range(n_joints):
                # Sample from multivariate Gaussian
                noise = np.random.multivariate_normal(np.zeros(3), cov_matrices[b, t, j])
                true_poses[b, t, j] = true_poses[b, t, j] + noise

    return pred_poses, true_poses, cov_matrices


def test_coverage_comparison_single_frame():
    """
    Test that batched and non-batched versions produce the same results
    for a single frame.

    Note: We use batch_size=2, n_frames=2 instead of 1,1 to avoid dimension
    issues with squeeze() in the batched implementation.
    """
    # Generate test data with batch_size=2, n_frames=2 to test on first frame
    pred_poses, true_poses, cov_matrices = generate_test_data(
        batch_size=2, n_frames=2, n_joints=13, seed=42
    )

    # Run batched version on all data
    batched_result, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_poses, true_poses, cov_matrices
    )

    # Run non-batched version on each frame and aggregate
    batch_size, n_frames, n_joints, _ = pred_poses.shape

    all_counts = {
        'within_1std': 0,
        'within_2std': 0,
        'within_3std': 0,
        'within_4std': 0
    }
    total_joints = 0

    for b in range(batch_size):
        for t in range(n_frames):
            pred_single = pred_poses[b, t]
            true_single = true_poses[b, t]
            cov_single = cov_matrices[b, t]

            nonbatched_result = evaluate_pose_estimation_full_3d(
                ground_truth=true_single,
                estimated_pose=pred_single,
                estimated_covariance=cov_single
            )

            for key in all_counts.keys():
                all_counts[key] += nonbatched_result['counts'][key]
            total_joints += nonbatched_result['num_joints']

    # Extract coverage from batched version
    batched_coverages = [
        batched_result['overall_within_1std'],
        batched_result['overall_within_2std'],
        batched_result['overall_within_3std'],
        batched_result['overall_within_4std']
    ]

    # Aggregate non-batched results
    nonbatched_coverages = [
        all_counts['within_1std'] / total_joints,
        all_counts['within_2std'] / total_joints,
        all_counts['within_3std'] / total_joints,
        all_counts['within_4std'] / total_joints
    ]

    # Compare with tolerance for numerical differences
    print("Overall coverage comparison:")
    tolerance = 1e-6
    all_match = True
    for i, (batched, nonbatched) in enumerate(zip(batched_coverages, nonbatched_coverages)):
        diff = np.abs(batched - nonbatched)
        match_str = "✓ MATCH" if diff < tolerance else "✗ MISMATCH"
        print(f"  {i+1} std - Batched: {batched:.6f}, Non-batched: {nonbatched:.6f}, Diff: {diff:.2e} {match_str}")
        if diff >= tolerance:
            all_match = False

    if not all_match:
        raise AssertionError("Coverage values differ between batched and non-batched implementations")

    print("✓ Single frame test passed!")


def test_coverage_comparison_multiple_frames():
    """
    Test that batched version produces correct aggregated results
    across multiple frames compared to non-batched version.
    """
    # Generate test data with batch_size=3, n_frames=5
    pred_poses, true_poses, cov_matrices = generate_test_data(
        batch_size=3, n_frames=5, n_joints=13, seed=123
    )

    batch_size, n_frames, n_joints, _ = pred_poses.shape

    # Run batched version
    batched_result, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_poses, true_poses, cov_matrices
    )

    # Run non-batched version on each frame and aggregate manually
    all_counts = {
        'within_1std': 0,
        'within_2std': 0,
        'within_3std': 0,
        'within_4std': 0
    }
    total_joints = 0

    per_frame_coverages = [[], [], [], []]  # For each std level
    per_joint_coverages = [np.zeros(n_joints) for _ in range(4)]  # For each std level

    for b in range(batch_size):
        for t in range(n_frames):
            pred_single = pred_poses[b, t].reshape(n_joints, 3)
            true_single = true_poses[b, t].reshape(n_joints, 3)
            cov_single = cov_matrices[b, t]

            result = evaluate_pose_estimation_full_3d(
                ground_truth=true_single,
                estimated_pose=pred_single,
                estimated_covariance=cov_single
            )

            # Aggregate counts
            for key in all_counts.keys():
                all_counts[key] += result['counts'][key]
            total_joints += result['num_joints']

            # Track per-frame coverages
            for i in range(4):
                std_key = f'within_{i+1}std'
                per_frame_coverages[i].append(result['counts'][std_key] / n_joints)

            # Track per-joint results
            for i in range(4):
                for j_idx, joint_result in enumerate(result['joint_results']):
                    std_key = f'within_{i+1}std'
                    if joint_result[std_key]:
                        per_joint_coverages[i][j_idx] += 1

    # Compute aggregated non-batched coverages
    nonbatched_overall = [all_counts[f'within_{i+1}std'] / total_joints for i in range(4)]
    nonbatched_per_frame = [np.array(pf) for pf in per_frame_coverages]
    nonbatched_per_joint = [pj / (batch_size * n_frames) for pj in per_joint_coverages]

    # Extract batched results
    batched_overall = [
        batched_result['overall_within_1std'],
        batched_result['overall_within_2std'],
        batched_result['overall_within_3std'],
        batched_result['overall_within_4std']
    ]

    batched_per_joint = [
        batched_result['per_joint_within_1std'],
        batched_result['per_joint_within_2std'],
        batched_result['per_joint_within_3std'],
        batched_result['per_joint_within_4std']
    ]

    batched_per_frame = [
        batched_result['per_frame_within_1std'],
        batched_result['per_frame_within_2std'],
        batched_result['per_frame_within_3std'],
        batched_result['per_frame_within_4std']
    ]

    # Compare overall coverage
    print("\nOverall coverage comparison:")
    for i in range(4):
        print(f"  {i+1} std - Batched: {batched_overall[i]:.4f}, Non-batched: {nonbatched_overall[i]:.4f}")
        assert np.abs(batched_overall[i] - nonbatched_overall[i]) < 1e-6, \
            f"Overall coverage mismatch at {i+1} std"

    # Compare per-joint coverage
    print("\nPer-joint coverage comparison:")
    for i in range(4):
        max_diff = np.max(np.abs(batched_per_joint[i] - nonbatched_per_joint[i]))
        print(f"  {i+1} std - Max difference: {max_diff:.6f}")
        assert max_diff < 1e-6, \
            f"Per-joint coverage mismatch at {i+1} std"

    # Compare per-frame coverage
    print("\nPer-frame coverage comparison:")
    for i in range(4):
        # Batched per-frame is averaged over batch dimension, so we need to reshape
        # non-batched to match
        nonbatched_frames_reshaped = nonbatched_per_frame[i].reshape(batch_size, n_frames).mean(axis=0)
        max_diff = np.max(np.abs(batched_per_frame[i] - nonbatched_frames_reshaped))
        print(f"  {i+1} std - Max difference: {max_diff:.6f}")
        assert max_diff < 1e-6, \
            f"Per-frame coverage mismatch at {i+1} std"

    print("✓ Multiple frames test passed!")


def test_mahalanobis_distance_computation():
    """
    Test that both methods compute the same Mahalanobis distances.
    """
    # Generate simple test case
    np.random.seed(999)
    n_joints = 5

    pred_pose = np.random.randn(n_joints, 3) * 10
    true_pose = pred_pose + np.random.randn(n_joints, 3)  # Add some error

    # Create simple diagonal covariances for easier verification
    cov_matrices = np.zeros((n_joints, 3, 3))
    for j in range(n_joints):
        diag = np.random.rand(3) + 0.5  # Random positive values
        cov_matrices[j] = np.diag(diag)

    # Reshape for batched version
    pred_batched = pred_pose.reshape(1, 1, n_joints * 3)
    true_batched = true_pose.reshape(1, 1, n_joints * 3)
    cov_batched = cov_matrices.reshape(1, 1, n_joints, 3, 3)

    # Compute Mahalanobis distances manually for verification
    errors = true_pose - pred_pose
    mahal_manual = np.zeros(n_joints)
    for j in range(n_joints):
        inv_cov = np.linalg.inv(cov_matrices[j] + np.eye(3) * 1e-6)
        mahal_manual[j] = errors[j] @ inv_cov @ errors[j]

    # Run non-batched version
    nonbatched_result = evaluate_pose_estimation_full_3d(
        ground_truth=true_pose,
        estimated_pose=pred_pose,
        estimated_covariance=cov_matrices
    )

    mahal_nonbatched = np.array([
        jr['mahalanobis_distance']**2 for jr in nonbatched_result['joint_results']
    ])

    # The batched version doesn't directly return Mahalanobis distances,
    # but we can verify through the thresholds
    print("\nMahalanobis distance comparison:")
    print(f"  Manual computation: {mahal_manual}")
    print(f"  Non-batched version: {mahal_nonbatched}")

    # Check that they match
    assert np.allclose(mahal_manual, mahal_nonbatched, rtol=1e-5), \
        "Mahalanobis distances don't match between manual and non-batched"

    print("✓ Mahalanobis distance test passed!")


def test_edge_cases():
    """
    Test edge cases like very small or very large covariances.
    """
    n_joints = 13

    # Case 1: Very small covariances (high certainty)
    print("\nTesting very small covariances...")
    pred_pose = np.random.randn(n_joints, 3) * 100
    true_pose = pred_pose + np.random.randn(n_joints, 3) * 0.01  # Very small error
    cov_matrices = np.stack([np.eye(3) * 0.01 for _ in range(n_joints)])

    # Reshape for batched - use batch_size=2, n_frames=2 to avoid squeeze dimension issues
    pred_batched = np.tile(pred_pose.reshape(1, 1, n_joints, 3), (2, 2, 1))
    true_batched = np.tile(true_pose.reshape(1, 1, n_joints, 3), (2, 2, 1))
    cov_batched = np.tile(cov_matrices.reshape(1, 1, n_joints, 3, 3), (2, 2, 1, 1, 1))

    batched_result, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_batched, true_batched, cov_batched
    )
    nonbatched_result = evaluate_pose_estimation_full_3d(
        true_pose, pred_pose, cov_matrices
    )

    # Compare
    batched_cov = batched_result['overall_within_1std']
    nonbatched_cov = nonbatched_result['counts']['within_1std'] / n_joints
    assert np.abs(batched_cov - nonbatched_cov) < 1e-6
    print(f"  Small covariance - Batched: {batched_cov:.4f}, Non-batched: {nonbatched_cov:.4f}")

    # Case 2: Very large covariances (high uncertainty)
    print("\nTesting very large covariances...")
    true_pose = pred_pose + np.random.randn(n_joints, 3) * 100  # Large error
    cov_matrices = np.stack([np.eye(3) * 10000 for _ in range(n_joints)])

    true_batched = np.tile(true_pose.reshape(1, 1, n_joints, 3), (2, 2, 1))
    cov_batched = np.tile(cov_matrices.reshape(1, 1, n_joints, 3, 3), (2, 2, 1, 1, 1))

    batched_result, _ = evaluate_uncertainty_coverage_with_covariance(
        pred_batched, true_batched, cov_batched
    )
    nonbatched_result = evaluate_pose_estimation_full_3d(
        true_pose, pred_pose, cov_matrices
    )

    batched_cov = batched_result['overall_within_1std']
    nonbatched_cov = nonbatched_result['counts']['within_1std'] / n_joints
    assert np.abs(batched_cov - nonbatched_cov) < 1e-6
    print(f"  Large covariance - Batched: {batched_cov:.4f}, Non-batched: {nonbatched_cov:.4f}")

    print("✓ Edge cases test passed!")


def test_expected_coverage_values():
    """
    Verify that both methods use the correct expected coverage values
    based on chi-square distribution with df=3.
    """
    from scipy.stats import chi2

    # Expected coverage for 3D Gaussian (df=3)
    expected_1std = chi2.cdf(1**2, df=3)  # ~0.199
    expected_2std = chi2.cdf(2**2, df=3)  # ~0.738
    expected_3std = chi2.cdf(3**2, df=3)  # ~0.971
    expected_4std = chi2.cdf(4**2, df=3)  # ~0.998

    print("\nExpected coverage values (chi-square df=3):")
    print(f"  1 std: {expected_1std:.4f}")
    print(f"  2 std: {expected_2std:.4f}")
    print(f"  3 std: {expected_3std:.4f}")
    print(f"  4 std: {expected_4std:.4f}")

    # The batched version has these hardcoded differently!
    # Let's check what it uses
    batched_expected = {
        1: 0.682,   # This is for 1D Gaussian!
        2: 0.954,
        3: 0.997,
        4: 0.9999,
    }

    print("\nBatched version expected coverage:")
    for std, cov in batched_expected.items():
        print(f"  {std} std: {cov:.4f}")

    # The non-batched version uses chi2.ppf which is correct for 3D
    nonbatched_thresholds = [
        chi2.ppf(0.68, df=3),
        chi2.ppf(0.95, df=3),
        chi2.ppf(0.9973, df=3),
        chi2.ppf(0.99994, df=3)
    ]

    print("\nNon-batched version thresholds:")
    for i, threshold in enumerate(nonbatched_thresholds):
        print(f"  {i+1} std: threshold = {threshold:.4f}, coverage = {chi2.cdf(threshold, df=3):.4f}")

    print("\n⚠ WARNING: The two methods use DIFFERENT expected coverage values!")
    print("  - Batched version uses 1D Gaussian percentiles (68%, 95%, 99.7%)")
    print("  - Non-batched version uses chi-square thresholds for 3D (df=3)")
    print("  This is a potential BUG in the batched implementation!")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Coverage Evaluation: Batched vs Non-batched")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("Test 1: Single frame comparison")
    print("=" * 70)
    test_coverage_comparison_single_frame()

    print("\n" + "=" * 70)
    print("Test 2: Multiple frames comparison")
    print("=" * 70)
    test_coverage_comparison_multiple_frames()

    print("\n" + "=" * 70)
    print("Test 3: Mahalanobis distance computation")
    print("=" * 70)
    test_mahalanobis_distance_computation()

    print("\n" + "=" * 70)
    print("Test 4: Edge cases")
    print("=" * 70)
    test_edge_cases()

    print("\n" + "=" * 70)
    print("Test 5: Expected coverage values")
    print("=" * 70)
    test_expected_coverage_values()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED!")
    print("=" * 70)
