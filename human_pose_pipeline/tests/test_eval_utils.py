"""Unit tests for human_pose_pipeline/utils/eval_utils.py."""

import unittest
import numpy as np
import jax.numpy as jnp
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from human_pose_pipeline.utils.eval_utils import (
    evaluate_pose_prediction_scores_np,
    evaluate_pose_prediction_scores_jax,
    evaluate_uncertainty_coverage_jax,
    evaluate_uncertainty_coverage_with_covariance,
)

# Shared test dimensions
B, T, J = 3, 4, 2


def _make_uniform_error_data(error_val=1.0):
    """All valid predictions with a constant per-joint error of `error_val`."""
    targets = np.zeros((B, T, J, 3), dtype=np.float64)
    predictions = np.zeros((B, T, J, 3), dtype=np.float64)
    predictions[..., 0] = error_val  # shift x → norm([error_val, 0, 0]) = error_val
    return predictions, targets


def _make_masked_data(error_val=1.0):
    """Like _make_uniform_error_data but batch-0 / frame-3 is all-zero (invalid)."""
    predictions, targets = _make_uniform_error_data(error_val)
    predictions[0, 3] = 0.0  # all-zero → invalid frame
    return predictions, targets


class TestEvaluatePosePredictionScoresNp(unittest.TestCase):

    def test_output_shapes(self):
        predictions, targets = _make_uniform_error_data()
        mpjpe, std, per_t, per_t_std, per_j, per_j_std = evaluate_pose_prediction_scores_np(
            predictions, targets
        )
        self.assertIsInstance(mpjpe, float)
        self.assertIsInstance(std, float)
        self.assertEqual(per_t.shape, (T,))
        self.assertEqual(per_t_std.shape, (T,))
        self.assertEqual(per_j.shape, (J,))
        self.assertEqual(per_j_std.shape, (J,))

    def test_uniform_error_all_valid(self):
        """When all predictions have the same error, MPJPE equals that error."""
        predictions, targets = _make_uniform_error_data(error_val=2.5)
        mpjpe, std, per_t, _, per_j, _ = evaluate_pose_prediction_scores_np(
            predictions, targets
        )
        self.assertAlmostEqual(mpjpe, 2.5, places=6)
        self.assertAlmostEqual(std, 0.0, places=6)
        np.testing.assert_allclose(per_t, 2.5, atol=1e-6)
        np.testing.assert_allclose(per_j, 2.5, atol=1e-6)

    def test_masking_excludes_zero_frames(self):
        """Invalid (all-zero) frames must be excluded from all statistics.

        Setup: error = 1.0 for all valid frames.
          Batch 0, frame 3 is invalid (pred = 0 = target → error = 0, but masked).
        Without masking overall MPJPE ≈ 22/24 ≈ 0.917; with masking it must be 1.0.
        """
        predictions, targets = _make_masked_data(error_val=1.0)
        mpjpe, _, per_t, _, per_j, _ = evaluate_pose_prediction_scores_np(
            predictions, targets
        )
        self.assertAlmostEqual(mpjpe, 1.0, places=6)
        np.testing.assert_allclose(per_t, 1.0, atol=1e-6)
        np.testing.assert_allclose(per_j, 1.0, atol=1e-6)

    def test_per_time_masking_is_local(self):
        """per_time_errors[t] must only average over batches that are valid at t.

        Batch 0 / frame 3 is invalid; valid batches at frame 3 still have error 1.0,
        so per_time_errors[3] = 1.0.  Without masking it would be < 1.0.
        """
        predictions, targets = _make_masked_data(error_val=1.0)
        _, _, per_t, _, _, _ = evaluate_pose_prediction_scores_np(predictions, targets)
        # All time steps including the partially-masked one should give 1.0
        self.assertAlmostEqual(float(per_t[3]), 1.0, places=6)

    def test_zero_error_when_predictions_equal_targets(self):
        targets = np.random.randn(B, T, J, 3)
        predictions = targets.copy()
        mpjpe, std, per_t, _, per_j, _ = evaluate_pose_prediction_scores_np(
            predictions, targets
        )
        self.assertAlmostEqual(mpjpe, 0.0, places=6)
        self.assertAlmostEqual(std, 0.0, places=6)


class TestEvaluatePosePredictionScoresJax(unittest.TestCase):

    def test_output_shapes(self):
        predictions, targets = _make_uniform_error_data()
        mpjpe, std, per_t, per_t_std, per_j, per_j_std = evaluate_pose_prediction_scores_jax(
            jnp.array(predictions), jnp.array(targets)
        )
        self.assertEqual(per_t.shape, (T,))
        self.assertEqual(per_t_std.shape, (T,))
        self.assertEqual(per_j.shape, (J,))
        self.assertEqual(per_j_std.shape, (J,))

    def test_uniform_error_all_valid(self):
        predictions, targets = _make_uniform_error_data(error_val=2.5)
        mpjpe, std, per_t, _, per_j, _ = evaluate_pose_prediction_scores_jax(
            jnp.array(predictions), jnp.array(targets)
        )
        self.assertAlmostEqual(float(mpjpe), 2.5, places=5)
        self.assertAlmostEqual(float(std), 0.0, places=5)
        np.testing.assert_allclose(np.array(per_t), 2.5, atol=1e-5)
        np.testing.assert_allclose(np.array(per_j), 2.5, atol=1e-5)

    def test_masking_excludes_zero_frames(self):
        predictions, targets = _make_masked_data(error_val=1.0)
        mpjpe, _, per_t, _, per_j, _ = evaluate_pose_prediction_scores_jax(
            jnp.array(predictions), jnp.array(targets)
        )
        self.assertAlmostEqual(float(mpjpe), 1.0, places=5)
        np.testing.assert_allclose(np.array(per_t), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.array(per_j), 1.0, atol=1e-5)

    def test_per_time_masking_is_local(self):
        predictions, targets = _make_masked_data(error_val=1.0)
        _, _, per_t, _, _, _ = evaluate_pose_prediction_scores_jax(
            jnp.array(predictions), jnp.array(targets)
        )
        self.assertAlmostEqual(float(per_t[3]), 1.0, places=5)

    def test_agrees_with_numpy_all_valid(self):
        """NumPy and JAX implementations must agree on fully valid data."""
        rng = np.random.default_rng(0)
        predictions = rng.standard_normal((B, T, J, 3))
        targets = rng.standard_normal((B, T, J, 3))

        np_out = evaluate_pose_prediction_scores_np(predictions, targets)
        jax_out = evaluate_pose_prediction_scores_jax(
            jnp.array(predictions), jnp.array(targets)
        )

        self.assertAlmostEqual(float(jax_out[0]), np_out[0], places=4)  # mpjpe
        self.assertAlmostEqual(float(jax_out[1]), np_out[1], places=4)  # std
        np.testing.assert_allclose(np.array(jax_out[2]), np_out[2], atol=1e-4)  # per_t
        np.testing.assert_allclose(np.array(jax_out[4]), np_out[4], atol=1e-4)  # per_j

    def test_agrees_with_numpy_with_masked_frames(self):
        """NumPy and JAX must agree when some frames are masked."""
        predictions, targets = _make_masked_data(error_val=3.0)
        # Add extra invalid frames to stress-test
        predictions[1, 0] = 0.0
        predictions[2, 2] = 0.0

        np_out = evaluate_pose_prediction_scores_np(predictions, targets)
        jax_out = evaluate_pose_prediction_scores_jax(
            jnp.array(predictions), jnp.array(targets)
        )

        self.assertAlmostEqual(float(jax_out[0]), np_out[0], places=4)
        np.testing.assert_allclose(np.array(jax_out[2]), np_out[2], atol=1e-4)
        np.testing.assert_allclose(np.array(jax_out[4]), np_out[4], atol=1e-4)


class TestEvaluateUncoverageCoverageJax(unittest.TestCase):

    def _identity_L(self, B, T, J):
        """Identity Cholesky factor → unit isotropic covariance."""
        L = np.zeros((B, T, J, 3, 3), dtype=np.float64)
        for i in range(3):
            L[..., i, i] = 1.0
        return jnp.array(L)

    def test_zero_diff_all_inside(self):
        """When predictions equal targets, all Mahalanobis distances are 0 → all inside."""
        poses = np.ones((B, T, J, 3), dtype=np.float64)
        L = self._identity_L(B, T, J)
        results = evaluate_uncertainty_coverage_jax(
            jnp.array(poses), jnp.array(poses), L, std_multipliers=[1, 2]
        )
        # Empirical coverage = 1.0, so error = expected - 1.0 <= 0
        for err in results:
            self.assertLessEqual(float(err), 0.0)

    def test_output_length_matches_multipliers(self):
        poses = np.ones((B, T, J, 3), dtype=np.float64)
        L = self._identity_L(B, T, J)
        multipliers = [1, 2, 3]
        results = evaluate_uncertainty_coverage_jax(
            jnp.array(poses), jnp.array(poses), L, std_multipliers=multipliers
        )
        self.assertEqual(len(results), len(multipliers))

    def test_masking_excludes_zero_pred_frames(self):
        """Frames where pred_poses == 0 must be ignored even if true_poses != 0.

        Without masking, those frames would have large Mahalanobis distances,
        lowering empirical coverage.  With masking the result must equal the
        all-valid baseline (predictions == targets → empirical = 1).
        """
        true_poses = np.ones((B, T, J, 3), dtype=np.float64)
        pred_poses = np.ones((B, T, J, 3), dtype=np.float64)
        # Invalidate batch 0, frame 3: pred = 0 but true = 1 → huge Mahal if unmasked
        pred_poses[0, 3] = 0.0

        L = self._identity_L(B, T, J)
        results_masked = evaluate_uncertainty_coverage_jax(
            jnp.array(pred_poses), jnp.array(true_poses), L, std_multipliers=[1]
        )
        results_clean = evaluate_uncertainty_coverage_jax(
            jnp.array(true_poses), jnp.array(true_poses), L, std_multipliers=[1]
        )
        # Both should be the same because the invalid frame is excluded
        self.assertAlmostEqual(float(results_masked[0]), float(results_clean[0]), places=5)


class TestEvaluateUncoverageCoverageWithCovariance(unittest.TestCase):

    def _identity_cov(self, B, T, J):
        cov = np.zeros((B, T, J, 3, 3), dtype=np.float64)
        for i in range(3):
            cov[..., i, i] = 1.0
        return cov

    def test_output_keys_and_shapes(self):
        poses = np.ones((B, T, J, 3), dtype=np.float64)
        cov = self._identity_cov(B, T, J)
        stats, within_stds = evaluate_uncertainty_coverage_with_covariance(poses, poses, cov)

        for i in range(1, 5):
            self.assertIn(f'overall_within_{i}std', stats)
            self.assertEqual(stats[f'per_joint_within_{i}std'].shape, (J,))
            self.assertEqual(stats[f'per_frame_within_{i}std'].shape, (T,))
        self.assertEqual(len(within_stds), 4)

    def test_zero_diff_all_inside(self):
        """Identical predictions and targets → empirical coverage = 1 for all thresholds."""
        poses = np.ones((B, T, J, 3), dtype=np.float64)
        cov = self._identity_cov(B, T, J)
        stats, _ = evaluate_uncertainty_coverage_with_covariance(poses, poses, cov)
        for i in range(1, 5):
            self.assertAlmostEqual(stats[f'overall_within_{i}std'], 1.0, places=5)

    def test_masking_excludes_zero_pred_frames(self):
        """Frames where pred_poses == 0 but true_poses != 0 must be ignored.

        Without masking those frames have large Mahalanobis distances and
        drag empirical coverage below 1.  With masking the result must equal
        the all-valid baseline.
        """
        true_poses = np.ones((B, T, J, 3), dtype=np.float64)
        pred_poses = np.ones((B, T, J, 3), dtype=np.float64)
        pred_poses[0, 3] = 0.0  # invalid: pred=0, true=1

        cov = self._identity_cov(B, T, J)
        stats_masked, _ = evaluate_uncertainty_coverage_with_covariance(
            pred_poses, true_poses, cov
        )
        stats_clean, _ = evaluate_uncertainty_coverage_with_covariance(
            true_poses, true_poses, cov
        )
        for i in range(1, 5):
            self.assertAlmostEqual(
                stats_masked[f'overall_within_{i}std'],
                stats_clean[f'overall_within_{i}std'],
                places=5,
            )


if __name__ == '__main__':
    unittest.main()
