"""
Tests for the batched buffer functions introduced for queue-based pose pipeline.

Verifies that:
  1. fill_pose_buffer_batched with B=1 produces the same result as fill_pose_buffer.
  2. process_pose_output_batched with B=1 produces the same result as process_pose_output.
  3. For invalid frames, fill_pose_buffer_batched uses motion_prediction_buffer[b] (not [0]).
  4. Intermediate buffer states have the correct sliding-window structure.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import torch

from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    fill_pose_buffer,
    fill_pose_buffer_batched,
    process_pose_output,
    process_pose_output_batched,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

T = 5   # INPUT_HORIZON_LENGTH (small for tests)
J = 3   # N_JOINTS (small for tests)
P = 4   # PREDICTION_HORIZON_LENGTH (small for tests)

RNG = np.random.default_rng(42)


def _rand(shape):
    return jnp.array(RNG.random(shape).astype(np.float32))


def make_buffers():
    pts_buf = _rand([T, J, 3])
    cov_buf = _rand([T, J, 3, 3])
    val_buf = jnp.ones([T], dtype=jnp.float32)
    mot_buf = _rand([P, J, 3])
    mot_cov = _rand([P, J, 3, 3])
    return pts_buf, cov_buf, val_buf, mot_buf, mot_cov


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFillPoseBufferBatchedB1:
    """B=1 batched call must equal the single-frame function."""

    def test_valid_frame(self):
        pts_buf, cov_buf, val_buf, mot_buf, mot_cov = make_buffers()
        new_pts = _rand([J, 3])
        new_cov = _rand([J, 3, 3])

        # Single-frame reference
        ref_pts, ref_cov, ref_val, ref_good = fill_pose_buffer(
            points_3d_buffer=pts_buf,
            covariance_buffer=cov_buf,
            pose_valid_buffer=val_buf,
            points_3d=new_pts,
            covariance=new_cov,
            is_valid=True,
            motion_prediction_buffer=mot_buf,
            motion_uncertainty_buffer=mot_cov,
        )

        # Batched B=1
        (
            fin_pts, fin_cov, fin_val,
            inter_pts, inter_cov, inter_val,
            good_batch,
        ) = fill_pose_buffer_batched(
            points_3d_buffer=pts_buf,
            covariance_buffer=cov_buf,
            pose_valid_buffer=val_buf,
            points_3d_batch=new_pts[None],   # [1, J, 3]
            covariance_batch=new_cov[None],  # [1, J, 3, 3]
            is_valid_batch=jnp.array([True]),
            motion_prediction_buffer=mot_buf,
            motion_uncertainty_buffer=mot_cov,
        )

        np.testing.assert_allclose(np.array(fin_pts), np.array(ref_pts), atol=1e-6)
        np.testing.assert_allclose(np.array(fin_cov), np.array(ref_cov), atol=1e-6)
        np.testing.assert_allclose(np.array(fin_val), np.array(ref_val), atol=1e-6)
        assert bool(good_batch[0]) == ref_good

        # Intermediate state (B=1) must equal the final state
        np.testing.assert_allclose(np.array(inter_pts[0]), np.array(fin_pts), atol=1e-6)

    def test_invalid_frame_uses_motion_prediction_buffer_0(self):
        pts_buf, cov_buf, val_buf, mot_buf, mot_cov = make_buffers()
        new_pts = _rand([J, 3])
        new_cov = _rand([J, 3, 3])

        ref_pts, ref_cov, ref_val, _ = fill_pose_buffer(
            points_3d_buffer=pts_buf,
            covariance_buffer=cov_buf,
            pose_valid_buffer=val_buf,
            points_3d=new_pts,
            covariance=new_cov,
            is_valid=False,
            motion_prediction_buffer=mot_buf,
            motion_uncertainty_buffer=mot_cov,
        )

        (fin_pts, fin_cov, fin_val, _, _, _, _) = fill_pose_buffer_batched(
            points_3d_buffer=pts_buf,
            covariance_buffer=cov_buf,
            pose_valid_buffer=val_buf,
            points_3d_batch=new_pts[None],
            covariance_batch=new_cov[None],
            is_valid_batch=jnp.array([False]),
            motion_prediction_buffer=mot_buf,
            motion_uncertainty_buffer=mot_cov,
        )

        np.testing.assert_allclose(np.array(fin_pts), np.array(ref_pts), atol=1e-6)
        np.testing.assert_allclose(np.array(fin_cov), np.array(ref_cov), atol=1e-6)


class TestFillPoseBufferBatchedFallback:
    """For B>1 invalid frames, each frame b must use motion_prediction_buffer[b]."""

    def test_fallback_uses_successive_motion_predictions(self):
        pts_buf, cov_buf, val_buf, mot_buf, mot_cov = make_buffers()
        B = 3
        new_pts = _rand([B, J, 3])
        new_cov = _rand([B, J, 3, 3])
        is_valid = jnp.array([False, False, False])

        (fin_pts, fin_cov, fin_val, inter_pts, inter_cov, inter_val, good_batch) = \
            fill_pose_buffer_batched(
                points_3d_buffer=pts_buf,
                covariance_buffer=cov_buf,
                pose_valid_buffer=val_buf,
                points_3d_batch=new_pts,
                covariance_batch=new_cov,
                is_valid_batch=is_valid,
                motion_prediction_buffer=mot_buf,
                motion_uncertainty_buffer=mot_cov,
            )

        # The last T entries of the final buffer should contain mot_buf[0], mot_buf[1], mot_buf[2]
        # at positions [-3], [-2], [-1].
        for b in range(B):
            np.testing.assert_allclose(
                np.array(fin_pts[T - B + b]),
                np.array(mot_buf[b]),
                atol=1e-6,
                err_msg=f'Frame b={b}: expected mot_buf[{b}] in final buffer',
            )

    def test_intermediate_state_structure(self):
        """intermediate_pts[b] should equal the buffer state after inserting frames 0..b."""
        pts_buf, cov_buf, val_buf, mot_buf, mot_cov = make_buffers()
        B = 2
        new_pts = _rand([B, J, 3])
        new_cov = _rand([B, J, 3, 3])
        is_valid = jnp.array([True, True])

        (_, _, _, inter_pts, _, _, _) = fill_pose_buffer_batched(
            points_3d_buffer=pts_buf,
            covariance_buffer=cov_buf,
            pose_valid_buffer=val_buf,
            points_3d_batch=new_pts,
            covariance_batch=new_cov,
            is_valid_batch=is_valid,
            motion_prediction_buffer=mot_buf,
            motion_uncertainty_buffer=mot_cov,
        )

        # inter_pts[0] = [pts_buf[1:], new_pts[0]]
        expected_0 = jnp.concatenate([pts_buf[1:], new_pts[0:1]], axis=0)
        np.testing.assert_allclose(np.array(inter_pts[0]), np.array(expected_0), atol=1e-6)

        # inter_pts[1] = [pts_buf[2:], new_pts[0], new_pts[1]]
        expected_1 = jnp.concatenate([pts_buf[2:], new_pts[0:1], new_pts[1:2]], axis=0)
        np.testing.assert_allclose(np.array(inter_pts[1]), np.array(expected_1), atol=1e-6)


class TestProcessPoseOutputBatchedB1:
    """process_pose_output_batched with B=1 must equal process_pose_output."""

    def _make_torch_outputs(self, B=1):
        pts = torch.tensor(RNG.random([B, J, 3]).astype(np.float32))
        cov = torch.tensor(RNG.random([B, J, 3, 3]).astype(np.float32))
        is_ood = torch.zeros(B, dtype=torch.bool)
        detected = torch.ones(B, dtype=torch.bool)
        return pts, cov, is_ood, detected

    def test_valid_single_frame(self):
        pts_buf, cov_buf, val_buf, mot_buf, mot_cov = make_buffers()
        pts, cov, is_ood, detected = self._make_torch_outputs(B=1)

        # Reference: old function (B=1, is_valid=True)
        ref_pts_buf, ref_cov_buf, ref_val_buf, ref_good = process_pose_output(
            points_3d=pts,
            C_3d_all=cov,
            is_valid=True,
            points_3d_buffer=pts_buf,
            covariance_buffer=cov_buf,
            pose_valid_buffer=val_buf,
            motion_prediction_buffer=mot_buf,
            motion_uncertainty_buffer=mot_cov,
        )

        # Batched version
        (fin_pts_buf, fin_cov_buf, fin_val_buf, _, _, _, good_batch) = \
            process_pose_output_batched(
                points_3d=pts,
                C_3d_all=cov,
                pose_is_ood=is_ood,
                human_detected=detected,
                points_3d_buffer=pts_buf,
                covariance_buffer=cov_buf,
                pose_valid_buffer=val_buf,
                motion_prediction_buffer=mot_buf,
                motion_uncertainty_buffer=mot_cov,
            )

        np.testing.assert_allclose(np.array(fin_pts_buf), np.array(ref_pts_buf), atol=1e-6)
        np.testing.assert_allclose(np.array(fin_cov_buf), np.array(ref_cov_buf), atol=1e-6)
        np.testing.assert_allclose(np.array(fin_val_buf), np.array(ref_val_buf), atol=1e-6)
        assert bool(good_batch[0]) == ref_good

    def test_ood_frame_is_invalid(self):
        pts_buf, cov_buf, val_buf, mot_buf, mot_cov = make_buffers()
        pts, cov, _, detected = self._make_torch_outputs(B=1)
        is_ood = torch.ones(1, dtype=torch.bool)  # OOD → invalid

        ref_pts_buf, _, _, _ = process_pose_output(
            points_3d=pts, C_3d_all=cov, is_valid=False,
            points_3d_buffer=pts_buf, covariance_buffer=cov_buf,
            pose_valid_buffer=val_buf,
            motion_prediction_buffer=mot_buf, motion_uncertainty_buffer=mot_cov,
        )

        (fin_pts_buf, *_) = process_pose_output_batched(
            points_3d=pts, C_3d_all=cov,
            pose_is_ood=is_ood, human_detected=detected,
            points_3d_buffer=pts_buf, covariance_buffer=cov_buf,
            pose_valid_buffer=val_buf,
            motion_prediction_buffer=mot_buf, motion_uncertainty_buffer=mot_cov,
        )

        np.testing.assert_allclose(np.array(fin_pts_buf), np.array(ref_pts_buf), atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
