"""
Equivalence test: sequential (B=1) vs batched (B=2) pose pipeline.

Mirrors the structure of eval_full_pipeline_rgbd_yolo.py but skips YOLO/depth
and injects synthetic 3D pose + covariance data directly.  Both pipeline
variants process the same 2000 frames; results must satisfy:
  - MPJPE difference  < 0.5 mm
  - 99%-likelihood coverage difference < 0.1 percentage points

Skipped automatically if the motion model file is not found.
"""

import os
import pytest
import numpy as np
import jax.numpy as jnp

from human_pose_pipeline.pose_estimation.inference_helper import initialize_jax_models
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    fill_pose_buffer,
    fill_pose_buffer_batched,
)
from human_pose_pipeline.motion_prediction.inference_helper import (
    run_motion_prediction,
    run_motion_prediction_batched,
)
from human_pose_pipeline.pose_estimation.inference_helper_batched import (
    update_motion_prediction_buffer,
)
from human_pose_pipeline.utils.eval_utils import (
    evaluate_pose_prediction_scores_np,
    simple_coverage_stats_sara,
    convert_covariance_matrices_to_set,
)
from human_pose_pipeline.motion_prediction.rgbd_yolo_settings import (
    INPUT_HORIZON_LENGTH,
    PREDICTION_HORIZON_LENGTH,
    N_JOINTS,
    OOD_THRESHOLD as MOTION_OOD_THRESHOLD,
    N_CORRECT_POSES_REQUIRED,
    COV_CALIBRATION_CT,
    COV_CALIBRATION_IT,
    COV_CALIBRATION_FACTORS,
    SET_LIKELIHOOD,
)

# ── Constants ──────────────────────────────────────────────────────────────────

N_FRAMES    = 2000
BATCH_SIZE  = 2
MPJPE_TOL   = 0.5    # mm
COV_TOL     = 0.1    # percentage points

_WORKSPACE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..')
)
_MOTION_MODEL_PATH = os.path.join(
    _WORKSPACE_ROOT,
    'human_pose_pipeline/models/motion_prediction/final_model/dct_pose_transformer.pickle',
)


# ── Synthetic data generation ──────────────────────────────────────────────────

def _generate_synthetic_sequence(n_frames, n_joints, rng):
    """Generate a smooth random-walk pose sequence in mm with per-frame covariances."""
    # Starting skeleton: joints spread across roughly 1m in height, in mm
    base = rng.standard_normal((n_joints, 3)).astype(np.float32) * 200.0
    base[:, 2] += 1000.0  # ~1 m above origin

    # Slow random walk (step_std ≈ 5 mm per frame)
    steps = rng.standard_normal((n_frames, n_joints, 3)).astype(np.float32) * 5.0
    poses = base[None] + np.cumsum(steps, axis=0)  # [N, J, 3]

    # Small isotropic covariances (~50 mm² diagonal)
    covs = np.zeros((n_frames, n_joints, 3, 3), dtype=np.float32)
    for j in range(n_joints):
        for d in range(3):
            covs[:, j, d, d] = 50.0 + rng.random(n_frames).astype(np.float32) * 20.0

    # 90% of frames are valid
    is_valid = rng.random(n_frames) < 0.90

    return poses, covs, is_valid


# ── Sequential pipeline (B=1) ─────────────────────────────────────────────────

def _run_sequential(poses, covs, is_valid, motion_jit_fn, params, batch_stats):
    """Process all frames one by one (mirrors eval_full_pipeline_rgbd_yolo.py)."""
    pts_buf  = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3])
    cov_buf  = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3, 3])
    val_buf  = jnp.zeros([INPUT_HORIZON_LENGTH])
    mot_buf  = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3])
    mot_cov  = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3, 3])

    motions_pred  = []
    motions_cov   = []
    motions_gt    = []
    frame_counter = 0

    for i in range(len(poses)):
        pts_buf, cov_buf, val_buf, buf_good = fill_pose_buffer(
            points_3d_buffer=pts_buf,
            covariance_buffer=cov_buf,
            pose_valid_buffer=val_buf,
            points_3d=jnp.array(poses[i]),
            covariance=jnp.array(covs[i]),
            is_valid=bool(is_valid[i]),
            motion_prediction_buffer=mot_buf,
            motion_uncertainty_buffer=mot_cov,
        )

        if frame_counter >= INPUT_HORIZON_LENGTH - 1 and buf_good:
            mot_buf, mot_cov, _, _, _, _, motion_pred, motion_cov_cal, _ = \
                run_motion_prediction(
                    points_3d_buffer=pts_buf,
                    covariance_buffer=cov_buf,
                    pose_valid_buffer=val_buf,
                    motion_prediction_buffer=mot_buf,
                    motion_uncertainty_buffer=mot_cov,
                    motion_prediction_jit_fn=motion_jit_fn,
                    motion_prediction_params=params,
                    motion_prediction_batch_stats=batch_stats,
                    motion_ood_score_fn=None,
                    n_joints=N_JOINTS,
                    input_horizon_length=INPUT_HORIZON_LENGTH,
                    prediction_horizon_length=PREDICTION_HORIZON_LENGTH,
                    ood_threshold=MOTION_OOD_THRESHOLD,
                    calibration_ct=COV_CALIBRATION_CT,
                    calibration_it=COV_CALIBRATION_IT,
                    calibration_factors=COV_CALIBRATION_FACTORS,
                    n_correct_poses_required=N_CORRECT_POSES_REQUIRED,
                    set_likelihood=SET_LIKELIHOOD,
                )

            # Collect prediction and next-P-frames GT
            gt_end = i + 1 + PREDICTION_HORIZON_LENGTH
            if gt_end <= len(poses):
                motions_pred.append(np.array(motion_pred))
                motions_cov.append(np.array(motion_cov_cal))
                motions_gt.append(poses[i + 1 : gt_end])

        frame_counter += 1

    return (
        np.array(motions_pred),    # [M, P, J, 3]
        np.array(motions_cov),     # [M, P, J, 3, 3]
        np.array(motions_gt),      # [M, P, J, 3]
    )


# ── Batched pipeline (B=2) ────────────────────────────────────────────────────

def _run_batched(poses, covs, is_valid, motion_jit_fn, params, batch_stats):
    """Process all frames in batches of BATCH_SIZE."""
    pts_buf  = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3])
    cov_buf  = jnp.zeros([INPUT_HORIZON_LENGTH, N_JOINTS, 3, 3])
    val_buf  = jnp.zeros([INPUT_HORIZON_LENGTH])
    mot_buf  = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3])
    mot_cov  = jnp.zeros([PREDICTION_HORIZON_LENGTH, N_JOINTS, 3, 3])

    motions_pred  = []
    motions_cov   = []
    motions_gt    = []
    frame_counter = 0

    n = len(poses)
    for start in range(0, n, BATCH_SIZE):
        batch_idx = list(range(start, min(start + BATCH_SIZE, n)))
        B = len(batch_idx)

        pts_batch  = jnp.array(poses[batch_idx])      # [B, J, 3]
        cov_batch  = jnp.array(covs[batch_idx])       # [B, J, 3, 3]
        valid_batch = jnp.array(is_valid[batch_idx].astype(bool))  # [B]

        (
            pts_buf, cov_buf, val_buf,
            inter_pts, inter_cov, inter_val,
            buf_good_batch,
        ) = fill_pose_buffer_batched(
            points_3d_buffer=pts_buf,
            covariance_buffer=cov_buf,
            pose_valid_buffer=val_buf,
            points_3d_batch=pts_batch,
            covariance_batch=cov_batch,
            is_valid_batch=valid_batch,
            motion_prediction_buffer=mot_buf,
            motion_uncertainty_buffer=mot_cov,
        )

        mot_buf, mot_cov, _, _, _, valid_motions, motion_pred_batch, motion_cov_batch = \
            run_motion_prediction_batched(
                points_3d_buffers=inter_pts,
                covariance_buffers=inter_cov,
                pose_valid_buffers=inter_val,
                motion_prediction_buffer=mot_buf,
                motion_uncertainty_buffer=mot_cov,
                motion_prediction_jit_fn=motion_jit_fn,
                motion_prediction_params=params,
                motion_prediction_batch_stats=batch_stats,
                motion_ood_score_fn=None,
                n_joints=N_JOINTS,
                input_horizon_length=INPUT_HORIZON_LENGTH,
                prediction_horizon_length=PREDICTION_HORIZON_LENGTH,
                ood_threshold=MOTION_OOD_THRESHOLD,
                calibration_ct=COV_CALIBRATION_CT,
                calibration_it=COV_CALIBRATION_IT,
                calibration_factors=COV_CALIBRATION_FACTORS,
                n_correct_poses_required=N_CORRECT_POSES_REQUIRED,
                set_likelihood=SET_LIKELIHOOD,
                pose_buffer_good_batch=buf_good_batch,
                frame_counter=frame_counter,
            )

        for b, global_i in enumerate(batch_idx):
            # Collect whenever frame is ready and buffer is warm — same criterion
            # as the sequential pipeline (independent of valid_motions[b]).
            frame_ready = (frame_counter + b) >= INPUT_HORIZON_LENGTH - 1
            if frame_ready and bool(buf_good_batch[b]):
                gt_end = global_i + 1 + PREDICTION_HORIZON_LENGTH
                if gt_end <= n:
                    motions_pred.append(np.array(motion_pred_batch[b]))
                    motions_cov.append(np.array(motion_cov_batch[b]))
                    motions_gt.append(poses[global_i + 1 : gt_end])

        frame_counter += B

    return (
        np.array(motions_pred),    # [M, P, J, 3]
        np.array(motions_cov),     # [M, P, J, 3, 3]
        np.array(motions_gt),      # [M, P, J, 3]
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

def _mpjpe(pred, gt):
    """Mean per-joint position error in mm.  pred/gt: [M, P, J, 3]."""
    err = np.linalg.norm(pred - gt, axis=-1)   # [M, P, J]
    return float(err.mean())


def _coverage_99(pred, cov, gt):
    """Fraction of ground-truth joints within the 99%-likelihood ellipsoid."""
    # set_radius: [M, P, J]  (uses convert_covariance_matrices_to_set)
    radius = np.array(convert_covariance_matrices_to_set(cov, likelihood=SET_LIKELIHOOD))
    dist   = np.linalg.norm(gt - pred, axis=-1)   # [M, P, J]
    within = (dist <= radius).mean()
    return float(within) * 100.0   # percentage


# ── Pytest fixture & test ──────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def motion_model():
    if not os.path.exists(_MOTION_MODEL_PATH):
        pytest.skip(f'Motion model not found: {_MOTION_MODEL_PATH}')
    jit_fn, params, batch_stats = initialize_jax_models(_MOTION_MODEL_PATH)
    return jit_fn, params, batch_stats


@pytest.fixture(scope='module')
def synthetic_data():
    rng = np.random.default_rng(0)
    poses, covs, is_valid = _generate_synthetic_sequence(N_FRAMES, N_JOINTS, rng)
    return poses, covs, is_valid


def test_batched_vs_sequential_equivalence(motion_model, synthetic_data):
    """Sequential (B=1) and batched (B=2) pipelines must produce near-identical results."""
    jit_fn, params, batch_stats = motion_model
    poses, covs, is_valid = synthetic_data

    print(f'\nRunning sequential pipeline on {N_FRAMES} frames...')
    pred_seq, cov_seq, gt_seq = _run_sequential(
        poses, covs, is_valid, jit_fn, params, batch_stats
    )

    print(f'Running batched pipeline (B={BATCH_SIZE}) on {N_FRAMES} frames...')
    pred_bat, cov_bat, gt_bat = _run_batched(
        poses, covs, is_valid, jit_fn, params, batch_stats
    )

    assert len(pred_seq) > 0, 'Sequential pipeline produced no motion predictions'
    assert len(pred_bat) > 0, 'Batched pipeline produced no motion predictions'
    assert len(pred_seq) == len(pred_bat), (
        f'Prediction count mismatch: sequential={len(pred_seq)}, batched={len(pred_bat)}'
    )
    n = len(pred_seq)
    print(f'Comparing {n} motion predictions...')

    # ── MPJPE of each pipeline vs the common ground truth ─────────────────────
    mpjpe_seq = _mpjpe(pred_seq, gt_seq)
    mpjpe_bat = _mpjpe(pred_bat, gt_bat)
    mpjpe_diff = abs(mpjpe_seq - mpjpe_bat)

    print(f'MPJPE sequential: {mpjpe_seq:.4f} mm')
    print(f'MPJPE batched:    {mpjpe_bat:.4f} mm')
    print(f'MPJPE difference: {mpjpe_diff:.6f} mm  (tolerance: {MPJPE_TOL} mm)')

    assert mpjpe_diff < MPJPE_TOL, (
        f'MPJPE difference {mpjpe_diff:.4f} mm exceeds tolerance {MPJPE_TOL} mm'
    )

    # ── 99%-likelihood coverage ────────────────────────────────────────────────
    cov_seq_pct = _coverage_99(pred_seq, cov_seq, gt_seq)
    cov_bat_pct = _coverage_99(pred_bat, cov_bat, gt_bat)
    cov_diff    = abs(cov_seq_pct - cov_bat_pct)

    print(f'Coverage sequential: {cov_seq_pct:.4f}%')
    print(f'Coverage batched:    {cov_bat_pct:.4f}%')
    print(f'Coverage difference: {cov_diff:.6f}%  (tolerance: {COV_TOL}%)')

    assert cov_diff < COV_TOL, (
        f'Coverage difference {cov_diff:.4f}% exceeds tolerance {COV_TOL}%'
    )


if __name__ == '__main__':
    # Allow running directly as a script for quick iteration
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_model_path', default=_MOTION_MODEL_PATH)
    parser.add_argument('--n_frames', type=int, default=N_FRAMES)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.motion_model_path):
        raise FileNotFoundError(f'Motion model not found: {args.motion_model_path}')

    print(f'Loading motion model from {args.motion_model_path}...')
    jit_fn, params, batch_stats = initialize_jax_models(args.motion_model_path)

    rng = np.random.default_rng(args.seed)
    poses, covs, is_valid = _generate_synthetic_sequence(args.n_frames, N_JOINTS, rng)

    pred_seq, cov_seq, gt_seq = _run_sequential(poses, covs, is_valid, jit_fn, params, batch_stats)
    pred_bat, cov_bat, gt_bat = _run_batched(poses, covs, is_valid, jit_fn, params, batch_stats)

    mpjpe_diff = abs(_mpjpe(pred_seq, gt_seq) - _mpjpe(pred_bat, gt_bat))
    cov_diff   = abs(_coverage_99(pred_seq, cov_seq, gt_seq) - _coverage_99(pred_bat, cov_bat, gt_bat))

    print(f'\nMPJPE seq={_mpjpe(pred_seq, gt_seq):.4f}mm  bat={_mpjpe(pred_bat, gt_bat):.4f}mm  diff={mpjpe_diff:.6f}mm')
    print(f'Coverage seq={_coverage_99(pred_seq, cov_seq, gt_seq):.4f}%  bat={_coverage_99(pred_bat, cov_bat, gt_bat):.4f}%  diff={cov_diff:.6f}%')
    print('PASS' if mpjpe_diff < MPJPE_TOL and cov_diff < COV_TOL else 'FAIL')
