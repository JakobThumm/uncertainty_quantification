"""
DCTPoseTransformer with Residual Log-Likelihood Estimation (RLE) uncertainty head.

Replaces the Cholesky covariance head with:
  - A per-joint per-coordinate sigma predictor (heteroscedastic scale)
  - A RealNVP normalizing flow that models the distribution of normalized
    residuals r = (y - mu) / sigma

The flow architecture mirrors the YOLO26 Pose RealNVP (ultralytics/nn/modules/block.py),
extended from 2D to 3D joint coordinates.

References:
    RLE paper: https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Human_Pose_Regression_With_Residual_Log-Likelihood_Estimation_ICCV_2021_paper.pdf
    Real NVP:  https://arxiv.org/abs/1605.08803
    YOLO impl: ultralytics/ultralytics/nn/modules/block.py::RealNVP
"""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from human_pose_pipeline.motion_prediction.h36m_settings import (
    N_JOINTS,
    REDUCED_TIMESTEP,
    REDUCED_JOINT_INDICES,
)
from src.models.dct_pose_transformer_pytorch_attn import (
    DCTPoseTransformerBlockPyTorch,
    UncertaintyEmbedding,
    get_dct_matrix,
)

# ---------------------------------------------------------------------------
# RealNVP coupling masks for 3D input.
# Shape (6, 3): mask[i][d] == 1 means dimension d is kept fixed in layer i.
# Alternating [[1,1,0], [0,0,1]] so every dim is transformed across the 6 layers.
# ---------------------------------------------------------------------------
_MASKS_3D = jnp.array([[1, 1, 0], [0, 0, 1]] * 3, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _CouplingMLP(nn.Module):
    """Two-hidden-layer MLP used for scale / translate in each coupling layer."""

    hidden_dim: int
    out_dim: int
    use_tanh: bool  # True for scale network (bounds scale outputs)

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.silu(x)
        x = nn.Dense(self.out_dim)(x)
        if self.use_tanh:
            x = jnp.tanh(x)
        return x


class RealNVPFlax(nn.Module):
    """
    RealNVP normalizing flow for 3D normalised pose residuals.

    Operates unconditionally on r = (y - mu) / sigma ∈ ℝ³ per joint.
    Uses 6 affine coupling layers with alternating masks (adapted from the
    YOLO26 Pose 2D implementation to 3D).

    The key method for training is `log_prob`, which returns log p(r) —
    the "log_phi" term in the RLE loss.
    """

    hidden_dim: int = 64
    n_layers: int = 6  # must be even; _MASKS_3D has period 2

    def setup(self):
        self.s_nets = [
            _CouplingMLP(hidden_dim=self.hidden_dim, out_dim=3, use_tanh=True, name=f"s_{i}")
            for i in range(self.n_layers)
        ]
        self.t_nets = [
            _CouplingMLP(hidden_dim=self.hidden_dim, out_dim=3, use_tanh=False, name=f"t_{i}")
            for i in range(self.n_layers)
        ]

    def backward_p(self, x):
        """
        Map residuals x from data space to latent space (inverse flow direction).

        Applies coupling layers in REVERSE order, consistent with Real NVP.
        Each layer: z_unmasked = (x_unmasked - t(x_masked)) * exp(-s(x_masked))

        Args:
            x: [N, 3] normalised residuals

        Returns:
            (z, log_det): latent vectors [N, 3] and log |det dz/dx| [N].
                          log_det is the log-determinant of the Jacobian of
                          the inverse mapping (data → latent).
        """
        z = x
        log_det = jnp.zeros(x.shape[0])

        for i in reversed(range(self.n_layers)):
            mask = _MASKS_3D[i]                              # [3]
            z_masked = mask * z                              # [N, 3] — fixed part fed into nets
            s = self.s_nets[i](z_masked) * (1.0 - mask)    # [N, 3] — nonzero only at unmasked dims
            t = self.t_nets[i](z_masked) * (1.0 - mask)    # [N, 3]
            z = (1.0 - mask) * (z - t) * jnp.exp(-s) + z_masked
            log_det = log_det - jnp.sum(s, axis=-1)         # inverse det = -sum(s) per layer

        return z, log_det

    def log_prob(self, x):
        """
        Compute log p(x) under the learned flow distribution.

        Uses the change-of-variables formula:
            log p(x) = log p_Z(z) + log |det dz/dx|
        where z = backward_p(x) and p_Z = N(0, I_3).

        Args:
            x: [N, 3] normalised residuals

        Returns:
            log_phi: [N] log probability in data space
        """
        z, log_det = self.backward_p(x)
        log_prior = -0.5 * jnp.sum(z ** 2, axis=-1) - 0.5 * 3 * math.log(2 * math.pi)
        return log_prior + log_det

    def __call__(self, x):
        """Default call returns log_prob (used during training via RLE loss)."""
        return self.log_prob(x)


class RLESigmaHead(nn.Module):
    """
    Predicts per-joint per-coordinate log-sigma (log scale) for the RLE head.

    Uses stopped-gradient transformer features (analogous to UncertaintyHeadCov).
    Outputs are in the same unit as pred_poses (millimetres after unit_conversion).
    """

    d_model: int
    seq_len: int
    seq_len_output: int
    num_joints: int

    def setup(self):
        n_out = self.seq_len_output * self.num_joints * 3
        self.fc0 = nn.Dense(512)
        self.fc1 = nn.Dense(256)
        self.log_sigma_out = nn.Dense(n_out)

    def __call__(self, features):
        """
        Args:
            features: [B, seq_len, d_model] transformer features (detached)

        Returns:
            log_sigma: [B, seq_len_output, num_joints, 3]
        """
        B = features.shape[0]
        flat = features.reshape(B, -1)
        h = nn.relu(self.fc0(flat))
        h = nn.relu(self.fc1(h))
        log_sigma = self.log_sigma_out(h).reshape(B, self.seq_len_output, self.num_joints, 3)
        return log_sigma


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def rle_loss(pred_poses, y_true, log_sigma, log_phi, residual=True):
    """
    Residual Log-Likelihood Estimation loss.

    Matches the YOLO26 RLE formulation (ultralytics/utils/loss.py::RLELoss):

        loss_per_dim = log_sigma - log_phi [+ log(2*sigma) + |error|]

    With ``residual=True`` (default), a Laplace base-distribution term is added
    on top of the flow. The flow then models the residual deviation from Laplace,
    which is more robust to outliers than a pure Gaussian NLL.

    Args:
        pred_poses: [B, T, J, 3]  predicted mean poses in mm
        y_true:     [B, T, J, 3]  ground truth poses in mm
        log_sigma:  [B, T, J, 3]  predicted log-scale per coordinate
        log_phi:    [B*T*J]       log probability under the flow (from RealNVPFlax.log_prob)
        residual:   if True, add Laplace residual term (recommended)

    Returns:
        Scalar loss (mean over batch × timesteps × joints × coordinates).
    """
    sigma = jnp.exp(log_sigma)                           # [B, T, J, 3]
    error = (y_true - pred_poses) / (sigma + 1e-6)       # normalised residual

    B, T, J, D = log_sigma.shape
    N = B * T * J

    log_sigma_flat = log_sigma.reshape(N, D)   # [N, D]
    log_phi_flat = log_phi.reshape(N, 1)        # [N, 1] — broadcast over D (matching YOLO)
    sigma_flat = sigma.reshape(N, D)
    error_flat = error.reshape(N, D)

    loss = log_sigma_flat - log_phi_flat        # [N, D]

    if residual:
        # Laplace residual: log(2σ) + |error|  (derived from -log Laplace(y; μ, σ))
        loss = loss + jnp.log(sigma_flat * 2.0) + jnp.abs(error_flat)

    return jnp.mean(loss)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DCTPoseTransformerRLE(nn.Module):
    """
    DCT Pose Transformer with a RealNVP-based RLE uncertainty head.

    Architecture is identical to DCTPoseTransformer except the Cholesky
    covariance head (UncertaintyHeadCov) is replaced by:
      1. RLESigmaHead  — predicts per-joint per-coord log-sigma
      2. RealNVPFlax   — normalising flow over 3D normalised residuals

    Training loop example::

        pred_poses, (log_sigma, log_phi) = model.apply(
            params, x, y_true=y_true_3d, train=True
        )
        loss = rle_loss(pred_poses, y_true_3d, log_sigma, log_phi)

    Inference example::

        pred_poses, (log_sigma, _) = model.apply(params, x, train=False)
        sigma = jnp.exp(log_sigma)  # [B, T_out, J, 3] per-coord uncertainty in mm

    Args:
        input_dim:       Number of pose coordinates (N_JOINTS * 3).
        d_model:         Transformer embedding dimension.
        nhead:           Number of attention heads.
        num_layers:      Number of transformer blocks.
        seq_len:         Input sequence length.
        seq_len_output:  Output sequence length.
        unit_conversion: mm → m conversion factor (default 1000.0).
        dropout:         Dropout probability.
        reduced_size:    If True, return a single flat pose vector (for OOD eval).
        flow_hidden_dim: Hidden size in RealNVP coupling MLPs.
        flow_n_layers:   Number of RealNVP coupling layers (must be even).
    """

    input_dim: int = 39
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    seq_len: int = 50
    seq_len_output: int = 10
    unit_conversion: float = 1000.0
    dropout: float = 0.0
    reduced_size: bool = False
    flow_hidden_dim: int = 64
    flow_n_layers: int = 6

    def __post_init__(self):
        self.dct_mat, self.idct_mat = get_dct_matrix(self.seq_len)
        self.reduced_timestep = REDUCED_TIMESTEP
        self.reduced_joints = jnp.array(REDUCED_JOINT_INDICES)
        return super().__post_init__()

    @nn.compact
    def __call__(self, x, y_true=None, train=True):
        """
        Forward pass.

        Args:
            x:      [B, seq_len, input_dim] pose sequence, optionally with
                    uncertainty covariances appended along the last axis.
            y_true: Optional [B, seq_len_output, num_joints * 3] ground-truth
                    poses in mm. When provided, the flow is evaluated at the
                    normalised residuals and ``log_phi`` is returned for use
                    in ``rle_loss``. Pass ``None`` at inference time.
            train:  Enables dropout when True.

        Returns:
            If ``reduced_size=True``:
                pred_poses [B, reduced_dim]

            Otherwise:
                (pred_poses [B, T_out, input_dim],
                 (log_sigma [B, T_out, J, 3],
                  log_phi   [B*T_out*J] or None))
        """
        deterministic = not train
        batch_size = x.shape[0]
        input_dim = x.shape[2]

        # ------------------------------------------------------------------
        # 1. Split input into poses and optional input uncertainties
        # ------------------------------------------------------------------
        if input_dim == N_JOINTS * 3:
            input_uncertainty = None
        else:
            pose_dim = N_JOINTS * 3
            input_pose = x[:, :, :pose_dim]
            input_uncertainty = x[:, :, pose_dim:].reshape(
                batch_size, self.seq_len, N_JOINTS, 3, 3
            )
            x = input_pose
            input_dim = pose_dim

        # ------------------------------------------------------------------
        # 2. Offset subtraction, DCT, unit conversion
        # ------------------------------------------------------------------
        offset = x[:, -1:, :]
        x = x - offset
        x = jnp.transpose(
            jnp.matmul(jnp.transpose(x, axes=(0, 2, 1)), jnp.transpose(self.dct_mat)),
            (0, 2, 1),
        )
        x = x / self.unit_conversion  # → metres

        # ------------------------------------------------------------------
        # 3. Input embedding + positional encoding
        # ------------------------------------------------------------------
        x = nn.Dense(self.d_model, name="input_embed_0")(x)
        x = nn.LayerNorm(name="input_embed_norm")(x)
        x = nn.gelu(x)

        freq_pos_embed = self.param(
            "freq_pos_embed",
            nn.initializers.normal(stddev=1.0),
            (1, self.seq_len, self.d_model),
        )
        x = x + freq_pos_embed

        # ------------------------------------------------------------------
        # 4. Optional uncertainty conditioning (same as base model)
        # ------------------------------------------------------------------
        uncertainty_features = None
        if input_uncertainty is not None:
            uncertainty_embedding = UncertaintyEmbedding(
                self.d_model, seq_len=self.seq_len, num_joints=N_JOINTS,
                name="uncertainty_embedding",
            )
            input_uncertainty = input_uncertainty / (self.unit_conversion ** 2)
            uncertainty_features = uncertainty_embedding(input_uncertainty)
            x = x + uncertainty_features

        # ------------------------------------------------------------------
        # 5. Transformer backbone
        # ------------------------------------------------------------------
        features = []
        for i in range(self.num_layers):
            block = DCTPoseTransformerBlockPyTorch(
                self.d_model, self.nhead, dropout=self.dropout,
                name=f"transformer_block_{i}",
            )
            x = block(x, deterministic=deterministic)
            features.append(x)

        # ------------------------------------------------------------------
        # 6. DCT-domain pose decoder (frequency-split, same as base model)
        # ------------------------------------------------------------------
        half_dim = x.shape[-1] // 2
        low_freq = x[..., :half_dim]
        high_freq = x[..., half_dim:]

        low_freq_features = (input_dim + 1) // 2
        high_freq_features = input_dim - low_freq_features

        low_freq_out = nn.Dense(low_freq_features, name="low_freq_decoder")(low_freq)
        high_freq_out = nn.Dense(high_freq_features, name="high_freq_decoder")(high_freq)

        freq_poses = jnp.concatenate([low_freq_out, high_freq_out], axis=-1)

        # IDCT + unit conversion + offset
        freq_poses = freq_poses * self.unit_conversion          # → mm
        pred_poses_seq = jnp.transpose(
            jnp.matmul(
                jnp.transpose(freq_poses, (0, 2, 1)),
                jnp.transpose(self.idct_mat, (1, 0)),
            ),
            (0, 2, 1),
        )
        pred_poses_seq = pred_poses_seq[:, : self.seq_len_output, :] + offset
        # pred_poses_seq: [B, T_out, input_dim]

        # ------------------------------------------------------------------
        # 7. RLE uncertainty head (sigma + flow)
        #    Uses stopped-gradient features, same pattern as UncertaintyHeadCov.
        # ------------------------------------------------------------------
        num_joints = input_dim // 3

        features_detached = jax.lax.stop_gradient(features[-1])
        if uncertainty_features is not None:
            unc_feat_detached = jax.lax.stop_gradient(uncertainty_features)
        else:
            unc_feat_detached = None

        # Sigma prediction
        sigma_head = RLESigmaHead(
            d_model=self.d_model,
            seq_len=self.seq_len,
            seq_len_output=self.seq_len_output,
            num_joints=num_joints,
            name="rle_sigma_head",
        )
        # Optionally fuse uncertainty features into sigma prediction
        if unc_feat_detached is not None:
            sigma_features = features_detached + unc_feat_detached
        else:
            sigma_features = features_detached

        log_sigma = sigma_head(sigma_features)   # [B, T_out, J, 3]

        # Normalising flow
        flow = RealNVPFlax(
            hidden_dim=self.flow_hidden_dim,
            n_layers=self.flow_n_layers,
            name="rle_flow",
        )

        # ------------------------------------------------------------------
        # 8. Compute flow log-probability if ground truth is provided
        # ------------------------------------------------------------------
        log_phi = None
        if y_true is not None:
            # y_true: [B, T_out, J*3] in mm  →  [B, T_out, J, 3]
            y_true_3d = y_true.reshape(batch_size, self.seq_len_output, num_joints, 3)
            pred_3d = pred_poses_seq.reshape(batch_size, self.seq_len_output, num_joints, 3)
            sigma = jnp.exp(log_sigma)                              # [B, T_out, J, 3]
            r = (y_true_3d - pred_3d) / (sigma + 1e-6)             # [B, T_out, J, 3]
            r_flat = r.reshape(-1, 3)                               # [B*T_out*J, 3]
            log_phi = flow.log_prob(r_flat)                         # [B*T_out*J]

        # ------------------------------------------------------------------
        # 9. Return
        # ------------------------------------------------------------------
        if self.reduced_size:
            pred_poses_timestep = pred_poses_seq[:, self.reduced_timestep, :]
            pred_poses_joints = pred_poses_timestep.reshape(batch_size, -1, 3)
            reduced = pred_poses_joints[:, self.reduced_joints, :]
            return reduced.reshape(batch_size, -1)

        return pred_poses_seq, (log_sigma, log_phi)
