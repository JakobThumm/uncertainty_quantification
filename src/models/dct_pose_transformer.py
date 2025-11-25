"""
The DCTPoseTransformer performs human motion prediction using a transformer architecture.
It incorporates frequency-aware attention mechanisms and predicts pose uncertainties.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax

from human_pose_pipeline.motion_prediction.h36m_settings import (
    N_JOINTS,
    REDUCED_TIMESTEP,
    REDUCED_JOINT_INDICES
)


def pose_prediction_loss(pred_poses, target_poses):
    """Simple MSE loss for pose predictions."""
    return jnp.mean(jnp.abs(pred_poses - target_poses))


def gaussian_nll_from_cholesky(y_pred, y_true, L, include_const=True, lambda_var=0.001, lambda_cov=0.01):
    """
    Gaussian NLL using Cholesky factor L such that Sigma = L L^T.

    Args:
        y_pred: [B, T, J, 3]
        y_true: [B, T, J, 3]
        L:      [B, T, J, 3, 3] (lower triangular, positive diag)
        include_const: whether to include k*log(2*pi)

    Returns:
        mean NLL (scalar)
    """
    diff = y_true - y_pred
    B, T, J, C = diff.shape
    N = B * T * J

    diff = diff.reshape(N, C, 1)
    Lf = L.reshape(N, C, C)

    # Stability (optional)
    eps = 1e-6
    Lf = Lf.at[..., jnp.arange(C), jnp.arange(C)].add(eps)

    # Mahalanobis: solve L m = diff
    m = jax.lax.linalg.triangular_solve(
        Lf, diff, lower=True, left_side=True
    )
    mahal = jnp.sum(m.squeeze(-1)**2, axis=-1)

    # log det
    diag_L = jnp.diagonal(Lf, axis1=-2, axis2=-1)
    log_det = 2.0 * jnp.sum(jnp.log(diag_L), axis=-1)

    if include_const:
        k_log_2pi = C * jnp.log(2.0 * jnp.pi)
    else:
        k_log_2pi = 0.0

    nll = 0.5 * (mahal + log_det + k_log_2pi)
    nll = jnp.mean(nll.reshape(B, T, J))

    # Variances from Cholesky L
    var_x = L[..., 0, 0]**2
    var_y = L[..., 1, 0]**2 + L[..., 1, 1]**2
    var_z = L[..., 2, 0]**2 + L[..., 2, 1]**2 + L[..., 2, 2]**2

    # Variance regularization
    inv_var = 1.0 / (var_x + var_y + var_z + 1e-6)
    reg_var = lambda_var * jnp.mean(inv_var)

    cov = L @ jnp.swapaxes(L, -1, -2)
    off_diag = cov[..., jnp.tril_indices(3, k=-1)]
    reg_cov = lambda_cov * jnp.mean(jnp.abs(off_diag))

    return nll + reg_var + reg_cov


class FrequencyAwareAttention(nn.Module):
    """
    Custom attention module that weights different frequency components differently.
    Extends standard multi-head attention with learnable frequency importance weights.
    """

    d_model: int
    nhead: int

    @nn.compact
    def __call__(self, x):
        """
        Apply frequency-weighted attention to input sequence.

        Args:
            x: Input sequence [seq_len, batch_size, d_model]

        Returns:
            Attention output with same shape as input
        """
        # Learnable weights for each frequency component
        freq_weights = self.param("freq_weights", nn.initializers.ones, (1, 1, self.d_model))

        weighted_x = x * freq_weights
        
        # Potential improvements, not tested.
        # from flash_attention_jax import flash_attention
        # from aqt import quantized_einsum

        mha = nn.MultiHeadDotProductAttention(
            num_heads=self.nhead,
            qkv_features=self.d_model,
            out_features=self.d_model,
            name="mha",
            normalize_qk=True,  # Enables training with higher LR.
            force_fp32_for_softmax=True,  # Better numerical stability for mixed precision data (prob. not needed)
            kernel_init=nn.initializers.xavier_uniform(),  # Kernel initialization
            out_kernel_init=nn.initializers.zeros,  # Output kernel init
            # attention_fn=flash_attention,  # Drop-in replacement
            # qk_attn_weights_einsum_cls=lambda: quantized_einsum,
            # attn_weights_value_einsum_cls=lambda: quantized_einsum,
            precision=jax.lax.Precision.HIGHEST,  # vs DEFAULT or HIGH
        )

        # Self-attention: query = key = value
        attn_output = mha(weighted_x, weighted_x, weighted_x, deterministic=True)

        return attn_output


class DCTPoseTransformerBlock(nn.Module):
    """
    Transformer block with separate processing paths for low and high frequency components.
    Combines frequency-aware attention with frequency-specific feed-forward networks.
    """

    d_model: int
    nhead: int
    dim_feedforward: int = 1024

    @nn.compact
    def __call__(self, x):
        """
        Process input through attention and frequency-specific networks.

        Args:
            x: Input features [seq_len, batch_size, d_model]

        Returns:
            Processed features with same shape as input
        """
        # Layer normalization and attention with residual
        norm1 = nn.LayerNorm(name="norm1")
        norm1_x = norm1(x)

        freq_attn = FrequencyAwareAttention(self.d_model, self.nhead, name="freq_attn")
        attn_output = freq_attn(norm1_x)
        x = x + attn_output

        # Layer normalization
        norm2 = nn.LayerNorm(name="norm2")
        norm2_x = norm2(x)

        # Split and process frequency components separately
        half_dim = norm2_x.shape[-1] // 2
        low_freq = norm2_x[..., :half_dim]
        high_freq = norm2_x[..., half_dim:]

        # Low frequency network
        low_freq_out = nn.Dense(self.dim_feedforward // 2, name="low_freq_0")(low_freq)
        low_freq_out = nn.gelu(low_freq_out)
        low_freq_out = nn.Dense(self.d_model // 2, name="low_freq_1")(low_freq_out)

        # High frequency network
        high_freq_out = nn.Dense(self.dim_feedforward // 2, name="high_freq_0")(high_freq)
        high_freq_out = nn.gelu(high_freq_out)
        high_freq_out = nn.Dense(self.d_model // 2, name="high_freq_1")(high_freq_out)

        # Combine and add residual
        ff_output = jnp.concatenate([low_freq_out, high_freq_out], axis=-1)
        x = x + ff_output

        return x


class UncertaintyEmbedding(nn.Module):
    """
    Modified module to process input covariance matrices.
    Processes input uncertainties parallel to main network.
    Learns how much uncertainty information should influence the main prediction.
    """

    d_model: int
    seq_len: int = 50
    num_joints: int = N_JOINTS

    def setup(self):
        """Setup layers that should always exist."""
        # Process each 3x3 covariance matrix first
        # cov_encoder: 9 -> 32 -> d_model
        self.cov_encoder_0 = nn.Dense(32)
        self.cov_encoder_1 = nn.Dense(self.d_model)

        # joint_encoder: num_joints * d_model -> d_model
        self.joint_encoder_0 = nn.Dense(self.d_model)
        self.joint_encoder_norm = nn.LayerNorm()

        # Learnable scaling factor for uncertainty influence
        self.uncertainty_scale = self.param("uncertainty_scale", nn.initializers.zeros, (1,))

    def __call__(self, uncertainty):
        """
        Embed and scale uncertainty features.

        Args:
            uncertainty: Input uncertainty covariance matrices [batch_size, seq_len, num_joints, 3, 3]

        Returns:
            Scaled uncertainty embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, num_joints, _, _ = uncertainty.shape

        # Reshape to process each covariance matrix
        flat_covs = uncertainty.reshape(-1, 9)  # Flatten each 3x3 matrix

        # Process each 3x3 covariance matrix first
        # cov_encoder: 9 -> 32 -> d_model
        x = self.cov_encoder_0(flat_covs)
        x = nn.relu(x)
        encoded_covs = self.cov_encoder_1(x)

        # Reshape back to [batch_size, seq_len, num_joints, d_model]
        encoded_covs = encoded_covs.reshape(batch_size, seq_len, num_joints, -1)

        # Process across joints
        joint_features = encoded_covs.reshape(batch_size, seq_len, -1)  # [batch, seq, joints*d_model]

        # joint_encoder: num_joints * d_model -> d_model
        x = self.joint_encoder_0(joint_features)
        x = self.joint_encoder_norm(x)
        uncertainty_features = nn.gelu(x)

        # Apply scaling
        scale = nn.sigmoid(self.uncertainty_scale)
        return uncertainty_features * scale


class UncertaintyHead(nn.Module):
    """
    Predicts pose uncertainties using both pose features and embedded uncertainties.
    Outputs variance parameters and covariance matrix factors for each joint.

    This was used in Marians experiments. We use the UncertaintyHeadCov in the paper.
    """

    d_model: int
    seq_len: int
    seq_len_output: int
    num_joints: int = 22
    coords_per_joint: int = 3

    def setup(self):
        """Setup layers that should always exist."""
        params_per_joint = self.coords_per_joint * 2

        # Initialize all layers with normal(std=0.01)
        kernel_init = nn.initializers.normal(stddev=0.01)
        bias_init = nn.initializers.zeros

        # MLP for processing pose features
        self.mlp_0 = nn.Dense(1024, kernel_init=kernel_init, bias_init=bias_init)
        self.mlp_1 = nn.Dense(512, kernel_init=kernel_init, bias_init=bias_init)
        self.mlp_2 = nn.Dense(
            self.seq_len_output * self.num_joints * params_per_joint,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )

        # Network for processing embedded uncertainties (always create)
        self.unc_proc_0 = nn.Dense(512, kernel_init=kernel_init, bias_init=bias_init)
        self.unc_proc_1 = nn.Dense(
            self.seq_len_output * self.num_joints * params_per_joint,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )

        # Learnable weight for combining pose-based and explicit uncertainties
        self.uncertainty_weight = self.param("uncertainty_weight", nn.initializers.zeros, (1,))

    def __call__(self, features, uncertainty_features=None):
        """
        Predict uncertainty parameters from features and optional explicit uncertainties.

        Args:
            features: Pose features [seq_len, batch_size, d_model]
            uncertainty_features: Optional explicit uncertainty features

        Returns:
            tuple: (variance parameters, covariance parameters)
        """
        batch_size = features.shape[1]
        params_per_joint = self.coords_per_joint * 2

        # Flatten features [batch_size, seq_len * d_model]
        flattened = jnp.transpose(features, (1, 0, 2)).reshape(batch_size, -1)

        # MLP for processing pose features
        x = self.mlp_0(flattened)
        x = nn.relu(x)
        x = self.mlp_1(x)
        x = nn.relu(x)
        uncertainty_from_features = self.mlp_2(x)

        if uncertainty_features is not None:
            # Process explicit uncertainties
            uncertainty_flat = jnp.transpose(uncertainty_features, (1, 0, 2)).reshape(batch_size, -1)
            x_unc = self.unc_proc_0(uncertainty_flat)
            x_unc = nn.relu(x_unc)
            processed_uncertainty = self.unc_proc_1(x_unc)

            # Combine with learnable weight
            weight = nn.sigmoid(self.uncertainty_weight)
            uncertainty_params = (1 - weight) * uncertainty_from_features + weight * processed_uncertainty
        else:
            uncertainty_params = uncertainty_from_features

        # Reshape and split
        uncertainty_params = uncertainty_params.reshape(
            batch_size, self.seq_len_output, self.num_joints, params_per_joint
        )

        log_vars = uncertainty_params[..., : self.coords_per_joint]
        raw_covs = uncertainty_params[..., self.coords_per_joint :]

        return log_vars, raw_covs


class UncertaintyHeadCov(nn.Module):
    """
    Predicts 3x3 covariance matrices per joint by outputting
    a valid Cholesky factor L (lower triangular, positive diag).
    """

    d_model: int
    seq_len: int
    seq_len_output: int
    num_joints: int = 22
    coords_per_joint: int = 3

    def setup(self):
        # Number of parameters needed to define a Cholesky L for 3D:
        # L = [[l11,   0,   0],
        #      [l21, l22,   0],
        #      [l31, l32, l33]]
        l_params_per_joint = 6  # (l11, l21, l22, l31, l32, l33)

        kernel_init = nn.initializers.normal(stddev=0.01)
        bias_init = nn.initializers.zeros

        # MLP for pose features
        self.mlp_0 = nn.Dense(1024, kernel_init=kernel_init, bias_init=bias_init)
        self.mlp_1 = nn.Dense(512, kernel_init=kernel_init, bias_init=bias_init)
        self.mlp_2 = nn.Dense(
            self.seq_len_output * self.num_joints * l_params_per_joint,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )

        # MLP for uncertainty feature fusion
        self.unc_proc_0 = nn.Dense(512, kernel_init=kernel_init, bias_init=bias_init)
        self.unc_proc_1 = nn.Dense(
            self.seq_len_output * self.num_joints * l_params_per_joint,
            kernel_init=kernel_init,
            bias_init=bias_init,
        )

        # Learnable fusion weight
        self.uncertainty_weight = self.param(
            "uncertainty_weight",
            nn.initializers.zeros,
            (1,),
        )

    def __call__(self, features, uncertainty_features=None):
        batch_size = features.shape[1]
        L_params_per_joint = 6

        # Flatten features: [B, T*D]
        flat_feat = jnp.transpose(features, (1, 0, 2)).reshape(batch_size, -1)

        # Pose feature branch
        x = nn.relu(self.mlp_0(flat_feat))
        x = nn.relu(self.mlp_1(x))
        unc_from_feat = self.mlp_2(x)

        # Optional explicit uncertainty branch
        if uncertainty_features is not None:
            flat_unc = jnp.transpose(uncertainty_features, (1, 0, 2)).reshape(batch_size, -1)
            u = nn.relu(self.unc_proc_0(flat_unc))
            unc_proc = self.unc_proc_1(nn.relu(u))
            w = nn.sigmoid(self.uncertainty_weight)
            unc_params = (1 - w) * unc_from_feat + w * unc_proc
        else:
            unc_params = unc_from_feat

        # Reshape into [B, T, J, 6]
        unc_params = unc_params.reshape(
            batch_size,
            self.seq_len_output,
            self.num_joints,
            L_params_per_joint,
        )

        # Split parameters
        l11_raw, l21, l22_raw, l31, l32, l33_raw = jnp.split(unc_params, 6, axis=-1)

        # Ensure positive diagonals using softplus
        eps = 1e-6
        l11 = nn.softplus(l11_raw) + eps
        l22 = nn.softplus(l22_raw) + eps
        l33 = nn.softplus(l33_raw) + eps

        # Build Cholesky matrix L
        L = jnp.zeros((*l11.shape[:-1], 3, 3), dtype=l11.dtype)
        L = L.at[..., 0, 0].set(l11[..., 0])
        L = L.at[..., 1, 0].set(l21[..., 0])
        L = L.at[..., 1, 1].set(l22[..., 0])
        L = L.at[..., 2, 0].set(l31[..., 0])
        L = L.at[..., 2, 1].set(l32[..., 0])
        L = L.at[..., 2, 2].set(l33[..., 0])

        # Covariance = L @ L^T
        cov = L @ jnp.swapaxes(L, -1, -2)

        return cov, L


def get_dct_matrix(N):
    """Compute the Discrete Cosine Transform (DCT) matrix and its inverse.

    Args:
        N (int): Size of the DCT matrix.
    Returns:
        tuple: (DCT matrix, Inverse DCT matrix), both of shape (N, N).
    """
    dct_m = jnp.eye(N)
    for k in range(N):
        for i in range(N):
            w = jnp.sqrt(2 / N)
            if k == 0:
                w = jnp.sqrt(1 / N)
            dct_m = dct_m.at[k, i].set(w * jnp.cos(jnp.pi * (i + 1 / 2) * k / N))
    idct_m = jnp.linalg.inv(dct_m)
    return dct_m, idct_m


class DCTPoseTransformer(nn.Module):
    """
    Main model for pose prediction with uncertainty estimation.
    Combines frequency-aware transformer with uncertainty prediction.
    """

    input_dim: int = 39
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    seq_len: int = 50
    seq_len_output: int = 10
    unit_conversion: float = 1000.0
    # Use a reduced output size for faster OOD evaluation
    reduced_size: bool = False

    def __post_init__(self) -> None:
        self.dct_mat, self.idct_mat = get_dct_matrix(self.seq_len)
        self.reduced_timestep = REDUCED_TIMESTEP
        self.reduced_joints = jnp.array(REDUCED_JOINT_INDICES)
        return super().__post_init__()

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        Forward pass through the model.

        Args:
            x: Input pose sequence [batch_size, seq_len, input_dim]
            train: Whether in training mode (currently unused)

        Returns:
            If reduced_size=True: predicted poses [batch_size, reduced_output_dim]
            If reduced_size=False: tuple (predicted poses, (cov, L))
                - predicted poses: [batch_size, seq_len_output, input_dim]
                - cov: Covariance matrices of the predictions. Shape: (batch_size, seq_len_output, num_joints, 3, 3)
                - L: Cholesky factors of the covariance matrices (used in gaussian_nll_from_cholesky loss).
                     Shape: (batch_size, seq_len_output, num_joints, 3, 3)
        """
        batch_size = x.shape[0]
        input_dim = x.shape[2]
        if input_dim == N_JOINTS * 3:
            input_uncertainty = None
        else:
            # Split input into poses and uncertainties
            pose_dim = N_JOINTS * 3
            input_pose = x[:, :, :pose_dim]
            input_uncertainty = x[:, :, pose_dim:]
            # Reshape uncertainty to [batch_size, seq_len, num_joints, 3, 3]
            input_uncertainty = input_uncertainty.reshape(
                batch_size, self.seq_len, N_JOINTS, 3, 3
            )
            x = input_pose
        offset = x[:, -1:, :]
        # Subtract offset
        x = x - offset
        # Apply DCT to input poses
        x = jnp.transpose(jnp.matmul(jnp.transpose(x, axes=(0, 2, 1)), jnp.transpose(self.dct_mat)), (0, 2, 1))
        # Convert to meters
        x = x / self.unit_conversion

        # Pose embedding
        x = nn.Dense(self.d_model, name="input_embed_0")(x)
        x = nn.LayerNorm(name="input_embed_norm")(x)
        x = nn.gelu(x)

        # Transpose to [seq_len, batch_size, d_model]
        x = jnp.transpose(x, (1, 0, 2))

        # Learnable frequency-based positional encoding
        freq_pos_embed = self.param(
            "freq_pos_embed", nn.initializers.normal(stddev=1.0), (self.seq_len, 1, self.d_model)
        )
        x = x + freq_pos_embed

        # Parallel uncertainty processing path
        # Always create the module (Flax requires consistent parameter tree)
        uncertainty_embedding = UncertaintyEmbedding(
            self.d_model, seq_len=self.seq_len, num_joints=N_JOINTS, name="uncertainty_embedding"
        )

        uncertainty_features = None
        if input_uncertainty is not None:
            # Convert to meters squared
            input_uncertainty = input_uncertainty / (self.unit_conversion**2)
            # Process uncertainty -> [batch, seq, d_model]
            uncertainty_features = uncertainty_embedding(input_uncertainty)
            # Transpose to match x's shape: [seq_len, batch_size, d_model]
            uncertainty_features = jnp.transpose(uncertainty_features, (1, 0, 2))
            # Add to main features
            x = x + uncertainty_features / self.unit_conversion

        # Pass through transformer blocks
        features = []
        for i in range(self.num_layers):
            block = DCTPoseTransformerBlock(self.d_model, self.nhead, name=f"transformer_block_{i}")
            x = block(x)
            features.append(x)

        # Decode poses
        x = jnp.transpose(x, (1, 0, 2))  # [batch_size, seq_len, d_model]
        half_dim = x.shape[-1] // 2
        low_freq = x[..., :half_dim]
        high_freq = x[..., half_dim:]

        # Frequency decoders
        low_freq_features = (self.input_dim + 1) // 2
        high_freq_features = self.input_dim - low_freq_features

        low_freq_out = nn.Dense(low_freq_features, name="low_freq_decoder")(low_freq)
        high_freq_out = nn.Dense(high_freq_features, name="high_freq_decoder")(high_freq)

        freq_poses = jnp.concatenate([low_freq_out, high_freq_out], axis=-1)

        # Predict uncertainties (using detached features in training)
        num_joints = self.input_dim // 3
        uncertainty_head = UncertaintyHeadCov(
            d_model=self.d_model,
            seq_len=self.seq_len,
            seq_len_output=self.seq_len_output,
            num_joints=num_joints,
            coords_per_joint=3,
            name="uncertainty_head",
        )

        # Process uncertainties with detached (stopped gradient) features
        features_detached = jax.lax.stop_gradient(features[-1])
        if uncertainty_features is not None:
            uncertainty_features_detached = jax.lax.stop_gradient(uncertainty_features)
        else:
            uncertainty_features_detached = None

        cov, L = uncertainty_head(features_detached, uncertainty_features_detached)

        # Convert to mm
        freq_poses = freq_poses * self.unit_conversion
        cov = cov * (self.unit_conversion**2)
        # Don't use unit conversion for L as the loss is too large.
        L = L * self.unit_conversion

        # Apply IDCT
        pred_poses = jnp.transpose(
            jnp.matmul(
                jnp.transpose(freq_poses, (0, 2, 1)),
                jnp.transpose(self.idct_mat, (1, 0))
            ), (0, 2, 1))

        # Add offset
        pred_poses = pred_poses[:, :self.seq_len_output, :] + offset

        if self.reduced_size:
            # Extract only the specified timestep and joints
            pred_poses_timestep = pred_poses[:, self.reduced_timestep, :]  # [batch_size, input_dim]
            pred_poses_timestep = pred_poses_timestep.reshape(batch_size, -1, 3)  # [batch_size, num_joints, 3]
            reduced_output = pred_poses_timestep[:, self.reduced_joints, :]  # [batch_size, len(reduced_joints), 3]
            pred_poses = reduced_output.reshape(batch_size, -1)  # [batch_size, len(reduced_joints)*3]
            # Similarly reduce uncertainty parameters
            # log_vars = log_vars[:, self.reduced_timestep, self.reduced_joints, :]
            # raw_covs = raw_covs[:, self.reduced_timestep, self.reduced_joints, :]

            # The OOD detection only works for a single output tensor
            return pred_poses
        else:
            return pred_poses, (cov, L)
