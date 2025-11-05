"""
The DCTPoseTransformer performs human motion prediction using a transformer architecture.
It incorporates frequency-aware attention mechanisms and predicts pose uncertainties.
"""

from typing import Union
import jax.numpy as jnp
from flax import linen as nn
from numpy import ndarray

from human_pose_pipeline.motion_prediction.h36m_settings import REDUCED_TIMESTEP, REDUCED_JOINT_INDICES


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

        # MultiheadAttention in Flax
        # Note: Flax doesn't have direct MultiheadAttention, we need to implement it
        # using nn.MultiHeadDotProductAttention or implement manually
        mha = nn.MultiHeadDotProductAttention(
            num_heads=self.nhead, qkv_features=self.d_model, out_features=self.d_model, name="mha"
        )

        # Self-attention: query = key = value
        attn_output = mha(weighted_x, weighted_x)  # torch is three inputs, query,key,value, why here 2

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
    Processes input uncertainties parallel to main network.
    Learns how much uncertainty information should influence the main prediction.
    """

    uncertainty_dim: int
    d_model: int

    @nn.compact
    def __call__(self, uncertainty):
        """
        Embed and scale uncertainty features.

        Args:
            uncertainty: Input uncertainty features

        Returns:
            Scaled uncertainty embeddings
        """
        # Embedding network
        x = nn.Dense(self.d_model, name="embed_0")(uncertainty)
        x = nn.LayerNorm(name="embed_norm")(x)
        x = nn.gelu(x)

        # Learnable scale (initialized to 0)
        uncertainty_scale = self.param("uncertainty_scale", nn.initializers.zeros, (1,))

        scale = nn.sigmoid(uncertainty_scale)
        return x * scale


class UncertaintyHead(nn.Module):
    """
    Predicts pose uncertainties using both pose features and embedded uncertainties.
    Outputs variance parameters and covariance matrix factors for each joint.
    """

    d_model: int
    seq_len: int
    seq_len_output: int
    num_joints: int = 22
    coords_per_joint: int = 3

    def setup(self):
        """Setup layers that should always exist."""
        params_per_joint = self.coords_per_joint * 2

        # MLP for processing pose features
        self.mlp_0 = nn.Dense(1024)
        self.mlp_1 = nn.Dense(512)
        self.mlp_2 = nn.Dense(
            self.seq_len_output * self.num_joints * params_per_joint,
            kernel_init=nn.initializers.normal(stddev=0.01),
            bias_init=nn.initializers.zeros,
        )

        # Network for processing embedded uncertainties (always create)
        self.unc_proc_0 = nn.Dense(512)
        self.unc_proc_1 = nn.Dense(
            self.seq_len_output * self.num_joints * params_per_joint,
            kernel_init=nn.initializers.normal(stddev=0.01),
            bias_init=nn.initializers.zeros,
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

        var_params = uncertainty_params[..., : self.coords_per_joint]
        cov_params = uncertainty_params[..., self.coords_per_joint :]

        return var_params, cov_params


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
    def __call__(self, x, input_uncertainty=None, train: bool = True):
        """
        Forward pass through the model.

        Args:
            x: Input pose sequence [batch_size, seq_len, input_dim]
            input_uncertainty: Optional external uncertainty information (currently unused)
            train: Whether in training mode (currently unused)

        Returns:
            tuple: (predicted poses, (variance parameters, covariance parameters))
        """
        batch_size = x.shape[0]
        offset = x[:, -1:, :]
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
        uncertainty_head = UncertaintyHead(
            self.d_model,
            self.seq_len,
            self.seq_len_output,
            num_joints=num_joints,
            coords_per_joint=3,
            name="uncertainty_head",
        )

        var_params, cov_params = uncertainty_head(features[-1], None)

        # Convert to mm
        freq_poses = freq_poses * self.unit_conversion

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
            # var_params = var_params[:, self.reduced_timestep, self.reduced_joints, :]
            # cov_params = cov_params[:, self.reduced_timestep, self.reduced_joints, :]

            # The OOD detection only works for a single output tensor
            return pred_poses
        else:
            return pred_poses, (var_params, cov_params)
