"""
PyTorch-compatible MultiheadAttention implementation in JAX/Flax.

This module provides a JAX/Flax implementation that mirrors PyTorch's nn.MultiheadAttention
architecture and behavior, enabling easier model porting and comparison.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
from typing import Optional, Tuple, Callable
from flax.typing import Array, PRNGKey, Dtype, Initializer


class PyTorchMultiheadAttention(nn.Module):
    """
    JAX/Flax implementation of PyTorch's nn.MultiheadAttention.

    Mirrors the interface and behavior of torch.nn.MultiheadAttention for easier
    compatibility between PyTorch and JAX implementations.

    Attributes:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability (default: 0.0)
        bias: If True, adds bias to input/output projection layers (default: True)
        add_bias_kv: If True, adds bias to key and value sequences (default: False)
        add_zero_attn: If True, adds new batch of zeros to key/value sequences (default: False)
        kdim: Total number of features for keys (default: None, uses embed_dim)
        vdim: Total number of features for values (default: None, uses embed_dim)
        batch_first: If True, input/output tensors are (batch, seq, feature) (default: False)
    """

    embed_dim: int
    num_heads: int
    dropout: float = 0.0
    bias: bool = True
    add_bias_kv: bool = False
    add_zero_attn: bool = False
    kdim: Optional[int] = None
    vdim: Optional[int] = None
    batch_first: bool = False
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32

    def setup(self):
        """Initialize projection layers and parameters."""
        # Compute derived values (don't assign to self in setup)
        kdim = self.kdim if self.kdim is not None else self.embed_dim
        vdim = self.vdim if self.vdim is not None else self.embed_dim
        _qkv_same_embed_dim = kdim == self.embed_dim and vdim == self.embed_dim

        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"

        head_dim = self.embed_dim // self.num_heads

        # Initialize projections using Xavier uniform (matching PyTorch default)
        kernel_init = nn.initializers.xavier_uniform()
        bias_init = nn.initializers.zeros

        if _qkv_same_embed_dim:
            # Combined QKV projection (like PyTorch's in_proj_weight)
            self.in_proj = nn.Dense(
                3 * self.embed_dim,
                use_bias=self.bias,
                kernel_init=kernel_init,
                bias_init=bias_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name='in_proj'
            )
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
        else:
            # Separate Q, K, V projections
            self.in_proj = None
            self.q_proj = nn.Dense(
                self.embed_dim,
                use_bias=False,
                kernel_init=kernel_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name='q_proj'
            )
            self.k_proj = nn.Dense(
                self.embed_dim,
                use_bias=False,
                kernel_init=kernel_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name='k_proj'
            )
            self.v_proj = nn.Dense(
                self.embed_dim,
                use_bias=False,
                kernel_init=kernel_init,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name='v_proj'
            )

        # Output projection
        self.out_proj = nn.Dense(
            self.embed_dim,
            use_bias=self.bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='out_proj'
        )

        # Optional bias for keys and values
        if self.add_bias_kv:
            self.bias_k = self.param(
                'bias_k',
                nn.initializers.xavier_normal(),
                (1, 1, self.embed_dim)
            )
            self.bias_v = self.param(
                'bias_v',
                nn.initializers.xavier_normal(),
                (1, 1, self.embed_dim)
            )

        # Dropout
        if self.dropout > 0.0:
            self.attn_dropout = nn.Dropout(self.dropout)

    def _transpose_for_scores(self, x: Array) -> Array:
        """
        Reshape from [batch, seq_len, embed_dim] to [batch, num_heads, seq_len, head_dim]
        or from [seq_len, batch, embed_dim] to [batch, num_heads, seq_len, head_dim]
        """
        head_dim = self.embed_dim // self.num_heads
        if self.batch_first:
            # Input: [batch, seq_len, embed_dim]
            batch_size, seq_len, _ = x.shape
            x = x.reshape(batch_size, seq_len, self.num_heads, head_dim)
            return jnp.transpose(x, (0, 2, 1, 3))  # [batch, num_heads, seq_len, head_dim]
        else:
            # Input: [seq_len, batch, embed_dim]
            seq_len, batch_size, _ = x.shape
            x = x.reshape(seq_len, batch_size, self.num_heads, head_dim)
            return jnp.transpose(x, (1, 2, 0, 3))  # [batch, num_heads, seq_len, head_dim]

    def _transpose_from_scores(self, x: Array) -> Array:
        """
        Reshape from [batch, num_heads, seq_len, head_dim] back to original format
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = jnp.transpose(x, (0, 2, 1, 3))  # [batch, seq_len, num_heads, head_dim]
        x = x.reshape(batch_size, seq_len, self.embed_dim)

        if not self.batch_first:
            x = jnp.transpose(x, (1, 0, 2))  # [seq_len, batch, embed_dim]

        return x

    def _merge_masks(
        self,
        attn_mask: Optional[Array],
        key_padding_mask: Optional[Array],
        batch_size: int,
        seq_len_q: int,
        seq_len_k: int
    ) -> Optional[Array]:
        """
        Merge attention mask and key padding mask into a single mask.

        Args:
            attn_mask: Attention mask of shape [seq_len_q, seq_len_k] or
                      [batch * num_heads, seq_len_q, seq_len_k]
            key_padding_mask: Padding mask of shape [batch, seq_len_k]
            batch_size: Batch size
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length

        Returns:
            Merged mask of shape [batch, num_heads, seq_len_q, seq_len_k]
        """
        merged_mask = None

        # Handle key padding mask
        if key_padding_mask is not None:
            # Expand to [batch, 1, 1, seq_len_k]
            merged_mask = key_padding_mask[:, None, None, :]
            # Broadcast to [batch, num_heads, seq_len_q, seq_len_k]
            merged_mask = jnp.broadcast_to(
                merged_mask,
                (batch_size, self.num_heads, seq_len_q, seq_len_k)
            )

        # Handle attention mask
        if attn_mask is not None:
            if attn_mask.ndim == 2:
                # Shape: [seq_len_q, seq_len_k]
                # Expand to [1, 1, seq_len_q, seq_len_k]
                attn_mask_expanded = attn_mask[None, None, :, :]
                # Broadcast to [batch, num_heads, seq_len_q, seq_len_k]
                attn_mask_expanded = jnp.broadcast_to(
                    attn_mask_expanded,
                    (batch_size, self.num_heads, seq_len_q, seq_len_k)
                )
            elif attn_mask.ndim == 3:
                # Shape: [batch * num_heads, seq_len_q, seq_len_k]
                attn_mask_expanded = attn_mask.reshape(
                    batch_size, self.num_heads, seq_len_q, seq_len_k
                )
            else:
                raise ValueError(f"attn_mask must be 2D or 3D, got {attn_mask.ndim}D")

            if merged_mask is None:
                merged_mask = attn_mask_expanded
            else:
                # Combine masks using logical AND
                merged_mask = jnp.logical_and(merged_mask, attn_mask_expanded)

        return merged_mask

    @nn.compact
    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        key_padding_mask: Optional[Array] = None,
        need_weights: bool = True,
        attn_mask: Optional[Array] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        deterministic: Optional[bool] = None,
    ) -> Tuple[Array, Optional[Array]]:
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor. Shape depends on batch_first:
                - If batch_first=True: [batch, seq_len_q, embed_dim]
                - If batch_first=False: [seq_len_q, batch, embed_dim]
            key: Key tensor. Shape depends on batch_first:
                - If batch_first=True: [batch, seq_len_k, kdim]
                - If batch_first=False: [seq_len_k, batch, kdim]
            value: Value tensor. Shape depends on batch_first:
                - If batch_first=True: [batch, seq_len_k, vdim]
                - If batch_first=False: [seq_len_k, batch, vdim]
            key_padding_mask: Boolean mask indicating padding positions in key.
                Shape: [batch, seq_len_k]. True indicates positions to ignore.
            need_weights: If True, returns attention weights
            attn_mask: Attention mask. Shape: [seq_len_q, seq_len_k] or
                      [batch * num_heads, seq_len_q, seq_len_k]
            average_attn_weights: If True and need_weights=True, returns averaged weights
            is_causal: If True, applies causal masking
            deterministic: If False, applies dropout

        Returns:
            Tuple of (attn_output, attn_output_weights)
            - attn_output: Attention output with same shape as query
            - attn_output_weights: Attention weights if need_weights=True, else None
        """
        # Handle batch dimension
        if self.batch_first:
            batch_size, seq_len_q, _ = query.shape
            _, seq_len_k, _ = key.shape
        else:
            seq_len_q, batch_size, _ = query.shape
            seq_len_k, batch_size_k, _ = key.shape
            assert batch_size == batch_size_k, "Batch sizes must match"

        # Transpose if not batch_first
        if not self.batch_first:
            query = jnp.transpose(query, (1, 0, 2))  # [batch, seq_len_q, embed_dim]
            key = jnp.transpose(key, (1, 0, 2))      # [batch, seq_len_k, kdim]
            value = jnp.transpose(value, (1, 0, 2))  # [batch, seq_len_k, vdim]

        # Project Q, K, V
        if self.in_proj is not None:
            # Combined projection
            qkv = self.in_proj(query)  # Assumes query=key=value for self-attention
            q, k, v = jnp.split(qkv, 3, axis=-1)
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        # Add bias to key and value if specified
        if self.add_bias_kv:
            k = jnp.concatenate([k, jnp.broadcast_to(self.bias_k, (batch_size, 1, self.embed_dim))], axis=1)
            v = jnp.concatenate([v, jnp.broadcast_to(self.bias_v, (batch_size, 1, self.embed_dim))], axis=1)
            seq_len_k += 1

            if key_padding_mask is not None:
                # Add False to padding mask for the added bias positions
                key_padding_mask = jnp.concatenate(
                    [key_padding_mask, jnp.zeros((batch_size, 1), dtype=jnp.bool_)],
                    axis=1
                )

        # Add zero attention if specified
        if self.add_zero_attn:
            k = jnp.concatenate([k, jnp.zeros((batch_size, 1, self.embed_dim), dtype=k.dtype)], axis=1)
            v = jnp.concatenate([v, jnp.zeros((batch_size, 1, self.embed_dim), dtype=v.dtype)], axis=1)
            seq_len_k += 1

            if key_padding_mask is not None:
                key_padding_mask = jnp.concatenate(
                    [key_padding_mask, jnp.zeros((batch_size, 1), dtype=jnp.bool_)],
                    axis=1
                )

        # Reshape for multi-head attention
        # [batch, seq_len, embed_dim] -> [batch, num_heads, seq_len, head_dim]
        q = self._transpose_for_scores(q)
        k = self._transpose_for_scores(k)
        v = self._transpose_for_scores(v)

        # Compute attention scores
        # [batch, num_heads, seq_len_q, head_dim] @ [batch, num_heads, head_dim, seq_len_k]
        # -> [batch, num_heads, seq_len_q, seq_len_k]
        head_dim = self.embed_dim // self.num_heads
        attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
        attention_scores = attention_scores / jnp.sqrt(head_dim)

        # Apply causal mask if specified
        if is_causal:
            causal_mask = jnp.tril(jnp.ones((seq_len_q, seq_len_k), dtype=jnp.bool_))
            if attn_mask is not None:
                attn_mask = jnp.logical_and(attn_mask, causal_mask)
            else:
                attn_mask = causal_mask

        # Merge and apply masks
        merged_mask = self._merge_masks(
            attn_mask, key_padding_mask, batch_size, seq_len_q, seq_len_k
        )

        if merged_mask is not None:
            # Apply mask by setting masked positions to large negative value
            attention_scores = jnp.where(
                merged_mask,
                attention_scores,
                jnp.finfo(attention_scores.dtype).min
            )

        # Apply softmax
        attention_probs = jax.nn.softmax(attention_scores, axis=-1)

        # Apply dropout
        if self.dropout > 0.0 and deterministic is not None and not deterministic:
            attention_probs = self.attn_dropout(attention_probs)

        # Compute attention output
        # [batch, num_heads, seq_len_q, seq_len_k] @ [batch, num_heads, seq_len_k, head_dim]
        # -> [batch, num_heads, seq_len_q, head_dim]
        context = jnp.matmul(attention_probs, v)

        # Reshape back to original format
        # [batch, num_heads, seq_len_q, head_dim] -> [batch, seq_len_q, embed_dim]
        # or [seq_len_q, batch, embed_dim] if not batch_first
        context = self._transpose_from_scores(context)

        # Output projection
        output = self.out_proj(context)

        # Prepare attention weights for return
        attn_weights = None
        if need_weights:
            if average_attn_weights:
                # Average over heads: [batch, num_heads, seq_len_q, seq_len_k] -> [batch, seq_len_q, seq_len_k]
                attn_weights = jnp.mean(attention_probs, axis=1)
            else:
                # Return per-head weights: [batch, num_heads, seq_len_q, seq_len_k]
                attn_weights = attention_probs

            # Adjust for batch_first
            if not self.batch_first:
                # Not needed for weights, they're always [batch, ...]
                pass

        return output, attn_weights


# Utility function for creating causal masks
def create_causal_mask(seq_len: int) -> Array:
    """Create a causal mask for autoregressive attention."""
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))


# Utility function for creating padding masks
def create_padding_mask(lengths: Array, max_len: int) -> Array:
    """
    Create padding mask from sequence lengths.

    Args:
        lengths: Array of sequence lengths [batch]
        max_len: Maximum sequence length

    Returns:
        Boolean mask [batch, max_len] where True indicates valid positions
    """
    batch_size = lengths.shape[0]
    positions = jnp.arange(max_len)[None, :]  # [1, max_len]
    lengths = lengths[:, None]  # [batch, 1]
    return positions < lengths  # [batch, max_len]
