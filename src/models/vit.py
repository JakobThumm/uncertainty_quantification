import jax.numpy as jnp
from flax import linen as nn
from typing import Any

class ViT(nn.Module):
    img_size: int = 28
    patch_size: int = 7
    in_channels: int = 1
    num_classes: int = 10
    dim: int = 64
    depth: int = 4
    num_heads: int = 1
    mlp_dim: int = 128
    dropout_rate: float = 0.1

    def setup(self):
        # convolutional patch embedder
        self.patch_proj = nn.Conv(
            features=self.dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            name="patch_proj"
        )
        self.num_patches = (self.img_size // self.patch_size) ** 2

        # learned [CLS] token + absolute pos embedding
        self.cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=1.0),
            (1, 1, self.dim)
        )
        self.pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=1.0),
            (1, 1 + self.num_patches, self.dim)
        )

        # stack of transformer blocks
        self.blocks = [
            EncoderBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate
            )
            for _ in range(self.depth)
        ]

        # final classification head
        self.classifier = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(self.num_classes)
        ])

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        # pull in attention‐mask and relative_position_index from wrapper
        attn_mask = self.variable(
            "attention_mask", "mask",
            lambda: jnp.ones(
                (1, 1, self.num_patches + 1, self.num_patches + 1),
                dtype=jnp.bool_
            )
        ).value
        rel_index = self.variable(
            "relative_position_index", "index",
            lambda: jnp.zeros(
                (self.num_patches + 1, self.num_patches + 1),
                dtype=jnp.int32
            )
        ).value

        # patch embedding → (B, N, dim)
        b = x.shape[0]
        x = self.patch_proj(x)                  # (B, H', W', dim)
        x = x.reshape(b, -1, self.dim)          # (B, N, dim)

        # prepend [CLS] and add absolute pos
        cls = jnp.tile(self.cls_token, (b, 1, 1))    # (B, 1, dim)
        x = jnp.concatenate([cls, x], axis=1)        # (B, N+1, dim)
        x = x + self.pos_embed[:, : x.shape[1], :]   # (1, N+1, dim) broadcast

        # transformer blocks with mask
        for block in self.blocks:
            x = block(x, attn_mask, deterministic)

        # classify via the [CLS] token
        return self.classifier(x[:, 0])


class EncoderBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attn_mask: jnp.ndarray,
        deterministic: bool
    ) -> jnp.ndarray:
        # -- Self‐Attention with mask --
        y = nn.LayerNorm(name="ln1")(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            dropout_rate=self.dropout_rate,
            name="attn"
        )(
            y, y,
            mask=attn_mask,
            deterministic=deterministic
        )
        x = x + y

        # -- MLP block --
        y = nn.LayerNorm(name="ln2")(x)
        y = nn.Dense(self.mlp_dim, name="mlp_fc1")(y)
        y = nn.gelu(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(self.dim, name="mlp_fc2")(y)
        y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)

        return x + y
