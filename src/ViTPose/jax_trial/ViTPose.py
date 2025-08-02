import jax
import jax.numpy as jnp
from flax import nnx


# import jax.numpy as jnp
# from jax import nn as jnn
# from flax import linen as nn

# ---------------------------------------------------------------------------#
# 1.  Three-step de-convolution head (identical to PyTorch ViTPose-Full logic)
# ---------------------------------------------------------------------------#
class ViTPoseHead(nnx.Module):
    def __init__(self, embed_dim: int, num_keypoints: int, *, rngs: nnx.Rngs = nnx.Rngs(0)):
        m1, m2, m3 = embed_dim // 2, embed_dim // 4, embed_dim // 8   # C → C/2 → C/4 → C/8

        self.reshape = nnx.Conv(embed_dim, embed_dim, (1, 1), (1, 1), padding="VALID",
                                use_bias=True, rngs=rngs)

        self.deconv1 = nnx.Sequential(
            nnx.ConvTranspose(embed_dim, m1, (4, 4), (2, 2), padding="SAME", rngs=rngs),
            nnx.BatchNorm(m1, use_running_average=False, rngs=rngs),
            jax.nn.relu,
        )
        self.deconv2 = nnx.Sequential(
            nnx.ConvTranspose(m1, m2, (4, 4), (2, 2), padding="SAME", rngs=rngs),
            nnx.BatchNorm(m2, use_running_average=False, rngs=rngs),
            jax.nn.relu,
        )
        self.deconv3 = nnx.Sequential(
            nnx.ConvTranspose(m2, m3, (4, 4), (2, 2), padding="SAME", rngs=rngs),
            nnx.BatchNorm(m3, use_running_average=False, rngs=rngs),
            jax.nn.relu,
        )

        self.out_conv = nnx.Conv(m3, num_keypoints, (1, 1), (1, 1), padding="VALID",
                                 use_bias=True, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x is NHWC (B, H/16, W/16, C)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.out_conv(x)                    # (B, H/2, W/2, K)
        return jnp.transpose(x, (0, 3, 1, 2))   # → (B, K, H/2, W/2)


# ---------------------------------------------------------------------------#
# 2.  ViTPose – minimal diff from your VisionTransformer, only the head part
# ---------------------------------------------------------------------------#
class ViTPose_base(nnx.Module):
    def __init__(
        self,
        num_keypoints: int = 17,
        in_channels: int = 3,
        img_size: int | tuple[int, int] = (256, 192),
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        hidden_size: int = 768,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        # ---------- image geometry -----------------------------------------
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        img_h, img_w = img_size
        self.grid_h = img_h // patch_size          # e.g. 256//16 = 16
        self.grid_w = img_w // patch_size          # e.g. 192//16 = 12
        n_patches = self.grid_h * self.grid_w

        # ---------- patch embed --------------------------------------------
        self.patch_embeddings = nnx.Conv(
            in_channels, hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID", use_bias=True, rngs=rngs
        )

        # ---------- positional + CLS ---------------------------------------
        init = jax.nn.initializers.truncated_normal(0.02)
        self.position_embeddings = nnx.Param(init(rngs.params(), (1, n_patches + 1, hidden_size), jnp.float32))
        self.cls_token = nnx.Param(jnp.zeros((1, 1, hidden_size)))
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        # ---------- transformer encoder (unchanged) ------------------------
        self.encoder = nnx.Sequential(*[
            TransformerEncoder(hidden_size, mlp_dim, num_heads, dropout_rate, rngs=rngs)
            for _ in range(num_layers)
        ])
        self.final_norm = nnx.LayerNorm(hidden_size, rngs=rngs)

        # ---------- pose head ----------------------------------------------
        self.pose_head = ViTPoseHead(hidden_size, num_keypoints, rngs=rngs)

    # ----------------------------------------------------------------------
    def __call__(self, x: jax.Array) -> jax.Array:
        # 1) patchify --------------------------------------------------------
        patches = self.patch_embeddings(x)                 # (B, H/16, W/16, C)
        B = patches.shape[0]
        patches = patches.reshape(B, -1, patches.shape[-1])  # (B, N, C)

        # 2) add CLS + pos-emb ---------------------------------------------
        cls = jnp.tile(self.cls_token, [B, 1, 1])
        tokens = jnp.concatenate([cls, patches], axis=1)
        tokens = self.dropout(tokens + self.position_embeddings)

        # 3) transformer ----------------------------------------------------
        tokens = self.final_norm(self.encoder(tokens))     # (B, 1+N, C)

        # 4) drop CLS, restore grid ----------------------------------------
        patch_tokens = tokens[:, 1:, :]                    # (B, N, C)
        patch_tokens = patch_tokens.reshape(B, self.grid_h, self.grid_w, -1)  # NHWC

        # 5) de-conv head ---------------------------------------------------
        return self.pose_head(patch_tokens)                # (B, K, H/2, W/2)


class TransformerEncoder(nnx.Module):
    """
    A single transformer encoder block in the ViT model, inheriting from `flax.nnx.Module`.

    Args:
        hidden_size (int): Input/output embedding dimensionality.
        mlp_dim (int): Dimension of the feed-forward/MLP block hidden layer.
        num_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate. Defaults to 0.0.
        rngs (flax.nnx.Rngs): A set of named `flax.nnx.RngStream` objects that generate a stream of JAX pseudo-random number generator (PRNG) keys. Defaults to `flax.nnx.Rngs(0)`.
    """
    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        # First layer normalization using `flax.nnx.LayerNorm`
        # before we apply Multi-Head Attentn.
        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        # The Multi-Head Attention layer (using `flax.nnx.MultiHeadAttention`).
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            decode=False,
            deterministic=False,
            rngs=rngs,
        )
        # Second layer normalization using `flax.nnx.LayerNorm`.
        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)

        # The MLP for point-wise feedforward (using `flax.nnx.Sequential`, `flax.nnx.Linear, flax.nnx.Dropout`)
        # with the GeLU activation function (`flax.nnx.gelu`).
        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size, mlp_dim, rngs=rngs),
            nnx.gelu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(mlp_dim, hidden_size, rngs=rngs),
            nnx.Dropout(dropout_rate, rngs=rngs),
        )

    # The forward pass through the transformer encoder block.
    def __call__(self, x: jax.Array) -> jax.Array:
        # The Multi-Head Attention layer with layer normalization.
        x = x + self.attn(self.norm1(x))
        # The feed-forward network with layer normalization.
        x = x + self.mlp(self.norm2(x))
        return x
    

# ---------------------------------------------------------------------------#
# 3.  Quick smoke test
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    rngs = nnx.Rngs(0)
    model = ViTPose_base(num_keypoints=13, img_size=(256, 192), rngs=rngs)

    # dummy = jax.random.normal(rngs.keys()[0], (2, 256, 192, 3))  # NHWC
    dummy = jnp.ones((4, 256, 192, 3))
    heatmaps = model(dummy)
    print("heatmaps:", heatmaps.shape)  # → (2, 17, 128, 96)


    # Real training 
