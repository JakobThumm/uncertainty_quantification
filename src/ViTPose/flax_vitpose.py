# flax_vitpose.py
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple
import torch
import torch.nn as nn
import json

import torch
from flax import linen as nn

# Load config
with open("config_train_vitpose.json", "r") as f:
    config = json.load(f)
# Extract basic parameters
IMG_SIZE = tuple(config["IMG_SIZE"])
NUM_KEYPOINTS = config["NUM_KEYPOINTS"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
LR = config["LR"]
NUM_FRAMES = config["NUM_FRAMES"]
EMBED_DIM = config["EMBED_DIM"]
DEPTH = config["DEPTH"]
MLP_RATIO = config["MLP_RATIO"]
PATCH = config["PATCH"]
HEAD = config["HEAD"]
PATIENCE = config["PATIENCE"]
HEATMAP_SIZE = config["HEATMAP_SIZE"]
VIT_MODEL = config["VIT_MODEL"]
HEATMAP_SIZE = None


# --- Heatmap Generator ---
def generate_heatmaps_from_2d(joints, out_size, sigma=2):
    B, J, _ = joints.shape
    H, W = out_size
    heatmaps = torch.zeros((B, J, H, W), dtype=torch.float32, device=joints.device)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=joints.device),
        torch.arange(W, device=joints.device),
        indexing='ij'
    )
    for b in range(B):
        for j in range(J):
            x, y = joints[b, j]
            if x < 0 or y < 0:
                continue
            x_hm = x * (W / IMG_SIZE[1])
            y_hm = y * (H / IMG_SIZE[0])
            heatmaps[b, j] = torch.exp(-((xx - x_hm)**2 + (yy - y_hm)**2) / (2 * sigma**2))
    return heatmaps

# --- Prediction Decoding ---
def get_max_preds(heatmaps):
    B, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, K, -1)
    max_vals, idxs = torch.max(heatmaps_reshaped, dim=2)
    coords = torch.zeros((B, K, 2), dtype=torch.float32, device=heatmaps.device)
    coords[..., 0] = (idxs % W) * (IMG_SIZE[1] / W)
    coords[..., 1] = (idxs // W) * (IMG_SIZE[0] / H)
    return coords

class EncoderBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool):
        # LayerNorm + Self-Attention
        y = nn.LayerNorm(name="ln1")(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            out_features=self.dim,
            dropout_rate=self.dropout_rate,
            name="attn"
        )(y, y, deterministic=deterministic)
        x = x + y

        # MLP
        y = nn.LayerNorm(name="ln2")(x)
        y = nn.Dense(self.mlp_dim, name="mlp_fc1")(y)
        y = nn.gelu(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(self.dim, name="mlp_fc2")(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        return x + y


class ViTPoseFull(nn.Module):
    img_size: Tuple[int, int]      # e.g. (256,192)
    patch_size: int                # e.g. 16
    num_keypoints: int             # e.g. 13
    dim: int                       # embedding dim, e.g. 512
    depth: int                     # number of transformer blocks, e.g. 8
    num_heads: int                 # e.g. 8
    mlp_dim: int                   # hidden dim in MLP, e.g. dim*2
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        B = x.shape[0]
        H, W = self.img_size
        ph = pw = self.patch_size
        grid_h, grid_w = H // ph, W // pw
        num_patches = grid_h * grid_w

        # 1) Patch embedding
        x = nn.Conv(
            features=self.dim,
            kernel_size=(ph, pw),
            strides=(ph, pw),
            name="patch_proj"
        )(x)                            # (B, H/ph, W/pw, dim)
        x = x.reshape(B, num_patches, self.dim)  # (B, N, dim)

        # 2) cls token + pos embed
        cls = self.param(
            "cls_token",
            nn.initializers.normal(stddev=1.0),
            (1, 1, self.dim)
        )
        cls = jnp.tile(cls, (B, 1, 1))  # (B,1,dim)
        x = jnp.concatenate([cls, x], axis=1)  # (B, N+1, dim)

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=1.0),
            (1, num_patches+1, self.dim)
        )
        x = x + pos_embed[:, : x.shape[1], :]

        # 3) Transformer blocks
        for i in range(self.depth):
            x = EncoderBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f"encoderblock_{i}"
            )(x, deterministic=deterministic)

        # 4) Drop CLS, reshape patches → feature map
        feats = x[:, 1:, :]                          # (B, N, dim)
        feats = feats.reshape(B, grid_h, grid_w, self.dim)
        feats = jnp.transpose(feats, (0, 3, 1, 2))    # (B, dim, grid_h, grid_w)

        # 5) 1×1 conv to refine
        feats = nn.Conv(features=self.dim,
                        kernel_size=(1, 1),
                        name="reshape_conv")(feats)

        # 6) Upsampling head
        m1, m2, m3 = self.dim // 2, self.dim // 4, self.dim // 8
        for idx, (out_ch, bn_name) in enumerate([(m1, "bn1"), (m2, "bn2"), (m3, "bn3")], start=1):
            feats = nn.ConvTranspose(
                features=out_ch,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='SAME',
                name=f"up{idx}"
            )(feats)
            feats = nn.BatchNorm(use_running_average=deterministic,
                                 name=bn_name)(feats)
            feats = nn.relu(feats)

        # 7) final 1×1 → heatmaps
        out = nn.Conv(
            features=self.num_keypoints,
            kernel_size=(1, 1),
            name="final_conv"
        )(feats)  # (B, num_keypoints, H_out, W_out)

        return out
