#!/usr/bin/env python3
import os
import pickle
import json
from functools import partial
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.core import FrozenDict
from flax.training import train_state
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from flax_vitpose import ViTPoseFull, generate_heatmaps_from_2d, get_max_preds
from H36MPoseDataset import Human36mDataset

# --- Config ---
with open("config_train_vitpose.json", "r") as f:
    cfg = json.load(f)

IMG_SIZE      = tuple(cfg["IMG_SIZE"])    # e.g. (256,192)
NUM_KEYPOINTS = cfg["NUM_KEYPOINTS"]
BATCH_SIZE    = cfg["BATCH_SIZE"]
EPOCHS        = cfg["EPOCHS"]
LR            = cfg["LR"]
NUM_FRAMES    = cfg["NUM_FRAMES"]
PATCH         = cfg["PATCH"]
HEAD          = cfg["HEAD"]
DEPTH         = cfg["DEPTH"]
MLP_RATIO     = cfg["MLP_RATIO"]
PATIENCE      = cfg["PATIENCE"]

SAVE_PATH = "vitpose_flax_h36m.pkl"
LOG_PATH  = "training_log_flax.txt"

HEATMAP_SIZE = None  # Will be inferred on first forward pass

# --- Extend TrainState to hold batch_stats ---
class FlaxPoseTrainState(train_state.TrainState):
    batch_stats: Any

# --- Create initial state ---
def create_train_state(rng, model, learning_rate):
    dummy = jnp.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), jnp.float32)
    variables = model.init(rng, dummy, deterministic=False)
    params      = variables['params']
    batch_stats = variables['batch_stats']
    tx = optax.adam(learning_rate)
    return FlaxPoseTrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats
    )

# --- Training step with BatchNorm updates ---
@jax.jit
def train_step(state: FlaxPoseTrainState,
               frames: jnp.ndarray,
               gt_hms:  jnp.ndarray) -> Tuple[FlaxPoseTrainState, jnp.ndarray]:
    def loss_fn(params):
        vars = {'params': params, 'batch_stats': state.batch_stats}
        (preds, _), new_vars = state.apply_fn(
            vars,
            frames,
            deterministic=False,
            mutable=['batch_stats']
        )
        loss = jnp.mean((preds - gt_hms) ** 2)
        return loss, new_vars['batch_stats']

    (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)
    return state, loss

# --- Evaluation step (no stats mutation) ---
@jax.jit
def eval_step(state: FlaxPoseTrainState, frames: jnp.ndarray) -> jnp.ndarray:
    vars = {'params': state.params, 'batch_stats': state.batch_stats}
    preds, _ = state.apply_fn(vars, frames, deterministic=True)
    return preds

# --- Compute PCK on validation set ---
def evaluate(state: FlaxPoseTrainState, dataloader: DataLoader) -> Dict[float, float]:
    thresholds = [5.0, 10.0, 20.0]
    pck_totals = {t: 0 for t in thresholds}
    pck_count = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        poses = batch["pose_13"].numpy()      # (B, J, 2)
        frames = batch["frame"].numpy()       # (B, 3, H, W)
        # to NHWC
        frames = np.transpose(frames, (0,2,3,1)).astype(np.float32)
        preds_jax = eval_step(state, jnp.array(frames))
        preds = np.array(preds_jax)           # (B, J, h, w)

        # Resize preds → HEATMAP_SIZE with torch
        preds_t = torch.from_numpy(preds)
        preds_t = nn.functional.interpolate(
            preds_t, size=HEATMAP_SIZE, mode="bilinear", align_corners=False
        )
        pred_coords = get_max_preds(preds_t)
        true_coords = torch.from_numpy(poses).to(pred_coords.device)

        for pc, tc in zip(pred_coords, true_coords):
            dists = torch.norm(pc - tc, dim=1)
            for t in thresholds:
                pck_totals[t] += (dists < t).sum().item()
            pck_count += tc.shape[0]

    return {t: pck_totals[t] / pck_count for t in thresholds}


def main():
    global HEATMAP_SIZE
    rng = jax.random.PRNGKey(42)

    # --- Data transforms & loaders ---
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    train_ds = Human36mDataset(
        base_directory="/home/skyle/datasets/H36M_FREI",
        split="train",
        num_frames_per_video=NUM_FRAMES,
        transform=transform,
        image_size=IMG_SIZE
    )
    val_ds = Human36mDataset(
        base_directory="/home/skyle/datasets/H36M_FREI",
        split="validation",
        num_frames_per_video=NUM_FRAMES // 4,
        transform=transform,
        image_size=IMG_SIZE
    )

    # Use single‐process loading to avoid JAX+fork deadlocks
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model & optimizer state ---
    model = ViTPoseFull(
        img_size=IMG_SIZE,
        patch_size=PATCH,
        num_keypoints=NUM_KEYPOINTS,
        dim=cfg["EMBED_DIM"],
        depth=DEPTH,
        num_heads=HEAD,
        mlp_dim=int(cfg["EMBED_DIM"] * MLP_RATIO),
        dropout_rate=0.1
    )
    state = create_train_state(rng, model, LR)

    best_pck5 = 0.0
    epochs_no_improve = 0

    with open(LOG_PATH, "w") as log_f:
        log_f.write("Epoch\tLoss\tPCK5\tPCK10\tPCK20\n")

        for epoch in range(1, EPOCHS + 1):
            # --- TRAINING ---
            running_loss = 0.0
            for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")):
                poses = batch["pose_13"]   # (B, J, 2) torch.Tensor
                frames = batch["frame"]    # (B, 3, H, W)

                if epoch == 1 and i == 0:
                    # Infer heatmap size
                    dummy = jnp.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), jnp.float32)
                    out = state.apply_fn(
                        {"params": state.params, "batch_stats": state.batch_stats},
                        dummy,
                        deterministic=True
                    )
                    #  HEATMAP_SIZE = out.shape[2:]
                    # Flax’s nn.Conv work in NHWC, but PyTorch‐style code you’ve been 
                    # treating everything as NCHW (batch, channels, height, width)
                    # 
                    print("out:", out.shape)
                    HEATMAP_SIZE = out.shape[1:3]
                    print(f"[INFO] Inferred HEATMAP_SIZE = {HEATMAP_SIZE}")

                # Prepare inputs
                frames_np = np.transpose(frames.numpy(), (0,2,3,1)).astype(np.float32)
                gt_hm = generate_heatmaps_from_2d(poses.numpy(), HEATMAP_SIZE)

                # Train step
                state, loss = train_step(state, jnp.array(frames_np), jnp.array(gt_hm))
                running_loss += float(loss)

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch} → Loss: {avg_loss:.4f}")

            # --- EVALUATION ---
            pcks = evaluate(state, val_loader)
            print(f"PCK@5: {pcks[5.0]:.4f}, @10: {pcks[10.0]:.4f}, @20: {pcks[20.0]:.4f}")

            # Log metrics
            log_f.write(f"{epoch}\t{avg_loss:.4f}\t"
                        f"{pcks[5.0]:.4f}\t{pcks[10.0]:.4f}\t{pcks[20.0]:.4f}\n")
            log_f.flush()

            # Early stopping & checkpoint
            if pcks[5.0] > best_pck5:
                best_pck5 = pcks[5.0]
                epochs_no_improve = 0
                print(f"[INFO] New best PCK@5: {best_pck5:.4f}, saving checkpoint.")
                with open(SAVE_PATH, "wb") as fh:
                    pickle.dump(state.params, fh)
            else:
                epochs_no_improve += 1
                print(f"[INFO] No improvement ({epochs_no_improve}/{PATIENCE}).")
                if epochs_no_improve >= PATIENCE:
                    print("[EARLY STOPPING]")
                    break


if __name__ == "__main__":
    main()
