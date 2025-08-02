import os, time, json, math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax import struct

from ViTPose import ViTPose_base                      # ← your nnx model
from H36MPoseDataset import Human36mDataset      # ← unchanged

# add this import near the top
from flax.nnx import filterlib



# ---------- identical helpers you gave (PyTorch) ---------------------------
from heatmaps_utils import generate_heatmaps_from_2d, get_max_preds  # or inline
import pickle, gzip
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import json

# Load config
 # ---------- config -----------------------------------------------------
with open("config_train_vitpose.json") as f:
    cfg = json.load(f)
IMG_SIZE        = tuple(cfg["IMG_SIZE"])          # (256,192)
NUM_KEYPOINTS   = cfg["NUM_KEYPOINTS"]
BATCH_SIZE      = cfg["BATCH_SIZE"]
EPOCHS          = cfg["EPOCHS"]
LR              = cfg["LR"]
PATCH           = cfg["PATCH"]



def torch_batch_to_jax(batch_imgs):
    """(B, C, H, W) torch.Tensor → (B, H, W, C) jnp.float32"""
    return jnp.asarray(batch_imgs.permute(0, 2, 3, 1).contiguous().numpy())


def params_from_model(model):
    # Replace every nnx.Param leaf by its .value
    return jax.tree_util.tree_map(
        lambda leaf: leaf.value if isinstance(leaf, nnx.Param) else leaf,
        model,
        is_leaf=lambda x: isinstance(x, nnx.Param),
    )




# ------------------------------------------------------------------ #
# 1.  TrainState: keep both the model *and* the array-tree ‘params’
# ------------------------------------------------------------------ #
@struct.dataclass
class TrainState:
    model: ViTPose_base                       # full nnx module
    params: any                               # PyTree of arrays
    tx: optax.GradientTransformation
    opt_state: optax.OptState

    def apply_gradients(self, grads):
        # grads is an array-tree with the same structure as self.params
        updates, new_opt_state = self.tx.update(grads, self.opt_state,
                                                self.params)
        new_params = optax.apply_updates(self.params, updates)

        # nnx.apply_updates gives us an updated module in one shot
        new_model = nnx.apply_updates(self.model, updates)

        return self.replace(model=new_model,
                            params=new_params,
                            opt_state=new_opt_state)


def mse_loss(pred, gt):
    return jnp.mean((pred - gt) ** 2)

def pckh(pred_coords, gt_coords, thresh=0.5):   # simple PCKh @0.5
    dists = jnp.linalg.norm(pred_coords - gt_coords, axis=-1)  # (B,K)
    # head size = dist between head-top & upper-neck joints (0 & 1 here)
    head = jnp.linalg.norm(gt_coords[:, 0] - gt_coords[:, 1], axis=-1, keepdims=True) + 1e-6
    correct = dists < (thresh * head)
    return jnp.mean(correct)

def heatmaps_to_coords(heatmaps):
    B, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(B, K, -1)
    idx = jnp.argmax(flat, axis=-1)
    xs = (idx % W) * (IMG_SIZE[1] / W)
    ys = (idx // W) * (IMG_SIZE[0] / H)
    return jnp.stack([xs, ys], axis=-1)         # (B,K,2)


# ------------------------------------------------------------------ #
# 3.  loss/grad functions now take & return the *array* tree -------- #
# ------------------------------------------------------------------ #
@jax.jit
def train_step(state: TrainState, batch_imgs, batch_hm):
    def loss_fn(pure_params):
        # Re-inject params into the forward model
        preds = nnx.apply(state.model, pure_params)(batch_imgs)
        loss  = mse_loss(preds, batch_hm)
        return loss, preds

    (loss, preds), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, preds


@jax.jit
def eval_step(model: ViTPose_base, batch_imgs):
    return model(batch_imgs)


def main():
   

    # ---------- datasets (unchanged) --------------------------------------
    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    train_ds = Human36mDataset("/home/skyle/datasets/H36M_FREI", 'train',
                               cfg["NUM_FRAMES"], tfm, IMG_SIZE)
    val_ds   = Human36mDataset("/home/skyle/datasets/H36M_FREI", 'validation',
                               cfg["NUM_FRAMES"]//4, tfm, IMG_SIZE)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=4)

    # ------------------------------------------------------------------ #
    # 2.  build model & optimiser -------------------------------------- #
    rngs  = nnx.Rngs(0)
    model = ViTPose_base(num_keypoints=NUM_KEYPOINTS,
                        img_size=IMG_SIZE, patch_size=PATCH,
                        rngs=rngs)

    params   = params_from_model(model)        # <-- NEW
    tx       = optax.adamw(LR, weight_decay=1e-4)
    opt_state = tx.init(params)                # arrays, so this now works

    state = TrainState(model=model,
                    params=params,
                    tx=tx,
                    opt_state=opt_state)

    # ---------- training loop ---------------------------------------------
    best_val = math.inf
    for epoch in range(1, EPOCHS+1):
        # ----- train ------------------------------------------------------
        model.train = True
        train_loss = []
        for imgs_t, kp2d_t in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            # torch ➔ jax
            imgs_j = torch_batch_to_jax(imgs_t)
            hmaps  = generate_heatmaps_from_2d(kp2d_t, (IMG_SIZE[0]//2, IMG_SIZE[1]//2))
            hmaps_j = jnp.asarray(hmaps.numpy())
            state, loss, _ = train_step(state, imgs_j, hmaps_j)
            train_loss.append(loss.item())

        # ----- validation -------------------------------------------------
        model_eval = state.model
        val_loss, val_pck = [], []
        for imgs_t, kp2d_t in val_loader:
            imgs_j = torch_batch_to_jax(imgs_t)
            preds  = eval_step(model_eval, imgs_j)
            hmaps  = generate_heatmaps_from_2d(kp2d_t, (IMG_SIZE[0]//2, IMG_SIZE[1]//2))
            loss   = mse_loss(preds, jnp.asarray(hmaps.numpy()))
            coords_pred = heatmaps_to_coords(preds)
            coords_gt   = jnp.asarray(kp2d_t.numpy())
            val_loss.append(loss.item())
            val_pck.append(pckh(coords_pred, coords_gt))

        mean_tr = np.mean(train_loss)
        mean_vl = np.mean(val_loss)
        mean_pk = np.mean(val_pck)
        print(f"[{epoch}] train={mean_tr:.4f} | val={mean_vl:.4f} | PCKh@0.5={mean_pk:.3f}")

        # ----- checkpoint -------------------------------------------------
        if mean_vl < best_val:
            best_val = mean_vl
            nnx.save(state.model, f"best_vitpose_epoch{epoch}.msgpack")

            
            with gzip.open("vitpose_params.pkl.gz", "wb") as f:
                pickle.dump(state.params, f)


    print("Finished training.")


if __name__ == "__main__":
    main()
