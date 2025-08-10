import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

# train_regressflow_flax.py
import math, functools, time, pathlib, pickle
from typing import Any, Dict, Tuple, Optional
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.core.frozen_dict import freeze, unfreeze
import json
import datetime
# ---- import your model ----

from src.models import RegressFlowFlax
# ---------------------------
# 1) Utilities: data & loaders
# ---------------------------
def make_dummy_loader(batch_size: int,
                      num_batches: int,
                      image_size=(256, 192),
                      num_joints=17,
                      nchw=True):
    """
    Yields synthetic samples for smoke-testing the pipeline.
    X: [B, C, H, W] in [0,1], Y: [B, J, 2] pixel coords in [0,H/W)
    """
    H, W = image_size
    C = 3
    for _ in range(num_batches):
        imgs = np.random.rand(batch_size, C, H, W).astype(np.float32) if nchw \
             else np.random.rand(batch_size, H, W, C).astype(np.float32)
        # random GT joints in pixel space
        y = np.stack([
            np.random.uniform(0, W, size=(batch_size, num_joints)),
            np.random.uniform(0, H, size=(batch_size, num_joints)),
        ], axis=-1).astype(np.float32)  # [B, J, 2]
        # optional visibility mask (1.0 visible; 0.0 not used)
        vis = np.ones((batch_size, num_joints, 1), dtype=np.float32)
        yield imgs, y, vis

# ------------------------------------------------
# 2) Gaussian NLL for 2D joints (per-joint, summed)
# ------------------------------------------------
def gaussian_nll_2d(pred: Dict[str, jnp.ndarray],
                    target_xy: jnp.ndarray,
                    vis_mask: Optional[jnp.ndarray] = None,
                    eps: float = 1e-6) -> jnp.ndarray:
    """
    pred:
      - pred_jts: [B,J,2] mean (mu_x, mu_y)
      - log_variance: [B,J,2] -> log(var_x), log(var_y)
      - covariance: [B,J] -> cov_xy
    target_xy: [B,J,2]
    vis_mask:  [B,J,1] (optional, 0/1)
    returns scalar mean loss
    """
    mu = pred["pred_jts"]               # [B,J,2]
    log_var = pred["log_variance"]      # [B,J,2]
    cov_xy = pred["covariance"]         # [B,J]

    var_x = jnp.exp(log_var[..., 0]) + eps
    var_y = jnp.exp(log_var[..., 1]) + eps
    cov_xy = jnp.clip(cov_xy, -(var_x * var_y)**0.5 + 1e-6, (var_x * var_y)**0.5 - 1e-6)

    dx = target_xy[..., 0] - mu[..., 0]  # [B,J]
    dy = target_xy[..., 1] - mu[..., 1]

    det = var_x * var_y - cov_xy**2       # [B,J]
    det = jnp.clip(det, eps, 1e6)

    # Mahalanobis term
    inv_xx =  var_y / det
    inv_yy =  var_x / det
    inv_xy = -cov_xy / det

    quad = inv_xx * dx * dx + 2.0 * inv_xy * dx * dy + inv_yy * dy * dy  # [B,J]

    # log-likelihood (2D Gaussian)
    log_norm = jnp.log(det) + 2.0 * jnp.log(2.0 * math.pi)  # [B,J]
    nll = 0.5 * (log_norm + quad)                            # [B,J]

    if vis_mask is not None:
        vis = jnp.squeeze(vis_mask, axis=-1)  # [B,J]
        nll = nll * vis
        denom = jnp.maximum(jnp.sum(vis, axis=(1,)), 1.0)
        loss = jnp.sum(nll, axis=1) / denom
    else:
        loss = jnp.mean(nll, axis=1)  # per-sample

    return jnp.mean(loss)  # scalar

# -----------------------------
# 3) Train state with batch norm
# -----------------------------
class TrainState(train_state.TrainState):
    batch_stats: Any

# -----------------------------
# 4) Create model & init params
# -----------------------------
def create_model(rng, num_joints=17, image_size=(256,192), fc_filters=(1024,),
                 accept_nchw=True, batch_size=8):
    model = RegressFlowFlax(
        num_joints=num_joints,
        image_size=image_size,
        fc_filters=fc_filters,
        accept_nchw=accept_nchw
    )
    H, W = image_size
    dummy_x = jnp.zeros((batch_size, 3, H, W), jnp.float32) if accept_nchw \
            else jnp.zeros((batch_size, H, W, 3), jnp.float32)
    variables = model.init(rng, dummy_x, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})
    return model, params, batch_stats

# -----------------------------
# 5) Optimizer / schedule
# -----------------------------
def create_tx(
    base_lr=3e-4, weight_decay=1e-4, warmup_steps=1000, total_steps=100_000, clip_norm=1.0
):
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=base_lr * 0.1
    )
    tx = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=sched, weight_decay=weight_decay),
    )
    return tx, sched

# -----------------------------
# 6) Train / eval steps (jit)
# -----------------------------
def loss_fn_apply(model, params, batch_stats, x, y, vis, rng, train: bool):
    variables = {"params": params, "batch_stats": batch_stats}
    if train:
        (pred, new_state) = model.apply(
            variables, x, train=True, rngs={"dropout": rng}, mutable=["batch_stats"]
        )
        new_batch_stats = new_state["batch_stats"]
    else:
        pred = model.apply(variables, x, train=False)
        new_batch_stats = batch_stats
    loss = gaussian_nll_2d(pred, y, vis_mask=vis)
    return loss, (pred, new_batch_stats)

@functools.partial(jax.jit, static_argnames=("model",))
def train_step(state: TrainState, model, batch, rng) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
    x, y, vis = batch
    def _loss(params):
        loss, (pred, new_bs) = loss_fn_apply(model, params, state.batch_stats, x, y, vis, rng, train=True)
        return loss, (pred, new_bs)
    grads, (pred, new_batch_stats) = jax.grad(_loss, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=new_batch_stats)
    return state, {"loss": gaussian_nll_2d(pred, y, vis)}

@functools.partial(jax.jit, static_argnames=("model",))
def eval_step(state: TrainState, model, batch) -> Dict[str, jnp.ndarray]:
    x, y, vis = batch
    loss, (pred, _) = loss_fn_apply(model, state.params, state.batch_stats, x, y, vis, rng=None, train=False)
    # simple L2 metric on means
    l2 = jnp.sqrt(jnp.sum((pred["pred_jts"] - y)**2, axis=-1))  # [B,J]
    if vis is not None:
        v = jnp.squeeze(vis, -1)
        l2 = (l2 * v)
        denom = jnp.maximum(jnp.sum(v, axis=1), 1.0)
        l2 = jnp.sum(l2, axis=1) / denom
    else:
        l2 = jnp.mean(l2, axis=1)
    return {"loss": loss, "mpjpe": jnp.mean(l2)}  # MPJPE in pixels

# -----------------------------
# 7) Main training loop
# -----------------------------
def main():
    # --- config ---
    seed = 0
    workdir = pathlib.Path("./exp_dummy")
    workdir.mkdir(parents=True, exist_ok=True)

    num_epochs = 5
    steps_per_epoch = 500
    batch_size = 4 
    image_size = (256, 192)
    num_joints = 17
    accept_nchw = True

    base_lr = 3e-4
    weight_decay = 1e-4
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = max(1000, int(0.05 * total_steps))

    # --- model / state ---
    rng = jax.random.PRNGKey(seed)
    model, params, batch_stats = create_model(rng, num_joints, image_size, fc_filters=(1024,), accept_nchw=accept_nchw, batch_size=batch_size)
    tx, sched = create_tx(base_lr, weight_decay, warmup_steps, total_steps, clip_norm=1.0)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats)

    # --- loaders (replace with real COCO/MSCOCO loader) ---
    train_loader = make_dummy_loader(batch_size, steps_per_epoch, image_size, num_joints, nchw=accept_nchw)
    val_loader   = make_dummy_loader(batch_size, 50, image_size, num_joints, nchw=accept_nchw)

    best_val = float("inf")
    gstep = 0

    for epoch in range(1, num_epochs + 1):
        # TRAIN
        t0 = time.time()
        losses = []
        for step, (x_np, y_np, vis_np) in enumerate(train_loader, start=1):
            gstep += 1
            rng, steprng = jax.random.split(rng)
            x = jnp.asarray(x_np)
            y = jnp.asarray(y_np)
            vis = jnp.asarray(vis_np)
            state, metrics = train_step(state, model, (x, y, vis), steprng)
            losses.append(float(metrics["loss"]))
        dt = time.time() - t0
        print(f"[Epoch {epoch}] train loss={np.mean(losses):.4f}  ({dt:.1f}s) lr≈{sched(gstep):.2e}")

        # EVAL
        val_losses, val_mpjpe = [], []
        for x_np, y_np, vis_np in val_loader:
            metrics = eval_step(state, model, (jnp.asarray(x_np), jnp.asarray(y_np), jnp.asarray(vis_np)))
            val_losses.append(float(metrics["loss"]))
            val_mpjpe.append(float(metrics["mpjpe"]))
        vloss, vmpjpe = np.mean(val_losses), np.mean(val_mpjpe)
        print(f"          valid loss={vloss:.4f}  MPJPE(px)={vmpjpe:.2f}")

        # checkpoint best
        if vloss < best_val:
            best_val = vloss

            # ---------- build save folder ----------
            now_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_folder = f"./exp/{'dummy_dataset'}"  # replace 'dummy_dataset' with your dataset name
            save_folder += f"/{model.__class__.__name__}"
            save_folder += f"/seed_{seed}"
            os.makedirs(save_folder, exist_ok=True)

            save_name = f"epoch{epoch}_{now_string}"

            # ---------- prepare dicts ----------
            model_dict = {
                "params": state.params,
                "batch_stats": state.batch_stats,
            }
            stats_dict = {
                "train_loss": float(np.mean(losses)),
                "val_loss": vloss,
                "val_mpjpe": vmpjpe,
                "epoch": epoch,
            }
            args_dict = {
                "dataset": "dummy_dataset",
                "model": model.__class__.__name__,
                "seed": seed,
                "image_size": image_size,
                "num_joints": num_joints,
                "batch_size": batch_size,
                "base_lr": base_lr,
                "weight_decay": weight_decay,
            }

            # ---------- save pickles + json ----------
            print(f"Saving to {save_folder}/{save_name}")
            with open(f"{save_folder}/{save_name}_params.pickle", "wb") as f:
                pickle.dump(model_dict, f)
            with open(f"{save_folder}/{save_name}_stats.pickle", "wb") as f:
                pickle.dump(stats_dict, f)
            with open(f"{save_folder}/{save_name}_args.json", "w") as f:
                json.dump(args_dict, f, indent=2, sort_keys=True)

            print(f"          ✅ saved best checkpoint (epoch {epoch})")

    print("Training done. Best valid loss:", best_val)

    # # Export final params
    # with open(workdir / "final_params.pkl", "wb") as f:
    #     pickle.dump({"params": state.params, "batch_stats": state.batch_stats}, f)
    # print("Saved weights to", workdir / "final_params.pkl")

if __name__ == "__main__":
    main()
