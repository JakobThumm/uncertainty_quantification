import pickle
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" # false for small
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
import argparse
import json
import datetime

from src.datasets import augmented_dataloader_from_string, get_output_dim
from src.models import model_from_string, pretrained_model_from_string
from src.training.trainer import gradient_descent
from src.training.trainer_fancy import gradient_descent_fancy

from src.models import RegressFlowFlax
from src.models import RealNVP
from src.datasets.h36m import get_h36m
import jax.numpy as jnp
import jax
parser = argparse.ArgumentParser()
# dataset hyperparams
parser.add_argument("--dataset", type=str, choices=["H36M", "Sinusoidal", "UCI", "MNIST", "FMNIST", "SVHN", "CIFAR-10", "CIFAR-100", "CelebA", "ImageNet"], default="MNIST")
parser.add_argument("--data_path", type=str, default="../datasets/", help="Root path of dataset")
parser.add_argument("--n_samples", default=None, type=int, help="Number of datapoint to use. None means all")
parser.add_argument("--uci_type", type=str, choices=["concrete", "boston", "energy", "kin8nm", "wine", "yacht"], default=None)

# model hyperparams
parser.add_argument("--model", type=str, choices=["MLP", "LeNet", "LeNet_h", "GoogleNet", "ConvNeXt", "ConvNeXt_L", "ConvNeXt_XL", "ResNet", "ResNet_NoNorm", "ResNet50", "ResNet50PreAct", "VAN_tiny", "VAN_small", "VAN_base", "VAN_large", "SWIN_tiny", "SWIN_large", "ViT_mnist"], default="MLP", help="Model architecture.")
parser.add_argument("--activation_fun", type=str, choices=["tanh", "relu"], default="tanh", help="Model activation function.")
parser.add_argument("--mlp_hidden_dim", default=20, type=int, help="Hidden dims of the MLP.")
parser.add_argument("--mlp_num_layers", default=1, type=int, help="Number of layers in the MLP.")

# training hyperparams
parser.add_argument("--seed", default=420, type=int)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32) # 128 original
parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "adamw", "rmsprop"], default="adam")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--decrease_learning_rate", action="store_true", required=False, default=False)
parser.add_argument("--weight_decay", type=float, default=None)
parser.add_argument("--momentum", type=float, default=None)
parser.add_argument("--likelihood", type=str, choices=["regression", "classification", "binary_multiclassification"], default="classification")

parser.add_argument("--default_hyperparams", action="store_true", required=False, default=False)
parser.add_argument("--fancy", action="store_true", required=False, default=False)

# extra regularizer
parser.add_argument("--regularizer", type=str, choices=["log_determinant_ggn", "log_determinant_ntk"], default=None)
parser.add_argument("--regularizer_hutch_samples", type=int, default=10)
parser.add_argument("--regularizer_prec_prior", type=float, default=1.)
parser.add_argument("--regularizer_prec_lik", type=float, default=1.)
parser.add_argument("--n_warmup_epochs", type=int, default=0)


# storage
parser.add_argument("--run_name", type=str, default=None, help="Fix the save file name. If None it's set to starting time")
parser.add_argument("--run_name_pretrained", type=str, default=None, help="Run name from which to load pretrained parameters. If None parameters are randomly initialized")
parser.add_argument("--model_save_path", type=str, default="../models", help="Root where to save models")
parser.add_argument("--test_every_n_epoch", type=int, default=20, help="Frequency of computing validation stats")

# print more stuff
parser.add_argument("--verbose", action="store_true", required=False, default=False)

# -----------------------------
# 3) Train state with batch norm
# -----------------------------
# import math, functools, time, pathlib, pickle
# from typing import Any, Dict, Tuple, Optional
# import jax
# import jax.numpy as jnp
# import optax
# import numpy as np
# from flax import linen as nn
# from flax.training import train_state, checkpoints
# from flax.core.frozen_dict import freeze, unfreeze
# import json
# import datetime

# =============================
# Train states
# =============================

import math, functools, time, os, datetime
from typing import Any, Dict, Tuple, Optional, Sequence, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from flax.core.frozen_dict import freeze, unfreeze, FrozenDict

class ModelState(train_state.TrainState):
    batch_stats: Any  # for BN etc.

FlowState = train_state.TrainState  # simple alias; flow has no batch_stats


# =============================
# Optimizer / schedule
# =============================
def create_tx(base_lr=3e-4, weight_decay=1e-4, warmup_steps=1000, total_steps=100000, clip_norm=1.0):
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(1, total_steps - warmup_steps),
        end_value=base_lr * 0.1
    )
    tx = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=sched, weight_decay=weight_decay),
    )
    return tx, sched


# =============================
# RLE-like term
# =============================
AMP = 1.0 / math.sqrt(2.0 * math.pi)
EPS = 1e-6

def logQ(gt_uv: jnp.ndarray, pred_jts: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """
    Mirror of your Torch formula:
      log(sigma / AMP) + |gt - pred| / (sqrt(2) * sigma + 1e-9)
    Shapes broadcast to (B, J, 2).
    """
    return jnp.log(sigma / AMP + EPS) + jnp.abs(gt_uv - pred_jts) / (math.sqrt(2.0) * sigma + 1e-9)


# =============================
# NF loss (uses RealNVP)
# =============================
def compute_nf_loss(
    flow_apply,                   # e.g., flow.apply
    flow_params: FrozenDict,      # flow params
    pred_jts: jnp.ndarray,        # (B, K, 2)
    target_uv: jnp.ndarray,       # (B, K, 2)
    sigma: jnp.ndarray,           # scalar or (B,K,1) or (B,K,2)
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """
    Implements: nf_loss = log(sigma) - log_phi
      where log_phi = flow.log_prob( (pred - gt) / sigma )
    Returns scalar mean loss and metrics dict.
    """
    # Residuals normalized by sigma
    bar_mu = (pred_jts - target_uv) / (sigma + EPS)  # (B, K, 2)

    B, K, _ = bar_mu.shape
    x = bar_mu.reshape(B * K, 2)

    # Log prob under flow
    log_phi = flow_apply({'params': flow_params}, x, method=RealNVP.log_prob)  # (B*K,)
    log_phi = log_phi.reshape(B, K, 1)

    # Log sigma — support scalar, (B,K,1), or (B,K,2)
    log_sigma = jnp.log(sigma + EPS)
    if log_sigma.ndim == 3 and log_sigma.shape[-1] == 2:  # (B,K,2) -> (B,K,1) by mean over coords
        log_sigma = jnp.mean(log_sigma, axis=-1, keepdims=True)
    elif log_sigma.ndim == 0:  # scalar -> (B,K,1)
        log_sigma = jnp.full((B, K, 1), log_sigma)

    nf_per = log_sigma - log_phi              # (B, K, 1)
    nf_loss = jnp.mean(nf_per)                # scalar

    metrics = {
        'nf_loss_mean': nf_loss,
        'log_phi_mean': jnp.mean(log_phi),
        'log_sigma_mean': jnp.mean(log_sigma),
    }
    return nf_loss, metrics

# TODO: check
import jax
import jax.numpy as jnp
from flax.core import FrozenDict

EPS = 1e-6

def compute_nf_loss_rle(
    flow_apply,              # e.g., flow.apply
    flow_params: FrozenDict, # flow params
    pred_jts: jnp.ndarray,   # (B, K, 2)
    target_uv: jnp.ndarray,  # (B, K, 2)
    sigma: jnp.ndarray,      # scalar or (B,K,1)/(B,K,2), broadcastable
    detach_residual: bool = True,
):
    """
    RLE paper-style residual term:
      L_flow = - E[ log G_theta( (pred - gt) / sigma ) ]

    NOTE: No extra log(sigma) here. Optionally detach residuals so the flow
          does not backprop into the regressor (gradient shortcut).
    """
    # normalized residuals
    bar_mu = (pred_jts - target_uv) / (sigma + EPS)      # (B, K, 2)
    B, K, _ = bar_mu.shape
    x = bar_mu.reshape(B * K, 2)

    if detach_residual:
        x = jax.lax.stop_gradient(x)

    log_g = flow_apply({'params': flow_params}, x, method=lambda m, z: m.log_prob(z))  # (B*K,)
    log_g = log_g.reshape(B, K, 1)

    nf_loss = -jnp.mean(log_g)  # negative log-likelihood of residual factor
    metrics = {
        "nf_logprob_mean": jnp.mean(log_g),
        "nf_loss_mean": nf_loss,
    }
    return nf_loss, metrics



# =============================
# Loss wrapper: RLE + lambda_nf * NF
# =============================
# def total_loss_with_nf(outputs: Dict[str, jnp.ndarray],
#                        gt_uv: jnp.ndarray,
#                        flow_apply,
#                        flow_params: FrozenDict,
#                        lambda_nf: float = 0.1) -> Tuple[jnp.ndarray, Dict[str, Any]]:
#     pred_jts = outputs["pred_jts"]            # (B,K,2)
#     sigma    = outputs.get("sigma", 1.0)      # scalar or (B,K,1/2)
#     gt_uv    = gt_uv.reshape(pred_jts.shape)

#     # RLE-like (sum over coords, mean over batch/joints)
#     q = logQ(gt_uv, pred_jts, sigma)          # (B,K,2)
#     rle_loss = jnp.mean(jnp.sum(q, axis=-1))  # scalar

#     # NF loss
#     nf_loss, nf_metrics = compute_nf_loss(
#         flow_apply=flow_apply,
#         flow_params=flow_params,
#         pred_jts=pred_jts,
#         target_uv=gt_uv,
#         sigma=sigma,
#     )

#     total = rle_loss + lambda_nf * nf_loss
#     metrics = {
#         "loss_total": total,
#         "loss_rle": rle_loss,
#         **nf_metrics
#     }
#     return total, metrics


def total_loss_with_nf(outputs, gt_uv, flow_apply, flow_params, lambda_nf: float = 1.0):
    pred_jts = outputs["pred_jts"]            # (B,K,2)
    sigma    = outputs.get("sigma", 1.0)      # scalar or (B,K,1/2)
    gt_uv    = gt_uv.reshape(pred_jts.shape)

    # Base Q (Laplace) NLL
    q = logQ(gt_uv, pred_jts, sigma)          # (B,K,2)  -- this is NLL per coord
    rle_q = jnp.mean(jnp.sum(q, axis=-1))     # scalar

    # Residual flow NLL (RLE-consistent, no extra log(sigma))
    nf_loss, nf_metrics = compute_nf_loss_rle(
        flow_apply=flow_apply,
        flow_params=flow_params,
        pred_jts=pred_jts,
        target_uv=gt_uv,
        sigma=sigma,
        detach_residual=True,                 # gradient shortcut per paper
    )

    total = rle_q + lambda_nf * nf_loss
    metrics = {"loss_total": total, "loss_rle_q": rle_q, **nf_metrics}
    return total, metrics


# =============================
# Create model & flow
# =============================
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


def create_flow(rng, num_coupling_layers: int = 4, hidden_sizes: Sequence[int] = (64, 64)):
    # Alternate masks for 2D
    masks = []
    for i in range(num_coupling_layers):
        if i % 2 == 0:
            masks.append(jnp.array([1., 0.], dtype=jnp.float32))
        else:
            masks.append(jnp.array([0., 1.], dtype=jnp.float32))
    flow = RealNVP(masks=masks, hidden_sizes=hidden_sizes)
    # init using log_prob method; input (N,2)
    params = flow.init(rng, jnp.zeros((1, 2), jnp.float32), method=RealNVP.log_prob)["params"]
    return flow, params


# =============================
# Train / Eval steps (JIT)
# =============================
def loss_fn_apply(model_apply_fn,
                  model_params,
                  batch_stats,
                  flow_apply_fn,
                  flow_params,
                  x,
                  y_uv,
                  rng,
                  train: bool,
                  lambda_nf: float):
    variables = {"params": model_params, "batch_stats": batch_stats}

    if train:
        (outputs, new_state) = model_apply_fn(
            variables, x, train=True, rngs={"dropout": rng}, mutable=["batch_stats"]
        )
        new_batch_stats = new_state["batch_stats"]
    else:
        outputs = model_apply_fn(variables, x, train=False)
        new_batch_stats = batch_stats

    loss, metrics = total_loss_with_nf(outputs, y_uv, flow_apply_fn, flow_params, lambda_nf)
    return loss, (outputs, new_batch_stats, metrics)


@functools.partial(jax.jit, static_argnames=("model_apply_fn", "flow_apply_fn", "train"))
def train_step(model_state: ModelState,
               flow_state: FlowState,
               x,
               y_uv,
               rng,
               lambda_nf: float,
               *,
               model_apply_fn,
               flow_apply_fn,
               train: bool = True):
    """
    Jointly update model and flow params.
    """
    def joint_loss_fn(all_params, batch_stats):
        p_model = all_params["model"]
        p_flow  = all_params["flow"]
        loss, (outputs, new_bs, metrics) = loss_fn_apply(
            model_apply_fn,
            p_model, batch_stats,
            flow_apply_fn,
            p_flow,
            x, y_uv, rng, train, lambda_nf
        )
        return loss, (outputs, new_bs, metrics)

    all_params = {"model": model_state.params, "flow": flow_state.params}
    (loss, (outputs, new_batch_stats, metrics)), grads = jax.value_and_grad(
        joint_loss_fn, has_aux=True
    )(all_params, model_state.batch_stats)

    # split grads
    grads_model = grads["model"]
    grads_flow  = grads["flow"]

    # apply
    new_model_state = model_state.apply_gradients(grads=grads_model)
    new_model_state = new_model_state.replace(batch_stats=new_batch_stats)
    new_flow_state  = flow_state.apply_gradients(grads=grads_flow)

    return new_model_state, new_flow_state, loss, outputs, metrics


@functools.partial(jax.jit, static_argnums=(5,))
def eval_step(model_state: ModelState,
              flow_state: FlowState,
              x,
              y_uv,
              lambda_nf: float,
              model_apply_fn):
    loss, (outputs, _bs, metrics) = loss_fn_apply(
        model_apply_fn,
        model_state.params, model_state.batch_stats,
        flow_state.apply_fn,
        flow_state.params,
        x, y_uv,
        rng=None,
        train=False,
        lambda_nf=lambda_nf
    )
    return loss, outputs, metrics


# =============================
# Helpers
# =============================
def _to_jnp(arr):
    # Accepts torch.Tensor or numpy; returns jnp.float32
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return jnp.asarray(arr).astype(jnp.float32)


def _maybe_nchw(x_np, accept_nchw=True):
    # If dataloader returns NHWC, convert to NCHW for your model if needed
    if accept_nchw:
        return x_np
    else:
        return jnp.transpose(x_np, (0, 2, 3, 1))

    

# =============================
# Main training loop
# =============================
if __name__ == "__main__":
    # ---- config ----
    seed = 0
    n_epochs = 20
    batch_size = 4
    steps_per_epoch = 100
    image_size = (256, 192)
    num_joints = 13
    accept_nchw = True
    base_lr = 3e-4
    weight_decay = 1e-4
    lambda_nf = 0.1                 # weight of NF loss
    total_steps = n_epochs * steps_per_epoch
    warmup_steps = max(100, int(0.05 * total_steps))

    rng = jax.random.PRNGKey(seed)
    rng, rng_model, rng_flow = jax.random.split(rng, 3)

    # ---- data ----
    train_loader, valid_loader, _ = get_h36m(
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        download=False,
    )
    print(f"Train set size {len(train_loader.dataset)}, Validation set size {len(valid_loader.dataset)}")

    # ---- model ----
    model, model_params, batch_stats = create_model(
        rng_model, num_joints, image_size, fc_filters=(1024,), accept_nchw=accept_nchw, batch_size=batch_size
    )
    model_tx, model_sched = create_tx(base_lr, weight_decay, warmup_steps, total_steps, clip_norm=1.0)
    model_state = ModelState.create(apply_fn=model.apply, params=model_params, tx=model_tx, batch_stats=batch_stats)

    # ---- flow ----
    flow, flow_params = create_flow(rng_flow, num_coupling_layers=4, hidden_sizes=(64, 64))
    flow_tx, _ = create_tx(base_lr, weight_decay, warmup_steps, total_steps, clip_norm=1.0)
    flow_state = FlowState.create(apply_fn=flow.apply, params=flow_params, tx=flow_tx)

    # Optional: dataset-specific
    # dataset = "H36M"
    # output_dim = get_output_dim(dataset)

    # ---- training ----
    best_val = float("inf")
    rng_loop = rng

    for epoch in range(1, n_epochs + 1):
        # ---------- TRAIN ----------
        t0 = time.time()
        train_losses = []
        for id,batch in enumerate(train_loader):
            x_b = _to_jnp(batch[0])   # shape (B, C, H, W) or (B, H, W, C)
            y_b = _to_jnp(batch[1])   # (B, J, 2)

            # x_b = _maybe_nchw(x_b, accept_nchw=accept_nchw)
            rng_loop, rng_step = jax.random.split(rng_loop)
            model_state, flow_state, loss, _outs, _metrics = train_step(
                model_state, flow_state, x_b, y_b, rng_step, lambda_nf,
                model_apply_fn=model.apply, flow_apply_fn=flow.apply, train=True
            )
            print(f"epoch:{epoch}, step{epoch}: loss={loss}")
            train_losses.append(loss)

        train_loss = float(jnp.mean(jnp.asarray(train_losses)))

        # ---------- EVAL ----------
        val_losses = []
        val_mse = []
        for batch in valid_loader:
            x_b = _to_jnp(batch[0])
            y_b = _to_jnp(batch[1])
            # x_b = _maybe_nchw(x_b, accept_nchw=accept_nchw)

            vloss, outs, _vmetrics = eval_step(model_state, flow_state, x_b, y_b, lambda_nf, model.apply)
            val_losses.append(vloss)

            pred = outs["pred_jts"]
            gt = y_b.reshape(pred.shape)
            val_mse.append(jnp.mean((pred - gt) ** 2))

        val_loss = float(jnp.mean(jnp.asarray(val_losses)))
        val_mse = float(jnp.mean(jnp.asarray(val_mse)))

        if val_loss < best_val:
            best_val = val_loss
            print(f"          ✅ new best (epoch {epoch})  val_loss={val_loss:.6f}  val_mse={val_mse:.6f}")

        dt = time.time() - t0
        print(f"[Epoch {epoch:03d}]  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  val_mse={val_mse:.6f}  ({dt:.1f}s)")
    assert False

    params_dict, stats_dict = gradient_descent(
                model, 
                train_loader, 
                valid_loader, 
                args_dict,
                pretrained_params_dict = None if args.run_name_pretrained is None else pretrained_params_dict
            )
    
    model_dict = {"model": args.model, **params_dict}


    ####################################
    ### save params and dictionaries ###
    # first folder is dataset
    save_folder = f"{args.model_save_path}/{args.dataset}"
    if args.n_samples is not None:
        save_folder += f"_samples{args.n_samples}"
    # second folder is model
    if args.model == "MLP":
        save_folder += f"/MLP_depth{args.mlp_num_layers}_hidden{args.mlp_hidden_dim}"
    else:
        save_folder += f"/{args.model}"
    # third folder is seed
    save_folder += f"/seed_{args.seed}"
    os.makedirs(save_folder, exist_ok=True)
    
    if args.run_name is not None:
        save_name = f"{args.run_name}"
    else:
        save_name = f"started_{now_string}"

    print(f"Saving to {save_folder}/{save_name}")
    pickle.dump(model_dict, open(f"{save_folder}/{save_name}_params.pickle", "wb"))
    pickle.dump(stats_dict, open(f"{save_folder}/{save_name}_stats.pickle", "wb"))
    with open(f"{save_folder}/{save_name}_args.json", "w") as f:
        json.dump(args_dict, f)