# ============================================================
# Train RegressFlowFlax + RealNVP (paper-clean RLE loss, JAX/Flax)
# ============================================================

import os, time, math, functools
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" # false for small
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

from typing import Any, Dict, Tuple, Optional, Sequence, List

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
from flax.core import FrozenDict

# ---- bring your code ----
from src.models import RegressFlowFlax, RegressFlow
from src.datasets.h36m import get_h36m
from src.datasets import augmented_dataloader_from_string, get_output_dim

import torch
from easydict import EasyDict
CONFIG = EasyDict({
    'DATA_PRESET': {
        'TYPE': 'simple',
        'SIGMA': 2,
        'NUM_JOINTS': 17,
        'IMAGE_SIZE': [256, 192],  # Height, Width
        'HEATMAP_SIZE': [64, 48]
    },
    'MODEL': {
        'TYPE': 'RegressFlow',
        'NUM_LAYERS': 50,
        'NUM_FC_FILTERS': [-1],
        'HIDDEN_LIST': [-1],
        'PRETRAINED': '',
        'TRY_LOAD': ''
    },
    'TEST': {
        'FLIP_TEST': True,
        'HEATMAP2COORD': 'coord'
    },
    'LOSS': {
        'TYPE': 'RLELoss'
    }
})

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


def _to_jnp(arr):
    if hasattr(arr, "numpy"):  # torch tensor
        arr = arr.numpy()
    return jnp.asarray(arr).astype(jnp.float32)

if __name__ == "__main__":
    # ---- config ----
    seed = 0
    n_epochs = 20
    batch_size = 16
    steps_per_epoch = 300
    image_size = (256, 192)
    num_joints = 13
    accept_nchw = True
    base_lr = 3e-4
    weight_decay = 1e-4
    total_steps = n_epochs * steps_per_epoch
    warmup_steps = max(100, int(0.05 * total_steps))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # paper-clean knobs
    DETACH_RESIDUAL = True                 # gradient shortcut
    SIZE_AVG_BATCH_ONLY = True             # match repo: sum over elems, divide by B

    rng = jax.random.PRNGKey(seed)
    rng, rng_model, rng_flow, rng_loop = jax.random.split(rng, 4)

    # ---- data ----
    train_loader, valid_loader, _ = get_h36m(
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        download=False,
    )
    print(f"Train set size {len(train_loader.dataset)}, Validation set size {len(valid_loader.dataset)}")


    # ---- load pytorch model -------
    checkpoint_path = "/home/skyle/Desktop/uq_benchmark/models/RegressFlow/finetuned_h36m_model.pth"
    cfg = {
        'PRESET': CONFIG.DATA_PRESET,
        'NUM_LAYERS': CONFIG.MODEL.NUM_LAYERS,
        'NUM_FC_FILTERS': CONFIG.MODEL.NUM_FC_FILTERS,
        'HIDDEN_LIST': CONFIG.MODEL.HIDDEN_LIST,
        'PRETRAINED': CONFIG.MODEL.PRETRAINED,
        'TRY_LOAD': CONFIG.MODEL.TRY_LOAD
    }
    model_regressflow_torch = RegressFlow(
                                        PRESET=cfg['PRESET'],
                                        NUM_LAYERS=cfg['NUM_LAYERS'],
                                        NUM_FC_FILTERS=cfg['NUM_FC_FILTERS'],
                                        HIDDEN_LIST=cfg['HIDDEN_LIST'],
                                        PRETRAINED=cfg['PRETRAINED'],
                                        TRY_LOAD=cfg['TRY_LOAD']
                                    )
    state_dict = torch.load(checkpoint_path, map_location='cuda')
    model_regressflow_torch.load_state_dict(state_dict)
    model_regressflow_torch.to(device)

    # ----------- Flax model -----------
    model_regressflow_flax = RegressFlowFlax(
        preset_cfg=cfg['PRESET'],
        fc_filters=cfg['NUM_FC_FILTERS'],
        accept_nchw=accept_nchw
    )
    H, W = image_size
    dummy_x = jnp.zeros((batch_size, 3, H, W), jnp.float32) if accept_nchw \
            else jnp.zeros((batch_size, H, W, 3), jnp.float32)
    variables = model_regressflow_flax.init(rng, dummy_x, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})
    # ---- model ----
    # model, model_params, batch_stats = create_model(
    #     rng_model, num_joints, image_size, fc_filters=(1024,), accept_nchw=accept_nchw, batch_size=batch_size
    # )

    # CONFIG.MODEL.NUM_FC_FILTERS
    for i,batch in enumerate(train_loader):
        x_tensor = batch[0].to(device)
        y_tensor = batch[1].to(device)
        print("x_tensor:", x_tensor.shape)
        x_b = _to_jnp(batch[0])          # (B,C,H,W) or (B,H,W,C) depending on your model
        y_b = _to_jnp(batch[1])          # (B,K,2)
        print("x_b:", x_b.shape)
        y_pred_torch = model_regressflow_torch(x_tensor)

        batch_stats = variables.get("batch_stats", {})  # keep BN running stats
        vars_for_apply = {"params": params, "batch_stats": batch_stats}
        y_pred_jax = model_regressflow_flax.apply(
                        vars_for_apply, x_b, train=False, mutable=False
                    )

        break