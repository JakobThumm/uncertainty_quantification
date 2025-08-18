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

import numpy as np
from flax.core import freeze, unfreeze

def _to_cpu_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)

def _t2f_conv(w_torch: torch.Tensor) -> jnp.ndarray:
    w = w_torch.detach().cpu().permute(2, 3, 1, 0).contiguous().numpy()  # (kh,kw,in,out)
    return jnp.asarray(w, dtype=jnp.float32)

def _t2f_dense(w_torch: torch.Tensor) -> jnp.ndarray:
    w = w_torch.detach().cpu().t().contiguous().numpy()  # (in,out)
    return jnp.asarray(w, dtype=jnp.float32)

def _assign(dst_tree, path_list, array, desc):
    """Navigate dict path and assign; assert shape match."""
    node = dst_tree
    for k in path_list[:-1]:
        if k not in node:
            raise KeyError(f"Missing path segment '{k}' while setting {desc}")
        node = node[k]
    leaf = path_list[-1]
    if leaf not in node:
        raise KeyError(f"Missing leaf '{leaf}' while setting {desc}")
    if hasattr(node[leaf], "shape"):
        expect = tuple(node[leaf].shape)
        got = tuple(array.shape)
        if expect != got:
            raise ValueError(f"Shape mismatch for {desc}: expected {expect}, got {got}")
    node[leaf] = array

def _set_bn(back_params, back_stats, module_path, tkey_prefix, sd, desc_prefix):
    # params: scale/bias
    _assign(back_params, module_path + ["scale"],
            _to_cpu_np(sd[f"{tkey_prefix}.weight"]), f"{desc_prefix}.scale")
    _assign(back_params, module_path + ["bias"],
            _to_cpu_np(sd[f"{tkey_prefix}.bias"]), f"{desc_prefix}.bias")
    # batch_stats: mean/var
    _assign(back_stats, module_path + ["mean"],
            _to_cpu_np(sd[f"{tkey_prefix}.running_mean"]), f"{desc_prefix}.running_mean")
    _assign(back_stats, module_path + ["var"],
            _to_cpu_np(sd[f"{tkey_prefix}.running_var"]), f"{desc_prefix}.running_var")



def _map_block(p_block, s_block, sd, tprefix, has_downsample, stage_idx, block_idx):
    # conv1/bn1
    _assign(p_block, ["Conv_0", "kernel"], _t2f_conv(sd[f"{tprefix}.conv1.weight"]),
            f"stage{stage_idx}.block{block_idx}.conv1")
    _set_bn(p_block, s_block, ["BatchNorm_0"], f"{tprefix}.bn1", sd,
            f"stage{stage_idx}.block{block_idx}.bn1")
    # conv2/bn2
    _assign(p_block, ["Conv_1", "kernel"], _t2f_conv(sd[f"{tprefix}.conv2.weight"]),
            f"stage{stage_idx}.block{block_idx}.conv2")
    _set_bn(p_block, s_block, ["BatchNorm_1"], f"{tprefix}.bn2", sd,
            f"stage{stage_idx}.block{block_idx}.bn2")
    # conv3/bn3
    _assign(p_block, ["Conv_2", "kernel"], _t2f_conv(sd[f"{tprefix}.conv3.weight"]),
            f"stage{stage_idx}.block{block_idx}.conv3")
    _set_bn(p_block, s_block, ["BatchNorm_2"], f"{tprefix}.bn3", sd,
            f"stage{stage_idx}.block{block_idx}.bn3")

    if has_downsample:
        # downsample conv/bn exist only for block 0 of each stage in your Torch
        _assign(p_block, ["Conv_3", "kernel"], _t2f_conv(sd[f"{tprefix}.downsample.0.weight"]),
                f"stage{stage_idx}.block{block_idx}.downsample.conv")
        _set_bn(p_block, s_block, ["BatchNorm_3"], f"{tprefix}.downsample.1", sd,
                f"stage{stage_idx}.block{block_idx}.downsample.bn")



def _maybe_unwrap_state_dict(sd):
    # un-DDP + unwrap common wrappers
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out


def transfer_regressflow_torch_to_flax(
    sd_in: Dict[str, torch.Tensor],
    flax_variables: Dict[str, Any],
    fc_filters: Sequence[int],
):
    sd = _maybe_unwrap_state_dict(sd_in)
    params = unfreeze(flax_variables["params"])
    batch_stats = unfreeze(flax_variables.get("batch_stats", {}))

    # ----- Backbone root names (assert they exist) -----
    if "ResNet50Backbone_0" not in params:
        raise KeyError("Could not find 'ResNet50Backbone_0' in Flax params. "
                       "Print the tree and adjust the name if different.")
    if "ResNet50Backbone_0" not in batch_stats:
        raise KeyError("Could not find 'ResNet50Backbone_0' in Flax batch_stats.")

    p_back = params["ResNet50Backbone_0"]
    s_back = batch_stats["ResNet50Backbone_0"]

    # stem conv/bn
    _assign(p_back, ["Conv_0", "kernel"], _t2f_conv(sd["preact.conv1.weight"]), "stem.conv1")
    _set_bn(p_back, s_back, ["BatchNorm_0"], "preact.bn1", sd, "stem.bn1")

    # stages config
    stage_names = ["layer1", "layer2", "layer3", "layer4"]
    nblocks = [3, 4, 6, 3]

    for s_idx, (lname, nb) in enumerate(zip(stage_names, nblocks)):
        stage_key = f"BottleneckStage_{s_idx}"
        if stage_key not in p_back or stage_key not in s_back:
            raise KeyError(f"Missing {stage_key} in Flax backbone. Check printed tree.")
        p_stage = p_back[stage_key]
        s_stage = s_back[stage_key]
        for b in range(nb):
            blk_key = f"Bottleneck_{b}"
            if blk_key not in p_stage or blk_key not in s_stage:
                raise KeyError(f"Missing {stage_key}/{blk_key} in Flax tree. Check printed tree.")
            p_blk = p_stage[blk_key]
            s_blk = s_stage[blk_key]
            tprefix = f"preact.{lname}.{b}"
            has_downsample = (b == 0)
            _map_block(p_blk, s_blk, sd, tprefix, has_downsample, s_idx, b)

    # ----- FC tower (if any) -----
    # Torch fcs contains triplets [Linear, BatchNorm1d, ReLU] for each width>0, else Identity
    torch_idx = 0
    bn_mod = 0
    for i, width in enumerate(fc_filters):
        if width <= 0:
            continue
        # Dense_i
        dense_key = f"Dense_{i}"
        if dense_key not in params:
            raise KeyError(f"Missing {dense_key} in Flax params (FC tower).")
        _assign(params, [dense_key, "kernel"],
                _t2f_dense(sd[f"fcs.{torch_idx}.weight"]),
                f"fcs.{torch_idx}.weight")
        _assign(params, [dense_key, "bias"],
                _to_cpu_np(sd[f"fcs.{torch_idx}.bias"]),
                f"fcs.{torch_idx}.bias")
        torch_idx += 1  # move to BN

        bn_key = f"BatchNorm_{bn_mod}"
        if bn_key not in params or bn_key not in batch_stats:
            raise KeyError(f"Missing {bn_key} in Flax (FC BN).")
        _set_bn(params, batch_stats, [bn_key], f"fcs.{torch_idx}", sd, f"fcs.{torch_idx}")
        torch_idx += 2  # skip ReLU (index+1) -> next Linear
        bn_mod += 1

    # ----- Heads -----
    # Make sure you switched Flax heads to Torch-compatible (log_variance/raw_cov) code.
    # coord head -> LinearNorm_0
    _assign(params, ["LinearNorm_0", "kernel"],
            _to_cpu_np(sd["fc_coord.linear.weight"]), "fc_coord.weight")
    _assign(params, ["LinearNorm_0", "bias"],
            _to_cpu_np(sd["fc_coord.linear.bias"]), "fc_coord.bias")

    # log-variance head -> LinearNorm_1
    _assign(params, ["LinearNorm_1", "kernel"],
            _to_cpu_np(sd["fc_sigma.linear.weight"]), "fc_sigma.weight")
    _assign(params, ["LinearNorm_1", "bias"],
            _to_cpu_np(sd["fc_sigma.linear.bias"]), "fc_sigma.bias")

    # raw covariance head -> LinearNorm_2
    _assign(params, ["LinearNorm_2", "kernel"],
            _to_cpu_np(sd["fc_sigma2.linear.weight"]), "fc_sigma2.weight")
    _assign(params, ["LinearNorm_2", "bias"],
            _to_cpu_np(sd["fc_sigma2.linear.bias"]), "fc_sigma2.bias")

    return freeze(params), freeze(batch_stats)


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
    checkpoint_path = "/home/skyle/Desktop/uq_benchmark/models/H36M/RegressFlow/seed_420/finetuned_h36m_model.pth"
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
    state_dict = torch.load(checkpoint_path, map_location='cuda') # cuda
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

    def print_tree(tree, prefix=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                print_tree(v, f"{prefix}{k}/")
        else:
            print(prefix[:-1], ":", getattr(tree, "shape", None))

    print("=== FLAX PARAM KEYS ===")
    print_tree(unfreeze(variables["params"]))
    print("=== FLAX BATCH_STATS KEYS ===")
    if "batch_stats" in variables:
        print_tree(unfreeze(variables["batch_stats"]))
    else:
        print("(no batch_stats)")


    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    # ----------- transfer Torch → Flax -----------
    # use the already-loaded torch model (weights in model_regressflow_torch)
    sd_full = model_regressflow_torch.state_dict()

    new_params, new_batch_stats = transfer_regressflow_torch_to_flax(
        sd_in=sd_full,
        flax_variables=variables,
        fc_filters=cfg['NUM_FC_FILTERS'],
    )

    params = new_params
    batch_stats = new_batch_stats

    # --- Probe Batch Norm
    # sd = model_regressflow_torch.state_dict()

    # 1) Stem conv
    t_stem = sd_full['preact.conv1.weight'].detach().cpu().numpy()             # (64,3,7,7)
    f_stem = np.array(new_params['ResNet50Backbone_0']['Conv_0']['kernel'])# (7,7,3,64)
    print("stem conv max abs diff:",
        np.max(np.abs(np.transpose(t_stem, (2,3,1,0)) - f_stem)))

    # 2) A deep conv (layer3.5.conv2)
    t_k = sd_full['preact.layer3.5.conv2.weight'].detach().cpu().numpy()       # (256,256,3,3)
    f_k = np.array(new_params['ResNet50Backbone_0']
                ['BottleneckStage_2']['Bottleneck_5']['Conv_1']['kernel'])
    print("layer3.5.conv2 max abs diff:",
        np.max(np.abs(np.transpose(t_k, (2,3,1,0)) - f_k)))

    # 3) Coord head kernel (should match *exactly*)
    t_fc = sd_full['fc_coord.linear.weight'].detach().cpu().numpy()            # (34,2048)
    f_fc = np.array(new_params['LinearNorm_0']['kernel'])                 # (34,2048)
    print("fc_coord kernel max abs diff:", np.max(np.abs(t_fc - f_fc)))

    # --- Proble Batch Norm

    t_rm = sd_full['preact.bn1.running_mean'].detach().cpu().numpy()
    f_rm = np.array(new_batch_stats['ResNet50Backbone_0']['BatchNorm_0']['mean'])
    print("bn1 running_mean max abs diff:", np.max(np.abs(t_rm - f_rm)))

    t_rv = sd_full['preact.bn1.running_var'].detach().cpu().numpy()
    f_rv = np.array(new_batch_stats['ResNet50Backbone_0']['BatchNorm_0']['var'])
    print("bn1 running_var max abs diff:", np.max(np.abs(t_rv - f_rv)))


    # ----------- quick numeric sanity check -----------
    model_regressflow_torch.eval()
    with torch.no_grad():
        x_t = torch.randn(batch_size, 3, H, W, device=device)
        out_t = model_regressflow_torch(x_t)
        feat_t = out_t['feat'].cpu().numpy()
        tj = out_t['pred_jts'].cpu().numpy()
        tlv = out_t['log_variance'].cpu().numpy()
        tcv = out_t['covariance'].cpu().numpy()

    x_j = jnp.asarray(x_t.detach().cpu().numpy())
    # model_regressflow_flax.eval()
    out_f = model_regressflow_flax.apply(
        {'params': params, 'batch_stats': batch_stats},
        x_j, train=False, mutable=False
    )

    feat_f = np.array(out_f['feat'])
    fj = np.array(out_f['pred_jts'])
    flv = np.array(out_f['log_variance'])
    fcv = np.array(out_f['covariance'])
    print(f"[Sanity] feat max abs diff: {np.max(np.abs(feat_t - feat_f)):.6f}")
    print(f"[Sanity] pred_jts max abs diff: {np.max(np.abs(tj - fj)):.6f}")
    print(f"[Sanity] log_variance max abs diff: {np.max(np.abs(tlv - flv)):.6f}")
    print(f"[Sanity] covariance max abs diff: {np.max(np.abs(tcv - fcv)):.6f}")

    # assert False
    # ---- model ----
    # model, model_params, batch_stats = create_model(
    #     rng_model, num_joints, image_size, fc_filters=(1024,), accept_nchw=accept_nchw, batch_size=batch_size
    # )

    # CONFIG.MODEL.NUM_FC_FILTERS
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            x_tensor = batch[0].to(device)             # torch input
            x_b = _to_jnp(batch[0])                    # same batch to JAX (NCHW)
            
            y_pred_torch = model_regressflow_torch(x_tensor)['pred_jts'].cpu().numpy()

            out_f = model_regressflow_flax.apply(
                {'params': params, 'batch_stats': batch_stats},  # <-- use transferred stats
                x_b, train=False, mutable=False
            )
            y_pred_jax = np.array(out_f['pred_jts'])
            print("y_pred_jax:", y_pred_jax.shape)
            print(f"pred_jts max abs diff: {np.max(np.abs(y_pred_torch - y_pred_jax)):.6f}")
            break

    # import pickle, json, os
    # from flax.core import FrozenDict
    # from flax.serialization import to_state_dict, from_state_dict
    # save_folder = "/home/skyle/Desktop/uq_benchmark/models/RegressFlow"
    # params_dict = {'params': params, 'batch_stats':batch_stats}
    # save_name = "model"
    # model_dict = {"model": "regressflow", **params_dict}
    # pickle.dump(model_dict, open(f"{save_folder}/{save_name}_params.pickle", "wb"))
