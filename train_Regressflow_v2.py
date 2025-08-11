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
from src.models import RegressFlowFlax
from src.datasets.h36m import get_h36m
from src.datasets import augmented_dataloader_from_string, get_output_dim

# ------------------------------------------------------------

# --- replace your MLP with this ---
class MLP(nn.Module):
    hidden_sizes: Sequence[int]
    out_dim: int
    final_kernel_init: Any = nn.initializers.zeros
    final_bias_init: Any = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        for hs in self.hidden_sizes:
            x = nn.Dense(hs)(x)
            x = nn.leaky_relu(x)
        x = nn.Dense(self.out_dim,
                     kernel_init=self.final_kernel_init,
                     bias_init=self.final_bias_init)(x)
        return x


# --- in RealNVP.setup: zero-init last layer for both s,t ---
class RealNVP(nn.Module):
    masks: List[jnp.ndarray]
    hidden_sizes: Sequence[int] = (64, 64)
    s_scale: float = 2.0  # cap |s| ≤ s_scale (exp(2)≈7.39)

    def setup(self):
        zero = nn.initializers.zeros
        self.s_nets = [MLP(self.hidden_sizes, out_dim=2,
                           final_kernel_init=zero, final_bias_init=zero)
                       for _ in self.masks]
        self.t_nets = [MLP(self.hidden_sizes, out_dim=2,
                           final_kernel_init=zero, final_bias_init=zero)
                       for _ in self.masks]

    # z -> x
    def forward_p(self, z):
        x = z
        for i, m in enumerate(self.masks):
            m_ = jnp.asarray(m)
            x_ = x * m_
            s_raw = self.s_nets[i](x_)
            # <-- BOUND s -->
            s = jnp.tanh(s_raw) * self.s_scale
            s = s * (1.0 - m_)
            t = self.t_nets[i](x_) * (1.0 - m_)
            x = x_ + (1.0 - m_) * (x * jnp.exp(s) + t)
        return x

    # x -> z
    def backward_p(self, x):
        z = x
        log_det_J = jnp.zeros(x.shape[0])
        for i in reversed(range(len(self.masks))):
            m_ = jnp.asarray(self.masks[i])
            z_ = m_ * z
            s_raw = self.s_nets[i](z_)
            # <-- same bound here -->
            s = jnp.tanh(s_raw) * self.s_scale
            s = s * (1.0 - m_)
            t = self.t_nets[i](z_) * (1.0 - m_)
            z = (1.0 - m_) * (z - t) * jnp.exp(-s) + z_
            log_det_J = log_det_J - jnp.sum(s, axis=-1)
        return z, log_det_J
    
    @staticmethod
    def _std_normal_log_prob(z: jnp.ndarray) -> jnp.ndarray:
        """Sum of log N(0,I) over features (here D=2), returns (B,)"""
        D = z.shape[-1]
        return -0.5 * (jnp.sum(z**2, axis=-1) + D * jnp.log(2.0 * jnp.pi))

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """Log p_X(x) under the flow with standard normal base. Returns (B,)."""
        z, log_det = self.backward_p(x)           # z: (B,2), log_det: (B,)
        base_logp = self._std_normal_log_prob(z)  # (B,)
        return base_logp + log_det

    # Optional: so flow.apply(..., x) works without method=
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.log_prob(x)


# =============================
# Train states
# =============================
class ModelState(train_state.TrainState):
    batch_stats: Any  # for BN etc.

FlowState = train_state.TrainState  # flow has no batch_stats

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
# RLE loss (paper-clean)
# =============================
AMP = 1.0 / math.sqrt(2.0 * math.pi)
EPS = 1e-6

def logQ_jax(gt_uv, pred_jts, sigma):
    sigma_safe = jnp.clip(sigma, a_min=1e-2)   # <-- add safety floor
    return jnp.log(sigma_safe / AMP) + jnp.abs(gt_uv - pred_jts) / (math.sqrt(2.0) * sigma_safe)

def rle_loss_paper_jax(outputs, labels, flow_apply, flow_params,
                       detach_residual=True, size_average_batch_only=True):
    pred_jts = outputs["pred_jts"]                  # (B,K,2)
    sigma    = jnp.clip(outputs["sigma"], a_min=1e-2)

    # labels["target_uv"] may be (B, K*2); reshape to match predictions
    gt_uv = labels["target_uv"].reshape(pred_jts.shape)   # (B,K,2)

    # >>> fixed: unit weights <<<
    B, K, _ = pred_jts.shape
    w    = jnp.ones((B, K, 2), dtype=pred_jts.dtype)      # (B,K,2)
    w_nf = jnp.ones((B, K, 1), dtype=pred_jts.dtype)      # (B,K,1)

    # Laplace term (per-coord NLL), masked safely (redundant here since w==1)
    q = logQ_jax(gt_uv, pred_jts, sigma)                  # (B,K,2)
    q = jnp.where(w > 0, q, 0.0)

    # Flow term: −log Gθ( (pred−gt)/σ )
    bar_mu = (pred_jts - gt_uv) / sigma                   # (B,K,2)
    x = bar_mu.reshape(B * K, 2)
    if detach_residual:
        x = jax.lax.stop_gradient(x)

    log_g = flow_apply({'params': flow_params}, x, method=lambda m, z: m.log_prob(z))
    log_g = log_g.reshape(B, K, 1)                        # force (B,K,1)

    nf = -log_g
    nf = jnp.where(w_nf > 0, nf, 0.0)                     # keeps pattern consistent

    # Broadcast add: (B,K,1) + (B,K,2) -> (B,K,2)
    per_elem = nf + q

    loss = jnp.sum(per_elem) / B if size_average_batch_only else jnp.mean(per_elem)
    metrics = {
        "loss_rle_q": jnp.mean(jnp.sum(q, axis=-1)),
        "loss_nf": jnp.mean(nf),
        "log_g_mean": jnp.mean(log_g),
    }
    return loss, metrics



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
    masks = []
    for i in range(num_coupling_layers):
        masks.append(jnp.array([1., 0.], dtype=jnp.float32) if i % 2 == 0
                     else jnp.array([0., 1.], dtype=jnp.float32))
    flow = RealNVP(masks=masks, hidden_sizes=hidden_sizes)
    flow_params = flow.init(rng_flow, jnp.zeros((1, 2), jnp.float32))["params"]
    return flow, flow_params

# =============================
# Train / Eval steps (paper-clean)
# =============================
def loss_fn_apply(model_apply_fn,
                  model_params,
                  batch_stats,
                  flow_apply_fn,
                  flow_params,
                  x,
                  labels: Dict[str, jnp.ndarray],
                  rng,
                  train: bool,
                  detach_residual: bool,
                  size_average_batch_only: bool):
    variables = {"params": model_params, "batch_stats": batch_stats}
    if train:
        (outputs, new_state) = model_apply_fn(
            variables, x, train=True, rngs={"dropout": rng}, mutable=["batch_stats"]
        )
        new_batch_stats = new_state["batch_stats"]
    else:
        outputs = model_apply_fn(variables, x, train=False)
        new_batch_stats = batch_stats

    loss, metrics = rle_loss_paper_jax(
        outputs, labels, flow_apply_fn, flow_params,
        detach_residual=detach_residual,
        size_average_batch_only=size_average_batch_only
    )
    return loss, (outputs, new_batch_stats, metrics)

@functools.partial(
    jax.jit,
    static_argnames=("model_apply_fn","flow_apply_fn","train","detach_residual","size_average_batch_only")
)
def train_step(model_state: ModelState,
               flow_state: FlowState,
               x,
               labels,
               rng,
               *,
               model_apply_fn,
               flow_apply_fn,
               train: bool,
               detach_residual: bool,
               size_average_batch_only: bool):
    """Joint update of model and flow."""
    def joint_loss_fn(all_params, batch_stats):
        p_model = all_params["model"]
        p_flow  = all_params["flow"]
        loss, (outputs, new_bs, metrics) = loss_fn_apply(
            model_apply_fn, p_model, batch_stats,
            flow_apply_fn, p_flow,
            x, labels, rng, train,
            detach_residual, size_average_batch_only
        )
        return loss, (outputs, new_bs, metrics)

    all_params = {"model": model_state.params, "flow": flow_state.params}
    (loss, (outputs, new_batch_stats, metrics)), grads = jax.value_and_grad(
        joint_loss_fn, has_aux=True
    )(all_params, model_state.batch_stats)

    grads_model = grads["model"]
    grads_flow  = grads["flow"]

    new_model_state = model_state.apply_gradients(grads=grads_model).replace(batch_stats=new_batch_stats)
    new_flow_state  = flow_state.apply_gradients(grads=grads_flow)
    return new_model_state, new_flow_state, loss, outputs, metrics

@functools.partial(
    jax.jit,
    static_argnames=("model_apply_fn","detach_residual","size_average_batch_only")
)
def eval_step(model_state: ModelState,
              flow_state: FlowState,
              x,
              labels,
              *,
              model_apply_fn,
              detach_residual: bool,
              size_average_batch_only: bool):
    """Flow is not used in eval_step metrics except via loss; we can keep it outside jit if desired."""
    loss, (outputs, _bs, metrics) = loss_fn_apply(
        model_apply_fn,
        model_state.params, model_state.batch_stats,
        flow_state.apply_fn, flow_state.params,
        x, labels, rng=None, train=False,
        detach_residual=detach_residual,
        size_average_batch_only=size_average_batch_only
    )
    return loss, outputs, metrics

# =============================
# Helpers
# =============================
def _to_jnp(arr):
    if hasattr(arr, "numpy"):  # torch tensor
        arr = arr.numpy()
    return jnp.asarray(arr).astype(jnp.float32)

# =============================
# Main (example wiring)
# =============================
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

    # ---- model ----
    model, model_params, batch_stats = create_model(
        rng_model, num_joints, image_size, fc_filters=(1024,), accept_nchw=accept_nchw, batch_size=batch_size
    )
    model_tx, _ = create_tx(base_lr, weight_decay, warmup_steps, total_steps, clip_norm=1.0)
    model_state = ModelState.create(apply_fn=model.apply, params=model_params, tx=model_tx, batch_stats=batch_stats)

    # ---- flow ----
    flow, flow_params = create_flow(rng_flow, num_coupling_layers=4, hidden_sizes=(64, 64))
    flow_tx, _ = create_tx(base_lr, weight_decay, warmup_steps, total_steps, clip_norm=1.0)
    flow_state = FlowState.create(apply_fn=flow.apply, params=flow_params, tx=flow_tx)

    # ---- training loop ----
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_losses = []

        # -------- TRAIN --------
        for i,batch in enumerate(train_loader):
            x_b = _to_jnp(batch[0])          # (B,C,H,W) or (B,H,W,C) depending on your model
            y_b = _to_jnp(batch[1])          # (B,K,2)

            # build labels dict; add weights if your loader has them
            labels = {"target_uv": y_b}
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                labels["target_uv_weight"] = _to_jnp(batch[2])  # expected (B,K,2) or (B,K,1) broadcastable
            else:
                labels["target_uv_weight"] = jnp.ones_like(y_b)

            rng_loop, rng_step = jax.random.split(rng_loop)
            model_state, flow_state, loss, _outs, _metrics = train_step(
                model_state, flow_state, x_b, labels, rng_step,
                model_apply_fn=model.apply,
                flow_apply_fn=flow.apply,
                train=True,
                detach_residual=DETACH_RESIDUAL,
                size_average_batch_only=SIZE_AVG_BATCH_ONLY
            )
            train_losses.append(loss)
            if i%20==0:
                print(f"Step {i},  loss={loss}")

        train_loss = float(jnp.mean(jnp.asarray(train_losses)))

        # -------- EVAL --------
        val_losses = []
        val_mse = []
        for batch in valid_loader:
            x_b = _to_jnp(batch[0])
            y_b = _to_jnp(batch[1])
            labels = {"target_uv": y_b}
            if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                labels["target_uv_weight"] = _to_jnp(batch[2])
            else:
                labels["target_uv_weight"] = jnp.ones_like(y_b)

            vloss, outs, _ = eval_step(
                model_state, flow_state, x_b, labels,
                model_apply_fn=model.apply,
                detach_residual=DETACH_RESIDUAL,
                size_average_batch_only=SIZE_AVG_BATCH_ONLY
            )
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
