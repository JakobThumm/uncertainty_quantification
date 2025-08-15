# transfer_lenet_pt_to_flax.py
import math
import re
import argparse
import numpy as np

# --- PyTorch bits ---
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- JAX/Flax bits ---
import jax
import jax.numpy as jnp
from flax import linen as nn_flax
from flax.core import freeze, unfreeze
from flax.serialization import to_bytes, from_bytes

torch.set_default_dtype(torch.float64)

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)
# ============== PyTorch LeNet (same as training) ==============
class LeNetPT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Input: [B,1,32,32]
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)   # 32->28
        self.conv2 = nn.Conv2d(6,16, kernel_size=5, stride=1, padding=0)   # 14->10
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

        # same init used in the training script (optional)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):  # x: [B,1,32,32]
        x = F.relu(self.conv1(x))                   # [B,6,28,28]
        x = F.avg_pool2d(x, kernel_size=2)         # [B,6,14,14]
        x = F.relu(self.conv2(x))                   # [B,16,10,10]
        x = F.avg_pool2d(x, kernel_size=2)         # [B,16,5,5]
        x = torch.flatten(x, 1)                     # [B,400]
        x = F.relu(self.fc1(x))                     # [B,120]
        x = F.relu(self.fc2(x))                     # [B,84]
        x = self.fc3(x)                             # [B,10]
        return x

# ============== Flax LeNet (matching shapes) ==============
class LeNetFlax(nn_flax.Module):
    num_classes: int = 10
    accept_nchw: bool = True  # accept [B,1,32,32] and transpose inside

    def setup(self):
        # Attribute names mirror PyTorch layer names for easy mapping
        self.conv1 = nn_flax.Conv(features=6,  kernel_size=(5,5), padding='VALID', use_bias=True)
        self.conv2 = nn_flax.Conv(features=16, kernel_size=(5,5), padding='VALID', use_bias=True)

        self.fc1   = nn_flax.Dense(features=120, use_bias=True)
        self.fc2   = nn_flax.Dense(features=84,  use_bias=True)
        self.fc3   = nn_flax.Dense(features=self.num_classes, use_bias=True)

    def __call__(self, x, train: bool = False):
        # If input is NCHW, flip to NHWC for Flax convs
        if self.accept_nchw:
            x = jnp.transpose(x, (0, 2, 3, 1))  # [B,H,W,C]

        x = nn_flax.relu(self.conv1(x))                         # [B,28,28,6]
        x = nn_flax.avg_pool(x, window_shape=(2,2), strides=(2,2))  # [B,14,14,6]
        x = nn_flax.relu(self.conv2(x))                         # [B,10,10,16] 

        # after conv2 + pool
        x = nn_flax.avg_pool(x, window_shape=(2,2), strides=(2,2))  # NHWC, [B,5,5,16]

        # ✅ Make flatten order match PyTorch (NCHW) , 
        # in this way I match my Jax model match pytorch
        x = jnp.transpose(x, (0, 3, 1, 2))  # [B,16,5,5]
        x = x.reshape((x.shape[0], -1))     # [B,400]
        # x = nn_flax.avg_pool(x, window_shape=(2,2), strides=(2,2))  # [B,5,5,16]
        # x = x.reshape((x.shape[0], -1))                         # [B,400]
        x = nn_flax.relu(self.fc1(x))                           # [B,120]
        x = nn_flax.relu(self.fc2(x))                           # [B,84]
        x = self.fc3(x)                                         # [B,10]
        return x

# ============== Helpers for mapping weights ==============
def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()

def map_conv_weight(w_torch: torch.Tensor) -> np.ndarray:
    # Torch: [out, in, kH, kW] -> Flax: [kH, kW, in, out]
    return to_np(w_torch).transpose(2, 3, 1, 0)

def map_linear_weight(w_torch: torch.Tensor) -> np.ndarray:
    # Torch: [out, in] -> Flax: [in, out]
    return to_np(w_torch).T

def key_allowed(key: str, include_regex, exclude_regex) -> bool:
    if include_regex and re.search(include_regex, key) is None:
        return False
    if exclude_regex and re.search(exclude_regex, key) is not None:
        return False
    return True

# ============== Core transfer function ==============
def transfer_lenet_torch_to_flax(torch_sd: dict,
                                 flax_model: LeNetFlax,
                                 include: str | None = None,
                                 exclude: str | None = None,
                                 seed: int = 0):
    """
    torch_sd: PyTorch state_dict
    include/exclude: optional regex to filter torch keys for partial transfer
    returns: Flax variables dict ({"params": ..., ...})
    """
    # Init Flax variables to get correct tree & shapes
    key = jax.random.PRNGKey(seed)
    dummy = np.zeros((1, 1, 32, 32), dtype=np.float32)  # NCHW dummy (pytorch)
    vars_flax = flax_model.init(key, dummy, train=False)
    params_f = unfreeze(vars_flax["params"])

    # Map conv1
    if key_allowed("conv1.weight", include, exclude):
        assert params_f["conv1"]["kernel"].shape == (5,5,1,6)
        params_f["conv1"]["kernel"] = map_conv_weight(torch_sd["conv1.weight"])
    if key_allowed("conv1.bias", include, exclude):
        assert params_f["conv1"]["bias"].shape == (6,)
        params_f["conv1"]["bias"] = to_np(torch_sd["conv1.bias"])

    # Map conv2
    if key_allowed("conv2.weight", include, exclude):
        assert params_f["conv2"]["kernel"].shape == (5,5,6,16)
        params_f["conv2"]["kernel"] = map_conv_weight(torch_sd["conv2.weight"])
    if key_allowed("conv2.bias", include, exclude):
        assert params_f["conv2"]["bias"].shape == (16,)
        params_f["conv2"]["bias"] = to_np(torch_sd["conv2.bias"])

    # Map fc1
    if key_allowed("fc1.weight", include, exclude):
        assert params_f["fc1"]["kernel"].shape == (16*5*5, 120)
        params_f["fc1"]["kernel"] = map_linear_weight(torch_sd["fc1.weight"])
    if key_allowed("fc1.bias", include, exclude):
        assert params_f["fc1"]["bias"].shape == (120,)
        params_f["fc1"]["bias"] = to_np(torch_sd["fc1.bias"])

    # Map fc2
    if key_allowed("fc2.weight", include, exclude):
        assert params_f["fc2"]["kernel"].shape == (120, 84)
        params_f["fc2"]["kernel"] = map_linear_weight(torch_sd["fc2.weight"])
    if key_allowed("fc2.bias", include, exclude):
        assert params_f["fc2"]["bias"].shape == (84,)
        params_f["fc2"]["bias"] = to_np(torch_sd["fc2.bias"])

    # Map fc3
    if key_allowed("fc3.weight", include, exclude):
        assert params_f["fc3"]["kernel"].shape == (84, flax_model.num_classes)
        params_f["fc3"]["kernel"] = map_linear_weight(torch_sd["fc3.weight"])
    if key_allowed("fc3.bias", include, exclude):
        assert params_f["fc3"]["bias"].shape == (flax_model.num_classes,)
        params_f["fc3"]["bias"] = to_np(torch_sd["fc3.bias"])

    new_vars = {"params": freeze(params_f)}
    return new_vars

# ============== Sanity check ==============
def sanity_check(torch_model: LeNetPT, flax_model: LeNetFlax, flax_vars, tol=1e-4):
    torch_model.eval()
    with torch.no_grad():
        x_pt = torch.randn(2, 1, 32, 32)
        y_pt = torch_model(x_pt).cpu().numpy()

    x_np = x_pt.numpy().astype(np.float32)
    y_fx = flax_model.apply(flax_vars, x_np, train=False)
    y_fx = np.array(y_fx)

    max_abs = np.max(np.abs(y_pt - y_fx))
    print(f"[Sanity] Max abs diff: {max_abs:.6f}")
    if not np.allclose(y_pt, y_fx, rtol=1e-4, atol=tol):
        print("  ⚠️  Outputs differ beyond tolerance (expected with different dtypes/libs).")
        print("     If very large, double-check mappings and layer shapes.")
    else:
        print("  ✅ Torch and Flax outputs match closely.")

# ============== CLI ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--torch_ckpt", type=str, default="lenet_pt.pth",
                    help="Path to the trained PyTorch state_dict.")
    ap.add_argument("--flax_out", type=str, default="lenet_flax.msgpack",
                    help="Where to save Flax variables (msgpack bytes).")
    ap.add_argument("--include", type=str, default=None,
                    help="Regex: only keys matching will be transferred (e.g., '^conv').")
    ap.add_argument("--exclude", type=str, default=None,
                    help="Regex: keys matching will be excluded (e.g., '^fc3\\.').")
    ap.add_argument("--no_check", action="store_true",
                    help="Skip the numerical sanity check.")
    args = ap.parse_args()

    # 1) Load PyTorch model/weights
    torch_model = LeNetPT()
    sd = torch.load(args.torch_ckpt, map_location="cpu")
    torch_model = LeNetPT().double()  # ensure weights/bias are float64
    torch_model.load_state_dict(sd)
    
    # 2) Build Flax model
    flax_model = LeNetFlax(num_classes=10, accept_nchw=True)

    # 3) Transfer (optionally partial with include/exclude)
    flax_vars = transfer_lenet_torch_to_flax(sd, flax_model,
                                             include=args.include,
                                             exclude=args.exclude,
                                             seed=0)

    # 4) Optional sanity check
    if not args.no_check:
        sanity_check(torch_model, flax_model, flax_vars, tol=1e-4)

    # 5) Save Flax variables as msgpack bytes
    b = to_bytes(flax_vars)
    with open(args.flax_out, "wb") as f:
        f.write(b)
    print(f"✅ Saved Flax variables to: {args.flax_out}")

    # (Optional) Example of loading back:
    # with open(args.flax_out, "rb") as f:
    #     loaded_vars = from_bytes(flax_vars, f.read())
    # y = flax_model.apply(loaded_vars, np.zeros((1,1,32,32), np.float32))

if __name__ == "__main__":
    main()
