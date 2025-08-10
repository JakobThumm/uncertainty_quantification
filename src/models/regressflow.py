# model_flax_regressflow.py
from typing import Sequence, Tuple, Optional
import jax
import jax.numpy as jnp
from flax import linen as nn

he_init = nn.initializers.variance_scaling(2.0, mode="fan_out", distribution="truncated_normal")

def global_avg_pool_2d(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(x, axis=(1, 2), keepdims=True)

class LinearNorm(nn.Module):
    in_features: int
    out_features: int
    use_bias: bool = True
    divide_by_input_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        w = self.param("kernel", nn.initializers.xavier_uniform(), (self.out_features, self.in_features))
        y = x @ w.T
        if self.divide_by_input_norm:
            denom = jnp.linalg.norm(x, axis=1, keepdims=True) + 1e-8
            y = y / denom
        if self.use_bias:
            b = self.param("bias", nn.initializers.zeros, (self.out_features,))
            y = y + b
        return y

class Bottleneck(nn.Module):
    inplanes: int
    planes: int
    stride: int = 1
    use_downsample: bool = False
    bn_momentum: float = 0.1
    bn_epsilon: float = 1e-5
    expansion: int = 4

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        y = nn.Conv(self.planes, (1, 1), use_bias=False, kernel_init=he_init)(x)
        y = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(y, use_running_average=not train)
        y = nn.relu(y)

        y = nn.Conv(self.planes, (3, 3), strides=(self.stride, self.stride),
                    padding=((1,1),(1,1)), use_bias=False, kernel_init=he_init)(y)
        y = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(y, use_running_average=not train)
        y = nn.relu(y)

        y = nn.Conv(self.planes * self.expansion, (1, 1), use_bias=False, kernel_init=he_init)(y)
        y = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(y, use_running_average=not train)

        if self.use_downsample:
            residual = nn.Conv(self.planes * self.expansion, (1, 1),
                               strides=(self.stride, self.stride),
                               use_bias=False, kernel_init=he_init)(x)
            residual = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(
                residual, use_running_average=not train
            )
        return nn.relu(y + residual)

class BottleneckStage(nn.Module):
    inplanes: int
    planes: int
    blocks: int
    stride: int
    bn_momentum: float = 0.1
    bn_epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = Bottleneck(self.inplanes, self.planes, stride=self.stride, use_downsample=True,
                       bn_momentum=self.bn_momentum, bn_epsilon=self.bn_epsilon)(x, train=train)
        inplanes = self.planes * Bottleneck.expansion
        for _ in range(1, self.blocks):
            x = Bottleneck(inplanes, self.planes, stride=1, use_downsample=False,
                           bn_momentum=self.bn_momentum, bn_epsilon=self.bn_epsilon)(x, train=train)
        return x, inplanes

class ResNet50Backbone(nn.Module):
    bn_momentum: float = 0.1
    bn_epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding=((3,3),(3,3)), use_bias=False, kernel_init=he_init)(x)
        x = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(x, use_running_average=not train)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        inplanes = 64
        for planes, blocks, stride in [(64,3,1), (128,4,2), (256,6,2), (512,3,2)]:
            x, inplanes = BottleneckStage(inplanes, planes, blocks, stride,
                                          bn_momentum=self.bn_momentum,
                                          bn_epsilon=self.bn_epsilon)(x, train=train)
        return x  # [B, H/32, W/32, 2048]

class RegressFlowFlax(nn.Module):
    num_joints: int
    image_size: Tuple[int, int]             # (H, W), not used directly here
    fc_filters: Sequence[int]               # e.g., [-1] (identity)
    accept_nchw: bool = True               # keep your PyTorch input layout

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x can be NCHW or NHWC; convert to NHWC for Flax convs
        if self.accept_nchw:
            # x: [B, C, H, W] -> [B, H, W, C]
            x = jnp.transpose(x, (0, 2, 3, 1))

        feat = ResNet50Backbone()(x, train=train)              # [B, H/32, W/32, 2048]
        feat = global_avg_pool_2d(feat).reshape((feat.shape[0], -1))  # [B, 2048]

        h = feat
        for width in self.fc_filters:
            if width > 0:
                h = nn.Dense(width, kernel_init=nn.initializers.xavier_uniform())(h)
                h = nn.BatchNorm(momentum=0.1, epsilon=1e-5)(h, use_running_average=not train)
                h = nn.relu(h)

        out_ch = h.shape[-1]

        coord = LinearNorm(out_ch, self.num_joints * 2, use_bias=True, divide_by_input_norm=True)(h)
        coord = coord.reshape((coord.shape[0], self.num_joints, 2))

        log_variance = LinearNorm(out_ch, self.num_joints * 2, use_bias=True, divide_by_input_norm=False)(h)
        log_variance = log_variance.reshape((log_variance.shape[0], self.num_joints, 2))

        raw_cov_xy = LinearNorm(out_ch, self.num_joints, use_bias=True, divide_by_input_norm=False)(h)

        log_var_x = log_variance[:, :, 0]
        log_var_y = log_variance[:, :, 1]
        var_x = jnp.exp(log_var_x)
        var_y = jnp.exp(log_var_y)
        cov_xy = jnp.tanh(raw_cov_xy) * jnp.sqrt(var_x * var_y)

        sigma = jnp.exp(0.5 * log_variance)
        scores = 1.0 - jax.nn.sigmoid(log_variance)
        scores = jnp.mean(scores, axis=2, keepdims=True)

        return {
            "pred_jts": coord,                # [B, 17, 2]
            "sigma": sigma,                   # [B, 17, 2]
            "log_variance": log_variance,     # [B, 17, 2]
            "covariance": cov_xy,             # [B, 17]
            "maxvals": scores.astype(jnp.float32),
            "nf_loss": None,
            "pure_sigma": log_variance,
        }
