# model_flax_regressflow.py
from typing import List, Sequence, Tuple, Optional, Type, Union
import jax
import jax.numpy as jnp
from flax import linen as nn

SIGMA_MIN = 1e-2  # lower bound for per-coord scale
SIGMA_REF = 0.2  # for confidence mapping when using softplus
RHO_EPS = 1e-3  # keep correlation away from ±1

RESNET_ARCHITECTURES = {
    'resnet18': [(64, 2, 1), (128, 2, 2), (256, 2, 2), (512, 2, 2)],
    'resnet34': [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)],
    'resnet50': [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)],
    'resnet101': [(64, 3, 1), (128, 4, 2), (256, 23, 2), (512, 3, 2)],
    'resnet152': [(64, 3, 1), (128, 8, 2), (256, 36, 2), (512, 3, 2)],
}

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


class BasicBlock(nn.Module):
    """BasicBlock used for ResNet18/34 archtitectures."""
    planes: int
    stride: int = 1
    use_downsample: bool = False
    bn_momentum: float = 0.9  # 1- torch
    bn_epsilon: float = 1e-5
    expansion: int = 1

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        # Conv 1
        y = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=he_init
        )(x)
        # Batch Norm 1
        y = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(y, use_running_average=not train)
        # ReLu
        y = nn.relu(y)
        # Conv 2
        y = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=he_init,
        )(y)
        # Batch Norm 2
        y = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(y, use_running_average=not train)
        # Downsample
        if self.use_downsample:
            residual = nn.Conv(
                self.planes * self.expansion,
                (1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                kernel_init=he_init,
            )(x)
            residual = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(
                residual, use_running_average=not train
            )
        # Add residual, final ReLu
        return nn.relu(y + residual)


class Bottleneck(nn.Module):
    """Bottleneck block used for ResNet 50/101/152 architectures."""
    planes: int
    stride: int = 1
    use_downsample: bool = False
    bn_momentum: float = 0.9  # 1- torch
    bn_epsilon: float = 1e-5
    expansion: int = 4

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x

        # Conv 1
        y = nn.Conv(
            features=self.planes,
            kernel_size=(1, 1),
            use_bias=False,
            kernel_init=he_init
        )(x)
        # Batch Norm 1
        y = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(y, use_running_average=not train)
        # ReLu
        y = nn.relu(y)
        # Conv 2
        y = nn.Conv(
            features=self.planes,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=((1, 1), (1, 1)),
            use_bias=False,
            kernel_init=he_init,
        )(y)
        # Batch Norm 2
        y = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(y, use_running_average=not train)
        # ReLu
        y = nn.relu(y)
        # Conv 3
        y = nn.Conv(self.planes * self.expansion, (1, 1), use_bias=False, kernel_init=he_init)(y)
        # Batch Norm 3
        y = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(y, use_running_average=not train)
        # Downsample
        if self.use_downsample:
            residual = nn.Conv(
                self.planes * self.expansion,
                (1, 1),
                strides=(self.stride, self.stride),
                use_bias=False,
                kernel_init=he_init,
            )(x)
            residual = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(
                residual, use_running_average=not train
            )
        # Add residual, final ReLu
        return nn.relu(y + residual)


# Corresponds to make_layer call in Marian's PyTorch ResNet implementation
class BottleneckStage(nn.Module):
    blockClass: Union[Type[BasicBlock], Type[Bottleneck]]
    planes: int
    blocks: int
    stride: int
    inplanes: int = 64
    bn_momentum: float = 0.9  # 1-torch
    bn_epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, train: bool = True):
        use_downsample = self.stride != 1 or self.inplanes != self.planes * self.blockClass.expansion
        x = self.blockClass(
            planes=self.planes,
            stride=self.stride,
            use_downsample=use_downsample,
            bn_momentum=self.bn_momentum,
            bn_epsilon=self.bn_epsilon,
        )(x, train=train)
        for _ in range(1, self.blocks):
            x = self.blockClass(
                planes=self.planes,
                stride=1,
                use_downsample=False,
                bn_momentum=self.bn_momentum,
                bn_epsilon=self.bn_epsilon,
            )(x, train=train)
        return x


class ResNet50Backbone(nn.Module):
    blockClass: Union[Type[BasicBlock], Type[Bottleneck]] = Bottleneck
    architecture_str: str = "resnet50"
    bn_momentum: float = 0.9  # 1-torch
    bn_epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x, train: bool = True):
        # Conv 1
        x = nn.Conv(
            features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=((3, 3), (3, 3)),
            use_bias=False,
            kernel_init=he_init
        )(x)
        # Batch Norm 1
        x = nn.BatchNorm(momentum=self.bn_momentum, epsilon=self.bn_epsilon)(x, use_running_average=not train)
        # ReLu
        x = nn.relu(x)
        # Max Pool
        x = nn.max_pool(
            inputs=x,
            window_shape=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1))
        )

        for planes, blocks, stride in RESNET_ARCHITECTURES[self.architecture_str]:
            x = BottleneckStage(
                blockClass=self.blockClass,
                planes=planes,
                blocks=blocks,
                stride=stride,
                bn_momentum=self.bn_momentum,
                bn_epsilon=self.bn_epsilon
            )(x, train=train)
        return x  # [B, H/32, W/32, 2048]


class RegressFlowFlax(nn.Module):
    num_joints: int
    fc_filters: Sequence[int]  # e.g., [-1] (identity)
    architecture_str: str = "resnet50"
    accept_nchw: bool = True
    predict_aleatoric_uncertainty: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        RESNET_BLOCKS = {
            'resnet18': BasicBlock,
            'resnet34': BasicBlock,
            'resnet50': Bottleneck,
            'resnet101': Bottleneck,
            'resnet152': Bottleneck,
        }
        # x can be NCHW or NHWC; convert to NHWC for Flax convs
        if self.accept_nchw:
            # x: [B, C, H, W] -> [B, H, W, C]
            x = jnp.transpose(x, (0, 2, 3, 1))

        feat = ResNet50Backbone(
            blockClass=RESNET_BLOCKS[self.architecture_str],
            architecture_str=self.architecture_str
        )(x, train=train)  # [B, H/32, W/32, 2048]
        feat = global_avg_pool_2d(feat).reshape((feat.shape[0], -1))  # [B, 2048]

        h = feat
        for width in self.fc_filters:
            if width > 0:
                h = nn.Dense(width, kernel_init=nn.initializers.xavier_uniform())(h)
                h = nn.BatchNorm(momentum=0.1, epsilon=1e-5)(h, use_running_average=not train)
                h = nn.relu(h)
            else:
                # Identity: no change
                pass

        out_ch = h.shape[-1]

        # --- coordinate head (identical semantics) ---
        coord = LinearNorm(out_ch, self.num_joints * 2, use_bias=True, divide_by_input_norm=True)(h)
        if not self.predict_aleatoric_uncertainty:
            # coord = coord.reshape((coord.shape[0], self.num_joints, 2))
            return coord

        coord = coord.reshape((coord.shape[0], self.num_joints, 2))

        # --- log-variance head (Torch-compatible) ---
        # Torch: fc_sigma outputs log-variance directly
        log_variance = LinearNorm(out_ch, self.num_joints * 2, use_bias=True, divide_by_input_norm=False)(
            h
        )
        log_variance = log_variance.reshape((log_variance.shape[0], self.num_joints, 2))
        var_x = jnp.exp(log_variance[:, :, 0])
        var_y = jnp.exp(log_variance[:, :, 1])
        sigma = jnp.exp(0.5 * log_variance)  # (B,K,2)

        # --- raw covariance head (Torch-compatible) ---
        # Torch: fc_sigma2 outputs raw_cov_xy, then cov_xy = tanh(raw) * sqrt(var_x * var_y)
        raw_cov = LinearNorm(out_ch, self.num_joints, use_bias=True, divide_by_input_norm=False)(
            h
        )  # (B,K)
        cov_xy = jnp.tanh(raw_cov) * jnp.sqrt(var_x * var_y)

        # --- confidence (Torch-compatible) ---
        scores = 1.0 - jax.nn.sigmoid(log_variance)  # (B,K,2)
        scores = jnp.mean(scores, axis=2, keepdims=True).astype(jnp.float32)

        return {
            "feat": feat,  # debug
            "pred_jts": coord,
            "sigma": sigma,
            "log_variance": log_variance,
            "covariance": cov_xy,
            "maxvals": scores,
            "nf_loss": None,
            "pure_sigma": log_variance,
        }
