import jax.numpy as jnp
from jax import nn as jnn
from flax import linen as nn

# Same structure as LeNet, support extract hidden layer
class LeNet_h(nn.Module):
    output_dim: int
    act_fn : callable

    @nn.compact
    def __call__(self, x, return_hidden: bool = False):
        if len(x.shape) != 4:
            x = jnp.expand_dims(x, 0)
        x = nn.Conv(features=6, kernel_size=(5, 5), strides=(1, 1), padding=((0, 0), (0, 0)))(x)
        x = self.act_fn(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=(1, 1), padding=((0, 0), (0, 0)))(x)
        x = self.act_fn(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding=((0, 0), (0, 0)))
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=120)(x)
        x = self.act_fn(x)
        hidden = x   # <— grab this
        x = nn.Dense(features=84)(x)
        x = self.act_fn(x)
        logits = nn.Dense(features=self.output_dim)(x)

        if return_hidden:
            return hidden, logits
        else:
            return logits