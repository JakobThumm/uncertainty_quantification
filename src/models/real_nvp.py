# real_nvp.py
from typing import Sequence, List
import jax
import jax.numpy as jnp
from flax import linen as nn

# =============================
# RealNVP (Flax, 2D)
# =============================
class MLP(nn.Module):
    hidden_sizes: Sequence[int]
    out_dim: int  # 2 for 2D

    @nn.compact
    def __call__(self, x):
        for hs in self.hidden_sizes:
            x = nn.Dense(hs)(x)
            x = nn.leaky_relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class RealNVP(nn.Module):
    """
    RealNVP for 2D inputs with affine coupling layers.
    masks: list of (2,) float arrays with 0/1 entries, e.g.
      [jnp.array([1., 0.]), jnp.array([0., 1.]), ...]
    """
    masks: List[jnp.ndarray]
    hidden_sizes: Sequence[int] = (64, 64)

    def setup(self):
        self.s_nets = [MLP(self.hidden_sizes, out_dim=2) for _ in self.masks]
        self.t_nets = [MLP(self.hidden_sizes, out_dim=2) for _ in self.masks]

    # z -> x
    def forward_p(self, z):
        x = z
        for i, m in enumerate(self.masks):
            m_ = jnp.asarray(m)
            x_ = x * m_
            s = self.s_nets[i](x_) * (1.0 - m_)
            t = self.t_nets[i](x_) * (1.0 - m_)
            x = x_ + (1.0 - m_) * (x * jnp.exp(s) + t)
        return x

    # x -> z ; returns (z, log|det J|)
    def backward_p(self, x):
        z = x
        log_det_J = jnp.zeros(x.shape[0])
        for i in reversed(range(len(self.masks))):
            m_ = jnp.asarray(self.masks[i])
            z_ = m_ * z
            s = self.s_nets[i](z_) * (1.0 - m_)
            t = self.t_nets[i](z_) * (1.0 - m_)
            z = (1.0 - m_) * (z - t) * jnp.exp(-s) + z_
            log_det_J = log_det_J - jnp.sum(s, axis=-1)
        return z, log_det_J

    @staticmethod
    def _std_normal_log_prob(z):
        D = z.shape[-1]
        return -0.5 * (jnp.sum(z**2, axis=-1) + D * jnp.log(2.0 * jnp.pi))

    def log_prob(self, x):
        z, log_det = self.backward_p(x)
        base_logp = self._std_normal_log_prob(z)
        return base_logp + log_det

    def sample(self, rng, batch_size: int):
        z = jax.random.normal(rng, (batch_size, 2))
        return self.forward_p(z)
