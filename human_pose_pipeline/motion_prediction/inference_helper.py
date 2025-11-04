"""Helper functions for motion prediction inference."""
import jax.numpy as jnp


def compute_covariance_matrices(log_var, raw_cov):
    """Compute covariance matrices from predicted log-variances and raw covariance factors.

    Args:
        log_var: Log variances [B, T, J, 3]
        raw_cov: Raw covariance factors [B, T, J, 3]
    Returns:
        cov_matrix: Covariance matrices [B, T, J, 3, 3]
    """
    B, T, J, C = log_var.shape

    # Compute variances
    variance = jnp.exp(log_var)

    var_x, var_y, var_z = variance[..., 0], variance[..., 1], variance[..., 2]

    # Construct Cholesky factors for each frame separately
    L = jnp.zeros((B, T, J, C, C))
    eps = 0

    # Lower triangular Cholesky factor (using JAX's immutable array updates)
    L = L.at[..., 0, 0].set(jnp.sqrt(var_x + eps) * 1000)
    L = L.at[..., 1, 0].set(raw_cov[..., 0] * jnp.sqrt(var_x + eps) * 1000)
    L = L.at[..., 1, 1].set(jnp.sqrt(var_y + eps) * 1000)
    L = L.at[..., 2, 0].set(raw_cov[..., 1] * jnp.sqrt(var_x + eps) * 1000)
    L = L.at[..., 2, 1].set(raw_cov[..., 2] * jnp.sqrt(var_y + eps) * 1000)
    L = L.at[..., 2, 2].set(jnp.sqrt(var_z + eps) * 1000)

    # Compute full covariance matrix from Cholesky factors
    cov_matrix = jnp.matmul(L, jnp.matrix_transpose(L))
    return cov_matrix
