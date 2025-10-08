"""Weight quality diagnostics."""

import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def effective_sample_size(weights: Array) -> Array:
    """
    Compute effective sample size (ESS).

    ESS = (sum w)^2 / sum(w^2)

    Lower values indicate more extreme weights (fewer "effective" samples).
    ESS = n means uniform weights.

    Parameters
    ----------
    weights : jax.Array, shape (n_samples,)
        Sample weights

    Returns
    -------
    ess : jax.Array (scalar)
        Effective sample size
    """
    return jnp.sum(weights) ** 2 / jnp.sum(weights**2)
