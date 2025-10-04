"""Weight extraction utilities for permutation weighting."""

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


def extract_weights(
    discriminator_fn: Callable,
    params: dict,
    X: Array,
    A: Array,
    AX: Array,
    eps: float = 1e-7,
) -> Array:
    """
    Extract importance weights from trained discriminator.

    Converts discriminator probabilities to density ratio weights using
    the formula: w(a, x) = (1 - η(a, x)) / η(a, x)
    where η(a, x) = p(C=1 | a, x) is the probability from the permuted distribution.

    Parameters
    ----------
    discriminator_fn : Callable
        Discriminator function (params, a, x) -> logits
    params : dict
        Trained discriminator parameters
    X : jax.Array, shape (n, d_x)
        Covariates
    A : jax.Array, shape (n, d_a)
        Treatments
    eps : float, optional
        Numerical stability constant (default: 1e-7)

    Returns
    -------
    weights : jax.Array, shape (n,)
        Importance weights

    Notes
    -----
    The discriminator outputs logits for p(C=1 | a, x), where C=1 indicates
    the permuted distribution. The weight formula w = (1 - η) / η gives the
    density ratio dP/dQ where P is the observed distribution and Q is the
    balanced (permuted) distribution. This up-weights observed pairs that
    look like they came from the observed distribution (low η).
    """
    # Get discriminator probabilities
    logits = discriminator_fn(params, A, X, AX)
    eta = jax.nn.sigmoid(logits)

    # Clip to avoid division issues
    eta_clipped = jnp.clip(eta, eps, 1 - eps)

    # Convert to density ratio: w = η / (1 - η)
    # Up-weight units that look PERMUTED (high η) to make observed distribution balanced
    weights = eta_clipped / (1 - eta_clipped)

    return weights
