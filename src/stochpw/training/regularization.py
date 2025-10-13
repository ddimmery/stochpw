"""Regularization functions for permutation weighting.

This module provides regularization functions that operate on the output
weights rather than model parameters, encouraging desirable properties
of the reweighting scheme.
"""

import jax.numpy as jnp
from jax import Array


def entropy_penalty(weights: Array, eps: float = 1e-7) -> Array:
    """
    Entropy regularization on weights.

    Penalizes weights that diverge from uniform, encouraging smoother
    reweighting and better effective sample size.

    The entropy of normalized weights is computed as:
        H = -sum(p * log(p)) where p = weights / sum(weights)

    We return -H (negative entropy) as a penalty, since lower entropy
    (more peaked weights) should be penalized.

    Parameters
    ----------
    weights : Array, shape (n,)
        Importance weights
    eps : float, default=1e-7
        Small constant for numerical stability

    Returns
    -------
    penalty : Array
        Negative entropy penalty (higher = more concentrated weights)

    Notes
    -----
    - Uniform weights have maximum entropy
    - Highly concentrated weights have low entropy (high penalty)
    - Encourages effective sample size close to n
    """
    # Normalize weights to probability distribution
    p = weights / (jnp.sum(weights) + eps)
    p = jnp.clip(p, eps, 1.0)  # Avoid log(0)

    # Compute entropy: H = -sum(p * log(p))
    entropy = -jnp.sum(p * jnp.log(p))

    # Return negative entropy as penalty (we want to maximize entropy)
    return -entropy


def lp_weight_penalty(weights: Array, p: float = 2.0) -> Array:
    """
    L_p penalty on weight deviations from uniform.

    Penalizes weights that deviate from 1, encouraging more uniform weighting.
    Different values of p produce different behaviors:
    - p=1: L1 penalty (sparse, robust to outliers)
    - p=2: L2 penalty (smooth, sensitive to large deviations)

    Parameters
    ----------
    weights : Array, shape (n,)
        Importance weights
    p : float, default=2.0
        The power for the L_p norm (must be >= 1)

    Returns
    -------
    penalty : Array
        L_p penalty on weight deviations

    Notes
    -----
    Computed as sum(|weights - 1|^p), which penalizes deviation
    from uniform weights.
    """
    deviations = jnp.abs(weights - 1.0)
    return jnp.sum(deviations**p)
