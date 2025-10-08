"""Regularization functions for discriminator training."""

import jax
import jax.numpy as jnp
from jax import Array


def l2_param_penalty(params: dict) -> Array:
    """
    L2 regularization penalty on parameters.

    Penalizes large parameter values to encourage simpler models
    and prevent overfitting.

    Parameters
    ----------
    params : dict
        Model parameters (PyTree structure)

    Returns
    -------
    penalty : float
        L2 penalty value (sum of squared parameters)
    """
    # Flatten all parameters and compute squared L2 norm
    leaves = jax.tree_util.tree_leaves(params)
    penalty_sum = sum(jnp.sum(param**2) for param in leaves)
    # Ensure we return an Array type
    return jnp.asarray(penalty_sum)
