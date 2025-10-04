"""Diagnostic utilities for assessing balance and weight quality."""

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


def standardized_mean_difference(X: Array, A: Array, weights: Array) -> Array:
    """
    Compute weighted standardized mean difference for each covariate.

    For binary treatment, computes SMD between weighted treatment groups.
    For continuous treatment, computes weighted correlation with covariates.

    Parameters
    ----------
    X : jax.Array, shape (n_samples, n_features)
        Covariates
    A : jax.Array, shape (n_samples, 1) or (n_samples,)
        Treatments
    weights : jax.Array, shape (n_samples,)
        Sample weights

    Returns
    -------
    smd : jax.Array, shape (n_features,)
        SMD or correlation for each covariate
    """
    # Ensure A is 1D for this computation
    if A.ndim == 2:
        A = A.squeeze()

    # Check if A is binary
    unique_a = jnp.unique(A)
    is_binary = len(unique_a) == 2

    if is_binary:
        # Binary treatment: compute SMD
        a0, a1 = unique_a[0], unique_a[1]
        mask_0 = A == a0
        mask_1 = A == a1

        # Weighted means
        weights_0 = weights * mask_0
        weights_1 = weights * mask_1

        sum_weights_0 = jnp.sum(weights_0)
        sum_weights_1 = jnp.sum(weights_1)

        mean_0 = jnp.average(X, axis=0, weights=weights_0)
        mean_1 = jnp.average(X, axis=0, weights=weights_1)

        # Weighted standard deviations
        var_0 = jnp.sum(weights_0[:, None] * (X - mean_0) ** 2, axis=0) / (sum_weights_0 + 1e-10)
        var_1 = jnp.sum(weights_1[:, None] * (X - mean_1) ** 2, axis=0) / (sum_weights_1 + 1e-10)

        # Pooled standard deviation
        pooled_std = jnp.sqrt((var_0 + var_1) / 2)

        # SMD
        smd = (mean_1 - mean_0) / (pooled_std + 1e-10)

    else:
        # Continuous treatment: compute weighted correlation
        # Normalize weights
        w_norm = weights / jnp.sum(weights)

        # Weighted means
        mean_A = jnp.sum(w_norm * A)
        mean_X = jnp.sum(w_norm[:, None] * X, axis=0)

        # Weighted covariance
        cov = jnp.sum(w_norm[:, None] * (A[:, None] - mean_A) * (X - mean_X), axis=0)

        # Weighted standard deviations
        std_A = jnp.sqrt(jnp.sum(w_norm * (A - mean_A) ** 2))
        std_X = jnp.sqrt(jnp.sum(w_norm[:, None] * (X - mean_X) ** 2, axis=0))

        # Correlation
        smd = cov / (std_A * std_X + 1e-10)

    return smd
