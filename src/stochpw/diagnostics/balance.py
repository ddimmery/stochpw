"""Balance diagnostics for treatment-covariate distributions."""

import jax.numpy as jnp
from jax import Array


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


def standardized_mean_difference_se(X: Array, A: Array, weights: Array) -> Array:
    """
    Compute standard errors for standardized mean differences.

    Uses the bootstrap-style approximation for weighted SMD standard errors.

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
    se : jax.Array, shape (n_features,)
        Standard error for each covariate's SMD
    """
    # Ensure A is 1D
    if A.ndim == 2:
        A = A.squeeze()

    # Check if A is binary
    unique_a = jnp.unique(A)
    is_binary = len(unique_a) == 2

    if is_binary:
        # Binary treatment: bootstrap-style SE
        a0, a1 = unique_a[0], unique_a[1]
        mask_0 = A == a0
        mask_1 = A == a1

        # Effective sample sizes
        weights_0 = weights * mask_0
        weights_1 = weights * mask_1
        ess_0 = jnp.sum(weights_0) ** 2 / jnp.sum(weights_0**2)
        ess_1 = jnp.sum(weights_1) ** 2 / jnp.sum(weights_1**2)

        # Approximate SE using ESS
        # SE(SMD) ≈ sqrt(1/n_0 + 1/n_1 + SMD²/(2*(n_0 + n_1)))
        # Use ESS instead of n
        smd = standardized_mean_difference(X, A, weights)
        se = jnp.sqrt(1.0 / ess_0 + 1.0 / ess_1 + smd**2 / (2.0 * (ess_0 + ess_1)))

    else:
        # Continuous treatment: approximate SE for correlation
        n_eff = jnp.sum(weights) ** 2 / jnp.sum(weights**2)
        # SE(correlation) ≈ 1/sqrt(n)
        se = jnp.ones(X.shape[1]) / jnp.sqrt(n_eff)

    return se
