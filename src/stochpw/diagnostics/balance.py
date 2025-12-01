"""Balance diagnostics for treatment-covariate distributions."""

import jax.numpy as jnp
from jax import Array, jit


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
    a_1d = A.squeeze() if A.ndim == 2 else A

    # Check if A is binary
    unique_a = jnp.unique(a_1d)
    is_binary = len(unique_a) == 2

    if is_binary:
        # Binary treatment: compute SMD
        a0, a1 = unique_a[0], unique_a[1]
        mask_0 = a_1d == a0
        mask_1 = a_1d == a1

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
        mean_a = jnp.sum(w_norm * a_1d)
        mean_X = jnp.sum(w_norm[:, None] * X, axis=0)

        # Weighted covariance
        cov = jnp.sum(w_norm[:, None] * (a_1d[:, None] - mean_a) * (X - mean_X), axis=0)

        # Weighted standard deviations
        std_a = jnp.sqrt(jnp.sum(w_norm * (a_1d - mean_a) ** 2))
        std_X = jnp.sqrt(jnp.sum(w_norm[:, None] * (X - mean_X) ** 2, axis=0))

        # Correlation
        smd = cov / (std_a * std_X + 1e-10)

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
    a_1d = A.squeeze() if A.ndim == 2 else A

    # Check if A is binary
    unique_a = jnp.unique(a_1d)
    is_binary = len(unique_a) == 2

    if is_binary:
        # Binary treatment: bootstrap-style SE
        a0, a1 = unique_a[0], unique_a[1]
        mask_0 = a_1d == a0
        mask_1 = a_1d == a1

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


@jit
def _rbf_kernel(X: Array, Y: Array, sigma: float) -> Array:
    """
    Compute RBF (Gaussian) kernel matrix between two sets of samples.

    Parameters
    ----------
    X : jax.Array, shape (n_samples_x, n_features)
        First set of samples
    Y : jax.Array, shape (n_samples_y, n_features)
        Second set of samples
    sigma : float
        Bandwidth parameter for RBF kernel

    Returns
    -------
    K : jax.Array, shape (n_samples_x, n_samples_y)
        Kernel matrix
    """
    # Compute pairwise squared Euclidean distances
    X_sq = jnp.sum(X**2, axis=1, keepdims=True)
    Y_sq = jnp.sum(Y**2, axis=1, keepdims=True)
    distances_sq = X_sq + Y_sq.T - 2 * jnp.dot(X, Y.T)
    # Apply RBF kernel
    gamma = 1.0 / (2 * sigma**2)
    return jnp.exp(-gamma * distances_sq)


def maximum_mean_discrepancy(
    X: Array, A: Array, weights: Array, sigma: float | None = None
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between weighted treatment groups.

    MMD measures distributional distance between groups using a kernel-based approach.
    Unlike SMD which compares means feature-by-feature, MMD captures higher-order
    moments and interactions between features.

    For binary treatment, computes MMD between weighted treatment and control groups.
    For continuous treatment, this function is not applicable and returns NaN.

    Parameters
    ----------
    X : jax.Array, shape (n_samples, n_features)
        Covariates
    A : jax.Array, shape (n_samples, 1) or (n_samples,)
        Treatments (must be binary)
    weights : jax.Array, shape (n_samples,)
        Sample weights
    sigma : float, optional
        Bandwidth parameter for RBF kernel. If None, uses median heuristic:
        sigma = median(pairwise distances) / sqrt(2)

    Returns
    -------
    mmd : float
        MMD statistic (non-negative, 0 means identical distributions)

    Notes
    -----
    The MMD is defined as:

    .. math::
        MMD^2 = E[k(X_1, X_1')] - 2E[k(X_1, X_0)] + E[k(X_0, X_0')]

    where X_1, X_1' are from the treated group and X_0, X_0' are from control.
    This implementation uses weighted expectations based on the provided weights.

    References
    ----------
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
    A kernel two-sample test. Journal of Machine Learning Research, 13(1), 723-773.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from stochpw import maximum_mean_discrepancy
    >>> X = jnp.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    >>> A = jnp.array([0, 0, 1, 1])
    >>> weights = jnp.ones(4)
    >>> mmd = maximum_mean_discrepancy(X, A, weights)
    """
    # Ensure A is 1D
    a_1d = A.squeeze() if A.ndim == 2 else A

    # Check if A is binary
    unique_a = jnp.unique(a_1d)
    is_binary = len(unique_a) == 2

    if not is_binary:
        # MMD only applicable for binary treatment
        return float("nan")

    # Split into treatment groups
    a0, a1 = unique_a[0], unique_a[1]
    mask_0 = a_1d == a0
    mask_1 = a_1d == a1

    X_0 = X[mask_0]
    X_1 = X[mask_1]
    weights_0 = weights[mask_0]
    weights_1 = weights[mask_1]

    # Normalize weights within each group
    weights_0_norm = weights_0 / jnp.sum(weights_0)
    weights_1_norm = weights_1 / jnp.sum(weights_1)

    # Compute bandwidth using median heuristic if not provided
    sigma_val: float
    if sigma is None:
        # Sample a subset for efficiency if dataset is large
        n_samples = min(1000, X.shape[0])
        if X.shape[0] > n_samples:
            indices = jnp.linspace(0, X.shape[0] - 1, n_samples, dtype=jnp.int32)
            X_sample = X[indices]
        else:
            X_sample = X

        # Compute pairwise distances
        dists_sq = jnp.sum(X_sample**2, axis=1, keepdims=True)
        pairwise_dists_sq = dists_sq + dists_sq.T - 2 * jnp.dot(X_sample, X_sample.T)
        pairwise_dists = jnp.sqrt(jnp.maximum(pairwise_dists_sq, 0))

        # Median heuristic
        positive_dists = pairwise_dists[pairwise_dists > 0]
        if positive_dists.size > 0:
            median_dist = float(jnp.median(positive_dists))
            sigma_val = float(median_dist / jnp.sqrt(2.0))
        else:
            # All distances are zero (constant features) - use default
            sigma_val = 1.0

        # Avoid zero or very small sigma
        sigma_val = float(jnp.maximum(sigma_val, 0.1))
    else:
        sigma_val = sigma

    # Compute kernel matrices
    K_00 = _rbf_kernel(X_0, X_0, sigma_val)
    K_11 = _rbf_kernel(X_1, X_1, sigma_val)
    K_01 = _rbf_kernel(X_0, X_1, sigma_val)

    # Compute weighted kernel expectations
    # E[k(X_0, X_0')] = sum_i sum_j w_i w_j k(x_i, x_j)
    term_00 = jnp.sum(weights_0_norm[:, None] * weights_0_norm[None, :] * K_00)
    term_11 = jnp.sum(weights_1_norm[:, None] * weights_1_norm[None, :] * K_11)
    term_01 = jnp.sum(weights_0_norm[:, None] * weights_1_norm[None, :] * K_01)

    # MMD^2 = E[k(X_1,X_1')] - 2*E[k(X_0,X_1)] + E[k(X_0,X_0')]
    mmd_sq = term_11 - 2 * term_01 + term_00

    # Return MMD (take square root, ensure non-negative)
    mmd = jnp.sqrt(jnp.maximum(mmd_sq, 0.0))

    return float(mmd)
