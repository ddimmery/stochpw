"""Advanced balance diagnostics."""

from typing import Any

import jax.numpy as jnp
from jax import Array


def roc_curve(
    weights: Array, true_labels: Array, max_points: int = 100
) -> tuple[Array, Array, Array]:
    """
    Compute ROC curve from weights for discriminator performance.

    Given weights w(x,a), infers eta(x,a) = w(x,a) / (1 + w(x,a)) and
    computes the ROC curve for discriminating between observed and
    permuted data.

    Parameters
    ----------
    weights : jax.Array, shape (n_samples,)
        Sample weights from permutation weighting
    true_labels : jax.Array, shape (n_samples,)
        True binary labels (0=observed, 1=permuted)
    max_points : int, default=100
        Maximum number of points in the ROC curve (for computational efficiency)

    Returns
    -------
    fpr : jax.Array
        False positive rates at each threshold
    tpr : jax.Array
        True positive rates at each threshold
    thresholds : jax.Array
        Thresholds used to compute fpr and tpr

    Notes
    -----
    The ROC curve is the most important diagnostic for discriminator quality.
    A good discriminator will have high AUC (area under curve), indicating
    it can successfully distinguish between observed and permuted data.

    The discriminator probability eta is inferred from weights as:
        eta(x,a) = w(x,a) / (1 + w(x,a))

    Examples
    --------
    >>> # After fitting a weighter
    >>> weights = weighter.predict(X, A)
    >>> # Create permuted data and labels
    >>> weights_perm = weighter.predict(X, A_permuted)
    >>> all_weights = jnp.concatenate([weights, weights_perm])
    >>> labels = jnp.concatenate([jnp.zeros(len(weights)), jnp.ones(len(weights_perm))])
    >>> fpr, tpr, thresholds = roc_curve(all_weights, labels)
    >>> auc = jnp.trapezoid(tpr, fpr)
    """
    # Infer eta from weights: eta = w / (1 + w)
    eta = weights / (1.0 + weights)

    # Use linearly spaced thresholds for efficiency
    min_eta = float(jnp.min(eta))
    max_eta = float(jnp.max(eta))
    thresholds = jnp.linspace(max_eta + 1e-6, min_eta - 1e-6, max_points)

    # Compute TPR and FPR for each threshold using vectorized operations
    n_positive = float(jnp.sum(true_labels == 1))
    n_negative = float(jnp.sum(true_labels == 0))

    # Vectorized computation
    # For each threshold, count predictions >= threshold
    predictions_matrix = eta[:, jnp.newaxis] >= thresholds[jnp.newaxis, :]

    # True positives: predictions==1 AND true_labels==1
    tp = jnp.sum(predictions_matrix & (true_labels[:, jnp.newaxis] == 1), axis=0)

    # False positives: predictions==1 AND true_labels==0
    fp = jnp.sum(predictions_matrix & (true_labels[:, jnp.newaxis] == 0), axis=0)

    # Compute rates
    tpr_array = tp / n_positive if n_positive > 0 else jnp.zeros_like(tp)
    fpr_array = fp / n_negative if n_negative > 0 else jnp.zeros_like(fp)

    return fpr_array, tpr_array, thresholds


def calibration_curve(
    discriminator_probs: Array, true_labels: Array, num_bins: int = 10
) -> tuple[Array, Array, Array]:
    """
    Compute calibration curve for discriminator predictions.

    A well-calibrated discriminator should have predicted probabilities
    that match the true frequencies of the labels.

    Parameters
    ----------
    discriminator_probs : jax.Array, shape (n_samples,)
        Predicted probabilities from discriminator (values between 0 and 1)
    true_labels : jax.Array, shape (n_samples,)
        True binary labels (0 or 1)
    num_bins : int, default=10
        Number of bins to divide probability range into

    Returns
    -------
    bin_centers : jax.Array, shape (num_bins,)
        Center of each probability bin
    true_frequencies : jax.Array, shape (num_bins,)
        Actual frequency of positive class in each bin
    counts : jax.Array, shape (num_bins,)
        Number of samples in each bin

    Notes
    -----
    Perfect calibration means true_frequencies == bin_centers for all bins.
    """
    # Create bins
    bin_edges = jnp.linspace(0, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Digitize predictions into bins
    bin_indices = jnp.digitize(discriminator_probs, bin_edges[1:-1])

    # Compute true frequency and count per bin
    true_frequencies = jnp.zeros(num_bins)
    counts = jnp.zeros(num_bins)

    for i in range(num_bins):
        mask = bin_indices == i
        count = jnp.sum(mask)
        counts = counts.at[i].set(count)

        if count > 0:
            freq = jnp.sum(true_labels[mask]) / count
            true_frequencies = true_frequencies.at[i].set(freq)
        else:
            # For empty bins, use bin center as default
            true_frequencies = true_frequencies.at[i].set(bin_centers[i])

    return bin_centers, true_frequencies, counts


def weight_statistics(weights: Array) -> dict[str, float]:
    """
    Compute comprehensive statistics about weight distribution.

    Parameters
    ----------
    weights : jax.Array, shape (n_samples,)
        Sample weights

    Returns
    -------
    stats : dict
        Dictionary with weight statistics:
        - mean: Mean weight
        - std: Standard deviation of weights
        - min: Minimum weight
        - max: Maximum weight
        - cv: Coefficient of variation (std/mean)
        - entropy: Entropy of normalized weights
        - max_ratio: Ratio of max to min weight
        - n_extreme: Number of weights > 10x mean

    Notes
    -----
    Useful for diagnosing weight quality and potential issues with
    extreme or highly variable weights.
    """
    mean_w = float(jnp.mean(weights))
    std_w = float(jnp.std(weights))
    min_w = float(jnp.min(weights))
    max_w = float(jnp.max(weights))

    # Coefficient of variation
    cv = std_w / mean_w if mean_w > 0 else float("inf")

    # Entropy of normalized weights
    w_norm = weights / jnp.sum(weights)
    # Add small constant to avoid log(0)
    entropy = float(-jnp.sum(w_norm * jnp.log(w_norm + 1e-10)))

    # Maximum weight ratio
    max_ratio = max_w / min_w if min_w > 0 else float("inf")

    # Number of extreme weights (> 10x mean)
    n_extreme = int(jnp.sum(weights > 10 * mean_w))

    return {
        "mean": mean_w,
        "std": std_w,
        "min": min_w,
        "max": max_w,
        "cv": cv,
        "entropy": entropy,
        "max_ratio": max_ratio,
        "n_extreme": n_extreme,
    }


def balance_report(X: Array, A: Array, weights: Array) -> dict[str, Any]:
    """
    Generate comprehensive balance report.

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
    report : dict
        Comprehensive balance report with:
        - smd: Array of SMD per covariate
        - max_smd: Maximum absolute SMD across covariates
        - mean_smd: Mean absolute SMD across covariates
        - ess: Effective sample size
        - ess_ratio: ESS / n_samples
        - weight_stats: Dictionary of weight distribution statistics
        - n_samples: Number of samples
        - n_features: Number of features
        - treatment_type: 'binary' or 'continuous'

    Notes
    -----
    This function provides a complete overview of balance quality after
    weighting, useful for reporting and model diagnostics.
    """
    from .balance import standardized_mean_difference
    from .weights import effective_sample_size as ess_fn

    # Ensure A is 1D for type detection
    A_flat = A.squeeze() if A.ndim == 2 else A

    # Detect treatment type
    unique_a = jnp.unique(A_flat)
    is_binary = len(unique_a) == 2
    treatment_type = "binary" if is_binary else "continuous"

    # Compute SMD
    smd = standardized_mean_difference(X, A, weights)
    max_smd = float(jnp.max(jnp.abs(smd)))
    mean_smd = float(jnp.mean(jnp.abs(smd)))

    # Compute ESS
    ess = float(ess_fn(weights))
    n_samples = len(weights)
    ess_ratio = ess / n_samples

    # Weight statistics
    w_stats = weight_statistics(weights)

    return {
        "smd": smd,
        "max_smd": max_smd,
        "mean_smd": mean_smd,
        "ess": ess,
        "ess_ratio": ess_ratio,
        "weight_stats": w_stats,
        "n_samples": n_samples,
        "n_features": X.shape[1],
        "treatment_type": treatment_type,
    }
