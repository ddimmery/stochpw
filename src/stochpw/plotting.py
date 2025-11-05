"""Visualization utilities for permutation weighting diagnostics."""

from typing import Any

import jax.numpy as jnp
import pandas as pd
from jax import Array
from plotnine import (
    aes,
    element_text,
    geom_bar,
    geom_histogram,
    geom_hline,
    geom_line,
    geom_point,
    geom_vline,
    ggplot,
    labs,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_minimal,
)

from .diagnostics import (
    effective_sample_size,
    standardized_mean_difference,
    standardized_mean_difference_se,
)


def plot_balance_diagnostics(
    X: Array,
    A: Array,
    weights: Array,
    unweighted_weights: Array | None = None,
    feature_names: list[str] | None = None,
):
    """
    Create balance diagnostic plot with standard errors.

    Parameters
    ----------
    X : jax.Array, shape (n_samples, n_features)
        Covariates
    A : jax.Array, shape (n_samples, 1) or (n_samples,)
        Treatments
    weights : jax.Array, shape (n_samples,)
        Sample weights after permutation weighting
    unweighted_weights : jax.Array, optional
        Uniform weights for comparison. If None, uses uniform weights.
    feature_names : list of str, optional
        Names of features for plot labels

    Returns
    -------
    plot : plotnine.ggplot
        SMD comparison plot with error bars

    Examples
    --------
    >>> from stochpw import PermutationWeighter
    >>> from stochpw.plotting import plot_balance_diagnostics
    >>> weighter = PermutationWeighter()
    >>> weighter.fit(X, A)
    >>> weights = weighter.predict(X, A)
    >>> plot = plot_balance_diagnostics(X, A, weights)
    >>> print(plot)
    """
    if unweighted_weights is None:
        unweighted_weights = jnp.ones(len(weights))

    if feature_names is None:
        feature_names = [f"X{i}" for i in range(X.shape[1])]

    # Compute SMD and SE for both weighted and unweighted
    smd_unweighted = standardized_mean_difference(X, A, unweighted_weights)
    smd_weighted = standardized_mean_difference(X, A, weights)
    se_unweighted = standardized_mean_difference_se(X, A, unweighted_weights)
    se_weighted = standardized_mean_difference_se(X, A, weights)

    # Create dataframe with SE
    n_features = len(feature_names)
    smd_df = pd.DataFrame(
        {
            "feature": feature_names * 2,
            "smd": jnp.concatenate([jnp.abs(smd_unweighted), jnp.abs(smd_weighted)]),
            "se": jnp.concatenate([se_unweighted, se_weighted]),
            "type": ["Unweighted"] * n_features + ["Weighted"] * n_features,
        }
    )

    # Add position for dodging
    smd_df["position"] = smd_df.groupby("feature").cumcount()

    from plotnine import geom_errorbar, position_dodge

    p = (
        ggplot(smd_df, aes(x="feature", y="smd", fill="type"))
        + geom_bar(stat="identity", position=position_dodge(width=0.9), alpha=0.7)
        + geom_errorbar(
            aes(ymin="smd - 1.96 * se", ymax="smd + 1.96 * se"),
            position=position_dodge(width=0.9),
            width=0.25,
        )
        + geom_hline(yintercept=0.1, linetype="dashed", color="red", alpha=0.5)
        + labs(
            title="Balance Improvement (with 95% CI)",
            x="Feature",
            y="Absolute Standardized Mean Difference",
            fill="",
        )
        + theme_minimal()
        + theme(axis_text_x=element_text(rotation=45, hjust=1))
    )

    return p


def plot_weight_distribution(weights: Array):
    """
    Visualize weight distribution.

    Parameters
    ----------
    weights : jax.Array, shape (n_samples,)
        Sample weights

    Returns
    -------
    plot : plotnine.ggplot
        Weight distribution histogram with ESS info
    """
    ess = effective_sample_size(weights)
    n = len(weights)

    weight_df = pd.DataFrame({"weight": jnp.array(weights)})
    mean_weight = float(jnp.mean(weights))

    p = (
        ggplot(weight_df, aes(x="weight"))
        + geom_histogram(bins=50, fill="steelblue", alpha=0.7, color="black")
        + geom_vline(xintercept=mean_weight, linetype="dashed", color="red")
        + labs(
            title=f"Weight Distribution (ESS: {ess:.0f}/{n}, Ratio: {ess / n:.2%})",
            x="Weight",
            y="Frequency",
        )
        + theme_minimal()
    )

    return p


def plot_training_history(history: dict[str, Any]):
    """
    Plot training loss and other metrics over epochs.

    Parameters
    ----------
    history : dict
        Training history dictionary with 'loss' key and optional other metrics

    Returns
    -------
    plot : plotnine.ggplot
        Training history plot
    """
    epochs = jnp.arange(1, len(history["loss"]) + 1)
    hist_df = pd.DataFrame({"epoch": epochs, "loss": jnp.array(history["loss"])})

    p = (
        ggplot(hist_df, aes(x="epoch", y="loss"))
        + geom_line(color="blue", size=1)
        + labs(title="Training History", x="Epoch", y="Loss")
        + theme_minimal()
    )

    return p


def plot_calibration_curve(
    bin_centers: Array,
    true_frequencies: Array,
    counts: Array,
):
    """
    Plot calibration curve for discriminator predictions.

    Parameters
    ----------
    bin_centers : jax.Array, shape (num_bins,)
        Center of each probability bin
    true_frequencies : jax.Array, shape (num_bins,)
        Actual frequency of positive class in each bin
    counts : jax.Array, shape (num_bins,)
        Number of samples in each bin

    Returns
    -------
    plot : plotnine.ggplot
        Calibration curve plot

    Notes
    -----
    A well-calibrated discriminator will have points close to the diagonal line.
    """
    # Only use bins with samples
    mask = counts > 0
    cal_df = pd.DataFrame(
        {
            "predicted": jnp.concatenate([bin_centers[mask], jnp.array([0.0, 1.0])]),
            "observed": jnp.concatenate([true_frequencies[mask], jnp.array([0.0, 1.0])]),
            "type": ["Calibration"] * int(jnp.sum(mask)) + ["Perfect", "Perfect"],
        }
    )

    p = (
        ggplot(cal_df, aes(x="predicted", y="observed", color="type"))
        + geom_line(size=1)
        + geom_point(data=pd.DataFrame(cal_df[cal_df["type"] == "Calibration"]), size=3)
        + labs(
            title="Calibration Curve",
            x="Predicted Probability",
            y="True Frequency",
            color="",
        )
        + theme_minimal()
        + scale_x_continuous(limits=(0, 1))
        + scale_y_continuous(limits=(0, 1))
    )

    return p


def plot_roc_curve(fpr: Array, tpr: Array, auc: float | None = None):
    """
    Plot ROC curve for discriminator performance evaluation.

    This is the most important diagnostic for assessing discriminator quality.
    The ROC curve shows the trade-off between true positive rate and false
    positive rate at different classification thresholds.

    Parameters
    ----------
    fpr : jax.Array
        False positive rates
    tpr : jax.Array
        True positive rates
    auc : float, optional
        Area under the ROC curve. If not provided, will be computed.

    Returns
    -------
    plot : plotnine.ggplot
        ROC curve plot

    Notes
    -----
    A good discriminator will have an ROC curve that bows toward the top-left
    corner (high TPR, low FPR) with AUC close to 1.0. An AUC of 0.5 indicates
    random guessing.

    The discriminator's ability to distinguish observed from permuted data
    directly relates to the quality of the estimated weights.

    Examples
    --------
    >>> from stochpw import PermutationWeighter, roc_curve
    >>> from stochpw.plotting import plot_roc_curve
    >>> # After fitting
    >>> weighter.fit(X, A)
    >>> weights = weighter.predict(X, A)
    >>> # Create permuted data
    >>> A_perm = A[jax.random.permutation(key, len(A))]
    >>> weights_perm = weighter.predict(X, A_perm)
    >>> # Compute ROC
    >>> all_weights = jnp.concatenate([weights, weights_perm])
    >>> labels = jnp.concatenate([jnp.zeros(len(weights)), jnp.ones(len(weights_perm))])
    >>> fpr, tpr, _ = roc_curve(all_weights, labels)
    >>> auc = jnp.trapezoid(tpr, fpr)
    >>> plot = plot_roc_curve(fpr, tpr, auc)
    >>> print(plot)
    """
    # Compute AUC if not provided
    if auc is None:
        # Use trapezoidal rule for AUC
        auc = float(jnp.trapezoid(tpr, fpr))

    # Create dataframes for actual ROC and diagonal reference
    roc_df = pd.DataFrame(
        {
            "fpr": jnp.concatenate([fpr, jnp.array([0.0, 1.0])]),
            "tpr": jnp.concatenate([tpr, jnp.array([0.0, 1.0])]),
            "type": ["ROC Curve"] * len(fpr) + ["Random Classifier", "Random Classifier"],
        }
    )

    title = f"ROC Curve (AUC = {auc:.3f})"

    p = (
        ggplot(roc_df, aes(x="fpr", y="tpr", color="type"))
        + geom_line(size=1)
        + geom_point(data=pd.DataFrame(roc_df[roc_df["type"] == "ROC Curve"]), size=2, alpha=0.6)
        + labs(
            title=title,
            x="False Positive Rate",
            y="True Positive Rate",
            color="",
        )
        + theme_minimal()
        + scale_x_continuous(limits=(0, 1))
        + scale_y_continuous(limits=(0, 1))
    )

    return p
