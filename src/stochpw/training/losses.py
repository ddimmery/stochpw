"""Loss functions for discriminator training."""

import jax
import jax.numpy as jnp
import optax
from jax import Array


@jax.jit
def logistic_loss(logits: Array, labels: Array) -> Array:
    """
    Binary cross-entropy loss for discriminator.

    Uses numerically stable log-sigmoid implementation.

    Parameters
    ----------
    logits : jax.Array, shape (batch_size,)
        Raw discriminator outputs
    labels : jax.Array, shape (batch_size,)
        Binary labels (0 or 1)

    Returns
    -------
    loss : float
        Scalar loss value
    """
    # Use optax's stable implementation
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()


@jax.jit
def exponential_loss(logits: Array, labels: Array) -> Array:
    """
    Exponential loss for density ratio estimation.

    This is a proper scoring rule that can be more robust than logistic loss
    for density ratio estimation. It directly optimizes the exponential of
    the log density ratio.

    Parameters
    ----------
    logits : jax.Array, shape (batch_size,)
        Raw discriminator outputs
    labels : jax.Array, shape (batch_size,)
        Binary labels (0 or 1)

    Returns
    -------
    loss : float
        Scalar loss value

    Notes
    -----
    The exponential loss is defined as:
        L = E[exp(-y * f(x))]
    where y ∈ {-1, +1} and f(x) are the logits.
    We convert labels from {0, 1} to {-1, +1} for this formulation.
    """
    # Convert labels from {0, 1} to {-1, +1}
    y = 2 * labels - 1
    # Exponential loss: E[exp(-y * logits)]
    return jnp.mean(jnp.exp(-y * logits))


@jax.jit
def brier_loss(logits: Array, labels: Array) -> Array:
    """
    Brier score loss for probabilistic predictions.

    The Brier score is the mean squared error between predicted probabilities
    and true labels. It's a proper scoring rule that encourages well-calibrated
    predictions.

    Parameters
    ----------
    logits : jax.Array, shape (batch_size,)
        Raw discriminator outputs
    labels : jax.Array, shape (batch_size,)
        Binary labels (0 or 1)

    Returns
    -------
    loss : float
        Scalar loss value

    Notes
    -----
    The Brier score is defined as:
        BS = (1/n) * Σ(p_i - y_i)²
    where p_i = σ(logits_i) is the predicted probability and y_i is the true label.
    """
    # Convert logits to probabilities
    probs = jax.nn.sigmoid(logits)
    # Brier score: mean squared error
    return jnp.mean((probs - labels) ** 2)
