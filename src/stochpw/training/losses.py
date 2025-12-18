"""Loss functions for discriminator training."""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import optax
from jax import Array


class BaseLoss(ABC):
    """
    Abstract base class for loss functions.

    All loss functions should inherit from this class and implement
    the __call__ method.
    """

    @abstractmethod
    def __call__(self, logits: Array, labels: Array) -> Array:
        """
        Compute loss for given logits and labels.

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
        pass


class LogisticLoss(BaseLoss):
    """
    Binary cross-entropy loss for discriminator.

    Uses numerically stable log-sigmoid implementation via optax.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from stochpw.training import LogisticLoss
    >>> loss_fn = LogisticLoss()
    >>> logits = jnp.array([2.0, -1.0])
    >>> labels = jnp.array([1.0, 0.0])
    >>> loss = loss_fn(logits, labels)
    """

    def __call__(self, logits: Array, labels: Array) -> Array:
        """Compute binary cross-entropy loss."""
        return optax.sigmoid_binary_cross_entropy(logits, labels).mean()


class ExponentialLoss(BaseLoss):
    """
    Exponential loss for density ratio estimation.

    This is a proper scoring rule that can be more robust than logistic loss
    for density ratio estimation. It directly optimizes the exponential of
    the log density ratio.

    Notes
    -----
    The exponential loss is defined as:
        L = E[exp(-y * f(x))]
    where y ∈ {-1, +1} and f(x) are the logits.
    We convert labels from {0, 1} to {-1, +1} for this formulation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from stochpw.training import ExponentialLoss
    >>> loss_fn = ExponentialLoss()
    >>> logits = jnp.array([2.0, -1.0])
    >>> labels = jnp.array([1.0, 0.0])
    >>> loss = loss_fn(logits, labels)
    """

    def __call__(self, logits: Array, labels: Array) -> Array:
        """Compute exponential loss."""
        # Convert labels from {0, 1} to {-1, +1}
        y = 2 * labels - 1
        # Exponential loss: E[exp(-y * logits)]
        return jnp.mean(jnp.exp(-y * logits))


class BrierLoss(BaseLoss):
    """
    Brier score loss for probabilistic predictions.

    The Brier score is the mean squared error between predicted probabilities
    and true labels. It's a proper scoring rule that encourages well-calibrated
    predictions.

    Notes
    -----
    The Brier score is defined as:
        BS = (1/n) * Σ(p_i - y_i)²
    where p_i = σ(logits_i) is the predicted probability and y_i is the true label.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from stochpw.training import BrierLoss
    >>> loss_fn = BrierLoss()
    >>> logits = jnp.array([2.0, -1.0])
    >>> labels = jnp.array([1.0, 0.0])
    >>> loss = loss_fn(logits, labels)
    """

    def __call__(self, logits: Array, labels: Array) -> Array:
        """Compute Brier score loss."""
        # Convert logits to probabilities
        probs = jax.nn.sigmoid(logits)
        # Brier score: mean squared error
        return jnp.mean((probs - labels) ** 2)
