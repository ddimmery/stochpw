"""Regularization functions for permutation weighting.

This module provides regularization classes that operate on the output
weights rather than model parameters, encouraging desirable properties
of the reweighting scheme.
"""

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array


class BaseRegularizer(ABC):
    """
    Abstract base class for regularizers.

    All regularizers should inherit from this class and implement
    the __call__ method.
    """

    @abstractmethod
    def __call__(self, weights: Array) -> Array:
        """
        Compute regularization penalty for given weights.

        Parameters
        ----------
        weights : Array, shape (n,)
            Importance weights

        Returns
        -------
        penalty : Array
            Regularization penalty (scalar)
        """
        pass


class EntropyRegularizer(BaseRegularizer):
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
    eps : float, default=1e-7
        Small constant for numerical stability

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from stochpw.training import EntropyRegularizer
    >>> regularizer = EntropyRegularizer()
    >>> weights = jnp.array([1.0, 1.0, 1.0, 1.0])  # Uniform weights
    >>> penalty = regularizer(weights)
    >>> # Uniform weights have high entropy (low penalty)

    Notes
    -----
    - Uniform weights have maximum entropy (minimum penalty)
    - Highly concentrated weights have low entropy (high penalty)
    - Encourages effective sample size close to n
    """

    def __init__(self, eps: float = 1e-7):
        self.eps: float = eps

    def __call__(self, weights: Array) -> Array:
        """Compute negative entropy penalty."""
        # Normalize weights to probability distribution
        p = weights / (jnp.sum(weights) + self.eps)
        p = jnp.clip(p, self.eps, 1.0)  # Avoid log(0)

        # Compute entropy: H = -sum(p * log(p))
        entropy = -jnp.sum(p * jnp.log(p))

        # Return negative entropy as penalty (we want to maximize entropy)
        return -entropy


class LpRegularizer(BaseRegularizer):
    """
    L_p penalty on weight deviations from uniform.

    Penalizes weights that deviate from 1, encouraging more uniform weighting.
    Different values of p produce different behaviors:
    - p=1: L1 penalty (sparse, robust to outliers)
    - p=2: L2 penalty (smooth, sensitive to large deviations)

    Parameters
    ----------
    p : float, default=2.0
        The power for the L_p norm (must be >= 1)
    strength : float, default=1.0
        Strength of the regularization penalty (multiplicative factor)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from stochpw.training import LpRegularizer
    >>> # L2 regularizer
    >>> regularizer = LpRegularizer(p=2.0)
    >>> weights = jnp.array([1.0, 1.0, 2.0, 0.5])
    >>> penalty = regularizer(weights)
    >>>
    >>> # L1 regularizer
    >>> regularizer = LpRegularizer(p=1.0)
    >>> penalty = regularizer(weights)

    Notes
    -----
    Computed as sum(|weights - 1|^p), which penalizes deviation
    from uniform weights (weight=1 for each observation).
    """

    def __init__(self, p: float = 2.0, strength: float = 1.0):
        if p < 1.0:
            raise ValueError(f"p must be >= 1, got {p}")
        self.p: float = p
        self.strength: float = strength

    def __call__(self, weights: Array) -> Array:
        """Compute L_p penalty on weight deviations."""
        deviations = jnp.abs(weights - 1.0)
        return self.strength * jnp.sum(deviations**self.p)


class NoRegularizer(BaseRegularizer):
    """
    No-op regularizer that returns zero penalty.

    Useful as a default when no regularization is desired,
    avoiding None checks throughout the codebase.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from stochpw.training import NoRegularizer
    >>> regularizer = NoRegularizer()
    >>> weights = jnp.array([1.0, 2.0, 3.0])
    >>> penalty = regularizer(weights)
    >>> assert penalty == 0.0
    """

    def __call__(self, weights: Array) -> Array:
        """Return zero penalty."""
        return jnp.array(0.0)
