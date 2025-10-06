"""Abstract base class for discriminator models."""

from abc import ABC, abstractmethod
from typing import Any

from jax import Array

PyTree = Any  # Type alias for JAX PyTree


class BaseDiscriminator(ABC):
    """
    Abstract base class for discriminator models in permutation weighting.

    All discriminator models must implement the `init_params` and `apply` methods
    to define how parameters are initialized and how logits are computed from
    treatments (A), covariates (X), and their interactions (AX).

    The discriminator's role is to distinguish between:
    - Observed pairs (X, A) with label C=0
    - Permuted pairs (X, A') with label C=1

    Examples
    --------
    >>> class MyDiscriminator(BaseDiscriminator):
    ...     def init_params(self, rng_key, d_a, d_x):
    ...         # Initialize parameters
    ...         return {"w": jax.random.normal(rng_key, (d_a + d_x,))}
    ...
    ...     def apply(self, params, a, x, ax):
    ...         # Compute logits
    ...         return jnp.dot(jnp.concatenate([a, x], axis=-1), params["w"])
    """

    @abstractmethod
    def init_params(self, rng_key: Array, d_a: int, d_x: int) -> PyTree:
        """
        Initialize discriminator parameters.

        Parameters
        ----------
        rng_key : jax.Array
            Random key for parameter initialization
        d_a : int
            Dimension of treatment vector
        d_x : int
            Dimension of covariate vector

        Returns
        -------
        params : PyTree
            Initialized parameters (any JAX-compatible PyTree structure)
        """
        pass

    @abstractmethod
    def apply(self, params: PyTree, a: Array, x: Array, ax: Array) -> Array:
        """
        Compute discriminator logits.

        Parameters
        ----------
        params : PyTree
            Model parameters (output of init_params)
        a : jax.Array, shape (batch_size, d_a) or (batch_size,)
            Treatment assignments
        x : jax.Array, shape (batch_size, d_x)
            Covariates
        ax : jax.Array, shape (batch_size, d_a * d_x)
            Pre-computed first-order interactions A ⊗ X

        Returns
        -------
        logits : jax.Array, shape (batch_size,)
            Discriminator logits for p(C=1 | a, x)

        Notes
        -----
        The logits represent the raw output before sigmoid activation.
        They are used in the logistic loss: BCE(sigmoid(logits), labels).
        """
        pass

    def __call__(self, params: PyTree, a: Array, x: Array, ax: Array) -> Array:
        """
        Convenience method to call apply.

        Allows discriminator to be called as: discriminator(params, a, x, ax).
        """
        return self.apply(params, a, x, ax)
