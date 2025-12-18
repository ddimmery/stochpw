"""Permutation strategies for treatment assignments."""

from abc import ABC, abstractmethod

import jax.random as random
from jax import Array


class BasePermuter(ABC):
    """
    Abstract base class for treatment permutation strategies.

    All permuters should inherit from this class and implement
    the permute method.
    """

    @abstractmethod
    def permute(self, A: Array, X: Array, rng_key: Array) -> tuple[Array, Array]:
        """
        Permute treatment assignments to create synthetic comparison data.

        Parameters
        ----------
        A : jax.Array, shape (n, d_a)
            Treatment assignments
        X : jax.Array, shape (n, d_x)
            Covariates
        rng_key : jax.random.PRNGKey
            Random key for permutation

        Returns
        -------
        A_perm : jax.Array, shape (n, d_a)
            Permuted treatment assignments
        X_perm : jax.Array, shape (n, d_x)
            Covariates (may be unchanged or modified depending on strategy)
        """
        pass


class RandomPermuter(BasePermuter):
    """
    Standard random permutation of treatment assignments.

    This is the default permutation strategy used in permutation weighting.
    It randomly shuffles treatment assignments within the batch while keeping
    covariates fixed, creating the product distribution P(A)P(X).

    For multi-dimensional treatments, entire treatment vectors are permuted
    together (not individual treatment dimensions independently).

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from stochpw.training import RandomPermuter
    >>> permuter = RandomPermuter()
    >>> A = jnp.array([[0.0], [1.0], [0.0], [1.0]])
    >>> X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    >>> rng_key = jax.random.PRNGKey(42)
    >>> A_perm, X_perm = permuter.permute(A, X, rng_key)
    >>> # A_perm is a shuffled version of A
    >>> # X_perm equals X (covariates unchanged)

    Notes
    -----
    This implements within-batch permutation, which ensures that:
    - P(A) and P(X) are identical in observed and permuted distributions
    - Only the joint P(A,X) differs between the two distributions
    - The discriminator learns the association between A and X
    """

    def permute(self, A: Array, X: Array, rng_key: Array) -> tuple[Array, Array]:
        """
        Randomly permute treatment assignments.

        Parameters
        ----------
        A : jax.Array, shape (n, d_a)
            Treatment assignments
        X : jax.Array, shape (n, d_x)
            Covariates
        rng_key : jax.random.PRNGKey
            Random key for permutation

        Returns
        -------
        A_perm : jax.Array, shape (n, d_a)
            Randomly permuted treatment assignments
        X_perm : jax.Array, shape (n, d_x)
            Unchanged covariates (same as input X)
        """
        n_samples = A.shape[0]
        perm_indices = random.permutation(rng_key, n_samples)
        A_perm = A[perm_indices]
        X_perm = X  # Covariates unchanged
        return A_perm, X_perm
