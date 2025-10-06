"""Linear discriminator for permutation weighting."""

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from .base import BaseDiscriminator

PyTree = Any


class LinearDiscriminator(BaseDiscriminator):
    """
    Linear discriminator using A, X, and A*X interactions.

    The discriminator computes logits as:
        logit = w_a^T A + w_x^T X + w_ax^T (A ⊗ X) + b

    This allows the model to learn from:
    - Marginal treatment effects (w_a)
    - Marginal covariate effects (w_x)
    - Treatment-covariate interactions (w_ax)

    The explicit inclusion of A*X interactions is critical for linear models
    because within-batch permutation ensures P(A) and P(X) are identical
    in observed vs permuted batches. Only the joint distribution P(A,X)
    differs, which linear models need explicit interaction terms to capture.

    Examples
    --------
    >>> from stochpw.models import LinearDiscriminator
    >>> import jax
    >>>
    >>> discriminator = LinearDiscriminator()
    >>> params = discriminator.init_params(jax.random.PRNGKey(0), d_a=1, d_x=3)
    >>> # params contains: w_a, w_x, w_ax, b
    """

    def init_params(self, rng_key: Array, d_a: int, d_x: int) -> dict:
        """
        Initialize linear discriminator parameters.

        Uses Xavier/Glorot initialization for weights and small random
        initialization for bias.

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
        params : dict
            Dictionary with keys:
            - 'w_a': Array of shape (d_a,) - treatment weights
            - 'w_x': Array of shape (d_x,) - covariate weights
            - 'w_ax': Array of shape (d_a * d_x,) - interaction weights
            - 'b': scalar - bias term
        """
        interaction_dim = d_a * d_x
        total_dim = d_a + d_x + interaction_dim

        w_key, b_key = jax.random.split(rng_key)

        # Xavier/Glorot initialization
        std = jnp.sqrt(2.0 / total_dim)
        w_a = jax.random.normal(jax.random.fold_in(w_key, 0), (d_a,)) * std
        w_x = jax.random.normal(jax.random.fold_in(w_key, 1), (d_x,)) * std
        w_ax = jax.random.normal(jax.random.fold_in(w_key, 2), (interaction_dim,)) * std
        b = jax.random.normal(b_key, ()) * 0.01

        return {"w_a": w_a, "w_x": w_x, "w_ax": w_ax, "b": b}

    def apply(self, params: dict, a: Array, x: Array, ax: Array) -> Array:
        """
        Compute linear discriminator logits using A, X, and A*X.

        Parameters
        ----------
        params : dict
            Parameters with keys 'w_a', 'w_x', 'w_ax', and 'b'
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
        """
        # Ensure a is 2D
        if a.ndim == 1:
            a = a.reshape(-1, 1)

        # Linear transformation: w_a^T A + w_x^T X + w_ax^T (A*X) + b
        logits = (
            jnp.dot(a, params["w_a"])
            + jnp.dot(x, params["w_x"])
            + jnp.dot(ax, params["w_ax"])
            + params["b"]
        )

        return logits
