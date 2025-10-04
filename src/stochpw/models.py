"""Discriminator model architectures for permutation weighting."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

PyTree = Any  # Type alias for JAX PyTree


def create_linear_discriminator(d_a: int, d_x: int) -> tuple[Callable, Callable]:
    """
    Create a linear discriminator using A, X, and A*X interactions.

    logit = w_a^T A + w_x^T X + w_ax^T (A ⊗ X) + b

    This allows the model to learn from:
    - Marginal treatment effects (w_a)
    - Marginal covariate effects (w_x)
    - Treatment-covariate interactions (w_ax)

    Parameters
    ----------
    d_a : int
        Dimension of treatment vector
    d_x : int
        Dimension of covariate vector

    Returns
    -------
    init_fn : Callable
        Function (rng_key) -> params to initialize parameters
    apply_fn : Callable
        Function (params, a, x, ax) -> logits to compute discriminator output
    """
    interaction_dim = d_a * d_x
    total_dim = d_a + d_x + interaction_dim

    def init_fn(rng_key: Array) -> dict:
        """Initialize linear discriminator parameters."""
        w_key, b_key = jax.random.split(rng_key)

        # Xavier/Glorot initialization
        std = jnp.sqrt(2.0 / total_dim)
        w_a = jax.random.normal(jax.random.fold_in(w_key, 0), (d_a,)) * std
        w_x = jax.random.normal(jax.random.fold_in(w_key, 1), (d_x,)) * std
        w_ax = jax.random.normal(jax.random.fold_in(w_key, 2), (interaction_dim,)) * std
        b = jax.random.normal(b_key, ()) * 0.01

        return {"w_a": w_a, "w_x": w_x, "w_ax": w_ax, "b": b}

    def apply_fn(params: dict, a: Array, x: Array, ax: Array) -> Array:
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

    return init_fn, apply_fn
