"""Batch creation for permutation weighting training."""

import jax.numpy as jnp
from jax import Array

from ..data import TrainingBatch
from .permutation import BasePermuter, RandomPermuter


def create_training_batch(
    X: Array,
    A: Array,
    batch_indices: Array,
    rng_key: Array,
    permuter: BasePermuter | None = None,
) -> TrainingBatch:
    """
    Create a training batch with observed and permuted pairs.

    Includes first-order interactions (A*X) which are critical for the
    discriminator to learn the association between treatment and covariates.

    Parameters
    ----------
    X : jax.Array, shape (n, d_x)
        Covariates
    A : jax.Array, shape (n, d_a)
        Treatments
    batch_indices : jax.Array, shape (batch_size,)
        Indices for this batch
    rng_key : jax.random.PRNGKey
        PRNG key for permutation
    permuter : BasePermuter, optional
        Permutation strategy. If None, uses RandomPermuter().

    Returns
    -------
    TrainingBatch
        Batch with concatenated observed and permuted data, including interactions
    """
    # Use default permuter if not specified
    if permuter is None:
        permuter = RandomPermuter()

    # Sample observed batch
    X_obs = X[batch_indices]
    A_obs = A[batch_indices]

    # Create permuted batch using permuter
    # This creates the product distribution P(A)P(X) within the batch
    batch_size = len(batch_indices)
    A_perm, X_perm = permuter.permute(A_obs, X_obs, rng_key)

    # Compute interactions: outer product A ⊗ X
    # For each sample, creates all A_i * X_j combinations
    interactions_obs = jnp.einsum("bi,bj->bij", A_obs, X_obs).reshape(batch_size, -1)
    interactions_perm = jnp.einsum("bi,bj->bij", A_perm, X_perm).reshape(batch_size, -1)

    # Concatenate and label
    X_batch = jnp.concatenate([X_obs, X_perm])
    A_batch = jnp.concatenate([A_obs, A_perm])
    interactions_batch = jnp.concatenate([interactions_obs, interactions_perm])
    C_batch = jnp.concatenate(
        [
            jnp.zeros(batch_size),  # Observed: C=0
            jnp.ones(batch_size),  # Permuted: C=1
        ]
    )

    return TrainingBatch(X=X_batch, A=A_batch, C=C_batch, AX=interactions_batch)
