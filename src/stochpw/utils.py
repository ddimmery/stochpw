"""Utility functions for input validation and data handling."""

import jax.numpy as jnp
import numpy as np
from jax import Array


def _check_array_validity(arr: Array, name: str) -> None:
    """Check array for NaNs and Infs."""
    if jnp.any(jnp.isnan(arr)) or jnp.any(jnp.isinf(arr)):
        raise ValueError(f"{name} contains NaN or Inf values")


def _validate_treatment_variation(A: Array) -> None:
    """Check that treatment has sufficient variation."""
    if A.shape[1] == 1:  # Single treatment
        unique_vals = jnp.unique(A)
        if len(unique_vals) < 2:
            raise ValueError(
                f"A must have at least 2 unique values for discrimination, "
                f"found {len(unique_vals)}"
            )


def validate_inputs(X: Array | np.ndarray, A: Array | np.ndarray) -> tuple[Array, Array]:
    """
    Validate and convert inputs to JAX arrays.

    Checks:
    - X and A have compatible shapes (same number of samples)
    - No NaNs or Infs
    - Sufficient variation in A
    - Converts numpy arrays to JAX arrays if needed

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Covariates
    A : array-like, shape (n_samples,) or (n_samples, n_treatments)
        Intervention/treatment assignments

    Returns
    -------
    X_jax : jax.Array
        Validated and converted X
    A_jax : jax.Array
        Validated and converted A

    Raises
    ------
    ValueError
        If inputs are invalid (incompatible shapes, NaNs, no variation, etc.)
    """
    # Convert to JAX arrays if needed
    if isinstance(X, np.ndarray):
        X = jnp.array(X)
    if isinstance(A, np.ndarray):
        A = jnp.array(A)

    # Check shapes
    if X.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")

    # Ensure A is at least 1D
    if A.ndim == 0:
        raise ValueError("A must be at least 1-dimensional, got scalar")
    if A.ndim == 1:
        A = A.reshape(-1, 1)  # Make it (n, 1) for consistency
    if A.ndim > 2:
        raise ValueError(f"A must be 1 or 2-dimensional, got shape {A.shape}")

    # Check number of samples match
    if X.shape[0] != A.shape[0]:
        raise ValueError(
            f"X and A must have same number of samples: " f"X has {X.shape[0]}, A has {A.shape[0]}"
        )

    # Check for NaNs or Infs
    _check_array_validity(X, "X")
    _check_array_validity(A, "A")

    # Check for sufficient variation in A
    _validate_treatment_variation(A)

    # Check minimum sample size
    if X.shape[0] < 10:
        raise ValueError(f"Need at least 10 samples for training, got {X.shape[0]}")

    return X, A


def permute_treatment(A: Array, rng_key: Array) -> Array:
    """
    Randomly permute treatment assignments.

    For multi-dimensional A, this permutes entire rows (treatment vectors)
    together, not individual elements.

    Parameters
    ----------
    A : jax.Array, shape (batch_size,) or (batch_size, n_treatments)
        Intervention assignments
    rng_key : jax.random.PRNGKey
        Random key for permutation

    Returns
    -------
    A_perm : jax.Array, same shape as A
        Permuted treatment assignments
    """
    import jax.random as random

    n_samples = A.shape[0]
    perm_indices = random.permutation(rng_key, n_samples)
    return A[perm_indices]
