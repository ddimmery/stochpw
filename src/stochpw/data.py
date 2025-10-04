"""Data structures for permutation weighting."""

from dataclasses import dataclass
from typing import Any

from jax import Array

PyTree = Any  # Type alias for JAX PyTree


@dataclass(frozen=True)
class TrainingBatch:
    """A batch of data for discriminator training."""

    X: Array  # Covariates, shape (2 * batch_size, d_x)
    A: Array  # Treatments, shape (2 * batch_size, d_a)
    C: Array  # Labels (0=observed, 1=permuted), shape (2 * batch_size,)
    AX: Array  # A*X interactions, shape (2 * batch_size, d_a * d_x)


@dataclass(frozen=True)
class WeightedData:
    """Data with computed importance weights."""

    X: Array  # Covariates, shape (n, d_x)
    A: Array  # Treatments, shape (n, d_a)
    weights: Array  # Importance weights, shape (n,)


@dataclass
class TrainingState:
    """State of the training process."""

    params: PyTree  # Model parameters
    opt_state: Any  # Optimizer state (Optax OptState)
    rng_key: Array  # RNG key for reproducibility
    epoch: int  # Current epoch
    history: dict  # Training history


@dataclass(frozen=True)
class TrainingStepResult:
    """Result of a single training step."""

    state: TrainingState  # Updated training state
    loss: float  # Loss value for this step
