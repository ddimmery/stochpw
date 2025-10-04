"""Training utilities for permutation weighting discriminators."""

from typing import Callable

import jax
import jax.numpy as jnp
import optax
from jax import Array

from .data import TrainingBatch, TrainingState, TrainingStepResult
from .utils import permute_treatment


def create_training_batch(
    X: Array, A: Array, batch_indices: Array, rng_key: Array
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

    Returns
    -------
    TrainingBatch
        Batch with concatenated observed and permuted data, including interactions
    """
    # Sample observed batch
    X_obs = X[batch_indices]
    A_obs = A[batch_indices]

    # Create permuted batch by shuffling treatments WITHIN the batch
    # This creates the product distribution P(A)P(X) within the batch
    batch_size = len(batch_indices)
    X_perm = X_obs  # Same covariates (not shuffled)
    A_perm = permute_treatment(A_obs, rng_key)  # Shuffle treatments within batch

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


def train_step(
    state: TrainingState,
    batch: TrainingBatch,
    discriminator_fn: Callable,
    optimizer: optax.GradientTransformation,
) -> TrainingStepResult:
    """
    Single training step (JIT-compiled).

    Computes loss, gradients, and updates parameters.

    Parameters
    ----------
    state : TrainingState
        Current training state
    batch : TrainingBatch
        Training batch
    discriminator_fn : Callable
        Discriminator function (params, a, x, ax) -> logits
    optimizer : optax.GradientTransformation
        Optax optimizer

    Returns
    -------
    TrainingStepResult
        Updated state and loss value
    """

    def loss_fn(params):
        logits = discriminator_fn(params, batch.A, batch.X, batch.AX)
        return logistic_loss(logits, batch.C)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)

    new_state = TrainingState(
        params=params,
        opt_state=opt_state,
        rng_key=state.rng_key,
        epoch=state.epoch,
        history=state.history,
    )

    return TrainingStepResult(state=new_state, loss=loss)


def fit_discriminator(
    X: Array,
    A: Array,
    discriminator_fn: Callable,
    init_params: dict,
    optimizer: optax.GradientTransformation,
    num_epochs: int,
    batch_size: int,
    rng_key: Array,
) -> tuple[dict, dict]:
    """
    Complete training loop for discriminator.

    Parameters
    ----------
    X : jax.Array, shape (n, d_x)
        Covariates
    A : jax.Array, shape (n, d_a)
        Treatments
    discriminator_fn : Callable
        Discriminator function (params, a, x, ax) -> logits
    init_params : dict
        Initial parameters
    optimizer : optax.GradientTransformation
        Optax optimizer
    num_epochs : int
        Number of training epochs
    batch_size : int
        Mini-batch size
    rng_key : jax.random.PRNGKey
        Random key for reproducibility

    Returns
    -------
    params : dict
        Fitted discriminator parameters
    history : dict
        Training history with keys 'loss' (list of losses per epoch)
    """
    n = X.shape[0]
    opt_state = optimizer.init(init_params)

    # Initialize state
    state = TrainingState(
        params=init_params,
        opt_state=opt_state,
        rng_key=rng_key,
        epoch=0,
        history={"loss": []},
    )

    for epoch in range(num_epochs):
        # Split RNG key for this epoch
        epoch_key, state.rng_key = jax.random.split(state.rng_key)

        # Shuffle data
        perm = jax.random.permutation(epoch_key, n)
        X_shuffled = X[perm]
        A_shuffled = A[perm]

        # Train on batches
        epoch_losses = []
        num_batches = n // batch_size

        for i in range(num_batches):
            batch_key, epoch_key = jax.random.split(epoch_key)

            # Get batch indices
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_indices = jnp.arange(start_idx, end_idx)

            # Create training batch
            batch = create_training_batch(X_shuffled, A_shuffled, batch_indices, batch_key)

            # Training step
            result = train_step(state, batch, discriminator_fn, optimizer)
            state = result.state
            epoch_losses.append(float(result.loss))

        # Record epoch loss
        mean_epoch_loss = jnp.mean(jnp.array(epoch_losses))
        state.history["loss"].append(float(mean_epoch_loss))
        state.epoch = epoch + 1

    return state.params, state.history
