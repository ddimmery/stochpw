"""Training loop for permutation weighting discriminators."""

from typing import Callable

import jax
import jax.numpy as jnp
import optax
from jax import Array

from ..data import TrainingState, TrainingStepResult
from .batch import create_training_batch
from .losses import logistic_loss


def train_step(
    state: TrainingState,
    batch,
    discriminator_fn: Callable,
    optimizer: optax.GradientTransformation,
    loss_fn_type: Callable[[Array, Array], Array] = logistic_loss,
    regularization_fn: Callable[[Array], Array] | None = None,
    regularization_strength: float = 0.0,
    eps: float = 1e-7,
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
    loss_fn_type : Callable, default=logistic_loss
        Loss function (logits, labels) -> loss
    regularization_fn : Callable, optional
        Regularization function on weights (weights) -> penalty
    regularization_strength : float, default=0.0
        Strength of regularization penalty
    eps : float, default=1e-7
        Numerical stability constant for weight computation

    Returns
    -------
    TrainingStepResult
        Updated state and loss value
    """

    def loss_fn(params):
        logits = discriminator_fn(params, batch.A, batch.X, batch.AX)
        loss = loss_fn_type(logits, batch.C)

        # Add weight-based regularization if specified
        if regularization_fn is not None and regularization_strength > 0:
            # Compute weights from discriminator output (only for observed data)
            # Filter to C=0 (observed data) for weight computation
            observed_mask = batch.C == 0
            observed_logits = logits[observed_mask]
            eta = jax.nn.sigmoid(observed_logits)
            eta_clipped = jnp.clip(eta, eps, 1 - eps)
            weights = eta_clipped / (1 - eta_clipped)

            # Apply regularization on weights
            penalty = regularization_fn(weights)
            loss = loss + regularization_strength * penalty

        return loss

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
    loss_fn: Callable[[Array, Array], Array] = logistic_loss,
    regularization_fn: Callable[[Array], Array] | None = None,
    regularization_strength: float = 0.0,
    early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 1e-4,
    eps: float = 1e-7,
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
    loss_fn : Callable, default=logistic_loss
        Loss function (logits, labels) -> loss
    regularization_fn : Callable, optional
        Regularization function on weights (weights) -> penalty
    regularization_strength : float, default=0.0
        Strength of regularization penalty
    early_stopping : bool, default=False
        Whether to use early stopping based on validation loss
    patience : int, default=10
        Number of epochs to wait for improvement before stopping
    min_delta : float, default=1e-4
        Minimum change in loss to qualify as improvement
    eps : float, default=1e-7
        Numerical stability constant for weight computation

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

    # Early stopping state
    best_loss = float("inf")
    best_params = init_params
    epochs_without_improvement = 0

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
            result = train_step(
                state,
                batch,
                discriminator_fn,
                optimizer,
                loss_fn_type=loss_fn,
                regularization_fn=regularization_fn,
                regularization_strength=regularization_strength,
                eps=eps,
            )
            state = result.state
            epoch_losses.append(float(result.loss))

        # Record epoch loss
        mean_epoch_loss = jnp.mean(jnp.array(epoch_losses))
        state.history["loss"].append(float(mean_epoch_loss))
        state.epoch = epoch + 1

        # Early stopping logic
        if early_stopping:
            if mean_epoch_loss < best_loss - min_delta:
                best_loss = mean_epoch_loss
                best_params = state.params
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                # Restore best parameters
                state.params = best_params
                break

    return state.params, state.history
