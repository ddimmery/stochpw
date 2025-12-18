"""Training loop for permutation weighting discriminators."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jax import Array

from ..data import TrainingBatch, TrainingState, TrainingStepResult
from ..types import PyTree
from .batch import create_training_batch
from .early_stopping import BaseEarlyStopping, NoEarlyStopping
from .losses import BaseLoss, LogisticLoss
from .permutation import BasePermuter, RandomPermuter
from .regularization import BaseRegularizer, NoRegularizer


def train_step(
    state: TrainingState,
    batch: TrainingBatch,
    discriminator_fn: Callable[[PyTree, Array, Array, Array], Array],
    optimizer: optax.GradientTransformation,
    loss_fn: BaseLoss,
    regularizer: BaseRegularizer,
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
    loss_fn : BaseLoss
        Loss function instance
    regularizer : BaseRegularizer
        Regularization instance
    eps : float, default=1e-7
        Numerical stability constant for weight computation

    Returns
    -------
    TrainingStepResult
        Updated state and loss value
    """

    def loss_fn_total(params: PyTree) -> Array:
        logits = discriminator_fn(params, batch.A, batch.X, batch.AX)
        loss = loss_fn(logits, batch.C)

        # Add weight-based regularization
        # Compute weights from discriminator output (only for observed data)
        # Filter to C=0 (observed data) for weight computation
        observed_mask = batch.C == 0
        observed_logits = logits[observed_mask]
        eta = jax.nn.sigmoid(observed_logits)
        eta_clipped = jnp.clip(eta, eps, 1 - eps)
        weights = eta_clipped / (1 - eta_clipped)

        # Apply regularization on weights
        penalty = regularizer(weights)
        loss = loss + penalty

        return loss

    loss, grads = jax.value_and_grad(loss_fn_total)(state.params)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)

    new_state = TrainingState(
        params=params,  # type: ignore[arg-type]
        opt_state=opt_state,
        rng_key=state.rng_key,
        epoch=state.epoch,
        history=state.history,
    )

    return TrainingStepResult(state=new_state, loss=loss)


def fit_discriminator(
    X: Array,
    A: Array,
    discriminator_fn: Callable[[PyTree, Array, Array, Array], Array],
    init_params: PyTree,
    optimizer: optax.GradientTransformation,
    num_epochs: int,
    batch_size: int,
    rng_key: Array,
    loss_fn: BaseLoss | None = None,
    regularizer: BaseRegularizer | None = None,
    early_stopping: BaseEarlyStopping | None = None,
    permuter: BasePermuter | None = None,
    eps: float = 1e-7,
) -> tuple[PyTree, dict[str, list[float]]]:
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
    loss_fn : BaseLoss, optional
        Loss function instance. If None, uses LogisticLoss().
    regularizer : BaseRegularizer, optional
        Regularization instance. If None, uses NoRegularizer().
    early_stopping : BaseEarlyStopping, optional
        Early stopping instance. If None, uses NoEarlyStopping().
    permuter : BasePermuter, optional
        Permutation strategy. If None, uses RandomPermuter().
    eps : float, default=1e-7
        Numerical stability constant for weight computation

    Returns
    -------
    params : dict
        Fitted discriminator parameters
    history : dict
        Training history with keys 'loss' (list of losses per epoch)
    """
    # Set defaults
    if loss_fn is None:
        loss_fn = LogisticLoss()
    if regularizer is None:
        regularizer = NoRegularizer()
    if early_stopping is None:
        early_stopping = NoEarlyStopping()
    if permuter is None:
        permuter = RandomPermuter()
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

    # Reset early stopping
    early_stopping.reset()

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
            batch = create_training_batch(
                X_shuffled, A_shuffled, batch_indices, batch_key, permuter=permuter
            )

            # Training step
            result = train_step(
                state,
                batch,
                discriminator_fn,
                optimizer,
                loss_fn=loss_fn,
                regularizer=regularizer,
                eps=eps,
            )
            state = result.state
            epoch_losses.append(float(result.loss))

        # Record epoch loss
        mean_epoch_loss = jnp.mean(jnp.array(epoch_losses))
        state.history["loss"].append(float(mean_epoch_loss))
        state.epoch = epoch + 1

        # Update early stopping
        early_stopping.update(float(mean_epoch_loss), state.params)

        # Check if should stop early
        if early_stopping.should_stop():
            # Restore best parameters if available
            best_params = early_stopping.get_best_params()
            if best_params is not None:
                state.params = best_params
            break

    return state.params, state.history
