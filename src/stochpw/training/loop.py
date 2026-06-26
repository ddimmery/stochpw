"""Training loop for permutation weighting discriminators."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from jax import Array

from ..data import TrainingBatch, TrainingState, TrainingStepResult
from ..types import OptimizerState, PyTree
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


def _build_step(
    discriminator_fn: Callable[[PyTree, Array, Array, Array], Array],
    optimizer: optax.GradientTransformation,
    loss_fn: BaseLoss,
    regularizer: BaseRegularizer,
    eps: float = 1e-7,
) -> Callable[[PyTree, OptimizerState, TrainingBatch], tuple[PyTree, OptimizerState, Array]]:
    """
    Build a single jit-compatible training step closure.

    The discriminator, optimizer, loss, and regularizer are captured by closure
    (held static) so the returned ``step`` traces only over ``(params, opt_state,
    batch)``. This is the building block shared by :func:`make_train_step` (which
    jits it for standalone use) and :func:`make_scan_epoch` (which inlines it
    inside a ``lax.scan``).

    Parameters
    ----------
    discriminator_fn : Callable
        Discriminator function ``(params, a, x, ax) -> logits``.
    optimizer : optax.GradientTransformation
        Optax optimizer.
    loss_fn : BaseLoss
        Loss function instance.
    regularizer : BaseRegularizer
        Regularization instance.
    eps : float, default=1e-7
        Numerical stability constant for weight computation.

    Returns
    -------
    step : Callable
        ``step(params, opt_state, batch) -> (params, opt_state, loss)`` with the
        loss kept on-device.
    """

    def step(
        params: PyTree, opt_state: OptimizerState, batch: TrainingBatch
    ) -> tuple[PyTree, OptimizerState, Array]:
        def loss_fn_total(p: PyTree) -> Array:
            logits = discriminator_fn(p, batch.A, batch.X, batch.AX)
            loss = loss_fn(logits, batch.C)

            # Observed rows are the static first half: create_training_batch
            # concatenates observed-then-permuted, so the observed (C=0) rows are
            # logits[:half]. A static slice is jit-safe, unlike the boolean-mask
            # indexing in train_step (which raises NonConcreteBooleanIndexError
            # under jit).
            half = batch.C.shape[0] // 2
            observed_logits = logits[:half]
            eta = jax.nn.sigmoid(observed_logits)
            eta_clipped = jnp.clip(eta, eps, 1 - eps)
            weights = eta_clipped / (1 - eta_clipped)

            return loss + regularizer(weights)

        loss, grads = jax.value_and_grad(loss_fn_total)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss  # type: ignore[return-value]  # loss on-device

    return step


def make_train_step(
    discriminator_fn: Callable[[PyTree, Array, Array, Array], Array],
    optimizer: optax.GradientTransformation,
    loss_fn: BaseLoss,
    regularizer: BaseRegularizer,
    eps: float = 1e-7,
) -> Callable[[PyTree, OptimizerState, TrainingBatch], tuple[PyTree, OptimizerState, Array]]:
    """
    Build a ``jax.jit``-compiled single training step.

    The returned step is compiled once and can be reused across batches of the
    same shape without re-tracing autodiff or re-dispatching primitives eagerly,
    which is the dominant cost of the un-jitted :func:`train_step`. The
    discriminator, optimizer, loss, and regularizer are held static (closed over);
    only ``(params, opt_state, batch)`` are traced.

    Parameters
    ----------
    discriminator_fn : Callable
        Discriminator function ``(params, a, x, ax) -> logits``.
    optimizer : optax.GradientTransformation
        Optax optimizer.
    loss_fn : BaseLoss
        Loss function instance.
    regularizer : BaseRegularizer
        Regularization instance.
    eps : float, default=1e-7
        Numerical stability constant for weight computation.

    Returns
    -------
    step : Callable
        Jitted ``step(params, opt_state, batch) -> (params, opt_state, loss)``
        with the loss kept on-device.
    """
    return jax.jit(_build_step(discriminator_fn, optimizer, loss_fn, regularizer, eps))


def make_scan_epoch(
    step: Callable[[PyTree, OptimizerState, TrainingBatch], tuple[PyTree, OptimizerState, Array]],
    permuter: BasePermuter,
    n: int,
    num_batches: int,
    batch_size: int,
) -> Callable[[PyTree, OptimizerState, Array, Array, Array], tuple[PyTree, OptimizerState, Array]]:
    """
    Build a ``jax.jit``-compiled single training epoch using ``jax.lax.scan``.

    Collapses the per-batch Python loop into one compiled scan, eliminating
    Python-level per-step dispatch. The data shape (``n``, ``num_batches``,
    ``batch_size``), the ``permuter``, and the ``step`` closure are held static.

    The PRNG sequence reproduces the un-jitted loop exactly: the shuffle consumes
    ``epoch_key`` directly, and the scan carry's initial key is *also*
    ``epoch_key``, with the body deriving each per-batch key via
    ``bk, k = jax.random.split(k)``. This left-fold matches the
    ``batch_key, epoch_key = jax.random.split(epoch_key)`` chain of the original
    loop, preserving numerical equivalence.

    Parameters
    ----------
    step : Callable
        Per-step closure ``(params, opt_state, batch) -> (params, opt_state,
        loss)`` (e.g. from :func:`_build_step`).
    permuter : BasePermuter
        Within-batch permutation strategy.
    n : int
        Number of samples.
    num_batches : int
        Batches per epoch (``n // batch_size``; the trailing partial batch is
        dropped, matching the original loop and the uniform-shape requirement of
        ``lax.scan``).
    batch_size : int
        Mini-batch size.

    Returns
    -------
    epoch_fn : Callable
        Jitted ``epoch_fn(params, opt_state, X, A, epoch_key) -> (params,
        opt_state, mean_loss)`` with the mean epoch loss kept on-device.
    """

    def epoch_fn(
        params: PyTree, opt_state: OptimizerState, X: Array, A: Array, epoch_key: Array
    ) -> tuple[PyTree, OptimizerState, Array]:
        # Shuffle once per epoch (consumes epoch_key directly, as in the original).
        perm = jax.random.permutation(epoch_key, n)
        X_shuffled = X[perm]
        A_shuffled = A[perm]

        # Per-step contiguous index slices into the shuffled data.
        idx = jnp.arange(num_batches * batch_size).reshape(num_batches, batch_size)

        def body(
            carry: tuple[PyTree, OptimizerState, Array], batch_indices: Array
        ) -> tuple[tuple[PyTree, OptimizerState, Array], Array]:
            p, opt_s, key = carry
            batch_key, key = jax.random.split(key)
            batch = create_training_batch(
                X_shuffled, A_shuffled, batch_indices, batch_key, permuter=permuter
            )
            p, opt_s, loss = step(p, opt_s, batch)
            return (p, opt_s, key), loss

        # Carry initial key is epoch_key itself (matches the original left-fold).
        (params, opt_state, _), losses = jax.lax.scan(body, (params, opt_state, epoch_key), idx)
        return params, opt_state, jnp.mean(losses)

    return jax.jit(epoch_fn)


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

    # Build the compiled per-step and per-epoch artifacts once. The epoch loop
    # stays in Python so early-stopping and history bookkeeping remain simple;
    # the inner batch loop is collapsed into a single jitted lax.scan (the perf
    # win). num_batches drops the trailing partial batch (n // batch_size), which
    # matches the original loop and the uniform-shape requirement of lax.scan.
    num_batches = n // batch_size
    step = _build_step(discriminator_fn, optimizer, loss_fn, regularizer, eps)
    epoch_fn = make_scan_epoch(step, permuter, n, num_batches, batch_size)

    for epoch in range(num_epochs):
        # Split RNG key for this epoch (unchanged from the original loop).
        epoch_key, state.rng_key = jax.random.split(state.rng_key)

        # Run one compiled epoch; the mean loss stays on-device until synced once
        # here (one host transfer per epoch instead of one per batch).
        params, opt_state, mean_epoch_loss = epoch_fn(
            state.params, state.opt_state, X, A, epoch_key
        )
        state.params = params
        state.opt_state = opt_state
        mean_epoch_loss_f = float(mean_epoch_loss)

        # Record epoch loss
        state.history["loss"].append(mean_epoch_loss_f)
        state.epoch = epoch + 1

        # Update early stopping
        early_stopping.update(mean_epoch_loss_f, state.params)

        # Check if should stop early
        if early_stopping.should_stop():
            # Restore best parameters if available
            best_params = early_stopping.get_best_params()
            if best_params is not None:
                state.params = best_params
            break

    return state.params, state.history
