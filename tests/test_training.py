"""Tests for stochpw.training module."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from stochpw.data import TrainingBatch, TrainingState
from stochpw.models import LinearDiscriminator, MLPDiscriminator
from stochpw.training import (
    BrierLoss,
    EarlyStopping,
    EntropyRegularizer,
    ExponentialLoss,
    LogisticLoss,
    LpRegularizer,
    NoEarlyStopping,
    NoRegularizer,
    RandomPermuter,
    create_training_batch,
    fit_discriminator,
    make_scan_epoch,
    make_train_step,
    train_step,
)


class TestCreateTrainingBatch:
    """Tests for create_training_batch function."""

    def test_batch_creation(self):
        """Test basic batch creation."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]])
        batch_indices = jnp.array([0, 1])
        key = jax.random.PRNGKey(42)

        batch = create_training_batch(X, A, batch_indices, key)

        assert isinstance(batch, TrainingBatch)
        assert batch.X.shape == (4, 2)  # 2 observed + 2 permuted
        assert batch.A.shape == (4, 1)
        assert batch.C.shape == (4,)
        assert batch.AX.shape == (4, 2)  # d_a * d_x = 1 * 2

    def test_labeling_correct(self):
        """Test that labels are correct (0=observed, 1=permuted)."""
        X = jnp.array([[1.0], [2.0]])
        A = jnp.array([[0.0], [1.0]])
        batch_indices = jnp.array([0, 1])
        key = jax.random.PRNGKey(0)

        batch = create_training_batch(X, A, batch_indices, key)

        # First half should be 0 (observed)
        assert jnp.all(batch.C[:2] == 0.0)
        # Second half should be 1 (permuted)
        assert jnp.all(batch.C[2:] == 1.0)

    def test_observed_data_unchanged(self):
        """Test that observed portion matches original data."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        A = jnp.array([[0.1], [0.5], [0.9]])
        batch_indices = jnp.array([0, 2])
        key = jax.random.PRNGKey(0)

        batch = create_training_batch(X, A, batch_indices, key)

        batch_size = len(batch_indices)
        # Observed X should match
        assert jnp.allclose(batch.X[:batch_size], X[batch_indices])
        # Observed A should match
        assert jnp.allclose(batch.A[:batch_size], A[batch_indices])

    def test_permuted_x_unchanged(self):
        """Test that permuted batch has same X as observed."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        A = jnp.array([[0.0], [1.0]])
        batch_indices = jnp.array([0, 1])
        key = jax.random.PRNGKey(0)

        batch = create_training_batch(X, A, batch_indices, key)

        batch_size = len(batch_indices)
        # Permuted X should equal observed X
        assert jnp.allclose(batch.X[:batch_size], batch.X[batch_size:])

    def test_permuted_a_shuffled(self):
        """Test that permuted A is shuffled within batch."""
        # Use distinct values to test shuffling
        X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        A = jnp.array([[0.1], [0.2], [0.3], [0.4]])
        batch_indices = jnp.array([0, 1, 2, 3])
        key = jax.random.PRNGKey(42)

        batch = create_training_batch(X, A, batch_indices, key)

        batch_size = len(batch_indices)
        A_obs = batch.A[:batch_size]
        A_perm = batch.A[batch_size:]

        # Permuted should be a reordering of observed (not identical order)
        # Check that sets are equal
        assert jnp.allclose(jnp.sort(A_obs.flatten()), jnp.sort(A_perm.flatten()))

    def test_interaction_computation(self):
        """Test that AX interactions are computed correctly."""
        X = jnp.array([[1.0, 2.0]])
        A = jnp.array([[3.0]])
        batch_indices = jnp.array([0])
        key = jax.random.PRNGKey(0)

        batch = create_training_batch(X, A, batch_indices, key)

        # Expected interactions for observed: [3*1, 3*2] = [3, 6]
        expected_ax = jnp.array([[3.0, 6.0]])
        assert jnp.allclose(batch.AX[0:1], expected_ax)

    def test_multivariate_treatment(self):
        """Test batch creation with multivariate treatment."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        A = jnp.array([[0.5, 0.3], [0.2, 0.8]])
        batch_indices = jnp.array([0, 1])
        key = jax.random.PRNGKey(0)

        batch = create_training_batch(X, A, batch_indices, key)

        assert batch.AX.shape == (4, 4)  # d_a * d_x = 2 * 2

    def test_different_seeds_different_permutations(self):
        """Test that different seeds produce different permutations."""
        X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        A = jnp.array([[0.1], [0.2], [0.3], [0.4]])
        batch_indices = jnp.array([0, 1, 2, 3])

        batch1 = create_training_batch(X, A, batch_indices, jax.random.PRNGKey(0))
        batch2 = create_training_batch(X, A, batch_indices, jax.random.PRNGKey(1))

        # Permuted portions should likely differ
        assert not jnp.allclose(batch1.A[4:], batch2.A[4:])


class TestLogisticLoss:
    """Tests for LogisticLoss class."""

    def test_perfect_prediction(self):
        """Test loss with perfect predictions."""
        loss_fn = LogisticLoss()
        logits = jnp.array([10.0, -10.0])
        labels = jnp.array([1.0, 0.0])

        loss = loss_fn(logits, labels)

        assert loss < 0.01  # Should be very small

    def test_worst_prediction(self):
        """Test loss with worst predictions."""
        loss_fn = LogisticLoss()
        logits = jnp.array([-10.0, 10.0])
        labels = jnp.array([1.0, 0.0])

        loss = loss_fn(logits, labels)

        assert loss > 5.0  # Should be large

    def test_neutral_prediction(self):
        """Test loss with neutral predictions (logits=0)."""
        loss_fn = LogisticLoss()
        logits = jnp.array([0.0, 0.0])
        labels = jnp.array([1.0, 0.0])

        loss = loss_fn(logits, labels)

        # log(2) ≈ 0.693
        assert jnp.isclose(loss, jnp.log(2.0), atol=0.01)

    def test_loss_positive(self):
        """Test that loss is always positive."""
        loss_fn = LogisticLoss()
        logits = jax.random.normal(jax.random.PRNGKey(0), (100,))
        labels = jax.random.bernoulli(jax.random.PRNGKey(1), 0.5, (100,))

        loss = loss_fn(logits, labels)

        assert loss >= 0.0

    def test_gradient_exists(self):
        """Test that gradient can be computed."""
        loss_fn = LogisticLoss()
        logits = jnp.array([0.5, -0.5])
        labels = jnp.array([1.0, 0.0])

        grad_fn = jax.grad(lambda logit: loss_fn(logit, labels))
        grads = grad_fn(logits)

        assert grads.shape == logits.shape
        assert jnp.all(jnp.isfinite(grads))


class TestTrainStep:
    """Tests for train_step function."""

    def test_train_step_execution(self):
        """Test that train_step executes without error."""
        from stochpw.training import LogisticLoss, NoRegularizer

        d_a, d_x = 1, 2
        discriminator = LinearDiscriminator()

        def init_fn(key):
            return discriminator.init_params(key, d_a, d_x)

        apply_fn = discriminator.apply

        key = jax.random.PRNGKey(42)
        params = init_fn(key)

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        state = TrainingState(
            params=params, opt_state=opt_state, rng_key=key, epoch=0, history={"loss": []}
        )

        # Create batch
        batch = TrainingBatch(
            X=jnp.ones((4, d_x)),
            A=jnp.ones((4, d_a)),
            C=jnp.array([0.0, 0.0, 1.0, 1.0]),
            AX=jnp.ones((4, d_a * d_x)),
        )

        loss_fn = LogisticLoss()
        regularizer = NoRegularizer()

        result = train_step(state, batch, apply_fn, optimizer, loss_fn, regularizer)

        assert result.loss >= 0.0
        assert result.state.params is not None

    def test_parameters_updated(self):
        """Test that parameters are updated after train_step."""
        from stochpw.training import LogisticLoss, NoRegularizer

        d_a, d_x = 1, 2
        discriminator = LinearDiscriminator()

        def init_fn(key):
            return discriminator.init_params(key, d_a, d_x)

        apply_fn = discriminator.apply

        key = jax.random.PRNGKey(42)
        params = init_fn(key)
        original_params = {k: v.copy() for k, v in params.items()}

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        state = TrainingState(params=params, opt_state=opt_state, rng_key=key, epoch=0, history={})

        batch = TrainingBatch(
            X=jnp.array([[1.0, 2.0]]),
            A=jnp.array([[1.0]]),
            C=jnp.array([1.0]),
            AX=jnp.array([[1.0, 2.0]]),
        )

        loss_fn = LogisticLoss()
        regularizer = NoRegularizer()

        result = train_step(state, batch, apply_fn, optimizer, loss_fn, regularizer)

        # Parameters should have changed
        assert not jnp.allclose(result.state.params["w_a"], original_params["w_a"])


class TestFitDiscriminator:
    """Tests for fit_discriminator function."""

    def test_fit_execution(self):
        """Test that fit_discriminator executes without error."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]])

        d_a, d_x = A.shape[1], X.shape[1]
        discriminator = LinearDiscriminator()

        def init_fn(key):
            return discriminator.init_params(key, d_a, d_x)

        apply_fn = discriminator.apply

        key = jax.random.PRNGKey(42)
        params = init_fn(key)
        optimizer = optax.adam(1e-3)

        final_params, history = fit_discriminator(
            X=X,
            A=A,
            discriminator_fn=apply_fn,
            init_params=params,
            optimizer=optimizer,
            num_epochs=5,
            batch_size=2,
            rng_key=key,
        )

        assert final_params is not None
        assert "loss" in history
        assert len(history["loss"]) == 5

    def test_loss_decreases(self):
        """Test that loss generally decreases during training."""
        # Create synthetic data with clear pattern
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (100, 5))
        A = jax.random.bernoulli(jax.random.PRNGKey(1), 0.5, (100,)).astype(float).reshape(-1, 1)

        d_a, d_x = A.shape[1], X.shape[1]
        discriminator = LinearDiscriminator()

        def init_fn(key):
            return discriminator.init_params(key, d_a, d_x)

        apply_fn = discriminator.apply

        params = init_fn(jax.random.PRNGKey(2))
        optimizer = optax.adam(1e-2)

        final_params, history = fit_discriminator(
            X=X,
            A=A,
            discriminator_fn=apply_fn,
            init_params=params,
            optimizer=optimizer,
            num_epochs=50,
            batch_size=32,
            rng_key=jax.random.PRNGKey(3),
        )

        # Loss should decrease
        assert history["loss"][-1] < history["loss"][0]

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        A = jnp.array([[0.0], [1.0]])

        d_a, d_x = A.shape[1], X.shape[1]
        discriminator = LinearDiscriminator()

        def init_fn(key):
            return discriminator.init_params(key, d_a, d_x)

        apply_fn = discriminator.apply

        key = jax.random.PRNGKey(42)

        params1 = init_fn(key)
        optimizer1 = optax.adam(1e-3)
        final_params1, history1 = fit_discriminator(
            X, A, apply_fn, params1, optimizer1, num_epochs=3, batch_size=2, rng_key=key
        )

        params2 = init_fn(key)
        optimizer2 = optax.adam(1e-3)
        final_params2, history2 = fit_discriminator(
            X, A, apply_fn, params2, optimizer2, num_epochs=3, batch_size=2, rng_key=key
        )

        # Results should be identical
        assert jnp.allclose(final_params1["w_a"], final_params2["w_a"])
        assert jnp.allclose(history1["loss"][-1], history2["loss"][-1])


def _fit_discriminator_reference(
    X,
    A,
    discriminator_fn,
    init_params,
    optimizer,
    num_epochs,
    batch_size,
    rng_key,
    loss_fn=None,
    regularizer=None,
    early_stopping=None,
    permuter=None,
    eps=1e-7,
):
    """Verbatim copy of the pre-jit ``fit_discriminator`` body.

    This is the reference implementation against which the jitted/scanned
    ``fit_discriminator`` must stay numerically equivalent. It drives the
    (unchanged) public ``train_step`` per batch with an eager Python loop and a
    per-step host sync, exactly as the original loop did. Do not "optimize" this
    function — it exists solely to pin down the original numerics.
    """
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

    state = TrainingState(
        params=init_params,
        opt_state=opt_state,
        rng_key=rng_key,
        epoch=0,
        history={"loss": []},
    )

    early_stopping.reset()

    for epoch in range(num_epochs):
        epoch_key, state.rng_key = jax.random.split(state.rng_key)

        perm = jax.random.permutation(epoch_key, n)
        X_shuffled = X[perm]
        A_shuffled = A[perm]

        epoch_losses = []
        num_batches = n // batch_size

        for i in range(num_batches):
            batch_key, epoch_key = jax.random.split(epoch_key)

            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_indices = jnp.arange(start_idx, end_idx)

            batch = create_training_batch(
                X_shuffled, A_shuffled, batch_indices, batch_key, permuter=permuter
            )

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

        mean_epoch_loss = jnp.mean(jnp.array(epoch_losses))
        state.history["loss"].append(float(mean_epoch_loss))
        state.epoch = epoch + 1

        early_stopping.update(float(mean_epoch_loss), state.params)

        if early_stopping.should_stop():
            best_params = early_stopping.get_best_params()
            if best_params is not None:
                state.params = best_params
            break

    return state.params, state.history


def _make_equiv_data(n=120, d_x=5, d_a=1, seed=0):
    """Confounded synthetic data with treatment-covariate signal to learn."""
    key = jax.random.PRNGKey(seed)
    x_key, a_key = jax.random.split(key)
    X = jax.random.normal(x_key, (n, d_x))
    coef = jnp.linspace(0.5, -0.3, num=min(d_x, 3))
    logits = X[:, : coef.shape[0]] @ coef + 0.2
    propensity = jax.nn.sigmoid(logits)
    A = jax.random.bernoulli(a_key, propensity, (n,)).astype(jnp.float32).reshape(-1, 1)
    return X, A


class TestJitEquivalence:
    """The jitted/scanned loop must match the pre-jit loop within fp tolerance."""

    # (discriminator factory, loss, regularizer) combinations to verify.
    CASES = [
        ("linear-logistic-none", LinearDiscriminator, LogisticLoss(), NoRegularizer()),
        (
            "linear-logistic-lp",
            LinearDiscriminator,
            LogisticLoss(),
            LpRegularizer(p=2.0, strength=0.01),
        ),
        ("linear-logistic-entropy", LinearDiscriminator, LogisticLoss(), EntropyRegularizer()),
        ("linear-exponential", LinearDiscriminator, ExponentialLoss(), NoRegularizer()),
        ("linear-brier", LinearDiscriminator, BrierLoss(), NoRegularizer()),
        (
            "mlp-logistic-none",
            lambda: MLPDiscriminator(hidden_dims=[16, 8]),
            LogisticLoss(),
            NoRegularizer(),
        ),
        (
            "mlp-logistic-lp",
            lambda: MLPDiscriminator(hidden_dims=[16, 8]),
            LogisticLoss(),
            LpRegularizer(p=2.0, strength=0.01),
        ),
    ]

    @pytest.mark.parametrize("name,disc_factory,loss,regularizer", CASES, ids=[c[0] for c in CASES])
    def test_matches_reference(self, name, disc_factory, loss, regularizer):
        """Fitted params and per-epoch loss match the pre-jit reference."""
        X, A = _make_equiv_data()
        d_a, d_x = A.shape[1], X.shape[1]
        disc = disc_factory()
        init_params = disc.init_params(jax.random.PRNGKey(0), d_a, d_x)
        train_key = jax.random.PRNGKey(123)

        # Two fresh optimizers (optax state is stateless given init, but be safe).
        ref_params, ref_hist = _fit_discriminator_reference(
            X,
            A,
            disc.apply,
            init_params,
            optax.adam(1e-2),
            num_epochs=8,
            batch_size=32,
            rng_key=train_key,
            loss_fn=loss,
            regularizer=regularizer,
        )
        new_params, new_hist = fit_discriminator(
            X,
            A,
            disc.apply,
            init_params,
            optax.adam(1e-2),
            num_epochs=8,
            batch_size=32,
            rng_key=train_key,
            loss_fn=loss,
            regularizer=regularizer,
        )

        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6),
            new_params,
            ref_params,
        )
        np.testing.assert_allclose(
            np.array(new_hist["loss"]), np.array(ref_hist["loss"]), rtol=1e-5, atol=1e-6
        )

    def test_early_stopping_equivalence(self):
        """Early stopping (restoring best params) matches the reference."""
        X, A = _make_equiv_data()
        d_a, d_x = A.shape[1], X.shape[1]
        disc = LinearDiscriminator()
        init_params = disc.init_params(jax.random.PRNGKey(1), d_a, d_x)
        train_key = jax.random.PRNGKey(7)

        ref_params, ref_hist = _fit_discriminator_reference(
            X,
            A,
            disc.apply,
            init_params,
            optax.adam(1e-2),
            num_epochs=50,
            batch_size=32,
            rng_key=train_key,
            early_stopping=EarlyStopping(patience=3, min_delta=1e-3),
        )
        new_params, new_hist = fit_discriminator(
            X,
            A,
            disc.apply,
            init_params,
            optax.adam(1e-2),
            num_epochs=50,
            batch_size=32,
            rng_key=train_key,
            early_stopping=EarlyStopping(patience=3, min_delta=1e-3),
        )

        # Same number of epochs run (early stop fired at the same point).
        assert len(new_hist["loss"]) == len(ref_hist["loss"])
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6),
            new_params,
            ref_params,
        )


class TestMakeTrainStep:
    """Tests for the standalone jitted step factory."""

    def test_make_train_step_matches_train_step(self):
        """One jitted step equals one un-jitted train_step (same numerics)."""
        X, A = _make_equiv_data(n=64)
        disc = LinearDiscriminator()
        d_a, d_x = A.shape[1], X.shape[1]
        params = disc.init_params(jax.random.PRNGKey(0), d_a, d_x)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        batch = create_training_batch(
            X, A, jnp.arange(0, 32), jax.random.PRNGKey(5), permuter=RandomPermuter()
        )

        # New jitted step.
        step = make_train_step(disc.apply, optimizer, LogisticLoss(), NoRegularizer())
        new_params, new_opt, new_loss = step(params, opt_state, batch)

        # Reference via the public train_step.
        state = TrainingState(
            params=params,
            opt_state=opt_state,
            rng_key=jax.random.PRNGKey(5),
            epoch=0,
            history={"loss": []},
        )
        result = train_step(
            state,
            batch,
            disc.apply,
            optimizer,
            loss_fn=LogisticLoss(),
            regularizer=NoRegularizer(),
        )

        np.testing.assert_allclose(float(new_loss), float(result.loss), rtol=1e-5, atol=1e-6)
        jax.tree.map(
            lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6),
            new_params,
            result.state.params,
        )

    def test_regularized_step_compiles_no_boolean_index_error(self):
        """The jitted step traces with a regularizer (static slice, not mask)."""
        X, A = _make_equiv_data(n=64)
        disc = LinearDiscriminator()
        d_a, d_x = A.shape[1], X.shape[1]
        params = disc.init_params(jax.random.PRNGKey(0), d_a, d_x)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        batch = create_training_batch(
            X, A, jnp.arange(0, 32), jax.random.PRNGKey(5), permuter=RandomPermuter()
        )
        step = make_train_step(
            disc.apply, optimizer, LogisticLoss(), LpRegularizer(p=2.0, strength=0.01)
        )
        # Force tracing/compilation; must not raise NonConcreteBooleanIndexError.
        new_params, new_opt, loss = step(params, opt_state, batch)
        jax.block_until_ready((new_params, loss))
        assert jnp.isfinite(loss)

    def test_make_scan_epoch_runs_one_epoch(self):
        """make_scan_epoch produces a usable jitted epoch function."""
        X, A = _make_equiv_data(n=64)
        disc = LinearDiscriminator()
        d_a, d_x = A.shape[1], X.shape[1]
        params = disc.init_params(jax.random.PRNGKey(0), d_a, d_x)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)
        step = make_train_step(disc.apply, optimizer, LogisticLoss(), NoRegularizer())
        epoch_fn = make_scan_epoch(step, RandomPermuter(), n=64, num_batches=2, batch_size=32)
        new_params, new_opt, mean_loss = epoch_fn(params, opt_state, X, A, jax.random.PRNGKey(9))
        assert jnp.isfinite(mean_loss)
        assert set(new_params.keys()) == set(params.keys())
