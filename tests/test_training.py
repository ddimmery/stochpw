"""Tests for stochpw.training module."""

import jax
import jax.numpy as jnp
import optax
from stochpw.data import TrainingBatch, TrainingState
from stochpw.models import create_linear_discriminator
from stochpw.training import create_training_batch, fit_discriminator, logistic_loss, train_step


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
    """Tests for logistic_loss function."""

    def test_perfect_prediction(self):
        """Test loss with perfect predictions."""
        logits = jnp.array([10.0, -10.0])
        labels = jnp.array([1.0, 0.0])

        loss = logistic_loss(logits, labels)

        assert loss < 0.01  # Should be very small

    def test_worst_prediction(self):
        """Test loss with worst predictions."""
        logits = jnp.array([-10.0, 10.0])
        labels = jnp.array([1.0, 0.0])

        loss = logistic_loss(logits, labels)

        assert loss > 5.0  # Should be large

    def test_neutral_prediction(self):
        """Test loss with neutral predictions (logits=0)."""
        logits = jnp.array([0.0, 0.0])
        labels = jnp.array([1.0, 0.0])

        loss = logistic_loss(logits, labels)

        # log(2) ≈ 0.693
        assert jnp.isclose(loss, jnp.log(2.0), atol=0.01)

    def test_loss_positive(self):
        """Test that loss is always positive."""
        logits = jax.random.normal(jax.random.PRNGKey(0), (100,))
        labels = jax.random.bernoulli(jax.random.PRNGKey(1), 0.5, (100,))

        loss = logistic_loss(logits, labels)

        assert loss >= 0.0

    def test_gradient_exists(self):
        """Test that gradient can be computed."""
        logits = jnp.array([0.5, -0.5])
        labels = jnp.array([1.0, 0.0])

        grad_fn = jax.grad(lambda logit: logistic_loss(logit, labels))
        grads = grad_fn(logits)

        assert grads.shape == logits.shape
        assert jnp.all(jnp.isfinite(grads))


class TestTrainStep:
    """Tests for train_step function."""

    def test_train_step_execution(self):
        """Test that train_step executes without error."""
        d_a, d_x = 1, 2
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

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

        result = train_step(state, batch, apply_fn, optimizer)

        assert result.loss >= 0.0
        assert result.state.params is not None

    def test_parameters_updated(self):
        """Test that parameters are updated after train_step."""
        d_a, d_x = 1, 2
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

        key = jax.random.PRNGKey(42)
        params = init_fn(key)
        original_params = {k: v.copy() for k, v in params.items()}

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        state = TrainingState(
            params=params, opt_state=opt_state, rng_key=key, epoch=0, history={}
        )

        batch = TrainingBatch(
            X=jnp.array([[1.0, 2.0]]),
            A=jnp.array([[1.0]]),
            C=jnp.array([1.0]),
            AX=jnp.array([[1.0, 2.0]]),
        )

        result = train_step(state, batch, apply_fn, optimizer)

        # Parameters should have changed
        assert not jnp.allclose(result.state.params["w_a"], original_params["w_a"])


class TestFitDiscriminator:
    """Tests for fit_discriminator function."""

    def test_fit_execution(self):
        """Test that fit_discriminator executes without error."""
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]])

        d_a, d_x = A.shape[1], X.shape[1]
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

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
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

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
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

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
