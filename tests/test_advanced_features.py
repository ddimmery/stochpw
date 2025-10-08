"""Tests for advanced features: alternative loss functions, regularization, and early stopping."""

import jax
import jax.numpy as jnp
from stochpw import (
    MLPDiscriminator,
    PermutationWeighter,
    brier_loss,
    exponential_loss,
    l2_param_penalty,
)


class TestAlternativeLossFunctions:
    """Test alternative loss functions."""

    def test_exponential_loss_perfect_prediction(self):
        """Exponential loss should be minimal for perfect predictions."""
        logits = jnp.array([10.0, 10.0, -10.0, -10.0])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])
        loss = exponential_loss(logits, labels)
        # For perfect predictions, exp(-y*logits) should be very small
        assert loss < 0.01

    def test_exponential_loss_worst_prediction(self):
        """Exponential loss should be large for wrong predictions."""
        logits = jnp.array([-10.0, -10.0, 10.0, 10.0])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])
        loss = exponential_loss(logits, labels)
        # For wrong predictions, exp(-y*logits) should be very large
        assert loss > 1000.0

    def test_exponential_loss_gradient_exists(self):
        """Exponential loss should have computable gradients."""
        logits = jnp.array([1.0, -1.0])
        labels = jnp.array([1.0, 0.0])

        def loss_fn(x):
            return exponential_loss(x, labels)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(logits)

        assert jnp.all(jnp.isfinite(grads))
        assert grads.shape == logits.shape

    def test_brier_loss_perfect_prediction(self):
        """Brier loss should be zero for perfect probabilistic predictions."""
        # Perfect predictions: logits -> sigmoid(10) ≈ 1, sigmoid(-10) ≈ 0
        logits = jnp.array([10.0, 10.0, -10.0, -10.0])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])
        loss = brier_loss(logits, labels)
        assert loss < 0.001

    def test_brier_loss_worst_prediction(self):
        """Brier loss should be large for wrong predictions."""
        logits = jnp.array([-10.0, -10.0, 10.0, 10.0])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])
        loss = brier_loss(logits, labels)
        # Brier score is (1-0)^2 = 1 for each wrong prediction
        assert loss > 0.99

    def test_brier_loss_neutral_prediction(self):
        """Brier loss for neutral predictions (0.5 probability)."""
        logits = jnp.array([0.0, 0.0, 0.0, 0.0])
        labels = jnp.array([1.0, 1.0, 0.0, 0.0])
        loss = brier_loss(logits, labels)
        # Brier score is (0.5-1)^2 = 0.25 and (0.5-0)^2 = 0.25
        assert jnp.abs(loss - 0.25) < 0.01

    def test_brier_loss_gradient_exists(self):
        """Brier loss should have computable gradients."""
        logits = jnp.array([1.0, -1.0])
        labels = jnp.array([1.0, 0.0])

        def loss_fn(x):
            return brier_loss(x, labels)

        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(logits)

        assert jnp.all(jnp.isfinite(grads))
        assert grads.shape == logits.shape

    def test_exponential_loss_with_weighter(self):
        """Test PermutationWeighter with exponential loss."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 20)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 20)

        weighter = PermutationWeighter(
            loss_fn=exponential_loss,
            num_epochs=10,
            batch_size=16,
            random_state=42,
        )
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (80,)
        assert jnp.all(jnp.isfinite(weights))
        assert jnp.all(weights > 0)

    def test_brier_loss_with_weighter(self):
        """Test PermutationWeighter with Brier loss."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 20)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 20)

        weighter = PermutationWeighter(
            loss_fn=brier_loss,
            num_epochs=10,
            batch_size=16,
            random_state=42,
        )
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (80,)
        assert jnp.all(jnp.isfinite(weights))
        assert jnp.all(weights > 0)


class TestRegularization:
    """Test regularization functionality."""

    def test_l2_param_penalty_zero_for_zero_params(self):
        """L2 penalty should be zero for zero parameters."""
        params = {"w": jnp.zeros((10,)), "b": jnp.zeros((1,))}
        penalty = l2_param_penalty(params)
        assert jnp.abs(penalty) < 1e-6

    def test_l2_param_penalty_nonzero_for_nonzero_params(self):
        """L2 penalty should be nonzero for nonzero parameters."""
        params = {"w": jnp.ones((10,)), "b": jnp.ones((1,))}
        penalty = l2_param_penalty(params)
        # Sum of 11 ones squared = 11
        assert jnp.abs(penalty - 11.0) < 1e-4

    def test_l2_param_penalty_nested_params(self):
        """L2 penalty should work with nested parameter structures."""
        params = {
            "layer1": {"w": jnp.ones((5,)), "b": jnp.ones((2,))},
            "layer2": {"w": jnp.ones((3,)), "b": jnp.ones((1,))},
        }
        penalty = l2_param_penalty(params)
        # Sum of (5+2+3+1) ones squared = 11
        assert jnp.abs(penalty - 11.0) < 1e-4

    def test_l2_param_penalty_gradient_exists(self):
        """L2 penalty should have computable gradients."""
        params = {"w": jnp.array([1.0, 2.0, 3.0])}

        grad_fn = jax.grad(l2_param_penalty)
        grads = grad_fn(params)

        assert "w" in grads
        assert jnp.all(jnp.isfinite(grads["w"]))

    def test_regularization_reduces_parameter_magnitude(self):
        """Regularization should lead to smaller parameter values."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 20)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 20)

        # Without regularization
        weighter_no_reg = PermutationWeighter(
            num_epochs=20,
            batch_size=16,
            random_state=42,
        )
        weighter_no_reg.fit(X, A)
        params_no_reg_norm = l2_param_penalty(weighter_no_reg.params_)

        # With strong regularization
        weighter_with_reg = PermutationWeighter(
            regularization_fn=l2_param_penalty,
            regularization_strength=0.1,
            num_epochs=20,
            batch_size=16,
            random_state=42,
        )
        weighter_with_reg.fit(X, A)
        params_with_reg_norm = l2_param_penalty(weighter_with_reg.params_)

        # Regularization should reduce parameter magnitude
        assert params_with_reg_norm < params_no_reg_norm

    def test_regularization_with_mlp(self):
        """Test regularization works with MLP discriminator."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 20)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 20)

        mlp = MLPDiscriminator(hidden_dims=[32, 16])
        weighter = PermutationWeighter(
            discriminator=mlp,
            regularization_fn=l2_param_penalty,
            regularization_strength=0.01,
            num_epochs=10,
            batch_size=16,
            random_state=42,
        )
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (80,)
        assert jnp.all(jnp.isfinite(weights))


class TestEarlyStopping:
    """Test early stopping functionality."""

    def test_early_stopping_stops_before_max_epochs(self):
        """Early stopping should terminate before max epochs when improvement stops."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 20)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 20)

        weighter = PermutationWeighter(
            early_stopping=True,
            patience=5,
            min_delta=0.01,  # Require at least 0.01 improvement (higher threshold)
            num_epochs=100,  # Set high, but should stop early
            batch_size=16,
            random_state=42,
        )
        weighter.fit(X, A)

        # Should have stopped before 100 epochs due to min_delta threshold
        assert len(weighter.history_["loss"]) < 100
        assert weighter.params_ is not None

    def test_early_stopping_with_min_delta(self):
        """Early stopping should respect min_delta parameter."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 20)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 20)

        # Use a very high min_delta to trigger early stopping quickly
        weighter = PermutationWeighter(
            early_stopping=True,
            patience=3,
            min_delta=0.05,  # Require at least 0.05 improvement (very high)
            num_epochs=100,
            batch_size=16,
            random_state=42,
        )
        weighter.fit(X, A)

        # Should stop early due to high min_delta threshold
        assert len(weighter.history_["loss"]) < 100
        assert weighter.params_ is not None

    def test_without_early_stopping_runs_all_epochs(self):
        """Without early stopping, should run all epochs."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 20)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 20)

        num_epochs = 20
        weighter = PermutationWeighter(
            early_stopping=False,
            num_epochs=num_epochs,
            batch_size=16,
            random_state=42,
        )
        weighter.fit(X, A)

        # Should have run all epochs
        assert len(weighter.history_["loss"]) == num_epochs

    def test_early_stopping_preserves_best_params(self):
        """Early stopping should restore best parameters."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 20)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 20)

        weighter = PermutationWeighter(
            early_stopping=True,
            patience=5,
            num_epochs=100,
            batch_size=16,
            random_state=42,
        )
        weighter.fit(X, A)

        # Should have valid fitted parameters
        assert weighter.params_ is not None
        weights = weighter.predict(X, A)
        assert jnp.all(jnp.isfinite(weights))


class TestCombinedFeatures:
    """Test combinations of advanced features."""

    def test_all_features_together(self):
        """Test using loss function, regularization, and early stopping together."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 20)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 20)

        weighter = PermutationWeighter(
            loss_fn=brier_loss,
            regularization_fn=l2_param_penalty,
            regularization_strength=0.01,
            early_stopping=True,
            patience=5,
            num_epochs=100,
            batch_size=16,
            random_state=42,
        )
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (80,)
        assert jnp.all(jnp.isfinite(weights))
        assert jnp.all(weights > 0)
        assert len(weighter.history_["loss"]) < 100

    def test_mlp_with_all_features(self):
        """Test MLP discriminator with all advanced features."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 20)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 20)

        mlp = MLPDiscriminator(hidden_dims=[64, 32], activation="tanh")
        weighter = PermutationWeighter(
            discriminator=mlp,
            loss_fn=exponential_loss,
            regularization_fn=l2_param_penalty,
            regularization_strength=0.005,
            early_stopping=True,
            patience=10,
            num_epochs=100,
            batch_size=16,
            random_state=42,
        )
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (80,)
        assert jnp.all(jnp.isfinite(weights))
        assert jnp.all(weights > 0)
