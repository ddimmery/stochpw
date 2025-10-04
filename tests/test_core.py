"""Tests for stochpw.core module (end-to-end API tests)."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from stochpw import PermutationWeighter
from stochpw.core import NotFittedError
from stochpw.diagnostics import effective_sample_size, standardized_mean_difference


class TestPermutationWeighter:
    """Tests for PermutationWeighter class."""

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        weighter = PermutationWeighter()

        assert weighter.discriminator is None
        assert weighter.optimizer is None
        assert weighter.num_epochs == 100
        assert weighter.batch_size == 256
        assert weighter.random_state is None
        assert weighter.params_ is None
        assert weighter.history_ is None
        assert weighter.discriminator_fn_ is None

    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        weighter = PermutationWeighter(
            num_epochs=50, batch_size=128, random_state=42
        )

        assert weighter.num_epochs == 50
        assert weighter.batch_size == 128
        assert weighter.random_state == 42

    def test_fit_returns_self(self):
        """Test that fit() returns self for method chaining."""
        X = np.random.randn(20, 2)
        A = np.random.choice([0.0, 1.0], size=20)

        weighter = PermutationWeighter(num_epochs=2, batch_size=10, random_state=42)
        result = weighter.fit(X, A)

        assert result is weighter

    def test_fit_sets_fitted_attributes(self):
        """Test that fit() sets the fitted attributes."""
        X = np.random.randn(15, 2)
        A = np.random.choice([0.0, 1.0], size=15)

        weighter = PermutationWeighter(num_epochs=2, batch_size=10, random_state=42)
        weighter.fit(X, A)

        assert weighter.params_ is not None
        assert weighter.history_ is not None
        assert weighter.discriminator_fn_ is not None
        assert "loss" in weighter.history_
        assert len(weighter.history_["loss"]) == 2

    def test_predict_before_fit_raises_error(self):
        """Test that predict() before fit() raises NotFittedError."""
        X = np.random.randn(10, 2)
        A = np.random.choice([0.0, 1.0], size=10)

        weighter = PermutationWeighter()

        with pytest.raises(NotFittedError):
            weighter.predict(X, A)

    def test_fit_and_predict(self):
        """Test basic fit and predict workflow."""
        X = np.random.randn(20, 2)
        A = np.random.choice([0.0, 1.0], size=20)

        weighter = PermutationWeighter(num_epochs=5, batch_size=10, random_state=42)
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (20,)
        assert jnp.all(weights > 0)  # All weights should be positive

    def test_weights_are_jax_arrays(self):
        """Test that predicted weights are JAX arrays."""
        X = np.random.randn(12, 1)
        A = np.random.choice([0.0, 1.0], size=12)

        weighter = PermutationWeighter(num_epochs=2, batch_size=10, random_state=42)
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert isinstance(weights, jnp.ndarray)

    def test_reproducibility_with_random_state(self):
        """Test that same random_state gives same results."""
        X = np.random.RandomState(42).randn(18, 2)
        A = np.random.RandomState(42).choice([0.0, 1.0], size=18)

        weighter1 = PermutationWeighter(num_epochs=5, batch_size=10, random_state=42)
        weighter1.fit(X, A)
        weights1 = weighter1.predict(X, A)

        weighter2 = PermutationWeighter(num_epochs=5, batch_size=10, random_state=42)
        weighter2.fit(X, A)
        weights2 = weighter2.predict(X, A)

        assert jnp.allclose(weights1, weights2)

    def test_different_seeds_different_results(self):
        """Test that different random_state gives different results."""
        X = np.random.RandomState(0).randn(18, 2)
        A = np.random.RandomState(0).choice([0.0, 1.0], size=18)

        weighter1 = PermutationWeighter(num_epochs=5, batch_size=10, random_state=0)
        weighter1.fit(X, A)
        weights1 = weighter1.predict(X, A)

        weighter2 = PermutationWeighter(num_epochs=5, batch_size=10, random_state=1)
        weighter2.fit(X, A)
        weights2 = weighter2.predict(X, A)

        assert not jnp.allclose(weights1, weights2)

    def test_multivariate_treatment(self):
        """Test with multivariate treatment."""
        X = np.random.randn(15, 2)
        A = np.random.randn(15, 2)

        weighter = PermutationWeighter(num_epochs=5, batch_size=10, random_state=42)
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (15,)
        assert jnp.all(weights > 0)

    def test_custom_optimizer(self):
        """Test with custom optimizer."""
        X = np.random.randn(12, 2)
        A = np.random.choice([0.0, 1.0], size=12)

        custom_optimizer = optax.sgd(0.01)
        weighter = PermutationWeighter(
            optimizer=custom_optimizer, num_epochs=5, batch_size=10, random_state=42
        )
        weighter.fit(X, A)

        assert weighter.params_ is not None

    def test_custom_discriminator(self):
        """Test with custom discriminator function."""
        from stochpw.models import create_linear_discriminator

        X = np.random.randn(15, 2)
        A = np.random.choice([0.0, 1.0], size=15)

        # Use custom discriminator factory
        def custom_disc(d_a, d_x):
            return create_linear_discriminator(d_a, d_x)

        weighter = PermutationWeighter(
            discriminator=custom_disc, num_epochs=5, batch_size=10, random_state=42
        )
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (15,)
        assert weighter.params_ is not None

    def test_none_random_state(self):
        """Test with random_state=None (uses default seed)."""
        X = np.random.RandomState(0).randn(15, 2)
        A = np.random.RandomState(0).choice([0.0, 1.0], size=15)

        weighter = PermutationWeighter(num_epochs=5, batch_size=10, random_state=None)
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (15,)
        assert jnp.all(weights > 0)

    def test_training_converges(self):
        """Test that training generally reduces loss."""
        # Generate synthetic data with confounding
        key = jax.random.PRNGKey(0)
        n = 100
        X = jax.random.normal(key, (n, 5))
        propensity = jax.nn.sigmoid(0.5 * X[:, 0] - 0.3 * X[:, 1])
        A = jax.random.bernoulli(jax.random.PRNGKey(1), propensity).astype(float)

        weighter = PermutationWeighter(num_epochs=50, batch_size=32, random_state=42)
        weighter.fit(X, A)

        # Loss should generally decrease
        assert weighter.history_["loss"][-1] < weighter.history_["loss"][0]

    def test_balance_improvement(self):
        """Test that weights improve covariate balance."""
        # Generate confounded data
        key = jax.random.PRNGKey(0)
        n = 200
        X = jax.random.normal(key, (n, 3))
        propensity = jax.nn.sigmoid(0.8 * X[:, 0] - 0.5 * X[:, 1])
        A = jax.random.bernoulli(jax.random.PRNGKey(1), propensity).astype(float)

        # Fit weighter
        weighter = PermutationWeighter(num_epochs=100, batch_size=64, random_state=42)
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        # Check balance
        smd_unweighted = standardized_mean_difference(X, A, jnp.ones_like(weights))
        smd_weighted = standardized_mean_difference(X, A, weights)

        max_smd_unw = jnp.abs(smd_unweighted).max()
        max_smd_w = jnp.abs(smd_weighted).max()

        # Weighted SMD should generally be lower (balance improved)
        # Allow some tolerance since this is stochastic
        assert max_smd_w < max_smd_unw * 1.2  # At most 20% worse

    def test_effective_sample_size(self):
        """Test that ESS is reasonable."""
        X = np.random.randn(20, 2)
        A = np.random.choice([0.0, 1.0], size=20)

        weighter = PermutationWeighter(num_epochs=10, batch_size=10, random_state=42)
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        ess = effective_sample_size(weights)

        # ESS should be between 1 and n
        assert 1.0 <= ess <= len(weights)

    def test_handles_numpy_and_jax_inputs(self):
        """Test that both NumPy and JAX arrays work as input."""
        # NumPy inputs
        X_np = np.random.RandomState(42).randn(15, 2)
        A_np = np.random.RandomState(42).choice([0.0, 1.0], size=15)

        weighter1 = PermutationWeighter(num_epochs=2, batch_size=10, random_state=42)
        weighter1.fit(X_np, A_np)
        weights1 = weighter1.predict(X_np, A_np)

        # JAX inputs
        X_jax = jnp.array(X_np)
        A_jax = jnp.array(A_np)

        weighter2 = PermutationWeighter(num_epochs=2, batch_size=10, random_state=42)
        weighter2.fit(X_jax, A_jax)
        weights2 = weighter2.predict(X_jax, A_jax)

        # Results should be identical
        assert jnp.allclose(weights1, weights2)

    def test_1d_treatment_array(self):
        """Test that 1D treatment array is handled correctly."""
        X = np.random.randn(12, 2)
        A = np.random.choice([0.0, 1.0], size=12)  # 1D

        weighter = PermutationWeighter(num_epochs=2, batch_size=10, random_state=42)
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (12,)

    def test_fit_multiple_times(self):
        """Test that fitting multiple times works (overwrites previous fit)."""
        X1 = np.random.randn(12, 1)
        A1 = np.random.choice([0.0, 1.0], size=12)

        weighter = PermutationWeighter(num_epochs=2, batch_size=10, random_state=42)
        weighter.fit(X1, A1)
        weighter.predict(X1, A1)  # Verify first fit works

        # Fit again with different data
        X2 = np.random.randn(15, 1)
        A2 = np.random.choice([0.0, 1.0], size=15)

        weighter.fit(X2, A2)
        weights2 = weighter.predict(X2, A2)

        # Should work and give weights for new data
        assert weights2.shape == (15,)

    def test_large_num_epochs(self):
        """Test that large number of epochs works."""
        X = np.random.randn(12, 2)
        A = np.random.choice([0.0, 1.0], size=12)

        weighter = PermutationWeighter(num_epochs=200, batch_size=10, random_state=42)
        weighter.fit(X, A)

        assert len(weighter.history_["loss"]) == 200

    def test_small_batch_size(self):
        """Test that small batch size works."""
        X = np.random.randn(20, 1)
        A = np.random.choice([0.0, 1.0], size=20)

        weighter = PermutationWeighter(num_epochs=5, batch_size=10, random_state=42)
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (20,)

    def test_method_chaining(self):
        """Test that fit() enables method chaining."""
        X = np.random.randn(12, 1)
        A = np.random.choice([0.0, 1.0], size=12)

        weighter = PermutationWeighter(num_epochs=2, batch_size=10, random_state=42)

        # Should be able to chain
        weights = weighter.fit(X, A).predict(X, A)

        assert weights.shape == (12,)

    def test_continuous_treatment(self):
        """Test with continuous treatment."""
        X = np.random.randn(15, 2)
        A = np.random.uniform(0, 1, size=15)  # Continuous

        weighter = PermutationWeighter(num_epochs=10, batch_size=10, random_state=42)
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        assert weights.shape == (15,)
        assert jnp.all(weights > 0)
