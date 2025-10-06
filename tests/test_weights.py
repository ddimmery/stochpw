"""Tests for stochpw.weights module."""

import jax
import jax.numpy as jnp
from stochpw.models import LinearDiscriminator
from stochpw.weights import extract_weights


class TestExtractWeights:
    """Tests for extract_weights function."""

    def test_weight_extraction(self):
        """Test basic weight extraction."""
        d_a, d_x = 1, 2
        discriminator = LinearDiscriminator()
        init_fn = lambda key: discriminator.init_params(key, d_a, d_x)
        apply_fn = discriminator.apply

        key = jax.random.PRNGKey(42)
        params = init_fn(key)

        X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        A = jnp.array([[0.0], [1.0]])
        AX = jnp.einsum("bi,bj->bij", A, X).reshape(2, -1)

        weights = extract_weights(apply_fn, params, X, A, AX)

        assert weights.shape == (2,)
        assert jnp.all(weights > 0)  # Weights should be positive

    def test_weight_formula(self):
        """Test that weight formula w = η/(1-η) is applied correctly."""
        # Create discriminator with known parameters
        def simple_apply(params, a, x, ax):
            # Return fixed logits
            return params["logit"] * jnp.ones(a.shape[0])

        params = {"logit": jnp.array(0.0)}  # logit=0 -> η=0.5 -> w=1.0

        X = jnp.array([[1.0]])
        A = jnp.array([[1.0]])
        AX = jnp.array([[1.0]])

        weights = extract_weights(simple_apply, params, X, A, AX)

        # η = sigmoid(0) = 0.5, w = 0.5 / (1 - 0.5) = 1.0
        assert jnp.isclose(weights[0], 1.0, atol=1e-6)

    def test_high_eta_high_weight(self):
        """Test that high η (permuted-looking) gives high weight."""
        def simple_apply(params, a, x, ax):
            return params["logit"] * jnp.ones(a.shape[0])

        # High positive logit -> high η -> high weight
        params = {"logit": jnp.array(5.0)}

        X = jnp.array([[1.0]])
        A = jnp.array([[1.0]])
        AX = jnp.array([[1.0]])

        weights = extract_weights(simple_apply, params, X, A, AX)

        # η ≈ sigmoid(5) ≈ 0.993, w = 0.993/(1-0.993) ≈ 141
        assert weights[0] > 10.0  # Should be large

    def test_low_eta_low_weight(self):
        """Test that low η (observed-looking) gives low weight."""
        def simple_apply(params, a, x, ax):
            return params["logit"] * jnp.ones(a.shape[0])

        # High negative logit -> low η -> low weight
        params = {"logit": jnp.array(-5.0)}

        X = jnp.array([[1.0]])
        A = jnp.array([[1.0]])
        AX = jnp.array([[1.0]])

        weights = extract_weights(simple_apply, params, X, A, AX)

        # η ≈ sigmoid(-5) ≈ 0.007, w = 0.007/(1-0.007) ≈ 0.007
        assert weights[0] < 0.1  # Should be small

    def test_numerical_stability_high_eta(self):
        """Test numerical stability when η close to 1."""
        def simple_apply(params, a, x, ax):
            return params["logit"] * jnp.ones(a.shape[0])

        # Very high logit -> η very close to 1
        params = {"logit": jnp.array(20.0)}

        X = jnp.array([[1.0]])
        A = jnp.array([[1.0]])
        AX = jnp.array([[1.0]])

        weights = extract_weights(simple_apply, params, X, A, AX)

        # Should not be inf or nan
        assert jnp.isfinite(weights[0])
        assert weights[0] > 0

    def test_numerical_stability_low_eta(self):
        """Test numerical stability when η close to 0."""
        def simple_apply(params, a, x, ax):
            return params["logit"] * jnp.ones(a.shape[0])

        # Very low logit -> η very close to 0
        params = {"logit": jnp.array(-20.0)}

        X = jnp.array([[1.0]])
        A = jnp.array([[1.0]])
        AX = jnp.array([[1.0]])

        weights = extract_weights(simple_apply, params, X, A, AX)

        # Should not be zero (due to clipping)
        assert jnp.isfinite(weights[0])
        assert weights[0] > 0

    def test_batch_processing(self):
        """Test that weights are computed independently for each sample."""
        def simple_apply(params, a, x, ax):
            # Different logits for each sample
            return jnp.array([0.0, 2.0, -2.0])

        params = {}

        X = jnp.ones((3, 2))
        A = jnp.ones((3, 1))
        AX = jnp.ones((3, 2))

        weights = extract_weights(simple_apply, params, X, A, AX)

        assert weights.shape == (3,)
        # Weights should be different
        assert not jnp.allclose(weights[0], weights[1])
        assert not jnp.allclose(weights[1], weights[2])

    def test_custom_eps(self):
        """Test that custom epsilon value works."""
        def simple_apply(params, a, x, ax):
            return jnp.array([100.0])  # Very high -> η ≈ 1

        params = {}
        X = jnp.array([[1.0]])
        A = jnp.array([[1.0]])
        AX = jnp.array([[1.0]])

        weights_default = extract_weights(simple_apply, params, X, A, AX)
        weights_custom = extract_weights(simple_apply, params, X, A, AX, eps=1e-5)

        # Both should be finite
        assert jnp.isfinite(weights_default[0])
        assert jnp.isfinite(weights_custom[0])

    def test_gradient_flow(self):
        """Test that gradients flow through weight extraction."""
        d_a, d_x = 1, 1
        discriminator = LinearDiscriminator()
        init_fn = lambda key: discriminator.init_params(key, d_a, d_x)
        apply_fn = discriminator.apply

        key = jax.random.PRNGKey(42)
        params = init_fn(key)

        X = jnp.array([[1.0]])
        A = jnp.array([[1.0]])
        AX = jnp.array([[1.0]])

        def loss_fn(params):
            weights = extract_weights(apply_fn, params, X, A, AX)
            return jnp.sum(weights)

        grads = jax.grad(loss_fn)(params)

        # Gradients should exist and be non-zero
        assert jnp.isfinite(grads["w_a"]).all()
        assert jnp.isfinite(grads["w_x"]).all()
        assert jnp.isfinite(grads["w_ax"]).all()

    def test_multivariate_treatment(self):
        """Test weight extraction with multivariate treatment."""
        d_a, d_x = 2, 3
        discriminator = LinearDiscriminator()
        init_fn = lambda key: discriminator.init_params(key, d_a, d_x)
        apply_fn = discriminator.apply

        key = jax.random.PRNGKey(42)
        params = init_fn(key)

        X = jnp.ones((5, d_x))
        A = jnp.ones((5, d_a))
        AX = jnp.ones((5, d_a * d_x))

        weights = extract_weights(apply_fn, params, X, A, AX)

        assert weights.shape == (5,)
        assert jnp.all(weights > 0)

    def test_jit_compilation(self):
        """Test that extract_weights can be JIT compiled."""
        d_a, d_x = 1, 2
        discriminator = LinearDiscriminator()
        init_fn = lambda key: discriminator.init_params(key, d_a, d_x)
        apply_fn = discriminator.apply

        key = jax.random.PRNGKey(42)
        params = init_fn(key)

        X = jnp.array([[1.0, 2.0]])
        A = jnp.array([[1.0]])
        AX = jnp.array([[1.0, 2.0]])

        jitted_extract = jax.jit(lambda p, x, a, ax: extract_weights(apply_fn, p, x, a, ax))

        weights = jitted_extract(params, X, A, AX)

        assert weights.shape == (1,)
        assert weights[0] > 0
