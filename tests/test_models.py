"""Tests for stochpw.models module."""

import jax
import jax.numpy as jnp
from stochpw.models import create_linear_discriminator


class TestLinearDiscriminator:
    """Tests for linear discriminator model."""

    def test_initialization(self):
        """Test discriminator initialization."""
        d_a, d_x = 1, 3
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

        key = jax.random.PRNGKey(42)
        params = init_fn(key)

        assert "w_a" in params
        assert "w_x" in params
        assert "w_ax" in params
        assert "b" in params
        assert params["w_a"].shape == (d_a,)
        assert params["w_x"].shape == (d_x,)
        assert params["w_ax"].shape == (d_a * d_x,)
        assert params["b"].shape == ()

    def test_parameter_shapes_multivariate(self):
        """Test parameter shapes with multivariate treatment."""
        d_a, d_x = 2, 5
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

        key = jax.random.PRNGKey(0)
        params = init_fn(key)

        assert params["w_a"].shape == (2,)
        assert params["w_x"].shape == (5,)
        assert params["w_ax"].shape == (10,)  # 2 * 5

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        d_a, d_x = 1, 3
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

        key = jax.random.PRNGKey(42)
        params = init_fn(key)

        batch_size = 16
        a = jnp.ones((batch_size, d_a))
        x = jnp.ones((batch_size, d_x))
        ax = jnp.ones((batch_size, d_a * d_x))

        logits = apply_fn(params, a, x, ax)

        assert logits.shape == (batch_size,)

    def test_forward_pass_1d_treatment(self):
        """Test forward pass with 1D treatment array."""
        d_a, d_x = 1, 3
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

        key = jax.random.PRNGKey(42)
        params = init_fn(key)

        batch_size = 8
        a = jnp.ones(batch_size)  # 1D
        x = jnp.ones((batch_size, d_x))
        ax = jnp.ones((batch_size, d_a * d_x))

        logits = apply_fn(params, a, x, ax)

        assert logits.shape == (batch_size,)

    def test_different_seeds_different_params(self):
        """Test that different random seeds produce different parameters."""
        d_a, d_x = 1, 2
        init_fn, _ = create_linear_discriminator(d_a, d_x)

        params1 = init_fn(jax.random.PRNGKey(0))
        params2 = init_fn(jax.random.PRNGKey(1))

        assert not jnp.allclose(params1["w_a"], params2["w_a"])
        assert not jnp.allclose(params1["w_x"], params2["w_x"])
        assert not jnp.allclose(params1["w_ax"], params2["w_ax"])

    def test_same_seed_same_params(self):
        """Test that same seed produces same parameters."""
        d_a, d_x = 1, 2
        init_fn, _ = create_linear_discriminator(d_a, d_x)

        params1 = init_fn(jax.random.PRNGKey(42))
        params2 = init_fn(jax.random.PRNGKey(42))

        assert jnp.allclose(params1["w_a"], params2["w_a"])
        assert jnp.allclose(params1["w_x"], params2["w_x"])
        assert jnp.allclose(params1["w_ax"], params2["w_ax"])
        assert jnp.allclose(params1["b"], params2["b"])

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        d_a, d_x = 1, 2
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

        key = jax.random.PRNGKey(42)
        params = init_fn(key)

        def loss_fn(params, a, x, ax):
            logits = apply_fn(params, a, x, ax)
            return jnp.mean(logits**2)

        a = jnp.array([[1.0]])
        x = jnp.array([[2.0, 3.0]])
        ax = jnp.array([[2.0, 3.0]])

        grads = jax.grad(loss_fn)(params, a, x, ax)

        # Check all gradients are non-zero
        assert not jnp.allclose(grads["w_a"], 0.0)
        assert not jnp.allclose(grads["w_x"], 0.0)
        assert not jnp.allclose(grads["w_ax"], 0.0)
        assert not jnp.allclose(grads["b"], 0.0)

    def test_linear_combination(self):
        """Test that discriminator computes correct linear combination."""
        d_a, d_x = 1, 2
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

        # Set known parameters
        params = {
            "w_a": jnp.array([1.0]),
            "w_x": jnp.array([2.0, 3.0]),
            "w_ax": jnp.array([0.5, 0.5]),
            "b": jnp.array(0.1),
        }

        a = jnp.array([[2.0]])  # 2 * 1.0 = 2.0
        x = jnp.array([[1.0, 1.0]])  # 1*2 + 1*3 = 5.0
        ax = jnp.array([[1.0, 1.0]])  # 1*0.5 + 1*0.5 = 1.0

        logits = apply_fn(params, a, x, ax)

        expected = 2.0 + 5.0 + 1.0 + 0.1  # 8.1
        assert jnp.allclose(logits, expected)

    def test_batch_independence(self):
        """Test that batch samples are processed independently."""
        d_a, d_x = 1, 2
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

        key = jax.random.PRNGKey(42)
        params = init_fn(key)

        # Process batch
        a_batch = jnp.array([[1.0], [2.0]])
        x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        ax_batch = jnp.array([[1.0, 2.0], [6.0, 8.0]])

        logits_batch = apply_fn(params, a_batch, x_batch, ax_batch)

        # Process individually
        logits_0 = apply_fn(params, a_batch[0:1], x_batch[0:1], ax_batch[0:1])
        logits_1 = apply_fn(params, a_batch[1:2], x_batch[1:2], ax_batch[1:2])

        assert jnp.allclose(logits_batch[0], logits_0[0])
        assert jnp.allclose(logits_batch[1], logits_1[0])

    def test_jit_compilation(self):
        """Test that apply_fn can be JIT compiled."""
        d_a, d_x = 1, 2
        init_fn, apply_fn = create_linear_discriminator(d_a, d_x)

        key = jax.random.PRNGKey(42)
        params = init_fn(key)

        jitted_apply = jax.jit(apply_fn)

        a = jnp.array([[1.0]])
        x = jnp.array([[2.0, 3.0]])
        ax = jnp.array([[2.0, 3.0]])

        # Should not raise
        logits = jitted_apply(params, a, x, ax)
        assert logits.shape == (1,)
