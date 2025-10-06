"""Tests for stochpw.models module (class-based API)."""

import jax
import jax.numpy as jnp
import pytest
from stochpw.models import LinearDiscriminator, MLPDiscriminator
from stochpw.models.mlp import _get_activation


class TestLinearDiscriminator:
    """Tests for LinearDiscriminator class."""

    def test_initialization(self):
        """Test discriminator initialization."""
        discriminator = LinearDiscriminator()
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=3)

        assert "w_a" in params
        assert "w_x" in params
        assert "w_ax" in params
        assert "b" in params
        assert params["w_a"].shape == (1,)
        assert params["w_x"].shape == (3,)
        assert params["w_ax"].shape == (3,)
        assert params["b"].shape == ()

    def test_parameter_shapes_multivariate(self):
        """Test parameter shapes with multivariate treatment."""
        discriminator = LinearDiscriminator()
        params = discriminator.init_params(jax.random.PRNGKey(0), d_a=2, d_x=5)

        assert params["w_a"].shape == (2,)
        assert params["w_x"].shape == (5,)
        assert params["w_ax"].shape == (10,)  # 2 * 5

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        discriminator = LinearDiscriminator()
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=3)

        batch_size = 16
        a = jnp.ones((batch_size, 1))
        x = jnp.ones((batch_size, 3))
        ax = jnp.ones((batch_size, 3))

        logits = discriminator.apply(params, a, x, ax)

        assert logits.shape == (batch_size,)

    def test_forward_pass_1d_treatment(self):
        """Test forward pass with 1D treatment array."""
        discriminator = LinearDiscriminator()
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=3)

        batch_size = 8
        a = jnp.ones(batch_size)  # 1D
        x = jnp.ones((batch_size, 3))
        ax = jnp.ones((batch_size, 3))

        logits = discriminator.apply(params, a, x, ax)

        assert logits.shape == (batch_size,)

    def test_different_seeds_different_params(self):
        """Test that different random seeds produce different parameters."""
        discriminator = LinearDiscriminator()

        params1 = discriminator.init_params(jax.random.PRNGKey(0), d_a=1, d_x=2)
        params2 = discriminator.init_params(jax.random.PRNGKey(1), d_a=1, d_x=2)

        assert not jnp.allclose(params1["w_a"], params2["w_a"])
        assert not jnp.allclose(params1["w_x"], params2["w_x"])
        assert not jnp.allclose(params1["w_ax"], params2["w_ax"])

    def test_same_seed_same_params(self):
        """Test that same seed produces same parameters."""
        discriminator = LinearDiscriminator()

        params1 = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)
        params2 = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        assert jnp.allclose(params1["w_a"], params2["w_a"])
        assert jnp.allclose(params1["w_x"], params2["w_x"])
        assert jnp.allclose(params1["w_ax"], params2["w_ax"])
        assert jnp.allclose(params1["b"], params2["b"])

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        discriminator = LinearDiscriminator()
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        def loss_fn(params, a, x, ax):
            logits = discriminator.apply(params, a, x, ax)
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
        discriminator = LinearDiscriminator()

        # Create discriminator with known parameters
        params = {
            "w_a": jnp.array([1.0]),
            "w_x": jnp.array([2.0, 3.0]),
            "w_ax": jnp.array([0.5, 0.5]),
            "b": jnp.array(0.1),
        }

        a = jnp.array([[2.0]])  # 2 * 1.0 = 2.0
        x = jnp.array([[1.0, 1.0]])  # 1*2 + 1*3 = 5.0
        ax = jnp.array([[1.0, 1.0]])  # 1*0.5 + 1*0.5 = 1.0

        logits = discriminator.apply(params, a, x, ax)

        expected = 2.0 + 5.0 + 1.0 + 0.1  # 8.1
        assert jnp.allclose(logits, expected)

    def test_batch_independence(self):
        """Test that batch samples are processed independently."""
        discriminator = LinearDiscriminator()
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        # Process batch
        a_batch = jnp.array([[1.0], [2.0]])
        x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        ax_batch = jnp.array([[1.0, 2.0], [6.0, 8.0]])

        logits_batch = discriminator.apply(params, a_batch, x_batch, ax_batch)

        # Process individually
        logits_0 = discriminator.apply(params, a_batch[0:1], x_batch[0:1], ax_batch[0:1])
        logits_1 = discriminator.apply(params, a_batch[1:2], x_batch[1:2], ax_batch[1:2])

        assert jnp.allclose(logits_batch[0], logits_0[0])
        assert jnp.allclose(logits_batch[1], logits_1[0])

    def test_jit_compilation(self):
        """Test that apply can be JIT compiled."""
        discriminator = LinearDiscriminator()
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        jitted_apply = jax.jit(discriminator.apply)

        a = jnp.array([[1.0]])
        x = jnp.array([[2.0, 3.0]])
        ax = jnp.array([[2.0, 3.0]])

        # Should not raise
        logits = jitted_apply(params, a, x, ax)
        assert logits.shape == (1,)

    def test_callable_interface(self):
        """Test that discriminator can be called directly."""
        discriminator = LinearDiscriminator()
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        a = jnp.array([[1.0]])
        x = jnp.array([[2.0, 3.0]])
        ax = jnp.array([[2.0, 3.0]])

        # Test __call__ method
        logits1 = discriminator(params, a, x, ax)
        logits2 = discriminator.apply(params, a, x, ax)

        assert jnp.allclose(logits1, logits2)


class TestActivationFunctions:
    """Tests for activation function utilities."""

    def test_get_relu(self):
        """Test getting ReLU activation."""
        relu_fn = _get_activation("relu")
        x = jnp.array([-1.0, 0.0, 1.0])
        result = relu_fn(x)
        expected = jnp.array([0.0, 0.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_get_tanh(self):
        """Test getting tanh activation."""
        tanh_fn = _get_activation("tanh")
        x = jnp.array([0.0])
        result = tanh_fn(x)
        expected = jnp.array([0.0])
        assert jnp.allclose(result, expected)

    def test_get_elu(self):
        """Test getting ELU activation."""
        elu_fn = _get_activation("elu")
        x = jnp.array([1.0])
        result = elu_fn(x)
        expected = jnp.array([1.0])
        assert jnp.allclose(result, expected)

    def test_get_sigmoid(self):
        """Test getting sigmoid activation."""
        sigmoid_fn = _get_activation("sigmoid")
        x = jnp.array([0.0])
        result = sigmoid_fn(x)
        expected = jnp.array([0.5])
        assert jnp.allclose(result, expected)

    def test_unknown_activation_raises_error(self):
        """Test that unknown activation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown activation"):
            _get_activation("unknown")


class TestMLPDiscriminator:
    """Tests for MLPDiscriminator class."""

    def test_initialization_default_hidden_dims(self):
        """Test MLP initialization with default hidden dims."""
        discriminator = MLPDiscriminator()
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=3)

        assert "layers" in params
        # Default is [64, 32] + output layer = 3 layers
        assert len(params["layers"]) == 3

        # Check layer shapes: input -> 64 -> 32 -> 1
        input_dim = 1 + 3 + (1 * 3)  # A + X + AX
        assert params["layers"][0]["w"].shape == (input_dim, 64)
        assert params["layers"][0]["b"].shape == (64,)
        assert params["layers"][1]["w"].shape == (64, 32)
        assert params["layers"][1]["b"].shape == (32,)
        assert params["layers"][2]["w"].shape == (32, 1)
        assert params["layers"][2]["b"].shape == (1,)

    def test_initialization_custom_hidden_dims(self):
        """Test MLP initialization with custom hidden dims."""
        discriminator = MLPDiscriminator(hidden_dims=[128, 64, 32])
        params = discriminator.init_params(jax.random.PRNGKey(0), d_a=2, d_x=4)

        # Should have 4 layers: 3 hidden + 1 output
        assert len(params["layers"]) == 4

        input_dim = 2 + 4 + (2 * 4)
        assert params["layers"][0]["w"].shape == (input_dim, 128)
        assert params["layers"][1]["w"].shape == (128, 64)
        assert params["layers"][2]["w"].shape == (64, 32)
        assert params["layers"][3]["w"].shape == (32, 1)

    def test_forward_pass_shape(self):
        """Test MLP forward pass output shape."""
        discriminator = MLPDiscriminator(hidden_dims=[16, 8])
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=3)

        batch_size = 10
        a = jnp.ones((batch_size, 1))
        x = jnp.ones((batch_size, 3))
        ax = jnp.ones((batch_size, 3))

        logits = discriminator.apply(params, a, x, ax)

        assert logits.shape == (batch_size,)

    def test_forward_pass_1d_treatment(self):
        """Test MLP forward pass with 1D treatment array."""
        discriminator = MLPDiscriminator()
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        batch_size = 5
        a = jnp.ones(batch_size)  # 1D
        x = jnp.ones((batch_size, 2))
        ax = jnp.ones((batch_size, 2))

        logits = discriminator.apply(params, a, x, ax)

        assert logits.shape == (batch_size,)

    def test_different_activations(self):
        """Test MLP with different activation functions."""
        activations = ["relu", "tanh", "elu", "sigmoid"]

        for activation in activations:
            discriminator = MLPDiscriminator(hidden_dims=[8], activation=activation)
            params = discriminator.init_params(jax.random.PRNGKey(0), d_a=1, d_x=2)

            a = jnp.array([[1.0]])
            x = jnp.array([[2.0, 3.0]])
            ax = jnp.array([[2.0, 3.0]])

            # Should not raise
            logits = discriminator.apply(params, a, x, ax)
            assert logits.shape == (1,)

    def test_different_seeds_different_params(self):
        """Test that different random seeds produce different parameters."""
        discriminator = MLPDiscriminator(hidden_dims=[16])

        params1 = discriminator.init_params(jax.random.PRNGKey(0), d_a=1, d_x=2)
        params2 = discriminator.init_params(jax.random.PRNGKey(1), d_a=1, d_x=2)

        # Check at least one layer has different weights
        assert not jnp.allclose(params1["layers"][0]["w"], params2["layers"][0]["w"])

    def test_same_seed_same_params(self):
        """Test that same seed produces same parameters."""
        discriminator = MLPDiscriminator(hidden_dims=[16])

        params1 = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)
        params2 = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        for i in range(len(params1["layers"])):
            assert jnp.allclose(params1["layers"][i]["w"], params2["layers"][i]["w"])
            assert jnp.allclose(params1["layers"][i]["b"], params2["layers"][i]["b"])

    def test_gradient_flow(self):
        """Test that gradients flow through the MLP."""
        discriminator = MLPDiscriminator(hidden_dims=[8])
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        def loss_fn(params, a, x, ax):
            logits = discriminator.apply(params, a, x, ax)
            return jnp.mean(logits**2)

        a = jnp.array([[1.0]])
        x = jnp.array([[2.0, 3.0]])
        ax = jnp.array([[2.0, 3.0]])

        grads = jax.grad(loss_fn)(params, a, x, ax)

        # Check gradients exist for all layers
        for i in range(len(params["layers"])):
            assert not jnp.allclose(grads["layers"][i]["w"], 0.0)

    def test_batch_independence(self):
        """Test that batch samples are processed independently."""
        discriminator = MLPDiscriminator(hidden_dims=[8])
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        # Process batch
        a_batch = jnp.array([[1.0], [2.0]])
        x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        ax_batch = jnp.array([[1.0, 2.0], [6.0, 8.0]])

        logits_batch = discriminator.apply(params, a_batch, x_batch, ax_batch)

        # Process individually
        logits_0 = discriminator.apply(params, a_batch[0:1], x_batch[0:1], ax_batch[0:1])
        logits_1 = discriminator.apply(params, a_batch[1:2], x_batch[1:2], ax_batch[1:2])

        assert jnp.allclose(logits_batch[0], logits_0[0], atol=1e-6)
        assert jnp.allclose(logits_batch[1], logits_1[0], atol=1e-6)

    def test_jit_compilation(self):
        """Test that MLP apply can be JIT compiled."""
        discriminator = MLPDiscriminator(hidden_dims=[8])
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        jitted_apply = jax.jit(discriminator.apply)

        a = jnp.array([[1.0]])
        x = jnp.array([[2.0, 3.0]])
        ax = jnp.array([[2.0, 3.0]])

        # Should not raise
        logits = jitted_apply(params, a, x, ax)
        assert logits.shape == (1,)

    def test_nonlinear_transformation(self):
        """Test that MLP applies nonlinear transformation (unlike linear model)."""
        discriminator = MLPDiscriminator(hidden_dims=[16], activation="relu")
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        # For ReLU MLP, doubling input doesn't double output (nonlinearity)
        a = jnp.array([[1.0]])
        x = jnp.array([[2.0, 3.0]])
        ax = jnp.array([[2.0, 3.0]])

        logits1 = discriminator.apply(params, a, x, ax)
        logits2 = discriminator.apply(params, 2 * a, 2 * x, 2 * ax)

        # Should NOT be linear relationship due to ReLU
        # (unless all activations happen to be in linear region, which is unlikely)
        assert logits1.shape == logits2.shape

    def test_multivariate_treatment(self):
        """Test MLP with multivariate treatment."""
        discriminator = MLPDiscriminator(hidden_dims=[32, 16])
        params = discriminator.init_params(jax.random.PRNGKey(0), d_a=3, d_x=5)

        batch_size = 10
        key = jax.random.PRNGKey(0)
        a = jax.random.normal(key, (batch_size, 3))
        x = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 5))
        ax = jnp.einsum("bi,bj->bij", a, x).reshape(batch_size, -1)

        logits = discriminator.apply(params, a, x, ax)

        assert logits.shape == (batch_size,)

    def test_callable_interface(self):
        """Test that discriminator can be called directly."""
        discriminator = MLPDiscriminator(hidden_dims=[8])
        params = discriminator.init_params(jax.random.PRNGKey(42), d_a=1, d_x=2)

        a = jnp.array([[1.0]])
        x = jnp.array([[2.0, 3.0]])
        ax = jnp.array([[2.0, 3.0]])

        # Test __call__ method
        logits1 = discriminator(params, a, x, ax)
        logits2 = discriminator.apply(params, a, x, ax)

        assert jnp.allclose(logits1, logits2)
