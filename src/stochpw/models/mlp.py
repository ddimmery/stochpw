"""Multi-layer perceptron discriminator for permutation weighting."""

from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from jax import Array

from .base import BaseDiscriminator

PyTree = Any
ActivationType = Literal["relu", "tanh", "elu", "sigmoid"]


def _get_activation(activation: ActivationType) -> Callable[[Array], Array]:
    """
    Get activation function by name.

    Parameters
    ----------
    activation : {'relu', 'tanh', 'elu', 'sigmoid'}
        Name of activation function

    Returns
    -------
    activation_fn : Callable
        Activation function that maps Array -> Array
    """
    activations = {
        "relu": jax.nn.relu,
        "tanh": jnp.tanh,
        "elu": jax.nn.elu,
        "sigmoid": jax.nn.sigmoid,
    }
    if activation not in activations:
        raise ValueError(
            f"Unknown activation '{activation}'. Choose from: {list(activations.keys())}"
        )
    return activations[activation]


class MLPDiscriminator(BaseDiscriminator):
    """
    Multi-layer perceptron (MLP) discriminator using A, X, and A*X interactions.

    The MLP processes concatenated features [A, X, A*X] through configurable
    hidden layers with specified activation functions, outputting a scalar logit.

    This provides more expressive power than linear discriminators for capturing
    complex relationships between treatments and covariates.

    Parameters
    ----------
    hidden_dims : list[int], optional
        List of hidden layer sizes. Default is [64, 32]
    activation : {'relu', 'tanh', 'elu', 'sigmoid'}, optional
        Activation function to use between layers. Default is 'relu'

    Examples
    --------
    >>> from stochpw.models import MLPDiscriminator
    >>> import jax
    >>>
    >>> # Default: 2-layer MLP with ReLU
    >>> discriminator = MLPDiscriminator()
    >>>
    >>> # Custom: 3-layer MLP with tanh
    >>> discriminator = MLPDiscriminator(hidden_dims=[128, 64, 32], activation='tanh')
    >>>
    >>> params = discriminator.init_params(jax.random.PRNGKey(0), d_a=1, d_x=3)
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        activation: ActivationType = "relu",
    ):
        if hidden_dims is None:
            hidden_dims = [64, 32]
        self.hidden_dims = hidden_dims
        self.activation = activation
        self._activation_fn = _get_activation(activation)
        self._use_he_init = activation in ("relu", "elu")

    def init_params(self, rng_key: Array, d_a: int, d_x: int) -> dict:
        """
        Initialize MLP discriminator parameters.

        Uses He initialization for ReLU-family activations and Xavier
        initialization for tanh/sigmoid activations. Biases are initialized to zero.

        Parameters
        ----------
        rng_key : jax.Array
            Random key for parameter initialization
        d_a : int
            Dimension of treatment vector
        d_x : int
            Dimension of covariate vector

        Returns
        -------
        params : dict
            Dictionary with key 'layers' containing a list of layer dicts,
            each with keys 'w' (weight matrix) and 'b' (bias vector)
        """
        interaction_dim = d_a * d_x
        input_dim = d_a + d_x + interaction_dim

        params = {"layers": []}
        layer_dims = [input_dim] + self.hidden_dims + [1]  # Output is scalar logit
        current_key = rng_key

        for i in range(len(layer_dims) - 1):
            current_key, layer_key = jax.random.split(current_key)
            w_key, b_key = jax.random.split(layer_key)

            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]

            # He initialization for ReLU, Xavier for others
            if self._use_he_init:
                std = jnp.sqrt(2.0 / in_dim)
            else:
                std = jnp.sqrt(2.0 / (in_dim + out_dim))

            w = jax.random.normal(w_key, (in_dim, out_dim)) * std
            b = jnp.zeros((out_dim,))  # Initialize biases to zero

            params["layers"].append({"w": w, "b": b})

        return params

    def apply(self, params: dict, a: Array, x: Array, ax: Array) -> Array:
        """
        Compute MLP discriminator logits using A, X, and A*X.

        Parameters
        ----------
        params : dict
            Parameters with key 'layers' containing list of layer dicts
        a : jax.Array, shape (batch_size, d_a) or (batch_size,)
            Treatment assignments
        x : jax.Array, shape (batch_size, d_x)
            Covariates
        ax : jax.Array, shape (batch_size, d_a * d_x)
            Pre-computed first-order interactions A ⊗ X

        Returns
        -------
        logits : jax.Array, shape (batch_size,)
            Discriminator logits for p(C=1 | a, x)
        """
        # Ensure a is 2D
        if a.ndim == 1:
            a = a.reshape(-1, 1)

        # Concatenate all features: [A, X, A*X]
        h = jnp.concatenate([a, x, ax], axis=-1)

        # Forward pass through hidden layers
        for i, layer in enumerate(params["layers"]):
            h = jnp.dot(h, layer["w"]) + layer["b"]

            # Apply activation to all layers except the last (output layer)
            if i < len(params["layers"]) - 1:
                h = self._activation_fn(h)

        # Output is shape (batch_size, 1), squeeze to (batch_size,)
        logits = h.squeeze(-1)

        return logits
