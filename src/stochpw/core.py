"""Main API for permutation weighting."""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from .models import BaseDiscriminator, LinearDiscriminator
from .training import fit_discriminator
from .utils import validate_inputs
from .weights import extract_weights


class NotFittedError(Exception):
    """Exception raised when predict is called before fit."""

    pass


class PermutationWeighter:
    """
    Main class for permutation weighting (sklearn-style API).

    Permutation weighting learns importance weights by training a discriminator
    to distinguish between observed (X, A) pairs and permuted (X, A') pairs,
    where treatments are randomly shuffled to break the association with covariates.

    Parameters
    ----------
    discriminator : BaseDiscriminator, optional
        Discriminator model instance. If None, uses LinearDiscriminator().
        Can be any subclass of BaseDiscriminator (e.g., LinearDiscriminator,
        MLPDiscriminator, or custom discriminator).
    optimizer : optax.GradientTransformation, optional
        Optax optimizer. If None, uses Adam with learning rate 1e-3.
    num_epochs : int, default=100
        Number of training epochs
    batch_size : int, default=256
        Mini-batch size for training
    random_state : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    params_ : dict
        Fitted discriminator parameters (set after fit)
    history_ : dict
        Training history with 'loss' key (set after fit)

    Examples
    --------
    >>> from stochpw import PermutationWeighter, LinearDiscriminator, MLPDiscriminator
    >>> import jax.numpy as jnp
    >>>
    >>> # Generate synthetic data
    >>> X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    >>> A = jnp.array([[0.0], [1.0], [0.0]])
    >>>
    >>> # Default: Linear discriminator
    >>> weighter = PermutationWeighter(num_epochs=50, random_state=42)
    >>> weighter.fit(X, A)
    >>> weights = weighter.predict(X, A)
    >>>
    >>> # MLP discriminator with custom architecture
    >>> mlp_disc = MLPDiscriminator(hidden_dims=[128, 64], activation='tanh')
    >>> weighter = PermutationWeighter(discriminator=mlp_disc, num_epochs=50, random_state=42)
    >>> weighter.fit(X, A)
    >>> weights = weighter.predict(X, A)
    """

    def __init__(
        self,
        discriminator: Optional[BaseDiscriminator] = None,
        optimizer: Optional[optax.GradientTransformation] = None,
        num_epochs: int = 100,
        batch_size: int = 256,
        random_state: Optional[int] = None,
    ):
        self.discriminator = discriminator if discriminator is not None else LinearDiscriminator()
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.random_state = random_state

        # Fitted attributes (set by fit())
        self.params_ = None
        self.history_ = None
        self._input_dim = None

    def fit(self, X: Array | np.ndarray, A: Array | np.ndarray) -> "PermutationWeighter":
        """
        Fit discriminator on data (sklearn-style).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Covariates
        A : array-like, shape (n_samples,) or (n_samples, n_treatments)
            Treatment assignments

        Returns
        -------
        self : PermutationWeighter
            Fitted estimator (for method chaining)
        """
        # Validate and convert inputs
        X, A = validate_inputs(X, A)

        # Set up RNG key
        if self.random_state is not None:
            rng_key = jax.random.PRNGKey(self.random_state)
        else:
            rng_key = jax.random.PRNGKey(0)

        # Determine dimensions
        d_a = A.shape[1]
        d_x = X.shape[1]

        # Initialize discriminator parameters
        init_key, train_key = jax.random.split(rng_key)
        init_params = self.discriminator.init_params(init_key, d_a, d_x)

        # Set up optimizer
        if self.optimizer is None:
            optimizer = optax.adam(1e-3)
        else:
            optimizer = self.optimizer

        # Fit discriminator
        self.params_, self.history_ = fit_discriminator(
            X=X,
            A=A,
            discriminator_fn=self.discriminator.apply,
            init_params=init_params,
            optimizer=optimizer,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            rng_key=train_key,
        )

        return self

    def predict(self, X: Array | np.ndarray, A: Array | np.ndarray) -> Array:
        """
        Predict importance weights for given data (sklearn-style).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Covariates
        A : array-like, shape (n_samples,) or (n_samples, n_treatments)
            Treatment assignments

        Returns
        -------
        weights : jax.Array, shape (n_samples,)
            Importance weights

        Raises
        ------
        NotFittedError
            If called before fit()
        """
        if self.params_ is None:
            raise NotFittedError(
                "This PermutationWeighter instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'predict'."
            )

        # Validate and convert inputs
        X, A = validate_inputs(X, A)

        # Compute interactions
        AX = jnp.einsum("bi,bj->bij", A, X).reshape(X.shape[0], -1)

        # Extract weights
        weights = extract_weights(self.discriminator.apply, self.params_, X, A, AX)

        return weights
