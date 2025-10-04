"""Tests for stochpw.utils module."""

import jax.numpy as jnp
import numpy as np
import pytest
from stochpw.utils import validate_inputs


class TestValidateInputs:
    """Tests for validate_inputs function."""

    def test_valid_inputs_2d(self):
        """Test validation with valid 2D inputs."""
        X = np.random.randn(15, 2)
        A = np.random.choice([0.0, 1.0], size=(15, 1))

        X_val, A_val = validate_inputs(X, A)

        assert isinstance(X_val, jnp.ndarray)
        assert isinstance(A_val, jnp.ndarray)
        assert X_val.shape == (15, 2)
        assert A_val.shape == (15, 1)

    def test_valid_inputs_1d_treatment(self):
        """Test validation with 1D treatment array."""
        X = np.random.randn(12, 2)
        A = np.random.choice([0.0, 1.0], size=12)

        X_val, A_val = validate_inputs(X, A)

        assert A_val.shape == (12, 1)  # Should be reshaped to 2D

    def test_converts_numpy_to_jax(self):
        """Test that NumPy arrays are converted to JAX arrays."""
        X = np.random.randn(10, 2)
        A = np.random.choice([0.0, 1.0], size=(10, 1))

        X_val, A_val = validate_inputs(X, A)

        assert isinstance(X_val, jnp.ndarray)
        assert isinstance(A_val, jnp.ndarray)

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched shapes raise ValueError."""
        X = np.random.randn(12, 2)
        A = np.array([[0.0]])  # Wrong number of samples

        with pytest.raises(ValueError, match="same number of samples"):
            validate_inputs(X, A)

    def test_nan_in_X_raises_error(self):
        """Test that NaN in X raises ValueError."""
        X = np.random.randn(10, 2)
        X[0, 0] = np.nan
        A = np.random.choice([0.0, 1.0], size=(10, 1))

        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            validate_inputs(X, A)

    def test_inf_in_X_raises_error(self):
        """Test that Inf in X raises ValueError."""
        X = np.random.randn(10, 2)
        X[0, 0] = np.inf
        A = np.random.choice([0.0, 1.0], size=(10, 1))

        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            validate_inputs(X, A)

    def test_nan_in_A_raises_error(self):
        """Test that NaN in A raises ValueError."""
        X = np.random.randn(10, 2)
        A = np.random.choice([0.0, 1.0], size=(10, 1))
        A[0, 0] = np.nan

        with pytest.raises(ValueError, match="A contains NaN or Inf"):
            validate_inputs(X, A)

    def test_inf_in_A_raises_error(self):
        """Test that Inf in A raises ValueError."""
        X = np.random.randn(10, 2)
        A = np.random.choice([0.0, 1.0], size=(10, 1))
        A[0, 0] = np.inf

        with pytest.raises(ValueError, match="A contains NaN or Inf"):
            validate_inputs(X, A)

    def test_constant_treatment_raises_error(self):
        """Test that constant treatment raises ValueError."""
        X = np.random.randn(12, 2)
        A = np.ones((12, 1))  # All same value

        with pytest.raises(ValueError, match="at least 2 unique values"):
            validate_inputs(X, A)

    def test_scalar_A_raises_error(self):
        """Test that scalar A raises ValueError."""
        X = np.random.randn(10, 2)
        A = np.array(0.5)  # Scalar (0-dimensional)

        with pytest.raises(ValueError, match="at least 1-dimensional"):
            validate_inputs(X, A)

    def test_too_few_samples_raises_error(self):
        """Test that fewer than 10 samples raises ValueError."""
        X = np.random.randn(9, 2)
        A = np.random.choice([0.0, 1.0], size=9)

        with pytest.raises(ValueError, match="at least 10 samples"):
            validate_inputs(X, A)

    def test_1d_X_raises_error(self):
        """Test that 1D X raises ValueError."""
        X = np.array([1.0, 2.0, 3.0])
        A = np.array([[0.0], [1.0], [0.0]])

        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            validate_inputs(X, A)

    def test_3d_X_raises_error(self):
        """Test that 3D X raises ValueError."""
        X = np.array([[[1.0]]])
        A = np.array([[0.0]])

        with pytest.raises(ValueError, match="X must be 2-dimensional"):
            validate_inputs(X, A)

    def test_3d_A_raises_error(self):
        """Test that 3D A raises ValueError."""
        X = np.array([[1.0, 2.0]])
        A = np.array([[[0.0]]])

        with pytest.raises(ValueError, match="A must be 1 or 2-dimensional"):
            validate_inputs(X, A)

    def test_multivariate_treatment(self):
        """Test with multivariate treatment."""
        X = np.random.randn(10, 2)
        A = np.random.randn(10, 2)

        X_val, A_val = validate_inputs(X, A)

        assert A_val.shape == (10, 2)

    def test_binary_treatment_variation(self):
        """Test binary treatment with variation."""
        X = np.random.randn(12, 1)
        A = np.random.choice([0.0, 1.0], size=12)

        X_val, A_val = validate_inputs(X, A)

        assert A_val.shape == (12, 1)

    def test_continuous_treatment_variation(self):
        """Test continuous treatment with variation."""
        X = np.random.randn(15, 1)
        A = np.random.uniform(0, 1, size=15)

        X_val, A_val = validate_inputs(X, A)

        assert A_val.shape == (15, 1)
