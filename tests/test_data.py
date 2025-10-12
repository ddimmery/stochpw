"""Tests for stochpw.data module."""

import jax
import jax.numpy as jnp
import pytest
from stochpw.data import TrainingBatch, TrainingState, TrainingStepResult, WeightedData


class TestTrainingBatch:
    """Tests for TrainingBatch dataclass."""

    def test_creation(self):
        """Test TrainingBatch creation."""
        X = jnp.array([[1.0, 2.0]])
        A = jnp.array([[0.0]])
        C = jnp.array([1.0])
        AX = jnp.array([[0.0, 0.0]])

        batch = TrainingBatch(X=X, A=A, C=C, AX=AX)

        assert jnp.array_equal(batch.X, X)
        assert jnp.array_equal(batch.A, A)
        assert jnp.array_equal(batch.C, C)
        assert jnp.array_equal(batch.AX, AX)

    def test_immutability(self):
        """Test that TrainingBatch is immutable."""
        batch = TrainingBatch(
            X=jnp.array([[1.0]]), A=jnp.array([[0.0]]), C=jnp.array([1.0]), AX=jnp.array([[0.0]])
        )

        with pytest.raises((AttributeError, ValueError, TypeError)):
            batch.X = jnp.array([[2.0]])

    def test_field_types(self):
        """Test field types."""
        batch = TrainingBatch(
            X=jnp.array([[1.0, 2.0]]),
            A=jnp.array([[0.0]]),
            C=jnp.array([1.0]),
            AX=jnp.array([[0.0, 0.0]]),
        )

        assert isinstance(batch.X, jnp.ndarray)
        assert isinstance(batch.A, jnp.ndarray)
        assert isinstance(batch.C, jnp.ndarray)
        assert isinstance(batch.AX, jnp.ndarray)


class TestWeightedData:
    """Tests for WeightedData dataclass."""

    def test_creation(self):
        """Test WeightedData creation."""
        X = jnp.array([[1.0, 2.0]])
        A = jnp.array([[0.0]])
        weights = jnp.array([1.0])

        data = WeightedData(X=X, A=A, weights=weights)

        assert jnp.array_equal(data.X, X)
        assert jnp.array_equal(data.A, A)
        assert jnp.array_equal(data.weights, weights)

    def test_immutability(self):
        """Test that WeightedData is immutable."""
        data = WeightedData(X=jnp.array([[1.0]]), A=jnp.array([[0.0]]), weights=jnp.array([1.0]))

        with pytest.raises((AttributeError, ValueError, TypeError)):
            data.weights = jnp.array([2.0])


class TestTrainingState:
    """Tests for TrainingState dataclass."""

    def test_creation(self):
        """Test TrainingState creation."""
        params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(0.0)}
        opt_state = None
        rng_key = jax.random.PRNGKey(0)
        epoch = 0
        history = {"loss": []}

        state = TrainingState(
            params=params, opt_state=opt_state, rng_key=rng_key, epoch=epoch, history=history
        )

        assert state.params == params
        assert state.opt_state is None
        assert jnp.array_equal(state.rng_key, rng_key)
        assert state.epoch == 0
        assert state.history == {"loss": []}

    def test_mutability(self):
        """Test that TrainingState is mutable (not frozen)."""
        state = TrainingState(
            params={},
            opt_state=None,
            rng_key=jax.random.PRNGKey(0),
            epoch=0,
            history={},
        )

        # Should be able to modify
        state.epoch = 1
        assert state.epoch == 1

    def test_history_accumulation(self):
        """Test that history can be accumulated."""
        history = {"loss": []}
        state = TrainingState(
            params={},
            opt_state=None,
            rng_key=jax.random.PRNGKey(0),
            epoch=0,
            history=history,
        )

        state.history["loss"].append(0.5)
        assert state.history["loss"] == [0.5]


class TestTrainingStepResult:
    """Tests for TrainingStepResult dataclass."""

    def test_creation(self):
        """Test TrainingStepResult creation."""
        state = TrainingState(
            params={}, opt_state=None, rng_key=jax.random.PRNGKey(0), epoch=0, history={}
        )
        loss = 0.5

        result = TrainingStepResult(state=state, loss=loss)

        assert result.state == state
        assert result.loss == 0.5

    def test_immutability(self):
        """Test that TrainingStepResult is immutable."""
        state = TrainingState(
            params={}, opt_state=None, rng_key=jax.random.PRNGKey(0), epoch=0, history={}
        )
        result = TrainingStepResult(state=state, loss=0.5)

        with pytest.raises((AttributeError, ValueError, TypeError)):
            result.loss = 0.3
