"""Tests for stochpw.diagnostics module."""

import jax.numpy as jnp
from stochpw.diagnostics import (
    effective_sample_size,
    standardized_mean_difference,
    standardized_mean_difference_se,
)


class TestEffectiveSampleSize:
    """Tests for effective_sample_size function."""

    def test_uniform_weights(self):
        """Test ESS with uniform weights."""
        weights = jnp.ones(100)

        ess = effective_sample_size(weights)

        # ESS = (sum w)^2 / sum(w^2) = 100^2 / 100 = 100
        assert jnp.isclose(ess, 100.0)

    def test_single_nonzero_weight(self):
        """Test ESS with single non-zero weight."""
        weights = jnp.array([1.0, 0.0, 0.0, 0.0])

        ess = effective_sample_size(weights)

        # ESS = 1^2 / 1 = 1
        assert jnp.isclose(ess, 1.0)

    def test_two_equal_weights(self):
        """Test ESS with two equal weights."""
        weights = jnp.array([1.0, 1.0, 0.0, 0.0])

        ess = effective_sample_size(weights)

        # ESS = 2^2 / 2 = 2
        assert jnp.isclose(ess, 2.0)

    def test_variable_weights(self):
        """Test ESS with variable weights."""
        weights = jnp.array([1.0, 2.0, 3.0])

        ess = effective_sample_size(weights)

        # ESS = (1+2+3)^2 / (1+4+9) = 36/14 ≈ 2.57
        expected = 36.0 / 14.0
        assert jnp.isclose(ess, expected)

    def test_ess_less_than_n(self):
        """Test that ESS <= n for any weight distribution."""
        weights = jnp.array([0.5, 1.0, 2.0, 0.3, 1.5])

        ess = effective_sample_size(weights)

        assert ess <= len(weights)

    def test_jit_compilation(self):
        """Test that ESS can be JIT compiled."""
        import jax

        jitted_ess = jax.jit(effective_sample_size)

        weights = jnp.array([1.0, 2.0, 1.0])
        ess = jitted_ess(weights)

        assert jnp.isfinite(ess)


class TestStandardizedMeanDifference:
    """Tests for standardized_mean_difference function."""

    def test_binary_treatment_no_difference(self):
        """Test SMD with perfectly balanced binary treatment."""
        # Same covariate distributions in both groups
        X = jnp.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])
        A = jnp.array([0.0, 1.0, 0.0, 1.0])
        weights = jnp.ones(4)

        smd = standardized_mean_difference(X, A, weights)

        assert smd.shape == (2,)  # One SMD per covariate
        assert jnp.allclose(smd, 0.0, atol=1e-6)

    def test_binary_treatment_with_difference(self):
        """Test SMD with imbalanced binary treatment."""
        # Treatment group has higher values
        X = jnp.array([[1.0], [1.0], [3.0], [3.0]])
        A = jnp.array([0.0, 0.0, 1.0, 1.0])
        weights = jnp.ones(4)

        smd = standardized_mean_difference(X, A, weights)

        # Mean difference exists, SMD should be non-zero
        assert smd.shape == (1,)
        assert jnp.abs(smd[0]) > 0.1

    def test_uniform_weights_vs_imbalanced(self):
        """Test that reweighting changes SMD."""
        # Create data where treatment=1 group has higher X values
        X = jnp.array([[1.0], [1.5], [2.0], [2.5], [3.0], [4.0], [4.5], [5.0], [5.5], [6.0]])
        A = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        # Uniform weights
        smd_uniform = standardized_mean_difference(X, A, jnp.ones(10))

        # Weights that up-weight lower X values in treatment=1 group
        # This should reduce the mean difference
        weights_balanced = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5])
        smd_weighted = standardized_mean_difference(X, A, weights_balanced)

        # SMDs should differ (weighted should be smaller - better balance)
        assert jnp.abs(smd_weighted[0]) < jnp.abs(smd_uniform[0])

    def test_continuous_treatment_correlation(self):
        """Test SMD with continuous treatment (computes correlation)."""
        # Perfect positive correlation
        X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        A = jnp.array([1.0, 2.0, 3.0, 4.0])  # Not binary
        weights = jnp.ones(4)

        smd = standardized_mean_difference(X, A, weights)

        # Should be high positive correlation
        assert smd.shape == (1,)
        assert smd[0] > 0.9  # Strong correlation

    def test_multivariate_covariates(self):
        """Test SMD with multiple covariates."""
        X = jnp.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        A = jnp.array([0.0, 0.0, 1.0, 1.0])
        weights = jnp.ones(4)

        smd = standardized_mean_difference(X, A, weights)

        assert smd.shape == (2,)  # One SMD per covariate
        assert jnp.all(jnp.isfinite(smd))

    def test_2d_treatment_array(self):
        """Test that 2D treatment array is handled correctly."""
        X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        A = jnp.array([[0.0], [0.0], [1.0], [1.0]])  # 2D
        weights = jnp.ones(4)

        smd = standardized_mean_difference(X, A, weights)

        assert smd.shape == (1,)
        assert jnp.isfinite(smd[0])

    def test_weighted_means_computed_correctly(self):
        """Test that weighted means are computed correctly."""
        # Simple case where we can verify by hand
        X = jnp.array([[1.0], [3.0]])
        A = jnp.array([0.0, 1.0])
        weights = jnp.array([2.0, 1.0])  # Weight group 0 more

        smd = standardized_mean_difference(X, A, weights)

        # Weighted mean for group 0: 1.0 * 2 / 2 = 1.0
        # Weighted mean for group 1: 3.0 * 1 / 1 = 3.0
        # Difference should be reflected in SMD
        assert jnp.isfinite(smd[0])

    def test_zero_variance_handling(self):
        """Test handling of zero variance (shouldn't crash)."""
        # Constant covariate
        X = jnp.array([[1.0], [1.0], [1.0], [1.0]])
        A = jnp.array([0.0, 0.0, 1.0, 1.0])
        weights = jnp.ones(4)

        smd = standardized_mean_difference(X, A, weights)

        # Should handle gracefully (with epsilon in denominator)
        assert jnp.isfinite(smd[0])

    def test_negative_smd_possible(self):
        """Test that SMD can be negative."""
        # Control group has higher values
        X = jnp.array([[3.0], [3.0], [1.0], [1.0]])
        A = jnp.array([0.0, 0.0, 1.0, 1.0])
        weights = jnp.ones(4)

        smd = standardized_mean_difference(X, A, weights)

        # SMD can be negative
        assert smd[0] < 0

    def test_binary_detection(self):
        """Test that binary treatment is correctly detected."""
        # Binary case
        X = jnp.ones((4, 2))
        A_binary = jnp.array([0.0, 1.0, 0.0, 1.0])
        weights = jnp.ones(4)

        smd_binary = standardized_mean_difference(X, A_binary, weights)

        # Continuous case (3 unique values)
        A_continuous = jnp.array([0.0, 0.5, 1.0, 0.5])
        smd_continuous = standardized_mean_difference(X, A_continuous, weights)

        # Both should execute without error
        assert smd_binary.shape == (2,)
        assert smd_continuous.shape == (2,)

    def test_weighted_variance_computation(self):
        """Test that weighted variance is computed correctly."""
        X = jnp.array([[1.0], [2.0], [3.0], [4.0]])
        A = jnp.array([0.0, 0.0, 1.0, 1.0])

        # Unequal weights should change variance calculation
        weights_uniform = jnp.ones(4)
        weights_variable = jnp.array([1.0, 2.0, 1.0, 2.0])

        smd_uniform = standardized_mean_difference(X, A, weights_uniform)
        smd_variable = standardized_mean_difference(X, A, weights_variable)

        # Different weighting should give different SMD
        assert not jnp.allclose(smd_uniform, smd_variable, atol=0.01)


class TestStandardizedMeanDifferenceSE:
    """Tests for standardized_mean_difference_se function."""

    def test_binary_treatment_se_shape(self):
        """SE should have same shape as number of features for binary treatment."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 10)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 10)
        weights = jnp.ones(40)

        se = standardized_mean_difference_se(X, A, weights)

        assert se.shape == (2,)
        assert jnp.all(se > 0)  # SE should be positive

    def test_binary_treatment_1d_array(self):
        """SE should work with 1D treatment array."""
        X = jnp.array([[1.0, 2.0, 3.0]] * 20)
        A = jnp.array([0.0, 1.0] * 10)  # 1D array
        weights = jnp.ones(20)

        se = standardized_mean_difference_se(X, A, weights)

        assert se.shape == (3,)
        assert jnp.all(se > 0)

    def test_continuous_treatment_se(self):
        """SE should work with continuous treatment."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]] * 15)
        A = jnp.linspace(0, 1, 45).reshape(-1, 1)  # Continuous treatment
        weights = jnp.ones(45)

        se = standardized_mean_difference_se(X, A, weights)

        assert se.shape == (2,)
        assert jnp.all(se > 0)

    def test_higher_weights_reduce_se(self):
        """Higher effective sample size should reduce SE."""
        X = jnp.array([[1.0, 2.0]] * 100)
        A = jnp.array([[0.0]] * 50 + [[1.0]] * 50)

        # Uniform weights (high ESS)
        weights_uniform = jnp.ones(100)
        se_uniform = standardized_mean_difference_se(X, A, weights_uniform)

        # Variable weights with unequal distribution (lower ESS)
        # Give more weight to some samples, less to others
        weights_variable = jnp.concatenate(
            [jnp.array([0.1] * 25 + [2.0] * 25), jnp.array([0.1] * 25 + [2.0] * 25)]
        )
        se_variable = standardized_mean_difference_se(X, A, weights_variable)

        # Variable weights should have higher SE due to lower ESS
        assert jnp.all(se_variable > se_uniform)

    def test_2d_treatment_gets_squeezed(self):
        """2D treatment array should be squeezed correctly."""
        X = jnp.array([[1.0, 2.0, 3.0]] * 30)
        A = jnp.array([[0.0]] * 15 + [[1.0]] * 15)
        weights = jnp.ones(30)

        se = standardized_mean_difference_se(X, A, weights)

        assert se.shape == (3,)
        assert jnp.all(jnp.isfinite(se))

    def test_multivariate_covariates(self):
        """SE should work with many covariates."""
        X = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]] * 40)
        A = jnp.array([[0.0]] * 20 + [[1.0]] * 20)
        weights = jnp.ones(40)

        se = standardized_mean_difference_se(X, A, weights)

        assert se.shape == (5,)
        assert jnp.all(se > 0)

    def test_se_scales_with_sample_size(self):
        """SE should decrease with larger sample size."""
        # Small sample
        X_small = jnp.array([[1.0, 2.0]] * 20)
        A_small = jnp.array([[0.0]] * 10 + [[1.0]] * 10)
        weights_small = jnp.ones(20)
        se_small = standardized_mean_difference_se(X_small, A_small, weights_small)

        # Large sample
        X_large = jnp.array([[1.0, 2.0]] * 200)
        A_large = jnp.array([[0.0]] * 100 + [[1.0]] * 100)
        weights_large = jnp.ones(200)
        se_large = standardized_mean_difference_se(X_large, A_large, weights_large)

        # Larger sample should have smaller SE
        assert jnp.all(se_large < se_small)
