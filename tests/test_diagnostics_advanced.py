"""Tests for advanced diagnostics functionality."""

import jax
import jax.numpy as jnp
import optax

from stochpw import balance_report, calibration_curve, weight_statistics


class TestCalibrationCurve:
    """Test calibration curve computation."""

    def test_perfect_calibration(self):
        """Test with perfectly calibrated predictions."""
        # Create perfectly calibrated predictions
        probs = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 10)
        labels = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1] * 10)

        bin_centers, true_freqs, counts = calibration_curve(probs, labels, num_bins=5)

        assert bin_centers.shape == (5,)
        assert true_freqs.shape == (5,)
        assert counts.shape == (5,)
        # All bins should have samples
        assert jnp.all(counts > 0)

    def test_poor_calibration(self):
        """Test with poorly calibrated predictions."""
        # Always predict high probability but labels are random
        probs = jnp.ones(100) * 0.9
        labels = jnp.array([0, 1] * 50)

        bin_centers, true_freqs, counts = calibration_curve(probs, labels, num_bins=10)

        # Most samples should be in high probability bin
        assert jnp.max(counts) > 50

    def test_num_bins(self):
        """Test different number of bins."""
        probs = jnp.linspace(0, 1, 100)
        labels = (probs > 0.5).astype(float)

        for num_bins in [5, 10, 20]:
            bin_centers, true_freqs, counts = calibration_curve(probs, labels, num_bins=num_bins)
            assert len(bin_centers) == num_bins
            assert len(true_freqs) == num_bins
            assert len(counts) == num_bins

    def test_edge_cases(self):
        """Test edge cases with extreme probabilities."""
        # All predictions at 0 or 1
        probs = jnp.array([0.0, 0.0, 1.0, 1.0])
        labels = jnp.array([0, 1, 0, 1])

        bin_centers, true_freqs, counts = calibration_curve(probs, labels, num_bins=5)

        assert jnp.all(jnp.isfinite(true_freqs))
        assert jnp.sum(counts) == len(probs)


class TestWeightStatistics:
    """Test weight statistics computation."""

    def test_uniform_weights(self):
        """Test with uniform weights."""
        weights = jnp.ones(100)
        stats = weight_statistics(weights)

        assert stats["mean"] == 1.0
        assert stats["std"] == 0.0
        assert stats["min"] == 1.0
        assert stats["max"] == 1.0
        assert stats["cv"] == 0.0
        assert stats["max_ratio"] == 1.0
        assert stats["n_extreme"] == 0

    def test_variable_weights(self):
        """Test with variable weights."""
        key = jax.random.PRNGKey(42)
        weights = jax.random.uniform(key, (100,), minval=0.1, maxval=2.0)
        stats = weight_statistics(weights)

        assert stats["mean"] > 0
        assert stats["std"] > 0
        assert stats["min"] < stats["max"]
        assert stats["cv"] > 0
        assert stats["max_ratio"] > 1
        assert 0 <= stats["entropy"] <= jnp.log(100)

    def test_extreme_weights(self):
        """Test detection of extreme weights."""
        # Most weights are 1, but a few are very large
        weights = jnp.concatenate(
            [
                jnp.ones(97),
                jnp.array([50.0, 100.0, 150.0]),  # 3 extreme weights
            ]
        )
        stats = weight_statistics(weights)

        # Mean is approximately 2.0, so weights > 20 are extreme
        # Should detect at least the 3 very large weights
        assert stats["n_extreme"] >= 3
        assert stats["max_ratio"] > 100

    def test_all_stats_present(self):
        """Test that all expected statistics are computed."""
        weights = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = weight_statistics(weights)

        required_keys = ["mean", "std", "min", "max", "cv", "entropy", "max_ratio", "n_extreme"]
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], float) or isinstance(stats[key], int)

    def test_entropy_calculation(self):
        """Test entropy is calculated correctly."""
        # Uniform weights should have higher entropy
        uniform_weights = jnp.ones(100)
        uniform_stats = weight_statistics(uniform_weights)

        # Concentrated weights should have lower entropy
        concentrated_weights = jnp.concatenate([jnp.array([100.0]), jnp.ones(99)])
        concentrated_stats = weight_statistics(concentrated_weights)

        assert uniform_stats["entropy"] > concentrated_stats["entropy"]


class TestBalanceReport:
    """Test comprehensive balance report."""

    def test_binary_treatment(self):
        """Test balance report with binary treatment."""
        # Create simple data
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100, 5))
        A = jax.random.bernoulli(key, 0.5, (100,))
        weights = jnp.ones(100)

        report = balance_report(X, A, weights)

        # Check all expected keys are present
        assert "smd" in report
        assert "max_smd" in report
        assert "mean_smd" in report
        assert "ess" in report
        assert "ess_ratio" in report
        assert "weight_stats" in report
        assert "n_samples" in report
        assert "n_features" in report
        assert "treatment_type" in report

        # Check values
        assert report["treatment_type"] == "binary"
        assert report["n_samples"] == 100
        assert report["n_features"] == 5
        assert report["smd"].shape == (5,)
        assert report["ess"] == 100.0  # Uniform weights
        assert report["ess_ratio"] == 1.0

    def test_continuous_treatment(self):
        """Test balance report with continuous treatment."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100, 3))
        A = jax.random.normal(key, (100,))
        weights = jnp.ones(100)

        report = balance_report(X, A, weights)

        assert report["treatment_type"] == "continuous"
        assert report["n_samples"] == 100
        assert report["n_features"] == 3

    def test_weighted_vs_unweighted(self):
        """Test that different weights produce different reports."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100, 3))
        A = jax.random.bernoulli(key, 0.5, (100,))

        # Uniform weights
        report_uniform = balance_report(X, A, jnp.ones(100))

        # Variable weights
        weights_var = jax.random.uniform(key, (100,))
        report_weighted = balance_report(X, A, weights_var)

        # ESS should be different
        assert report_uniform["ess"] > report_weighted["ess"]
        # Weight stats should be different
        assert report_uniform["weight_stats"]["cv"] == 0.0
        assert report_weighted["weight_stats"]["cv"] > 0.0

    def test_weight_statistics_included(self):
        """Test that weight statistics are properly included."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100, 3))
        A = jax.random.bernoulli(key, 0.5, (100,))
        weights = jax.random.uniform(key, (100,))

        report = balance_report(X, A, weights)

        # Check weight stats are included
        assert isinstance(report["weight_stats"], dict)
        assert "mean" in report["weight_stats"]
        assert "std" in report["weight_stats"]
        assert "cv" in report["weight_stats"]

    def test_2d_treatment_array(self):
        """Test with 2D treatment array (shape n, 1)."""
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100, 3))
        A = jax.random.bernoulli(key, 0.5, (100, 1))  # 2D array
        weights = jnp.ones(100)

        report = balance_report(X, A, weights)

        # Should handle 2D treatment array correctly
        assert report["treatment_type"] == "binary"
        assert report["n_samples"] == 100


class TestIntegrationWithPermutationWeighter:
    """Test new diagnostics with PermutationWeighter."""

    def test_balance_report_after_fit(self):
        """Test balance report with actual permutation weighting."""
        from stochpw import PermutationWeighter

        # Create confounded data
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (200, 5))
        # Treatment depends on X[0]
        A = (X[:, 0] + jax.random.normal(key, (200,)) * 0.5 > 0).astype(float)

        # Fit weighter
        weighter = PermutationWeighter(
            num_epochs=5,
            batch_size=50,
            random_state=42,
            optimizer=optax.rmsprop(learning_rate=0.1),
        )
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        # Generate balance report
        report_before = balance_report(X, A, jnp.ones(200))
        report_after = balance_report(X, A, weights)

        # Balance should improve
        assert report_after["max_smd"] < report_before["max_smd"]
        # ESS should be less than n (weights are not uniform)
        assert report_after["ess"] < 200

    def test_calibration_with_discriminator(self):
        """Test calibration curve with discriminator predictions."""
        from stochpw import LinearDiscriminator

        # Create simple discriminator
        discriminator = LinearDiscriminator()

        key = jax.random.PRNGKey(42)
        d_a, d_x = 1, 3
        params = discriminator.init_params(key, d_a, d_x)

        # Generate predictions
        A = jax.random.normal(key, (100, 1))
        X = jax.random.normal(key, (100, 3))
        AX = jnp.einsum("bi,bj->bij", A, X).reshape(100, -1)

        logits = discriminator.apply(params, A, X, AX)
        probs = jax.nn.sigmoid(logits)

        # Create labels (random for this test)
        labels = jax.random.bernoulli(key, 0.5, (100,))

        # Should not error
        bin_centers, true_freqs, counts = calibration_curve(probs, labels, num_bins=5)

        assert bin_centers.shape == (5,)
        assert jnp.all(jnp.isfinite(true_freqs))
