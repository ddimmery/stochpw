"""Tests for stochpw.plotting module."""

import jax
import jax.numpy as jnp
from plotnine import ggplot
from stochpw.plotting import (
    plot_balance_diagnostics,
    plot_calibration_curve,
    plot_roc_curve,
    plot_training_history,
    plot_weight_distribution,
)


class TestPlotBalanceDiagnostics:
    """Tests for plot_balance_diagnostics function."""

    def test_basic_plot_creation(self):
        """Test that a basic plot is created successfully."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]] * 10)
        A = jnp.array([[0.0], [1.0], [0.0], [1.0]] * 10)
        weights = jnp.ones(40)

        plot = plot_balance_diagnostics(X, A, weights)

        assert isinstance(plot, ggplot)

    def test_with_custom_feature_names(self):
        """Test plot with custom feature names."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]] * 10)
        A = jnp.array([[0.0], [1.0], [0.0]] * 10)
        weights = jnp.ones(30)
        feature_names = ["Age", "Income"]

        plot = plot_balance_diagnostics(X, A, weights, feature_names=feature_names)

        assert isinstance(plot, ggplot)

    def test_with_unweighted_weights(self):
        """Test plot with explicit unweighted weights."""
        X = jnp.array([[1.0, 2.0], [2.0, 3.0]] * 15)
        A = jnp.array([[0.0], [1.0]] * 15)
        weights = jnp.array([1.5, 0.8] * 15)
        unweighted_weights = jnp.ones(30)

        plot = plot_balance_diagnostics(X, A, weights, unweighted_weights=unweighted_weights)

        assert isinstance(plot, ggplot)

    def test_with_1d_treatment(self):
        """Test plot with 1D treatment array."""
        X = jnp.array([[1.0, 2.0, 3.0]] * 20)
        A = jnp.array([0.0, 1.0] * 10)  # 1D array
        weights = jnp.ones(20)

        plot = plot_balance_diagnostics(X, A, weights)

        assert isinstance(plot, ggplot)

    def test_multivariate_covariates(self):
        """Test plot with many covariates."""
        X = jnp.array([[1.0, 2.0, 3.0, 4.0, 5.0]] * 20)
        A = jnp.array([[0.0]] * 10 + [[1.0]] * 10)
        weights = jnp.ones(20)
        feature_names = ["X1", "X2", "X3", "X4", "X5"]

        plot = plot_balance_diagnostics(X, A, weights, feature_names=feature_names)

        assert isinstance(plot, ggplot)


class TestPlotWeightDistribution:
    """Tests for plot_weight_distribution function."""

    def test_uniform_weights(self):
        """Test plot with uniform weights."""
        weights = jnp.ones(100)

        plot = plot_weight_distribution(weights)

        assert isinstance(plot, ggplot)

    def test_variable_weights(self):
        """Test plot with variable weights."""
        weights = jnp.array([0.5] * 30 + [1.0] * 40 + [2.0] * 30)

        plot = plot_weight_distribution(weights)

        assert isinstance(plot, ggplot)

    def test_extreme_weights(self):
        """Test plot with extreme weight values."""
        weights = jnp.array([0.1] * 10 + [1.0] * 80 + [10.0] * 10)

        plot = plot_weight_distribution(weights)

        assert isinstance(plot, ggplot)

    def test_small_sample_size(self):
        """Test plot with small number of samples."""
        weights = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        plot = plot_weight_distribution(weights)

        assert isinstance(plot, ggplot)


class TestPlotTrainingHistory:
    """Tests for plot_training_history function."""

    def test_basic_training_history(self):
        """Test plot with basic loss history."""
        history = {"loss": jnp.array([1.0, 0.8, 0.6, 0.5, 0.4])}

        plot = plot_training_history(history)

        assert isinstance(plot, ggplot)

    def test_decreasing_loss(self):
        """Test plot with decreasing loss over time."""
        history = {"loss": jnp.exp(-jnp.linspace(0, 3, 50))}

        plot = plot_training_history(history)

        assert isinstance(plot, ggplot)

    def test_single_epoch(self):
        """Test plot with single epoch."""
        history = {"loss": jnp.array([0.5])}

        plot = plot_training_history(history)

        assert isinstance(plot, ggplot)

    def test_long_training(self):
        """Test plot with long training history."""
        history = {"loss": jnp.linspace(1.0, 0.1, 200)}

        plot = plot_training_history(history)

        assert isinstance(plot, ggplot)


class TestPlotCalibrationCurve:
    """Tests for plot_calibration_curve function."""

    def test_perfect_calibration(self):
        """Test plot with perfect calibration."""
        bin_centers = jnp.linspace(0, 1, 10)
        true_frequencies = bin_centers  # Perfect calibration
        counts = jnp.ones(10) * 10

        plot = plot_calibration_curve(bin_centers, true_frequencies, counts)

        assert isinstance(plot, ggplot)

    def test_poor_calibration(self):
        """Test plot with poor calibration."""
        bin_centers = jnp.linspace(0, 1, 10)
        true_frequencies = jnp.array([0.1, 0.2, 0.3, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8])
        counts = jnp.ones(10) * 15

        plot = plot_calibration_curve(bin_centers, true_frequencies, counts)

        assert isinstance(plot, ggplot)

    def test_with_empty_bins(self):
        """Test plot with some empty bins (zero counts)."""
        bin_centers = jnp.linspace(0, 1, 10)
        true_frequencies = jnp.linspace(0, 1, 10)
        counts = jnp.array([10, 0, 5, 0, 8, 12, 0, 6, 9, 11])  # Some empty bins

        plot = plot_calibration_curve(bin_centers, true_frequencies, counts)

        assert isinstance(plot, ggplot)

    def test_few_bins(self):
        """Test plot with few bins."""
        bin_centers = jnp.array([0.25, 0.75])
        true_frequencies = jnp.array([0.2, 0.8])
        counts = jnp.array([50, 50])

        plot = plot_calibration_curve(bin_centers, true_frequencies, counts)

        assert isinstance(plot, ggplot)


class TestPlotROCCurve:
    """Tests for plot_roc_curve function."""

    def test_perfect_classification(self):
        """Test plot with perfect ROC curve."""
        fpr = jnp.array([0.0, 0.0, 1.0])
        tpr = jnp.array([0.0, 1.0, 1.0])

        plot = plot_roc_curve(fpr, tpr)

        assert isinstance(plot, ggplot)

    def test_random_classification(self):
        """Test plot with random classifier."""
        fpr = jnp.linspace(0, 1, 50)
        tpr = jnp.linspace(0, 1, 50)

        plot = plot_roc_curve(fpr, tpr)

        assert isinstance(plot, ggplot)

    def test_with_explicit_auc(self):
        """Test plot with explicitly provided AUC."""
        fpr = jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        tpr = jnp.array([0.0, 0.5, 0.7, 0.85, 0.95, 1.0])
        auc = 0.85

        plot = plot_roc_curve(fpr, tpr, auc=auc)

        assert isinstance(plot, ggplot)

    def test_without_auc(self):
        """Test plot computes AUC when not provided."""
        fpr = jnp.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
        tpr = jnp.array([0.0, 0.4, 0.6, 0.8, 0.9, 1.0])

        plot = plot_roc_curve(fpr, tpr)  # No AUC provided

        assert isinstance(plot, ggplot)

    def test_good_discriminator(self):
        """Test plot with good discriminator performance."""
        # Simulate a good discriminator
        fpr = jnp.array([0.0, 0.05, 0.1, 0.2, 0.3, 1.0])
        tpr = jnp.array([0.0, 0.6, 0.8, 0.9, 0.95, 1.0])
        auc = 0.92

        plot = plot_roc_curve(fpr, tpr, auc=auc)

        assert isinstance(plot, ggplot)

    def test_many_points(self):
        """Test plot with many ROC points."""
        fpr = jnp.linspace(0, 1, 200)
        # Create a good ROC curve
        tpr = jnp.sqrt(fpr)  # Concave curve above diagonal

        plot = plot_roc_curve(fpr, tpr)

        assert isinstance(plot, ggplot)


class TestIntegrationWithPermutationWeighter:
    """Integration tests using PermutationWeighter."""

    def test_plot_after_fitting(self):
        """Test creating plots after fitting a weighter."""
        from stochpw import PermutationWeighter

        # Generate simple data
        key = jax.random.PRNGKey(42)
        X = jax.random.normal(key, (100, 3))
        A = jax.random.bernoulli(key, 0.5, (100, 1)).astype(float)

        # Fit weighter
        weighter = PermutationWeighter(num_epochs=10, batch_size=32, random_state=42)
        weighter.fit(X, A)
        weights = weighter.predict(X, A)

        # Test all plotting functions
        plot1 = plot_balance_diagnostics(X, A, weights)
        plot2 = plot_weight_distribution(weights)
        plot3 = plot_training_history(weighter.history_)

        assert isinstance(plot1, ggplot)
        assert isinstance(plot2, ggplot)
        assert isinstance(plot3, ggplot)
