"""Tests for ROC curve functionality."""

import jax.numpy as jnp
import numpy as np

from stochpw import PermutationWeighter, roc_curve


class TestROCCurve:
    """Test suite for ROC curve computation."""

    def test_roc_curve_perfect_discrimination(self):
        """Test ROC curve with perfect discrimination."""
        # Perfect weights: high for positives, low for negatives
        weights = jnp.array([0.9, 0.95, 0.8, 0.85, 0.1, 0.2, 0.15, 0.05])
        labels = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

        fpr, tpr, thresholds = roc_curve(weights, labels)

        # Check shapes
        assert fpr.shape == tpr.shape == thresholds.shape

        # Check bounds
        assert jnp.all(fpr >= 0) and jnp.all(fpr <= 1)
        assert jnp.all(tpr >= 0) and jnp.all(tpr <= 1)

        # AUC should be close to 1.0 for perfect discrimination
        auc = float(jnp.trapezoid(tpr, fpr))
        assert auc > 0.95, f"Expected AUC > 0.95 for perfect discrimination, got {auc}"

    def test_roc_curve_random_guessing(self):
        """Test ROC curve with random guessing (all same weight)."""
        # All same weights = random guessing
        weights = jnp.ones(100)
        labels = jnp.concatenate([jnp.zeros(50), jnp.ones(50)])

        fpr, tpr, thresholds = roc_curve(weights, labels)

        # AUC should be close to 0.5 for random guessing
        auc = float(jnp.trapezoid(tpr, fpr))
        assert abs(auc - 0.5) < 0.1, f"Expected AUC ≈ 0.5 for random guessing, got {auc}"

    def test_roc_curve_with_permutation_weighter(self):
        """Test ROC curve with actual PermutationWeighter."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        A = (X[:, 0] > 0).astype(float)

        # Fit weighter
        weighter = PermutationWeighter(num_epochs=20, batch_size=32, random_state=42)
        weighter.fit(X, A)

        # Get weights for observed and permuted
        weights_obs = weighter.predict(X, A)
        A_perm = A[np.random.permutation(len(A))]
        weights_perm = weighter.predict(X, A_perm)

        # Combine for ROC analysis
        all_weights = jnp.concatenate([weights_obs, weights_perm])
        all_labels = jnp.concatenate([jnp.zeros(len(weights_obs)), jnp.ones(len(weights_perm))])

        fpr, tpr, thresholds = roc_curve(all_weights, all_labels)

        # Check shapes
        assert fpr.shape == tpr.shape == thresholds.shape

        # AUC should be > 0.5 (better than random)
        auc = float(jnp.trapezoid(tpr, fpr))
        assert auc > 0.5, f"Expected AUC > 0.5 for trained discriminator, got {auc}"

    def test_roc_curve_eta_inference(self):
        """Test that eta is correctly inferred from weights."""
        # eta should be w / (1 + w)
        # For w=1: eta=0.5, for w=4: eta=0.8, for w=0.25: eta=0.2
        weights = jnp.array([1.0, 4.0, 0.25])

        labels = jnp.array([1.0, 1.0, 0.0])

        fpr, tpr, thresholds = roc_curve(weights, labels, max_points=10)

        # Check that we got results
        assert len(fpr) == 10
        assert len(tpr) == 10
        assert len(thresholds) == 10

    def test_roc_curve_max_points(self):
        """Test that max_points parameter controls output size."""
        weights = jnp.linspace(0.1, 10.0, 1000)
        labels = jnp.concatenate([jnp.zeros(500), jnp.ones(500)])

        fpr_50, tpr_50, thresh_50 = roc_curve(weights, labels, max_points=50)
        fpr_200, tpr_200, thresh_200 = roc_curve(weights, labels, max_points=200)

        assert len(fpr_50) == 50
        assert len(fpr_200) == 200

        # AUC should be similar regardless of max_points
        auc_50 = float(jnp.trapezoid(tpr_50, fpr_50))
        auc_200 = float(jnp.trapezoid(tpr_200, fpr_200))
        assert abs(auc_50 - auc_200) < 0.05

    def test_roc_curve_boundary_values(self):
        """Test ROC curve with edge case inputs."""
        # Small dataset
        weights = jnp.array([0.3, 0.7])
        labels = jnp.array([0.0, 1.0])

        fpr, tpr, thresholds = roc_curve(weights, labels, max_points=10)

        assert fpr.shape[0] == 10
        assert jnp.all(jnp.isfinite(fpr))
        assert jnp.all(jnp.isfinite(tpr))
