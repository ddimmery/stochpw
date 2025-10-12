"""stochpw - Permutation weighting for causal inference.

This package implements permutation weighting, a method for learning density ratios
via discriminative classification. It trains a discriminator to distinguish between
observed (X, A) pairs and permuted (X, A') pairs, then extracts importance weights
from the discriminator's predictions.

The package provides both a high-level sklearn-style API and low-level composable
components for integration into larger causal inference models.
"""

__version__ = "0.2.0"

# Main API
from .core import NotFittedError, PermutationWeighter
from .data import TrainingBatch, TrainingState, TrainingStepResult, WeightedData
from .diagnostics import (
    balance_report,
    calibration_curve,
    effective_sample_size,
    roc_curve,
    standardized_mean_difference,
    standardized_mean_difference_se,
    weight_statistics,
)
from .models import BaseDiscriminator, LinearDiscriminator, MLPDiscriminator

# Low-level components for composability
from .training import (
    brier_loss,
    create_training_batch,
    exponential_loss,
    fit_discriminator,
    l2_param_penalty,
    logistic_loss,
    train_step,
)
from .utils import permute_treatment, validate_inputs
from .weights import extract_weights

__all__ = [
    # Version
    "__version__",
    # Main API
    "PermutationWeighter",
    "NotFittedError",
    # Training utilities (for integration)
    "create_training_batch",
    "logistic_loss",
    "exponential_loss",
    "brier_loss",
    "l2_param_penalty",
    "train_step",
    "fit_discriminator",
    # Weight extraction
    "extract_weights",
    # Discriminator models
    "BaseDiscriminator",
    "LinearDiscriminator",
    "MLPDiscriminator",
    # Data structures
    "TrainingBatch",
    "WeightedData",
    "TrainingState",
    "TrainingStepResult",
    # Diagnostics
    "effective_sample_size",
    "standardized_mean_difference",
    "standardized_mean_difference_se",
    "weight_statistics",
    "balance_report",
    "calibration_curve",
    "roc_curve",
    # Utilities
    "validate_inputs",
    "permute_treatment",
]
