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
from .diagnostics import effective_sample_size, standardized_mean_difference
from .models import BaseDiscriminator, LinearDiscriminator, MLPDiscriminator

# Low-level components for composability
from .training import create_training_batch, fit_discriminator, logistic_loss, train_step
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
    # Utilities
    "validate_inputs",
    "permute_treatment",
]
