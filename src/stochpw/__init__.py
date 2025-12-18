"""stochpw - Permutation weighting for causal inference.

This package implements permutation weighting, a method for learning density ratios
via discriminative classification. It trains a discriminator to distinguish between
observed (X, A) pairs and permuted (X, A') pairs, then extracts importance weights
from the discriminator's predictions.

The package provides both a high-level sklearn-style API and low-level composable
components for integration into larger causal inference models.
"""

__version__ = "0.3.0"

# Main API
from .core import NotFittedError, PermutationWeighter
from .data import TrainingBatch, TrainingState, TrainingStepResult, WeightedData
from .diagnostics import (
    balance_report,
    calibration_curve,
    effective_sample_size,
    maximum_mean_discrepancy,
    roc_curve,
    standardized_mean_difference,
    standardized_mean_difference_se,
    weight_statistics,
)
from .models import BaseDiscriminator, LinearDiscriminator, MLPDiscriminator

# Low-level components for composability
from .training import (
    BaseEarlyStopping,
    BaseLoss,
    BasePermuter,
    BaseRegularizer,
    BrierLoss,
    EarlyStopping,
    EntropyRegularizer,
    ExponentialLoss,
    LogisticLoss,
    LpRegularizer,
    NoEarlyStopping,
    NoRegularizer,
    RandomPermuter,
    create_training_batch,
    fit_discriminator,
    train_step,
)
from .types import (
    BalanceReport,
    DiscriminatorParams,
    LinearParams,
    LossFn,
    MLPParams,
    OptimizerState,
    PyTree,
    TrainingHistory,
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
    "train_step",
    "fit_discriminator",
    # Loss functions
    "BaseLoss",
    "LogisticLoss",
    "ExponentialLoss",
    "BrierLoss",
    # Regularization
    "BaseRegularizer",
    "NoRegularizer",
    "EntropyRegularizer",
    "LpRegularizer",
    # Early stopping
    "BaseEarlyStopping",
    "NoEarlyStopping",
    "EarlyStopping",
    # Permutation
    "BasePermuter",
    "RandomPermuter",
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
    "maximum_mean_discrepancy",
    "weight_statistics",
    "balance_report",
    "calibration_curve",
    "roc_curve",
    # Utilities
    "validate_inputs",
    "permute_treatment",
    # Type aliases (for type annotations in user code)
    "PyTree",
    "LinearParams",
    "MLPParams",
    "DiscriminatorParams",
    "OptimizerState",
    "LossFn",
    "TrainingHistory",
    "BalanceReport",
]
