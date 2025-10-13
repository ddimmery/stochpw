"""Training utilities for permutation weighting discriminators."""

from .batch import create_training_batch
from .loop import fit_discriminator, train_step
from .losses import brier_loss, exponential_loss, logistic_loss
from .regularization import entropy_penalty, lp_weight_penalty

__all__ = [
    # Batch creation
    "create_training_batch",
    # Training loop
    "train_step",
    "fit_discriminator",
    # Loss functions
    "logistic_loss",
    "exponential_loss",
    "brier_loss",
    # Regularization
    "entropy_penalty",
    "lp_weight_penalty",
]
