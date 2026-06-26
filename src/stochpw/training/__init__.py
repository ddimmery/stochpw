"""Training utilities for permutation weighting discriminators."""

from .batch import create_training_batch
from .early_stopping import BaseEarlyStopping, EarlyStopping, NoEarlyStopping
from .loop import fit_discriminator, make_scan_epoch, make_train_step, train_step
from .losses import BaseLoss, BrierLoss, ExponentialLoss, LogisticLoss
from .permutation import BasePermuter, RandomPermuter
from .regularization import BaseRegularizer, EntropyRegularizer, LpRegularizer, NoRegularizer

__all__ = [
    # Batch creation
    "create_training_batch",
    # Training loop
    "train_step",
    "make_train_step",
    "make_scan_epoch",
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
]
