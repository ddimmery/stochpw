"""Early stopping for discriminator training."""

from abc import ABC, abstractmethod

from ..types import PyTree


class BaseEarlyStopping(ABC):
    """
    Abstract base class for early stopping strategies.

    All early stopping strategies should inherit from this class and
    implement the should_stop and update methods.
    """

    @abstractmethod
    def should_stop(self) -> bool:
        """
        Check whether training should stop.

        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        pass

    @abstractmethod
    def update(self, loss: float, params: PyTree) -> None:
        """
        Update early stopping state with new loss value.

        Parameters
        ----------
        loss : float
            Current epoch loss
        params : PyTree
            Current model parameters
        """
        pass

    @abstractmethod
    def get_best_params(self) -> PyTree | None:
        """
        Get the best parameters seen so far.

        Returns
        -------
        PyTree or None
            Best parameters, or None if no update has occurred
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset early stopping state."""
        pass


class EarlyStopping(BaseEarlyStopping):
    """
    Early stopping based on training loss.

    Monitors training loss and stops if no improvement is seen for
    a specified number of epochs (patience). Keeps track of the best
    parameters seen during training.

    Parameters
    ----------
    patience : int, default=10
        Number of epochs to wait for improvement before stopping
    min_delta : float, default=1e-4
        Minimum change in loss to qualify as improvement

    Examples
    --------
    >>> from stochpw.training import EarlyStopping
    >>> early_stop = EarlyStopping(patience=5, min_delta=1e-3)
    >>> # During training loop:
    >>> for epoch in range(100):
    ...     loss = train_epoch()
    ...     early_stop.update(loss, params)
    ...     if early_stop.should_stop():
    ...         best_params = early_stop.get_best_params()
    ...         break

    Attributes
    ----------
    best_loss : float
        Best loss value seen so far
    best_params : PyTree or None
        Parameters corresponding to best loss
    epochs_without_improvement : int
        Number of epochs since last improvement
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience: int = patience
        self.min_delta: float = min_delta

        self.best_loss: float = float("inf")
        self.best_params: PyTree | None = None
        self.epochs_without_improvement: int = 0

    def should_stop(self) -> bool:
        """Check if training should stop."""
        return self.epochs_without_improvement >= self.patience

    def update(self, loss: float, params: PyTree) -> None:
        """Update early stopping state with new loss."""
        if loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = loss
            self.best_params = params
            self.epochs_without_improvement = 0
        else:
            # No improvement
            self.epochs_without_improvement += 1

    def get_best_params(self) -> PyTree | None:
        """Get the best parameters seen so far."""
        return self.best_params

    def reset(self) -> None:
        """Reset early stopping state."""
        self.best_loss = float("inf")
        self.best_params = None
        self.epochs_without_improvement = 0


class NoEarlyStopping(BaseEarlyStopping):
    """
    No-op early stopping that never stops training.

    Useful as a default when early stopping is not desired,
    avoiding None checks throughout the codebase.

    Examples
    --------
    >>> from stochpw.training import NoEarlyStopping
    >>> early_stop = NoEarlyStopping()
    >>> # Will never trigger early stopping
    >>> early_stop.should_stop()  # Always returns False
    False
    """

    def should_stop(self) -> bool:
        """Never stop training."""
        return False

    def update(self, loss: float, params: PyTree) -> None:
        """No-op update."""
        pass

    def get_best_params(self) -> PyTree | None:
        """Return None (no best params tracked)."""
        return None

    def reset(self) -> None:
        """No-op reset."""
        pass
