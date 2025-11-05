"""Type definitions for stochpw.

This module provides type aliases for JAX-specific types to improve type safety
and reduce reliance on `Any` types throughout the codebase.
"""

from collections.abc import Callable
from typing import Any, TypeAlias

import jax.numpy as jnp
from jax import Array

# JAX array type (already available from jax import)
ArrayLike = Array | jnp.ndarray

# PyTree types - JAX PyTrees are recursive structures
# For discriminators: typically dict[str, Array] or dict[str, dict[str, Array]]
# We use a TypeAlias for clarity while accepting the limitation that PyTrees
# are fundamentally dynamic structures in JAX.
# Note: Using Any here is unavoidable as JAX PyTrees are inherently dynamic.
PyTree: TypeAlias = dict[str, Any] | list[Any] | tuple[Any, ...] | Array  # type: ignore[misc]

# Discriminator parameter types
# Linear discriminator: {"w_a": Array, "w_x": Array, "w_ax": Array, "b": Array}
LinearParams: TypeAlias = dict[str, Array]

# MLP discriminator: {"layers": [{"w": Array, "b": Array}, ...]}
MLPLayer: TypeAlias = dict[str, Array]
MLPParams: TypeAlias = dict[str, list[MLPLayer]]

# Generic discriminator params (union of known types)
DiscriminatorParams: TypeAlias = LinearParams | MLPParams | PyTree

# Type alias for optimizer state (from optax)
# Unfortunately optax doesn't provide good types yet, so we must use Any
OptimizerState: TypeAlias = Any  # type: ignore[misc]

# Type aliases for loss functions
# Loss functions map (logits, labels) -> scalar loss
LossFn: TypeAlias = Callable[[Array, Array], Array]

# Type aliases for gradient functions
# Value-and-gradient functions return (loss_value, gradients)
ValueAndGradFn: TypeAlias = Callable[[PyTree, Array, Array], tuple[Array, PyTree]]

# Type aliases for diagnostic/reporting structures
# Balance report contains various metrics about the weighting quality
BalanceReport: TypeAlias = dict[str, float | Array | dict[str, float] | str | int]

# Training history contains lists of metrics per epoch
TrainingHistory: TypeAlias = dict[str, list[float]]
