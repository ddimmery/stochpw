# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-06

### Added

- **Class-based discriminator architecture**: Introduced `BaseDiscriminator` abstract base class defining a consistent interface for all discriminator models
- **MLPDiscriminator**: Multi-layer perceptron discriminator with configurable architecture
  - Customizable hidden layer dimensions via `hidden_dims` parameter
  - Multiple activation function options: 'relu', 'tanh', 'elu', 'sigmoid'
  - Smart weight initialization (He for ReLU/ELU, Xavier for tanh/sigmoid)
  - Processes concatenated [A, X, A*X] features through neural network layers
- **LinearDiscriminator class**: Object-oriented version of the linear discriminator
  - Replaces functional `create_linear_discriminator` API
  - Maintains same functionality: uses A, X, and A*X interactions
  - Xavier/Glorot initialization for weights
- **Enhanced examples**: New `examples/mlp_discriminator.py` demonstrating:
  - Comparison of Linear vs MLP discriminators on nonlinear confounding
  - Different MLP architectures and activation functions
  - Performance benchmarking across discriminator types
- **Extensibility**: Users can now create custom discriminators by subclassing `BaseDiscriminator`

### Changed

- **BREAKING**: Refactored discriminator API from functional to class-based
  - **Before**: `init_fn, apply_fn = create_linear_discriminator(d_a, d_x)`
  - **After**: `discriminator = LinearDiscriminator()`
- **BREAKING**: `PermutationWeighter` now accepts discriminator class instances instead of factory functions
  - **Before**: `PermutationWeighter(discriminator=create_linear_discriminator)`
  - **After**: `PermutationWeighter(discriminator=LinearDiscriminator())`
  - Default behavior unchanged: uses `LinearDiscriminator()` when no discriminator specified
- **API simplification**: Removed discriminator-specific parameters from `PermutationWeighter`
  - Configuration now encapsulated in discriminator classes
  - Example: `MLPDiscriminator(hidden_dims=[128, 64], activation='tanh')`
- **Module reorganization**: Discriminator models moved to submodule
  - **Before**: `src/stochpw/models.py`
  - **After**: `src/stochpw/models/` (base.py, linear.py, mlp.py)

### Removed

- **BREAKING**: Removed functional discriminator API
  - `create_linear_discriminator()` - replaced by `LinearDiscriminator` class
  - `create_mlp_discriminator()` - replaced by `MLPDiscriminator` class

### Improved

- **Test coverage**: Increased from 100% (241 statements) to 99.34% (301 statements)
  - Added 23 new tests (129 total, up from 106)
  - Comprehensive testing for MLP discriminator (14 tests)
  - Enhanced LinearDiscriminator tests (11 tests)
  - Activation function utility tests (5 tests)
- **Type safety**: Full type annotations with abstract base class ensuring interface compliance
- **Documentation**: Enhanced docstrings with detailed parameter descriptions and usage examples
- **Code organization**: Better separation of concerns with discriminator-specific logic encapsulated in classes

### Migration Guide

For users upgrading from v0.1.0:

**Basic usage (unchanged)**:
```python
# This still works - default linear discriminator
weighter = PermutationWeighter()
weighter.fit(X, A)
weights = weighter.predict(X, A)
```

**Custom discriminator (updated API)**:
```python
# v0.1.0 (old)
from stochpw.models import create_linear_discriminator
def custom_disc(d_a, d_x):
    return create_linear_discriminator(d_a, d_x)
weighter = PermutationWeighter(discriminator=custom_disc)

# v0.2.0 (new)
from stochpw import LinearDiscriminator
weighter = PermutationWeighter(discriminator=LinearDiscriminator())
```

**Using MLP discriminator (new in v0.2.0)**:
```python
from stochpw import PermutationWeighter, MLPDiscriminator

# Default MLP (hidden_dims=[64, 32], activation='relu')
mlp = MLPDiscriminator()
weighter = PermutationWeighter(discriminator=mlp)

# Custom MLP architecture
mlp = MLPDiscriminator(hidden_dims=[128, 64, 32], activation='tanh')
weighter = PermutationWeighter(discriminator=mlp)
```

## [0.1.0] - 2025-01-06

### Added

- Initial release with core permutation weighting functionality
