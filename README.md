# stochpw - Permutation Weighting for Causal Inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-green.svg)](https://github.com/google/jax)

**Permutation weighting** learns importance weights for causal inference by training a discriminator to distinguish between observed treatment-covariate pairs and artificially permuted pairs.

## Installation

```bash
pip install stochpw  # Coming soon
```

For development:
```bash
git clone https://github.com/yourusername/stochpw.git
cd stochpw
poetry install
```

## Quick Start

```python
import jax.numpy as jnp
from stochpw import PermutationWeighter

# Your observational data
X = jnp.array(...)  # Covariates, shape (n_samples, n_features)
A = jnp.array(...)  # Treatments, shape (n_samples, 1)

# Fit permutation weighter (sklearn-style API)
weighter = PermutationWeighter(
    num_epochs=100,
    batch_size=256,
    random_state=42
)
weighter.fit(X, A)

# Predict importance weights
weights = weighter.predict(X, A)

# Use weights for causal inference (in external package)
# ate = weighted_estimator(Y, A, weights)
```

## How It Works

Permutation weighting estimates density ratios by:

1. **Training a discriminator** to distinguish:
   - Permuted pairs: (X, A') with label C=1 (treatments shuffled)
   - Observed pairs: (X, A) with label C=0 (original data)

2. **Extracting weights** from discriminator probabilities:
   ```
   w(a, x) = η(a, x) / (1 - η(a, x))
   ```
   where η(a, x) = p(C=1 | a, x)

3. **Using weights** for inverse probability weighting in causal effect estimation

## Composable Design

The package exposes low-level components for integration into larger models:

```python
from stochpw import (
    create_training_batch,
    logistic_loss,
    extract_weights,
    create_linear_discriminator
)

# Use in your custom architecture (e.g., DragonNet)
batch = create_training_batch(X, A, batch_indices, rng_key)
logits = my_discriminator(params, batch.A, batch.X)
loss = logistic_loss(logits, batch.C)
```

## Features

- **JAX-based**: Fast, GPU-compatible, auto-differentiable
- **Sklearn-style API**: Familiar `.fit()` and `.predict()` interface
- **Composable**: All components exposed for integration
- **Flexible**: Supports binary, continuous, and multi-dimensional treatments
- **Diagnostic tools**: ESS, SMD, and balance checks included

## References

Arbour, D., Dimmery, D., & Sondhi, A. (2021). **Permutation Weighting**. *International Conference on Machine Learning (ICML)*.

## License

MIT License - see LICENSE file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this package, please cite:

```bibtex
@software{stochpw2024,
  title = {stochpw: Permutation Weighting for Causal Inference},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/stochpw}
}
```
