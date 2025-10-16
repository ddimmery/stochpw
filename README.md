# stochpw - Permutation Weighting for Causal Inference

[![PyPI version](https://img.shields.io/pypi/v/stochpw.svg)](https://pypi.org/project/stochpw/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/ddimmery/stochpw/workflows/CI/badge.svg)](https://github.com/ddimmery/stochpw/actions)
[![codecov](https://codecov.io/gh/ddimmery/stochpw/branch/main/graph/badge.svg)](https://codecov.io/gh/ddimmery/stochpw)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Permutation weighting** learns importance weights for causal inference by training a discriminator to distinguish between observed treatment-covariate pairs and artificially permuted pairs.

## Installation

```bash
pip install stochpw
```

For development:
```bash
git clone https://github.com/ddimmery/stochpw.git
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

# Use weights for downstream task
# (tools for causal estimation not provided)
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

3. **Using weights** for balancing weights in causal effect estimation

## Composable Design

The package exposes low-level components for integration into larger models:

```python
from stochpw import (
    BaseDiscriminator,
    LinearDiscriminator,
    MLPDiscriminator,
    create_training_batch,
    logistic_loss,
    extract_weights,
)

# Use in your custom architecture (e.g., DragonNet)
batch = create_training_batch(X, A, batch_indices, rng_key)
logits = my_discriminator(params, batch.A, batch.X, batch.AX)
loss = logistic_loss(logits, batch.C)
```

## Features

- **JAX-based**: Fast, GPU-compatible, auto-differentiable
- **Sklearn-style API**: Familiar `.fit()` and `.predict()` interface
- **Composable**: All components exposed for integration
- **Flexible**: Supports binary, continuous, and multi-dimensional treatments
- **Diagnostic tools**: ESS, SMD, and balance checks included

## References

Arbour, D., Dimmery, D., & Sondhi, A. (2021). **Permutation Weighting**. In *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139:331-341.

## License

Apache-2.0 License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this package, please cite [the original paper](https://proceedings.mlr.press/v139/arbour21a.html):

```bibtex
@InProceedings{arbour21permutation,
  title = {Permutation Weighting},
  author = {Arbour, David and Dimmery, Drew and Sondhi, Arjun},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages = {331--341},
  year = {2021},
  editor = {Meila, Marina and Zhang, Tong},
  volume = {139},
  series = {Proceedings of Machine Learning Research},
  month = {18--24 Jul},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v139/arbour21a/arbour21a.pdf},
  url = {https://proceedings.mlr.press/v139/arbour21a.html}
}
```
