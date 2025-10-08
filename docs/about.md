---
hide:
  - navigation
---

## About the package

This package implements **permutation weighting** using stochastic gradient descent (SGD) with JAX.

### What is Permutation Weighting?

Permutation weighting is a method for learning importance weights for causal inference in observational studies. It was introduced in:

> Arbour, D., Dimmery, D., & Sondhi, A. (2021). Permutation Weighting. In *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139:331-341.

The key insight is to train a discriminator to distinguish between observed treatment-covariate pairs (X, A) and artificially permuted pairs (X, A'). The discriminator probabilities are then transformed into importance weights that can be used for causal effect estimation.

### Implementation Approach

This implementation uses:

- **JAX** for high-performance computing, automatic differentiation, and GPU acceleration
- **Optax** for flexible gradient-based optimization algorithms
- **Sklearn-style API** for familiar `.fit()` and `.predict()` interface
- **Class-based discriminators** allowing users to easily define custom architectures
- **Mini-batch SGD** enabling scalable training on large datasets

### Applications

Permutation weighting is particularly useful for:

- Learning importance weights for causal effect estimation in observational studies
- Handling arbitrary treatment types (binary, continuous, or multivariate)
- Balancing treatment and covariate distributions without parametric assumptions
- Situations where traditional propensity score methods may be insufficient
- Large-scale causal inference problems requiring scalable training

### Key Features

- **Flexible discriminators**: Linear and MLP architectures included, with easy extension to custom models
- **Composable design**: Low-level components exposed for integration into larger systems
- **Diagnostic tools**: Effective sample size (ESS) and standardized mean difference (SMD) for balance checking
- **Type-safe**: Full type annotations with pyright validation
- **Well-tested**: Comprehensive test suite with >99% coverage

## Original Paper Authors

- [David Arbour](https://darbour.github.io/)
- [Drew Dimmery](http://ddimmery.com)
- [Arjun Sondhi](https://asondhi.github.io/)
