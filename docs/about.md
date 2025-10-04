---
hide:
  - navigation
---

## About the package

This package implements **permutation weighting** using stochastic gradient descent (SGD) with NumPyro.

### What is Permutation Weighting?

Permutation weighting is a method for estimating average treatment effects (ATEs) in observational studies. It was introduced in:

> Arbour, D., Dimmery, D., & Sondhi, A. (2021). Permutation Weighting. *International Conference on Machine Learning (ICML)*.

The key insight of permutation weighting is to learn weights that balance treatment and control groups across all possible permutations of the observed data. This provides a principled approach to causal inference that goes beyond traditional propensity score methods.

### Why SGD?

The original paper suggested that stochastic gradient descent could be used to train permutation weighting models. This implementation realizes that vision by:

- Using **NumPyro** for probabilistic modeling with automatic differentiation
- Leveraging **JAX** for high-performance computing and GPU acceleration
- Employing **Optax** for flexible optimization algorithms
- Enabling scalable training on large datasets through mini-batch SGD

### Applications

Permutation weighting is particularly useful for:

- Estimating causal effects in observational studies
- Handling high-dimensional covariate spaces
- Situations where traditional matching or weighting methods struggle
- Large-scale causal inference problems

## Authors

- [Drew Dimmery](http://ddimmery.com)
- [David Arbour](https://darbour.github.io/)
- [Arjun Sondhi](https://scholar.google.com/citations?user=9fF8qNEAAAAJ)
