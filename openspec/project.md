# Project Context

## Purpose
stochpw is a JAX-based implementation of permutation weighting for causal
inference. It learns importance weights by training a discriminator to
distinguish observed (X, A) pairs from within-batch permuted (X, A') pairs.

## Tech Stack
- JAX / jaxlib (functional, JIT, autodiff), Optax (optimizers)
- NumPy, plotnine (viz), numpyro
- Tooling: pytest (+pytest-order), coverage, ruff, basedpyright, mkdocs

## Project Conventions
- sklearn-style API (`.fit()` / `.predict()`), class-based discriminators
  subclassing `BaseDiscriminator` with `init_params()` / `apply(params, a, x, ax)`.
- NumPy-style docstrings, full type hints, double quotes, 100-char lines.
- Target ~100% test coverage; tests mirror source modules under `tests/`.
- Numerical stability: clip probabilities with `eps=1e-7` before weight division.

## Important Constraints
- Public API in `src/stochpw/__init__.py` must remain backwards compatible.
- Within-batch permutation semantics and the weight formula `w = η/(1-η)` are
  load-bearing for correctness and must not change.

## External Dependencies
- JAX device backend is environment-dependent (CPU in CI/dev; GPU/TPU possible).
