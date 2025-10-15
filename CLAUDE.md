# CLAUDE.md - AI Assistant Guide for stochpw

This document helps AI assistants (like Claude) understand the stochpw project structure, conventions, and best practices for development.

## Project Overview

**stochpw** is a JAX-based Python package implementing permutation weighting for causal inference. It learns importance weights by training a discriminator to distinguish between observed treatment-covariate pairs (X, A) and artificially permuted pairs (X, A').

**Key characteristics:**
- JAX-first implementation (functional programming, JIT compilation, automatic differentiation)
- Sklearn-style API (`.fit()` and `.predict()` methods)
- Composable design - all components exposed for integration into larger causal inference models
- Type-annotated, well-tested, production-ready code

## Key Technical Concepts

### 1. Permutation Weighting Algorithm

The core algorithm trains a discriminator η(a,x) to predict whether a pair (a,x) is:
- **Observed** (label C=0): From the original data distribution P(A,X)
- **Permuted** (label C=1): From the product distribution P(A)P(X)

**Weight formula:** `w(a,x) = η(a,x) / (1 - η(a,x))`

These weights reweight the observed distribution to match the balanced permuted distribution.

### 2. Critical Design Decisions

**Interactions are Essential:**
- Linear discriminators MUST include A⊗X interaction terms
- Within-batch permutation ensures P(A) and P(X) are identical in both distributions
- Only the joint P(A,X) differs → need interactions to capture this
- Discriminator input: `logit = w_a^T A + w_x^T X + w_ax^T (A⊗X) + b`

**Class-Based Discriminator Architecture:**
- All discriminators inherit from `BaseDiscriminator` abstract class
- Two methods required: `init_params()` and `apply()`
- Discriminators accept 4 inputs: `(params, A, X, AX)` where AX = A⊗X interactions
- Users can create custom discriminators by subclassing `BaseDiscriminator`

**Composability:**
- All internal components are publicly exposed in `__init__.py`
- Functions like `create_training_batch()`, `logistic_loss()`, `extract_weights()` can be used in larger models
- Dataclasses (`TrainingBatch`, etc.) make data flow explicit and type-safe

## Development Guidelines

### Code Style

**Tools:**
- Linting: `ruff` (configured in pyproject.toml)
- Type checking: `pyright`
- Formatting: `ruff format` (double quotes, 100 char line length)
- Testing: `pytest` with `pytest-order` for test ordering

**Run quality checks:**
```bash
make lint        # Run ruff linting
make typecheck   # Run pyright
make test        # Run pytest
make coverage    # Generate coverage report
```

### Testing Philosophy

**Target: 100% code coverage** (currently 99.34%, only abstract class `pass` statements uncovered)

**Test structure:**
- One test file per source module
- Test edge cases: scalar A, extreme values, NaN/Inf, empty inputs
- Test numerical stability: η near 0 or 1, division by zero protection
- Test JAX compatibility: JIT compilation, gradient flow, reproducibility with seeds
- Test API usability: sklearn-style interface, method chaining

See existing test files in `tests/` for patterns and examples.

### Documentation Standards

**Docstring style: NumPy format** with type hints for all parameters and return values.

See existing code for examples. All public functions must have:
- Type hints
- NumPy-style docstrings
- Examples in docstring for complex functions
- Links to relevant papers/equations where applicable

### Adding New Features

**When adding a new discriminator:**
1. Create class in `src/stochpw/models/` inheriting from `BaseDiscriminator`
2. Implement `init_params()` and `apply()` methods
3. Accept signature: `apply(params, a, x, ax)` where ax = A⊗X interactions
4. Add to `models/__init__.py` exports
5. Add to main `__init__.py` exports
6. Create comprehensive tests in `tests/test_models.py`
7. Add example to `examples/` directory
8. Update documentation

**When adding a new loss function:**
1. Add to `src/stochpw/training/losses.py`
2. Decorate with `@jax.jit` for performance
3. Signature: `loss_fn(logits: Array, labels: Array) -> Array`
4. Add tests in `tests/test_training.py`
5. Export from `training/__init__.py` and main `__init__.py`
6. Add example usage to `examples/advanced_features.py`

**When adding a new diagnostic:**
1. Add to appropriate file in `src/stochpw/diagnostics/`
2. Return JAX arrays or standard Python types (avoid custom objects)
3. Add comprehensive tests including edge cases
4. Export from `diagnostics/__init__.py` and main `__init__.py`
5. Add visualization to `plotting.py` if applicable
6. Update `examples/diagnostics_demo.py`

### Example Files

**Format: Jupytext Percent Format**

All examples use jupytext percent format (`.py` files that convert to markdown):

```python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Example Title
# Description of the example

# %%
import time
# ... imports ...

# %% [markdown]
# ## Section Title

# %%
# Code for this section
```

**Requirements:**
- Jupytext header with percent format specification
- Markdown cells (`# %% [markdown]`) for documentation
- Code cells (`# %%`) for executable code
- Time execution with `time.time()` at start and end
- Print clear section headers with separators

**Generating documentation:**
```bash
poetry run python docs/gen_examples.py
```

This:
1. Converts `.py` → `.md` using jupytext
2. Runs each example to capture output
3. Appends output and plots to markdown
4. Saves to `docs/examples/`

See [basic_usage.py](examples/basic_usage.py) for a complete example in jupytext format.

## Common Tasks

### Running tests
```bash
poetry run pytest                          # Run all tests
poetry run pytest tests/test_core.py       # Run specific test file
poetry run pytest -k "test_balance"        # Run tests matching pattern
poetry run pytest --cov=src/stochpw        # Run with coverage
```

### Building documentation
```bash
poetry run mkdocs serve                    # Local preview at http://127.0.0.1:8000
poetry run mkdocs build                    # Build static site to site/
```

### Package release
```bash
make clean                      # Clean build artifacts
make build                      # Build package
poetry publish                  # Publish to PyPI (requires auth)
```

### Running examples
```bash
poetry run python examples/basic_usage.py           # Run basic example
poetry run python examples/mlp_discriminator.py     # Run MLP example
poetry run python examples/advanced_features.py     # Run advanced features example
poetry run python examples/diagnostics_demo.py      # Run diagnostics demo
poetry run python examples/lalonde_experiment.py    # Run Lalonde experiment
```

## Important Implementation Details

### Weight Formula Direction
**Correct formula:** `w = η / (1 - η)` where η = p(C=1|a,x)

This up-weights units that look **PERMUTED** (high η) to reweight the observed distribution toward the balanced permuted distribution. This is often confused - verify with balance tests!

### Batch Creation Strategy
Permutation happens **within batch**, not across entire dataset. This creates local product distribution P(A)P(X) within each batch. See `src/stochpw/training/batch.py` for implementation.

### Numerical Stability
Always clip probabilities before computing weights to avoid division by zero. Use `eps = 1e-7` as the clipping threshold.

### JAX Best Practices
- Use `@jax.jit` for performance-critical functions
- Manage PRNG keys explicitly (split before each random operation)
- Prefer functional style (pure functions without side effects)
- Use `jax.numpy` not `numpy` in core code (but accept both in validation)

## Git Workflow

**Branch naming:**
- `main` - stable releases
- `feature/description` - new features
- `fix/description` - bug fixes
- `docs/description` - documentation updates

**Commit messages:**
- Use conventional commits: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`
- Example: `feat(models): add attention-based discriminator`

**Before committing:**
```bash
make lint                       # Fix linting issues
make typecheck                  # Fix type errors
make test                       # Ensure tests pass
make coverage                   # Check coverage hasn't decreased
```

## Dependencies

**Core:** JAX, Optax, NumPy, plotnine
**Dev:** pytest, coverage, ruff, pyright, mkdocs

See `pyproject.toml` for complete list and version constraints.

## Reference Documentation

**Essential reading:**
- `.claude/DESIGN.md` - Architecture and design decisions
- `.claude/IMPLEMENTATION.md` - Implementation roadmap
- `docs/contributing.md` - Contribution guidelines
- Original paper: Arbour, Dimmery, & Sondhi (2021) - Permutation Weighting (ICML)

## Current Status (v0.2.0)

**Production ready:** Core permutation weighting with linear/MLP discriminators, sklearn-style API, alternative losses, regularization, early stopping, comprehensive diagnostics, visualization, 99%+ test coverage.

**Future work:** See `.claude/IMPLEMENTATION.md` for roadmap and planned features.

## Tips for AI Assistants

**When asked to add a feature:**
1. Check IMPLEMENTATION.md to see if it's planned and what phase it's in
2. Follow the existing patterns in similar modules
3. Create comprehensive tests (aim for 100% coverage)
4. Update relevant examples
5. Add to exports in `__init__.py` files
6. Update documentation

**When debugging:**
1. Check test files - they often contain examples of correct usage
2. Look at examples/ directory for working code patterns
3. Review DESIGN.md for understanding of core concepts
4. Pay attention to JAX-specific issues (random keys, JIT compilation, etc.)

**When refactoring:**
1. Run tests frequently to catch regressions early
2. Maintain backwards compatibility in public API
3. Update docstrings to reflect changes
4. Check that all examples still work

**When writing tests:**
1. Test both happy path and edge cases
2. Use descriptive test names: `test_<what>_<condition>_<expected>`
3. Include a docstring explaining what's being tested
4. Test numerical stability explicitly (extreme values, NaN/Inf)
5. Test JAX compatibility (JIT, grad, vmap if applicable)

## Quick Reference

- Design & architecture → `.claude/DESIGN.md`
- Implementation roadmap → `.claude/IMPLEMENTATION.md`
- Contribution guidelines → `docs/contributing.md`
- Usage examples → `examples/` directory
- API docs → https://github.com/ddimmery/stochpw
