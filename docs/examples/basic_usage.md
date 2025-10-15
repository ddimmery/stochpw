---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Basic Usage Example for stochpw

This example demonstrates the basic workflow of permutation weighting:
1. Generate synthetic confounded data
2. Fit a permutation weighter
3. Extract importance weights
4. Assess balance improvement

```python
import time

import jax
import jax.numpy as jnp
import optax
from stochpw import PermutationWeighter, effective_sample_size, standardized_mean_difference
```

## Generate Synthetic Data with Confounding

```python
start_time = time.time()

# Generate synthetic observational data with confounding
key = jax.random.PRNGKey(420)
n = 250

# Generate confounders
X_key, A_key = jax.random.split(key)
X = jax.random.normal(X_key, (n, 5))  # 5 covariates

# Treatment depends on covariates (confounding)
propensity = jax.nn.sigmoid(0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2)
A = jax.random.bernoulli(A_key, propensity, (n,)).astype(jnp.float32).reshape(-1, 1)

print("=" * 60)
print("Permutation Weighting Example")
print("=" * 60)
print(f"\nData: {n} samples, {X.shape[1]} covariates")
print(f"Treatment distribution: {A.mean():.2%} treated")
```

## Fit Permutation Weighter

```python
print("\nFitting permutation weighter...")
opt = optax.rmsprop(learning_rate=0.1)
weighter = PermutationWeighter(num_epochs=20, batch_size=250 // 4, random_state=42, optimizer=opt)

weighter.fit(X, A)
weights = weighter.predict(X, A)
```

## Assess Results

```python
# Assess balance
print("\n" + "-" * 60)
print("Results:")
print("-" * 60)

# Weight statistics
print("\nWeight statistics:")
print(f"  Range: [{weights.min():.3f}, {weights.max():.3f}]")
print(f"  Mean: {weights.mean():.3f}")
print(f"  Std: {weights.std():.3f}")

# Effective sample size
ess = effective_sample_size(weights)
ess_ratio = ess / len(weights)
print("\nEffective sample size:")
print(f"  ESS: {ess:.1f} / {len(weights)} ({ess_ratio:.1%})")

# Balance assessment
smd_unweighted = standardized_mean_difference(X, A, jnp.ones_like(weights))
smd_weighted = standardized_mean_difference(X, A, weights)

print("\nStandardized Mean Difference (SMD):")
print(f"  Max |SMD| (unweighted): {jnp.abs(smd_unweighted).max():.3f}")
print(f"  Max |SMD| (weighted):   {jnp.abs(smd_weighted).max():.3f}")

# Calculate improvement
max_smd_unweighted = jnp.abs(smd_unweighted).max()
max_smd_weighted = jnp.abs(smd_weighted).max()
improvement = (1 - max_smd_weighted / max_smd_unweighted) * 100
print(f"  Improvement: {improvement:.1f}%")
```

## Training History

```python
# Training history
assert weighter.history_ is not None  # Guaranteed after fit()
loss_history = weighter.history_["loss"]
initial_loss = loss_history[0]
final_loss = loss_history[-1]
print("\nTraining:")
print(f"  Initial loss: {initial_loss:.4f}")
print(f"  Final loss: {final_loss:.4f}")
print(f"  Loss reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
print(f"  Epochs: {len(loss_history)}")

# Show first few and last few losses
print(f"  First 10 losses: {[f'{loss:.2f}' for loss in loss_history[:10]]}")
print(f"  Last 10 losses: {[f'{loss:.2f}' for loss in loss_history[-10:]]}")

print("\n" + "=" * 60)
print("✓ Example completed successfully!")
elapsed_time = time.time() - start_time
print(f"⏱  Total execution time: {elapsed_time:.2f} seconds")
print("=" * 60)
```

## Output

```
============================================================
Permutation Weighting Example
============================================================

Data: 250 samples, 5 covariates
Treatment distribution: 58.00% treated

Fitting permutation weighter...

------------------------------------------------------------
Results:
------------------------------------------------------------

Weight statistics:
  Range: [0.270, 2.573]
  Mean: 1.130
  Std: 0.425

Effective sample size:
  ESS: 219.0 / 250 (87.6%)

Standardized Mean Difference (SMD):
  Max |SMD| (unweighted): 0.517
  Max |SMD| (weighted):   0.079
  Improvement: 84.8%

Training:
  Initial loss: 0.7540
  Final loss: 0.6894
  Loss reduction: 8.6%
  Epochs: 20
  First 10 losses: ['0.75', '0.70', '0.68', '0.69', '0.69', '0.69', '0.69', '0.69', '0.69', '0.70']
  Last 10 losses: ['0.69', '0.69', '0.68', '0.70', '0.70', '0.70', '0.69', '0.69', '0.68', '0.69']

============================================================
✓ Example completed successfully!
⏱  Total execution time: 4.01 seconds
============================================================
```


---

[View source on GitHub](https://github.com/ddimmery/stochpw/blob/main/examples/basic_usage.py){ .md-button }
