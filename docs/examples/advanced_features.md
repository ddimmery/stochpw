# Advanced Features Demo

This example demonstrates advanced features in stochpw:

1. Alternative loss functions (exponential, Brier)
2. Weight-based regularization (entropy penalty)
3. Early stopping


```python
import time

import jax
import jax.numpy as jnp

from stochpw import (
    MLPDiscriminator,
    PermutationWeighter,
    brier_loss,
    entropy_penalty,
    exponential_loss,
    standardized_mean_difference,
)
```

## Generate Synthetic Data


```python
def generate_data(n: int = 1000, seed: int = 42):
    """Generate synthetic data with treatment-covariate confounding."""
    key = jax.random.PRNGKey(seed)
    key1, key2, _key3 = jax.random.split(key, 3)

    # Generate covariates
    X = jax.random.normal(key1, (n, 5))

    # Treatment depends on covariates (confounding)
    propensity = jax.nn.sigmoid(X[:, 0] + 0.5 * X[:, 1])
    A = (jax.random.uniform(key2, (n,)) < propensity).astype(float)[:, None]

    return X, A


start_time = time.time()

print("=" * 60)
print("Advanced Features Demo")
print("=" * 60)

# Generate data
X, A = generate_data(n=1000, seed=42)
print(f"\nGenerated data: X.shape={X.shape}, A.shape={A.shape}")

# Compute initial imbalance
initial_smd = standardized_mean_difference(X, A, jnp.ones(X.shape[0]))
print(f"Initial SMD: {jnp.max(jnp.abs(initial_smd)):.4f}")
```

    ============================================================
    Advanced Features Demo
    ============================================================


    
    Generated data: X.shape=(1000, 5), A.shape=(1000, 1)


    Initial SMD: 0.8575


## 1. Default Configuration (Logistic Loss)


```python
print("\n" + "=" * 60)
print("1. Default Configuration (Logistic Loss)")
print("=" * 60)

weighter_default = PermutationWeighter(
    num_epochs=50,
    batch_size=128,
    random_state=42,
)
_ = weighter_default.fit(X, A)
weights_default = weighter_default.predict(X, A)
smd_default = standardized_mean_difference(X, A, weights_default)

print(f"Final SMD: {jnp.max(jnp.abs(smd_default)):.4f}")
assert weighter_default.history_ is not None
print(f"Training epochs: {len(weighter_default.history_['loss'])}")
print(f"Final loss: {weighter_default.history_['loss'][-1]:.4f}")
```

    
    ============================================================
    1. Default Configuration (Logistic Loss)
    ============================================================


    Final SMD: 0.8955
    Training epochs: 50
    Final loss: 0.7840


## 2. Alternative Loss: Exponential Loss


```python
print("\n" + "=" * 60)
print("2. Alternative Loss: Exponential Loss")
print("=" * 60)

weighter_exp = PermutationWeighter(
    loss_fn=exponential_loss,
    num_epochs=50,
    batch_size=128,
    random_state=42,
)
_ = weighter_exp.fit(X, A)
weights_exp = weighter_exp.predict(X, A)
smd_exp = standardized_mean_difference(X, A, weights_exp)

print(f"Final SMD: {jnp.max(jnp.abs(smd_exp)):.4f}")
assert weighter_exp.history_ is not None
print(f"Final loss: {weighter_exp.history_['loss'][-1]:.4f}")
```

    
    ============================================================
    2. Alternative Loss: Exponential Loss
    ============================================================


    Final SMD: 0.9809
    Final loss: 1.7735


## 3. Alternative Loss: Brier Score


```python
print("\n" + "=" * 60)
print("3. Alternative Loss: Brier Score")
print("=" * 60)

weighter_brier = PermutationWeighter(
    loss_fn=brier_loss,
    num_epochs=50,
    batch_size=128,
    random_state=42,
)
_ = weighter_brier.fit(X, A)
weights_brier = weighter_brier.predict(X, A)
smd_brier = standardized_mean_difference(X, A, weights_brier)

print(f"Final SMD: {jnp.max(jnp.abs(smd_brier)):.4f}")
assert weighter_brier.history_ is not None
print(f"Final loss: {weighter_brier.history_['loss'][-1]:.4f}")
```

    
    ============================================================
    3. Alternative Loss: Brier Score
    ============================================================


    Final SMD: 0.8714
    Final loss: 0.2809


## 4. With Entropy Regularization


```python
print("\n" + "=" * 60)
print("4. With Entropy Regularization")
print("=" * 60)

mlp = MLPDiscriminator(hidden_dims=[64, 32])
weighter_entropy_reg = PermutationWeighter(
    discriminator=mlp,
    regularization_fn=entropy_penalty,
    regularization_strength=0.01,
    num_epochs=50,
    batch_size=128,
    random_state=42,
)
_ = weighter_entropy_reg.fit(X, A)
weights_entropy_reg = weighter_entropy_reg.predict(X, A)
smd_entropy_reg = standardized_mean_difference(X, A, weights_entropy_reg)

# Compare weight entropy with and without regularization
mlp_no_reg = MLPDiscriminator(hidden_dims=[64, 32])
weighter_no_reg = PermutationWeighter(
    discriminator=mlp_no_reg,
    num_epochs=50,
    batch_size=128,
    random_state=42,
)
_ = weighter_no_reg.fit(X, A)
weights_no_reg = weighter_no_reg.predict(X, A)

# Compute negative entropy (penalty) for comparison
entropy_with_reg = -entropy_penalty(weights_entropy_reg)
entropy_without_reg = -entropy_penalty(weights_no_reg)

print(f"Final SMD: {jnp.max(jnp.abs(smd_entropy_reg)):.4f}")
print(f"Weight entropy (with regularization): {entropy_with_reg:.2f}")
print(f"Weight entropy (without regularization): {entropy_without_reg:.2f}")
print("Higher entropy = more uniform weights (better ESS)")
```

    
    ============================================================
    4. With Entropy Regularization
    ============================================================


    Final SMD: 0.1016
    Weight entropy (with regularization): 6.78
    Weight entropy (without regularization): 6.77
    Higher entropy = more uniform weights (better ESS)


## 5. With Early Stopping


```python
print("\n" + "=" * 60)
print("5. With Early Stopping")
print("=" * 60)

weighter_early = PermutationWeighter(
    early_stopping=True,
    patience=10,
    min_delta=0.001,
    num_epochs=200,  # Set high, but will stop early
    batch_size=128,
    random_state=42,
)
_ = weighter_early.fit(X, A)
weights_early = weighter_early.predict(X, A)
smd_early = standardized_mean_difference(X, A, weights_early)

print(f"Final SMD: {jnp.max(jnp.abs(smd_early)):.4f}")
assert weighter_early.history_ is not None
print(f"Stopped at epoch: {len(weighter_early.history_['loss'])}/200")
print(f"Epochs saved: {200 - len(weighter_early.history_['loss'])}")
```

    
    ============================================================
    5. With Early Stopping
    ============================================================


    Final SMD: 0.6395
    Stopped at epoch: 106/200
    Epochs saved: 94


## 6. All Features Combined


```python
print("\n" + "=" * 60)
print("6. All Features Combined")
print("=" * 60)

mlp_combined = MLPDiscriminator(hidden_dims=[128, 64, 32], activation="tanh")
weighter_combined = PermutationWeighter(
    discriminator=mlp_combined,
    loss_fn=brier_loss,
    regularization_fn=entropy_penalty,
    regularization_strength=0.005,
    early_stopping=True,
    patience=15,
    min_delta=0.001,
    num_epochs=200,
    batch_size=128,
    random_state=42,
)
_ = weighter_combined.fit(X, A)
weights_combined = weighter_combined.predict(X, A)
smd_combined = standardized_mean_difference(X, A, weights_combined)

print(f"Final SMD: {jnp.max(jnp.abs(smd_combined)):.4f}")
assert weighter_combined.history_ is not None
assert weighter_combined.params_ is not None
print(f"Stopped at epoch: {len(weighter_combined.history_['loss'])}/200")
print(f"Final loss: {weighter_combined.history_['loss'][-1]:.4f}")
entropy_combined = -entropy_penalty(weights_combined)
print(f"Weight entropy: {entropy_combined:.2f}")
```

    
    ============================================================
    6. All Features Combined
    ============================================================


    Final SMD: 0.1120
    Stopped at epoch: 29/200
    Final loss: 0.2120
    Weight entropy: 6.79


## Summary


```python
print("\n" + "=" * 60)
print("Summary: Balance Improvement")
print("=" * 60)
print(f"Initial:                    {jnp.max(jnp.abs(initial_smd)):.4f}")
print(f"Default (logistic):         {jnp.max(jnp.abs(smd_default)):.4f}")
print(f"Exponential loss:           {jnp.max(jnp.abs(smd_exp)):.4f}")
print(f"Brier loss:                 {jnp.max(jnp.abs(smd_brier)):.4f}")
print(f"With entropy regularization: {jnp.max(jnp.abs(smd_entropy_reg)):.4f}")
print(f"With early stopping:        {jnp.max(jnp.abs(smd_early)):.4f}")
print(f"All features combined:      {jnp.max(jnp.abs(smd_combined)):.4f}")

print("\n" + "=" * 60)
print("✓ Example completed successfully!")
elapsed_time = time.time() - start_time
print(f"⏱  Total execution time: {elapsed_time:.2f} seconds")
print("=" * 60)
```

    
    ============================================================
    Summary: Balance Improvement
    ============================================================
    Initial:                    0.8575
    Default (logistic):         0.8955
    Exponential loss:           0.9809
    Brier loss:                 0.8714
    With entropy regularization: 0.1016
    With early stopping:        0.6395
    All features combined:      0.1120
    
    ============================================================
    ✓ Example completed successfully!
    ⏱  Total execution time: 30.65 seconds
    ============================================================


---

[View source on GitHub](https://github.com/ddimmery/stochpw/blob/main/examples/advanced_features.py){ .md-button }
