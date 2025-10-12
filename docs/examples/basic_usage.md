# Basic Usage

Basic usage example for stochpw.

## Code

```python
import jax
import jax.numpy as jnp
from stochpw import PermutationWeighter, effective_sample_size, standardized_mean_difference

# Generate synthetic observational data with confounding
key = jax.random.PRNGKey(420)
n = 250

X_key, A_key = jax.random.split(key)
X = jax.random.normal(X_key, (n, 5))  # 5 covariates

# Treatment depends on covariates (confounding)
propensity = jax.nn.sigmoid(0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2)
A = jax.random.bernoulli(A_key, propensity, (n,)).astype(jnp.float32).reshape(-1, 1)

# Fit permutation weighter
weighter = PermutationWeighter(num_epochs=1000, batch_size=250, random_state=42)
weighter.fit(X, A)
weights = weighter.predict(X, A)

# Assess balance
ess = effective_sample_size(weights)
smd_unweighted = standardized_mean_difference(X, A, jnp.ones_like(weights))
smd_weighted = standardized_mean_difference(X, A, weights)

# Training history
loss_history = weighter.history_["loss"]
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
  Range: [0.468, 2.135]
  Mean: 1.000
  Std: 0.269

Effective sample size:
  ESS: 233.1 / 250 (93.2%)

Standardized Mean Difference (SMD):
  Max |SMD| (unweighted): 0.517
  Max |SMD| (weighted):   0.131
  Improvement: 74.8%

Training:
  Initial loss: 0.8058
  Final loss: 0.6849
  Loss reduction: 15.0%
  Epochs: 1000
  First 10 losses: ['0.81', '0.83', '0.80', '0.83', '0.82', '0.82', '0.82', '0.81', '0.79', '0.79']
  Last 10 losses: ['0.69', '0.69', '0.68', '0.68', '0.69', '0.68', '0.68', '0.69', '0.69', '0.68']

============================================================
✓ Example completed successfully!
============================================================
```

??? example "Full source code"

    ```python
    """Basic usage example for stochpw."""
    
    import jax
    import jax.numpy as jnp
    from stochpw import PermutationWeighter, effective_sample_size, standardized_mean_difference
    
    
    key_code = """import jax
    import jax.numpy as jnp
    from stochpw import PermutationWeighter, effective_sample_size, standardized_mean_difference
    
    # Generate synthetic observational data with confounding
    key = jax.random.PRNGKey(420)
    n = 250
    
    X_key, A_key = jax.random.split(key)
    X = jax.random.normal(X_key, (n, 5))  # 5 covariates
    
    # Treatment depends on covariates (confounding)
    propensity = jax.nn.sigmoid(0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2)
    A = jax.random.bernoulli(A_key, propensity, (n,)).astype(jnp.float32).reshape(-1, 1)
    
    # Fit permutation weighter
    weighter = PermutationWeighter(num_epochs=1000, batch_size=250, random_state=42)
    weighter.fit(X, A)
    weights = weighter.predict(X, A)
    
    # Assess balance
    ess = effective_sample_size(weights)
    smd_unweighted = standardized_mean_difference(X, A, jnp.ones_like(weights))
    smd_weighted = standardized_mean_difference(X, A, weights)
    
    # Training history
    loss_history = weighter.history_["loss"]
    """
    
    
    def main():
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
    
        # Fit permutation weighter
        print("\nFitting permutation weighter...")
        weighter = PermutationWeighter(num_epochs=1000, batch_size=250, random_state=42)
    
        weighter.fit(X, A)
        weights = weighter.predict(X, A)
    
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
        print("=" * 60)
    
    
    if __name__ == "__main__":
        main()
    
    ```

    [View on GitHub](https://github.com/ddimmery/stochpw/blob/main/examples/basic_usage.py){ .md-button }
