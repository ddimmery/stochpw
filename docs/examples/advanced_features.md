# Advanced Features

Demonstration of advanced features in stochpw.

This example shows how to use:

1. Alternative loss functions (exponential, Brier)
2. Weight-based regularization (entropy, L_p penalties)
3. Early stopping

## Code

```python
import jax.numpy as jnp
from stochpw import (
    MLPDiscriminator,
    PermutationWeighter,
    brier_loss,
    entropy_penalty,
    exponential_loss,
    lp_weight_penalty,
    standardized_mean_difference,
)

# Generate data with confounding
# X, A = generate_data(n=1000, seed=42)

# 1. Default Configuration (Logistic Loss)
weighter_default = PermutationWeighter(
    num_epochs=50,
    batch_size=128,
    random_state=42,
)
weighter_default.fit(X, A)
weights_default = weighter_default.predict(X, A)

# 2. Alternative Loss: Exponential Loss
weighter_exp = PermutationWeighter(
    loss_fn=exponential_loss,
    num_epochs=50,
    batch_size=128,
    random_state=42,
)
weighter_exp.fit(X, A)

# 3. Alternative Loss: Brier Score
weighter_brier = PermutationWeighter(
    loss_fn=brier_loss,
    num_epochs=50,
    batch_size=128,
    random_state=42,
)
weighter_brier.fit(X, A)

# 4. With Entropy Regularization
mlp = MLPDiscriminator(hidden_dims=[64, 32])
weighter_entropy = PermutationWeighter(
    discriminator=mlp,
    regularization_fn=entropy_penalty,
    regularization_strength=0.01,
    num_epochs=50,
    batch_size=128,
    random_state=42,
)
weighter_entropy.fit(X, A)

# 5. With L2 Weight Regularization
weighter_l2 = PermutationWeighter(
    regularization_fn=lambda w: lp_weight_penalty(w, p=2.0),
    regularization_strength=0.001,
    num_epochs=50,
    batch_size=128,
    random_state=42,
)
weighter_l2.fit(X, A)

# 6. With Early Stopping
weighter_early = PermutationWeighter(
    early_stopping=True,
    patience=10,
    min_delta=0.001,
    num_epochs=200,  # Set high, but will stop early
    batch_size=128,
    random_state=42,
)
weighter_early.fit(X, A)

# 7. All Features Combined
mlp_combined = MLPDiscriminator(hidden_dims=[128, 64, 32], activation="tanh")
weighter_combined = PermutationWeighter(
    discriminator=mlp_combined,
    loss_fn=brier_loss,
    regularization_fn=entropy_penalty,
    regularization_strength=0.01,
    early_stopping=True,
    patience=15,
    min_delta=0.001,
    num_epochs=200,
    batch_size=128,
    random_state=42,
)
weighter_combined.fit(X, A)
```

## Output

```
============================================================
Advanced Features Demo
============================================================

Generated data: X.shape=(1000, 5), A.shape=(1000, 1)
Initial SMD: 0.9872

============================================================
1. Default Configuration (Logistic Loss)
============================================================
Final SMD: 0.5034
Training epochs: 50
Final loss: 0.7007

============================================================
2. Alternative Loss: Exponential Loss
============================================================
Final SMD: 0.6182
Final loss: 1.1258

============================================================
3. Alternative Loss: Brier Score
============================================================
Final SMD: 0.4442
Final loss: 0.2504

============================================================
4. With Entropy Regularization
============================================================
Final SMD: 0.1521
Weight entropy (with regularization): 6.77
Weight entropy (without regularization): 6.75
Higher entropy = more uniform weights (better ESS)

============================================================
5. With Early Stopping
============================================================
Final SMD: 0.4851
Stopped at epoch: 77/200
Epochs saved: 123

============================================================
6. All Features Combined
============================================================
Final SMD: 0.2193
Stopped at epoch: 35/200
Final loss: 0.2107
Weight entropy: 6.80

============================================================
Summary: Balance Improvement
============================================================
Initial:                    0.9872
Default (logistic):         0.5034
Exponential loss:           0.6182
Brier loss:                 0.4442
With entropy regularization: 0.1521
With early stopping:        0.4851
All features combined:      0.2193
```

??? example "Full source code"

    ```python
    """
    Demonstration of advanced features in stochpw.
    
    This example shows how to use:
    
    1. Alternative loss functions (exponential, Brier)
    2. Weight-based regularization (entropy, L_p penalties)
    3. Early stopping
    """
    
    import jax.numpy as jnp
    from stochpw import (
        MLPDiscriminator,
        PermutationWeighter,
        brier_loss,
        entropy_penalty,
        exponential_loss,
        standardized_mean_difference,
    )
    
    key_code = """import jax.numpy as jnp
    from stochpw import (
        MLPDiscriminator,
        PermutationWeighter,
        brier_loss,
        entropy_penalty,
        exponential_loss,
        lp_weight_penalty,
        standardized_mean_difference,
    )
    
    # Generate data with confounding
    # X, A = generate_data(n=1000, seed=42)
    
    # 1. Default Configuration (Logistic Loss)
    weighter_default = PermutationWeighter(
        num_epochs=50,
        batch_size=128,
        random_state=42,
    )
    weighter_default.fit(X, A)
    weights_default = weighter_default.predict(X, A)
    
    # 2. Alternative Loss: Exponential Loss
    weighter_exp = PermutationWeighter(
        loss_fn=exponential_loss,
        num_epochs=50,
        batch_size=128,
        random_state=42,
    )
    weighter_exp.fit(X, A)
    
    # 3. Alternative Loss: Brier Score
    weighter_brier = PermutationWeighter(
        loss_fn=brier_loss,
        num_epochs=50,
        batch_size=128,
        random_state=42,
    )
    weighter_brier.fit(X, A)
    
    # 4. With Entropy Regularization
    mlp = MLPDiscriminator(hidden_dims=[64, 32])
    weighter_entropy = PermutationWeighter(
        discriminator=mlp,
        regularization_fn=entropy_penalty,
        regularization_strength=0.01,
        num_epochs=50,
        batch_size=128,
        random_state=42,
    )
    weighter_entropy.fit(X, A)
    
    # 5. With L2 Weight Regularization
    weighter_l2 = PermutationWeighter(
        regularization_fn=lambda w: lp_weight_penalty(w, p=2.0),
        regularization_strength=0.001,
        num_epochs=50,
        batch_size=128,
        random_state=42,
    )
    weighter_l2.fit(X, A)
    
    # 6. With Early Stopping
    weighter_early = PermutationWeighter(
        early_stopping=True,
        patience=10,
        min_delta=0.001,
        num_epochs=200,  # Set high, but will stop early
        batch_size=128,
        random_state=42,
    )
    weighter_early.fit(X, A)
    
    # 7. All Features Combined
    mlp_combined = MLPDiscriminator(hidden_dims=[128, 64, 32], activation="tanh")
    weighter_combined = PermutationWeighter(
        discriminator=mlp_combined,
        loss_fn=brier_loss,
        regularization_fn=entropy_penalty,
        regularization_strength=0.01,
        early_stopping=True,
        patience=15,
        min_delta=0.001,
        num_epochs=200,
        batch_size=128,
        random_state=42,
    )
    weighter_combined.fit(X, A)
    """
    
    
    def generate_data(n=1000, seed=42):
        """Generate synthetic data with treatment-covariate confounding."""
        import jax
    
        key = jax.random.PRNGKey(seed)
        key1, key2, key3 = jax.random.split(key, 3)
    
        # Generate covariates
        X = jax.random.normal(key1, (n, 5))
    
        # Treatment depends on covariates (confounding)
        propensity = jax.nn.sigmoid(X[:, 0] + 0.5 * X[:, 1])
        A = (jax.random.uniform(key2, (n,)) < propensity).astype(float)[:, None]
    
        return X, A
    
    
    def main():
        """Demonstrate advanced features."""
        print("=" * 60)
        print("Advanced Features Demo")
        print("=" * 60)
    
        # Generate data
        X, A = generate_data(n=1000, seed=42)
        print(f"\nGenerated data: X.shape={X.shape}, A.shape={A.shape}")
    
        # Compute initial imbalance
        initial_smd = standardized_mean_difference(X, A, jnp.ones(X.shape[0]))
        print(f"Initial SMD: {jnp.max(jnp.abs(initial_smd)):.4f}")
    
        print("\n" + "=" * 60)
        print("1. Default Configuration (Logistic Loss)")
        print("=" * 60)
    
        weighter_default = PermutationWeighter(
            num_epochs=50,
            batch_size=128,
            random_state=42,
        )
        weighter_default.fit(X, A)
        weights_default = weighter_default.predict(X, A)
        smd_default = standardized_mean_difference(X, A, weights_default)
    
        print(f"Final SMD: {jnp.max(jnp.abs(smd_default)):.4f}")
        assert weighter_default.history_ is not None
        print(f"Training epochs: {len(weighter_default.history_['loss'])}")
        print(f"Final loss: {weighter_default.history_['loss'][-1]:.4f}")
    
        print("\n" + "=" * 60)
        print("2. Alternative Loss: Exponential Loss")
        print("=" * 60)
    
        weighter_exp = PermutationWeighter(
            loss_fn=exponential_loss,
            num_epochs=50,
            batch_size=128,
            random_state=42,
        )
        weighter_exp.fit(X, A)
        weights_exp = weighter_exp.predict(X, A)
        smd_exp = standardized_mean_difference(X, A, weights_exp)
    
        print(f"Final SMD: {jnp.max(jnp.abs(smd_exp)):.4f}")
        assert weighter_exp.history_ is not None
        print(f"Final loss: {weighter_exp.history_['loss'][-1]:.4f}")
    
        print("\n" + "=" * 60)
        print("3. Alternative Loss: Brier Score")
        print("=" * 60)
    
        weighter_brier = PermutationWeighter(
            loss_fn=brier_loss,
            num_epochs=50,
            batch_size=128,
            random_state=42,
        )
        weighter_brier.fit(X, A)
        weights_brier = weighter_brier.predict(X, A)
        smd_brier = standardized_mean_difference(X, A, weights_brier)
    
        print(f"Final SMD: {jnp.max(jnp.abs(smd_brier)):.4f}")
        assert weighter_brier.history_ is not None
        print(f"Final loss: {weighter_brier.history_['loss'][-1]:.4f}")
    
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
        weighter_entropy_reg.fit(X, A)
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
        weighter_no_reg.fit(X, A)
        weights_no_reg = weighter_no_reg.predict(X, A)
    
        # Compute negative entropy (penalty) for comparison
        entropy_with_reg = -entropy_penalty(weights_entropy_reg)
        entropy_without_reg = -entropy_penalty(weights_no_reg)
    
        print(f"Final SMD: {jnp.max(jnp.abs(smd_entropy_reg)):.4f}")
        print(f"Weight entropy (with regularization): {entropy_with_reg:.2f}")
        print(f"Weight entropy (without regularization): {entropy_without_reg:.2f}")
        print("Higher entropy = more uniform weights (better ESS)")
    
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
        weighter_early.fit(X, A)
        weights_early = weighter_early.predict(X, A)
        smd_early = standardized_mean_difference(X, A, weights_early)
    
        print(f"Final SMD: {jnp.max(jnp.abs(smd_early)):.4f}")
        assert weighter_early.history_ is not None
        print(f"Stopped at epoch: {len(weighter_early.history_['loss'])}/200")
        print(f"Epochs saved: {200 - len(weighter_early.history_['loss'])}")
    
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
        weighter_combined.fit(X, A)
        weights_combined = weighter_combined.predict(X, A)
        smd_combined = standardized_mean_difference(X, A, weights_combined)
    
        print(f"Final SMD: {jnp.max(jnp.abs(smd_combined)):.4f}")
        assert weighter_combined.history_ is not None
        assert weighter_combined.params_ is not None
        print(f"Stopped at epoch: {len(weighter_combined.history_['loss'])}/200")
        print(f"Final loss: {weighter_combined.history_['loss'][-1]:.4f}")
        entropy_combined = -entropy_penalty(weights_combined)
        print(f"Weight entropy: {entropy_combined:.2f}")
    
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
    
    
    if __name__ == "__main__":
        main()
    
    ```

    [View on GitHub](https://github.com/ddimmery/stochpw/blob/main/examples/advanced_features.py){ .md-button }
