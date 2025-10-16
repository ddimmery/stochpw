# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: stochpw-py3.12
#     language: python
#     name: stochpw
# ---

# %% [markdown]
# # MLP Discriminator Example
#
# This example demonstrates using MLP (multilayer perceptron) discriminators
# for handling complex nonlinear confounding patterns.
#
# We compare:
# - Linear discriminator (default)
# - MLP with different architectures
# - MLP with different activation functions

# %%
import time

import jax
import jax.numpy as jnp
from stochpw import MLPDiscriminator, PermutationWeighter, standardized_mean_difference

# %% [markdown]
# ## Generate Data with Complex Nonlinear Confounding

# %%
start_time = time.time()

# Generate synthetic data with complex confounding
key = jax.random.PRNGKey(123)
n = 500

# Generate confounders with nonlinear relationships
X_key, A_key = jax.random.split(key)
X = jax.random.normal(X_key, (n, 5))

# Complex nonlinear propensity function
propensity = jax.nn.sigmoid(
    0.5 * X[:, 0] ** 2  # Nonlinear effect
    - 0.3 * X[:, 1] * X[:, 2]  # Interaction
    + jnp.sin(X[:, 3])  # Nonlinearity
    + 0.1
)
A = jax.random.bernoulli(A_key, propensity, (n,)).astype(jnp.float32).reshape(-1, 1)

print("=" * 70)
print("MLP Discriminator Example")
print("=" * 70)
print(f"\nData: {n} samples, {X.shape[1]} covariates")
print(f"Treatment: {A.mean():.1%} treated (nonlinear confounding)")

# %% [markdown]
# ## Compare Linear vs MLP Discriminators

# %%
print("\n" + "=" * 70)
print("Comparing Linear vs MLP Discriminators")
print("=" * 70)

configs = [
    ("Linear", None),
    ("MLP (default)", MLPDiscriminator()),
    ("MLP (small)", MLPDiscriminator(hidden_dims=[32])),
    ("MLP (large)", MLPDiscriminator(hidden_dims=[128, 64, 32])),
    ("MLP (tanh)", MLPDiscriminator(activation="tanh")),
]

results = []

for name, discriminator in configs:
    print(f"\n{name}:")
    print("-" * 40)

    weighter = PermutationWeighter(
        discriminator=discriminator,
        num_epochs=500,
        batch_size=500,
        random_state=42,
    )

    weighter.fit(X, A)
    weights = weighter.predict(X, A)

    # Calculate balance improvement
    smd_unweighted = standardized_mean_difference(X, A, jnp.ones_like(weights))
    smd_weighted = standardized_mean_difference(X, A, weights)

    max_smd_unw = jnp.abs(smd_unweighted).max()
    max_smd_w = jnp.abs(smd_weighted).max()
    improvement = (1 - max_smd_w / max_smd_unw) * 100

    print(f"  Max |SMD| (unweighted): {max_smd_unw:.3f}")
    print(f"  Max |SMD| (weighted):   {max_smd_w:.3f}")
    print(f"  Balance improvement:    {improvement:.1f}%")

    results.append((name, max_smd_unw, max_smd_w, improvement))

# %% [markdown]
# ## Summary Comparison

# %%
print("\n" + "=" * 70)
print("Summary Comparison")
print("=" * 70)
print(f"{'Model':<20} {'Unweighted SMD':<18} {'Weighted SMD':<15} {'Improvement':<12}")
print("-" * 70)
for name, unw, w, imp in results:
    print(f"{name:<20} {unw:>15.3f}   {w:>13.3f}   {imp:>10.1f}%")

print("\n" + "=" * 70)
print("✓ Example completed successfully!")
elapsed_time = time.time() - start_time
print(f"⏱  Total execution time: {elapsed_time:.2f} seconds")
print("=" * 70)
