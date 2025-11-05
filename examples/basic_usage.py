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
# # Basic Usage Example for stochpw
#
# This example demonstrates the basic workflow of permutation weighting:
# 1. Generate synthetic confounded data
# 2. Fit a permutation weighter
# 3. Extract importance weights
# 4. Assess balance improvement

# %%
import time

import jax
import jax.numpy as jnp
import optax

from stochpw import PermutationWeighter, effective_sample_size, standardized_mean_difference

start_time = time.time()

# %% [markdown]
# ## Generate Synthetic Data with Confounding

# %%
# Generate synthetic observational data with confounding
key = jax.random.PRNGKey(420)
n = 250

# Generate confounders
X_key, A_key = jax.random.split(key)
X = jax.random.normal(X_key, (n, 5))  # 5 covariates

# Treatment depends on covariates (confounding)
propensity = jax.nn.sigmoid(0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.2)
A = jax.random.bernoulli(A_key, propensity, (n,)).astype(jnp.float32).reshape(-1, 1)

print(f"Generated data: {n} samples, {X.shape[1]} covariates")
print(f"Treatment distribution: {A.mean():.2%} treated")

# %% [markdown]
# ## Fit Permutation Weighter

# %%
opt = optax.rmsprop(learning_rate=0.1)
weighter = PermutationWeighter(num_epochs=20, batch_size=250 // 4, random_state=42, optimizer=opt)

weighter.fit(X, A)
weights = weighter.predict(X, A)

print("Fitting complete!")

# %% [markdown]
# ## Weight Statistics

# %%
print("Weight statistics:")
print(f"  Range: [{weights.min():.3f}, {weights.max():.3f}]")
print(f"  Mean: {weights.mean():.3f}")
print(f"  Std: {weights.std():.3f}")

# %% [markdown]
# ## Effective Sample Size

# %%
ess = effective_sample_size(weights)
ess_ratio = ess / len(weights)
print("Effective sample size:")
print(f"  ESS: {ess:.1f} / {len(weights)} ({ess_ratio:.1%})")

# %% [markdown]
# ## Balance Assessment

# %%
smd_unweighted = standardized_mean_difference(X, A, jnp.ones_like(weights))
smd_weighted = standardized_mean_difference(X, A, weights)

print("Standardized Mean Difference (SMD):")
print(f"  Max |SMD| (unweighted): {jnp.abs(smd_unweighted).max():.3f}")
print(f"  Max |SMD| (weighted):   {jnp.abs(smd_weighted).max():.3f}")

max_smd_unweighted = jnp.abs(smd_unweighted).max()
max_smd_weighted = jnp.abs(smd_weighted).max()
improvement = (1 - max_smd_weighted / max_smd_unweighted) * 100
print(f"  Improvement: {improvement:.1f}%")

# %% [markdown]
# ## Training History

# %%
assert weighter.history_ is not None  # Guaranteed after fit()
loss_history = weighter.history_["loss"]
initial_loss = loss_history[0]
final_loss = loss_history[-1]

print("Training history:")
print(f"  Initial loss: {initial_loss:.4f}")
print(f"  Final loss: {final_loss:.4f}")
print(f"  Loss reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
print(f"  Epochs: {len(loss_history)}")
print(f"\n  First 10 losses: {[f'{loss:.2f}' for loss in loss_history[:10]]}")
print(f"  Last 10 losses: {[f'{loss:.2f}' for loss in loss_history[-10:]]}")

# %% [markdown]
# ## Summary

# %%
elapsed_time = time.time() - start_time
print("✓ Example completed successfully!")
print(f"⏱  Total execution time: {elapsed_time:.2f} seconds")
