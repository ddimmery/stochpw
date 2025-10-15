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

# Lalonde Experiment: Permutation Weighting for ATE Estimation

This example demonstrates using permutation weighting on the classic Lalonde (1986)
observational dataset to estimate the average treatment effect (ATE) of a job
training program on earnings.

The dataset combines experimental treatment units from the NSW program with
non-experimental control units, creating confounding and selection bias that
must be addressed to recover the experimental benchmark ATE of $1,794.

**Reference:**
LaLonde, R. J. (1986). "Evaluating the Econometric Evaluations of Training
Programs with Experimental Data". The American Economic Review, 76(4), 604-620.

```python
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from stochpw import (
    MLPDiscriminator,
    PermutationWeighter,
    effective_sample_size,
    standardized_mean_difference,
)
```

## Helper Functions

```python
def load_lalonde_nsw():
    """
    Load the Lalonde NSW (National Supported Work) observational dataset.

    This dataset contains observational (non-experimental) data from the LaLonde study,
    where NSW experimental treatment units are combined with non-experimental control
    units, creating confounding and selection bias.

    The dataset contains:
    - Treatment: participation in job training program (1=treated, 0=control)
    - Outcome: real earnings in 1978 (RE78)
    - Covariates: age, education, race, marital status, pre-treatment earnings

    Returns:
        dict: Dictionary with keys:
            - X: Covariates array, shape (n, d_x)
            - A: Treatment array, shape (n, 1)
            - Y: Outcome (earnings) array, shape (n, 1)
            - feature_names: List of covariate names
            - ate_benchmark: Experimental ATE estimate from RCT ($1,794)
    """
    # Find the data file - try multiple locations to handle different execution contexts
    current_dir = Path(__file__).parent

    # Try several possible locations
    possible_paths = [
        current_dir / "nsw_data.csv",  # Same directory as script
        current_dir.parent / "examples" / "nsw_data.csv",  # When run from project root
    ]

    data_file = None
    for path in possible_paths:
        if path.exists():
            data_file = path
            break

    if data_file is None:
        raise FileNotFoundError(
            "Data file not found. Tried:\n" +
            "\n".join(f"  - {p}" for p in possible_paths) +
            "\n\nPlease ensure nsw_data.csv is in the examples/ directory."
        )

    # Load data
    data = np.genfromtxt(data_file, delimiter=",", skip_header=1)

    # Extract treatment, outcome, and covariates
    A = data[:, 0]  # Treatment indicator (first column)
    Y = data[:, -1]  # RE78 earnings (last column)
    X = data[:, 1:9]  # All other columns are covariates

    # Feature names (from dataset documentation)
    feature_names = [
        "age",
        "education",
        "black",
        "hispanic",
        "married",
        "nodegree",
        "RE74",  # earnings in 1974
        "RE75",  # earnings in 1975
    ]

    # Experimental ATE benchmark (from LaLonde 1986 paper)
    # This is the "true" ATE estimated from the randomized experiment
    ate_benchmark = 1794.0

    return {
        "X": jnp.array(X),
        "A": jnp.array(A),
        "Y": jnp.array(Y),
        "feature_names": feature_names,
        "ate_benchmark": ate_benchmark,
    }


def estimate_ate(Y, A, weights):
    """
    Estimate the average treatment effect (ATE) using weighted means.

    ATE = E[Y(1) - Y(0)] = E[Y|A=1] - E[Y|A=0]

    Args:
        Y: Outcome array, shape (n, 1) or (n,)
        A: Treatment array, shape (n, 1) or (n,)
        weights: Importance weights, shape (n,)

    Returns:
        float: Estimated ATE
    """
    Y = Y.flatten()
    A = A.flatten()

    treated_mask = A == 1
    control_mask = A == 0

    # Weighted mean for treated
    weighted_y1 = jnp.sum(Y[treated_mask] * weights[treated_mask])
    weighted_n1 = jnp.sum(weights[treated_mask])
    mean_y1 = weighted_y1 / weighted_n1

    # Weighted mean for control
    weighted_y0 = jnp.sum(Y[control_mask] * weights[control_mask])
    weighted_n0 = jnp.sum(weights[control_mask])
    mean_y0 = weighted_y0 / weighted_n0

    ate = mean_y1 - mean_y0
    return float(ate)
```

## Load Dataset

```python
start_time = time.time()

print("=" * 70)
print("Lalonde Experiment: Permutation Weighting for ATE Estimation")
print("=" * 70)

# Load observational data
print("\nLoading Lalonde NSW observational dataset...")
data = load_lalonde_nsw()
X, A, Y = data["X"], data["A"], data["Y"]
feature_names = data["feature_names"]
ate_benchmark = data["ate_benchmark"]

n_treated = int(A.sum())
n_control = len(A) - n_treated

print("\nDataset statistics:")
print(f"  Total samples: {len(X)}")
print(f"  Treated: {n_treated}")
print(f"  Control: {n_control}")
print(f"  Covariates: {X.shape[1]}")
print(f"  Covariate names: {', '.join(feature_names)}")
```

## Experimental Benchmark (Ground Truth)

```python
print(f"\n{'='*70}")
print("Experimental Benchmark (Ground Truth)")
print(f"{'='*70}")
print(f"  Experimental ATE: ${ate_benchmark:.2f}")
print("  (From the original randomized controlled trial)")
```

## Naive Estimate (No Adjustment)

```python
print(f"\n{'='*70}")
print("Naive Estimate (No Adjustment)")
print(f"{'='*70}")

weights_naive = jnp.ones(len(X))
ate_naive = estimate_ate(Y, A, weights_naive)
naive_error = ate_naive - ate_benchmark
naive_pct_error = (naive_error / ate_benchmark) * 100

print(f"  Naive ATE: ${ate_naive:.2f}")
print(f"  Error: ${naive_error:.2f} ({naive_pct_error:+.1f}%)")

# Check initial balance
smd_naive = standardized_mean_difference(X, A, weights_naive)
print("\n  Covariate balance:")
print(f"    Max |SMD|: {jnp.abs(smd_naive).max():.3f}")
print("    (Values > 0.1 indicate imbalance)")

print("\n  Per-covariate imbalance:")
for i, (name, smd_val) in enumerate(zip(feature_names, smd_naive)):
    print(f"    {name:12s}: {smd_val:+.3f}")
```

## Permutation Weighting with Simple MLP

```python
print(f"\n{'='*70}")
print("Permutation Weighting (Simple MLP)")
print(f"{'='*70}")

# Fit with a simple MLP architecture
mlp_simple = MLPDiscriminator(hidden_dims=[3])
weighter_simple = PermutationWeighter(
    discriminator=mlp_simple,
    num_epochs=500,
    batch_size=len(X),  # Full batch
    random_state=42,
)

print("\nFitting weighter...")
weighter_simple.fit(X, A)
weights_simple = weighter_simple.predict(X, A)

# Estimate ATE
ate_pw_simple = estimate_ate(Y, A, weights_simple)
pw_error_simple = ate_pw_simple - ate_benchmark
pw_pct_error_simple = (pw_error_simple / ate_benchmark) * 100

print(f"\n  Permutation-weighted ATE: ${ate_pw_simple:.2f}")
print(f"  Error: ${pw_error_simple:.2f} ({pw_pct_error_simple:+.1f}%)")

# Check balance improvement
smd_pw_simple = standardized_mean_difference(X, A, weights_simple)
print("\n  Covariate balance after weighting:")
print(f"    Max |SMD|: {jnp.abs(smd_pw_simple).max():.3f}")
balance_improvement = (
    1 - jnp.abs(smd_pw_simple).max() / jnp.abs(smd_naive).max()
) * 100
print(f"    Balance improvement: {balance_improvement:.1f}%")

# ESS
ess_simple = effective_sample_size(weights_simple)
ess_ratio_simple = ess_simple / len(weights_simple)
print("\n  Effective sample size:")
print(f"    ESS: {ess_simple:.0f} / {len(weights_simple)} ({ess_ratio_simple:.1%})")
```

## Permutation Weighting with Larger MLP

```python
print(f"\n{'='*70}")
print("Permutation Weighting (Larger MLP)")
print(f"{'='*70}")

# Try a larger architecture
mlp_large = MLPDiscriminator(hidden_dims=[32, 16])
weighter_large = PermutationWeighter(
    discriminator=mlp_large,
    num_epochs=500,
    batch_size=len(X),
    random_state=42,
)

print("\nFitting weighter...")
weighter_large.fit(X, A)
weights_large = weighter_large.predict(X, A)

# Estimate ATE
ate_pw_large = estimate_ate(Y, A, weights_large)
pw_error_large = ate_pw_large - ate_benchmark
pw_pct_error_large = (pw_error_large / ate_benchmark) * 100

print(f"\n  Permutation-weighted ATE: ${ate_pw_large:.2f}")
print(f"  Error: ${pw_error_large:.2f} ({pw_pct_error_large:+.1f}%)")

# Check balance
smd_pw_large = standardized_mean_difference(X, A, weights_large)
print("\n  Covariate balance after weighting:")
print(f"    Max |SMD|: {jnp.abs(smd_pw_large).max():.3f}")
balance_improvement_large = (
    1 - jnp.abs(smd_pw_large).max() / jnp.abs(smd_naive).max()
) * 100
print(f"    Balance improvement: {balance_improvement_large:.1f}%")

# ESS
ess_large = effective_sample_size(weights_large)
ess_ratio_large = ess_large / len(weights_large)
print("\n  Effective sample size:")
print(f"    ESS: {ess_large:.0f} / {len(weights_large)} ({ess_ratio_large:.1%})")
```

## Summary Comparison

```python
print(f"\n{'='*70}")
print("Summary Comparison")
print(f"{'='*70}")

print(f"\n{'Method':<30} {'ATE Estimate':<15} {'Error':<15} {'% Error':<12}")
print("-" * 72)
print(f"{'Experimental (Benchmark)':<30} ${ate_benchmark:>12.2f}   {'---':>12}   {'---':>12}")
print(
    f"{'Naive (Unadjusted)':<30} ${ate_naive:>12.2f}  "
    f"${naive_error:>12.2f}  {naive_pct_error:>10.1f}%"
)
print(
    f"{'PW (Simple MLP)':<30} ${ate_pw_simple:>12.2f}  "
    f"${pw_error_simple:>12.2f}  {pw_pct_error_simple:>10.1f}%"
)
print(
    f"{'PW (Larger MLP)':<30} ${ate_pw_large:>12.2f}  "
    f"${pw_error_large:>12.2f}  {pw_pct_error_large:>10.1f}%"
)

print("\n  Improvement over naive:")
improvement_over_naive = abs(naive_error) - abs(pw_error_simple)
print(f"\n  Improvement over naive: ${improvement_over_naive:.2f}")

print(f"\n{'='*70}")
print("✓ Lalonde experiment completed successfully!")
elapsed_time = time.time() - start_time
print(f"⏱  Total execution time: {elapsed_time:.2f} seconds")
print(f"{'='*70}")
```

## Output

```
======================================================================
Lalonde Experiment: Permutation Weighting for ATE Estimation
======================================================================

Loading Lalonde NSW observational dataset...

Dataset statistics:
  Total samples: 458
  Treated: 177
  Control: 281
  Covariates: 8
  Covariate names: age, education, black, hispanic, married, nodegree, RE74, RE75

======================================================================
Experimental Benchmark (Ground Truth)
======================================================================
  Experimental ATE: $1794.00
  (From the original randomized controlled trial)

======================================================================
Naive Estimate (No Adjustment)
======================================================================
  Naive ATE: $4224.48
  Error: $2430.48 (+135.5%)

  Covariate balance:
    Max |SMD|: 0.899
    (Values > 0.1 indicate imbalance)

  Per-covariate imbalance:
    age         : +0.175
    education   : +0.323
    black       : +0.279
    hispanic    : -0.087
    married     : -0.220
    nodegree    : -0.086
    RE74        : -0.884
    RE75        : -0.899

======================================================================
Permutation Weighting (Simple MLP)
======================================================================

Fitting weighter...

  Permutation-weighted ATE: $5232.60
  Error: $3438.60 (+191.7%)

  Covariate balance after weighting:
    Max |SMD|: 0.767
    Balance improvement: 14.6%

  Effective sample size:
    ESS: 240 / 458 (52.4%)

======================================================================
Permutation Weighting (Larger MLP)
======================================================================

Fitting weighter...

  Permutation-weighted ATE: $1734.75
  Error: $-59.25 (-3.3%)

  Covariate balance after weighting:
    Max |SMD|: 30.765
    Balance improvement: -3323.7%

  Effective sample size:
    ESS: 1 / 458 (0.2%)

======================================================================
Summary Comparison
======================================================================

Method                         ATE Estimate    Error           % Error     
------------------------------------------------------------------------
Experimental (Benchmark)       $     1794.00            ---            ---
Naive (Unadjusted)             $     4224.48  $     2430.48       135.5%
PW (Simple MLP)                $     5232.60  $     3438.60       191.7%
PW (Larger MLP)                $     1734.75  $      -59.25        -3.3%

  Improvement over naive:

  Improvement over naive: $-1008.12

======================================================================
✓ Lalonde experiment completed successfully!
⏱  Total execution time: 17.38 seconds
======================================================================
```


---

[View source on GitHub](https://github.com/ddimmery/stochpw/blob/main/examples/lalonde_experiment.py){ .md-button }
