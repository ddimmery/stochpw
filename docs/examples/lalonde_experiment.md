# Lalonde Experiment

Lalonde experiment example with permutation weighting.

This example demonstrates using permutation weighting on the classic Lalonde (1986)
observational dataset to estimate the average treatment effect (ATE) of a job
training program on earnings.

The dataset combines experimental treatment units from the NSW program with
non-experimental control units, creating confounding and selection bias that
must be addressed to recover the experimental benchmark ATE of $1,794.

Reference:
    LaLonde, R. J. (1986). "Evaluating the Econometric Evaluations of Training
    Programs with Experimental Data". The American Economic Review, 76(4), 604-620.

## Code

```python
import jax.numpy as jnp
from stochpw import PermutationWeighter, MLPDiscriminator, effective_sample_size

# Load Lalonde NSW observational dataset
data = load_lalonde_nsw()
X, A, Y = data['X'], data['A'], data['Y']

# Fit permutation weighter with simple architecture
weighter = PermutationWeighter(
    discriminator=MLPDiscriminator(hidden_dims=[10]),
    num_epochs=500,
    batch_size=len(X),
    random_state=42,
)
weighter.fit(X, A)
weights = weighter.predict(X, A)

# Estimate ATE with permutation weights
ate_pw = estimate_ate(Y, A, weights)

# Compare to naive (unadjusted) estimate
ate_naive = estimate_ate(Y, A, jnp.ones_like(weights))

# Benchmark: Experimental ATE = $1,794
print(f"Naive ATE: ${ate_naive:.2f}")
print(f"Permutation-weighted ATE: ${ate_pw:.2f}")
print(f"Experimental benchmark: $1,794.00")
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
  Covariate names: age, education, black, hispanic, married, no_degree, earnings_1974, earnings_1975

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

======================================================================
Permutation Weighting
======================================================================

Fitting permutation weighter...

  Permutation-weighted ATE: $3058.82
  Error: $1264.82 (+70.5%)

  Weight diagnostics:
    Range: [0.000, 12.372]
    Mean: 0.580
    Std: 0.928
    ESS: 128.4 / 458 (28.0%)

  Covariate balance:
    Max |SMD| (naive): 0.899
    Max |SMD| (weighted): 0.462
    Improvement: 48.6%

  Balance by covariate:
    Covariate            Naive |SMD|     Weighted |SMD| 
    --------------------------------------------------
    age                          0.175           0.453
    education                    0.323           0.412
    black                        0.279           0.369
    hispanic                     0.087           0.013
    married                      0.220           0.157
    no_degree                    0.086           0.122
    earnings_1974                0.884           0.462
    earnings_1975                0.899           0.344

  Training:
    Initial loss: 375.4108
    Final loss: 0.5713
    Epochs: 5000

======================================================================
Summary
======================================================================

  Method                         ATE Estimate    Error           % Error   
  ----------------------------------------------------------------------
  Experimental Benchmark         $      1794.00   --              --        
  Naive (unadjusted)             $      4224.48   $     2430.48      135.5%
  Permutation Weighting          $      3058.82   $     1264.82       70.5%

  Improvement over naive: $1165.66

======================================================================
Lalonde experiment completed successfully!
======================================================================
```

??? example "Full source code"

    ```python
    """
    Lalonde experiment example with permutation weighting.
    
    This example demonstrates using permutation weighting on the classic Lalonde (1986)
    observational dataset to estimate the average treatment effect (ATE) of a job
    training program on earnings.
    
    The dataset combines experimental treatment units from the NSW program with
    non-experimental control units, creating confounding and selection bias that
    must be addressed to recover the experimental benchmark ATE of $1,794.
    
    Reference:
        LaLonde, R. J. (1986). "Evaluating the Econometric Evaluations of Training
        Programs with Experimental Data". The American Economic Review, 76(4), 604-620.
    """
    
    import os
    from pathlib import Path
    
    import jax.numpy as jnp
    import numpy as np
    from stochpw import (
        MLPDiscriminator,
        PermutationWeighter,
        effective_sample_size,
        standardized_mean_difference,
    )
    
    key_code = """import jax.numpy as jnp
    from stochpw import PermutationWeighter, MLPDiscriminator, effective_sample_size
    
    # Load Lalonde NSW observational dataset
    data = load_lalonde_nsw()
    X, A, Y = data['X'], data['A'], data['Y']
    
    # Fit permutation weighter with simple architecture
    weighter = PermutationWeighter(
        discriminator=MLPDiscriminator(hidden_dims=[10]),
        num_epochs=500,
        batch_size=len(X),
        random_state=42,
    )
    weighter.fit(X, A)
    weights = weighter.predict(X, A)
    
    # Estimate ATE with permutation weights
    ate_pw = estimate_ate(Y, A, weights)
    
    # Compare to naive (unadjusted) estimate
    ate_naive = estimate_ate(Y, A, jnp.ones_like(weights))
    
    # Benchmark: Experimental ATE = $1,794
    print(f"Naive ATE: ${ate_naive:.2f}")
    print(f"Permutation-weighted ATE: ${ate_pw:.2f}")
    print(f"Experimental benchmark: $1,794.00")
    """
    
    
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
        # Load data from CSV file
        script_dir = Path(__file__).parent
        data_path = script_dir / "nsw_data.csv"
    
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file not found at {data_path}. "
                "Please ensure nsw_data.csv is in the examples directory."
            )
    
        # Read CSV file (skip header)
        data = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    
        # Extract variables
        # Columns: treat, age, educ, black, hisp, married, nodegree, re74, re75, re78
        A = data[:, 0:1]  # Treatment
        Y = data[:, 9:10]  # Outcome (earnings in 1978)
    
        # Covariates: age, education, black, hispanic, married, nodegree, re74, re75
        X = data[:, 1:9]
    
        feature_names = [
            "age",
            "education",
            "black",
            "hispanic",
            "married",
            "no_degree",
            "earnings_1974",
            "earnings_1975",
        ]
    
        # Convert to JAX arrays
        X = jnp.array(X, dtype=jnp.float32)
        A = jnp.array(A, dtype=jnp.float32)
        Y = jnp.array(Y, dtype=jnp.float32)
    
        # Experimental benchmark from the original RCT
        # This is the "ground truth" we're trying to recover with observational methods
        ate_benchmark = 1794.0
    
        return {
            "X": X,
            "A": A,
            "Y": Y,
            "feature_names": feature_names,
            "ate_benchmark": ate_benchmark,
        }
    
    
    def estimate_ate(Y, A, weights):
        """
        Estimate average treatment effect using importance weights.
    
        Args:
            Y: Outcomes, shape (n, 1)
            A: Treatments, shape (n, 1)
            weights: Importance weights, shape (n,)
    
        Returns:
            float: Estimated ATE
        """
        # Reshape for broadcasting
        A_flat = A[:, 0]
        Y_flat = Y[:, 0]
    
        # Weighted means for treated and control
        treated_mask = A_flat == 1
        control_mask = A_flat == 0
    
        # Normalize weights within each group
        weights_treated = weights * treated_mask
        weights_treated = weights_treated / weights_treated.sum()
    
        weights_control = weights * control_mask
        weights_control = weights_control / weights_control.sum()
    
        # Weighted means
        y1_weighted = (Y_flat * weights_treated).sum()
        y0_weighted = (Y_flat * weights_control).sum()
    
        return float(y1_weighted - y0_weighted)
    
    
    def main():
        """Run Lalonde experiment with permutation weighting."""
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
    
        print(f"\nDataset statistics:")
        print(f"  Total samples: {len(X)}")
        print(f"  Treated: {n_treated}")
        print(f"  Control: {n_control}")
        print(f"  Covariates: {X.shape[1]}")
        print(f"  Covariate names: {', '.join(feature_names)}")
    
        # Benchmark ATE (from randomized experiment)
        print(f"\n{'='*70}")
        print("Experimental Benchmark (Ground Truth)")
        print(f"{'='*70}")
        print(f"  Experimental ATE: ${ate_benchmark:.2f}")
        print("  (From the original randomized controlled trial)")
    
        # Naive estimate (unweighted difference in means) on observational data
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
        print(f"\n  Covariate balance:")
        print(f"    Max |SMD|: {jnp.abs(smd_naive).max():.3f}")
        print(f"    (Values > 0.1 indicate imbalance)")
    
        # Permutation weighting on observational data
        print(f"\n{'='*70}")
        print("Permutation Weighting")
        print(f"{'='*70}")
    
        print("\nFitting permutation weighter...")
        weighter = PermutationWeighter(
            discriminator=MLPDiscriminator(hidden_dims=[32,8]),
            num_epochs=5000,
            batch_size=len(X),
            random_state=420,
            regularization_strength=.01,
        )
    
        weighter.fit(X, A)
        weights_pw = weighter.predict(X, A)
    
        # Estimate ATE
        ate_pw = estimate_ate(Y, A, weights_pw)
        pw_error = ate_pw - ate_benchmark
        pw_pct_error = (pw_error / ate_benchmark) * 100
    
        print(f"\n  Permutation-weighted ATE: ${ate_pw:.2f}")
        print(f"  Error: ${pw_error:.2f} ({pw_pct_error:+.1f}%)")
    
        # Weight diagnostics
        print(f"\n  Weight diagnostics:")
        print(f"    Range: [{weights_pw.min():.3f}, {weights_pw.max():.3f}]")
        print(f"    Mean: {weights_pw.mean():.3f}")
        print(f"    Std: {weights_pw.std():.3f}")
    
        ess = effective_sample_size(weights_pw)
        ess_ratio = ess / len(weights_pw)
        print(f"    ESS: {ess:.1f} / {len(weights_pw)} ({ess_ratio:.1%})")
    
        # Check balance improvement
        smd_pw = standardized_mean_difference(X, A, weights_pw)
        max_smd_naive = jnp.abs(smd_naive).max()
        max_smd_pw = jnp.abs(smd_pw).max()
        balance_improvement = (1 - max_smd_pw / max_smd_naive) * 100
    
        print(f"\n  Covariate balance:")
        print(f"    Max |SMD| (naive): {max_smd_naive:.3f}")
        print(f"    Max |SMD| (weighted): {max_smd_pw:.3f}")
        print(f"    Improvement: {balance_improvement:.1f}%")
    
        # Detailed balance by covariate
        print(f"\n  Balance by covariate:")
        print(f"    {'Covariate':<20} {'Naive |SMD|':<15} {'Weighted |SMD|':<15}")
        print(f"    {'-'*50}")
        for i, name in enumerate(feature_names):
            smd_n = abs(float(smd_naive[i]))
            smd_w = abs(float(smd_pw[i]))
            print(f"    {name:<20} {smd_n:>13.3f}   {smd_w:>13.3f}")
    
        # Training diagnostics
        assert weighter.history_ is not None
        loss_history = weighter.history_["loss"]
        print(f"\n  Training:")
        print(f"    Initial loss: {loss_history[0]:.4f}")
        print(f"    Final loss: {loss_history[-1]:.4f}")
        print(f"    Epochs: {len(loss_history)}")
    
        # Summary comparison
        print(f"\n{'='*70}")
        print("Summary")
        print(f"{'='*70}")
        print(f"\n  {'Method':<30} {'ATE Estimate':<15} {'Error':<15} {'% Error':<10}")
        print(f"  {'-'*70}")
        print(
            f"  {'Experimental Benchmark':<30} ${ate_benchmark:>13.2f}   "
            f"{'--':<13}   {'--':<10}"
        )
        print(
            f"  {'Naive (unadjusted)':<30} ${ate_naive:>13.2f}   "
            f"${naive_error:>12.2f}   {naive_pct_error:>8.1f}%"
        )
        print(
            f"  {'Permutation Weighting':<30} ${ate_pw:>13.2f}   "
            f"${pw_error:>12.2f}   {pw_pct_error:>8.1f}%"
        )
    
        improvement_over_naive = abs(naive_error) - abs(pw_error)
        print(f"\n  Improvement over naive: ${improvement_over_naive:.2f}")
    
        print(f"\n{'='*70}")
        print("Lalonde experiment completed successfully!")
        print(f"{'='*70}")
    
    
    if __name__ == "__main__":
        main()
    
    ```

    [View on GitHub](https://github.com/ddimmery/stochpw/blob/main/examples/lalonde_experiment.py){ .md-button }
