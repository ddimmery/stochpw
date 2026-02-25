"""Stochastic tests for stochpw using pytest-stochastic.

These tests verify statistical properties of the permutation weighting algorithm
across many random seeds, with mathematically-guaranteed false failure rates
via concentration inequalities.

Each test returns a scalar where negative values indicate the desired property holds.
The @stochastic_test decorator runs each test many times and uses concentration
inequalities to verify that the expected property holds in general.

Run `pytest --stochastic-tune` to discover variance estimates and save to
.stochastic.toml for dramatically faster subsequent runs (Bernstein bounds).
"""

import jax
import jax.numpy as jnp
import optax
from pytest_stochastic import stochastic_test

from stochpw import (
    MLPDiscriminator,
    PermutationWeighter,
    balance_report,
    roc_curve,
)
from stochpw.diagnostics import standardized_mean_difference
from stochpw.models import LinearDiscriminator
from stochpw.training import fit_discriminator


def _make_jax_key(rng):
    """Convert pytest-stochastic NumPy rng to JAX PRNGKey and integer seed."""
    seed = int(rng.integers(0, 2**31))
    return jax.random.PRNGKey(seed), seed


def _generate_confounded_data(key, n=30, n_features=3):
    """Generate synthetic confounded data with a strong treatment-covariate relationship."""
    key1, key2 = jax.random.split(key)
    X = jax.random.normal(key1, (n, n_features))
    propensity = jax.nn.sigmoid(0.5 * X[:, 0] - 0.3 * X[:, 1])
    A = jax.random.bernoulli(key2, propensity).astype(float)
    return X, A


@stochastic_test(
    expected=0, side="less", atol=0.15, bounds=(-0.5, 0.3), failure_prob=1e-6
)
def test_stochastic_training_converges(rng):
    """Test that PermutationWeighter training reduces loss."""
    key, seed = _make_jax_key(rng)
    X, A = _generate_confounded_data(key, n=30, n_features=3)

    weighter = PermutationWeighter(
        num_epochs=3,
        batch_size=30,
        random_state=seed,
        optimizer=optax.rmsprop(learning_rate=0.1),
    )
    weighter.fit(X, A)

    return float(weighter.history_["loss"][-1] - weighter.history_["loss"][0])


@stochastic_test(
    expected=0, side="less", atol=0.3, bounds=(-1.5, 1.0), failure_prob=1e-6
)
def test_stochastic_balance_improvement(rng):
    """Test that permutation weights improve covariate balance."""
    key, seed = _make_jax_key(rng)
    X, A = _generate_confounded_data(key, n=50, n_features=3)

    weighter = PermutationWeighter(
        num_epochs=5,
        batch_size=50,
        random_state=seed,
        optimizer=optax.rmsprop(learning_rate=0.1),
    )
    weighter.fit(X, A)
    weights = weighter.predict(X, A)

    smd_unweighted = standardized_mean_difference(X, A, jnp.ones_like(weights))
    smd_weighted = standardized_mean_difference(X, A, weights)

    max_smd_unw = float(jnp.abs(smd_unweighted).max())
    max_smd_w = float(jnp.abs(smd_weighted).max())

    return max_smd_w - max_smd_unw


@stochastic_test(
    expected=0, side="less", atol=0.15, bounds=(-0.6, 0.3), failure_prob=1e-6
)
def test_stochastic_mlp_convergence(rng):
    """Test that MLP discriminator training reduces loss."""
    key, seed = _make_jax_key(rng)
    X, A = _generate_confounded_data(key, n=30, n_features=3)

    weighter = PermutationWeighter(
        discriminator=MLPDiscriminator(hidden_dims=[16]),
        num_epochs=5,
        batch_size=30,
        random_state=seed,
        optimizer=optax.rmsprop(learning_rate=0.1),
    )
    weighter.fit(X, A)

    return float(weighter.history_["loss"][-1] - weighter.history_["loss"][0])


@stochastic_test(
    expected=0, side="less", atol=0.15, bounds=(-0.5, 0.3), failure_prob=1e-6
)
def test_stochastic_fit_discriminator_convergence(rng):
    """Test that fit_discriminator reduces loss over training."""
    key, _ = _make_jax_key(rng)
    key1, key2, key3, key4 = jax.random.split(key, 4)

    n = 30
    X = jax.random.normal(key1, (n, 3))
    A = jax.random.bernoulli(key2, 0.5, (n,)).astype(float).reshape(-1, 1)

    d_a, d_x = A.shape[1], X.shape[1]
    discriminator = LinearDiscriminator()
    params = discriminator.init_params(key3, d_a, d_x)
    optimizer = optax.adam(1e-2)

    _, history = fit_discriminator(
        X=X,
        A=A,
        discriminator_fn=discriminator.apply,
        init_params=params,
        optimizer=optimizer,
        num_epochs=5,
        batch_size=n,
        rng_key=key4,
    )

    return float(history["loss"][-1] - history["loss"][0])


@stochastic_test(
    expected=0, side="less", atol=0.15, bounds=(-0.5, 0.5), failure_prob=1e-6
)
def test_stochastic_roc_auc_above_chance(rng):
    """Test that a trained discriminator achieves AUC > 0.5."""
    key, seed = _make_jax_key(rng)
    key1, key2 = jax.random.split(key)

    n = 50
    X, A = _generate_confounded_data(key1, n=n, n_features=3)

    weighter = PermutationWeighter(
        num_epochs=5,
        batch_size=n,
        random_state=seed,
        optimizer=optax.rmsprop(learning_rate=0.1),
    )
    weighter.fit(X, A)

    weights_obs = weighter.predict(X, A)
    A_perm = jax.random.permutation(key2, A)
    weights_perm = weighter.predict(X, A_perm)

    all_weights = jnp.concatenate([weights_obs, weights_perm])
    all_labels = jnp.concatenate([jnp.zeros(len(weights_obs)), jnp.ones(len(weights_perm))])

    _, tpr, _ = roc_curve(all_weights, all_labels)
    fpr_vals = jnp.linspace(0, 1, len(tpr))
    auc = float(jnp.trapezoid(tpr, fpr_vals))

    return 0.5 - auc


@stochastic_test(
    expected=0, side="less", atol=0.3, bounds=(-1.5, 1.0), failure_prob=1e-6
)
def test_stochastic_balance_report_improvement(rng):
    """Test that balance report shows improvement after permutation weighting."""
    key, seed = _make_jax_key(rng)
    X, A = _generate_confounded_data(key, n=50, n_features=3)

    weighter = PermutationWeighter(
        num_epochs=5,
        batch_size=50,
        random_state=seed,
        optimizer=optax.rmsprop(learning_rate=0.1),
    )
    weighter.fit(X, A)
    weights = weighter.predict(X, A)

    report_before = balance_report(X, A, jnp.ones(len(A)))
    report_after = balance_report(X, A, weights)

    return float(report_after["max_smd"] - report_before["max_smd"])
