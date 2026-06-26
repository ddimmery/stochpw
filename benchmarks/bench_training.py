#!/usr/bin/env python
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Training-loop throughput benchmark for stochpw
#
# This is a **profiling/benchmark harness only** — it does not modify anything in
# `src/stochpw`. It measures *throughput* (steps/sec, samples/sec) of the
# permutation-weighting training loop, attributes wall-clock time across phases,
# and estimates the achievable gain from a fully-jitted, sync-free step.
#
# Run:
#     PYTHONPATH=src uv run python benchmarks/bench_training.py            # default grid
#     PYTHONPATH=src uv run python benchmarks/bench_training.py --full     # add large-n sweep
#     PYTHONPATH=src uv run python benchmarks/bench_training.py --profile  # + cProfile + jax trace
#
# Metrics convention:
#   * A "step" = one mini-batch gradient update.
#   * "samples/sec" counts 2 * batch_size per step (observed + permuted halves).
#   * Compilation (first 1-2 steps) is measured separately from steady-state.
#   * Every timed region calls jax.block_until_ready(...) before stopping the clock.

# %%
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from jax import Array

from stochpw import LinearDiscriminator, MLPDiscriminator, PermutationWeighter
from stochpw.models import BaseDiscriminator
from stochpw.training import (
    LogisticLoss,
    NoRegularizer,
    RandomPermuter,
    create_training_batch,
    train_step,
)
from stochpw.types import PyTree

# %% [markdown]
# ## Utilities


# %%
def env_info() -> None:
    """Print environment context relevant to throughput interpretation."""
    print("=" * 72)
    print("ENVIRONMENT")
    print("=" * 72)
    print(f"  jax version      : {jax.__version__}")
    print(f"  default backend  : {jax.default_backend()}")
    print(f"  devices          : {jax.devices()}")
    print("  XLA flags        : (see XLA_FLAGS env if set)")
    print()


def make_data(n: int, d_x: int, d_a: int, seed: int = 0) -> tuple[Array, Array]:
    """Generate synthetic confounded data, mirroring examples/basic_usage.py.

    Treatment depends on covariates so the discriminator has signal to learn.
    """
    key = jax.random.PRNGKey(seed)
    x_key, a_key = jax.random.split(key)
    X = jax.random.normal(x_key, (n, d_x))
    # Confounded propensity from first few covariates.
    coef = jnp.linspace(0.5, -0.3, num=min(d_x, 3))
    logits = X[:, : coef.shape[0]] @ coef + 0.2
    propensity = jax.nn.sigmoid(logits)
    if d_a == 1:
        A = jax.random.bernoulli(a_key, propensity, (n,)).astype(jnp.float32).reshape(-1, 1)
    else:
        # Multi-dim treatment: continuous, correlated with covariates.
        a_keys = jax.random.split(a_key, d_a)
        cols = [
            (propensity + 0.1 * jax.random.normal(a_keys[j], (n,))).reshape(-1, 1)
            for j in range(d_a)
        ]
        A = jnp.concatenate(cols, axis=1).astype(jnp.float32)
    return jax.block_until_ready(X), jax.block_until_ready(A)


def timed_steady_state(
    fn: Callable[[], object],
    *,
    warmup: int = 2,
    iters: int = 50,
) -> tuple[float, float]:
    """Time a no-arg callable, separating compile/warmup from steady state.

    Returns (warmup_first_call_sec, steady_state_median_sec_per_call).
    The callable must return a device value (or pytree) we can block on.
    """
    # First call: includes tracing + compilation.
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    first = time.perf_counter() - t0

    # Remaining warmup (cache should be warm now).
    for _ in range(max(0, warmup - 1)):
        jax.block_until_ready(fn())

    # Steady-state timing, per-call.
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - t0)
    return first, statistics.median(samples)


# %% [markdown]
# ## Benchmark grid configuration


# %%
@dataclass(frozen=True)
class Case:
    n: int
    d_x: int
    d_a: int
    batch_size: int
    disc: str  # "linear" | "mlp"

    def label(self) -> str:
        return (
            f"n={self.n:>7} d_x={self.d_x:>3} d_a={self.d_a} "
            f"bs={self.batch_size:>6} disc={self.disc:<6}"
        )


def make_discriminator(name: str) -> BaseDiscriminator:
    if name == "linear":
        return LinearDiscriminator()
    if name == "mlp":
        return MLPDiscriminator(hidden_dims=[64, 32])
    raise ValueError(name)


def default_grid(full: bool) -> list[Case]:
    cases: list[Case] = []
    # Example-scale cases (the throughput regime the user cares about).
    sizes = [(250, 62), (1_000, 256), (10_000, 256)]
    if full:
        sizes += [(100_000, 1024)]
    for n, bs in sizes:
        for disc in ("linear", "mlp"):
            cases.append(Case(n=n, d_x=5, d_a=1, batch_size=bs, disc=disc))
    # Wide covariates -> large A(x)X interaction term (d_a*d_x), linear only.
    cases.append(Case(n=1_000, d_x=50, d_a=1, batch_size=256, disc="linear"))
    # Full-batch variant.
    cases.append(Case(n=1_000, d_x=5, d_a=1, batch_size=1_000, disc="linear"))
    return cases


# %% [markdown]
# ## (1) End-to-end throughput via the public API
#
# Times `PermutationWeighter.fit()` exactly as a user would call it, and converts
# total wall-clock into steps/sec and samples/sec. This is the headline "as-is"
# number, inclusive of all Python-loop and host-sync overhead.


# %%
def bench_end_to_end(case: Case, num_epochs: int = 20) -> dict[str, float]:
    X, A = make_data(case.n, case.d_x, case.d_a, seed=0)
    steps_per_epoch = case.n // case.batch_size
    total_steps = steps_per_epoch * num_epochs

    opt = optax.adam(1e-3)
    weighter = PermutationWeighter(
        discriminator=make_discriminator(case.disc),
        optimizer=opt,
        num_epochs=num_epochs,
        batch_size=case.batch_size,
        random_state=42,
    )

    # Warm one fit so XLA caches are hot, then time a fresh fit.
    weighter_warm = PermutationWeighter(
        discriminator=make_discriminator(case.disc),
        optimizer=optax.adam(1e-3),
        num_epochs=1,
        batch_size=case.batch_size,
        random_state=42,
    )
    jax.block_until_ready(weighter_warm.fit(X, A).params_)

    t0 = time.perf_counter()
    weighter.fit(X, A)
    jax.block_until_ready(weighter.params_)
    elapsed = time.perf_counter() - t0

    steps_per_sec = total_steps / elapsed if elapsed > 0 else float("nan")
    samples_per_sec = steps_per_sec * 2 * case.batch_size
    return {
        "elapsed_s": elapsed,
        "total_steps": float(total_steps),
        "steps_per_sec": steps_per_sec,
        "samples_per_sec": samples_per_sec,
    }


# %% [markdown]
# ## (2) Phase attribution + (5) ideal jitted reference
#
# Builds the same per-step computation three ways and times steady-state cost:
#   * `batch`     : create_training_batch() alone (eager batch construction).
#   * `as_is`     : train_step() as called in the loop — re-traces value_and_grad
#                   every call, plus a per-step float(loss) host sync.
#   * `jitted`    : a jax.jit closure over (params, opt_state, batch) that keeps
#                   the loss on-device (no host sync). This is the achievable
#                   upper bound for the step compute on this hardware.
#
# The ratio jitted_throughput / as_is_throughput estimates the plausible gain.


# %%
def build_jitted_step(
    discriminator_fn: Callable[[PyTree, Array, Array, Array], Array],
    optimizer: optax.GradientTransformation,
    eps: float = 1e-7,
):
    loss_fn = LogisticLoss()
    regularizer = NoRegularizer()

    def step(params: PyTree, opt_state, batch_AX):
        A_b, X_b, AX_b, C_b = batch_AX

        def loss_total(p: PyTree) -> Array:
            logits = discriminator_fn(p, A_b, X_b, AX_b)
            loss = loss_fn(logits, C_b)
            # Observed rows are the static first half (create_training_batch
            # concatenates observed-then-permuted). Static slice keeps this
            # jit-compatible, unlike boolean-mask indexing in train_step.
            half = C_b.shape[0] // 2
            observed_logits = logits[:half]
            eta = jax.nn.sigmoid(observed_logits)
            eta_clipped = jnp.clip(eta, eps, 1 - eps)
            weights = eta_clipped / (1 - eta_clipped)
            return loss + regularizer(weights)

        loss, grads = jax.value_and_grad(loss_total)(params)
        updates, opt_state2 = optimizer.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss  # loss kept on-device

    return jax.jit(step)


def bench_phases(case: Case, iters: int = 50) -> dict[str, dict[str, float]]:
    X, A = make_data(case.n, case.d_x, case.d_a, seed=0)
    disc = make_discriminator(case.disc)
    key = jax.random.PRNGKey(7)
    init_key, perm_key, step_key = jax.random.split(key, 3)
    params = disc.init_params(init_key, case.d_a, case.d_x)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    permuter = RandomPermuter()

    bs = case.batch_size
    batch_indices = jnp.arange(0, bs)

    # --- (2a) batch creation alone -------------------------------------------
    def make_batch() -> object:
        b = create_training_batch(X, A, batch_indices, perm_key, permuter=permuter)
        return (b.A, b.X, b.AX, b.C)

    first_batch, med_batch = timed_steady_state(make_batch, iters=iters)

    # Reuse a fixed batch for the step microbenchmarks (isolate the step).
    batch = create_training_batch(X, A, batch_indices, perm_key, permuter=permuter)
    jax.block_until_ready((batch.A, batch.X, batch.AX, batch.C))

    # --- (2b) as-is train_step (re-traces value_and_grad + host sync) --------
    from stochpw.data import TrainingState

    state = TrainingState(
        params=params, opt_state=opt_state, rng_key=step_key, epoch=0, history={"loss": []}
    )

    def as_is_step() -> object:
        result = train_step(
            state,
            batch,
            disc.apply,
            optimizer,
            loss_fn=LogisticLoss(),
            regularizer=NoRegularizer(),
        )
        # Mirror the loop's per-step host sync.
        _ = float(result.loss)
        return result.state.params

    first_asis, med_asis = timed_steady_state(as_is_step, iters=iters)

    # --- (5) jitted, sync-free step ------------------------------------------
    jstep = build_jitted_step(disc.apply, optimizer)
    batch_tuple = (batch.A, batch.X, batch.AX, batch.C)
    p_ref = [params]
    os_ref = [opt_state]

    def jitted_step() -> object:
        p2, os2, loss = jstep(p_ref[0], os_ref[0], batch_tuple)
        p_ref[0] = p2
        os_ref[0] = os2
        return loss  # on-device; block happens in timer

    first_jit, med_jit = timed_steady_state(jitted_step, iters=iters)

    return {
        "batch": {"first": first_batch, "median": med_batch},
        "as_is": {"first": first_asis, "median": med_asis},
        "jitted": {"first": first_jit, "median": med_jit},
    }


# %% [markdown]
# ## (2c) Host-sync cost: per-step float(loss) vs accumulate-then-sync
#
# Isolates the cost of the device->host transfer done every batch at
# loop.py:205 (`epoch_losses.append(float(result.loss))`).


# %%
def bench_host_sync(case: Case, n_steps: int = 200) -> dict[str, float]:
    X, A = make_data(case.n, case.d_x, case.d_a, seed=0)
    disc = make_discriminator(case.disc)
    key = jax.random.PRNGKey(11)
    init_key, perm_key = jax.random.split(key)
    params = disc.init_params(init_key, case.d_a, case.d_x)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    permuter = RandomPermuter()
    bs = case.batch_size
    batch = create_training_batch(X, A, jnp.arange(0, bs), perm_key, permuter=permuter)
    batch_tuple = (batch.A, batch.X, batch.AX, batch.C)
    jstep = build_jitted_step(disc.apply, optimizer)

    # Warm.
    p, os_, lo = jstep(params, opt_state, batch_tuple)
    jax.block_until_ready((p, os_, lo))

    # Variant A: sync every step (float(loss)).
    p, os_ = params, opt_state
    t0 = time.perf_counter()
    for _ in range(n_steps):
        p, os_, loss = jstep(p, os_, batch_tuple)
        _ = float(loss)  # host sync each step
    sync_each = time.perf_counter() - t0

    # Variant B: keep losses on-device, single sync at the end.
    p, os_ = params, opt_state
    losses = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        p, os_, loss = jstep(p, os_, batch_tuple)
        losses.append(loss)  # stays on device
    _ = float(jnp.stack(losses).mean())  # one sync
    sync_once = time.perf_counter() - t0

    return {
        "sync_each_s": sync_each,
        "sync_once_s": sync_once,
        "speedup": sync_each / sync_once if sync_once > 0 else float("nan"),
        "steps": float(n_steps),
    }


# %% [markdown]
# ## Reporting


# %%
def run_grid(full: bool) -> None:
    cases = default_grid(full)
    print("=" * 72)
    print("(1) END-TO-END THROUGHPUT  (PermutationWeighter.fit, 20 epochs)")
    print("=" * 72)
    e2e: dict[str, dict[str, float]] = {}
    for c in cases:
        r = bench_end_to_end(c)
        e2e[c.label()] = r
        print(
            f"  {c.label()} | {r['steps_per_sec']:>10.1f} steps/s | "
            f"{r['samples_per_sec']:>13.0f} samples/s | {r['elapsed_s']:.3f}s"
        )
    print()

    print("=" * 72)
    print("(2)+(5) PER-STEP PHASE ATTRIBUTION  (steady-state median, ms/step)")
    print("=" * 72)
    print(f"  {'case':<52} {'batch':>9} {'as_is':>9} {'jitted':>9} {'gain x':>8}")
    for c in cases:
        ph = bench_phases(c)
        b = ph["batch"]["median"] * 1e3
        a = ph["as_is"]["median"] * 1e3
        j = ph["jitted"]["median"] * 1e3
        gain = a / j if j > 0 else float("nan")
        print(f"  {c.label():<52} {b:>9.3f} {a:>9.3f} {j:>9.3f} {gain:>7.1f}x")
    print()

    print("=" * 72)
    print("(2c) HOST-SYNC COST  (200 jitted steps; sync-each vs sync-once)")
    print("=" * 72)
    for c in cases:
        r = bench_host_sync(c)
        print(
            f"  {c.label()} | each={r['sync_each_s']:.4f}s "
            f"once={r['sync_once_s']:.4f}s | {r['speedup']:.2f}x"
        )
    print()


# %% [markdown]
# ## (3)+(4) Optional deep profiling: jax.profiler trace + cProfile


# %%
def run_deep_profile() -> None:
    case = Case(n=1_000, d_x=5, d_a=1, batch_size=256, disc="linear")
    X, A = make_data(case.n, case.d_x, case.d_a, seed=0)

    # (4) cProfile of a full fit at a small size where Python overhead dominates.
    print("=" * 72)
    print("(4) cProfile: PermutationWeighter.fit(n=1000, linear, 20 epochs)")
    print("=" * 72)
    weighter = PermutationWeighter(
        discriminator=LinearDiscriminator(),
        optimizer=optax.adam(1e-3),
        num_epochs=20,
        batch_size=case.batch_size,
        random_state=42,
    )
    # Warm caches first so cProfile reflects steady state, not compilation.
    jax.block_until_ready(
        PermutationWeighter(
            optimizer=optax.adam(1e-3), num_epochs=1, batch_size=case.batch_size, random_state=42
        )
        .fit(X, A)
        .params_
    )
    pr = cProfile.Profile()
    pr.enable()
    weighter.fit(X, A)
    jax.block_until_ready(weighter.params_)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())

    # (3) jax.profiler trace (view with TensorBoard / Perfetto).
    trace_dir = "/tmp/stochpw-trace"
    print("=" * 72)
    print(f"(3) jax.profiler trace -> {trace_dir}")
    print("=" * 72)
    try:
        with jax.profiler.trace(trace_dir):
            w = PermutationWeighter(
                optimizer=optax.adam(1e-3),
                num_epochs=10,
                batch_size=case.batch_size,
                random_state=42,
            )
            w.fit(X, A)
            jax.block_until_ready(w.params_)
        print(f"  Trace written. Inspect with: tensorboard --logdir {trace_dir}")
        print("  (or load the .trace.json.gz in https://ui.perfetto.dev)")
    except Exception as exc:  # pragma: no cover - profiler optional
        print(f"  jax.profiler.trace unavailable: {exc}")
    print()


# %% [markdown]
# ## Main


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="add large-n sweep (slow on CPU)")
    parser.add_argument("--profile", action="store_true", help="run cProfile + jax.profiler trace")
    args = parser.parse_args()

    t0 = time.time()
    env_info()
    run_grid(full=args.full)
    if args.profile:
        run_deep_profile()
    print(f"⏱  Total benchmark wall-clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
