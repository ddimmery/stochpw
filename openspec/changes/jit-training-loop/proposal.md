# Jit the training loop for throughput

## Why
Profiling (`benchmarks/bench_training.py`, results in `benchmarks/RESULTS.md`)
shows the training loop is **dispatch/trace-bound, not compute-bound**:

- End-to-end throughput is flat at ~70 steps/s (linear) / ~45 (mlp) from
  n=250 → n=10,000 — fixed per-step overhead, not work, sets the pace.
- `train_step` (`src/stochpw/training/loop.py:19`) is not `@jax.jit`-decorated,
  so every step re-traces `jax.value_and_grad` and dispatches each primitive
  eagerly. cProfile attributes **78% of `fit` time** to `train_step`, almost all
  of it autodiff tracing + eager dispatch repeated once per step.
- An identical jitted step is **65–170× faster** (~10–21 ms → ~0.07–0.27 ms).
- The per-step `float(loss)` host sync is **negligible on CPU** (~1×), so it is
  explicitly out of scope for the CPU win.

## What Changes
- **JIT-compile the single training step.** Wrap the gradient + optimizer update
  in a `jax.jit`'d function with the discriminator/optimizer/loss/regularizer
  held as static (closed-over) arguments and `(params, opt_state, batch)` as
  traced arguments. Keep the loss on-device (return it; do not `float()` inside).
- **Make the step jit-compatible.** Replace the boolean-mask indexing
  `logits[observed_mask]` (`loop.py:64`) with a static slice of the observed
  first half (`create_training_batch` always concatenates observed-then-permuted).
- **Fold batch construction into the compiled region.** Drive
  `create_training_batch` from batch *indices* + key so its `einsum` /
  `concatenate` / permutation run inside XLA instead of eagerly per step.
- **Collapse the inner batch loop with `jax.lax.scan`** over per-step keys/index
  slices, removing Python-level per-step dispatch. Epoch loop may remain Python
  (shuffle + early-stopping bookkeeping) or also be scanned in a later phase.
- **Move host syncs out of the hot loop.** Accumulate per-step losses on-device
  and sync once per epoch (correctness-neutral; sets up GPU/TPU portability).
- **Add a regression-style throughput check** wiring `benchmarks/bench_training.py`
  into a documented `make bench` target.

Public API (`PermutationWeighter`, `fit_discriminator`, `train_step` signatures)
remains backwards compatible; `train_step` keeps working for composability but is
no longer the per-step driver inside `fit_discriminator`.

## Impact
- Affected specs: `training` (new performance + jit-compatibility requirements).
- Affected code: `src/stochpw/training/loop.py` (primary),
  `src/stochpw/training/batch.py` (index-driven batch build),
  optionally `src/stochpw/training/permutation.py` (key-driven permutation under
  scan). `Makefile` gains a `bench` target.
- Numerics: weights/loss must be unchanged vs. current implementation for a fixed
  `random_state` (verified by existing tests + a new equivalence test).
- Expected throughput: ~5× from jitting the step alone; tens-of-× with batch
  folding + `lax.scan`, bounded by remaining Python/epoch overhead.
