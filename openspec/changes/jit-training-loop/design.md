# Design

## Context
The loop is dispatch/trace-bound (see `benchmarks/RESULTS.md`). The fix is to
move the per-step work into a single compiled artifact reused across steps,
without changing numerics or the public API.

## Goals / Non-Goals
- **Goals:** amortize tracing/compilation; remove eager per-step dispatch; keep
  loss on-device in the hot loop; preserve weights/loss bit-for-bit (within fp
  tolerance) for a fixed seed.
- **Non-Goals:** changing the permutation-weighting algorithm, the weight formula
  `w = η/(1-η)`, within-batch permutation semantics, or the public API surface.
  Removing the per-step host sync is *not* pursued as a CPU win (measured ~1×);
  it is done only as a side effect of scanning and for GPU/TPU portability.

## Decisions

### D1: JIT a step *factory*, not `train_step` directly
`train_step` takes non-array args (`discriminator_fn`, `optimizer`, `loss_fn`,
`regularizer`) that are awkward as `static_argnums` (optax transforms are
NamedTuples of closures; instances aren't reliably hashable). Instead introduce
`make_train_step(...) -> jax.jit(step)` that **closes over** those deps and traces
only `(params, opt_state, batch)`. This is the pattern validated in the
benchmark's `build_jitted_step` (65–170× faster).

`train_step` stays as a thin, un-jitted public function for composability; it is
simply no longer the per-step driver inside `fit_discriminator`.

### D2: Static slice instead of boolean mask
`logits[observed_mask]` (`loop.py:64`) is illegal under jit
(`NonConcreteBooleanIndexError`). `create_training_batch` concatenates
observed-then-permuted, so the observed rows are the static first half:
`half = C.shape[0] // 2; observed_logits = logits[:half]`. Equivalent result,
jit-safe, and avoids a dynamic gather.

### D3: Index-driven batch construction inside the compiled region
`create_training_batch` is already index-based (`X[batch_indices]`). Drive it from
`(batch_indices, key)` so the `einsum` interactions, `concatenate`s, and
`random.permutation` trace under XLA. This removes the ~2.5 ms/step eager floor
that dominates once the step is jitted.

### D4: `lax.scan` the inner batch loop
Per epoch: shuffle once (Python/JAX), then form a `(num_batches, batch_size)`
index matrix and a stacked key array, and `jax.lax.scan` the jitted step with
carry `(params, opt_state)`, stacking per-step losses as scan outputs. Collapses
`num_batches` Python iterations + dispatches into one compiled loop.

Remainder handling: today `num_batches = n // batch_size` drops the trailing
partial batch. `scan` requires uniform shapes, which **matches** this behavior —
preserve it (do not silently start training on a ragged last batch).

### D5: Epoch loop stays Python initially
Early stopping reads host-side loss and may copy best params; shuffling resplits
keys. Keeping epochs in Python keeps that bookkeeping simple. Scanning epochs is a
gated optional follow-up (task 6) once the inner-loop win is banked.

## Risks / Mitigations
- **Numerical drift** from reassociation under XLA fusion → assert closeness
  (`atol/rtol`) against the pre-change implementation for a fixed seed, not exact
  equality.
- **Recompilation on shape changes** (last-batch remainder, varying batch_size)
  → fixed batch shapes via D4; document that changing `batch_size` retriggers one
  compile (expected, amortized).
- **Regularizer/loss generality** — `regularizer(weights)` must stay traceable;
  current `NoRegularizer`/`Lp`/`Entropy` are pure JAX. Add a jit smoke test per
  regularizer + loss combination.

## Migration
No user-facing migration. Internally, `fit_discriminator` switches from calling
`train_step` per batch to building one jitted step (D1) and scanning it (D4).
