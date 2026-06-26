# Tasks

## 1. Make the step jit-compatible (no behavior change)
- [ ] 1.1 Replace boolean-mask `logits[observed_mask]` (`loop.py:64`) with a
      static slice of the observed first half (`logits[: C.shape[0] // 2]`).
- [ ] 1.2 Confirm `RandomPermuter.permute` and `create_training_batch` use only
      jit-safe ops (they do); note any dynamic-shape hazards.
- [ ] 1.3 Run existing tests — must stay green (pure refactor so far).

## 2. JIT the training step
- [ ] 2.1 Add a `make_train_step(discriminator_fn, optimizer, loss_fn,
      regularizer, eps)` factory returning a `jax.jit`'d `step(params, opt_state,
      batch) -> (params, opt_state, loss)` with non-array deps closed over.
- [ ] 2.2 Keep the loss on-device (return the array; remove inner `float()`).
- [ ] 2.3 Have `fit_discriminator` build the jitted step once before the loop and
      call it per batch; keep the standalone `train_step` for API compatibility.
- [ ] 2.4 Equivalence test: weights & per-epoch loss identical (within fp tol) to
      pre-change output for fixed `random_state`, across linear + MLP.

## 3. Fold batch creation into the compiled region
- [ ] 3.1 Refactor `create_training_batch` to be driven by `(X, A, batch_indices,
      key)` such that it traces cleanly under jit (already index-based; verify).
- [ ] 3.2 Either include batch construction inside the jitted step, or jit it
      separately; benchmark which removes the ~2.5 ms/step eager floor better.

## 4. Collapse the inner batch loop with `lax.scan`
- [ ] 4.1 Precompute per-epoch shuffle, then build per-step batch-index slices +
      split keys; `jax.lax.scan` the step over them, carrying (params, opt_state).
- [ ] 4.2 Accumulate per-step losses in the scan carry; sync once per epoch.
- [ ] 4.3 Handle the `n % batch_size` remainder consistently with current loop
      (drops trailing partial batch today — preserve that).

## 5. Tooling, verification, docs
- [ ] 5.1 Add `make bench` running `benchmarks/bench_training.py`; document in
      CLAUDE.md "Common Tasks".
- [ ] 5.2 Re-run the benchmark; record before/after steps/sec & samples/sec in
      `benchmarks/RESULTS.md`. Target: end-to-end moves toward the `jitted` column.
- [ ] 5.3 `make lint`, `make typecheck`, `make test`, `make coverage` all pass;
      coverage not decreased.
- [ ] 5.4 Update `.claude/IMPLEMENTATION.md` status if it tracks perf work.

## 6. (Optional, GPU/TPU-aware follow-up)
- [ ] 6.1 Optionally `lax.fori_loop`/`scan` the epoch loop where early stopping
      allows; gate behind a flag if it complicates Python-side bookkeeping.
