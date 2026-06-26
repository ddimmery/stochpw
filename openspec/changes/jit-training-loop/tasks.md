# Tasks

## 1. Make the step jit-compatible (no behavior change)
- [x] 1.1 Replace boolean-mask `logits[observed_mask]` (`loop.py:64`) with a
      static slice of the observed first half (`logits[: C.shape[0] // 2]`).
      Done in `_build_step` / `make_train_step`; original `train_step` keeps the
      mask (un-jitted, for API compat).
- [x] 1.2 Confirm `RandomPermuter.permute` and `create_training_batch` use only
      jit-safe ops (they do); note any dynamic-shape hazards. None — `len()` is on
      a static-shape index slice.
- [x] 1.3 Run existing tests — must stay green (pure refactor so far).

## 2. JIT the training step
- [x] 2.1 Add a `make_train_step(discriminator_fn, optimizer, loss_fn,
      regularizer, eps)` factory returning a `jax.jit`'d `step(params, opt_state,
      batch) -> (params, opt_state, loss)` with non-array deps closed over.
      `TrainingBatch` registered as a pytree so it crosses the jit boundary.
- [x] 2.2 Keep the loss on-device (return the array; remove inner `float()`).
- [x] 2.3 Have `fit_discriminator` build the compiled step/epoch once before the
      loop; keep the standalone `train_step` for API compatibility.
- [x] 2.4 Equivalence test: weights & per-epoch loss identical (within fp tol) to
      pre-change output for fixed `random_state`, across linear + MLP
      (`TestJitEquivalence`, kept reference implementation).

## 3. Fold batch creation into the compiled region
- [x] 3.1 Refactor `create_training_batch` to be driven by `(X, A, batch_indices,
      key)` such that it traces cleanly under jit (already index-based; verified).
- [x] 3.2 Batch construction runs inside the jitted/scanned epoch
      (`make_scan_epoch`), removing the ~2.5 ms/step eager floor.

## 4. Collapse the inner batch loop with `lax.scan`
- [x] 4.1 Shuffle per epoch, then `jax.lax.scan` the step over a
      `(num_batches, batch_size)` index matrix, carrying `(params, opt_state, key)`.
      The carry's initial key reproduces the original split chain exactly.
- [x] 4.2 Accumulate per-step losses as scan outputs; sync once per epoch.
- [x] 4.3 Handle the `n % batch_size` remainder consistently with current loop
      (drops trailing partial batch today — preserved).

## 5. Tooling, verification, docs
- [x] 5.1 Add `make bench` running `benchmarks/bench_training.py`; document in
      CLAUDE.md "Common Tasks".
- [x] 5.2 Re-run the benchmark; record before/after steps/sec & samples/sec in
      `benchmarks/RESULTS.md` (§6). End-to-end up to ~15–19× at n=10k.
- [x] 5.3 `make lint`, `make typecheck`, `make test`, `make coverage` all pass;
      coverage not decreased (98%; new `loop.py`/`data.py` code 100%).
- [~] 5.4 Update `.claude/IMPLEMENTATION.md` status if it tracks perf work.
      N/A — `.claude/IMPLEMENTATION.md` does not exist in this repo.

## 6. (Optional, GPU/TPU-aware follow-up)
- [ ] 6.1 Optionally `lax.fori_loop`/`scan` the epoch loop where early stopping
      allows; gate behind a flag if it complicates Python-side bookkeeping.
      (Out of scope — epoch loop intentionally stays in Python.)
