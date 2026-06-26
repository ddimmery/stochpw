# Training Spec Delta

## ADDED Requirements

### Requirement: Compiled single training step
The training loop SHALL execute each gradient step through a single
`jax.jit`-compiled function that is constructed once per `fit` call and reused
across all steps, so that autodiff tracing and primitive dispatch are not
repeated per step. The discriminator function, optimizer, loss, and regularizer
SHALL be held as static (closed-over) arguments; only `(params, opt_state,
batch)` SHALL be traced.

#### Scenario: Step is compiled once and reused
- **GIVEN** a `PermutationWeighter` configured with any discriminator, loss, and
  regularizer
- **WHEN** `fit` runs for multiple steps with a fixed `batch_size`
- **THEN** the per-step function is traced/compiled at most once for that batch
  shape
- **AND** subsequent steps reuse the compiled artifact (no per-step retrace)

#### Scenario: Numerical equivalence to the un-jitted loop
- **GIVEN** a fixed `random_state` and identical hyperparameters
- **WHEN** `fit` then `predict` is run under the compiled loop
- **THEN** the resulting weights and per-epoch loss history match the previous
  un-jitted implementation within floating-point tolerance (`rtol=1e-5`,
  `atol=1e-6`)

### Requirement: Jit-compatible observed-row selection
The step's regularization path SHALL select observed rows using a static slice of
the observed-then-permuted batch (the first half), not boolean-mask indexing, so
that the step traces cleanly under `jax.jit`.

#### Scenario: Regularized step compiles
- **GIVEN** a non-trivial regularizer (e.g. `LpRegularizer`)
- **WHEN** the compiled step is traced
- **THEN** tracing succeeds with no `NonConcreteBooleanIndexError`
- **AND** the regularization penalty equals the penalty computed on the observed
  half under the previous implementation

### Requirement: Host syncs excluded from the inner loop
The inner batch loop SHALL NOT transfer per-step scalars to host. Per-step losses
SHALL be accumulated on-device and synchronized at most once per epoch.

#### Scenario: No per-step device-to-host transfer
- **WHEN** an epoch of training executes
- **THEN** loss values are kept on-device during the batch loop
- **AND** the recorded per-epoch loss equals the mean of that epoch's step losses

### Requirement: Throughput benchmark target
The repository SHALL provide a documented command (`make bench`) that runs the
training-loop throughput benchmark and reports steps/sec and samples/sec, so that
performance changes are measurable and regressions are detectable.

#### Scenario: Benchmark runs and reports throughput
- **WHEN** `make bench` is executed
- **THEN** it runs `benchmarks/bench_training.py`
- **AND** prints steps/sec and samples/sec for the benchmark grid without
  modifying `src/stochpw`
