# Results

## Strongest Supported Findings

### 1. V2 is the clean supported path

The repository now exposes a clear supported path:
- `TopologicalAdamV2` for new work
- `TopologicalAdam` only for legacy comparison

### 2. Energy targeting is explicit in the implementation

The optimizer rescales the auxiliary fields toward a target-energy level directly in code. In the included synthetic diagnostics run, this produces very tight energy tracking.

This is an implementation fact plus an empirical observation. It is not yet a proof of any broader optimizer property.

### 3. `J_t` is a practical internal monitoring signal

In the included comparison run, `J_t` tends to fall as the loss falls. That makes it useful for monitoring and for a reconnection-style stopping heuristic.

## What Did Not Hold Up Cleanly

### `J_t` is not yet unique evidence of the topological correction

The supported comparison includes a control path with `eta=0` and `w_topo=0`. That control still shows strong `J_t`/loss correlation.

In the current seeded diagnostics run:
- control: `Pearson r(J_t, loss) ≈ 0.784`
- topological path: `Pearson r(J_t, loss) ≈ 0.785`
- recommended stopping epoch:
  - control path: epoch 3 by the absolute threshold
  - topological path: epoch 4 by the peak-drop rule

So the current repo should **not** claim:
- that `J_t` proves the topological mechanism is the source of convergence
- that the observed signal is unique to the topological correction term
- that the current experiment establishes a theorem-level optimizer result

## Current Classification

- Exact / implementation-level:
  - update equations as coded
  - target-energy rescaling step
  - version separation between V1 and V2
- Empirical:
  - stable synthetic diagnostics run
  - `J_t`/loss correlation in tested runs
  - useful stopping heuristic in the tested horizon
- Open:
  - benchmark-level advantage over Adam
  - transfer of the stopping rule to real workloads
  - mechanistic interpretation of `J_t` beyond the current experiments
