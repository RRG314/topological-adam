# Results

This repository reports wins, parity results, and losses. The benchmark
scripts tune each optimizer over the same learning-rate grid before reporting
comparisons.

## V3: recommended optimizer

The strongest current evidence for `TopologicalAdamV3` is:

- It reaches the measurement floor on the included ill-conditioned quadratic
  where tuned Adam stalls near 8e-5.
- It improves the included noisy teacher-student regression on 9 of 10 fresh
  seeds in the fresh-seed confirmation run.
- It is generally at parity with tuned Adam on ordinary noisy classification
  tasks, which is reported as parity rather than hidden.

V3 is the default recommendation because its coherence gate is designed to
close under minibatch noise.

## V4: trajectory-topology branch

`TopologicalAdamV4` is experimental and narrower. It targets trajectories with
loops or oscillation:

- On the synthetic rotating-field benchmark, V4 beats tuned Adam on 8 of 8
  fresh seeds.
- On sklearn digits MLP, V4 is at parity with tuned Adam.
- On noisy teacher-student regression, V4 loses to tuned Adam on 7 of 8 fresh
  seeds.

That result is intentional to report: V4 can react to incidental turning under
minibatch noise, so it is not recommended as a general stochastic optimizer.

## Persistent-homology evidence

The exact Vietoris-Rips H1 code in `topological_adam.persistence` is validated
as a detector:

- sampled circles produce a prominent normalized loop score;
- line, noise, degenerate, and straight-descent controls stay low;
- with `persistence_every > 0`, the H1 loop score feeds V4's momentum gate.

The benchmark table uses V4's default online turning/winding detector. Exact
H1 persistence is opt-in because it synchronizes to the host and is intended
for small trajectory windows or diagnostics.

## Reproduce

```bash
python examples/benchmark_v3_suite.py
python examples/confirm_fresh_seeds.py --results benchmark_v3_results.json
python examples/benchmark_v4_suite.py
python examples/make_topology_figures.py
```
