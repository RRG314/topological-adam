# Overview

`topological-adam` is a small PyTorch optimizer package for testing
Adam-compatible update rules with explicit reliability gates. It is organized
as an optimizer family rather than a single universal replacement for Adam.

## Recommended path

- Use `TopologicalAdamV3` for new experiments.
- V3 adds slow and fast gradient-EMA fields, uses their disagreement as a
  short-term trend signal, and gates that correction by alignment with the
  current gradient.
- With `w_topo=0` and `cautious=False`, V3 reduces to Adam or AdamW. This is
  covered by tests.

## Experimental path

- Use `TopologicalAdamV4` only for loop- or oscillation-structured dynamics.
- V4 records projected update trajectories and gates momentum using online
  turning/winding statistics. Optional exact Vietoris-Rips H1 persistence can
  feed the same gate for small trajectory windows.
- With `loop_gate=False`, V4 reduces to Adam or AdamW. Straight trajectories
  also keep the gate at one.

## Legacy paths

- `TopologicalAdam` and `TopologicalAdamV2` are retained for provenance and
  comparison.
- `TopologicalAdamSDS` is an experimental two-temperature efficiency gate.

## Where to look

- `README.md`: installation, quick usage, and summary evidence.
- `docs/trajectory-topology.md`: V4 design, limitations, and benchmarks.
- `docs/results.md`: current benchmark interpretation.
- `paper.md`: JOSS paper source.
- `tests/`: reduction, behavior, persistence, and integration tests.
