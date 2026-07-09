# Topological Adam

[![Release](https://img.shields.io/github/v/release/RRG314/topological-adam?display_name=tag)](https://github.com/RRG314/topological-adam/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/RRG314/topological-adam/blob/main/LICENSE)
[![Tests](https://github.com/RRG314/topological-adam/actions/workflows/tests.yml/badge.svg)](https://github.com/RRG314/topological-adam/actions/workflows/tests.yml)

Topological Adam is a PyTorch optimizer package for studying Adam-compatible
update rules with field-dynamics and trajectory-topology diagnostics. It keeps
the standard `torch.optim.Optimizer` interface while exposing reproducible
baselines, ablations, and internal metrics for optimizer research.

The current recommended optimizer is `TopologicalAdamV3`. `TopologicalAdamV4`
is an experimental branch for update trajectories with loop or oscillation
structure.

## Statement of Need

Optimizer papers and software releases are hard to evaluate when a proposed
method cannot be installed, reduced to a baseline, or reproduced under the same
tuning protocol as Adam. This repository provides an installable implementation
of several related Adam-family optimizers, together with tests, benchmark
scripts, stored benchmark outputs, and JOSS paper source.

The target users are machine-learning researchers, numerical-methods
researchers, and reviewers who want to inspect or benchmark Adam-style
mechanisms with explicit diagnostics. The package complements `torch.optim`
rather than replacing it: the main variants retain Adam-style state and include
disabled-mechanism settings that reduce to Adam or AdamW for tests and
ablations.

## Installation

Published releases are installed with `pip`:

```bash
pip install topological-adam
```

For JOSS review or unreleased development, install from the repository:

```bash
git clone https://github.com/RRG314/topological-adam.git
cd topological-adam
pip install -e .
pip install -r requirements-dev.txt
```

## Quickstart

```python
import torch.nn as nn
from topological_adam import TopologicalAdamV3

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
optimizer = TopologicalAdamV3(model.parameters(), lr=1e-3)
```

`TopologicalAdamV3` can be used anywhere a standard PyTorch optimizer is used:

```python
for batch, target in loader:
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(batch), target)
    loss.backward()
    optimizer.step()
```

## Optimizer Family

| Optimizer | Role | Notes |
| --- | --- | --- |
| `TopologicalAdam` | Legacy implementation | Preserved for backward compatibility and provenance. |
| `TopologicalAdamV2` | Previous supported branch | Adds reusable field diagnostics and stopping helpers. |
| `TopologicalAdamV3` | Recommended branch | Uses slow and fast gradient-EMA fields, a coherence gate, and optional cautious updates. With `w_topo=0, cautious=False`, it reduces exactly to Adam or AdamW. |
| `TopologicalAdamSDS` | Experimental branch | SDS-inspired optimizer candidate retained for comparison and further testing. |
| `TopologicalAdamV4` | Experimental trajectory-topology branch | Computes projected turning/winding statistics from the optimizer trajectory, with optional exact Vietoris-Rips H1 persistence. With `loop_gate=False`, it reduces exactly to Adam or AdamW. |

## Diagnostics

V3 exposes field-level diagnostics when `track_stats=True`:

```python
optimizer = TopologicalAdamV3(model.parameters(), lr=1e-3, track_stats=True)

# after optimizer.step()
stats = optimizer.field_metrics()
print(stats["energy"], stats["j_t"], stats["gate"], stats["align_cos"])
```

V4 exposes trajectory diagnostics for loop-structured dynamics:

```python
from topological_adam import TopologicalAdamV4

optimizer = TopologicalAdamV4(model.parameters(), lr=1e-3)

# after several optimizer steps
for metric in optimizer.trajectory_metrics():
    print(metric["gate"], metric["winding"], metric["kappa_ema"])
```

The exact persistent-homology path is opt-in because it is intended for small
trajectory windows and diagnostics:

```python
optimizer = TopologicalAdamV4(
    model.parameters(),
    lr=1e-3,
    n_planes=3,
    persistence_every=50,
)
```

See [docs/trajectory-topology.md](docs/trajectory-topology.md) for the V4
mechanism, limitations, and benchmark interpretation.

## Evidence and Reproducibility

The repository reports wins, parity results, and losses. Benchmark scripts tune
each optimizer over the same learning-rate grid before fresh-seed evaluation.

- V3 reaches the measurement floor on the included ill-conditioned quadratic
  where tuned Adam stalls near `8e-5`.
- V3 improves the included noisy teacher-student regression on 9 of 10 fresh
  seeds in `fresh_seed_confirmation.json`.
- V3 is generally at parity with tuned Adam on ordinary noisy classification
  tasks.
- V4 wins on the synthetic rotating-field task it targets, is at parity on
  sklearn digits MLP, and loses on noisy teacher-student regression.
- A small reviewer-facing reference benchmark trains a real-data sklearn digits
  MLP with Adam, AdamW, V3, and V4 under a documented tune-then-fresh protocol.

Entry points:

- Reference training benchmark: [docs/reference-training-benchmark.md](docs/reference-training-benchmark.md)
- V3 audit and benchmark details: [docs/v3-audit.md](docs/v3-audit.md)
- Results summary: [docs/results.md](docs/results.md)
- V4 trajectory topology: [docs/trajectory-topology.md](docs/trajectory-topology.md)
- Stored reference training results: [reference_training_results.json](reference_training_results.json)
- Stored V3 results: [benchmark_v3_results.json](benchmark_v3_results.json)
- Stored V4 results: [benchmark_v4_results.json](benchmark_v4_results.json)

Reproduce the stored benchmark outputs:

```bash
python examples/reference_training_benchmark.py --out reference_training_results.json
python examples/benchmark_v3_suite.py
python examples/confirm_fresh_seeds.py --results benchmark_v3_results.json
python examples/benchmark_v4_suite.py
python examples/make_topology_figures.py
```

## Tests and Packaging

Run the test suite:

```bash
python -m pytest tests/ -q
```

Build and check the package:

```bash
python -m build
python -m twine check dist/*
```

The tests cover exact Adam/AdamW reduction, optimizer state compatibility,
V3 field gating, V4 winding and persistence computations, trajectory gating
behavior, benchmark smoke paths, and legacy imports. GitHub Actions runs the
test suite on Python 3.10 and 3.12.

## Repository Layout

- `topological_adam/`: installable Python package.
- `examples/`: benchmark, reproduction, and figure-generation scripts.
- `tests/`: unit, integration, reduction, and smoke tests.
- `docs/`: reviewer-facing method notes, benchmark summaries, and figures.
- `paper.md` and `paper.bib`: JOSS paper source and references.
- `CITATION.cff`: citation metadata for software reuse.
- `CONTRIBUTING.md`: contribution, issue, and support guidelines.

## JOSS Materials

The JOSS paper source is [paper.md](paper.md), with references in
[paper.bib](paper.bib). Reviewer navigation notes are in
[docs/joss-readiness.md](docs/joss-readiness.md).

The branch version is `2.2.0`. The PyPI release should be created only after
the JOSS/readiness branch is merged and the release workflow is confirmed.

## Citation

If you use this package, cite [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
