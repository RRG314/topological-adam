# Topological Adam

[![Release](https://img.shields.io/github/v/release/RRG314/topological-adam?display_name=tag)](https://github.com/RRG314/topological-adam/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/RRG314/topological-adam/blob/main/LICENSE)
[![Tests](https://github.com/RRG314/topological-adam/actions/workflows/tests.yml/badge.svg)](https://github.com/RRG314/topological-adam/actions/workflows/tests.yml)

Topological Adam is a family of Adam-based PyTorch optimizers with two
documented branches:

| Branch | Optimizers | Status |
|---|---|---|
| Field dynamics | `TopologicalAdam`, `TopologicalAdamV2`, `TopologicalAdamV3`, `TopologicalAdamSDS` | V3 is recommended |
| Trajectory topology | `TopologicalAdamV4` | Experimental |

`TopologicalAdamV3` is the default recommendation. It augments Adam with slow
and fast gradient-EMA fields, uses their disagreement as a short-term trend
signal, and gates the correction by alignment with the current gradient. The
gate opens on coherent deterministic or ill-conditioned problems and closes
under minibatch noise. With `w_topo=0, cautious=False`, V3 reduces exactly to
Adam or AdamW.

`TopologicalAdamV4` is experimental. It computes topological invariants of the
optimizer's own recent update trajectory, not the loss landscape: projected
winding numbers and optional exact Vietoris-Rips H1 persistent homology. It is
designed for loop- or oscillation-structured dynamics and is documented in
[docs/trajectory-topology.md](docs/trajectory-topology.md).

```python
from topological_adam import TopologicalAdamV3

optimizer = TopologicalAdamV3(model.parameters(), lr=1e-3)
```

## What this package is

- A PyTorch optimizer package for testing Adam-style mechanisms.
- A reproducible benchmark harness with tuned baselines and fresh-seed checks.
- An honest record of wins, parity results, and losses.

## What this package is not

- It is not a claim about loss-surface topology.
- It is not topological data analysis as a general-purpose library.
- It is not a universal replacement for Adam.

## Repository map

- `topological_adam/`: installable Python package. V3, V4, and persistence
  helpers are exported from `topological_adam.__init__`.
- `examples/`: reproducible benchmark and figure-generation scripts.
- `tests/`: unit, integration, reduction, and benchmark-smoke tests.
- `docs/`: reviewer-facing method notes, benchmark summaries, and figures.
- `paper.md` and `paper.bib`: JOSS paper source and references.

The current GitHub/JOSS branch is versioned as 2.3.0. The PyPI package may
lag until a release is uploaded after this branch is merged.

## Evidence

Full V3 methodology and results are in [docs/v3-audit.md](docs/v3-audit.md),
`benchmark_v3_results.json`, and `fresh_seed_confirmation.json`. V4 results are
in `benchmark_v4_results.json`.

Summary:

- V3 reaches the measurement floor on the included ill-conditioned quadratic
  where tuned Adam stalls near 8e-5.
- V3 improves the included noisy teacher-student regression on 9 of 10 fresh
  seeds.
- V3 is generally at parity with tuned Adam on ordinary noisy classification
  tasks.
- V4 wins on the synthetic rotating-field task it targets, is parity on digits
  MLP, and loses on noisy teacher-student regression.

The benchmark suite labels synthetic and real-data tasks separately.

## Installation

For JOSS review or development, install directly from the repository:

```bash
git clone https://github.com/RRG314/topological-adam.git
cd topological-adam
pip install -e .
```

Development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Usage

```python
import torch.nn as nn
from topological_adam import TopologicalAdamV3

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
optimizer = TopologicalAdamV3(model.parameters(), lr=1e-3)
```

Inspect diagnostics:

```python
optimizer = TopologicalAdamV3(model.parameters(), lr=1e-3, track_stats=True)
# after optimizer.step()
stats = optimizer.field_metrics()
print(stats["energy"], stats["j_t"], stats["gate"], stats["align_cos"])
```

Run the experimental trajectory-topology optimizer:

```python
from topological_adam import TopologicalAdamV4

optimizer = TopologicalAdamV4(model.parameters(), lr=1e-3)
```

## Reproducing results

```bash
python examples/benchmark_v3_suite.py
python examples/confirm_fresh_seeds.py --results benchmark_v3_results.json
python examples/benchmark_v4_suite.py
python examples/make_topology_figures.py
```

## Tests

```bash
python -m pytest tests/ -q
```

The tests cover exact Adam/AdamW reduction, convergence-floor regressions,
state-dict compatibility, V3 field gating, V4 winding and persistence
computations, trajectory gating behavior, and legacy paths.

## Citation

If you use this package, cite [CITATION.cff](CITATION.cff). The JOSS paper
source is [paper.md](paper.md).

## License

MIT. See [LICENSE](LICENSE).
