# Topological Adam

[![Release](https://img.shields.io/github/v/release/RRG314/topological-adam?display_name=tag)](https://github.com/RRG314/topological-adam/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/RRG314/topological-adam/blob/main/LICENSE)
[![Status](https://img.shields.io/badge/Status-Supported%20V2%20%7C%20Experimental%20SDS-0f8f82)](https://github.com/RRG314/topological-adam/blob/main/docs/results.md)
[![Docs](https://img.shields.io/badge/Docs-Overview%20%7C%20Results-1f6feb)](https://github.com/RRG314/topological-adam/blob/main/docs/overview.md)

An experimental optimizer repository centered on two things:

1. a PyTorch optimizer family built around auxiliary field dynamics
2. a small, reproducible diagnostics workflow for tracking energy regulation and the internal coupling signal `J_t`

The repository's recommended path is **TopologicalAdamV2**. The original optimizer remains available as **TopologicalAdam** for comparison and provenance, and an SDS-inspired candidate branch is available as **TopologicalAdamSDS** for controlled experimental use.

## Current Release

- Release notes: [docs/releases/v2.1.0.md](docs/releases/v2.1.0.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Recommended starting point: `python examples/quickstart_v2.py`

## What This Repository Is

Topological Adam extends Adam with two auxiliary tensors, `alpha` and `beta`, plus a bounded correction term added to the update direction. The design is structurally informed by ideas that also appear in the sibling MHD closure repository, but this code is **not** a plasma simulator and it should not be read as a physical model.

What the repository is for:
- testing whether the auxiliary-field formulation is numerically useful
- tracking internal optimizer diagnostics such as field energy and `J_t`
- comparing a legacy implementation against the newer supported path

What the repository is not for:
- making physical claims about magnetized fluids
- claiming theorem-level optimizer guarantees that are not yet proved
- presenting the current experiments as production benchmark evidence

## Best User Path

Start here if you want the supported version:

```bash
pip install -e .
python examples/quickstart_v2.py
```

Run the main diagnostics comparison:

```bash
python ta_experiments.py
```

This comparison runs:
- a control path with `eta=0` and `w_topo=0`
- the recommended V2 path with active topological correction

## Versions

### Recommended: `TopologicalAdamV2`

Use V2 if you want:
- the supported default path
- structured field diagnostics
- deterministic initialization for reproducible demos
- the reconnection-style stopping helper

### Legacy: `TopologicalAdam`

Use V1 only if you need:
- the original implementation for comparison
- continuity with older experiments or package behavior

V1 is kept intentionally. It is not the default path for new work.

### Experimental candidate: `TopologicalAdamSDS`

Use the SDS branch only if you want to test the new two-temperature efficiency gate.

Current honest status:
- it is stable on the included benchmark suite
- it does not yet show a clear enough advantage over V2 to become the default
- it is kept because it is a real, testable branch rather than a speculative note

## Quickstart

```python
import torch.nn as nn
from topological_adam import TopologicalAdamV2

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

optimizer = TopologicalAdamV2(
    model.parameters(),
    lr=1e-3,
    eta=0.01,
    w_topo=0.01,
    target_energy=1.0,
    deterministic_init=True,
    track_stats=True,
)
```

After `optimizer.step()`, inspect the internal diagnostics:

```python
stats = optimizer.field_metrics()
print(stats["energy"], stats["j_t"], stats["alpha_beta_corr"])
```

## What `J_t` Means Here

`J_t` is an internal coupling diagnostic computed from the auxiliary fields and the current gradient. In this repository it is treated as a **numerical signal**, not a physical current.

Current honest status:
- `J_t` often decreases as training loss decreases in the supported synthetic diagnostics run
- `J_t` can be used to build a practical stopping heuristic
- the current control run also shows strong `J_t`/loss correlation, so this signal is **not yet evidence of a uniquely topological mechanism**

## Exact vs Empirical

Exact or implementation-level facts:
- Adam moment updates are implemented directly
- the auxiliary-field update equations are explicit in code
- target-energy rescaling is enforced directly by the optimizer implementation

Empirical findings in this repo:
- V2 can train simple models stably on the included synthetic diagnostics setup
- the field-energy target is maintained very tightly in the included experiments
- `J_t` often behaves like a useful convergence indicator on the tested runs

Still open or only partially supported:
- whether the topological correction improves optimization in a reliable benchmark sense
- when `J_t` is genuinely informative beyond gradient magnitude effects
- whether the stopping signal transfers beyond the current synthetic experiments

## Repository Map

- [docs/overview.md](docs/overview.md): where to start and which path is supported
- [docs/results.md](docs/results.md): current empirical picture, including what did and did not hold up
- [docs/mhd-connection.md](docs/mhd-connection.md): how this repo relates to the MHD closure repo
- `topological_adam/v1.py`: legacy implementation
- `topological_adam/v2.py`: recommended implementation
- `topological_adam/sds.py`: experimental SDS-inspired branch
- `topological_adam/stopping.py`: reconnection-style stopping heuristic
- `topological_adam/analysis.py`: reusable diagnostics workflow used by `ta_experiments.py`
- `topological_adam/benchmarks.py`: small candidate benchmark suite
- `examples/quickstart_v2.py`: shortest supported example
- `examples/reconnection_stopping_demo.py`: comparison demo with stopping logic
- `DISCOVERIES.md`: original April 2026 research memo, kept for provenance
- [docs/sds-candidate.md](docs/sds-candidate.md): current status of the SDS branch

## Relationship to the MHD Repository

The sibling repository [RRG314/MagnetoHydroDynamic-research](https://github.com/RRG314/MagnetoHydroDynamic-research) is the theory and closure research layer. This repository is the applied optimizer branch.

- MHD repo: mathematical and symbolic closure work
- Topological Adam repo: optimizer implementation and empirical diagnostics

## Status

- Supported path: `TopologicalAdamV2`
- Legacy path: `TopologicalAdam`
- Experimental path: `TopologicalAdamSDS`
- Diagnostics path: available and reproducible
- Stopping rule: implemented as a heuristic utility, not a theorem-backed guarantee
- Benchmark status: exploratory, not production-ready

## Documentation

- [CHANGELOG.md](CHANGELOG.md)
- [ROADMAP.md](ROADMAP.md)
- [docs/overview.md](docs/overview.md)
- [docs/results.md](docs/results.md)
- [docs/mhd-connection.md](docs/mhd-connection.md)
- [TOPOLOGICAL_ADAM_FINAL_REPORT.md](TOPOLOGICAL_ADAM_FINAL_REPORT.md)

## License

MIT. See [LICENSE](LICENSE).
