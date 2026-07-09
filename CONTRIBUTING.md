# Contributing to Topological Adam

Thanks for your interest in improving this project. Contributions of all kinds
are welcome: bug reports, benchmark results, documentation fixes, theoretical
analysis, and code.

## Getting set up

```bash
git clone https://github.com/RRG314/topological-adam.git
cd topological-adam
pip install -e .
pip install -r requirements-dev.txt
```

## Running the tests

```bash
python -m pytest tests/ -q
```

All tests must pass before a pull request can be merged. If you add or change
optimizer behavior, add tests for it. The V3 tests cover exact reduction,
convergence-floor regression, gate behavior, and state-dict compatibility.

## Running the benchmarks

```bash
python examples/benchmark_v3_suite.py --fast
python examples/benchmark_v3_suite.py
python examples/confirm_fresh_seeds.py --results benchmark_v3_results.json
python examples/benchmark_v4_suite.py
python examples/make_topology_figures.py
```

## Benchmark ground rules

Benchmark claims in this repository follow a fixed protocol:

1. Every optimizer being compared, including baselines, gets its learning rate
   tuned over the same grid.
2. Use held-out metrics where the task has data.
3. Confirm stochastic-task conclusions on seeds not used for tuning.
4. Report paired per-seed statistics, not only means.
5. Report parity and losses as plainly as wins.

## Reporting bugs and asking questions

Open a [GitHub Issue](https://github.com/RRG314/topological-adam/issues) with
your Python and PyTorch versions, a minimal reproduction script, and the
observed versus expected behavior. Questions and discussion are welcome in
issues as well.

## Pull requests

- Keep changes focused.
- Preserve the exact Adam/AdamW reduction for V3
  (`w_topo=0, cautious=False`) and V4 (`loop_gate=False`).
- Preserve legacy implementations unless a change is explicitly about legacy
  behavior.
- New hyperparameters need validation, docstring coverage, and tests.

## Current priorities

- Benchmark results on new tasks and hardware.
- GPU wall-clock validation.
- A fused or `foreach` implementation path.
- Convergence analysis of the gated update.
- Additional independent benchmarks for V3 and V4.

## Code of conduct

Be respectful and constructive. Critiques of methods and evidence are welcome;
personal attacks are not.
