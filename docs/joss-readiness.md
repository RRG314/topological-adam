# JOSS readiness notes

This file maps the repository to the main JOSS review expectations. It is not
part of the paper; it is a navigation aid for reviewers and maintainers.

## Paper

- Source: `paper.md`
- Bibliography: `paper.bib`
- Figure source: `examples/make_topology_figures.py`
- Figure output: `docs/figures/v4_trajectory_topology.png`

The paper includes the required JOSS sections: Summary, Statement of need,
State of the field, Software design, Research impact statement, AI usage
disclosure, Acknowledgements, and References.

## Software package

- Package metadata: `pyproject.toml`
- Import root: `topological_adam/__init__.py`
- Recommended optimizer: `TopologicalAdamV3`
- Experimental trajectory-topology optimizer: `TopologicalAdamV4`

V3 and V4 are part of the installable package and are exported from the
package root.

## Installation and tests

```bash
pip install -e .
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

The GitHub Actions workflow in `.github/workflows/tests.yml` runs the test
suite on Python 3.10 and 3.12.

## Benchmarks

- V3 benchmark suite: `examples/benchmark_v3_suite.py`
- V3 fresh-seed confirmation: `examples/confirm_fresh_seeds.py`
- V4 benchmark suite: `examples/benchmark_v4_suite.py`
- Stored outputs: `benchmark_v3_results.json`, `fresh_seed_confirmation.json`,
  and `benchmark_v4_results.json`

Benchmark claims in the paper are bounded to these scripts and stored outputs.
