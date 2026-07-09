# Changelog

## v2.2.0

- JOSS submission preparation: `paper.md`/`paper.bib`, reviewer-focused
  README, `CONTRIBUTING.md`, `CITATION.cff`, CI test workflow, JOSS paper
  build workflow, PyPI release workflow, and reviewer readiness notes.
- Added `TopologicalAdamV3` as the recommended optimizer path: coherence-gated
  gradient-EMA field dynamics, exact Adam/AdamW reduction, tuned-baseline
  audit (`docs/v3-audit.md`), and fresh-seed confirmation.
- Made the package name operational: `TopologicalAdamV4` (experimental) now
  computes genuine topological invariants of the optimizer's own recent
  trajectory and uses them to modulate the update.
  - Multi-plane winding/turning detector: rotation index (winding number) of
    the projected update-direction curve, measured in `n_planes` random 2-D
    projections; O(1)-memory `store_projections=False` mode with a verified
    identical trajectory.
  - New dependency-free `topological_adam/persistence.py`: exact
    Vietoris-Rips H0/H1 persistent homology over Z/2 (boundary-matrix
    reduction), with a scale-free loop score; opt-in gate term via
    `persistence_every`.
  - `trajectory_metrics()` and `trajectory_persistence()` diagnostics, off
    the hot path; exact bit-for-bit Adam reduction with `loop_gate=False`
    retained and tested.
- Added an honest V4 benchmark suite (`examples/benchmark_v4_suite.py`,
  results in `benchmark_v4_results.json`): per-optimizer learning-rate
  tuning, fresh-seed evaluation, paired statistics, every task labeled
  [synthetic] or [real data]; wins and losses both reported.
- Added a reviewer-facing real-data reference training benchmark
  (`examples/reference_training_benchmark.py`, results in
  `reference_training_results.json`) using the sklearn digits dataset and a
  documented tune-then-fresh protocol.
- Repositioned the documentation plainly: the family is a *specialized*
  optimizer collection, not a general Adam replacement; every benchmark in
  the README is labeled real-data or synthetic.
- Removed stale root-level research memos and legacy experiment entry points
  so the repository presents as a Python optimizer package rather than a
  project file dump.
- Rewrote `docs/trajectory-topology.md` around the explicit criteria for the
  name "Topological Adam" being honest, and updated `paper.md`/`paper.bib`
  (Whitney 1937; Edelsbrunner-Letscher-Zomorodian 2002;
  Zomorodian-Carlsson 2005).
- Packaging cleanup for PyPI: `pyproject.toml` is now the single source of
  truth (metadata, version, URLs, classifiers); `setup.py` reduced to a
  shim; added `__version__` and persistence exports to the package root; the
  source distribution now includes reviewer-facing docs, examples, paper
  files, tests, and stored JSON outputs.
- Test suite grown to 125 tests covering the persistence module and the V4
  topology integration.

## v2.1.0

- Separated the optimizer family into clear `v1`, `v2`, and experimental `sds` paths.
- Kept `TopologicalAdamV2` as the supported default and preserved `TopologicalAdam` for provenance.
- Promoted the reconnection-style stopping logic into supported code and documented it as a practical heuristic.
- Added reusable diagnostics and benchmark helpers instead of keeping that logic buried in one-off scripts.
- Added a benchmarked SDS-inspired candidate branch without promoting it beyond its evidence.
- Rewrote the README, overview, results, and release notes so the repository reads as a real optimizer project rather than a loose experiment dump.

## Unreleased

- Split the package internals into explicit `v1`, `v2`, `analysis`, and `stopping` modules.
- Kept `topological_adam.optimizer` as a compatibility shim so older imports still work.
- Promoted the reconnection-style stopping logic into supported code instead of leaving it inside an experiment script.
- Replaced the monkey-patched diagnostics experiment with reusable analysis functions.
- Rewrote the README and docs so `TopologicalAdamV2` is the default supported path and V1 is clearly legacy.
- Added quickstart and stopping demo examples.
- Corrected the repo narrative to distinguish implementation facts from empirical findings.
- Added an SDS-inspired experimental branch, `TopologicalAdamSDS`, plus a benchmark suite against `Adam` and `TopologicalAdamV2`.
