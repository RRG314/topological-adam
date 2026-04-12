# Changelog

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
