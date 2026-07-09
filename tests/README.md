# Test Suite

The tests are organized around optimizer behavior, PyTorch compatibility,
legacy-path preservation, and the V3/V4 mechanisms used in the JOSS paper.

## Main groups

- `test_optimizer_basic.py`, `test_optimizer_edge_cases.py`,
  `test_optimizer_integration.py`, and `test_convergence.py` cover the legacy
  optimizer interface, numerical stability, state handling, device behavior,
  and toy convergence checks.
- `test_v2_api.py` preserves the V1/V2 compatibility boundary.
- `test_sds_candidate.py` keeps the SDS experimental branch importable and
  numerically stable.
- `test_v3.py` covers exact Adam/AdamW reduction, convergence-floor
  regressions, coherence-gate behavior, field diagnostics, and state-dict
  round trips.
- `test_v4.py` covers exact Adam/AdamW reduction, projected winding detection,
  exact H0/H1 persistence, live trajectory persistence, persistence-fed
  gating, and state-dict behavior.
- `test_benchmarks.py` and `test_analysis.py` cover lightweight benchmark and
  diagnostics helpers, including a smoke test for the reviewer-facing
  reference training benchmark.

## Running tests

```bash
python -m pytest tests/ -q
```

Optional coverage run:

```bash
python -m pytest tests/ --cov=topological_adam --cov-report=html
```

CUDA-specific tests are skipped automatically when CUDA is unavailable.
