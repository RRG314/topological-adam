from __future__ import annotations

from topological_adam.benchmarks import run_candidate_benchmarks


def test_candidate_benchmarks_include_baseline_and_variants() -> None:
    payload = run_candidate_benchmarks(seeds=(0,))
    optimizer_names = {row["optimizer"] for row in payload["results"]}
    assert {"Adam", "TopologicalAdamV2", "TopologicalAdamSDS"} <= optimizer_names
