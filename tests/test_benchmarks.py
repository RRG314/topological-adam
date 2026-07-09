from __future__ import annotations

from examples.reference_training_benchmark import run_benchmark
from topological_adam.benchmarks import run_candidate_benchmarks


def test_candidate_benchmarks_include_baseline_and_variants() -> None:
    payload = run_candidate_benchmarks(seeds=(0,))
    optimizer_names = {row["optimizer"] for row in payload["results"]}
    assert {"Adam", "TopologicalAdamV2", "TopologicalAdamSDS"} <= optimizer_names


def test_reference_training_benchmark_smoke() -> None:
    payload = run_benchmark(
        optimizer_names=("Adam", "V3"),
        lr_grid=(1e-3,),
        tune_seeds=(0,),
        fresh_seeds=(5,),
        epochs=1,
        batch_size=256,
    )
    assert payload["benchmark"] == "sklearn_digits_mlp"
    assert set(payload["optimizers"]) == {"Adam", "V3"}
    assert payload["optimizers"]["Adam"]["best_lr"] == 1e-3
    assert payload["optimizers"]["V3"]["fresh"]["n"] == 1
