from __future__ import annotations

from topological_adam import ExperimentConfig, run_comparison


def test_instrumented_comparison_returns_expected_structure() -> None:
    results = run_comparison(
        ExperimentConfig(
            epochs=2,
            num_samples=128,
            input_dim=32,
            hidden_dim=32,
            num_classes=4,
            batch_size=64,
            deterministic_init=True,
            dataset_mode="clustered",
        )
    )
    assert set(results) == {"control", "topological"}
    for run in results.values():
        assert len(run["history"]) == 2
        assert "pearson_r_j_loss" in run
        assert "final_loss" in run
        assert "final_j_t" in run
