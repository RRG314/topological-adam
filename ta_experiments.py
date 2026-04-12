"""Run the supported Topological Adam diagnostics comparison."""

from topological_adam import ExperimentConfig, print_comparison_report, run_comparison


if __name__ == "__main__":
    config = ExperimentConfig(dataset_mode="memorization")
    results = run_comparison(config=config)
    print_comparison_report(results)
