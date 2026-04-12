from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topological_adam import ExperimentConfig, print_comparison_report, run_comparison


if __name__ == "__main__":
    results = run_comparison(ExperimentConfig(dataset_mode="memorization", deterministic_init=True))
    print_comparison_report(results)
