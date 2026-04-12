from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topological_adam.benchmarks import benchmark_report_json


if __name__ == "__main__":
    print(benchmark_report_json())
