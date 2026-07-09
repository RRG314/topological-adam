"""Topological Adam optimizer family."""

from .analysis import ExperimentConfig, print_comparison_report, run_comparison, run_instrumented_training
from .optimizer import TopologicalAdam, TopologicalAdamV2
from .persistence import h0_persistence, max_loop_score, rips_h1_persistence
from .sds import TopologicalAdamSDS
from .stopping import ReconnectionStoppingRule, StopDecision
from .v3 import TopologicalAdamV3
from .v4 import TopologicalAdamV4

__version__ = "2.3.0"

__all__ = [
    "ExperimentConfig",
    "ReconnectionStoppingRule",
    "StopDecision",
    "TopologicalAdam",
    "TopologicalAdamV2",
    "TopologicalAdamSDS",
    "TopologicalAdamV3",
    "TopologicalAdamV4",
    "__version__",
    "h0_persistence",
    "max_loop_score",
    "print_comparison_report",
    "rips_h1_persistence",
    "run_comparison",
    "run_instrumented_training",
]
