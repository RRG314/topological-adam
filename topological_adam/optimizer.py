"""Backward-compatible optimizer exports.

The supported implementation now lives in ``topological_adam.v1`` and
``topological_adam.v2``. This module remains so older imports keep working.
"""

from .v1 import TopologicalAdam
from .v2 import TopologicalAdamV2

__all__ = ["TopologicalAdam", "TopologicalAdamV2"]
