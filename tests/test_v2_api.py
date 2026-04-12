from __future__ import annotations

import torch
import torch.nn as nn

from topological_adam import TopologicalAdam, TopologicalAdamV2


def test_version_exports_are_distinct_classes() -> None:
    assert TopologicalAdam is not TopologicalAdamV2


def test_v2_tracks_expected_stats_keys() -> None:
    model = nn.Linear(10, 4)
    optimizer = TopologicalAdamV2(
        model.parameters(),
        lr=1e-3,
        deterministic_init=True,
        track_stats=True,
    )
    x = torch.randn(8, 10)
    y = model(x).sum()
    y.backward()
    optimizer.step()

    for key in ["energy", "alpha_norm", "beta_norm", "coupling_current", "coupling", "topo_ratio", "num_params"]:
        assert key in optimizer.stats

    metrics = optimizer.field_metrics()
    assert metrics["energy"] >= 0.0
    assert metrics["j_t"] >= 0.0


def test_v2_get_field_stats_returns_finite_values() -> None:
    model = nn.Linear(10, 4)
    optimizer = TopologicalAdamV2(
        model.parameters(),
        lr=1e-3,
        deterministic_init=True,
        track_stats=True,
    )
    x = torch.randn(8, 10)
    y = model(x).sum()
    y.backward()
    optimizer.step()

    energy, j_t, corr = optimizer.get_field_stats()
    assert energy >= 0.0
    assert j_t >= 0.0
    assert -1.0 <= corr <= 1.0
