from __future__ import annotations

import torch
import torch.nn as nn

from topological_adam import TopologicalAdamSDS
from topological_adam.shared import two_temperature_efficiency


def test_two_temperature_efficiency_is_bounded() -> None:
    alpha = torch.tensor([1.0, 2.0, 3.0])
    beta = torch.tensor([1.0, 1.0, 1.0])
    thermo = two_temperature_efficiency(alpha, beta)
    assert thermo["T_hot"] >= thermo["T_cold"] >= 0.0
    assert 0.0 <= thermo["eta_c"] <= 1.0


def test_sds_candidate_tracks_temperature_stats() -> None:
    model = nn.Linear(8, 4)
    optimizer = TopologicalAdamSDS(model.parameters(), lr=1e-3, deterministic_init=True)
    x = torch.randn(16, 8)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    for key in ["T_hot", "T_cold", "eta_c", "energy", "topo_ratio"]:
        assert key in optimizer.stats
        assert optimizer.stats[key] >= 0.0


def test_sds_candidate_reduces_quadratic_loss() -> None:
    param = nn.Parameter(torch.tensor([0.0, 0.0]))
    optimizer = TopologicalAdamSDS([param], lr=0.05, deterministic_init=True)
    initial = None
    final = None
    for _ in range(200):
        optimizer.zero_grad()
        loss = (param[0] - 3.0) ** 2 + (param[1] + 2.0) ** 2
        if initial is None:
            initial = float(loss.item())
        loss.backward()
        optimizer.step()
        final = float(loss.item())
    assert initial is not None and final is not None
    assert final < initial
    assert final < 1e-3
