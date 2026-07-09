"""Tests for TopologicalAdamV3."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from topological_adam import TopologicalAdamV3


def _quadratic_run(opt_factory, steps=200, lr=0.05, seed=0):
    torch.manual_seed(seed)
    p = nn.Parameter(torch.tensor([0.0, 0.0]))
    opt = opt_factory([p], lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = (p[0] - 3.0) ** 2 + (p[1] + 2.0) ** 2
        loss.backward()
        opt.step()
    return float(loss.detach())


class TestAdamEquivalence:
    def test_reduces_to_adam_when_disabled(self):
        torch.manual_seed(0)
        p1 = nn.Parameter(torch.randn(32))
        p2 = nn.Parameter(p1.detach().clone())
        o1 = torch.optim.Adam([p1], lr=0.01)
        o2 = TopologicalAdamV3([p2], lr=0.01, w_topo=0.0, cautious=False)
        target = torch.arange(32.0)
        for _ in range(100):
            for p, o in ((p1, o1), (p2, o2)):
                o.zero_grad()
                ((p - target) ** 2).sum().backward()
                o.step()
        assert torch.allclose(p1, p2, atol=1e-5)

    def test_reduces_to_adamw_when_disabled(self):
        torch.manual_seed(0)
        p1 = nn.Parameter(torch.randn(16))
        p2 = nn.Parameter(p1.detach().clone())
        o1 = torch.optim.AdamW([p1], lr=0.01, weight_decay=0.1)
        o2 = TopologicalAdamV3([p2], lr=0.01, w_topo=0.0, cautious=False, weight_decay=0.1)
        for _ in range(50):
            for p, o in ((p1, o1), (p2, o2)):
                o.zero_grad()
                (p ** 2).sum().backward()
                o.step()
        assert torch.allclose(p1, p2, atol=1e-5)


class TestConvergence:
    def test_quadratic_no_noise_floor(self):
        """The V2 failure case: V3 must converge deeply, not stall at ~1e-5."""
        loss = _quadratic_run(lambda ps, lr: TopologicalAdamV3(ps, lr=lr), steps=1000)
        assert loss < 1e-10

    def test_quadratic_beats_v2_regime(self):
        loss = _quadratic_run(lambda ps, lr: TopologicalAdamV3(ps, lr=lr), steps=200)
        assert loss < 1e-4  # V2 floors near 1.5e-5 permanently; Adam ~5e-8

    def test_rosenbrock_progress(self):
        torch.manual_seed(0)
        p = nn.Parameter(torch.tensor([-1.2, 1.0]))
        opt = TopologicalAdamV3([p], lr=0.02)
        for _ in range(2000):
            opt.zero_grad()
            loss = (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
            loss.backward()
            opt.step()
        assert float(loss.detach()) < 1e-3

    def test_mlp_trains(self):
        torch.manual_seed(0)
        x = torch.randn(256, 8)
        y = (x.sum(dim=1) > 0).long()
        model = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 2))
        opt = TopologicalAdamV3(model.parameters(), lr=1e-2)
        first = None
        for _ in range(200):
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            if first is None:
                first = float(loss.detach())
        assert float(loss.detach()) < first * 0.2


class TestMechanics:
    def test_fields_anneal_at_convergence(self):
        """No energy floor: field energy must decay as gradients vanish."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.tensor([5.0]))
        opt = TopologicalAdamV3([p], lr=0.1, track_stats=True)
        energies = []
        for _ in range(500):
            opt.zero_grad()
            (p ** 2).sum().backward()
            opt.step()
            energies.append(opt.stats["energy"])
        assert energies[-1] < energies[50] * 1e-3

    def test_skips_grad_none(self):
        p1 = nn.Parameter(torch.randn(4))
        p2 = nn.Parameter(torch.randn(4))
        opt = TopologicalAdamV3([p1, p2], lr=1e-3)
        (p1 ** 2).sum().backward()
        opt.step()  # p2 has no grad; must not raise

    def test_invalid_field_rates_raise(self):
        p = nn.Parameter(torch.randn(2))
        with pytest.raises(ValueError):
            TopologicalAdamV3([p], eta_fast=0.01, eta_slow=0.5)

    def test_invalid_eta_gate_raises(self):
        p = nn.Parameter(torch.randn(2))
        with pytest.raises(ValueError):
            TopologicalAdamV3([p], eta_gate=0.0)
        with pytest.raises(ValueError):
            TopologicalAdamV3([p], eta_gate=1.5)

    def test_state_dict_roundtrip(self):
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(8))
        opt = TopologicalAdamV3([p], lr=1e-3)
        for _ in range(3):
            opt.zero_grad()
            (p ** 2).sum().backward()
            opt.step()
        sd = opt.state_dict()
        p2 = nn.Parameter(p.detach().clone())
        opt2 = TopologicalAdamV3([p2], lr=1e-3)
        opt2.load_state_dict(sd)
        opt2.zero_grad()
        (p2 ** 2).sum().backward()
        opt2.step()  # must not raise

    def test_stats_and_field_metrics(self):
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(16))
        opt = TopologicalAdamV3([p], lr=1e-3, track_stats=True)
        for _ in range(5):
            opt.zero_grad()
            (p ** 2).sum().backward()
            opt.step()
        assert opt.stats["num_params"] == 1.0
        assert math.isfinite(opt.stats["coupling_current"])
        m = opt.field_metrics()
        for key in ("energy", "j_t", "alpha_norm", "beta_norm", "alpha_beta_corr"):
            assert math.isfinite(m[key])
        e, j, c = opt.get_field_stats()
        assert all(math.isfinite(v) for v in (e, j, c))

    def test_gate_stat_tracked_and_bounded(self):
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(16))
        opt = TopologicalAdamV3([p], lr=1e-2, cautious=False, coupling_gate=True,
                                track_stats=True)
        for _ in range(20):
            opt.zero_grad()
            (p ** 2).sum().backward()
            opt.step()
        assert 0.0 <= opt.stats["gate"] <= 1.0
        assert math.isfinite(opt.stats["gate"])

    def test_gate_opens_on_coherent_gradients(self):
        """On a deterministic quadratic the trend is coherent -> gate should open."""
        torch.manual_seed(0)
        p = nn.Parameter(torch.tensor([5.0, -3.0]))
        opt = TopologicalAdamV3([p], lr=1e-2, cautious=False, coupling_gate=True,
                                track_stats=True)
        for _ in range(50):
            opt.zero_grad()
            (p ** 2).sum().backward()
            opt.step()
        assert opt.stats["gate"] > 0.5

    def test_gate_off_path_runs(self):
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(8))
        opt = TopologicalAdamV3([p], lr=1e-2, coupling_gate=False)
        for _ in range(5):
            opt.zero_grad()
            (p ** 2).sum().backward()
            opt.step()
        assert torch.isfinite(p).all()

    def test_standalone_mechanism_deep_convergence(self):
        """The topological mechanism alone (no cautious mask) must have no noise
        floor: deep convergence on the quadratic like Adam."""
        loss = _quadratic_run(
            lambda ps, lr: TopologicalAdamV3(ps, lr=lr, cautious=False,
                                             coupling_gate=True),
            steps=1000,
        )
        assert loss < 1e-10

    def test_state_dict_roundtrip_preserves_gate(self):
        torch.manual_seed(0)
        p = nn.Parameter(torch.randn(8))
        opt = TopologicalAdamV3([p], lr=1e-3, coupling_gate=True)
        for _ in range(5):
            opt.zero_grad()
            (p ** 2).sum().backward()
            opt.step()
        sd = opt.state_dict()
        p2 = nn.Parameter(p.detach().clone())
        opt2 = TopologicalAdamV3([p2], lr=1e-3, coupling_gate=True)
        opt2.load_state_dict(sd)
        st = opt2.state[p2]
        assert "j_ema" in st and torch.isfinite(st["j_ema"])
        opt2.zero_grad()
        (p2 ** 2).sum().backward()
        opt2.step()

    def test_no_finite_issues_with_zero_grad(self):
        p = nn.Parameter(torch.randn(4))
        opt = TopologicalAdamV3([p], lr=1e-3)
        for _ in range(3):
            opt.zero_grad()
            p.grad = torch.zeros_like(p)
            opt.step()
        assert torch.isfinite(p).all()
