"""Tests for TopologicalAdamV4 (trajectory-topology gated Adam)."""

import copy
import math

import pytest
import torch

from topological_adam import TopologicalAdamV4


def _clone_params(seed=0, shape=(8,)):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(shape, generator=g)
    p1 = base.clone().requires_grad_(True)
    p2 = base.clone().requires_grad_(True)
    return p1, p2


class TestAdamReduction:
    def test_gate_off_matches_adam(self):
        p1, p2 = _clone_params()
        o1 = torch.optim.Adam([p1], lr=0.01)
        o2 = TopologicalAdamV4([p2], lr=0.01, loop_gate=False)
        g = torch.Generator().manual_seed(1)
        for _ in range(50):
            grad = torch.randn(p1.shape, generator=g)
            p1.grad = grad.clone()
            p2.grad = grad.clone()
            o1.step()
            o2.step()
        assert torch.allclose(p1, p2, atol=1e-5)

    def test_gate_off_matches_adamw(self):
        p1, p2 = _clone_params()
        o1 = torch.optim.AdamW([p1], lr=0.01, weight_decay=0.1)
        o2 = TopologicalAdamV4([p2], lr=0.01, weight_decay=0.1, loop_gate=False)
        g = torch.Generator().manual_seed(1)
        for _ in range(50):
            grad = torch.randn(p1.shape, generator=g)
            p1.grad = grad.clone()
            p2.grad = grad.clone()
            o1.step()
            o2.step()
        assert torch.allclose(p1, p2, atol=1e-5)

    def test_straight_trajectory_matches_adam_with_gate_on(self):
        """Constant gradient => zero turning => gate stays exactly 1 => Adam."""
        p1, p2 = _clone_params()
        o1 = torch.optim.Adam([p1], lr=0.01)
        o2 = TopologicalAdamV4([p2], lr=0.01, loop_gate=True)
        grad = torch.randn(p1.shape, generator=torch.Generator().manual_seed(2))
        for _ in range(50):
            p1.grad = grad.clone()
            p2.grad = grad.clone()
            o1.step()
            o2.step()
        assert torch.allclose(p1, p2, atol=1e-5)
        m = o2.trajectory_metrics()[0]
        assert m["gate"] == pytest.approx(1.0)
        assert abs(m["kappa_ema"]) < 1e-5
        assert abs(m["winding"]) < 1e-5


class TestWindingDetector:
    def test_rotating_gradient_field_detected(self):
        """Gradients rotating in a 2-D parameter space produce circulation."""
        p = torch.zeros(2, requires_grad=True)
        opt = TopologicalAdamV4([p], lr=1e-3)
        omega = 0.3  # radians per step
        for t in range(300):
            angle = omega * t
            p.grad = torch.tensor([math.cos(angle), math.sin(angle)])
            opt.step()
        m = opt.trajectory_metrics()[0]
        # Persistent one-directional turning: circulation and curvature agree
        # in magnitude, winding over the window is clearly nonzero, and the
        # gate has closed below 1.
        assert m["kappa_ema"] > 0.05
        assert abs(m["theta_ema"]) > 0.5 * m["kappa_ema"]
        assert abs(m["winding"]) > 0.5
        assert m["gate"] < 1.0

    def test_oscillating_gradient_closes_gate(self):
        """Sign-flipping gradients (pi turning per step) drive gate to floor."""
        p = torch.zeros(8, requires_grad=True)
        opt = TopologicalAdamV4([p], lr=1e-3, min_gate=0.1)
        grad = torch.randn(8, generator=torch.Generator().manual_seed(3))
        for t in range(120):
            p.grad = grad if t % 2 == 0 else -grad
            opt.step()
        m = opt.trajectory_metrics()[0]
        assert m["kappa_ema"] > 1.5  # approaching pi
        assert m["gate"] < 0.5
        # Oscillation is not circulation: signed turning stays small
        # relative to total curvature.
        assert abs(m["theta_ema"]) < 0.5 * m["kappa_ema"]

    def test_winding_number_of_full_loops(self):
        """~k full rotations within the window give winding close to k."""
        p = torch.zeros(2, requires_grad=True)
        opt = TopologicalAdamV4([p], lr=1e-3, window=128, betas=(0.5, 0.999))
        omega = 2.0 * math.pi / 32  # one full turn every 32 steps
        for t in range(128 + 64):
            angle = omega * t
            p.grad = torch.tensor([math.cos(angle), math.sin(angle)])
            opt.step()
        m = opt.trajectory_metrics()[0]
        # 128-step window at 32 steps/turn => ~4 turns (sign depends on the
        # random projection's orientation).
        assert 2.0 < abs(m["winding"]) < 6.0


class TestBehavior:
    def test_converges_on_quadratic(self):
        torch.manual_seed(0)
        p = torch.randn(10, requires_grad=True)
        target = torch.randn(10)
        opt = TopologicalAdamV4([p], lr=0.05)
        for _ in range(500):
            opt.zero_grad(set_to_none=True)
            loss = ((p - target) ** 2).sum()
            loss.backward()
            opt.step()
        assert ((p - target) ** 2).sum().item() < 1e-6

    def test_gating_helps_on_oscillatory_quadratic(self):
        """On a stiff quadratic driven into oscillation, the gate should not
        hurt, and typically helps, relative to gate-off (plain Adam)."""

        def run(loop_gate):
            torch.manual_seed(0)
            p = (torch.ones(2) * 3.0).requires_grad_(True)
            scales = torch.tensor([1.0, 100.0])
            opt = TopologicalAdamV4([p], lr=0.3, loop_gate=loop_gate)
            for _ in range(200):
                opt.zero_grad(set_to_none=True)
                loss = (scales * p ** 2).sum()
                loss.backward()
                opt.step()
            return (scales * p ** 2).sum().item()

        gated, plain = run(True), run(False)
        assert gated <= plain * 1.5  # never catastrophically worse
        assert gated < 1e-2  # and actually converges

    def test_zero_grad_steps_are_safe(self):
        p = torch.randn(4, requires_grad=True)
        opt = TopologicalAdamV4([p], lr=0.01)
        for _ in range(5):
            p.grad = torch.zeros(4)
            opt.step()
        assert torch.isfinite(p).all()
        m = opt.trajectory_metrics()[0]
        assert math.isfinite(m["gate"]) and m["gate"] == pytest.approx(1.0)

    def test_state_dict_roundtrip(self):
        p1, p2 = _clone_params(seed=4)
        o1 = TopologicalAdamV4([p1], lr=0.01)
        g = torch.Generator().manual_seed(5)
        for _ in range(10):
            p1.grad = torch.randn(p1.shape, generator=g)
            o1.step()
        o2 = TopologicalAdamV4([p2], lr=0.01)
        with torch.no_grad():
            p2.copy_(p1)
        # deepcopy: torch's load_state_dict may alias tensors from a live
        # state_dict when dtype/device already match, which would let
        # o1.step() mutate o2's state.
        o2.load_state_dict(copy.deepcopy(o1.state_dict()))
        grad = torch.randn(p1.shape, generator=g)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        o1.step()
        o2.step()
        assert torch.allclose(p1, p2, atol=1e-6)

    def test_invalid_args_raise(self):
        p = torch.randn(2, requires_grad=True)
        with pytest.raises(ValueError):
            TopologicalAdamV4([p], lr=-1.0)
        with pytest.raises(ValueError):
            TopologicalAdamV4([p], min_gate=0.0)
        with pytest.raises(ValueError):
            TopologicalAdamV4([p], rho=0.0)
        with pytest.raises(ValueError):
            TopologicalAdamV4([p], window=1)


class TestPersistence:
    """Tests for the exact Vietoris-Rips H1 persistence module."""

    def test_circle_has_prominent_loop(self):
        from topological_adam.persistence import max_loop_score, rips_h1_persistence

        pts = [
            [math.cos(2 * math.pi * k / 32), math.sin(2 * math.pi * k / 32)]
            for k in range(32)
        ]
        bars = rips_h1_persistence(pts)
        assert len(bars) >= 1
        # VR H1 of a circle of radius r dies at edge length sqrt(3)*r.
        birth, death = bars[0]
        assert death == pytest.approx(math.sqrt(3.0), rel=0.05)
        assert max_loop_score(pts) > 0.6

    def test_noise_and_line_score_low(self):
        from topological_adam.persistence import max_loop_score

        g = torch.Generator().manual_seed(0)
        noise = torch.randn(32, 2, generator=g)
        assert max_loop_score(noise) < 0.3
        line = torch.stack([torch.arange(32.0) * 0.1, torch.arange(32.0) * 0.2], 1)
        assert max_loop_score(line) == 0.0

    def test_h0_bar_count(self):
        from topological_adam.persistence import h0_persistence

        pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        bars = h0_persistence(pts)
        # n points => n H0 bars, exactly one infinite.
        assert len(bars) == 3
        assert sum(1 for _, d in bars if d == math.inf) == 1

    def test_degenerate_inputs(self):
        from topological_adam.persistence import max_loop_score, rips_h1_persistence

        assert rips_h1_persistence([[0.0, 0.0]]) == []
        assert max_loop_score([[1.0, 2.0], [1.0, 2.0]]) == 0.0
        # All-identical points: diameter 0 must not divide by zero.
        assert max_loop_score([[0.0, 0.0]] * 8) == 0.0


class TestTopologyIntegration:
    """The invariants are computed from and act on the real trajectory."""

    def _run_rotating(self, opt, steps=192, omega=2 * math.pi / 32):
        p = opt.param_groups[0]["params"][0]
        for t in range(steps):
            angle = omega * t
            p.grad = torch.tensor([math.cos(angle), math.sin(angle)])
            opt.step()

    def test_multi_plane_windings_consistent(self):
        """Every plane of a rotating 2-D field sees the same |winding|."""
        p = torch.zeros(2, requires_grad=True)
        opt = TopologicalAdamV4([p], lr=1e-3, window=128, n_planes=3,
                                betas=(0.5, 0.999))
        self._run_rotating(opt)
        m = opt.trajectory_metrics()[0]
        assert len(m["windings"]) == 3
        for w in m["windings"]:
            # 4 full turns fit in the 128-step window (sign is
            # projection-dependent, magnitude is not).
            assert 3.0 < abs(w) < 5.0

    def test_trajectory_persistence_detects_loop(self):
        p = torch.zeros(2, requires_grad=True)
        opt = TopologicalAdamV4([p], lr=1e-3, window=64, n_planes=2,
                                betas=(0.5, 0.999))
        self._run_rotating(opt, steps=96)
        rep = opt.trajectory_persistence()[0]
        assert rep["p_loop"] > 0.5  # a genuine, persistent H1 loop
        assert len(rep["loop_scores"]) == 2

    def test_trajectory_persistence_trivial_on_straight_descent(self):
        p = torch.ones(4, requires_grad=True)
        opt = TopologicalAdamV4([p], lr=1e-3, window=64)
        grad = torch.tensor([1.0, -0.5, 0.25, 2.0])
        for _ in range(96):
            p.grad = grad.clone()
            opt.step()
        rep = opt.trajectory_persistence()[0]
        assert rep["p_loop"] < 0.2

    def test_persistence_gate_closes_on_loops(self):
        """With persistence_every on, the H1 score feeds the gate."""
        p = torch.zeros(2, requires_grad=True)
        opt = TopologicalAdamV4([p], lr=1e-3, window=48, n_planes=2,
                                betas=(0.5, 0.999), persistence_every=16,
                                persistence_max_points=48)
        self._run_rotating(opt, steps=96)
        m = opt.trajectory_metrics()[0]
        assert m["p_loop"] > 0.5
        # Gate must reflect the persistent-loop term, not curvature alone.
        assert m["gate"] < 1.0 - 0.9 * m["p_loop"] + 1e-6

    def test_store_projections_false_matches_true(self):
        """Regenerated projections give the identical trajectory."""
        p1, p2 = _clone_params(seed=7)
        o1 = TopologicalAdamV4([p1], lr=0.01, store_projections=True)
        o2 = TopologicalAdamV4([p2], lr=0.01, store_projections=False)
        g = torch.Generator().manual_seed(3)
        for _ in range(40):
            grad = torch.randn(p1.shape, generator=g)
            p1.grad = grad.clone()
            p2.grad = grad.clone()
            o1.step()
            o2.step()
        assert torch.equal(p1, p2)

    def test_n_planes_one_still_works(self):
        p = torch.zeros(2, requires_grad=True)
        opt = TopologicalAdamV4([p], lr=1e-3, n_planes=1, betas=(0.5, 0.999))
        self._run_rotating(opt, steps=64)
        m = opt.trajectory_metrics()[0]
        assert m["kappa_ema"] > 0.05
        assert len(m["windings"]) == 1
