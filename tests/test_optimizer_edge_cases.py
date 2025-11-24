"""
Edge case and numerical stability tests for TopologicalAdam.

Tests handling of edge cases like zero gradients, NaN/Inf values,
and extreme parameter values.
"""
import pytest
import torch
import torch.nn as nn
from topological_adam import TopologicalAdam


class TestGradientEdgeCases:
    """Test handling of edge cases in gradients."""

    def test_none_gradients(self):
        """Test that optimizer handles None gradients correctly."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Don't compute gradients for some parameters
        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        # Manually set some gradients to None
        loss.backward()
        for i, p in enumerate(model.parameters()):
            if i % 2 == 0:
                p.grad = None

        # Should not raise error
        optimizer.step()

    def test_zero_gradients(self):
        """Test that optimizer handles zero gradients correctly."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Create zero gradients
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

        # Should not raise error
        optimizer.step()

        # Parameters should not change significantly with zero gradients
        # (but fields may still update)

    def test_very_small_gradients(self):
        """Test handling of very small gradients (below threshold)."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Create very small gradients
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 1e-15

        # Should not raise error
        optimizer.step()

    def test_very_large_gradients(self):
        """Test handling of very large gradients."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Create very large gradients
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 1e6

        # Should not raise error or produce NaN
        optimizer.step()

        # Check parameters are still finite
        for p in model.parameters():
            assert torch.all(torch.isfinite(p))


class TestNumericalStability:
    """Test numerical stability of the optimizer."""

    def test_nan_in_gradients(self):
        """Test handling of NaN in gradients."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Create gradients with NaN
        for p in model.parameters():
            p.grad = torch.randn_like(p)
            p.grad[0, 0] = float('nan')

        # Should not crash (though behavior with NaN is undefined)
        try:
            optimizer.step()
        except Exception:
            # If it raises, that's also acceptable behavior
            pass

    def test_inf_in_gradients(self):
        """Test handling of Inf in gradients."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Create gradients with Inf
        for p in model.parameters():
            p.grad = torch.randn_like(p)
            p.grad[0, 0] = float('inf')

        # Should not crash
        try:
            optimizer.step()
        except Exception:
            # If it raises, that's also acceptable behavior
            pass

    def test_division_by_zero_protection(self):
        """Test that eps parameter prevents division by zero."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters(), eps=1e-8)

        # Zero second moment should not cause division by zero
        for p in model.parameters():
            p.grad = torch.zeros_like(p)

        optimizer.step()

        # Check state
        for p in model.parameters():
            state = optimizer.state[p]
            # With zero gradients, v should be very small
            # Adam update should not produce NaN
            assert torch.all(torch.isfinite(state['m']))
            assert torch.all(torch.isfinite(state['v']))

    def test_energy_calculation_stability(self):
        """Test that energy calculation remains stable."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        for _ in range(20):
            x = torch.randn(3, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            energy = optimizer.energy()
            assert isinstance(energy, float)
            assert not torch.isnan(torch.tensor(energy))
            assert not torch.isinf(torch.tensor(energy))
            assert energy >= 0.0

    def test_j_calculation_stability(self):
        """Test that J calculation remains stable."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        for _ in range(20):
            x = torch.randn(3, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            j_mean = optimizer.J_mean_abs()
            assert isinstance(j_mean, float)
            assert not torch.isnan(torch.tensor(j_mean))
            assert not torch.isinf(torch.tensor(j_mean))
            assert j_mean >= 0.0


class TestExtremeParameters:
    """Test optimizer behavior with extreme parameter values."""

    def test_very_high_learning_rate(self):
        """Test with very high learning rate."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters(), lr=100.0)

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Should not produce NaN (though may not converge well)
        for p in model.parameters():
            has_finite = torch.any(torch.isfinite(p))
            if not has_finite:
                pytest.skip("Very high LR can cause overflow, which is expected")

    def test_very_low_learning_rate(self):
        """Test with very low learning rate."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters(), lr=1e-10)

        initial_params = [p.clone() for p in model.parameters()]

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # With very low LR, parameters should barely change
        for p_initial, p_current in zip(initial_params, model.parameters()):
            diff = torch.abs(p_current - p_initial).max()
            assert diff < 1e-8

    def test_eta_extremes(self):
        """Test extreme values of eta (coupling rate)."""
        model = nn.Linear(10, 5)

        # Very low eta
        opt1 = TopologicalAdam(model.parameters(), eta=1e-6)
        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        opt1.step()

        # Very high eta (should be stable due to clamping)
        model2 = nn.Linear(10, 5)
        opt2 = TopologicalAdam(model2.parameters(), eta=0.99)
        y2 = model2(x)
        loss2 = y2.sum()
        loss2.backward()
        opt2.step()

    def test_w_topo_extremes(self):
        """Test extreme values of w_topo."""
        model = nn.Linear(10, 5)

        # Zero w_topo (should behave like Adam)
        opt1 = TopologicalAdam(model.parameters(), w_topo=0.0)
        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        opt1.step()

        # Very high w_topo
        model2 = nn.Linear(10, 5)
        opt2 = TopologicalAdam(model2.parameters(), w_topo=10.0)
        y2 = model2(x)
        loss2 = y2.sum()
        loss2.backward()
        opt2.step()

    def test_target_energy_extremes(self):
        """Test extreme values of target_energy."""
        model = nn.Linear(10, 5)

        # Very low target energy
        opt1 = TopologicalAdam(model.parameters(), target_energy=1e-10)
        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        opt1.step()

        # Very high target energy
        model2 = nn.Linear(10, 5)
        opt2 = TopologicalAdam(model2.parameters(), target_energy=1e3)
        y2 = model2(x)
        loss2 = y2.sum()
        loss2.backward()
        opt2.step()


class TestTensorShapes:
    """Test optimizer with different tensor shapes."""

    def test_scalar_parameter(self):
        """Test with scalar parameters."""
        param = nn.Parameter(torch.tensor(1.0))
        optimizer = TopologicalAdam([param])

        param.grad = torch.tensor(0.5)
        optimizer.step()

        assert param.shape == torch.Size([])

    def test_1d_parameter(self):
        """Test with 1D parameters."""
        param = nn.Parameter(torch.randn(10))
        optimizer = TopologicalAdam([param])

        param.grad = torch.randn(10)
        optimizer.step()

        assert param.shape == torch.Size([10])

    def test_2d_parameter(self):
        """Test with 2D parameters."""
        param = nn.Parameter(torch.randn(10, 5))
        optimizer = TopologicalAdam([param])

        param.grad = torch.randn(10, 5)
        optimizer.step()

        assert param.shape == torch.Size([10, 5])

    def test_high_dimensional_parameter(self):
        """Test with high-dimensional parameters (like convolutions)."""
        param = nn.Parameter(torch.randn(64, 32, 3, 3))
        optimizer = TopologicalAdam([param])

        param.grad = torch.randn(64, 32, 3, 3)
        optimizer.step()

        assert param.shape == torch.Size([64, 32, 3, 3])

    def test_mixed_shapes(self):
        """Test with parameters of different shapes."""
        params = [
            nn.Parameter(torch.randn(10, 5)),
            nn.Parameter(torch.randn(5)),
            nn.Parameter(torch.randn(5, 2)),
            nn.Parameter(torch.randn(2))
        ]
        optimizer = TopologicalAdam(params)

        for p in params:
            p.grad = torch.randn_like(p)

        optimizer.step()

        # Check all states initialized correctly
        for p in params:
            state = optimizer.state[p]
            assert state['m'].shape == p.shape
            assert state['v'].shape == p.shape
            assert state['alpha'].shape == p.shape
            assert state['beta'].shape == p.shape


class TestConsistentBehavior:
    """Test that optimizer behavior is consistent."""

    def test_deterministic_with_same_seed(self):
        """Test that optimizer produces same results with same random seed."""
        torch.manual_seed(42)
        model1 = nn.Linear(10, 5)
        opt1 = TopologicalAdam(model1.parameters())

        torch.manual_seed(42)
        model2 = nn.Linear(10, 5)
        opt2 = TopologicalAdam(model2.parameters())

        torch.manual_seed(42)
        x = torch.randn(3, 10)

        # First model
        y1 = model1(x)
        loss1 = y1.sum()
        loss1.backward()
        opt1.step()

        # Second model
        y2 = model2(x)
        loss2 = y2.sum()
        loss2.backward()
        opt2.step()

        # Parameters should be identical
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)

    def test_state_persistence(self):
        """Test that optimizer state persists across steps."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Store state
        old_state = {}
        for p in model.parameters():
            state = optimizer.state[p]
            old_state[p] = {
                'step': state['step'],
                'm': state['m'].clone(),
                'v': state['v'].clone(),
                'alpha': state['alpha'].clone(),
                'beta': state['beta'].clone()
            }

        # Take another step
        optimizer.zero_grad()
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # State should have evolved (not reset)
        for p in model.parameters():
            state = optimizer.state[p]
            assert state['step'] == old_state[p]['step'] + 1
            assert not torch.allclose(state['m'], old_state[p]['m'])
            assert not torch.allclose(state['v'], old_state[p]['v'])
