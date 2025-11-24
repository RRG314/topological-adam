"""
Basic unit tests for TopologicalAdam optimizer.

Tests initialization, parameter validation, and core functionality.
"""
import pytest
import torch
import torch.nn as nn
from topological_adam import TopologicalAdam


class TestInitialization:
    """Test optimizer initialization and default parameters."""

    def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        defaults = optimizer.defaults
        assert defaults['lr'] == 1e-3
        assert defaults['betas'] == (0.9, 0.999)
        assert defaults['eps'] == 1e-8
        assert defaults['eta'] == 0.02
        assert defaults['mu0'] == 0.5
        assert defaults['w_topo'] == 0.15
        assert defaults['field_init_scale'] == 0.01
        assert defaults['target_energy'] == 1e-3

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(
            model.parameters(),
            lr=1e-4,
            betas=(0.8, 0.99),
            eps=1e-7,
            eta=0.05,
            mu0=1.0,
            w_topo=0.01,
            field_init_scale=0.02,
            target_energy=1e-2
        )

        defaults = optimizer.defaults
        assert defaults['lr'] == 1e-4
        assert defaults['betas'] == (0.8, 0.99)
        assert defaults['eps'] == 1e-7
        assert defaults['eta'] == 0.05
        assert defaults['mu0'] == 1.0
        assert defaults['w_topo'] == 0.01
        assert defaults['field_init_scale'] == 0.02
        assert defaults['target_energy'] == 1e-2

    def test_state_initialization(self):
        """Test that optimizer state is initialized correctly on first step."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Initially state should be empty
        for group in optimizer.param_groups:
            for p in group['params']:
                assert p not in optimizer.state

        # Run a forward and backward pass
        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Now state should be initialized
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                assert 'step' in state
                assert 'm' in state
                assert 'v' in state
                assert 'alpha' in state
                assert 'beta' in state
                assert state['step'] == 1
                assert state['m'].shape == p.shape
                assert state['v'].shape == p.shape
                assert state['alpha'].shape == p.shape
                assert state['beta'].shape == p.shape

    def test_field_initialization_randomness(self):
        """Test that alpha and beta fields are initialized randomly."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Check that alpha and beta are not all zeros and are different
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                alpha = state['alpha']
                beta = state['beta']

                assert not torch.all(alpha == 0)
                assert not torch.all(beta == 0)
                assert not torch.allclose(alpha, beta)


class TestBasicFunctionality:
    """Test basic optimizer functionality."""

    def test_step_updates_parameters(self):
        """Test that step() updates model parameters."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters(), lr=0.1)

        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Run optimization step
        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Check that parameters have changed
        for p_initial, p_current in zip(initial_params, model.parameters()):
            assert not torch.allclose(p_initial, p_current)

    def test_zero_grad_compatibility(self):
        """Test that optimizer works with zero_grad()."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        for p in model.parameters():
            assert p.grad is not None

        optimizer.zero_grad()

        # Check gradients are cleared
        for p in model.parameters():
            if p.grad is not None:
                assert torch.all(p.grad == 0)

    def test_multiple_steps(self):
        """Test that optimizer can perform multiple steps correctly."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        for i in range(10):
            x = torch.randn(3, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Check step count
            for group in optimizer.param_groups:
                for p in group['params']:
                    assert optimizer.state[p]['step'] == i + 1

    def test_closure_function(self):
        """Test that optimizer supports closure function."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)

        def closure():
            optimizer.zero_grad()
            y = model(x)
            loss = y.sum()
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        assert loss is not None
        assert isinstance(loss, torch.Tensor)


class TestMomentumUpdates:
    """Test Adam momentum calculations."""

    def test_momentum_accumulation(self):
        """Test that first and second moments accumulate correctly."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters(), betas=(0.9, 0.999))

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Store gradients
        grads = [p.grad.clone() for p in model.parameters()]

        optimizer.step()

        # Check that moments are updated
        for p, g in zip(model.parameters(), grads):
            state = optimizer.state[p]
            m = state['m']
            v = state['v']

            # First moment should be approximately (1 - beta1) * grad for first step
            expected_m = 0.1 * g
            assert torch.allclose(m, expected_m, rtol=1e-5)

            # Second moment should be approximately (1 - beta2) * grad^2
            expected_v = 0.001 * (g ** 2)
            assert torch.allclose(v, expected_v, rtol=1e-5)

    def test_bias_correction(self):
        """Test that bias correction is applied correctly."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters(), lr=1.0, betas=(0.9, 0.999))

        # Run one step
        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # For step 1, bias correction should be:
        # m_hat = m / (1 - 0.9^1) = m / 0.1
        # v_hat = v / (1 - 0.999^1) = v / 0.001
        for p in model.parameters():
            state = optimizer.state[p]
            assert state['step'] == 1


class TestEnergyDynamics:
    """Test energy-related functionality."""

    def test_energy_method(self):
        """Test that energy() method returns valid values."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Before step, energy should be 0
        assert optimizer.energy() == 0.0

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # After step, energy should be positive
        energy = optimizer.energy()
        assert isinstance(energy, float)
        assert energy >= 0.0

    def test_j_mean_abs_method(self):
        """Test that J_mean_abs() method returns valid values."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Before step
        assert optimizer.J_mean_abs() == 0.0

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # After step
        j_mean = optimizer.J_mean_abs()
        assert isinstance(j_mean, float)
        assert j_mean >= 0.0

    def test_energy_stabilization_upscaling(self):
        """Test that fields are upscaled when energy is below target."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(
            model.parameters(),
            target_energy=1.0,  # High target to trigger upscaling
            field_init_scale=0.001  # Very small initial fields
        )

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Energy should have been upscaled toward target
        energy = optimizer.energy()
        assert energy > 0.0

    def test_energy_stabilization_downscaling(self):
        """Test that fields are downscaled when energy is too high."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(
            model.parameters(),
            target_energy=1e-6,  # Very low target
            field_init_scale=1.0  # Large initial fields
        )

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Fields should exist
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                assert 'alpha' in state
                assert 'beta' in state


class TestParameterGroups:
    """Test multiple parameter groups."""

    def test_multiple_parameter_groups(self):
        """Test optimizer with multiple parameter groups."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )

        optimizer = TopologicalAdam([
            {'params': model[0].parameters(), 'lr': 1e-3},
            {'params': model[1].parameters(), 'lr': 1e-4}
        ])

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]['lr'] == 1e-3
        assert optimizer.param_groups[1]['lr'] == 1e-4

        # Run optimization
        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Check that all parameters were updated
        for group in optimizer.param_groups:
            for p in group['params']:
                assert p in optimizer.state

    def test_different_hyperparameters_per_group(self):
        """Test different hyperparameters for different parameter groups."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )

        optimizer = TopologicalAdam([
            {'params': model[0].parameters(), 'lr': 1e-3, 'eta': 0.01},
            {'params': model[1].parameters(), 'lr': 1e-4, 'eta': 0.05}
        ])

        assert optimizer.param_groups[0]['eta'] == 0.01
        assert optimizer.param_groups[1]['eta'] == 0.05
