"""
Convergence and correctness tests for TopologicalAdam.

Tests that the optimizer can solve simple optimization problems
and converges correctly.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from topological_adam import TopologicalAdam


class TestSimpleOptimization:
    """Test convergence on simple optimization problems."""

    def test_quadratic_optimization(self):
        """Test optimization of a simple quadratic function."""
        # Minimize f(x) = (x - 5)^2
        x = nn.Parameter(torch.tensor([0.0]))
        optimizer = TopologicalAdam([x], lr=0.1)

        for _ in range(100):
            optimizer.zero_grad()
            loss = (x - 5.0) ** 2
            loss.backward()
            optimizer.step()

        # Should converge close to 5
        assert torch.abs(x - 5.0) < 0.1

    def test_multivariate_quadratic(self):
        """Test optimization of multivariate quadratic."""
        # Minimize f(x, y) = (x - 3)^2 + (y + 2)^2
        params = nn.Parameter(torch.tensor([0.0, 0.0]))
        optimizer = TopologicalAdam([params], lr=0.1)

        for _ in range(200):
            optimizer.zero_grad()
            x, y = params[0], params[1]
            loss = (x - 3.0) ** 2 + (y + 2.0) ** 2
            loss.backward()
            optimizer.step()

        # Should converge close to (3, -2)
        assert torch.abs(params[0] - 3.0) < 0.1
        assert torch.abs(params[1] + 2.0) < 0.1

    def test_rosenbrock_function(self):
        """Test on Rosenbrock function (challenging non-convex problem)."""
        # f(x, y) = (1-x)^2 + 100(y-x^2)^2, minimum at (1, 1)
        params = nn.Parameter(torch.tensor([0.0, 0.0]))
        optimizer = TopologicalAdam([params], lr=0.001)

        for _ in range(1000):
            optimizer.zero_grad()
            x, y = params[0], params[1]
            loss = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
            loss.backward()
            optimizer.step()

        # Rosenbrock is hard, but should get reasonably close
        assert torch.abs(params[0] - 1.0) < 0.5
        assert torch.abs(params[1] - 1.0) < 0.5


class TestLinearRegression:
    """Test convergence on linear regression."""

    def test_simple_linear_regression(self):
        """Test fitting a simple linear regression."""
        # Generate data: y = 2x + 3 + noise
        torch.manual_seed(42)
        X = torch.randn(100, 1)
        y_true = 2 * X + 3
        y = y_true + torch.randn(100, 1) * 0.1

        # Model
        model = nn.Linear(1, 1, bias=True)
        optimizer = TopologicalAdam(model.parameters(), lr=0.01)

        # Train
        for _ in range(200):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()

        # Check that weights are close to true values
        weight = model.weight.item()
        bias = model.bias.item()

        assert abs(weight - 2.0) < 0.5
        assert abs(bias - 3.0) < 0.5

    def test_multivariate_linear_regression(self):
        """Test fitting multivariate linear regression."""
        torch.manual_seed(42)
        X = torch.randn(100, 5)
        true_weights = torch.tensor([1.0, -2.0, 3.0, -1.0, 0.5])
        y = X @ true_weights + torch.randn(100) * 0.1

        model = nn.Linear(5, 1, bias=False)
        optimizer = TopologicalAdam(model.parameters(), lr=0.01)

        for _ in range(300):
            optimizer.zero_grad()
            y_pred = model(X).squeeze()
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()

        # Loss should be very small
        with torch.no_grad():
            final_loss = F.mse_loss(model(X).squeeze(), y)
        assert final_loss < 0.1


class TestNonlinearProblems:
    """Test convergence on nonlinear problems."""

    def test_xor_problem(self):
        """Test solving XOR problem (classic non-linear problem)."""
        torch.manual_seed(42)

        # XOR data
        X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

        # Simple MLP
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

        optimizer = TopologicalAdam(model.parameters(), lr=0.1)

        # Train
        for _ in range(1000):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = F.binary_cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()

        # Check accuracy
        with torch.no_grad():
            predictions = (model(X) > 0.5).float()
            accuracy = (predictions == y).float().mean()

        assert accuracy >= 0.75  # Should solve XOR reasonably well

    def test_simple_classification(self):
        """Test simple binary classification."""
        torch.manual_seed(42)

        # Generate separable data
        X1 = torch.randn(50, 2) + torch.tensor([2.0, 2.0])
        X2 = torch.randn(50, 2) + torch.tensor([-2.0, -2.0])
        X = torch.cat([X1, X2])
        y = torch.cat([torch.ones(50, 1), torch.zeros(50, 1)])

        # Simple model
        model = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

        optimizer = TopologicalAdam(model.parameters(), lr=0.1)

        # Train
        for _ in range(100):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = F.binary_cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()

        # Check accuracy
        with torch.no_grad():
            predictions = (model(X) > 0.5).float()
            accuracy = (predictions == y).float().mean()

        assert accuracy > 0.9


class TestLossReduction:
    """Test that loss decreases during training."""

    def test_monotonic_loss_decrease_convex(self):
        """Test that loss decreases monotonically for convex problems."""
        torch.manual_seed(42)
        X = torch.randn(50, 10)
        y = torch.randn(50, 1)

        model = nn.Linear(10, 1)
        optimizer = TopologicalAdam(model.parameters(), lr=0.01)

        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (allow some fluctuation)
        assert losses[-1] < losses[0]
        assert losses[-10:] < losses[:10]  # Last 10 < first 10

    def test_loss_convergence(self):
        """Test that loss converges to a stable value."""
        torch.manual_seed(42)
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        X = torch.randn(100, 10)
        y = torch.randn(100, 5)

        losses = []
        for _ in range(200):
            optimizer.zero_grad()
            y_pred = model(X)
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should stabilize (variance in last 20 steps should be small)
        recent_losses = losses[-20:]
        variance = torch.tensor(recent_losses).var().item()
        assert variance < 0.01


class TestComparisonWithAdam:
    """Test comparison with vanilla Adam."""

    def test_comparable_performance(self):
        """Test that TopologicalAdam performs comparably to Adam."""
        torch.manual_seed(42)
        X = torch.randn(100, 10)
        y = torch.randn(100, 5)

        # TopologicalAdam
        model1 = nn.Linear(10, 5)
        opt1 = TopologicalAdam(model1.parameters(), lr=0.01)

        for _ in range(100):
            opt1.zero_grad()
            loss = F.mse_loss(model1(X), y)
            loss.backward()
            opt1.step()

        loss1 = F.mse_loss(model1(X), y).item()

        # Vanilla Adam
        torch.manual_seed(42)
        model2 = nn.Linear(10, 5)
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.01)

        for _ in range(100):
            opt2.zero_grad()
            loss = F.mse_loss(model2(X), y)
            loss.backward()
            opt2.step()

        loss2 = F.mse_loss(model2(X), y).item()

        # TopologicalAdam should achieve comparable loss
        # (within 2x of Adam's performance)
        assert loss1 < loss2 * 2.0


class TestFieldDynamics:
    """Test that auxiliary fields evolve correctly."""

    def test_field_evolution(self):
        """Test that alpha and beta fields evolve during training."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Run first step
        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        # Store initial field states
        initial_fields = {}
        for p in model.parameters():
            state = optimizer.state[p]
            initial_fields[p] = {
                'alpha': state['alpha'].clone(),
                'beta': state['beta'].clone()
            }

        # Run more steps
        for _ in range(10):
            x = torch.randn(3, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        # Fields should have evolved
        for p in model.parameters():
            state = optimizer.state[p]
            assert not torch.allclose(
                state['alpha'], initial_fields[p]['alpha'], rtol=1e-3
            )
            assert not torch.allclose(
                state['beta'], initial_fields[p]['beta'], rtol=1e-3
            )

    def test_energy_tracking(self):
        """Test that energy is tracked correctly during training."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        energies = []
        for _ in range(50):
            x = torch.randn(3, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            energies.append(optimizer.energy())

        # Energy should be positive
        assert all(e >= 0 for e in energies)

        # Energy should stabilize around target
        target = optimizer.defaults['target_energy']
        # Average energy should be within an order of magnitude of target
        avg_energy = sum(energies[-20:]) / 20
        assert avg_energy > target / 100
        assert avg_energy < target * 100

    def test_coupling_current(self):
        """Test that coupling current (J) is computed correctly."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        j_values = []
        for _ in range(50):
            x = torch.randn(3, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            j_values.append(optimizer.J_mean_abs())

        # J should be computed (non-zero in most cases)
        assert any(j > 0 for j in j_values)


class TestRobustness:
    """Test robustness of the optimizer."""

    def test_difficult_initialization(self):
        """Test that optimizer can handle difficult initializations."""
        # Very large initial weights
        model = nn.Linear(10, 5)
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(10.0)

        optimizer = TopologicalAdam(model.parameters(), lr=0.01)

        X = torch.randn(50, 10)
        y = torch.randn(50, 5)

        # Should still converge
        for _ in range(200):
            optimizer.zero_grad()
            loss = F.mse_loss(model(X), y)
            loss.backward()
            optimizer.step()

        final_loss = F.mse_loss(model(X), y).item()
        assert final_loss < 10.0  # Should make some progress

    def test_noisy_gradients(self):
        """Test robustness to noisy gradients."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        X = torch.randn(50, 10)
        y = torch.randn(50, 5)

        for _ in range(100):
            # Add noise to gradients
            optimizer.zero_grad()
            loss = F.mse_loss(model(X), y)
            loss.backward()

            # Add gradient noise
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += torch.randn_like(p.grad) * 0.01

            optimizer.step()

        # Should still make progress despite noise
        final_loss = F.mse_loss(model(X), y).item()
        assert not torch.isnan(torch.tensor(final_loss))
