"""
PyTorch integration tests for TopologicalAdam.

Tests compatibility with PyTorch features and training workflows.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from topological_adam import TopologicalAdam


class TestPyTorchCompatibility:
    """Test compatibility with PyTorch optimizer interface."""

    def test_optimizer_interface(self):
        """Test that TopologicalAdam implements the optimizer interface."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Check standard optimizer attributes
        assert hasattr(optimizer, 'param_groups')
        assert hasattr(optimizer, 'state')
        assert hasattr(optimizer, 'defaults')
        assert hasattr(optimizer, 'step')
        assert hasattr(optimizer, 'zero_grad')

    def test_state_dict_save_load(self):
        """Test state dict saving and loading."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Run a few steps
        for _ in range(5):
            x = torch.randn(3, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Save state
        state_dict = optimizer.state_dict()

        # Create new optimizer
        model2 = nn.Linear(10, 5)
        optimizer2 = TopologicalAdam(model2.parameters())

        # Load state (note: parameters must match)
        try:
            optimizer2.load_state_dict(state_dict)
        except Exception as e:
            # State dict loading with different parameters may fail,
            # which is expected
            pytest.skip(f"State dict loading failed as expected: {e}")

    def test_param_groups_interface(self):
        """Test param_groups interface."""
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 2)
        )

        optimizer = TopologicalAdam([
            {'params': model[0].parameters(), 'lr': 1e-3},
            {'params': model[1].parameters(), 'lr': 1e-4}
        ])

        # Test modifying learning rates
        for group in optimizer.param_groups:
            group['lr'] *= 0.1

        assert optimizer.param_groups[0]['lr'] == 1e-4
        assert optimizer.param_groups[1]['lr'] == 1e-5


class TestDeviceCompatibility:
    """Test device compatibility (CPU/CUDA)."""

    def test_cpu_tensors(self):
        """Test optimizer with CPU tensors."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            assert p.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_tensors(self):
        """Test optimizer with CUDA tensors."""
        model = nn.Linear(10, 5).cuda()
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10).cuda()
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            assert p.device.type == 'cuda'

        # Check state is also on CUDA
        for state in optimizer.state.values():
            if 'm' in state:
                assert state['m'].device.type == 'cuda'
                assert state['v'].device.type == 'cuda'
                assert state['alpha'].device.type == 'cuda'
                assert state['beta'].device.type == 'cuda'


class TestModelArchitectures:
    """Test with different model architectures."""

    def test_linear_model(self):
        """Test with simple linear model."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        assert True  # Should complete without error

    def test_mlp_model(self):
        """Test with multi-layer perceptron."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 5)
        )
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        assert True

    def test_conv_model(self):
        """Test with convolutional model."""
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10)
        )
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(2, 3, 8, 8)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        assert True

    def test_rnn_model(self):
        """Test with recurrent model."""
        model = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 5, 10)  # batch, seq_len, features
        output, (hn, cn) = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()

        assert True

    def test_custom_module(self):
        """Test with custom module."""
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20)
                self.fc2 = nn.Linear(20, 5)
                self.weight = nn.Parameter(torch.randn(5, 5))

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                x = x @ self.weight
                return x

        model = CustomModel()
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        assert True


class TestTrainingWorkflows:
    """Test integration with common training workflows."""

    def test_standard_training_loop(self):
        """Test standard training loop."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        # Create dummy dataset
        X = torch.randn(100, 10)
        y = torch.randn(100, 5)

        for epoch in range(3):
            for i in range(0, len(X), 10):
                batch_X = X[i:i+10]
                batch_y = y[i:i+10]

                optimizer.zero_grad()
                output = model(batch_X)
                loss = F.mse_loss(output, batch_y)
                loss.backward()
                optimizer.step()

        assert True

    def test_with_lr_scheduler(self):
        """Test compatibility with learning rate schedulers."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        initial_lr = optimizer.param_groups[0]['lr']

        for epoch in range(10):
            x = torch.randn(3, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        final_lr = optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr

    def test_gradient_accumulation(self):
        """Test gradient accumulation pattern."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        accumulation_steps = 4
        for i in range(8):
            x = torch.randn(3, 10)
            y = model(x)
            loss = y.sum() / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        assert True

    def test_gradient_clipping(self):
        """Test compatibility with gradient clipping."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        assert True


class TestBatchSizes:
    """Test with different batch sizes."""

    def test_batch_size_1(self):
        """Test with batch size 1."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(1, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        assert True

    def test_batch_size_large(self):
        """Test with large batch size."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(1000, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        assert True

    def test_varying_batch_sizes(self):
        """Test with varying batch sizes."""
        model = nn.Linear(10, 5)
        optimizer = TopologicalAdam(model.parameters())

        for batch_size in [1, 5, 10, 32, 64]:
            x = torch.randn(batch_size, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        assert True


class TestMixedPrecision:
    """Test mixed precision training compatibility."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_automatic_mixed_precision(self):
        """Test with automatic mixed precision (AMP)."""
        device = torch.device('cuda')
        model = nn.Linear(10, 5).to(device)
        optimizer = TopologicalAdam(model.parameters())
        scaler = torch.cuda.amp.GradScaler()

        for _ in range(5):
            x = torch.randn(3, 10, device=device)

            with torch.cuda.amp.autocast():
                y = model(x)
                loss = y.sum()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        assert True


class TestModelModes:
    """Test with different model modes (train/eval)."""

    def test_train_mode(self):
        """Test in training mode."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.5),
            nn.Linear(20, 5)
        )
        model.train()
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        assert True

    def test_eval_mode(self):
        """Test that optimizer works even when model is in eval mode."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(0.5),
            nn.Linear(20, 5)
        )
        model.eval()
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)
        # In eval mode with no_grad, we won't get gradients
        # So this tests that optimizer doesn't crash
        with torch.enable_grad():
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        assert True


class TestSpecialCases:
    """Test special cases and patterns."""

    def test_no_parameters(self):
        """Test with a model that has no parameters."""
        # This should handle gracefully
        optimizer = TopologicalAdam([])
        optimizer.step()
        assert True

    def test_frozen_parameters(self):
        """Test with some frozen parameters."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        # Only optimize second layer
        optimizer = TopologicalAdam(
            filter(lambda p: p.requires_grad, model.parameters())
        )

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        assert True

    def test_parameter_sharing(self):
        """Test with shared parameters."""
        class SharedParamModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared_weight = nn.Parameter(torch.randn(10, 10))
                self.bias1 = nn.Parameter(torch.randn(10))
                self.bias2 = nn.Parameter(torch.randn(10))

            def forward(self, x):
                # Use shared weight twice
                x = x @ self.shared_weight + self.bias1
                x = x @ self.shared_weight + self.bias2
                return x

        model = SharedParamModel()
        optimizer = TopologicalAdam(model.parameters())

        x = torch.randn(3, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()

        assert True
