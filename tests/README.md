# Test Suite for Topological Adam

This directory contains comprehensive tests for the TopologicalAdam optimizer.

## Test Files

### `test_optimizer_basic.py`
Basic unit tests covering:
- Initialization and default parameters
- State initialization
- Basic functionality (step, zero_grad)
- Momentum updates and bias correction
- Energy and J diagnostics
- Multiple parameter groups

### `test_optimizer_edge_cases.py`
Edge case and numerical stability tests:
- Gradient edge cases (None, zero, very small/large)
- NaN and Inf handling
- Numerical stability
- Extreme parameter values
- Different tensor shapes
- Consistent behavior and state persistence

### `test_optimizer_integration.py`
PyTorch integration tests:
- Optimizer interface compliance
- State dict save/load
- Device compatibility (CPU/CUDA)
- Different model architectures (Linear, MLP, Conv, RNN)
- Training workflows (LR schedulers, gradient accumulation, clipping)
- Mixed precision training
- Special cases (frozen parameters, parameter sharing)

### `test_convergence.py`
Convergence and correctness tests:
- Simple optimization problems (quadratic, Rosenbrock)
- Linear and nonlinear regression
- Classification problems (XOR, binary classification)
- Loss reduction and convergence
- Comparison with vanilla Adam
- Field dynamics and energy tracking
- Robustness to difficult initializations and noisy gradients

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_optimizer_basic.py
```

### Run specific test class
```bash
pytest tests/test_optimizer_basic.py::TestInitialization
```

### Run specific test
```bash
pytest tests/test_optimizer_basic.py::TestInitialization::test_default_parameters
```

### Run with coverage
```bash
pytest --cov=topological_adam --cov-report=html
```

### Run only fast tests (skip slow convergence tests)
```bash
pytest -m "not slow"
```

### Run only CUDA tests
```bash
pytest -m cuda
```

## Test Coverage

The test suite aims for comprehensive coverage of:
- ✅ Initialization and parameter validation
- ✅ Core optimizer functionality
- ✅ Adam momentum calculations
- ✅ Auxiliary field updates (alpha, beta)
- ✅ Energy stabilization mechanism
- ✅ Coupling current (J) computation
- ✅ Topological correction term
- ✅ Edge cases and error handling
- ✅ PyTorch integration
- ✅ Device compatibility
- ✅ Various model architectures
- ✅ Training workflows
- ✅ Convergence on toy problems
- ✅ Comparison with vanilla Adam

## Development

When adding new features to the optimizer:
1. Add corresponding tests
2. Ensure all existing tests still pass
3. Aim for >80% code coverage
4. Document any new test markers or special requirements
