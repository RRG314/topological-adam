# Topological Adam

Energy-Stabilized Optimizer Inspired by Field-Coupling Principles
Author: Steven Reid
ORCID: 0009-0003-9132-3410
Repository: [https://github.com/RRG314/topological-adam](https://github.com/RRG314/topological-adam)
PyPI: [https://pypi.org/project/topological-adam/](https://pypi.org/project/topological-adam/)

---

## Overview

Topological Adam is an **experimental** optimization method that extends Adam with an internal energy-stabilization mechanism. The approach incorporates **computational analogues** of field-coupling concepts that are structurally similar to those used in magnetohydrodynamics (MHD), but these ideas are **not used to model, predict, or simulate any physical systems**.

All added components, including the auxiliary fields α and β, the coupling current (J_t), and the energy term (E_t), are **mathematical constructs designed solely for numerical stability and gradient control**. They do not represent physical fields or physical quantities.

The optimizer has demonstrated **stable behavior** on standard machine-learning benchmarks and has shown reduced gradient variance and smoother convergence compared to Adam in several cases. It is intended for research and exploratory use and should be treated as an experimental optimization technique.

---

## Features

* Energy-stabilized gradient updates
* Auxiliary two-field system designed for numerical stability
* Reduced gradient variance in many training scenarios
* Compatible with existing PyTorch training pipelines
* Competitive accuracy on standard benchmarks
* Low runtime overhead relative to Adam

---

## Installation

PyPI installation:

```
pip install topological-adam
```

Install directly from the repository:

```
pip install git+https://github.com/RRG314/topological-adam.git
```

---

## Basic Usage (PyTorch)

```
from topological_adam import TopologicalAdam
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

optimizer = TopologicalAdam(
    model.parameters(),
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eta=0.05,        # coupling rate
    mu0=1.0,         # field permeability constant
    w_topo=0.01,     # strength of topological correction
    target_energy=1e-3     # target energy level
)
```

Training loop:

```
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

---

## Available Versions

The package provides two versions of the Topological Adam optimizer:

### TopologicalAdam (v1)

The original implementation with energy-stabilized gradient updates.

```python
from topological_adam import TopologicalAdam

optimizer = TopologicalAdam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    eta=0.02,
    mu0=0.5,
    w_topo=0.15,
    field_init_scale=0.01,
    target_energy=1e-3
)
```

### TopologicalAdamV2 (Clean, Stable)

Enhanced version with additional stability features:
- NaN/Inf protection for gradients
- Field norm clamping to prevent runaway behavior
- Separate energy floor and ceiling controls
- Topological correction clipping
- Gradient norm floor threshold
- Extended diagnostic methods

```python
from topological_adam import TopologicalAdamV2

optimizer = TopologicalAdamV2(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    eta=0.02,
    mu0=0.5,
    w_topo=0.15,
    field_init_scale=0.01,
    target_energy=1e-3,
    # Additional v2 parameters:
    energy_floor=1e-12,           # minimum energy threshold
    energy_ceiling_mult=10.0,      # energy ceiling multiplier
    max_field_norm=10.0,           # maximum field norm
    topo_clip=1.0,                 # topological correction clip value
    grad_norm_floor=1e-12          # gradient norm threshold
)
```

### When to Use Which Version

**Use TopologicalAdam (v1)** when:
- You want the original, proven implementation
- You're working with well-behaved gradients
- You prefer simplicity and fewer hyperparameters

**Use TopologicalAdamV2** when:
- Training stability is critical
- You're dealing with potentially unstable gradients or NaN/Inf issues
- You need more fine-grained control over field dynamics
- You want additional diagnostic information

### V2 Additional Diagnostic Methods

TopologicalAdamV2 provides enhanced diagnostics:

```python
# After optimizer.step()
energy = optimizer.last_energy()           # Total field energy
j_mean = optimizer.last_J_mean_abs()       # Mean coupling current magnitude
alpha_norm = optimizer.last_alpha_norm()   # Alpha field norm sum
beta_norm = optimizer.last_beta_norm()     # Beta field norm sum
```

---

# Troubleshooting

This section provides guidance for resolving common issues when using Topological Adam.

### 1. Training becomes unstable or diverges

Possible causes and solutions:

**a. coupling rate (`eta`) too high**
Large values amplify the auxiliary field dynamics.
Use values in the range 0.01 to 0.10.

**b. topological correction weight (`w_topo`) too large**
If too strong, the corrective term dominates the Adam update.
Recommended upper bound: 0.05.

**c. learning rate too high**
Topological Adam should generally use the same learning rates as Adam, but if instability appears, try reducing by a factor of two or ten.

---

### 2. Convergence becomes slower than Adam

**a. `E_target` too high**
When the energy target is large, the auxiliary fields remain amplified, which may slow early training. Lower the value toward 0.5 to 1.0.

**b. `w_topo` too low**
If the topological term is nearly zero, the optimizer behaves almost identically to Adam but with additional overhead. Increase `w_topo` slightly (for example: 0.005 to 0.02).

**c. network is extremely small**
For very small networks (such as single-layer models), Adam may train faster because the stabilizing effect has little impact.

---

### 3. Gradients vanish in early iterations

This can occur if the internal fields remain under the target energy.

**Solution:**
Increase `target_energy` slightly. Typical values are 0.5, 1.0, or 1.5.

---

### 4. Training is overly smooth and fails to escape plateaus

This can happen if the energy stabilization suppresses variation too aggressively.

**Solutions:**

* Increase `eta` slightly
* Increase `target_energy`
* Reduce `w_topo` if the tanh correction becomes too dominant

---

### 5. GPU memory usage is higher than expected

Topological Adam maintains two auxiliary vectors (α and β) for each parameter tensor.

**Solutions:**

* Reduce batch size
* Use mixed precision (`torch.cuda.amp`)
* Verify that unused tensors are not being retained in the computation graph

---

# Recommended Defaults

These settings reflect stable behavior across all benchmarks provided in the repository. They are appropriate starting points for most users.

```
optimizer = TopologicalAdam(
    model.parameters(),
    lr=1e-3,
    beta1=0.9,
    beta2=0.999,
    eta=0.05,
    mu0=1.0,
    w_topo=0.01,
    target_energy=1e-3,
    eps=1e-8
)
```

### Recommended default ranges

| Parameter | Default | Suggested Range | Notes                                        |
| --------- | ------- | --------------- | -------------------------------------------- |
| lr        | 1e-3    | same as Adam    | Lower if training is unstable                |
| eta       | 0.05    | 0.01 to 0.10    | Coupling rate; too high can destabilize      |
| mu0       | 1.0     | 1.0             | Treated as a scaling constant                |
| w_topo    | 0.01    | 0.001 to 0.05   | Controls strength of correction              |
| target_   | 1e-3    |                 | Controls internal energy; affects smoothness |
  energy
| eps       | 1e-8    | 1e-8            | Same as Adam                                 |

### General recommendations

* Begin with the defaults.
* Tune `eta`, `w_topo`, and `target_energy` only if you see clear instability or overly smooth optimization.
* If you are not experimenting with the method, treat the defaults as fixed and modify only the learning rate.

---

## Theoretical Summary

The update rules in Topological Adam are based on a **computational reinterpretation** of certain field-interaction ideas. These rules are **designed for optimization stability**, not for physical accuracy or simulation.

### 1. Standard Adam moments

[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
]

[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
]

### 2. Auxiliary fields and coupling current

Define the coupling current:

[
J_t = (\alpha_t - \beta_t)\cdot g_t
]

The auxiliary fields evolve according to:

[
\alpha_{t+1} = (1 - \eta)\alpha_t + \frac{\eta}{\mu_0} J_t
]

[
\beta_{t+1} = (1 - \eta)\beta_t - \frac{\eta}{\mu_0} J_t
]

### 3. Energy stabilization

A joint energy term is defined for numerical regulation:

[
E_t = \frac{1}{2}\langle \alpha_t^2 + \beta_t^2 \rangle
]

If (E_t) falls below a target value, the fields are scaled upward.
If it exceeds the target, they are scaled downward.
This maintains consistent internal dynamics and helps prevent unstable updates.

### 4. Final parameter update

[
\theta_{t+1} = \theta_t - \text{lr}
\left(
\frac{\hat{m}_t}{\sqrt{\hat{v}*t} + \epsilon}
+
w*{\text{topo}} \tanh(\alpha_t - \beta_t)
\right)
]

The topological term introduces a bounded corrective influence that smooths parameter transitions.

---

## Pseudocode (Simplified)

```
for each parameter p with gradient g:

    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g * g

    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)

    J = (alpha - beta) @ g

    alpha = (1 - eta) * alpha + (eta / mu0) * J
    beta  = (1 - eta) * beta  - (eta / mu0) * J

    if energy(alpha, beta) < E_target:
        rescale(alpha, beta)

    p -= lr * (m_hat / sqrt(v_hat + eps)
               + w_topo * tanh(alpha - beta))
```

---

## Experimental Benchmarks

All benchmarks used identical architectures and hyperparameters for both Adam and Topological Adam (learning rate 1e-3, β₁ = 0.9, β₂ = 0.999, five epochs).

### MNIST

| Epoch | Adam   | Topological Adam |
| ----- | ------ | ---------------- |
| 1     | 93.84% | 91.96%           |
| 2     | 95.50% | 95.39%           |
| 3     | 96.45% | 96.36%           |
| 4     | 96.82% | 96.75%           |
| 5     | 97.24% | 96.79%           |

### KMNIST

| Epoch | Adam   | Topological Adam |
| ----- | ------ | ---------------- |
| 1     | 80.86% | 81.36%           |
| 2     | 84.80% | 85.27%           |
| 3     | 86.81% | 86.83%           |
| 4     | 87.37% | 86.75%           |
| 5     | 88.67% | 88.77%           |

### CIFAR-10

| Epoch | Adam   | Topological Adam |
| ----- | ------ | ---------------- |
| 1     | 57.97% | 60.18%           |
| 2     | 65.64% | 65.81%           |
| 3     | 68.26% | 67.64%           |
| 4     | 69.05% | 70.78%           |
| 5     | 70.73% | 68.88%           |

These results indicate that the optimizer is stable and competitive, with particularly improved behavior in early and mid-training on several tasks.

---

## Recommended Hyperparameters

| Parameter | Description                        | Typical Range |
| --------- | ---------------------------------- | ------------- |
| eta       | coupling rate                      | 0.01 to 0.10  |
| mu0       | field permeability constant        | 1.0           |
| w_topo    | strength of topological correction | 0.001 to 0.05 |
| E_target  | target energy level                | 0.5 to 2.0    |

---

## Experimental Status and Intended Use

Topological Adam is an **experimental** optimizer.
Its design is informed by abstracted ideas from MHD-style coupling, but these ideas have been **adapted entirely for computational optimization**.
The method does **not** simulate or solve physics problems.

It is most appropriate for research and exploratory contexts, including:

* investigations of alternative gradient update rules
* stabilization studies
* scenarios where Adam exhibits gradient oscillation or instability
* reinforcement learning and other noisy gradient regimes

It is not promoted as a universal improvement over Adam; users should evaluate it for their specific applications.

---

## Citation

```
Reid, Steven. "Topological Adam: Energy-Stabilized Optimizer Inspired by Field-Coupling Principles." GitHub Repository: RRG314/topological-adam, 2025.
```

---

## License

MIT License.

