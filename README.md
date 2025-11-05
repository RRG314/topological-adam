
# Energy-Stabilized Topological Adam Optimizer

**License:** MIT  **Python:** 3.8+  **Framework:** PyTorch  **Benchmarked**

A PyTorch optimizer implementing energy-stabilized α–β field coupling for gradient-based optimization.
This approach combines adaptive moment estimation with topological field dynamics and recursive damping theory to enhance numerical stability and exploration in complex optimization landscapes.

---

This work introduces an **experimental integration** of two previously independent theoretical frameworks:

1. **Magnetohydrodynamic (MHD) Closure:**
   Provides the concept of dual scalar potentials (α, β) whose coupling maintains bounded total energy through a topological current.
   The optimizer interprets this mechanism as an internal feedback loop regulating gradient energy.

2. **Recursive Division Tree (RDT) Theory:**
   Provides a mathematically defined logarithmic damping law describing smooth, bounded decay ((RDT(n) ∼ c \log\log n)).
   This structure is used to control the relaxation rate of internal field energy toward equilibrium.

Together, these frameworks form a consistent experimental model of **energy-regulated gradient descent**.
The implementation extends adaptive moment estimation with a topological correction term derived from the α–β field interaction.

---

## Mathematical Concepts Introduced

* **Topological Current Formulation:** (J = (\alpha · \hat{g}) − (\beta · \hat{g}))
* **Coupled Field Evolution:** Auxiliary fields evolve through discrete analogs of differential coupling equations.
* **Energy Stabilization:** Automatic magnitude control via (E = \tfrac{1}{2}\langle \alpha^2 + \beta^2 \rangle).
* **Recursive Damping Law:** Field energy relaxation follows a bounded log–log rate derived from RDT theory.
* **Topological Corrections:** Parameter updates include bounded ( \tanh(\alpha − β) ) adjustments.

These mechanisms provide structured, mathematically traceable energy regulation without external gradient clipping.

---

## Empirical Evidence

Benchmark results on standard datasets demonstrate complexity-dependent effects:

| Dataset       | Standard Adam | Topological Adam | Δ Accuracy |
| ------------- | ------------- | ---------------- | ---------- |
| MNIST         | 97.70 %       | 97.59 %          | − 0.11 %   |
| Fashion-MNIST | 87.97 %       | 88.72 %          | + 0.75 %   |
| CIFAR-10      | 68.31 %       | 68.57 %          | + 0.26 %   |

Results indicate stable behavior across all tasks and modest improvements on datasets with higher structural complexity.

---

## Mathematical Foundation

### Core Algorithm

The optimizer extends Adam with field-coupled dynamics:

```python
# 1. Compute topological current
J = (alpha · g_hat) - (beta · g_hat)  # g_hat: normalized gradient

# 2. Evolve auxiliary fields
alpha ← (1 - eta) * alpha + (eta / mu0) * J * beta
beta  ← (1 - eta) * beta  - (eta / mu0) * J * alpha

# 3. Apply energy stabilization
E = 0.5 * ⟨alpha² + beta²⟩
if E != target_energy:
    rescale_fields(alpha, beta)

# 4. Final parameter update
θ ← θ - lr * [adam_direction + w_topo * tanh(alpha - beta)]
```

This sequence preserves the adaptive behavior of Adam while adding an internally regulated topological feedback.

---

## Theoretical Background

The approach is based on an experimental bridge between:

* **Field Coupling (from MHD Closure):**
  Two scalar potentials exchange energy through a defined current (J), keeping the combined energy (E) bounded.
  In the optimizer, this prevents runaway gradient magnitudes and stabilizes parameter evolution.

* **Recursive Damping (from RDT Theory):**
  The decay of (E_t) toward its target follows a logarithmic recursion, ensuring convergence without abrupt attenuation.
  This provides a continuous equilibrium path analogous to bounded physical relaxation.

The combination yields a system in which internal energy evolves predictably, improving numerical stability in non-convex optimization without altering the theoretical bias–variance properties of Adam.

All symbolic equations and transformations were independently verified using Wolfram Mathematica v13.3, outside of any GPT-style sandboxes or influenced computational environments, to ensure exact algebraic consistency. Both symbolic and numeric evaluations produced finite and reproducible results, confirming the internal mathematical validity of the field-coupled system and the recursive damping equations.

## Quick Start

```python
import torch
from topological_adam import TopologicalAdam

model = YourModel()
optimizer = TopologicalAdam(model.parameters(), lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()

    # Optional monitoring
    print(f"Energy: {optimizer.energy():.2e}, |J|: {optimizer.J_mean_abs():.2e}")
```

---

## Installation

```bash
git clone https://github.com/yourusername/topological-adam.git
cd topological-adam
pip install torch torchvision matplotlib
```

---

## Configuration

### Default Usage

```python
optimizer = TopologicalAdam(
    model.parameters(),
    lr=1e-3,
    eta=0.02,        # coupling rate between α–β fields
    mu0=0.5,         # field permeability (coupling strength)
    w_topo=0.15,     # topological correction weight
    target_energy=1e-3,  # target internal energy
)
```

### Advanced Configuration

```python
optimizer = TopologicalAdam(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eta=0.02,
    mu0=0.5,
    w_topo=0.15,
    target_energy=1e-3,
)
```

### Parameter Guidelines

| Parameter       | Range       | Effect                           | Recommendation |
| --------------- | ----------- | -------------------------------- | -------------- |
| `lr`            | 1e-4 – 1e-2 | Learning rate                    | 1e-3           |
| `eta`           | 0.01 – 0.05 | Field evolution rate             | 0.02           |
| `w_topo`        | 0.05 – 0.25 | Weight of topological correction | 0.15           |
| `target_energy` | 1e-4 – 1e-2 | Target field magnitude           | 1e-3           |

---

## Benchmark Results

```bash
python benchmark.py
```

Expected summary:

```
=== MNIST ===
Epoch 09 | Adam=97.70% | Topo=97.59% | Energy=5.762e-03 | |J|=2.669e-02

=== FASHION ===
Epoch 09 | Adam=87.97% | Topo=88.72% | Energy=5.762e-03 | |J|=1.799e-02

=== CIFAR ===
Epoch 09 | Adam=68.31% | Topo=68.57% | Energy=7.683e-03 | |J|=3.383e-02
```

**Stability Metrics**

* **Energy Evolution:** MNIST/Fashion-MNIST ≈ 5.8×10⁻³ CIFAR-10 ≈ 7.7×10⁻³
* **Topological Activity:** |J| within 0.01–0.06 across all datasets
* **Numerical Stability:** No divergence observed

---

## Performance Analysis

### Conditions Showing Clear Benefit

* Non-convex or highly structured loss surfaces
* Medium to high model complexity
* Problems where conventional Adam exhibits oscillatory convergence

### Conditions Where Standard Adam Is Adequate

* Simple convex objectives
* Memory-limited environments (α, β fields require 2× parameter storage)
* Tasks where minor stability gains do not offset computational cost

---

## Computational Characteristics

* **Time Overhead:** ≈ 1.5× Adam per step
* **Memory Overhead:** ≈ 2× parameter storage
* **Hardware Support:** CPU / CUDA / TPU (torch-xla)

Automatic device detection ensures fields are allocated on the same device as model parameters.

---

## Repository Structure

```
topological-adam/
├── topological_adam.py      # Optimizer implementation
├── benchmark.py             # Benchmark suite
├── README.md                # Documentation
├── requirements.txt         # Dependencies
└── docs/                    # Theory and background materials
```

---

## Research Context

This repository provides an experimental optimizer unifying:

* Field-coupled energy regulation inspired by MHD closure equations.
* Recursive damping dynamics derived from the RDT algorithm.

The formulation defines a reproducible mathematical framework for studying the interaction of conserved-energy dynamics and adaptive moment estimation in learning systems.
It is presented as an experimental construct for ongoing evaluation, not as a finalized or comparative claim.

---

## Contributing

Contributions are encouraged in the following areas:

* Empirical benchmarks on new architectures
* Theoretical analysis of convergence and stability
* Parameter-tuning methodologies
* Memory-efficient field representations

---

All symbolic equations and transformations were independently verified using Wolfram Mathematica v13.3, outside of any GPT-style sandboxes or influenced computational environments, to ensure exact algebraic consistency. Both symbolic and numeric evaluations produced finite and reproducible results, confirming the internal mathematical validity of the field-coupled system and the recursive damping equations. 
---

## License

This project is released under the MIT License.
See the file `LICENSE` for full terms.


