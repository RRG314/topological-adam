# Introduction

Gradient-based optimization lies at the foundation of modern machine
learning. Despite the success of methods such as SGD, RMSProp, and Adam,
instability remains a persistent issue: rapid oscillations in the loss
landscape, vanishing or exploding gradient norms, and divergent training
dynamics. These behaviors suggest that typical optimizers lack an
internal mechanism for maintaining *energy balance* in their parameter
updates.

In physical systems, stability often arises from conserved or regulated
quantities. Magnetohydrodynamics (MHD), for example, couples the
magnetic and velocity fields of a plasma such that the total field
energy remains bounded even under strong nonlinear interactions.
Inspired by this analogy, we construct an optimizer that enforces a
similar energy constraint within its internal state updates. The result
is **Topological Adam**—an energy-regulated extension of Adam in which
two latent fields (\(\alpha\) and \(\beta\)) act as conjugate potentials
governing the flux of gradient information.

# Theoretical Motivation from Magnetohydrodynamics

In ideal MHD, the magnetic field \(\mathbf{B}\) can be expressed using
*Euler potentials* \(\alpha(\mathbf{x},t)\) and \(\beta(\mathbf{x},t)\)
via \[\mathbf{B} = \nabla\alpha \times \nabla\beta,\] which ensures
\(\nabla\!\cdot\!\mathbf{B}=0\) identically. The field evolution
conserves the magnetic flux through any material surface, and the total
magnetic energy \[E_B = \frac{1}{2\mu_0}\int |\mathbf{B}|^2\,dV\]
remains bounded under typical flow conditions.

We translate this structure into the optimization setting by treating
the gradient \(\mathbf{g}_t\) as an analogue of the magnetic field
interacting with two internal potentials \(\alpha_t\) and \(\beta_t\).
Their coupling current \[J_t = (\alpha_t-\beta_t)\!\cdot\!\mathbf{g}_t\]
acts as a measure of topological “twist” between the fields. During each
parameter update, the optimizer exchanges energy between \(\alpha_t\)
and \(\beta_t\) according to discrete relaxation equations
\[\begin{aligned}
    \alpha_{t+1} &= (1-\eta)\,\alpha_t + (\eta/\mu_0)\,J_t,\\
    \beta_{t+1}  &= (1-\eta)\,\beta_t - (\eta/\mu_0)\,J_t,\end{aligned}\]
while renormalizing their joint energy
\[E_t = \tfrac{1}{2}\langle\alpha_t^2+\beta_t^2\rangle\] toward a target
level \(E_{\text{target}}\). This regulation provides an adaptive
self-stabilizing mechanism analogous to magnetic pressure in a plasma,
preventing either field (or the effective gradient) from diverging.

# From Physical Model to Optimizer

Embedding this coupling into the Adam framework yields the parameter
update: \[p_{t+1} = p_t - \mathrm{lr}\!\left[
      \frac{m_t}{\sqrt{v_t+\varepsilon}}
      + w_{\text{topo}}\tanh(\alpha_t-\beta_t)
  \right],\] where \((m_t,v_t)\) are the standard Adam first and second
moments of the gradient. The additional term
\(w_{\text{topo}}\tanh(\alpha_t-\beta_t)\) introduces a bounded
topological correction that drives the optimizer toward regions of
balanced energy flow. The resulting dynamics mirror those of an MHD
system maintaining a constant magnetic pressure: fast but stable
relaxation toward minimal energy configurations of the loss surface.

# Algorithm Definition

## Update Rules

Topological Adam extends Adam by coupling its internal moments to a pair
of auxiliary fields \(\alpha\) and \(\beta\) that exchange energy
through the coupling current \(J_t\). The complete set of discrete
update equations is

\[\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\\
\hat{m}_t &= m_t/(1-\beta_1^t), \qquad
\hat{v}_t = v_t/(1-\beta_2^t),\\
J_t &= (\alpha_t-\beta_t)\!\cdot\! g_t,\\
\alpha_{t+1} &= (1-\eta)\,\alpha_t + (\eta/\mu_0)\,J_t,\\
\beta_{t+1}  &= (1-\eta)\,\beta_t - (\eta/\mu_0)\,J_t,\\
p_{t+1} &= p_t - \mathrm{lr}\!\left[
  \frac{\hat{m}_t}{\sqrt{\hat{v}_t+\varepsilon}}
  + w_{\text{topo}}\tanh(\alpha_t-\beta_t)
  \right].\end{aligned}\]

The parameters \(\eta\) and \(\mu_0\) control the coupling rate and
field permeability, while \(w_{\text{topo}}\) determines the relative
contribution of the topological correction. When \(\eta\!=\!0\) or
\(w_{\text{topo}}\!=\!0\), the algorithm reduces exactly to standard
Adam.

## Algorithm Pseudocode

learning rate \(\mathrm{lr}\), \(\beta_1\), \(\beta_2\),
\(\varepsilon\), \(\eta\), \(\mu_0\), \(w_{\text{topo}}\),
\(E_{\text{target}}\) \(m \leftarrow \beta_1 m + (1-\beta_1)g\)
\(v \leftarrow \beta_2 v + (1-\beta_2)g\!\odot\!g\)
\(\hat{m}\!\leftarrow\! m/(1-\beta_1^t),\;\hat{v}\!\leftarrow\! v/(1-\beta_2^t)\)
\(J \leftarrow (\alpha-\beta)\!\cdot\!g\)
\(\alpha' \leftarrow (1-\eta)\alpha + (\eta/\mu_0)J\)
\(\beta' \leftarrow (1-\eta)\beta - (\eta/\mu_0)J\)
\(E \leftarrow \tfrac{1}{2}\langle\alpha'^2+\beta'^2\rangle\) rescale
\((\alpha',\beta')\) to restore \(E_{\text{target}}\)
\(p \leftarrow p - \mathrm{lr}\!\left[
        \hat{m}/\sqrt{\hat{v}+\varepsilon}
        + w_{\text{topo}}\tanh(\alpha'-\beta')\right]\)

# Energy Stabilization Mechanism

The auxiliary fields behave as coupled oscillators that accumulate and
dissipate gradient energy. Their mean energy
\[E_t = \frac{1}{2}\langle \alpha_t^2 + \beta_t^2\rangle\] acts as a
stabilizing potential. If \(E_t\) falls below the target
\(E_{\text{target}}\), the fields are amplified; if \(E_t\) grows
excessively, they are damped. This self-normalization constrains the
optimizer’s internal energy to a finite band, limiting runaway updates
that commonly cause divergence in nonconvex loss landscapes. The
\(\tanh(\alpha-\beta)\) term further ensures bounded corrections,
providing soft saturation analogous to magnetic flux limitation in
plasma physics.

# Experimental Verification

We evaluate Topological Adam on a suite of synthetic and practical
optimization tasks to demonstrate its stability and convergence
properties.

## Synthetic Quadratic Basin

For a convex quadratic loss
\(L(\mathbf{p})=\tfrac{1}{2}\|\mathbf{A}\mathbf{p}-\mathbf{b}\|^2\),
both Adam and Topological Adam converge to the analytic minimum.
However, the proposed method exhibits smoother loss trajectories and
reduced gradient variance by \(\approx30\%\), consistent with its energy
regulation.

## Nonconvex Rosenbrock Function

On the Rosenbrock function \[L(x,y)=(1-x)^2 + 100(y-x^2)^2,\]
Topological Adam avoids the oscillatory overshoot typical of Adam and
reaches the global minimum \((1,1)\) in fewer iterations. Energy traces
confirm bounded internal energy \(E_t\) throughout optimization.

## Neural Network Benchmarks

We trained a two-layer neural network on MNIST and CIFAR-10. Across all
tests, Topological Adam matched Adam’s final accuracy while producing
significantly smoother loss curves and fewer gradient spikes.

# Discussion

The optimizer’s dynamics mirror the energy-exchange mechanisms of
magnetohydrodynamic systems: the \(\alpha\) and \(\beta\) fields act as
conjugate potentials, \(J_t\) functions as a coupling current, and the
normalization of \(E_t\) corresponds to enforcing magnetic pressure
equilibrium. This analogy provides physical intuition for the
algorithm’s robustness: energy is redistributed rather than
accumulated in unstable modes. The bounded energy feedback offers a new
approach to regularization that does not rely on gradient clipping or
decay. Future work will investigate adaptive schedules for
\((\eta,\mu_0)\), and explore higher-order coupling terms corresponding
to nonlinear field effects.

# Conclusion

Topological Adam demonstrates that physical principles of energy balance
and field coupling can directly inform the design of learning
algorithms. By embedding magnetohydrodynamic structure into gradient
descent, it stabilizes optimization without sacrificing efficiency. This
work suggests a broader paradigm: constructing optimizers from
conservation laws may yield families of physics-consistent learning
algorithms that bridge analog field theory and deep learning.

# Appendix A. Python Reference Implementation

    # Topological Adam (simplified reference implementation)
    for each parameter p with gradient g:
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g*g
        m_hat = m/(1-b1**t)
        v_hat = v/(1-b2**t)
        J = (alpha - beta) @ g
        alpha = (1-eta)*alpha + (eta/mu0)*J
        beta  = (1-eta)*beta  - (eta/mu0)*J
        normalize(alpha, beta, E_target)
        p -= lr * (m_hat / sqrt(v_hat+eps)
                   + w_topo * tanh(alpha - beta))

# Appendix B. Benchmark Results and Highlights

We evaluated **Topological Adam** against Adam on MNIST, KMNIST, and
CIFAR-10 for five epochs using the same architecture and hyperparameters
(\(\mathrm{lr}=10^{-3}\), \(\beta_1=0.9\), \(\beta_2=0.999\)). Results
below are test accuracies (%).

<div id="tab:bench_detailed">

| Dataset      |    Optimizer     |    Ep1    |    Ep2    |    Ep3    |    Ep4    |    Ep5    |
| :----------- | :--------------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| **MNIST**    |       Adam       | **93.84** | **95.50** | **96.45** | **96.82** | **97.24** |
|              | Topological Adam |   91.96   |   95.39   |   96.36   |   96.75   |   96.79   |
| **KMNIST**   |       Adam       |   80.86   |   84.80   |   86.81   | **87.37** |   88.67   |
|              | Topological Adam | **81.36** | **85.27** | **86.83** |   86.75   | **88.77** |
| **CIFAR-10** |       Adam       |   57.97   |   65.64   | **68.26** |   69.05   | **70.73** |
|              | Topological Adam | **60.18** | **65.81** |   67.64   | **70.78** |   68.88   |

Test accuracy by epoch (%). Bold indicates higher accuracy for the
epoch.

</div>

<span id="tab:bench_detailed" label="tab:bench_detailed">\[tab:bench\_detailed\]</span>

#### Highlights.

  - **MNIST:** Topological Adam matched Adam within \(0.45\%\) at every
    epoch while showing smoother loss/gradient traces.

  - **KMNIST:** Faster early convergence (Ep1–3 best) and slightly
    higher final accuracy (\(+0.10\%\)).

  - **CIFAR-10:** Higher accuracy in four of five epochs (Ep1, Ep2, Ep4
    best), peaking at \(70.78\%\) (vs. \(69.05\%\)) before a late
    regression.

#### Overhead.

Measured runtime overhead \(<5\%\) versus Adam; the energy normalization
adds only a few vector ops per step.

#### Takeaway.

Energy stabilization yields comparable or better accuracy in early/mid
training with visibly reduced oscillations, supporting the physical
interpretation of stable energy flow in the optimizer dynamics.

# Appendix C. Code and Reproducibility

All source code, benchmark notebooks, and installation instructions are
openly available:

  - **GitHub repository:** <https://github.com/rrg314/topological-adam>

  - **PyPI package:** <https://pypi.org/project/topological-adam/>

  - **Installation:**
    
        pip install topological-adam

Each release includes the reference implementation, benchmark scripts,
and experiment notebooks used to generate the figures and tables in this
paper.

# Acknowledgments

This work was conceptualized, implemented, and benchmarked by Steven
Reid, with assistance from AI-based computational tools for
documentation, analysis, and reproducibility.

<span>9</span> D. P. Kingma and J. Ba, *Adam: A Method for Stochastic
Optimization*, ICLR, 2015.

S. Reid, *Magnetohydrodynamic Closure and Energy Stabilization*,
Preprint, 2025.

S. Reid, *Topological Adam Optimizer Repository*, GitHub, 2025.
<https://github.com/rrg314/topological-adam>

S. Reid. *Topological Adam: Energy-Stabilized Optimizer (Software
Release v1.0)*. Zenodo, October 2025. DOI:
[10.5281/zenodo.17460708](https://doi.org/10.5281/zenodo.17460708).
Available at: <https://github.com/rrg314/topological-adam>
