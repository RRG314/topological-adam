---
title: 'Topological Adam: a PyTorch optimizer family with coherence-gated field dynamics and trajectory-topology gating'
tags:
  - Python
  - PyTorch
  - optimization
  - machine learning
  - adaptive gradient methods
  - persistent homology
authors:
  - name: Steven Reid
    orcid: 0009-0003-9132-3410
    affiliation: 1
affiliations:
  - name: Independent Researcher, United States
    index: 1
date: 9 July 2026
bibliography: paper.bib
---

# Summary

`topological-adam` is a PyTorch [@paszke2019pytorch] package for testing
Adam-based optimizers [@kingma2014adam] that use structured auxiliary signals
instead of treating every momentum correction as equally reliable. The
recommended optimizer, `TopologicalAdamV3`, maintains slow and fast
exponential moving averages of each gradient tensor. Their bias-corrected
disagreement is a band-pass-filtered estimate of the recent gradient trend,
used to form an effective gradient before the Adam/AdamW update. A
coupling-current coherence gate scales that correction by its running
alignment with the current gradient, so it engages on coherent deterministic
or ill-conditioned problems and closes under minibatch noise. The package also
includes decoupled weight decay [@loshchilov2019decoupled], optional cautious
masking [@liang2024cautious], diagnostics, benchmark scripts, and older
optimizer versions retained for provenance.

The package name has two documented meanings. In V1-V3, "topological" records
the magnetohydrodynamic design lineage: paired scalar potentials can label
field-line structure and magnetic helicity is a topological invariant
[@moffatt1969; @reid2025closure]. These versions do not compute topological
invariants. The experimental `TopologicalAdamV4` makes the name operational by
computing invariants of the optimizer's own recent trajectory: the rotation
index of projected update directions [@whitney1937] and, optionally, an exact
Vietoris-Rips H1 persistent-homology barcode [@edelsbrunner2002;
@zomorodian2005]. These invariants belong to the projected trajectory, not the
loss landscape, and the persistence path is opt-in rather than part of the
default recommendation.

: Optimizer family and role. "Reduction" means bit-for-bit equivalence to Adam
or AdamW when the listed mechanism is disabled. \label{tab:family}

| Class | Mechanism | Status | Reduction |
|---|---|---|---|
| `TopologicalAdam` | Original auxiliary fields | Legacy | Tested |
| `TopologicalAdamV2` | Parameter-sourced fields, energy floor | Prior version | Tested |
| `TopologicalAdamV3` | Gradient-EMA fields and coherence gate | Recommended | Tested |
| `TopologicalAdamSDS` | Two-temperature efficiency gate | Experimental | Tested |
| `TopologicalAdamV4` | Winding and H1 trajectory-topology gate | Experimental | Tested |

# Statement of need

Optimizer papers commonly need four things: a precise update rule, comparison
to related optimizers, ablations that isolate the new mechanism, and benchmark
tables that include tuned baselines and negative results. This repository is
organized around those expectations. It addresses two practical problems in
new Adam variants: corrections can inject structured noise and reduce final
accuracy, and weak evaluations can make untuned baselines look artificially
bad. The package therefore provides exact Adam/AdamW reduction tests,
mechanism-level unit tests, per-optimizer learning-rate sweeps, held-out
metrics where data are present, fresh-seed confirmation, and published JSON
outputs for every benchmark table. In those reduction tests, V3 matches Adam
or AdamW when `w_topo=0` and `cautious=False`, while V4 matches Adam or AdamW
when `loop_gate=False`; straight V4 trajectories also keep the gate at one and
match Adam.

The target audience is optimization researchers and practitioners who want a
drop-in Adam variant that is conservative on ordinary stochastic tasks but can
use coherent gradient structure when it is present. V3 is the default
recommendation. V4 is a narrower research branch for loop- or
oscillation-structured dynamics.

# State of the field

Adam, AdamW, Nesterov acceleration, Adan, and cautious optimizers all motivate
the design space [@kingma2014adam; @loshchilov2019decoupled; @nesterov1983;
@xie2024adan; @liang2024cautious]. `topological-adam` does not replace those
standard tools; it implements a small family of PyTorch optimizers that can be
compared directly against them in ordinary training loops. The unique
contribution is not another fixed momentum formula but two reliability gates:
V3 gates a gradient-trend forecast by alignment, while V4 gates stale momentum
by trajectory topology. Existing topological data-analysis libraries can
compute persistence diagrams, but they are not optimizer implementations and
do not provide Adam-compatible state, exact-reduction tests, benchmark harnesses,
or update-rule diagnostics. A standalone package is therefore justified: the
mechanisms are experimental, auditable, and too specific for direct inclusion
in PyTorch's core optimizer set.

# Software design

The V3 design follows the lesson of the repository audit: additive field kicks
and field-energy floors can create a permanent convergence floor. V3 instead
sources both fields from the gradient stream, computes a fast-minus-slow trend
estimate, applies the correction through the effective gradient used by both
Adam moments, and uses only an energy ceiling. This preserves deep convergence
while exposing `field_metrics()` for off-hot-path diagnostics.

V4 is deliberately separate. It keeps Adam's moment state but records a small
rolling projection of the update direction into fixed random 2-D planes. The
online detector tracks turning angle, curvature, and windowed winding number;
an opt-in persistence refresh computes the exact H1 barcode of the recent
trajectory cloud. When the trajectory loops or oscillates, the momentum gate
blends toward the raw gradient; when the trajectory is topologically trivial,
the update reduces to Adam. \autoref{fig:v4} shows the full detector pipeline.

The H1 path is intentionally limited and concrete: for each stored 2-D
projection window, `topological_adam.persistence` builds the Vietoris-Rips
filtration from pairwise distances, reduces the edge-triangle boundary matrix
over Z/2, and reports the largest normalized interval as `p_loop`. Unit tests
check a sampled circle, line and noise controls, live trajectory persistence,
and that `p_loop` closes the V4 gate when `persistence_every > 0`. The V4
benchmark rows below use the default online turning/winding detector; the
exact barcode is shown in \autoref{fig:v4} and remains an opt-in diagnostic or
gate input for small trajectory windows.

![The V4 trajectory-topology detector, generated by
`examples/make_topology_figures.py`: projected update trajectory, detector and
gate traces, and the exact H1 barcode of the trajectory window.
\label{fig:v4}](docs/figures/v4_trajectory_topology.png)

# Research impact statement

The current impact evidence is reproducible near-term significance rather than
external adoption. V3's tuned benchmark suite shows its standalone
field-and-gate mechanism reaches the 1e-16 measurement floor on a
dimension-50 ill-conditioned quadratic where tuned Adam stalls near 8e-5, and
the full V3 optimizer improves held-out MSE on a noisy teacher-student task on
9 of 10 fresh seeds (paired t = +4.2). It is statistically indistinguishable
from tuned Adam on the included noisy classification, digits, and small
language-model tasks, which is reported as parity rather than hidden.

: Representative benchmark outcomes. Lower is better for all metrics shown.
V3 rows use the V3 suite with learning rates selected over the same grid on
five seeds. V4 rows use the V4 suite: learning rates are tuned on seeds 0-2
over the same grid and the table reports medians over eight fresh seeds.
\label{tab:benchmarks}

| Task | Evidence | Metric/protocol | Adam | V3/V4 | Outcome |
|---|---:|---|---:|---:|---|
| Ill-conditioned quadratic | synthetic | final optimality gap, tuned LR, 5 seeds | 8.2e-5 mean | V3: <= 1e-16 mean | V3 win |
| Teacher-student regression | synthetic | held-out MSE, 10 fresh seeds, paired test | 10-seed paired | V3 wins 9/10 | V3 win |
| Rotating field | synthetic | final squared norm, tuned LR, 8 fresh seeds | 9.4e-3 median | V4: 5.3e-3 median | V4 wins 8/8 |
| Digits MLP | real data | held-out cross-entropy, tuned LR, 8 fresh seeds | 0.104 median | V4: 0.104 median | parity |
| Teacher-student regression | synthetic | held-out MSE, tuned LR, 8 fresh seeds | 0.000769 median | V4: 0.000966 median | V4 loss |

These results support a bounded claim: V3 is a conservative default for
structured or low-noise gradients, and V4 is a specialized branch whose
topology gate helps on loop-structured synthetic dynamics but can hurt under
minibatch noise. The persistent-homology part is validated as a detector, not
reported as a broad performance win: unit and integration tests give a
prominent normalized H1 loop score on sampled-circle and rotating-trajectory
windows, low scores on line/noise/straight-descent controls, and a closed V4
gate when `persistence_every=16` feeds `p_loop` into the update rule. The
repository includes the benchmark scripts, figures, result JSON files, tests,
CI, contribution guide, citation metadata, and changelog needed for
independent review.

# AI usage disclosure

Anthropic Claude assisted with V3/V4 code review, documentation drafting,
JOSS-branch preparation, and benchmark/report organization. OpenAI Codex
(GPT-5, July 2026) assisted with repository readiness review, paper
restructuring, citation checking, and JOSS/optimizer-paper alignment. The
author reviewed, edited, and validated AI-assisted outputs, ran the reported
tests and benchmark scripts, and made the project design decisions. The author
remains responsible for correctness, originality, licensing, and claims.

# Acknowledgements

No external funding supported this work.

# References
