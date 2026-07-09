# Trajectory topology: TopologicalAdamV4

**Status: experimental.** V4 is the version of this package in which the name
"Topological Adam" is *operational*: genuine topological invariants of the
optimizer's own recent trajectory are computed and used to modulate the
update. The recommended production optimizer remains `TopologicalAdamV3`.
Nothing in V4 replaces or modifies V1–V3.

## The criteria for calling this "topological" — and how V4 meets them

For the name to be honest, four things must hold. V4 is built to satisfy all
four, and each is unit-tested.

**1. It computes genuine, correctly-named topological invariants.** Two of
them:

- The **rotation index (winding number)** of the projected update-direction
  curve over a rolling window: the degree of the tangent's map to the circle
  (Whitney, 1937). Measured independently in `n_planes` random 2-D
  projections, as the accumulated signed turning angle divided by 2π.
- The **persistent homology (H1 barcode)** of the recent trajectory point
  cloud: exact Vietoris–Rips persistence over Z/2 computed by standard
  boundary-matrix reduction in `topological_adam/persistence.py` — a small,
  dependency-free, exact implementation (not an approximation, not a renamed
  heuristic). Tests verify the textbook facts: the H1 class of a circle of
  radius r dies at edge length √3·r; a straight trajectory has an empty H1
  barcode; a noise cloud has only low-persistence classes.

**2. The invariants are computed from the actual optimization trajectory.**
The inputs are the optimizer's own bias-corrected update directions
`d_t = m̂_t / (√v̂_t + ε)`, projected per parameter tensor:
`z_t^(k) = (⟨d_t, u1_k⟩, ⟨d_t, u2_k⟩)` for plane k.

**3. The invariants are operational — they change the update.** The momentum
actually used is

```
gate   = clamp(1 − gate_gain · (kappa_mean/π + p_loop),  min_gate,  1)
m_used = gate · m̂ + (1 − gate) · g_t
```

where `kappa_mean` is the mean total curvature across planes (an upper bound
on the winding rate) and `p_loop` is the persistent-homology loop score
(refreshed every `persistence_every` steps when enabled). A topologically
trivial (straight) trajectory gives `gate = 1` and the step is **bit-for-bit
Adam** — also unit-tested.

**4. The claims are scoped correctly.** The invariants belong to the
*projected trajectory* curve/point cloud — not to the loss landscape. A
finite set of random planes reduces but cannot eliminate projection
blindness: rotation orthogonal to every sampled plane is invisible. The sign
of a winding number is projection-dependent; its magnitude is not. And the
evidence below does not support "better than Adam in general" — it supports
"a specialized mechanism that acts exactly when the trajectory is
topologically nontrivial."

## Design

`Adam + rolling projected trajectory + loop/winding detector (+ persistent
homology) + gate on momentum`:

1. **Adam core.** Standard bias-corrected moments; `loop_gate=False` is
   exactly `torch.optim.Adam`/AdamW (tested to bit precision).
2. **Rolling projected trajectory.** `n_planes` (default 2) fixed random
   2-D projections per parameter tensor, seeded by `proj_seed`. With
   `store_projections=False` the projection vectors are regenerated
   deterministically each step instead of stored — O(1) extra memory for
   large models, verified to produce the *identical* trajectory.
3. **Loop/winding detector.** Signed turning angle per plane
   `Δθ = atan2(z_prev × z, z_prev · z)`; EMAs `theta_ema` (circulation) and
   `kappa_ema` (total curvature); ring-buffer winding number per plane.
4. **Persistent homology (opt-in).** With `persistence_every=k > 0`, every k
   steps the exact H1 barcode of the buffered window is computed and its
   scale-free loop score (max persistence / cloud diameter; ~0.8 for a clean
   circle, ~0 for noise or a line) feeds the gate. Costs a host sync and
   ~0.1–2 s per refresh depending on `window` — that is why it is opt-in;
   the online winding/curvature statistics are the default detector.

![The V4 mechanism end to end: (a) projected update-direction trajectory
forming a loop, (b) winding detector and momentum gate, (c) exact H1
barcode of the trajectory window](figures/v4_trajectory_topology.png)

## Reading the detector

| kappa_ema | \|theta_ema\| vs kappa_ema | winding | p_loop | interpretation |
| --- | --- | --- | --- | --- |
| ≈ 0 | — | ≈ 0 | ≈ 0 | straight descent; V4 ≡ Adam |
| high | ≈ kappa_ema | large | high | circulation: the direction is looping |
| high | ≪ kappa_ema | ≈ 0 | low | oscillation: back-and-forth sign flips |

`trajectory_metrics()` returns per-tensor scalars plus per-plane lists;
`trajectory_persistence()` returns exact per-plane H1 loop scores. Both are
off the hot path.

## Evidence — reported honestly, losses included

Protocol (`examples/benchmark_v4_suite.py`, results in
`benchmark_v4_results.json`): every optimizer *including Adam* gets its
learning rate tuned per task on seeds 0–2 over the same grid (tuning
criterion: mean, so divergent learning rates are penalized); evaluation uses
8 fresh seeds (5–12) never seen during tuning; paired per-seed statistics.
Every task is labeled synthetic or real data.

| Task | tuned Adam (median) | V4 (median) | paired t (+ favors V4) | per-seed |
|---|---|---|---|---|
| Rotating/non-conservative field [synthetic] | 9.4e-3 | **5.3e-3** | +1.60 | **V4 8/8** |
| Stiff oscillatory quadratic [synthetic] | 6.2e-9 | 5.9e-9 | −1.53 | parity 4/8 |
| digits MLP, held-out CE [real data: sklearn digits] | 0.104 | 0.104 | +0.87 | parity 4/8 |
| Teacher–student regression [synthetic] | **7.7e-4** | 9.7e-4 | −2.71 | **Adam 7/8** |

![Per-seed fresh-seed results, tuned Adam vs V4 (below the diagonal = V4
better)](figures/v4_benchmark.png)

What this says, without spin:

- **The win is exactly where the topology says it should be.** The rotating
  field has a genuinely loop-structured gradient flow; V4 beats tuned Adam
  on all 8 fresh seeds there (V3 also does well on that task).
- **Parity on real data.** On sklearn digits (real data) V4 is statistically
  indistinguishable from tuned Adam. Do not expect gains on ordinary
  stochastic classification; expect Adam-equivalent behavior.
- **A real loss, reported.** On noisy teacher–student regression, tuned Adam
  beats V4 on 7 of 8 seeds. Mini-batch gradient noise produces incidental
  turning that partially closes the gate and slows V4. This is the price of
  the mechanism, and it is why V4 is a *specialized* optimizer, not a
  general Adam replacement. (V3, whose gate is built for noise, wins on this
  task.)
- These are small-scale CPU benchmarks. No large-scale claims are made.

## Usage

```python
from topological_adam import TopologicalAdamV4

# Default: online winding/curvature detector, 2 projection planes.
opt = TopologicalAdamV4(model.parameters(), lr=1e-3)

# Exact Adam (gate off):
opt = TopologicalAdamV4(model.parameters(), lr=1e-3, loop_gate=False)

# Full topology: 3 planes + exact persistent homology every 50 steps,
# minimal-memory projections for large models:
opt = TopologicalAdamV4(
    model.parameters(), lr=1e-3,
    n_planes=3, persistence_every=50, store_projections=False,
)

for step in range(n_steps):
    opt.zero_grad(set_to_none=True)
    loss = compute_loss()
    loss.backward()
    opt.step()

for m in opt.trajectory_metrics():
    print(m["shape"], f"gate={m['gate']:.3f}", f"winding={m['winding']:+.2f}")
for r in opt.trajectory_persistence():
    print(r["shape"], "H1 loop score:", round(r["p_loop"], 3))
```

Key hyperparameters: `gate_gain` (how aggressively topological signal closes
the gate), `min_gate` (floor; momentum is never fully removed), `rho` (EMA
rate), `window` (winding window and persistence cloud size), `n_planes`
(projection planes; more planes = less blindness, more memory when stored),
`persistence_every` (0 = off), `store_projections` (speed vs memory).

Cost: the detector adds a few reductions per tensor per step (no host sync
on the hot path); memory is `2 + 2·n_planes` extra tensors per parameter
with stored projections, or 2 (just Adam's moments) plus O(window) scalars
with `store_projections=False`.

Demo: [`examples/trajectory_topology_demo.py`](../examples/trajectory_topology_demo.py)
runs winding detection on a rotating field and a gated-vs-ungated stiff
quadratic in seconds on CPU.

## Relation to V1–V3

V1/V2 derive their update from MHD-inspired field equations (see the
[preprint](preprint/README.md)); V3 is the audited, benchmarked flagship
with a coherence gate on a coupling current — its gate is driven by
*signal alignment*, which is why it handles gradient noise well. In V1–V3
the word "topological" reflects the magnetic-topology *origin* of the field
equations. V4 is the version where topological invariants of the
optimization trajectory itself are measured and acted on during training.
The two gates are complementary: V3 for noisy stochastic tasks, V4 for
loop/oscillation-structured dynamics.
