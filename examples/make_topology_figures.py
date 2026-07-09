"""Generate the repository's V4 figures (reproducible, seconds on CPU).

Writes:
- docs/figures/v4_trajectory_topology.png : the mechanism. (a) the projected
  update-direction trajectory on a rotating (non-conservative) field, which
  is topologically nontrivial; (b) the detector and gate over training;
  (c) the exact H1 persistence barcode of the trajectory window.
- docs/figures/v4_benchmark.png : per-seed fresh-seed results of tuned Adam
  vs V4 from benchmark_v4_results.json (run examples/benchmark_v4_suite.py
  first if the JSON is missing).

Run:  python examples/make_topology_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topological_adam import TopologicalAdamV4, rips_h1_persistence  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIGDIR, exist_ok=True)


# ----------------------------------------------------------------------
# Figure 1: the mechanism on a rotating (non-conservative) gradient field
# ----------------------------------------------------------------------

def figure_mechanism(steps: int = 256, period: int = 32, seed: int = 5) -> str:
    """Circulating gradient g_t = [cos(2pi t/T), sin(2pi t/T)] (+ small noise):
    the update direction winds around the origin, so the projected trajectory
    is a genuine loop — the regime V4's detector exists for."""
    torch.manual_seed(seed)
    p = torch.zeros(2, requires_grad=True)
    opt = TopologicalAdamV4([p], lr=1e-3, window=64, persistence_every=period)

    gates, kappas, windings, zs = [], [], [], []
    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        ang = 2.0 * torch.pi * t / period
        p.grad = torch.tensor(
            [torch.cos(torch.tensor(ang)), torch.sin(torch.tensor(ang))]
        ) + 0.02 * torch.randn(2)
        opt.step()
        m = opt.trajectory_metrics()[0]
        gates.append(m["gate"])
        kappas.append(m["kappa_ema"])
        windings.append(m["winding"])
        st = opt.state[p]
        idx = (st["buf_idx"] - 1) % st["z_buf"].shape[0]
        zs.append(st["z_buf"][idx, 0].clone())

    st = opt.state[p]
    fill = st["buf_fill"]
    cloud = torch.roll(st["z_buf"][:, 0, :], -st["buf_idx"], dims=0)[-fill:]
    bars = rips_h1_persistence(cloud)

    z = torch.stack(zs)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    ax = axes[0]
    sc = ax.scatter(z[:, 0], z[:, 1], c=range(len(z)), cmap="viridis", s=8)
    ax.plot(z[:, 0], z[:, 1], lw=0.4, color="gray", alpha=0.5)
    ax.set_title("(a) Projected update-direction trajectory\n(circulating gradient, plane 0)")
    ax.set_xlabel(r"$\langle d_t, u_1\rangle$")
    ax.set_ylabel(r"$\langle d_t, u_2\rangle$")
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(sc, ax=ax, label="step")

    ax = axes[1]
    ax.plot(gates, label="gate", color="tab:red")
    ax.plot([k / torch.pi for k in kappas], label=r"$\kappa_{ema}/\pi$", color="tab:blue")
    ax.plot([abs(w) / 8 for w in windings], label="|winding| / 8", color="tab:green", lw=0.9)
    ax.set_title("(b) Detector and momentum gate")
    ax.set_xlabel("step")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="center right", fontsize=8)

    ax = axes[2]
    if bars:
        shown = bars[:12]
        for i, (b, d) in enumerate(shown):
            ax.hlines(i, b, d, lw=5, color="tab:purple")
            ax.plot([b, d], [i, i], "|", color="tab:purple", ms=10)
        diam = float(torch.cdist(cloud, cloud).max())
        best = max(d - b for b, d in bars)
        ax.set_ylim(-0.6, max(len(shown) - 0.4, 0.6))
        ax.set_yticks(range(len(shown)))
        ax.set_title(
            f"(c) Exact H1 barcode of the window\nloop score = {best / diam:.2f}"
        )
    else:
        ax.set_title("(c) Exact H1 barcode of the window\n(empty)")
    ax.set_xlabel("Vietoris–Rips scale")
    ax.set_ylabel("H1 class")

    fig.tight_layout()
    out = os.path.join(FIGDIR, "v4_trajectory_topology.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ----------------------------------------------------------------------
# Figure 2: benchmark per-seed scatter, tuned Adam vs V4
# ----------------------------------------------------------------------

def figure_benchmark() -> str | None:
    path = os.path.join(ROOT, "benchmark_v4_results.json")
    if not os.path.exists(path):
        print("benchmark_v4_results.json missing; run examples/benchmark_v4_suite.py")
        return None
    with open(path) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, len(results), figsize=(3.4 * len(results), 3.6))
    for ax, (task, entry) in zip(axes, results.items()):
        a = entry["fresh"]["adam"]
        v = entry["fresh"]["v4"]
        lo = min(min(a), min(v)) * 0.5
        hi = max(max(a), max(v)) * 2.0
        ax.plot([lo, hi], [lo, hi], color="gray", lw=0.8, ls="--")
        ax.scatter(a, v, s=28, color="tab:blue", zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        wins, n = entry["v4_vs_adam"]["v4_wins_of"]
        ax.set_title(f"{task}\nV4 better on {wins}/{n} seeds", fontsize=9)
        ax.set_xlabel("tuned Adam (final metric)", fontsize=8)
        ax.set_ylabel("V4 (final metric)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        "Fresh-seed results, per-optimizer tuned learning rates "
        "(below the diagonal = V4 better)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(FIGDIR, "v4_benchmark.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", figure_mechanism())
    out = figure_benchmark()
    if out:
        print("wrote", out)
