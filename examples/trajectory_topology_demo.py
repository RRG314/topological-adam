"""Demo: TopologicalAdamV4's trajectory-topology detector in action.

Two short experiments, both runnable on CPU in seconds:

1. **Winding detection.** A synthetic rotating gradient field in 2-D makes the
   Adam update direction trace circles. V4's projected-trajectory detector
   reports a nonzero winding number (rotation index over the rolling window),
   rising total curvature, and a gate that closes below 1 - while plain Adam
   has no notion that anything is looping.

2. **Oscillation-prone stiff quadratic.** Minimizing f(p) = p1^2 + 100 p2^2
   with a deliberately large learning rate drives Adam into sustained
   oscillation along the stiff axis. V4 detects the back-and-forth turning
   (high curvature, near-zero signed circulation), closes the momentum gate,
   and converges where plain Adam (gate off) oscillates.

Run:
    python trajectory_topology_demo.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topological_adam import TopologicalAdamV4  # noqa: E402


def demo_winding_detection():
    print("=" * 72)
    print("1. Winding detection on a rotating gradient field")
    print("=" * 72)
    print(__doc__.split("1. **Winding detection.**")[1].split("2. **")[0].strip(), "\n")

    p = torch.zeros(2, requires_grad=True)
    opt = TopologicalAdamV4([p], lr=1e-3, window=128, betas=(0.5, 0.999))
    omega = 2.0 * math.pi / 32  # one full rotation of the gradient every 32 steps

    print(f"{'step':>6} {'gate':>8} {'kappa_ema':>10} {'theta_ema':>10} {'winding':>9}")
    for t in range(192):
        angle = omega * t
        p.grad = torch.tensor([math.cos(angle), math.sin(angle)])
        opt.step()
        if (t + 1) % 32 == 0:
            m = opt.trajectory_metrics()[0]
            print(f"{t + 1:>6} {m['gate']:>8.3f} {m['kappa_ema']:>10.3f} "
                  f"{m['theta_ema']:>10.3f} {m['winding']:>9.2f}")

    m = opt.trajectory_metrics()[0]
    print(f"\nFinal winding over the {128}-step window: {m['winding']:+.2f} "
          f"(~{abs(m['winding']):.0f} full turns; sign depends on the random "
          "projection's orientation).")
    print("Signed circulation |theta_ema| ~= total curvature kappa_ema "
          "=> persistent one-directional turning, i.e. a loop, not noise.\n")


def demo_stiff_quadratic():
    print("=" * 72)
    print("2. Oscillation-prone stiff quadratic: gate on vs. gate off")
    print("=" * 72)

    scales = torch.tensor([1.0, 100.0])

    def run(loop_gate: bool):
        torch.manual_seed(0)
        p = (torch.ones(2) * 3.0).requires_grad_(True)
        opt = TopologicalAdamV4([p], lr=0.3, loop_gate=loop_gate)
        history = []
        most_closed = None
        for step in range(200):
            opt.zero_grad(set_to_none=True)
            loss = (scales * p ** 2).sum()
            loss.backward()
            opt.step()
            history.append(loss.item())
            # With loop_gate=False the detector never runs (exact-Adam fast
            # path), so there are no trajectory metrics to report.
            metrics = opt.trajectory_metrics()
            if metrics and (most_closed is None or metrics[0]["gate"] < most_closed["gate"]):
                most_closed = dict(metrics[0], step=step + 1)
        return history, most_closed

    gated, m_on = run(True)
    plain, _ = run(False)

    print(f"\nf(p) = p1^2 + 100*p2^2, lr = 0.3 (deliberately too high), 200 steps\n")
    print(f"{'step':>6} {'gate ON (V4)':>14} {'gate OFF (Adam)':>16}")
    for s in (10, 50, 100, 150, 200):
        print(f"{s:>6} {gated[s - 1]:>14.3e} {plain[s - 1]:>16.3e}")

    print(f"\nDetector at its most-closed point (step {m_on['step']}): "
          f"gate={m_on['gate']:.3f}, kappa_ema={m_on['kappa_ema']:.3f}, "
          f"theta_ema={m_on['theta_ema']:+.3f}")
    print("The update direction keeps turning instead of pointing steadily "
          "downhill (here the overshoot traces a spiral in the projected "
          "plane, so the turning is one-directional). The gate responds to "
          "total curvature, blending momentum toward the raw gradient; once "
          "the trajectory straightens out, the gate reopens toward 1.")


if __name__ == "__main__":
    demo_winding_detection()
    demo_stiff_quadratic()
