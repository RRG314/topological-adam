"""TopologicalAdamV4: trajectory-topology gated Adam (experimental).

This is the version of the optimizer in which the word "topological" is
*operational*: the optimizer computes genuine topological invariants of its
own recent update trajectory and uses them to modulate the update rule.

What makes this honestly "Topological Adam" — the criteria
----------------------------------------------------------
1. **Genuine invariants, correctly named.** Two are computed:

   - the **rotation index (winding number)** of the projected
     update-direction curve over a rolling window — the degree of the
     tangent's map to the circle (Whitney 1937), measured independently in
     ``n_planes`` random 2-D projections; and
   - the **persistent homology (H1 barcode)** of the recent trajectory
     point cloud — Vietoris-Rips persistence computed *exactly* (over Z/2,
     standard boundary-matrix reduction; see
     :mod:`topological_adam.persistence`), summarized as a scale-free loop
     persistence score.

   Both are textbook topological quantities, not renamed heuristics.

2. **Computed from the actual optimization trajectory.** The inputs are the
   optimizer's own bias-corrected update directions
   ``d_t = m_hat / (sqrt(v_hat) + eps)``, projected per parameter tensor.

3. **Operational, not decorative.** The invariants drive the momentum gate:
   ``gate = clamp(1 - gate_gain*(kappa_mean/pi + p_loop), min_gate, 1)`` and
   ``m_used = gate*m_hat + (1-gate)*grad``. A topologically trivial
   (straight) trajectory gives ``gate = 1`` and the step is *bit-for-bit*
   Adam; loops and oscillations close the gate. ``p_loop`` is the persistent
   H1 loop score, refreshed every ``persistence_every`` steps when enabled.

4. **Claims scoped correctly.** The invariants belong to the (projected)
   trajectory curve/point cloud, NOT to the loss landscape. Multiple
   projection planes reduce — but cannot eliminate — projection blindness;
   rotation orthogonal to all sampled planes is invisible. No claim of broad
   superiority over Adam is made: see docs/trajectory-topology.md for the
   honest evidence and its limits.

Design
------
``Adam + rolling projected trajectory + loop/winding detector (+ persistent
homology) + gate on momentum``:

- **Rolling projected trajectory.** For every parameter tensor, ``n_planes``
  pairs of fixed random unit vectors define 2-D projections of the update
  direction: ``z_t^(k) = (<d_t, u1_k>, <d_t, u2_k>)``. With
  ``store_projections=False`` the vectors are regenerated deterministically
  each step from ``proj_seed`` instead of stored (O(1) extra memory per
  plane at ~1 extra RNG pass per step).
- **Loop/winding detector.** Signed turning angles
  ``dtheta = atan2(z_prev x z, z_prev . z)`` per plane feed EMAs of
  circulation (``theta_ema``) and total curvature (``kappa_ema``), plus a
  ring buffer whose sum / 2*pi is the windowed winding number per plane.
- **Persistent homology (optional, ``persistence_every > 0``).** Every k
  steps, the exact H1 barcode of the buffered projected trajectory is
  computed off-device and its loop-persistence score is folded into the
  gate. This is the "persistent topology of recent updates" detector; it is
  exact but costs ~0.1-2 s per refresh depending on ``window``, so it is
  opt-in and infrequent.
- **Gate on momentum.** Curvature (which upper-bounds the winding rate) plus
  persistent loop score close the gate; the update blends stale momentum
  toward the current gradient exactly when the trajectory is topologically
  nontrivial.

Reading the detector: ``|theta_ema| ~ kappa_ema`` with large ``|winding|``
means circulation (loops); ``kappa_ema`` large with ``theta_ema`` near zero
and ``winding`` near zero means oscillation; both near zero means straight
descent (exact Adam). ``trajectory_metrics()`` and
``trajectory_persistence()`` expose everything off the hot path.

With ``loop_gate=False`` this optimizer is exactly Adam (plus optional
decoupled weight decay), verified by unit tests. No ``.item()`` calls or
other host synchronization occur on the hot path except the opt-in
persistence refresh.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from .persistence import max_loop_score

__all__ = ["TopologicalAdamV4"]

_TINY = 1e-24  # squared-norm floor below which turning is treated as zero


class TopologicalAdamV4(Optimizer):
    """Adam gated by topological invariants of its own update trajectory.

    Args:
        params: iterable of parameters or param groups.
        lr: learning rate.
        betas: Adam ``(beta1, beta2)``.
        eps: Adam epsilon.
        weight_decay: decoupled (AdamW-style) weight decay.
        loop_gate: enable the trajectory detector and momentum gate.
            ``False`` reduces the optimizer to exact Adam(W).
        gate_gain: how aggressively topological signal closes the gate:
            ``gate = clamp(1 - gate_gain*(kappa_mean/pi + p_loop),
            min_gate, 1)``.
        min_gate: floor of the momentum gate (never remove momentum
            entirely).
        rho: EMA rate of the turning statistics.
        window: ring-buffer length for the rolling winding number and the
            persistent-homology point cloud.
        n_planes: number of independent random 2-D projection planes per
            parameter tensor. More planes reduce projection blindness at
            2 extra tensors of parameter size each (when stored).
        proj_seed: seed for the fixed random projection vectors.
        store_projections: keep projection vectors in optimizer state
            (fast; 2*n_planes extra tensors per parameter). ``False``
            regenerates them deterministically each step (minimal memory).
        persistence_every: if > 0, refresh the exact persistent-homology
            loop score every this many steps and fold it into the gate.
            Costs host synchronization at each refresh; leave 0 to gate on
            the online winding/curvature statistics only.
        persistence_max_points: subsample cap for the persistence point
            cloud (H1 reduction is O(N^3) in points).
        track_stats: retained for API symmetry with V2/V3; detector state
            is always available through :meth:`trajectory_metrics`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        loop_gate: bool = True,
        gate_gain: float = 1.0,
        min_gate: float = 0.1,
        rho: float = 0.1,
        window: int = 64,
        n_planes: int = 2,
        proj_seed: int = 0x7A11,
        store_projections: bool = True,
        persistence_every: int = 0,
        persistence_max_points: int = 64,
        track_stats: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if not 0.0 < min_gate <= 1.0:
            raise ValueError(f"min_gate must be in (0, 1], got {min_gate}")
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if n_planes < 1:
            raise ValueError(f"n_planes must be >= 1, got {n_planes}")
        if persistence_every < 0:
            raise ValueError(
                f"persistence_every must be >= 0, got {persistence_every}"
            )
        if persistence_max_points < 4:
            raise ValueError(
                f"persistence_max_points must be >= 4, got {persistence_max_points}"
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            loop_gate=loop_gate,
            gate_gain=gate_gain,
            min_gate=min_gate,
            rho=rho,
            window=window,
            n_planes=n_planes,
            proj_seed=proj_seed,
            store_projections=store_projections,
            persistence_every=persistence_every,
            persistence_max_points=persistence_max_points,
            track_stats=track_stats,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _make_projections(p, group, index):
        """(2*n_planes, numel) row-normalized random projection matrix."""
        n_planes = group["n_planes"]
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(group["proj_seed"]) + 7919 * index)
        u = torch.randn((2 * n_planes, p.numel()), generator=gen, dtype=torch.float32)
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return u.to(dtype=p.dtype, device=p.device)

    def _init_state(self, p, group, index):
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p)
        state["exp_avg_sq"] = torch.zeros_like(p)
        if group["loop_gate"]:
            n_planes = group["n_planes"]
            window = group["window"]
            if group["store_projections"]:
                state["proj"] = self._make_projections(p, group, index)
            state["prev_z"] = torch.zeros(
                n_planes, 2, dtype=p.dtype, device=p.device
            )
            state["has_prev"] = False
            state["theta_ema"] = torch.zeros(
                n_planes, dtype=p.dtype, device=p.device
            )
            state["kappa_ema"] = torch.zeros(
                n_planes, dtype=p.dtype, device=p.device
            )
            state["dtheta_buf"] = torch.zeros(
                window, n_planes, dtype=p.dtype, device=p.device
            )
            state["z_buf"] = torch.zeros(
                window, n_planes, 2, dtype=p.dtype, device=p.device
            )
            state["buf_idx"] = 0
            state["buf_fill"] = 0
            state["buf_sum"] = torch.zeros(
                n_planes, dtype=p.dtype, device=p.device
            )
            state["p_loop"] = torch.zeros((), dtype=p.dtype, device=p.device)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]
            use_gate = group["loop_gate"]
            gain = group["gate_gain"]
            min_gate = group["min_gate"]
            rho = group["rho"]
            window = group["window"]
            pers_every = group["persistence_every"]

            for index, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "TopologicalAdamV4 does not support sparse gradients"
                    )

                state = self.state[p]
                if len(state) == 0:
                    self._init_state(p, group, index)

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                m_hat = m / bc1
                denom = (v / bc2).sqrt_().add_(eps)

                if not use_gate:
                    p.addcdiv_(m_hat, denom, value=-lr)
                    continue

                # --- rolling projected trajectory -------------------------
                direction = m_hat / denom
                proj = state.get("proj")
                if proj is None:
                    proj = self._make_projections(p, group, index)
                z = (proj @ direction.reshape(-1)).view(-1, 2)  # (n_planes, 2)

                idx = state["buf_idx"]
                if state["has_prev"]:
                    prev = state["prev_z"]
                    cross = prev[:, 0] * z[:, 1] - prev[:, 1] * z[:, 0]
                    dot = (prev * z).sum(dim=1)
                    sq = (z * z).sum(dim=1) * (prev * prev).sum(dim=1)
                    dtheta = torch.where(
                        sq > _TINY,
                        torch.atan2(cross, dot),
                        torch.zeros_like(cross),
                    )
                    # --- loop / winding detector --------------------------
                    state["theta_ema"].mul_(1.0 - rho).add_(dtheta, alpha=rho)
                    state["kappa_ema"].mul_(1.0 - rho).add_(
                        dtheta.abs(), alpha=rho
                    )
                    buf = state["dtheta_buf"]
                    state["buf_sum"].add_(dtheta - buf[idx])
                    buf[idx] = dtheta
                state["z_buf"][idx] = z
                state["buf_idx"] = (idx + 1) % window
                state["buf_fill"] = min(state["buf_fill"] + 1, window)
                state["prev_z"].copy_(z)
                state["has_prev"] = True

                # --- persistent homology refresh (opt-in, off-device) -----
                if (
                    pers_every > 0
                    and state["buf_fill"] >= window
                    and t % pers_every == 0
                ):
                    state["p_loop"].fill_(
                        self._loop_persistence(state, group)
                    )

                # --- gate on momentum --------------------------------------
                loop_signal = state["kappa_ema"].mean() / math.pi
                gate = (1.0 - gain * (loop_signal + state["p_loop"])).clamp(
                    min=min_gate, max=1.0
                )
                m_used = gate * m_hat + (1.0 - gate) * grad
                p.addcdiv_(m_used, denom, value=-lr)

        return loss

    # ------------------------------------------------------------------
    # diagnostics (off the hot path)
    # ------------------------------------------------------------------

    @staticmethod
    def _loop_persistence(state, group) -> float:
        """Max H1 loop-persistence score across planes for one tensor."""
        window = group["window"]
        fill = state["buf_fill"]
        if fill < 8:
            return 0.0
        idx = state["buf_idx"]
        zb = state["z_buf"]
        if fill >= window:
            pts_all = torch.cat([zb[idx:], zb[:idx]], dim=0)
        else:
            pts_all = zb[:fill]
        cap = group["persistence_max_points"]
        if pts_all.shape[0] > cap:
            sel = torch.linspace(
                0, pts_all.shape[0] - 1, cap, dtype=torch.long
            )
            pts_all = pts_all[sel]
        best = 0.0
        for k in range(pts_all.shape[1]):
            score = max_loop_score(pts_all[:, k, :])
            if score > best:
                best = score
        return best

    @torch.no_grad()
    def trajectory_persistence(self):
        """Exact persistent-homology loop scores per tensor (syncs host).

        Returns a list of dicts with the tensor ``shape``, per-plane H1
        ``loop_scores`` (scale-free prominence of the most persistent loop
        in each projected trajectory, in [0, 1]), and their max ``p_loop``.
        Computed from the same rolling window the winding detector uses.
        """
        out = []
        for group in self.param_groups:
            if not group["loop_gate"]:
                continue
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "z_buf" not in state:
                    continue
                window = group["window"]
                fill = state["buf_fill"]
                idx = state["buf_idx"]
                zb = state["z_buf"]
                if fill >= window:
                    pts_all = torch.cat([zb[idx:], zb[:idx]], dim=0)
                else:
                    pts_all = zb[:fill]
                cap = group["persistence_max_points"]
                if pts_all.shape[0] > cap:
                    sel = torch.linspace(
                        0, pts_all.shape[0] - 1, cap, dtype=torch.long
                    )
                    pts_all = pts_all[sel]
                scores = [
                    max_loop_score(pts_all[:, k, :])
                    for k in range(pts_all.shape[1])
                ]
                out.append(
                    dict(
                        shape=tuple(p.shape),
                        loop_scores=scores,
                        p_loop=max(scores) if scores else 0.0,
                    )
                )
        return out

    @torch.no_grad()
    def trajectory_metrics(self):
        """Per-parameter detector diagnostics (off the hot path; syncs host).

        Returns a list of dicts per parameter tensor:

        - ``gate``: the momentum gate currently in effect;
        - ``kappa_ema``: mean total curvature across planes (radians/step);
        - ``theta_ema``: signed circulation of the plane where it is
          strongest;
        - ``winding``: rolling winding number of the plane where its
          magnitude is largest;
        - ``windings`` / ``theta_emas`` / ``kappa_emas``: per-plane values;
        - ``p_loop``: last persistent-homology loop score used by the gate
          (0 unless ``persistence_every > 0``).
        """
        out = []
        for group in self.param_groups:
            if not group["loop_gate"]:
                continue
            gain = group["gate_gain"]
            min_gate = group["min_gate"]
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "kappa_ema" not in state:
                    continue
                kappas = state["kappa_ema"].tolist()
                thetas = state["theta_ema"].tolist()
                windings = (state["buf_sum"] / (2.0 * math.pi)).tolist()
                kappa_mean = sum(kappas) / len(kappas)
                p_loop = float(state["p_loop"])
                gate = min(
                    max(
                        1.0 - gain * (kappa_mean / math.pi + p_loop),
                        min_gate,
                    ),
                    1.0,
                )
                theta_star = max(thetas, key=abs)
                winding_star = max(windings, key=abs)
                out.append(
                    dict(
                        shape=tuple(p.shape),
                        gate=gate,
                        theta_ema=theta_star,
                        kappa_ema=kappa_mean,
                        winding=winding_star,
                        theta_emas=thetas,
                        kappa_emas=kappas,
                        windings=windings,
                        p_loop=p_loop,
                    )
                )
        return out
