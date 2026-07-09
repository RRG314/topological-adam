"""TopologicalAdamV3: field-coupled Adam with gradient-sourced auxiliary fields.

Design rationale (what changed vs V2 and why)
=============================================

V2's auxiliary fields ``alpha``/``beta`` are initialized from parameter values
(or noise) and evolve only through a *single scalar* coupling per tensor, so the
correction ``tanh(alpha - beta)`` is nearly orthogonal to the gradient
(measured mean cosine ~ -0.04 on the repo's own benchmark tasks). Combined with
the target-energy *floor*, the correction behaves as a non-vanishing structured
perturbation: it prevents convergence below a noise floor on deterministic
problems (quadratic: ~1e-5 vs Adam's ~5e-8) and is too small to matter
elsewhere.

V3 keeps the topological-field identity — two coupled auxiliary fields, a
coupling current ``J_t``, rotation dynamics, and energy regulation — but makes
three changes so the field disagreement carries real optimization signal:

1. **Gradient-sourced fields.** ``alpha`` is a slow exponential moving average
   of the gradient (long-term field memory) and ``beta`` is a fast one (recent
   field state). Their disagreement ``d = beta - alpha`` is a band-pass
   filtered gradient: it points along the *recent trend* of the gradient.
   The optional rotation coupling driven by ``J_t`` is retained.

2. **Extrapolated effective gradient instead of an additive kick.** The field
   disagreement enters through ``g_eff = g + w_topo * d`` — a one-step
   gradient forecast (Nesterov-style look-ahead; closely related to the
   gradient-difference term in Adan, Xie et al. 2022). Both Adam moments are
   computed on ``g_eff``, so the whole update remains a single, consistently
   preconditioned contracting system: near a smooth minimum the correction
   anneals with the gradient and full convergence depth is preserved (this is
   what the additive V2-style term structurally cannot do).

3. **Energy ceiling instead of a floor.** Field energy naturally anneals to
   zero at convergence because the fields are EMAs of the gradient. The only
   regulation left is a safety ceiling against blow-ups. No rescale-up floor.

Additionally V3 supports:

- **Cautious masking** (optional, on by default): update components that point
  against the current gradient are zeroed and the mask is renormalized
  (Liang et al. 2024, "Cautious Optimizers"). Guarantees the field correction
  never fights instantaneous descent.
- **Decoupled weight decay** (AdamW-style).
- No ``.item()`` host-device syncs in the hot path (stats are opt-in).
- Exact reduction to Adam(W) when ``w_topo=0`` and ``cautious=False``.

``J_t`` remains available as a diagnostic with the same meaning as in v1/v2:
mean absolute coupling ``|d * g|`` (compatible with ``stopping.py``), plus a
new alignment cosine ``cos(d, g)``.
"""

from __future__ import annotations

import torch


class TopologicalAdamV3(torch.optim.Optimizer):
    """Adam(W) augmented with gradient-sourced coupled auxiliary fields.

    Args:
        params: iterable of parameters.
        lr: learning rate.
        betas: Adam first/second moment decay.
        eps: Adam epsilon.
        weight_decay: decoupled (AdamW-style) weight decay. 0 disables.
        w_topo: weight of the field-disagreement extrapolation in the
            effective gradient ``g_eff = g + w_topo * d``. 0 recovers plain
            Adam(W) (+ cautious mask if enabled).
        eta_slow: EMA rate of the slow field ``alpha`` (long-term memory).
        eta_fast: EMA rate of the fast field ``beta`` (recent state).
        eta: strength of the rotation/exchange coupling between the fields,
            driven by the coupling current J. 0 disables rotation.
        mu0: coupling permeability (divides the rotation strength).
        max_field_norm: safety ceiling on each field's RMS, expressed as a
            multiple of the current preconditioner scale; caps blow-ups only,
            never inflates the fields (no energy floor).
        coupling_gate: if True (default), scale the field correction by the
            coherence of the coupling current: gate = |EMA of cos(d, g)|.
            Coherent field/gradient coupling (deterministic trends, valley
            traversal) engages the correction; incoherent coupling (minibatch
            noise) shuts it off and the optimizer falls back to Adam(W).
        eta_gate: EMA rate for the coupling-current coherence estimate.
        cautious: if True, zero update components whose sign disagrees with
            the current gradient and renormalize (cautious masking).
        track_stats: if True, populate ``self.stats`` each step (costs a few
            host syncs; disable in performance-critical loops).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        w_topo: float = 0.1,
        eta_slow: float = 0.02,
        eta_fast: float = 0.25,
        eta: float = 0.03,
        mu0: float = 1.0,
        max_field_norm: float = 10.0,
        coupling_gate: bool = True,
        eta_gate: float = 0.1,
        cautious: bool = True,
        track_stats: bool = False,
    ):
        if not 0.0 <= eta_slow <= 1.0 or not 0.0 <= eta_fast <= 1.0:
            raise ValueError("eta_slow and eta_fast must be in [0, 1]")
        if eta_fast <= eta_slow:
            raise ValueError("eta_fast must exceed eta_slow (fast field must be faster)")
        if not 0.0 < eta_gate <= 1.0:
            raise ValueError("eta_gate must be in (0, 1]")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            w_topo=w_topo,
            eta_slow=eta_slow,
            eta_fast=eta_fast,
            eta=eta,
            mu0=mu0,
            max_field_norm=max_field_norm,
            coupling_gate=coupling_gate,
            eta_gate=eta_gate,
            cautious=cautious,
            track_stats=track_stats,
        )
        super().__init__(params, defaults)
        self.stats = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict[str, float]:
        return {
            "energy": 0.0,
            "alpha_norm": 0.0,
            "beta_norm": 0.0,
            "coupling_current": 0.0,  # mean |d * g|  (J_t, compatible with stopping.py)
            "coupling_cos": 0.0,      # cos(d, g): alignment of field disagreement
            "topo_ratio": 0.0,        # ||w_topo * d|| / ||g||
            "gate": 0.0,              # coherence gate value |EMA(J)| in [0, 1]
            "cautious_frac": 0.0,     # fraction of coordinates kept by the mask
            "num_params": 0.0,
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        track_any = any(g["track_stats"] for g in self.param_groups)
        if track_any:
            self.stats = self._empty_stats()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            w_topo = group["w_topo"]
            eta_slow = group["eta_slow"]
            eta_fast = group["eta_fast"]
            eta = group["eta"]
            mu0 = group["mu0"]
            max_field_norm = group["max_field_norm"]
            coupling_gate = group["coupling_gate"]
            eta_gate = group["eta_gate"]
            cautious = group["cautious"]
            track = group["track_stats"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("TopologicalAdamV3 does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    if w_topo != 0.0:
                        state["alpha"] = torch.zeros_like(p)  # slow field
                        state["beta"] = torch.zeros_like(p)   # fast field

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                # ---- field dynamics (gradient-sourced, two timescales) ------------
                if w_topo != 0.0:
                    alpha, beta = state["alpha"], state["beta"]
                    alpha.mul_(1 - eta_slow).add_(grad, alpha=eta_slow)
                    beta.mul_(1 - eta_fast).add_(grad, alpha=eta_fast)

                    if eta != 0.0:
                        # rotation/exchange coupling driven by the coupling current
                        # J = cos(alpha - beta, g): dimensionless, in [-1, 1]
                        diff = alpha - beta
                        J = diff.mul(grad).sum() / (
                            diff.norm().clamp_min(1e-12) * grad.norm().clamp_min(1e-12)
                        )
                        rot = (eta / mu0) * J
                        alpha_prev = alpha.clone()
                        alpha.add_(beta, alpha=rot)
                        beta.add_(alpha_prev, alpha=-rot)

                    # bias-corrected field disagreement = gradient trend direction
                    bc_slow = 1 - (1 - eta_slow) ** t
                    bc_fast = 1 - (1 - eta_fast) ** t
                    d = beta / bc_fast - alpha / bc_slow

                    # ---- coupling-current coherence gate --------------------------
                    # J = cos(d, g) is the (normalized) coupling current. When the
                    # field disagreement is coherent with the gradient stream over
                    # time (|EMA of J| -> 1), the trend is real signal and the
                    # correction engages. When d is minibatch noise, J fluctuates
                    # around zero, its EMA vanishes, and the correction shuts off.
                    if coupling_gate:
                        J = d.mul(grad).sum() / (
                            d.norm().clamp_min(1e-12) * grad.norm().clamp_min(1e-12)
                        )
                        if "j_ema" not in state:
                            state["j_ema"] = torch.zeros((), device=p.device, dtype=J.dtype)
                        j_ema = state["j_ema"]
                        j_ema.mul_(1 - eta_gate).add_(J, alpha=eta_gate)
                        gate = j_ema.abs().clamp(max=1.0)
                    else:
                        gate = None

                    # extrapolated (predicted) gradient
                    if gate is not None:
                        g_eff = grad.add(d * gate, alpha=w_topo)
                    else:
                        g_eff = grad.add(d, alpha=w_topo)
                else:
                    g_eff = grad

                # ---- Adam moments on the effective gradient -----------------------
                m.mul_(beta1).add_(g_eff, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g_eff, g_eff, value=1 - beta2)
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                denom = (v / bc2).sqrt_().add_(eps)
                update = (m / bc1).div_(denom)

                # ---- energy ceiling on fields (safety only, uses denom scale) -----
                if w_topo != 0.0 and max_field_norm > 0:
                    cap = max_field_norm * denom.mean()
                    for f in (alpha, beta):
                        f_rms = f.pow(2).mean().sqrt()
                        f.mul_((cap / f_rms.clamp_min(1e-12)).clamp(max=1.0))

                # ---- cautious mask ------------------------------------------------
                if cautious:
                    mask = (update * grad > 0).to(update.dtype)
                    kept = mask.mean().clamp_min(1e-3)
                    update = update.mul_(mask).div_(kept)

                # ---- parameter update (decoupled weight decay) --------------------
                if wd != 0.0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)

                # ---- diagnostics --------------------------------------------------
                if track:
                    if w_topo != 0.0:
                        dvec = d.reshape(-1)
                        gvec = grad.reshape(-1)
                        dn = dvec.norm().clamp_min(1e-12)
                        gn = gvec.norm().clamp_min(1e-12)
                        self.stats["coupling_cos"] += float((dvec @ gvec) / (dn * gn))
                        self.stats["coupling_current"] += float(d.mul(grad).abs().mean())
                        self.stats["energy"] += 0.5 * float((alpha.pow(2) + beta.pow(2)).mean())
                        self.stats["alpha_norm"] += float(alpha.norm())
                        self.stats["beta_norm"] += float(beta.norm())
                        self.stats["topo_ratio"] += float(w_topo * dn / gn)
                        if coupling_gate:
                            self.stats["gate"] += float(gate)
                    if cautious:
                        self.stats["cautious_frac"] += float(mask.mean())
                    self.stats["num_params"] += 1.0

        if track_any and self.stats["num_params"] > 0:
            n = self.stats["num_params"]
            for k in self.stats:
                if k != "num_params":
                    self.stats[k] /= n
        return loss

    # -- diagnostics API compatible with v1/v2 ---------------------------------
    def field_metrics(self) -> dict[str, float]:
        total = {"energy": 0.0, "j_t": 0.0, "alpha_norm": 0.0, "beta_norm": 0.0, "alpha_beta_corr": 0.0}
        count = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "alpha" not in state:
                    continue
                alpha, beta = state["alpha"], state["beta"]
                total["energy"] += 0.5 * float((alpha.pow(2) + beta.pow(2)).mean())
                total["alpha_norm"] += float(alpha.norm())
                total["beta_norm"] += float(beta.norm())
                a = alpha.reshape(-1) - alpha.mean()
                b = beta.reshape(-1) - beta.mean()
                dnm = float(a.norm() * b.norm())
                total["alpha_beta_corr"] += float(a @ b) / dnm if dnm > 1e-12 else 0.0
                if p.grad is not None:
                    total["j_t"] += float((alpha - beta).mul(p.grad).abs().mean())
                count += 1
        if count == 0:
            return total
        return {k: v / count for k, v in total.items()}

    def get_field_stats(self) -> tuple[float, float, float]:
        metrics = self.field_metrics()
        return metrics["energy"], metrics["j_t"], metrics["alpha_beta_corr"]
