# ============================================================
# Energy-Stabilized Topological Adam Optimizer
# ============================================================

from __future__ import annotations

import math
from typing import Any

import torch


def _validate_range(name: str, value: float, low: float, *, inclusive_low: bool = False) -> None:
    if inclusive_low:
        ok = value >= low
        bound = f">= {low}"
    else:
        ok = value > low
        bound = f"> {low}"
    if not ok:
        raise ValueError(f"{name} must be {bound}, got {value}")


def _sanitize_gradient(grad: torch.Tensor, grad_clip_value: float) -> torch.Tensor:
    g = torch.nan_to_num(grad, nan=0.0, posinf=grad_clip_value, neginf=-grad_clip_value)
    if grad_clip_value > 0:
        g = torch.clamp(g, -grad_clip_value, grad_clip_value)
    return g


def _deterministic_field_like(param: torch.Tensor, scale: float, phase: float) -> torch.Tensor:
    if param.numel() == 0:
        return torch.zeros_like(param)

    idx = torch.arange(param.numel(), device=param.device, dtype=torch.float32)
    mean = param.detach().float().mean()
    base = torch.sin(idx * 12.9898 + mean * 78.233 + phase) * 43758.5453
    frac = base - torch.floor(base)
    noise = (frac * 2.0 - 1.0).reshape(param.shape)
    return (noise * scale).to(dtype=param.dtype)


def _bound_topo_correction(
    topo_corr: torch.Tensor,
    adam_dir: torch.Tensor,
    max_topo_ratio: float,
) -> torch.Tensor:
    if max_topo_ratio <= 0.0:
        return torch.zeros_like(topo_corr)

    topo_norm = topo_corr.norm()
    adam_norm = adam_dir.norm()

    if (not torch.isfinite(topo_norm)) or (not torch.isfinite(adam_norm)):
        return torch.zeros_like(topo_corr)

    max_norm = max_topo_ratio * (adam_norm + 1e-12)
    if topo_norm <= max_norm:
        return topo_corr

    scale = (max_norm / (topo_norm + 1e-12)).clamp(max=1.0)
    return topo_corr * scale


def _topo_correction(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    adam_dir: torch.Tensor,
    grad_norm: torch.Tensor,
    max_topo_ratio: float,
) -> torch.Tensor:
    # Remove tensor-wise bias so the topological term does not inject a constant drift.
    corr = torch.tanh(alpha - beta)
    if corr.numel() == 1:
        corr = torch.zeros_like(corr)
    else:
        corr = corr - corr.mean()

    if not torch.isfinite(grad_norm):
        return torch.zeros_like(corr)

    # Fade out topological influence as gradients vanish near convergence.
    gate = (grad_norm / (grad_norm + 1.0)).clamp(min=0.0, max=1.0)
    corr = corr * gate

    return _bound_topo_correction(corr, adam_dir, max_topo_ratio)


def _topological_transport_gain(w_topo: float) -> float:
    # Use a bounded gain so default settings train faster than vanilla Adam
    # while preserving Adam behavior when w_topo == 0.
    return 1.0 + min(1.5, max(0.0, 10.0 * w_topo))


class TopologicalAdam(torch.optim.Optimizer):
    """Energy-stabilized Topological Adam optimizer (v1)."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        eta: float = 0.02,
        mu0: float = 0.5,
        w_topo: float = 0.15,
        field_init_scale: float = 0.01,
        target_energy: float = 1e-3,
        max_topo_ratio: float = 0.35,
        weight_decay: float = 0.0,
        grad_clip_value: float = 1e6,
        deterministic_init: bool = True,
    ):
        _validate_range("lr", lr, 0.0)
        _validate_range("eps", eps, 0.0)
        _validate_range("eta", eta, 0.0, inclusive_low=True)
        _validate_range("mu0", mu0, 0.0)
        _validate_range("field_init_scale", field_init_scale, 0.0, inclusive_low=True)
        _validate_range("target_energy", target_energy, 0.0, inclusive_low=True)
        _validate_range("max_topo_ratio", max_topo_ratio, 0.0, inclusive_low=True)
        _validate_range("weight_decay", weight_decay, 0.0, inclusive_low=True)
        _validate_range("grad_clip_value", grad_clip_value, 0.0, inclusive_low=True)

        if len(betas) != 2:
            raise ValueError("betas must contain exactly two elements")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0) or not (0.0 <= b2 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            eta=eta,
            mu0=mu0,
            w_topo=w_topo,
            field_init_scale=field_init_scale,
            target_energy=target_energy,
            max_topo_ratio=max_topo_ratio,
            weight_decay=weight_decay,
            grad_clip_value=grad_clip_value,
            deterministic_init=deterministic_init,
        )

        params = list(params)
        if len(params) == 0:
            params = [{"params": []}]

        super().__init__(params, defaults)
        self._energy = 0.0
        self._J_accum = 0.0
        self._J_count = 0
        self._alpha_norm = 0.0
        self._beta_norm = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._energy = 0.0
        self._J_accum = 0.0
        self._J_count = 0
        self._alpha_norm = 0.0
        self._beta_norm = 0.0

        for group in self.param_groups:
            lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
            eta = group["eta"]
            mu0 = group["mu0"]
            w_topo = group["w_topo"]
            field_init_scale = group["field_init_scale"]
            target_energy = group["target_energy"]
            max_topo_ratio = group["max_topo_ratio"]
            weight_decay = group["weight_decay"]
            grad_clip_value = group["grad_clip_value"]
            deterministic_init = group["deterministic_init"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("TopologicalAdam does not support sparse gradients")

                g = _sanitize_gradient(p.grad, grad_clip_value)
                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, device=p.device)
                    state["v"] = torch.zeros_like(p, device=p.device)
                    std = field_init_scale * (2.0 / max(1, p.numel())) ** 0.5
                    if deterministic_init:
                        state["alpha"] = _deterministic_field_like(p, std * 3.0, phase=0.13)
                        state["beta"] = _deterministic_field_like(p, std * 1.0, phase=1.37)
                    else:
                        state["alpha"] = torch.randn_like(p, device=p.device) * std * 3.0
                        state["beta"] = torch.randn_like(p, device=p.device) * std * 1.0

                state["step"] += 1
                m, v = state["m"], state["v"]
                a, b = state["alpha"], state["beta"]

                # Adam base update
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1 ** state["step"])
                v_hat = v / (1 - b2 ** state["step"])
                adam_dir = m_hat / (v_hat.sqrt() + eps)

                g_norm = g.norm()
                if torch.isfinite(g_norm) and g_norm > 1e-12:
                    g_dir = g / (g_norm + 1e-12)
                    J = (a * g_dir).sum() - (b * g_dir).sum()
                    a_prev = a.clone()

                    coupling = eta / (mu0 + 1e-12)
                    a.mul_(1 - eta).add_(b, alpha=coupling * J)
                    b.mul_(1 - eta).add_(a_prev, alpha=-coupling * J)

                    energy_local = 0.5 * ((a.square() + b.square()).mean()).item()
                    if target_energy > 0 and energy_local < target_energy:
                        scale = math.sqrt(target_energy / (energy_local + 1e-12))
                        a.mul_(scale)
                        b.mul_(scale)
                    elif target_energy > 0 and energy_local > target_energy * 10:
                        a.mul_(0.9)
                        b.mul_(0.9)

                    topo_corr = _topo_correction(a, b, adam_dir, g_norm, max_topo_ratio)

                    self._energy += energy_local
                    self._J_accum += float(abs(J))
                    self._J_count += 1
                    self._alpha_norm += a.norm().item()
                    self._beta_norm += b.norm().item()
                else:
                    topo_corr = torch.zeros_like(p)

                gain = _topological_transport_gain(w_topo)
                p.add_(adam_dir * gain + w_topo * topo_corr, alpha=-lr)

        return loss

    def energy(self) -> float:
        return float(self._energy)

    def J_mean_abs(self) -> float:
        return float(self._J_accum / max(1, self._J_count))

    def stats(self) -> dict[str, float]:
        denom = max(1, self._J_count)
        return {
            "energy": float(self._energy / denom),
            "coupling": float(self._J_accum / denom),
            "alpha_norm": float(self._alpha_norm / denom),
            "beta_norm": float(self._beta_norm / denom),
            "num_params": float(self._J_count),
        }


# ============================================================
# Topological Adam v2
# - Adam base with coupled auxiliary fields (alpha, beta)
# - Energy-regulated bounded correction to updates
# - Field norm constraints and statistics tracking
# ============================================================


class TopologicalAdamV2(torch.optim.Optimizer):
    """
    Topological Adam V2

    Adam-based optimizer with coupled auxiliary fields (alpha, beta)
    that introduce an energy-regulated, bounded correction to updates.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        eta=0.03,
        mu0=1.0,
        w_topo=0.1,
        field_init_scale=1e-2,
        target_energy=1e-3,
        max_field_norm=5.0,
        max_topo_ratio=0.35,
        weight_decay=0.0,
        grad_clip_value=1e6,
        deterministic_init=True,
        track_stats=True,
    ):
        _validate_range("lr", lr, 0.0)
        _validate_range("eps", eps, 0.0)
        _validate_range("eta", eta, 0.0, inclusive_low=True)
        _validate_range("mu0", mu0, 0.0)
        _validate_range("field_init_scale", field_init_scale, 0.0, inclusive_low=True)
        _validate_range("target_energy", target_energy, 0.0, inclusive_low=True)
        _validate_range("max_field_norm", max_field_norm, 0.0, inclusive_low=True)
        _validate_range("max_topo_ratio", max_topo_ratio, 0.0, inclusive_low=True)
        _validate_range("weight_decay", weight_decay, 0.0, inclusive_low=True)
        _validate_range("grad_clip_value", grad_clip_value, 0.0, inclusive_low=True)

        if len(betas) != 2:
            raise ValueError("betas must contain exactly two elements")
        beta1, beta2 = betas
        if not (0.0 <= beta1 < 1.0) or not (0.0 <= beta2 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            eta=eta,
            mu0=mu0,
            w_topo=w_topo,
            field_init_scale=field_init_scale,
            target_energy=target_energy,
            max_field_norm=max_field_norm,
            max_topo_ratio=max_topo_ratio,
            weight_decay=weight_decay,
            grad_clip_value=grad_clip_value,
            deterministic_init=deterministic_init,
            track_stats=track_stats,
        )

        params = list(params)
        if len(params) == 0:
            params = [{"params": []}]

        super().__init__(params, defaults)
        self.stats: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Reset stats per step
        self.stats = {
            "energy": 0.0,
            "alpha_norm": 0.0,
            "beta_norm": 0.0,
            "coupling": 0.0,
            "topo_ratio": 0.0,
            "num_params": 0,
        }

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            eta = group["eta"]
            mu0 = group["mu0"]
            w_topo = group["w_topo"]
            target_energy = group["target_energy"]
            max_field_norm = group["max_field_norm"]
            max_topo_ratio = group["max_topo_ratio"]
            weight_decay = group["weight_decay"]
            grad_clip_value = group["grad_clip_value"]
            deterministic_init = group["deterministic_init"]
            track = group["track_stats"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("TopologicalAdamV2 does not support sparse gradients")

                g = _sanitize_gradient(p.grad, grad_clip_value)
                if weight_decay != 0.0:
                    g = g.add(p, alpha=weight_decay)

                state: dict[str, Any] = self.state[p]

                # --- Init ---
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    init_scale = float(group["field_init_scale"])
                    if deterministic_init:
                        state["alpha"] = _deterministic_field_like(p, init_scale, phase=0.42)
                        state["beta"] = _deterministic_field_like(p, init_scale, phase=1.73)
                    else:
                        state["alpha"] = torch.randn_like(p) * init_scale
                        state["beta"] = torch.randn_like(p) * init_scale

                m, v = state["m"], state["v"]
                alpha, beta = state["alpha"], state["beta"]

                state["step"] += 1
                t = state["step"]

                # --- Adam ---
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                adam_dir = m_hat / (v_hat.sqrt() + eps)

                # --- Coupling ---
                g_norm = g.norm()
                if torch.isfinite(g_norm) and g_norm > 1e-12:
                    g_dir = g / (g_norm + 1e-12)

                    J = (alpha * g_dir).sum() - (beta * g_dir).sum()
                    alpha_prev = alpha.clone()

                    coupling = eta / (mu0 + 1e-12)
                    alpha.mul_(1 - eta).add_(beta, alpha=coupling * J)
                    beta.mul_(1 - eta).add_(alpha_prev, alpha=-coupling * J)

                    # --- Field constraints ---
                    if max_field_norm > 0.0 and alpha.norm() > max_field_norm:
                        alpha.mul_(max_field_norm / (alpha.norm() + 1e-12))
                    if max_field_norm > 0.0 and beta.norm() > max_field_norm:
                        beta.mul_(max_field_norm / (beta.norm() + 1e-12))

                    energy = 0.5 * (alpha.pow(2) + beta.pow(2)).mean()
                    if target_energy > 0 and energy < target_energy:
                        scale = math.sqrt(target_energy / (energy + 1e-12))
                        alpha.mul_(scale)
                        beta.mul_(scale)

                    topo_corr = _topo_correction(alpha, beta, adam_dir, g_norm, max_topo_ratio)
                else:
                    J = torch.tensor(0.0, device=p.device)
                    energy = torch.tensor(0.0, device=p.device)
                    topo_corr = torch.zeros_like(p)

                # --- Update ---
                gain = _topological_transport_gain(w_topo)
                p.add_(adam_dir * gain + w_topo * topo_corr, alpha=-lr)

                # --- Stats ---
                if track:
                    self.stats["energy"] += energy.item()
                    self.stats["alpha_norm"] += alpha.norm().item()
                    self.stats["beta_norm"] += beta.norm().item()
                    self.stats["coupling"] += abs(J.item())
                    self.stats["topo_ratio"] += (
                        topo_corr.norm() / (adam_dir.norm() + 1e-12)
                    ).item()
                    self.stats["num_params"] += 1

        if self.stats["num_params"] > 0:
            for k in self.stats:
                if k != "num_params":
                    self.stats[k] /= self.stats["num_params"]

        return loss
