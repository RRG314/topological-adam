from __future__ import annotations

import math

import torch

from .shared import centered_correlation, directional_current, mean_abs_coupling_current


class TopologicalAdam(torch.optim.Optimizer):
    """Original Topological Adam implementation retained as the legacy path."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        eta: float = 0.02,
        mu0: float = 0.5,
        w_topo: float = 0.15,
        field_init_scale: float = 0.01,
        target_energy: float = 1e-3,
    ):
        params = list(params)
        self._dummy_param = None
        if not params:
            self._dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            params = [self._dummy_param]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            eta=eta,
            mu0=mu0,
            w_topo=w_topo,
            field_init_scale=field_init_scale,
            target_energy=target_energy,
        )
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
            eta, mu0, w_topo, field_init_scale, target_energy = (
                group["eta"],
                group["mu0"],
                group["w_topo"],
                group["field_init_scale"],
                group["target_energy"],
            )

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, device=p.device)
                    state["v"] = torch.zeros_like(p, device=p.device)
                    state["alpha"] = torch.tanh(p.detach()).to(p.device) * field_init_scale
                    state["beta"] = torch.cos(p.detach()).to(p.device) * (field_init_scale * 0.5)

                state["step"] += 1
                m, v, alpha, beta = state["m"], state["v"], state["alpha"], state["beta"]

                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1 ** state["step"])
                v_hat = v / (1 - b2 ** state["step"])
                adam_dir = m_hat / (v_hat.sqrt() + eps)

                coupling = directional_current(alpha, beta, g)
                if torch.isfinite(coupling):
                    alpha_prev = alpha.clone()
                    alpha.mul_(1 - eta).add_(beta, alpha=(eta / mu0) * coupling)
                    beta.mul_(1 - eta).add_(alpha_prev, alpha=-(eta / mu0) * coupling)

                    energy_local = 0.5 * ((alpha.pow(2) + beta.pow(2)).mean()).item()
                    if energy_local < target_energy:
                        scale = math.sqrt(target_energy / (energy_local + 1e-12))
                        alpha.mul_(scale)
                        beta.mul_(scale)
                    elif energy_local > target_energy * 10:
                        alpha.mul_(0.9)
                        beta.mul_(0.9)

                    topo_corr = torch.tanh(alpha - beta)
                    topo_norm = topo_corr.norm()
                    adam_norm = adam_dir.norm()
                    if torch.isfinite(topo_norm) and torch.isfinite(adam_norm):
                        if topo_norm > 0 and adam_norm > 0:
                            max_topo = 0.02 * adam_norm
                            scale = float(min(1.0, (max_topo / (topo_norm + 1e-12)).item()))
                            topo_corr = topo_corr * scale
                        else:
                            topo_corr = torch.zeros_like(p)
                    else:
                        topo_corr = torch.zeros_like(p)
                    self._energy += energy_local
                    self._J_accum += float(abs(coupling.item()))
                    self._J_count += 1
                    self._alpha_norm += alpha.norm().item()
                    self._beta_norm += beta.norm().item()
                else:
                    topo_corr = torch.zeros_like(p)

                p.add_(adam_dir + w_topo * topo_corr, alpha=-lr)
        return loss

    def energy(self) -> float:
        return self._energy

    def J_mean_abs(self) -> float:
        return self._J_accum / max(1, self._J_count)

    def field_metrics(self) -> dict[str, float]:
        total_energy = 0.0
        total_current = 0.0
        total_corr = 0.0
        total_alpha = 0.0
        total_beta = 0.0
        count = 0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p)
                if not state or "alpha" not in state:
                    continue
                alpha = state["alpha"]
                beta = state["beta"]
                total_energy += 0.5 * float((alpha.pow(2) + beta.pow(2)).mean().item())
                total_alpha += float(alpha.norm().item())
                total_beta += float(beta.norm().item())
                total_corr += centered_correlation(alpha, beta)
                if p.grad is not None:
                    total_current += mean_abs_coupling_current(alpha, beta, p.grad)
                count += 1
        if count == 0:
            return {
                "energy": 0.0,
                "j_t": 0.0,
                "alpha_norm": 0.0,
                "beta_norm": 0.0,
                "alpha_beta_corr": 0.0,
            }
        return {
            "energy": total_energy / count,
            "j_t": total_current / count,
            "alpha_norm": total_alpha / count,
            "beta_norm": total_beta / count,
            "alpha_beta_corr": total_corr / count,
        }

    def get_field_stats(self) -> tuple[float, float, float]:
        metrics = self.field_metrics()
        return metrics["energy"], metrics["j_t"], metrics["alpha_beta_corr"]
