from __future__ import annotations

import math

import torch

from .shared import (
    bounded_topological_correction,
    centered_correlation,
    deterministic_field_like,
    directional_current,
    mean_abs_coupling_current,
    two_temperature_efficiency,
)


class TopologicalAdamSDS(torch.optim.Optimizer):
    """Experimental SdS-inspired branch with a two-temperature efficiency gate.

    This optimizer keeps the V2 field dynamics, but gates the topological
    correction by a bounded Carnot-style efficiency derived from the two
    auxiliary fields. It is an experimental branch, not the default path.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        eta: float = 0.03,
        mu0: float = 1.0,
        w_topo: float = 0.1,
        field_init_scale: float = 1e-2,
        target_energy: float = 1e-3,
        max_field_norm: float = 5.0,
        max_topo_ratio: float | None = None,
        deterministic_init: bool = False,
        efficiency_power: float = 1.0,
        min_efficiency: float = 0.0,
        track_stats: bool = True,
    ):
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
            deterministic_init=deterministic_init,
            efficiency_power=efficiency_power,
            min_efficiency=min_efficiency,
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
            "coupling_current": 0.0,
            "coupling": 0.0,
            "topo_ratio": 0.0,
            "T_hot": 0.0,
            "T_cold": 0.0,
            "eta_c": 0.0,
            "num_params": 0.0,
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.stats = self._empty_stats()

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
            deterministic_init = group["deterministic_init"]
            field_init_scale = group["field_init_scale"]
            efficiency_power = max(0.0, float(group["efficiency_power"]))
            min_efficiency = max(0.0, min(1.0, float(group["min_efficiency"])))
            track = group["track_stats"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    if deterministic_init:
                        state["alpha"] = deterministic_field_like(p, field_init_scale, phase=0.13)
                        state["beta"] = deterministic_field_like(p, field_init_scale, phase=1.37)
                    else:
                        state["alpha"] = torch.randn_like(p) * field_init_scale
                        state["beta"] = torch.randn_like(p) * field_init_scale

                m = state["m"]
                v = state["v"]
                alpha = state["alpha"]
                beta = state["beta"]

                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                adam_dir = m_hat / (v_hat.sqrt() + eps)

                coupling = directional_current(alpha, beta, grad)
                if torch.isfinite(coupling):
                    thermo = two_temperature_efficiency(alpha, beta)
                    coupling_gain = max(min_efficiency, thermo["eta_c"] ** efficiency_power)
                    alpha_prev = alpha.clone()
                    alpha.mul_(1 - eta).add_(beta, alpha=(eta / mu0) * coupling * (1.0 + coupling_gain))
                    beta.mul_(1 - eta).add_(alpha_prev, alpha=-(eta / mu0) * coupling * (1.0 + coupling_gain))
                else:
                    thermo = {"T_hot": 0.0, "T_cold": 0.0, "eta_c": 0.0}
                    coupling_gain = 0.0
                    coupling = grad.new_zeros(())

                alpha_norm = alpha.norm()
                beta_norm = beta.norm()
                if torch.isfinite(alpha_norm) and alpha_norm > max_field_norm:
                    alpha.mul_(max_field_norm / (alpha_norm + 1e-12))
                if torch.isfinite(beta_norm) and beta_norm > max_field_norm:
                    beta.mul_(max_field_norm / (beta_norm + 1e-12))

                energy = 0.5 * (alpha.pow(2) + beta.pow(2)).mean()
                if target_energy > 0.0 and torch.isfinite(energy) and energy > 0:
                    if energy < target_energy:
                        scale = math.sqrt(target_energy / (float(energy.item()) + 1e-12))
                        alpha.mul_(scale)
                        beta.mul_(scale)
                        energy = 0.5 * (alpha.pow(2) + beta.pow(2)).mean()

                thermo = two_temperature_efficiency(alpha, beta)
                efficiency_gain = max(min_efficiency, thermo["eta_c"] ** efficiency_power)
                topo_corr = torch.tanh(alpha - beta)
                topo_corr = efficiency_gain * bounded_topological_correction(topo_corr, adam_dir, max_topo_ratio)
                p.add_(adam_dir + w_topo * topo_corr, alpha=-lr)

                if track:
                    coupling_current = mean_abs_coupling_current(alpha, beta, grad)
                    adam_norm = float(adam_dir.norm().item()) if torch.isfinite(adam_dir.norm()) else 0.0
                    topo_norm = float(topo_corr.norm().item()) if torch.isfinite(topo_corr.norm()) else 0.0
                    topo_ratio = topo_norm / (adam_norm + 1e-12)
                    self.stats["energy"] += float(energy.item()) if torch.isfinite(energy) else 0.0
                    self.stats["alpha_norm"] += float(alpha.norm().item())
                    self.stats["beta_norm"] += float(beta.norm().item())
                    self.stats["coupling_current"] += coupling_current
                    self.stats["coupling"] += coupling_current
                    self.stats["topo_ratio"] += topo_ratio
                    self.stats["T_hot"] += thermo["T_hot"]
                    self.stats["T_cold"] += thermo["T_cold"]
                    self.stats["eta_c"] += efficiency_gain
                    self.stats["num_params"] += 1.0

        if self.stats["num_params"] > 0:
            for key in (
                "energy",
                "alpha_norm",
                "beta_norm",
                "coupling_current",
                "coupling",
                "topo_ratio",
                "T_hot",
                "T_cold",
                "eta_c",
            ):
                self.stats[key] /= self.stats["num_params"]

        return loss

    def field_metrics(self) -> dict[str, float]:
        total_energy = 0.0
        total_current = 0.0
        total_corr = 0.0
        total_alpha = 0.0
        total_beta = 0.0
        total_t_hot = 0.0
        total_t_cold = 0.0
        total_eta_c = 0.0
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
                thermo = two_temperature_efficiency(alpha, beta)
                total_t_hot += thermo["T_hot"]
                total_t_cold += thermo["T_cold"]
                total_eta_c += thermo["eta_c"]
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
                "T_hot": 0.0,
                "T_cold": 0.0,
                "eta_c": 0.0,
            }
        return {
            "energy": total_energy / count,
            "j_t": total_current / count,
            "alpha_norm": total_alpha / count,
            "beta_norm": total_beta / count,
            "alpha_beta_corr": total_corr / count,
            "T_hot": total_t_hot / count,
            "T_cold": total_t_cold / count,
            "eta_c": total_eta_c / count,
        }

    def get_field_stats(self) -> tuple[float, float, float]:
        metrics = self.field_metrics()
        return metrics["energy"], metrics["j_t"], metrics["alpha_beta_corr"]
