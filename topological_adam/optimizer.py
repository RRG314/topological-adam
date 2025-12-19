# ============================================================
# Energy-Stabilized Topological Adam Optimizer
# ============================================================

import math
import torch

class TopologicalAdam(torch.optim.Optimizer):
    """Energy-Stabilized Topological Adam Optimizer"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 eta=0.02, mu0=0.5, w_topo=0.15, field_init_scale=0.01,
                 target_energy=1e-3):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        eta=eta, mu0=mu0, w_topo=w_topo,
                        field_init_scale=field_init_scale,
                        target_energy=target_energy)
        super().__init__(params, defaults)
        self._energy = 0.0
        self._J_accum = 0.0
        self._J_count = 0
        self._alpha_norm = 0.0
        self._beta_norm = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure else None
        self._energy = 0.0
        self._J_accum = 0.0
        self._J_count = 0
        self._alpha_norm = 0.0
        self._beta_norm = 0.0

        for group in self.param_groups:
            lr, (b1, b2), eps = group['lr'], group['betas'], group['eps']
            eta, mu0, w_topo, field_init_scale, target_energy = (
                group['eta'], group['mu0'], group['w_topo'],
                group['field_init_scale'], group['target_energy']
            )

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p, device=p.device)
                    state['v'] = torch.zeros_like(p, device=p.device)
                    std = field_init_scale * (2.0 / p.numel()) ** 0.5
                    state['alpha'] = torch.randn_like(p, device=p.device) * std * 3.0
                    state['beta'] = torch.randn_like(p, device=p.device) * std * 1.0

                state['step'] += 1
                m, v, a, b = state['m'], state['v'], state['alpha'], state['beta']

                # Adam base update
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1 ** state['step'])
                v_hat = v / (1 - b2 ** state['step'])
                adam_dir = m_hat / (v_hat.sqrt() + eps)

                g_norm = g.norm()
                if torch.isfinite(g_norm) and g_norm > 1e-12:
                    g_dir = g / (g_norm + 1e-12)
                    j_alpha = (a * g_dir).sum()
                    j_beta = (b * g_dir).sum()
                    J = j_alpha - j_beta
                    a_prev = a.clone()

                    a.mul_(1 - eta).add_(b, alpha=(eta / mu0) * J)
                    b.mul_(1 - eta).add_(a_prev, alpha=-(eta / mu0) * J)

                    energy_local = 0.5 * ((a**2 + b**2).mean()).item()
                    if energy_local < target_energy:
                        scale = math.sqrt(target_energy / (energy_local + 1e-12))
                        a.mul_(scale)
                        b.mul_(scale)
                    elif energy_local > target_energy * 10:
                        a.mul_(0.9)
                        b.mul_(0.9)

                    topo_corr = torch.tanh(a - b)
                    self._energy += energy_local
                    self._J_accum += float(abs(J))
                    self._J_count += 1
                    self._alpha_norm += a.norm().item()
                    self._beta_norm += b.norm().item()
                else:
                    topo_corr = torch.zeros_like(p)

                p.add_(adam_dir + w_topo * topo_corr, alpha=-lr)
        return loss

    def energy(self):
        return self._energy

    def J_mean_abs(self):
        return self._J_accum / max(1, self._J_count)


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
        track_stats=True,
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
            track_stats=track_stats,
        )
        super().__init__(params, defaults)

        # Global stats (optional)
        self.stats = {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

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
            track = group["track_stats"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # --- Init ---
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["alpha"] = torch.randn_like(p) * group["field_init_scale"]
                    state["beta"] = torch.randn_like(p) * group["field_init_scale"]

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
                g_norm = g.norm() + 1e-12
                g_dir = g / g_norm

                J = (alpha * g_dir).sum() - (beta * g_dir).sum()
                alpha_prev = alpha.clone()

                alpha.mul_(1 - eta).add_(beta, alpha=(eta / mu0) * J)
                beta.mul_(1 - eta).add_(alpha_prev, alpha=-(eta / mu0) * J)

                # --- Field constraints ---
                if alpha.norm() > max_field_norm:
                    alpha.mul_(max_field_norm / (alpha.norm() + 1e-12))
                if beta.norm() > max_field_norm:
                    beta.mul_(max_field_norm / (beta.norm() + 1e-12))

                energy = 0.5 * (alpha.pow(2) + beta.pow(2)).mean()
                if energy < target_energy:
                    scale = math.sqrt(target_energy / (energy + 1e-12))
                    alpha.mul_(scale)
                    beta.mul_(scale)

                topo_corr = torch.tanh(alpha - beta)

                # --- Update ---
                p.add_(adam_dir + w_topo * topo_corr, alpha=-lr)

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
