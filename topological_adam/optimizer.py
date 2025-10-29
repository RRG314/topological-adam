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
