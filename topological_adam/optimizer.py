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
# Topological Adam v2 (Clean, Stable)
# - Adam base + topological correction field tanh(alpha - beta)
# - alpha/beta J-coupled dynamics with energy targeting
# - Stability guards: NaN/Inf protection, field norm clamp, safe scaling
# ============================================================


class TopologicalAdamV2(torch.optim.Optimizer):
    """
    Clean Topological Adam v2.

    Params:
      lr: Adam learning rate.
      betas: Adam (beta1, beta2).
      eps: Adam epsilon.

      eta: coupling step size for alpha/beta dynamics.
      mu0: coupling stiffness (larger => slower coupling).
      w_topo: weight of topo correction added to Adam direction.

      field_init_scale: initial scale for alpha/beta fields.
      target_energy: target mean energy for fields (mean((a^2+b^2)/2)).

      energy_floor: if energy drops below this, rescale up.
      energy_ceiling_mult: if energy exceeds target_energy * mult, damp fields.
      max_field_norm: clamp alpha/beta norms to avoid runaway.

      topo_clip: clamp topo correction magnitude elementwise.
      grad_norm_floor: if grad norm < this, skip field dynamics for that param.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        eta=0.02,
        mu0=0.5,
        w_topo=0.15,
        field_init_scale=0.01,
        target_energy=1e-3,
        energy_floor=1e-12,
        energy_ceiling_mult=10.0,
        max_field_norm=10.0,
        topo_clip=1.0,
        grad_norm_floor=1e-12,
    ):
        if lr <= 0:
            raise ValueError("lr must be > 0")
        if eps <= 0:
            raise ValueError("eps must be > 0")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError("betas must be in [0,1)")
        if mu0 <= 0:
            raise ValueError("mu0 must be > 0")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            eta=eta,
            mu0=mu0,
            w_topo=w_topo,
            field_init_scale=field_init_scale,
            target_energy=target_energy,
            energy_floor=energy_floor,
            energy_ceiling_mult=energy_ceiling_mult,
            max_field_norm=max_field_norm,
            topo_clip=topo_clip,
            grad_norm_floor=grad_norm_floor,
        )
        super().__init__(params, defaults)

        # optional diagnostics (global per step)
        self._last_energy = 0.0
        self._last_J_mean_abs = 0.0
        self._last_alpha_norm = 0.0
        self._last_beta_norm = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # reset diagnostics
        energy_sum = 0.0
        J_abs_sum = 0.0
        J_count = 0
        a_norm_sum = 0.0
        b_norm_sum = 0.0

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]

            eta = group["eta"]
            mu0 = group["mu0"]
            w_topo = group["w_topo"]

            field_init_scale = group["field_init_scale"]
            target_energy = group["target_energy"]
            energy_floor = group["energy_floor"]
            energy_ceiling_mult = group["energy_ceiling_mult"]
            max_field_norm = group["max_field_norm"]
            topo_clip = group["topo_clip"]
            grad_norm_floor = group["grad_norm_floor"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if not torch.isfinite(g).all():
                    # Skip update if gradient has NaNs/Infs
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # alpha/beta init scaled by param size
                    std = field_init_scale * (2.0 / max(1, p.numel())) ** 0.5
                    state["alpha"] = torch.randn_like(p) * (3.0 * std)
                    state["beta"]  = torch.randn_like(p) * (1.0 * std)

                state["step"] += 1
                t = state["step"]

                m = state["m"]
                v = state["v"]
                a = state["alpha"]
                b = state["beta"]

                # --------------------------
                # Adam base moments
                # --------------------------
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)

                # Bias correction
                m_hat = m / (1 - b1 ** t)
                v_hat = v / (1 - b2 ** t)

                denom = v_hat.sqrt().add_(eps)
                adam_dir = m_hat / denom  # "direction" (like Adam step vector)

                # --------------------------
                # Topological field dynamics
                # --------------------------
                g_norm = g.norm()
                if torch.isfinite(g_norm) and g_norm > grad_norm_floor:
                    g_dir = g / (g_norm + 1e-12)

                    # J coupling = projection difference along gradient direction
                    j_alpha = (a * g_dir).sum()
                    j_beta  = (b * g_dir).sum()
                    J = j_alpha - j_beta

                    # coupled update (energy-like exchange)
                    a_prev = a.clone()
                    a.mul_(1 - eta).add_(b, alpha=(eta / mu0) * J)
                    b.mul_(1 - eta).add_(a_prev, alpha=-(eta / mu0) * J)

                    # energy targeting (mean energy density)
                    energy_local = 0.5 * (a.pow(2).mean() + b.pow(2).mean()).item()

                    # if too low: rescale up toward target
                    if energy_local < target_energy and target_energy > 0:
                        scale = math.sqrt(target_energy / max(energy_local, energy_floor))
                        a.mul_(scale)
                        b.mul_(scale)

                    # if too high: damp down
                    if energy_local > target_energy * energy_ceiling_mult:
                        a.mul_(0.9)
                        b.mul_(0.9)

                    # field norm clamp
                    a_norm = a.norm().item()
                    if a_norm > max_field_norm:
                        a.mul_(max_field_norm / (a.norm() + 1e-12))

                    b_norm = b.norm().item()
                    if b_norm > max_field_norm:
                        b.mul_(max_field_norm / (b.norm() + 1e-12))

                    # diagnostics
                    energy_sum += energy_local
                    J_abs_sum += float(J.abs().item())
                    J_count += 1
                    a_norm_sum += float(a.norm().item())
                    b_norm_sum += float(b.norm().item())

                    topo_corr = torch.tanh(a - b)
                    if topo_clip is not None:
                        topo_corr = torch.clamp(topo_corr, -topo_clip, topo_clip)
                else:
                    topo_corr = torch.zeros_like(p)

                # --------------------------
                # Final parameter update
                # --------------------------
                # p <- p - lr*(adam_dir + w_topo*topo_corr)
                p.add_(adam_dir + w_topo * topo_corr, alpha=-lr)

        # store diagnostics
        self._last_energy = energy_sum
        self._last_J_mean_abs = (J_abs_sum / max(1, J_count))
        self._last_alpha_norm = a_norm_sum
        self._last_beta_norm = b_norm_sum

        return loss

    # --- optional diagnostics accessors ---
    def last_energy(self):
        return self._last_energy

    def last_J_mean_abs(self):
        return self._last_J_mean_abs

    def last_alpha_norm(self):
        return self._last_alpha_norm

    def last_beta_norm(self):
        return self._last_beta_norm
