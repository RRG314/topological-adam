from __future__ import annotations

import torch


def deterministic_field_like(tensor: torch.Tensor, scale: float, phase: float) -> torch.Tensor:
    """Create a deterministic field pattern tied to tensor values and indices."""
    flat = tensor.detach().reshape(-1).to(dtype=torch.float32)
    idx = torch.arange(flat.numel(), device=tensor.device, dtype=torch.float32)
    signal = torch.sin(idx * 0.173 + flat * 0.37 + phase)
    return (signal.reshape_as(tensor) * float(scale)).to(dtype=tensor.dtype)


def centered_correlation(alpha: torch.Tensor, beta: torch.Tensor) -> float:
    a = alpha.detach().reshape(-1).float()
    b = beta.detach().reshape(-1).float()
    if a.numel() < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if not torch.isfinite(denom) or denom <= 1e-12:
        return 0.0
    return float((a * b).sum().item() / (denom.item() + 1e-12))


def mean_abs_coupling_current(alpha: torch.Tensor, beta: torch.Tensor, grad: torch.Tensor) -> float:
    current = ((alpha - beta) * grad).abs().mean()
    if not torch.isfinite(current):
        return 0.0
    return float(current.item())


def directional_current(alpha: torch.Tensor, beta: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    g_norm = grad.norm()
    if not torch.isfinite(g_norm) or g_norm <= 1e-12:
        return grad.new_zeros(())
    g_dir = grad / (g_norm + 1e-12)
    return (alpha * g_dir).sum() - (beta * g_dir).sum()


def bounded_topological_correction(
    topo_corr: torch.Tensor,
    adam_dir: torch.Tensor,
    max_ratio: float | None,
) -> torch.Tensor:
    if max_ratio is None:
        return topo_corr
    topo_norm = topo_corr.norm()
    adam_norm = adam_dir.norm()
    if not torch.isfinite(topo_norm) or not torch.isfinite(adam_norm):
        return torch.zeros_like(topo_corr)
    if topo_norm <= 0 or adam_norm <= 0 or max_ratio <= 0:
        return torch.zeros_like(topo_corr)
    scale = min(1.0, float(max_ratio) * float(adam_norm.item()) / (float(topo_norm.item()) + 1e-12))
    return topo_corr * scale


def two_temperature_efficiency(alpha: torch.Tensor, beta: torch.Tensor) -> dict[str, float]:
    """Compute a bounded two-temperature efficiency from the auxiliary fields."""
    alpha_rms = float(alpha.detach().float().pow(2).mean().sqrt().item())
    beta_rms = float(beta.detach().float().pow(2).mean().sqrt().item())
    t_hot = max(alpha_rms, beta_rms)
    t_cold = min(alpha_rms, beta_rms)
    if t_hot <= 1e-12:
        eta_c = 0.0
    else:
        eta_c = max(0.0, min(1.0, 1.0 - t_cold / (t_hot + 1e-12)))
    return {
        "T_hot": t_hot,
        "T_cold": t_cold,
        "eta_c": eta_c,
    }
