from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .stopping import ReconnectionStoppingRule
from .v2 import TopologicalAdamV2


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 1729
    epochs: int = 10
    num_samples: int = 1000
    input_dim: int = 28 * 28
    hidden_dim: int = 256
    num_classes: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    eta: float = 0.01
    w_topo: float = 0.01
    mu0: float = 1.0
    target_energy: float = 1.0
    deterministic_init: bool = True
    dataset_mode: str = "memorization"


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int = 28 * 28, hidden_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def make_synthetic_loader(config: ExperimentConfig) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    if config.dataset_mode == "clustered":
        centers = torch.randn(config.num_classes, config.input_dim, generator=generator) * 0.4
        labels = torch.randint(0, config.num_classes, (config.num_samples,), generator=generator)
        features = centers[labels] + 0.5 * torch.randn(config.num_samples, config.input_dim, generator=generator)
    else:
        features = torch.randn(config.num_samples, config.input_dim, generator=generator) * 0.5
        labels = torch.randint(0, config.num_classes, (config.num_samples,), generator=generator)
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, generator=generator)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: TopologicalAdamV2,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    step_loss: list[float] = []
    step_energy: list[float] = []
    step_j: list[float] = []
    step_corr: list[float] = []
    step_grad_norm: list[float] = []

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()

        grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm_sq += float(param.grad.norm().item()) ** 2
        optimizer.step()
        energy, j_t, corr = optimizer.get_field_stats()

        total_loss += float(loss.item())
        num_batches += 1
        step_loss.append(float(loss.item()))
        step_energy.append(float(energy))
        step_j.append(float(j_t))
        step_corr.append(float(corr))
        step_grad_norm.append(math.sqrt(grad_norm_sq))

    return {
        "loss": total_loss / max(1, num_batches),
        "E_t": float(np.mean(step_energy)) if step_energy else 0.0,
        "J_t": float(np.mean(step_j)) if step_j else 0.0,
        "alpha_beta_corr": float(np.mean(step_corr)) if step_corr else 0.0,
        "grad_norm": float(np.mean(step_grad_norm)) if step_grad_norm else 0.0,
        "step_loss": step_loss,
        "step_E": step_energy,
        "step_J": step_j,
    }


def run_instrumented_training(
    *,
    config: ExperimentConfig | None = None,
    use_topology: bool = True,
) -> dict[str, Any]:
    config = config or ExperimentConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = torch.device("cpu")
    model = SimpleMLP(config.input_dim, config.hidden_dim, config.num_classes).to(device)
    loader = make_synthetic_loader(config)
    criterion = nn.CrossEntropyLoss()
    optimizer = TopologicalAdamV2(
        model.parameters(),
        lr=config.lr,
        eta=config.eta if use_topology else 0.0,
        w_topo=config.w_topo if use_topology else 0.0,
        mu0=config.mu0,
        target_energy=config.target_energy if use_topology else 1e-3,
        deterministic_init=config.deterministic_init,
        track_stats=True,
    )

    history: list[dict[str, Any]] = []
    all_losses: list[float] = []
    all_j: list[float] = []
    stopper = ReconnectionStoppingRule(peak_ratio=0.3, absolute_threshold=1e-3, warmup_steps=3)
    stop_decision = None

    for epoch in range(config.epochs):
        metrics = _train_epoch(model, loader, optimizer, criterion, device)
        history.append(metrics)
        all_losses.extend(metrics["step_loss"])
        all_j.extend(metrics["step_J"])
        decision = stopper.update(step=epoch, j_t=metrics["J_t"], loss=metrics["loss"])
        if decision.should_stop and stop_decision is None:
            stop_decision = {
                "epoch": decision.step,
                "reason": decision.reason,
                "ratio_to_peak": decision.ratio_to_peak,
                "peak_j_t": decision.peak_j_t,
                "current_j_t": decision.current_j_t,
            }

    correlation = _pearson_corr(all_j, all_losses)
    return {
        "mode": "topological" if use_topology else "control",
        "config": config.__dict__,
        "history": history,
        "pearson_r_j_loss": correlation,
        "stop_decision": stop_decision,
        "final_loss": history[-1]["loss"],
        "final_j_t": history[-1]["J_t"],
    }


def run_comparison(config: ExperimentConfig | None = None) -> dict[str, Any]:
    config = config or ExperimentConfig()
    control = run_instrumented_training(config=config, use_topology=False)
    topo = run_instrumented_training(config=config, use_topology=True)
    return {"control": control, "topological": topo}


def print_comparison_report(results: dict[str, Any]) -> None:
    for label, run in results.items():
        title = "Topological Adam" if label == "topological" else "Control (w_topo=0, eta=0)"
        print("=" * 80)
        print(title)
        print("=" * 80)
        print("Epoch | Loss      | E_t       | J_t       | alpha_beta_corr | grad_norm")
        print("-" * 80)
        for epoch, row in enumerate(run["history"]):
            print(
                f"{epoch:5d} | {row['loss']:9.6f} | {row['E_t']:9.6f} | {row['J_t']:9.6f} | "
                f"{row['alpha_beta_corr']:15.6f} | {row['grad_norm']:9.6f}"
            )
        print()
        print(f"Pearson r(J_t, loss): {run['pearson_r_j_loss']:.6f}")
        if run["stop_decision"] is None:
            print("Recommended stop: no trigger in tested horizon")
        else:
            stop = run["stop_decision"]
            print(
                "Recommended stop: "
                f"epoch {stop['epoch']} ({stop['reason']}, ratio_to_peak={stop['ratio_to_peak']:.3f})"
            )
        print()
