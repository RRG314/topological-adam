from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .sds import TopologicalAdamSDS
from .v2 import TopologicalAdamV2


@dataclass(frozen=True)
class BenchmarkSummary:
    task: str
    optimizer: str
    mean_loss: float
    std_loss: float
    mean_accuracy: float | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "task": self.task,
            "optimizer": self.optimizer,
            "mean_loss": self.mean_loss,
            "std_loss": self.std_loss,
        }
        if self.mean_accuracy is not None:
            payload["mean_accuracy"] = self.mean_accuracy
        return payload


def _build_optimizer(name: str, params, lr: float):
    if name == "Adam":
        return optim.Adam(params, lr=lr)
    if name == "TopologicalAdamV2":
        return TopologicalAdamV2(params, lr=lr, deterministic_init=True, track_stats=True)
    if name == "TopologicalAdamSDS":
        return TopologicalAdamSDS(params, lr=lr, deterministic_init=True, track_stats=True)
    raise ValueError(f"Unknown optimizer benchmark target: {name}")


def _quadratic(seed: int, opt_name: str) -> dict[str, float]:
    torch.manual_seed(seed)
    params = nn.Parameter(torch.tensor([0.0, 0.0]))
    optimizer = _build_optimizer(opt_name, [params], lr=0.05)
    for _ in range(200):
        optimizer.zero_grad()
        loss = (params[0] - 3.0) ** 2 + (params[1] + 2.0) ** 2
        loss.backward()
        optimizer.step()
    return {"loss": float(loss.item())}


def _linear(seed: int, opt_name: str) -> dict[str, float]:
    torch.manual_seed(seed)
    x = torch.randn(128, 4)
    weights = torch.tensor([1.5, -2.0, 0.5, 3.0])
    y = x @ weights + 0.1 * torch.randn(128)
    model = nn.Linear(4, 1, bias=False)
    optimizer = _build_optimizer(opt_name, model.parameters(), lr=0.02)
    for _ in range(200):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x).squeeze(), y)
        loss.backward()
        optimizer.step()
    return {"loss": float(loss.item())}


def _xor(seed: int, opt_name: str) -> dict[str, float]:
    torch.manual_seed(seed)
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1), nn.Sigmoid())
    optimizer = _build_optimizer(opt_name, model.parameters(), lr=0.05)
    for _ in range(500):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = (model(x) > 0.5).float()
        acc = float((preds == y).float().mean().item())
        loss = float(F.binary_cross_entropy(model(x), y).item())
    return {"loss": loss, "accuracy": acc}


def _clustered(seed: int, opt_name: str) -> dict[str, float]:
    torch.manual_seed(seed)
    centers = torch.tensor([[1.5, 1.5], [1.5, -1.5], [-1.5, 1.5], [-1.5, -1.5]])
    labels = torch.arange(4).repeat_interleave(64)
    x = centers[labels] + 0.4 * torch.randn(labels.numel(), 2)
    model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 4))
    optimizer = _build_optimizer(opt_name, model.parameters(), lr=0.01)
    for _ in range(150):
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(x)
        loss = float(F.cross_entropy(logits, labels).item())
        acc = float((logits.argmax(dim=1) == labels).float().mean().item())
    return {"loss": loss, "accuracy": acc}


def run_candidate_benchmarks(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict[str, Any]:
    tasks: dict[str, Callable[[int, str], dict[str, float]]] = {
        "quadratic": _quadratic,
        "linear_regression": _linear,
        "xor": _xor,
        "clustered_classification": _clustered,
    }
    optimizers = ["Adam", "TopologicalAdamV2", "TopologicalAdamSDS"]
    results: list[dict[str, Any]] = []
    for task_name, task_fn in tasks.items():
        for opt_name in optimizers:
            rows = [task_fn(seed, opt_name) for seed in seeds]
            summary = BenchmarkSummary(
                task=task_name,
                optimizer=opt_name,
                mean_loss=mean(row["loss"] for row in rows),
                std_loss=pstdev(row["loss"] for row in rows),
                mean_accuracy=mean(row["accuracy"] for row in rows) if "accuracy" in rows[0] else None,
            )
            results.append(summary.as_dict())
    return {"seeds": list(seeds), "results": results}


def benchmark_report_json(seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> str:
    return json.dumps(run_candidate_benchmarks(seeds=seeds), indent=2)
