"""Reference real-data training benchmark for reviewers.

This script is intentionally small and local: it trains a PyTorch MLP on the
scikit-learn digits dataset, which ships with scikit-learn and requires no
network download. It gives reviewers a concrete real-data training workflow
without turning the JOSS paper into a large optimizer study.

Default protocol:
  - optimizers: Adam, AdamW, TopologicalAdamV3, TopologicalAdamV4
  - learning-rate grid: 3e-4, 1e-3, 3e-3, 1e-2
  - tuning seeds: 0, 1, 2
  - fresh evaluation seeds: 5, 6, 7, 8, 9
  - metric: held-out cross-entropy and accuracy

Run:
  python examples/reference_training_benchmark.py

Quick smoke run:
  python examples/reference_training_benchmark.py --quick --out tmp/reference_training_quick.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from topological_adam import TopologicalAdamV3, TopologicalAdamV4


DEFAULT_OPTIMIZERS = ("Adam", "AdamW", "V3", "V4")
DEFAULT_LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2)
DEFAULT_TUNE_SEEDS = (0, 1, 2)
DEFAULT_FRESH_SEEDS = (5, 6, 7, 8, 9)
_SPLIT_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_names(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def make_optimizer(name: str, params, lr: float):
    params = list(params)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    if name == "V3":
        return TopologicalAdamV3(params, lr=lr)
    if name == "V4":
        return TopologicalAdamV4(params, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def load_digits_split(seed: int):
    """Return a deterministic standardized train/test split."""
    if seed in _SPLIT_CACHE:
        return _SPLIT_CACHE[seed]
    x, y = load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    x_train = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    x_test = torch.tensor((x_test - mean) / std, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    split = (x_train, y_train, x_test, y_test)
    _SPLIT_CACHE[seed] = split
    return split


def build_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


def train_once(
    optimizer_name: str,
    lr: float,
    seed: int,
    *,
    epochs: int,
    batch_size: int,
) -> dict[str, float | int]:
    """Train one model and return held-out metrics."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x_train, y_train, x_test, y_test = load_digits_split(seed)
    model = build_model(seed)
    optimizer = make_optimizer(optimizer_name, model.parameters(), lr)
    n_train = x_train.shape[0]

    start = time.perf_counter()
    for _ in range(epochs):
        permutation = torch.randperm(n_train, generator=generator)
        for start_idx in range(0, n_train, batch_size):
            idx = permutation[start_idx : start_idx + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        logits = model(x_test)
        loss = F.cross_entropy(logits, y_test)
        accuracy = (logits.argmax(dim=1) == y_test).float().mean()

    return {
        "seed": seed,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "seconds": elapsed,
    }


def summarize(rows: list[dict[str, float | int]]) -> dict[str, object]:
    """Summarize per-seed benchmark rows."""
    losses = [float(row["loss"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    seconds = [float(row["seconds"]) for row in rows]
    return {
        "n": len(rows),
        "loss_mean": statistics.mean(losses),
        "loss_median": statistics.median(losses),
        "loss_std": statistics.pstdev(losses) if len(losses) > 1 else 0.0,
        "accuracy_mean": statistics.mean(accuracies),
        "accuracy_median": statistics.median(accuracies),
        "accuracy_std": statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0,
        "seconds_mean": statistics.mean(seconds),
        "per_seed": rows,
    }


def run_benchmark(
    *,
    optimizer_names: tuple[str, ...] = DEFAULT_OPTIMIZERS,
    lr_grid: tuple[float, ...] = DEFAULT_LR_GRID,
    tune_seeds: tuple[int, ...] = DEFAULT_TUNE_SEEDS,
    fresh_seeds: tuple[int, ...] = DEFAULT_FRESH_SEEDS,
    epochs: int = 12,
    batch_size: int = 64,
) -> dict[str, object]:
    """Run the full tune-then-fresh benchmark protocol."""
    payload: dict[str, object] = {
        "benchmark": "sklearn_digits_mlp",
        "dataset": "sklearn.datasets.load_digits",
        "model": "Linear(64,64)-ReLU-Linear(64,10)",
        "metric": "held-out cross-entropy and accuracy",
        "protocol": {
            "optimizers": list(optimizer_names),
            "lr_grid": list(lr_grid),
            "tune_seeds": list(tune_seeds),
            "fresh_seeds": list(fresh_seeds),
            "epochs": epochs,
            "batch_size": batch_size,
            "adamw_weight_decay": 1e-2,
            "lr_selection": "lowest mean held-out cross-entropy on tuning seeds",
        },
        "optimizers": {},
    }

    optimizer_results = {}
    for name in optimizer_names:
        print(f"\n== {name} ==", flush=True)
        tuning: dict[str, object] = {}
        best_lr = None
        best_loss = float("inf")
        for lr in lr_grid:
            rows = [
                train_once(name, lr, seed, epochs=epochs, batch_size=batch_size)
                for seed in tune_seeds
            ]
            summary = summarize(rows)
            tuning[f"{lr:g}"] = summary
            mean_loss = float(summary["loss_mean"])
            print(
                f"tune lr={lr:g}: loss={mean_loss:.4f}, "
                f"acc={float(summary['accuracy_mean']):.4f}",
                flush=True,
            )
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_lr = lr

        assert best_lr is not None
        fresh_rows = [
            train_once(name, best_lr, seed, epochs=epochs, batch_size=batch_size)
            for seed in fresh_seeds
        ]
        fresh = summarize(fresh_rows)
        optimizer_results[name] = {
            "best_lr": best_lr,
            "tuning": tuning,
            "fresh": fresh,
        }
        print(
            f"fresh lr={best_lr:g}: loss median={float(fresh['loss_median']):.4f}, "
            f"acc median={float(fresh['accuracy_median']):.4f}",
            flush=True,
        )

    if "Adam" in optimizer_results:
        adam_rows = {
            int(row["seed"]): row
            for row in optimizer_results["Adam"]["fresh"]["per_seed"]
        }
        comparisons: dict[str, object] = {}
        for name, result in optimizer_results.items():
            if name == "Adam":
                continue
            rows = result["fresh"]["per_seed"]
            paired = []
            for row in rows:
                seed = int(row["seed"])
                adam = adam_rows[seed]
                paired.append(
                    {
                        "seed": seed,
                        "loss_delta_vs_adam": float(adam["loss"]) - float(row["loss"]),
                        "accuracy_delta_vs_adam": float(row["accuracy"])
                        - float(adam["accuracy"]),
                    }
                )
            comparisons[f"{name}_vs_Adam"] = {
                "positive_loss_delta_favors_optimizer": True,
                "optimizer_loss_wins": sum(
                    1 for row in paired if row["loss_delta_vs_adam"] > 0
                ),
                "optimizer_accuracy_wins": sum(
                    1 for row in paired if row["accuracy_delta_vs_adam"] > 0
                ),
                "n": len(paired),
                "paired": paired,
            }
        payload["comparisons"] = comparisons

    payload["optimizers"] = optimizer_results
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reference_training_results.json")
    parser.add_argument("--quick", action="store_true", help="run a short smoke protocol")
    parser.add_argument("--optimizers", default=",".join(DEFAULT_OPTIMIZERS))
    parser.add_argument("--lr-grid", default=",".join(f"{lr:g}" for lr in DEFAULT_LR_GRID))
    parser.add_argument("--tune-seeds", default=",".join(str(s) for s in DEFAULT_TUNE_SEEDS))
    parser.add_argument("--fresh-seeds", default=",".join(str(s) for s in DEFAULT_FRESH_SEEDS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    optimizers = _parse_names(args.optimizers)
    lr_grid = _parse_floats(args.lr_grid)
    tune_seeds = _parse_ints(args.tune_seeds)
    fresh_seeds = _parse_ints(args.fresh_seeds)
    epochs = args.epochs

    if args.quick:
        lr_grid = (1e-3, 3e-3)
        tune_seeds = (0,)
        fresh_seeds = (5,)
        epochs = min(epochs, 3)

    payload = run_benchmark(
        optimizer_names=optimizers,
        lr_grid=lr_grid,
        tune_seeds=tune_seeds,
        fresh_seeds=fresh_seeds,
        epochs=epochs,
        batch_size=args.batch_size,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
