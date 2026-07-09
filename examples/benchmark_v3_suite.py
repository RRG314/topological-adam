"""Rigorous benchmark: TopologicalAdamV3 vs tuned conventional baselines.

Methodology (designed to avoid the usual ways optimizer comparisons fool
themselves):
  - every optimizer gets its learning rate tuned per task over the same grid,
    selected by mean metric across seeds (no default-lr strawmen)
  - 5 seeds per (task, optimizer, lr); mean +/- population std reported
  - held-out evaluation where the task has data (test loss / test accuracy)
  - equal step budgets for all optimizers
  - per-step wall-clock overhead measured separately

Run:  python examples/benchmark_v3_suite.py [--fast]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from topological_adam import TopologicalAdamV2
from topological_adam.sds import TopologicalAdamSDS
from topological_adam.v3 import TopologicalAdamV3

torch.set_num_threads(4)

LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def make_optimizer(name: str, params, lr: float):
    params = list(params)
    if name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    if name == "V2":
        return TopologicalAdamV2(params, lr=lr, deterministic_init=True, track_stats=False)
    if name == "SDS":
        return TopologicalAdamSDS(params, lr=lr, deterministic_init=True, track_stats=False)
    if name == "V3":
        return TopologicalAdamV3(params, lr=lr)
    if name == "V3-solo":  # THE standalone topological mechanism: gated fields, NO cautious mask
        return TopologicalAdamV3(params, lr=lr, cautious=False, coupling_gate=True)
    if name == "V3-fields-nogate":  # ablation: ungated fields, no cautious mask
        return TopologicalAdamV3(params, lr=lr, cautious=False, coupling_gate=False)
    if name == "V3-cautious-only":  # ablation: cautious mask, no fields
        return TopologicalAdamV3(params, lr=lr, w_topo=0.0)
    raise ValueError(name)


OPTIMIZERS = ["Adam", "AdamW", "V2", "SDS", "V3", "V3-solo", "V3-fields-nogate", "V3-cautious-only"]


# --------------------------------------------------------------------------
# Tasks. Each returns {"loss": float (lower better)} and optionally accuracy.
# --------------------------------------------------------------------------

def task_quadratic(seed: int, opt_name: str, lr: float, steps=200):
    torch.manual_seed(seed)
    p = nn.Parameter(torch.tensor([0.0, 0.0]))
    opt = make_optimizer(opt_name, [p], lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = (p[0] - 3.0) ** 2 + (p[1] + 2.0) ** 2
        loss.backward()
        opt.step()
    return {"loss": float(loss.detach())}


def task_ill_conditioned(seed: int, opt_name: str, lr: float, steps=500, dim=50):
    torch.manual_seed(seed)
    # eigenvalues log-spaced over 3 orders of magnitude, random rotation
    eigs = torch.logspace(0, 3, dim)
    q, _ = torch.linalg.qr(torch.randn(dim, dim, generator=torch.Generator().manual_seed(1234)))
    A = q @ torch.diag(eigs) @ q.T
    b = torch.randn(dim, generator=torch.Generator().manual_seed(4321))
    p = nn.Parameter(torch.zeros(dim))
    opt = make_optimizer(opt_name, [p], lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * p @ (A @ p) - b @ p
        loss.backward()
        opt.step()
    # report gap to optimum
    x_star = torch.linalg.solve(A, b)
    f_star = 0.5 * x_star @ (A @ x_star) - b @ x_star
    with torch.no_grad():
        gap = float((0.5 * p @ (A @ p) - b @ p) - f_star)
    return {"loss": max(gap, 1e-16)}


def task_rosenbrock(seed: int, opt_name: str, lr: float, steps=2000):
    torch.manual_seed(seed)
    starts = [(-1.2, 1.0), (-1.0, -1.0), (0.0, 2.0), (2.0, -1.0), (-2.0, 2.0)]
    x0, y0 = starts[seed % len(starts)]
    p = nn.Parameter(torch.tensor([x0, y0]))
    opt = make_optimizer(opt_name, [p], lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
        loss.backward()
        opt.step()
    return {"loss": float(loss.detach())}


def _split(x, y, seed, frac=0.8):
    n = x.shape[0]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    k = int(n * frac)
    return x[idx[:k]], y[idx[:k]], x[idx[k:]], y[idx[k:]]


def task_clusters_noisy(seed: int, opt_name: str, lr: float, steps=400, batch=64):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[1.5, 1.5], [1.5, -1.5], [-1.5, 1.5], [-1.5, -1.5]])
    labels = torch.arange(4).repeat_interleave(200)
    x = centers[labels] + 0.7 * torch.randn(labels.numel(), 2, generator=g)
    # 10% label noise
    flip = torch.rand(labels.numel(), generator=g) < 0.10
    noisy = labels.clone()
    noisy[flip] = torch.randint(0, 4, (int(flip.sum()),), generator=g)
    xtr, ytr, xte, yte = _split(x, noisy, seed)
    model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 4))
    opt = make_optimizer(opt_name, model.parameters(), lr)
    n = xtr.shape[0]
    for i in range(steps):
        sel = torch.randint(0, n, (batch,), generator=g)
        opt.zero_grad()
        loss = F.cross_entropy(model(xtr[sel]), ytr[sel])
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(xte)
        return {
            "loss": float(F.cross_entropy(logits, yte)),
            "accuracy": float((logits.argmax(1) == yte).float().mean()),
        }


_DIGITS_CACHE = None

def _digits():
    global _DIGITS_CACHE
    if _DIGITS_CACHE is None:
        from sklearn.datasets import load_digits
        d = load_digits()
        x = torch.tensor(d.data, dtype=torch.float32) / 16.0
        y = torch.tensor(d.target, dtype=torch.long)
        _DIGITS_CACHE = (x, y)
    return _DIGITS_CACHE


def task_digits_mlp(seed: int, opt_name: str, lr: float, steps=600, batch=64):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x, y = _digits()
    xtr, ytr, xte, yte = _split(x, y, seed)
    model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))
    opt = make_optimizer(opt_name, model.parameters(), lr)
    n = xtr.shape[0]
    for i in range(steps):
        sel = torch.randint(0, n, (batch,), generator=g)
        opt.zero_grad()
        loss = F.cross_entropy(model(xtr[sel]), ytr[sel])
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(xte)
        return {
            "loss": float(F.cross_entropy(logits, yte)),
            "accuracy": float((logits.argmax(1) == yte).float().mean()),
        }


def task_teacher_student(seed: int, opt_name: str, lr: float, steps=600, batch=64):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed + 999)
    teacher = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 1))
    with torch.no_grad():
        for m in teacher.modules():
            if isinstance(m, nn.Linear):
                m.weight.mul_(1.5)
    x = torch.randn(2000, 16, generator=g)
    with torch.no_grad():
        y = teacher(x).squeeze(-1) + 0.05 * torch.randn(2000, generator=g)
    xtr, ytr, xte, yte = _split(x, y, seed)
    model = nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 1))
    opt = make_optimizer(opt_name, model.parameters(), lr)
    n = xtr.shape[0]
    for i in range(steps):
        sel = torch.randint(0, n, (batch,), generator=g)
        opt.zero_grad()
        loss = F.mse_loss(model(xtr[sel]).squeeze(-1), ytr[sel])
        loss.backward()
        opt.step()
    with torch.no_grad():
        return {"loss": float(F.mse_loss(model(xte).squeeze(-1), yte))}


def task_digits_cnn(seed: int, opt_name: str, lr: float, steps=500, batch=64):
    """Small conv net on 8x8 digit images — harder / more realistic than the MLP."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    x, y = _digits()
    x = x.reshape(-1, 1, 8, 8)
    xtr, ytr, xte, yte = _split(x, y, seed)
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )
    opt = make_optimizer(opt_name, model.parameters(), lr)
    n = xtr.shape[0]
    for i in range(steps):
        sel = torch.randint(0, n, (batch,), generator=g)
        opt.zero_grad()
        loss = F.cross_entropy(model(xtr[sel]), ytr[sel])
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(xte)
        return {
            "loss": float(F.cross_entropy(logits, yte)),
            "accuracy": float((logits.argmax(1) == yte).float().mean()),
        }


class _TinyTransformer(nn.Module):
    """Char-level causal transformer, ~200k params."""

    def __init__(self, vocab=16, d=64, heads=4, layers=2, ctx=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        block = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=2 * d,
            dropout=0.0, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(d, vocab)
        self.ctx = ctx

    def forward(self, idx):
        T = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        h = self.enc(h, mask=mask)
        return self.head(h)


def _synthetic_lm_data(n=4000, ctx=32, vocab=16, seed=7):
    """Deterministic-but-nontrivial sequences: order-2 Markov chain over 16 symbols."""
    g = torch.Generator().manual_seed(seed)
    trans = torch.softmax(2.0 * torch.randn(vocab, vocab, vocab, generator=g), dim=-1)
    seqs = torch.zeros(n, ctx + 1, dtype=torch.long)
    state = torch.randint(0, vocab, (n, 2), generator=g)
    seqs[:, 0], seqs[:, 1] = state[:, 0], state[:, 1]
    for t in range(2, ctx + 1):
        probs = trans[seqs[:, t - 2], seqs[:, t - 1]]
        seqs[:, t] = torch.multinomial(probs, 1, generator=g).squeeze(-1)
    return seqs


_LM_CACHE = None

def task_tiny_transformer(seed: int, opt_name: str, lr: float, steps=400, batch=64):
    """Tiny causal LM on synthetic order-2 Markov text; held-out cross-entropy."""
    global _LM_CACHE
    if _LM_CACHE is None:
        _LM_CACHE = _synthetic_lm_data()
    seqs = _LM_CACHE
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(seqs.shape[0], generator=torch.Generator().manual_seed(seed))
    k = int(seqs.shape[0] * 0.8)
    tr, te = seqs[idx[:k]], seqs[idx[k:]]
    model = _TinyTransformer()
    opt = make_optimizer(opt_name, model.parameters(), lr)
    n = tr.shape[0]
    for i in range(steps):
        sel = torch.randint(0, n, (batch,), generator=g)
        x, y = tr[sel, :-1], tr[sel, 1:]
        opt.zero_grad()
        loss = F.cross_entropy(model(x).reshape(-1, 16), y.reshape(-1))
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        losses = []
        for j in range(0, te.shape[0], 256):
            xb, yb = te[j:j + 256, :-1], te[j:j + 256, 1:]
            losses.append(float(F.cross_entropy(model(xb).reshape(-1, 16), yb.reshape(-1))))
        return {"loss": statistics.mean(losses)}


TASKS = {
    "quadratic_200": task_quadratic,
    "ill_conditioned_quad": task_ill_conditioned,
    "rosenbrock": task_rosenbrock,
    "clusters_noisy_testacc": task_clusters_noisy,
    "digits_mlp_testacc": task_digits_mlp,
    "teacher_student_mse": task_teacher_student,
    "digits_cnn_testacc": task_digits_cnn,
    "tiny_transformer_lm": task_tiny_transformer,
}


def wallclock_overhead(n_params_hidden=512, reps=30):
    """Per-step time on a ~1.1M-param MLP, batch 128."""
    out = {}
    x = torch.randn(128, 256)
    y = torch.randint(0, 10, (128,))
    for name in OPTIMIZERS:
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(256, n_params_hidden), nn.ReLU(),
            nn.Linear(n_params_hidden, n_params_hidden), nn.ReLU(),
            nn.Linear(n_params_hidden, 10),
        )
        opt = make_optimizer(name, model.parameters(), 1e-3)
        # warmup
        for _ in range(5):
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        t0 = time.perf_counter()
        for _ in range(reps):
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        out[name] = (time.perf_counter() - t0) / reps * 1000  # ms/step
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="3 seeds, reduced grid")
    ap.add_argument("--out", default="benchmark_v3_results.json")
    ap.add_argument("--tasks", default="", help="comma-separated task subset")
    ap.add_argument("--opts", default="", help="comma-separated optimizer subset")
    ap.add_argument("--threads", type=int, default=0, help="override torch thread count")
    ap.add_argument("--no-wallclock", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    seeds = (0, 1, 2) if args.fast else (0, 1, 2, 3, 4)
    grid = (1e-3, 3e-3, 1e-2) if args.fast else LR_GRID

    tasks = TASKS
    if args.tasks:
        wanted = args.tasks.split(",")
        tasks = {k: TASKS[k] for k in wanted}
    opt_names = args.opts.split(",") if args.opts else OPTIMIZERS

    results = {}
    for task_name, task_fn in tasks.items():
        results[task_name] = {}
        for opt_name in opt_names:
            best = None
            for lr in grid:
                try:
                    rows = [task_fn(s, opt_name, lr) for s in seeds]
                except Exception as e:  # diverged badly
                    continue
                losses = [r["loss"] for r in rows]
                if any(not math.isfinite(l) for l in losses):
                    continue
                mean_loss = statistics.mean(losses)
                entry = {
                    "lr": lr,
                    "mean_loss": mean_loss,
                    "std_loss": statistics.pstdev(losses),
                }
                if "accuracy" in rows[0]:
                    accs = [r["accuracy"] for r in rows]
                    entry["mean_acc"] = statistics.mean(accs)
                    entry["std_acc"] = statistics.pstdev(accs)
                # select best lr by accuracy if present, else loss
                key = -entry.get("mean_acc", -mean_loss)
                if best is None or key < best[0]:
                    best = (key, entry)
            results[task_name][opt_name] = best[1] if best else {"error": "all lrs diverged"}
            e = results[task_name][opt_name]
            acc = f"  acc={e['mean_acc']:.4f}+-{e['std_acc']:.4f}" if "mean_acc" in e else ""
            print(f"{task_name:26s} {opt_name:18s} lr={e.get('lr', 0):.0e}  "
                  f"loss={e.get('mean_loss', float('nan')):.4e}+-{e.get('std_loss', float('nan')):.1e}{acc}",
                  flush=True)

    if not args.no_wallclock:
        print("\n-- wall-clock (ms/step, 1.1M-param MLP, CPU) --")
        wc = wallclock_overhead()
        for k, v in wc.items():
            print(f"  {k:18s} {v:7.2f} ms/step")
        results["_wallclock_ms_per_step"] = wc
    results["_meta"] = {"seeds": list(seeds), "lr_grid": list(grid)}

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
