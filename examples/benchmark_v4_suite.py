"""Honest benchmark of TopologicalAdamV4 against tuned Adam and V3.

Protocol (same spirit as the V3 audit):
- every optimizer, including Adam, gets its learning rate tuned per task on
  TUNE_SEEDS over the same grid;
- evaluation uses FRESH_SEEDS never seen during tuning;
- paired per-seed statistics are reported;
- every task is labeled [synthetic] or [real data];
- results are written to benchmark_v4_results.json verbatim, wins and losses.

V4 targets a specific regime: trajectories with loops/oscillation
(ill-conditioned or high-lr dynamics). On ordinary stochastic tasks the
expected honest result is parity with Adam, not gains.

Run:  python benchmark_v4_suite.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topological_adam import TopologicalAdamV3, TopologicalAdamV4  # noqa: E402

LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
TUNE_SEEDS = (0, 1, 2)
FRESH_SEEDS = (5, 6, 7, 8, 9, 10, 11, 12)


def make_opt(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "v3":
        return TopologicalAdamV3(params, lr=lr)
    if name == "v4":
        return TopologicalAdamV4(params, lr=lr)
    raise ValueError(name)


# ----------------------------------------------------------------------
# Tasks. Each returns a scalar "final metric" (lower is better).
# ----------------------------------------------------------------------

def task_stiff_quadratic(opt_name, lr, seed, steps=300):
    """[synthetic] Oscillation-prone stiff quadratic (V4's target regime)."""
    torch.manual_seed(seed)
    scales = torch.tensor([1.0, 30.0, 100.0, 300.0])
    p = (torch.randn(4) * 2.0 + 1.0).requires_grad_(True)
    opt = make_opt(opt_name, [p], lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = (scales * p ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float((scales * p ** 2).sum())


def task_rotating_field(opt_name, lr, seed, steps=400):
    """[synthetic] Quadratic + rotational gradient component (loops)."""
    torch.manual_seed(seed)
    p = (torch.randn(2) * 2.0).requires_grad_(True)
    opt = make_opt(opt_name, [p], lr)
    rot = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            g = 2.0 * p + 3.0 * (rot @ p.detach())  # non-conservative field
        p.grad = g
        opt.step()
    with torch.no_grad():
        return float((p ** 2).sum())


def _digits_data(seed):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-6] = 1.0  # zero-variance pixels: don't divide by ~0
    Xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Xte = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    return Xtr, Xte, torch.tensor(ytr), torch.tensor(yte)


def task_digits_mlp(opt_name, lr, seed, epochs=12, batch=64):
    """[real data] sklearn digits, small MLP, held-out cross-entropy."""
    torch.manual_seed(seed)
    Xtr, Xte, ytr, yte = _digits_data(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10)
    )
    opt = make_opt(opt_name, model.parameters(), lr)
    lossf = torch.nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return float(lossf(model(Xte), yte))


def task_teacher_student(opt_name, lr, seed, steps=400):
    """[synthetic] Teacher-student regression, held-out MSE."""
    torch.manual_seed(seed)
    teacher = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)
    )
    for q in teacher.parameters():
        q.requires_grad_(False)
    student = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)
    )
    opt = make_opt(opt_name, student.parameters(), lr)
    Xte = torch.randn(512, 16)
    for _ in range(steps):
        X = torch.randn(128, 16)
        opt.zero_grad(set_to_none=True)
        loss = ((student(X) - teacher(X)) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((student(Xte) - teacher(Xte)) ** 2).mean())


TASKS = {
    "stiff_quadratic [synthetic]": task_stiff_quadratic,
    "rotating_field [synthetic]": task_rotating_field,
    "digits_mlp [real data]": task_digits_mlp,
    "teacher_student [synthetic]": task_teacher_student,
}
OPTS = ("adam", "v3", "v4")


def tune(task_fn, opt_name):
    """Pick the lr with the best MEAN over tuning seeds.

    Mean (not median) so that an lr that diverges on any tuning seed is
    heavily penalized — a median criterion can select an unstable lr that
    looks fine on 2 of 3 seeds and then explodes on fresh seeds.
    """
    best_lr, best = None, math.inf
    for lr in LR_GRID:
        vals = [task_fn(opt_name, lr, s) for s in TUNE_SEEDS]
        mean = sum(vals) / len(vals)
        if mean < best:
            best, best_lr = mean, lr
    return best_lr


def paired_t(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    if sd == 0.0:
        return 0.0
    return mean / (sd / math.sqrt(n))


def main():
    results = {}
    for tname, fn in TASKS.items():
        print(f"\n=== {tname} ===")
        entry = {"tuned_lr": {}, "fresh": {}}
        for opt in OPTS:
            lr = tune(fn, opt)
            entry["tuned_lr"][opt] = lr
            vals = [fn(opt, lr, s) for s in FRESH_SEEDS]
            entry["fresh"][opt] = vals
            med = statistics.median(vals)
            print(f"  {opt:>5}: tuned lr={lr:.0e}  fresh median={med:.4g}")
        t_adam = paired_t(entry["fresh"]["adam"], entry["fresh"]["v4"])
        wins = sum(
            1
            for a, b in zip(entry["fresh"]["adam"], entry["fresh"]["v4"])
            if b < a
        )
        entry["v4_vs_adam"] = {
            "paired_t": t_adam,
            "v4_wins_of": [wins, len(FRESH_SEEDS)],
        }
        print(
            f"  V4 vs tuned Adam: paired t={t_adam:+.2f} "
            f"(positive favors V4), per-seed wins {wins}/{len(FRESH_SEEDS)}"
        )
        results[tname] = entry

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "benchmark_v4_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
