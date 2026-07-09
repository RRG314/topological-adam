"""Fresh-seed paired confirmation for the standalone topological mechanism.

Takes the per-(task, optimizer) learning rates tuned on seeds 0-4 by
benchmark_v3_suite.py, then re-runs the *stochastic* tasks on 10 fresh seeds
(5-14) that were never used for tuning.  Reports per-seed paired comparisons
and paired t statistics of V3-solo (gated fields, NO cautious mask) and V3
(full) against each baseline.

Run after the suite:  python examples/confirm_fresh_seeds.py \
    --results benchmark_v3_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_v3_suite import TASKS  # noqa: E402

FRESH_SEEDS = tuple(range(5, 15))

# stochastic tasks only: deterministic ones are seed-independent, already exact
CONFIRM_TASKS = (
    "teacher_student_mse",
    "digits_mlp_testacc",
    "digits_cnn_testacc",
    "clusters_noisy_testacc",
    "tiny_transformer_lm",
)

CANDIDATES = ("V3-solo", "V3")
BASELINES = ("Adam", "AdamW", "V2")


def paired_t(diffs):
    n = len(diffs)
    mean = statistics.mean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else float("nan")
    if sd == 0:
        return mean, float("inf") if mean != 0 else 0.0
    return mean, mean / (sd / math.sqrt(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="benchmark_v3_results.json")
    ap.add_argument("--out", default="fresh_seed_confirmation.json")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tasks", default="", help="comma-separated subset of CONFIRM_TASKS")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    confirm_tasks = args.tasks.split(",") if args.tasks else CONFIRM_TASKS

    with open(args.results) as f:
        tuned = json.load(f)

    out = {}
    for task in confirm_tasks:
        if task not in tuned:
            print(f"skip {task}: not in results")
            continue
        task_fn = TASKS[task]
        names = set(CANDIDATES) | set(BASELINES)
        runs = {}
        for name in names:
            entry = tuned[task].get(name, {})
            lr = entry.get("lr")
            if lr is None:
                print(f"skip {task}/{name}: no tuned lr")
                continue
            rows = [task_fn(s, name, lr) for s in FRESH_SEEDS]
            runs[name] = {
                "lr": lr,
                "loss": [r["loss"] for r in rows],
                "acc": [r["accuracy"] for r in rows] if "accuracy" in rows[0] else None,
            }
            m = statistics.mean(runs[name]["loss"])
            a = (f"  acc={statistics.mean(runs[name]['acc']):.4f}"
                 if runs[name]["acc"] else "")
            print(f"{task:26s} {name:10s} lr={lr:.0e} loss={m:.4e}{a}", flush=True)

        comps = {}
        for cand in CANDIDATES:
            if cand not in runs:
                continue
            for base in BASELINES:
                if base not in runs:
                    continue
                # compare on accuracy if available (higher better), else loss
                if runs[cand]["acc"] is not None:
                    a, b = runs[cand]["acc"], runs[base]["acc"]
                    diffs = [x - y for x, y in zip(a, b)]  # >0 means cand wins
                    wins = sum(d > 0 for d in diffs)
                    ties = sum(d == 0 for d in diffs)
                else:
                    a, b = runs[cand]["loss"], runs[base]["loss"]
                    diffs = [y - x for x, y in zip(a, b)]  # >0 means cand wins
                    wins = sum(d > 0 for d in diffs)
                    ties = sum(d == 0 for d in diffs)
                mean_d, t = paired_t(diffs)
                comps[f"{cand}_vs_{base}"] = {
                    "wins": wins, "ties": ties, "n": len(diffs),
                    "mean_diff": mean_d, "paired_t": t,
                }
                print(f"  {cand} vs {base}: {wins}/{len(diffs)} wins  "
                      f"mean diff={mean_d:+.4e}  t={t:+.2f}", flush=True)
        out[task] = {"runs": {k: v for k, v in runs.items()}, "comparisons": comps}

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
