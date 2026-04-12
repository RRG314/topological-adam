from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from topological_adam import ReconnectionStoppingRule, TopologicalAdamV2


torch.manual_seed(1729)
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))
optimizer = TopologicalAdamV2(
    model.parameters(),
    lr=1e-3,
    eta=0.01,
    w_topo=0.01,
    target_energy=1.0,
    deterministic_init=True,
    track_stats=True,
)
stopper = ReconnectionStoppingRule(peak_ratio=0.3, absolute_threshold=1e-3, warmup_steps=3)
criterion = nn.CrossEntropyLoss()

features = torch.randn(256, 32)
labels = torch.randint(0, 4, (256,))

for epoch in range(10):
    optimizer.zero_grad()
    logits = model(features)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    stats = optimizer.field_metrics()
    decision = stopper.update(epoch, stats["j_t"], loss.item())
    print(
        f"epoch={epoch:02d} loss={loss.item():.4f} E_t={stats['energy']:.4f} "
        f"J_t={stats['j_t']:.6f} corr={stats['alpha_beta_corr']:.4f}"
    )
    if decision.should_stop:
        print(f"Stopping suggestion at epoch {epoch}: {decision.reason}")
        break
