from __future__ import annotations

from topological_adam import ReconnectionStoppingRule


def test_reconnection_stopping_triggers_after_peak_decay() -> None:
    rule = ReconnectionStoppingRule(peak_ratio=0.3, warmup_steps=2, min_steps_after_peak=1)
    decision = rule.evaluate_history([0.01, 0.012, 0.011, 0.0025, 0.0015], [2.0, 1.7, 1.2, 0.4, 0.2])
    assert decision is not None
    assert decision.should_stop
    assert decision.step == 3


def test_reconnection_stopping_waits_through_warmup() -> None:
    rule = ReconnectionStoppingRule(peak_ratio=0.5, warmup_steps=4, min_steps_after_peak=1)
    decision = rule.evaluate_history([0.01, 0.02, 0.005, 0.004], [2.0, 1.5, 0.9, 0.4])
    assert decision is None
