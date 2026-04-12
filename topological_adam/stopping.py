from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class StopDecision:
    should_stop: bool
    reason: str
    step: int
    current_j_t: float
    peak_j_t: float
    ratio_to_peak: float


@dataclass
class ReconnectionStoppingRule:
    """Heuristic stopping rule based on decay from the observed J_t peak."""

    peak_ratio: float = 0.3
    absolute_threshold: float = 1e-3
    warmup_steps: int = 3
    min_steps_after_peak: int = 1
    patience: int = 1
    require_loss_fraction: float | None = None
    peak_j_t: float = field(default=0.0, init=False)
    peak_step: int = field(default=-1, init=False)
    initial_loss: float | None = field(default=None, init=False)
    _hits: int = field(default=0, init=False)

    def reset(self) -> None:
        self.peak_j_t = 0.0
        self.peak_step = -1
        self.initial_loss = None
        self._hits = 0

    def update(self, step: int, j_t: float, loss: float | None = None) -> StopDecision:
        current_j_t = float(j_t)
        if self.initial_loss is None and loss is not None:
            self.initial_loss = float(loss)
        if current_j_t > self.peak_j_t:
            self.peak_j_t = current_j_t
            self.peak_step = step
            self._hits = 0

        ratio_to_peak = current_j_t / self.peak_j_t if self.peak_j_t > 0 else 1.0
        after_warmup = step >= self.warmup_steps
        after_peak = self.peak_step >= 0 and (step - self.peak_step) >= self.min_steps_after_peak
        relative_hit = self.peak_j_t > 0 and ratio_to_peak <= self.peak_ratio
        absolute_hit = self.absolute_threshold is not None and current_j_t <= self.absolute_threshold
        loss_hit = True
        if self.require_loss_fraction is not None and loss is not None and self.initial_loss not in (None, 0.0):
            loss_hit = float(loss) <= float(self.initial_loss) * float(self.require_loss_fraction)

        if after_warmup and after_peak and loss_hit and (relative_hit or absolute_hit):
            self._hits += 1
        else:
            self._hits = 0

        if self._hits >= self.patience:
            if relative_hit:
                reason = f"J_t fell below {self.peak_ratio:.3f} of its observed peak"
            else:
                reason = f"J_t fell below the absolute threshold {self.absolute_threshold:.3e}"
            return StopDecision(
                should_stop=True,
                reason=reason,
                step=step,
                current_j_t=current_j_t,
                peak_j_t=self.peak_j_t,
                ratio_to_peak=ratio_to_peak,
            )

        return StopDecision(
            should_stop=False,
            reason="continue",
            step=step,
            current_j_t=current_j_t,
            peak_j_t=self.peak_j_t,
            ratio_to_peak=ratio_to_peak,
        )

    def evaluate_history(
        self,
        j_history: Sequence[float],
        loss_history: Sequence[float] | None = None,
    ) -> StopDecision | None:
        self.reset()
        for step, j_t in enumerate(j_history):
            loss = None if loss_history is None else loss_history[step]
            decision = self.update(step=step, j_t=float(j_t), loss=loss)
            if decision.should_stop:
                return decision
        return None
