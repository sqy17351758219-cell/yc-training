"""Component 1 — constraint-balanced controller (Lagrange multiplier λ).

We treat over-refusal as a *constraint* rather than a hand-tuned data ratio:

    max_pi   E_harmful[ SafeScore ]
    s.t.     E_benign[ refusal ]  <=  tau

Online primal-dual update of the multiplier λ >= 0:

    λ  <-  clip( λ + eta * (ORR_ema - tau),  λ_min, λ_max )

CRITICAL: λ must NOT be applied as a reward multiplier. GRPO normalizes
advantages within each generation group (same state, N samples), so any
constant scale on a group's reward is cancelled by the z-score normalization.
Instead λ controls the **fraction of benign states** in the next batch:

    p_benign = clip( λ / (1 + λ),  pb_lo, pb_hi )

λ = 1  ->  p_benign = 0.5 (the current hand-tuned default). As ORR exceeds tau,
λ grows, more benign recovery states enter training, and ORR is pushed back
down — the balance point is learned, not set.

Torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LambdaController:
    tau: float = 0.04            # target ORR ceiling (base-model level)
    eta: float = 0.05            # dual step size
    lam: float = 1.0             # current multiplier (lam_init)
    lam_min: float = 0.05
    lam_max: float = 20.0
    ema: float = 0.9             # EMA smoothing of measured batch ORR
    pb_lo: float = 0.2           # clamp on benign ratio
    pb_hi: float = 0.8
    _orr_ema: float = -1.0       # -1 sentinel: not yet initialized
    history: List[Tuple[float, float, float]] = None  # (orr_ema, lam, p_benign)

    def __post_init__(self) -> None:
        if self.history is None:
            self.history = []
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError("tau must be in [0,1]")
        if not self.lam_min <= self.lam <= self.lam_max:
            self.lam = min(max(self.lam, self.lam_min), self.lam_max)

    def _clip_lam(self, x: float) -> float:
        return min(max(x, self.lam_min), self.lam_max)

    def _clip_pb(self, x: float) -> float:
        return min(max(x, self.pb_lo), self.pb_hi)

    def update(self, orr_batch: float) -> float:
        """Feed the measured over-refusal rate of the last benign batch.

        Returns the updated λ. Uses an EMA of the (noisy) per-batch ORR before
        the dual step so the ratio does not thrash.
        """
        if not 0.0 <= orr_batch <= 1.0:
            raise ValueError(f"orr_batch must be a rate in [0,1], got {orr_batch}")
        if self._orr_ema < 0:
            self._orr_ema = orr_batch
        else:
            self._orr_ema = self.ema * self._orr_ema + (1.0 - self.ema) * orr_batch
        self.lam = self._clip_lam(self.lam + self.eta * (self._orr_ema - self.tau))
        self.history.append((self._orr_ema, self.lam, self.benign_ratio()))
        return self.lam

    def benign_ratio(self) -> float:
        """Fraction of benign states to sample into the next batch."""
        return self._clip_pb(self.lam / (1.0 + self.lam))

    def split_counts(self, total: int) -> Tuple[int, int]:
        """Split `total` states into (n_harmful, n_benign) for the next batch."""
        n_benign = round(self.benign_ratio() * total)
        n_benign = min(max(n_benign, 0), total)
        return total - n_benign, n_benign

    @property
    def orr_ema(self) -> float:
        return max(self._orr_ema, 0.0)
