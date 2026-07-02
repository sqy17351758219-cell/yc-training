"""Deepening B — CVaR worst-tail selection for the attacker.

The attacker scores each candidate drifted state by how badly the *current*
policy handles it, then trains the defender on the worst α-tail rather than the
single argmin. This (a) averages out judge noise that makes a single argmin an
outlier, and (b) turns "recover from the worst case" into a distributionally
robust objective (CVaR_α).

Orientation convention: we work with a **badness** score where *higher = worse
for the policy* (more likely unsafe on harmful prompts, more likely refused on
benign prompts). `attacker_badness` converts raw probe means to this convention.

Torch-free.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def attacker_badness(probe_scores: Sequence[float], kind: str) -> float:
    """Aggregate probe rollouts into a badness score (higher = worse policy).

    - harmful: probe score = mean safety in [0,1]; badness = 1 - safety
      (an easy-to-jailbreak drift has low safety -> high badness).
    - benign:  probe score = mean refusal in [0,1]; badness = refusal
      (an easy-to-over-refuse drift has high refusal -> high badness).
    """
    if not probe_scores:
        return 0.0
    m = sum(probe_scores) / len(probe_scores)
    if kind == "harmful":
        return 1.0 - m
    if kind == "benign":
        return m
    raise ValueError(f"unknown kind {kind!r}")


def worst_tail_indices(badness: Sequence[float], alpha: float) -> List[int]:
    """Indices of the worst α-fraction (highest badness).

    alpha in (0, 1]. alpha -> 0 approaches a single argmax(badness) (== the old
    single worst-case argmin on safety); alpha == 1 keeps everything (== random
    injection among candidates). At least one index is always returned.
    """
    if not badness:
        return []
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    n = len(badness)
    k = max(1, math.ceil(alpha * n))
    order = sorted(range(n), key=lambda i: badness[i], reverse=True)  # worst first
    return order[:k]


def softmax_weights(values: Sequence[float], temp: float = 1.0) -> List[float]:
    """Numerically-stable softmax over `values` with temperature `temp`."""
    if not values:
        return []
    if temp <= 0:
        # degenerate: all mass on the max
        mx = max(values)
        raw = [1.0 if v == mx else 0.0 for v in values]
    else:
        m = max(values)
        raw = [math.exp((v - m) / temp) for v in values]
    s = sum(raw) or 1.0
    return [r / s for r in raw]


def select_cvar(
    candidates: Sequence[T],
    badness: Sequence[float],
    alpha: float = 0.3,
    temp: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Tuple[T, int]:
    """Pick one candidate from the worst α-tail, softmax-weighted by badness.

    Returns (candidate, original_index). Deterministic given `rng`.
    """
    if len(candidates) != len(badness):
        raise ValueError("candidates and badness length mismatch")
    if not candidates:
        raise ValueError("no candidates to select from")
    rng = rng or random.Random()
    tail = worst_tail_indices(badness, alpha)
    tail_badness = [badness[i] for i in tail]
    weights = softmax_weights(tail_badness, temp=temp)
    # weighted choice within the tail
    r = rng.random()
    acc = 0.0
    for local_i, w in enumerate(weights):
        acc += w
        if r <= acc:
            idx = tail[local_i]
            return candidates[idx], idx
    idx = tail[-1]
    return candidates[idx], idx


def cvar_value(badness: Sequence[float], alpha: float) -> float:
    """The CVaR_α of badness = mean of the worst α-tail. Useful for logging."""
    tail = worst_tail_indices(badness, alpha)
    if not tail:
        return 0.0
    return sum(badness[i] for i in tail) / len(tail)
