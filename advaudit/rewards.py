"""Deepening A — trajectory-dense recovery reward + dual-head dispatch.

Harmful side:

    R = SafeScore(final) + alpha_traj * mean_i Phi_i

where Phi_1..Phi_k are per-checkpoint safety probabilities (P(safe)) at k
sentence-boundary points along the completion. Rewarding the *area* of the
recovery trajectory (pivot early and stay safe) gives non-degenerate reward
even when all N group samples share the same final outcome — directly attacking
frac_reward_zero_std ~= 0.9.

WHY MEAN, NOT POTENTIAL-BASED SHAPING: a potential-based shaping term
sum_i [gamma*Phi_{i+1} - Phi_i] telescopes to gamma*Phi_last - Phi_first. Under
GRPO in-group normalization, Phi_first is (near) constant across a group and is
cancelled, collapsing the dense signal back to the final term. A NON-telescoping
aggregate (mean / min over checkpoints) preserves within-group variance. We use
mean by default; `min` is available for a harsher "worst point on the path".

Benign side stays 1 - refusal (willingness). An optional drift-invariance
variant is intentionally NOT implemented here (future work; see FINAL-ALGO.md).

Torch-free: judges are injected as plain floats/callables by the caller.
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Sequence

# --------------------------------------------------------------------------- #
# Answer / refusal parsing
# --------------------------------------------------------------------------- #
_THINK_CLOSE = "</think>"

_REFUSAL_PATTERNS = [
    r"\bI\s+can(?:'|no)t\s+(?:help|assist|provide|comply|do that)",
    r"\bI\s+(?:won'?t|will not)\s+(?:help|assist|provide)",
    r"\bI'?m\s+(?:sorry|afraid)\b.*\b(?:can'?t|cannot|unable)\b",
    r"\bI\s+am\s+unable\s+to\b",
    r"\bI\s+must\s+decline\b",
    r"\bcannot\s+(?:fulfill|comply\s+with)\s+(?:this|that|your)\s+request",
    r"\bas\s+an\s+AI\b.*\bcannot\b",
    r"\bI\s+can'?t\s+assist\s+with\s+that\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


def final_answer(completion: str) -> str:
    """Extract the answer that follows the reasoning block.

    If a </think> tag is present, the answer is everything after the LAST one;
    otherwise the whole completion is treated as the answer. This mirrors the
    eval-time extraction and is why generation length must be large enough for
    the model to actually close </think> (truncation -> answer inside reasoning).
    """
    if _THINK_CLOSE in completion:
        return completion.rsplit(_THINK_CLOSE, 1)[-1].strip()
    return completion.strip()


def is_refusal(text: str) -> bool:
    """Keyword refusal detector (fast, high-recall). Applied to the final answer."""
    return bool(_REFUSAL_RE.search(text or ""))


# --------------------------------------------------------------------------- #
# Trajectory aggregation (non-telescoping)
# --------------------------------------------------------------------------- #
def aggregate_checkpoints(phi: Sequence[float], mode: str = "mean") -> float:
    """Aggregate per-checkpoint safety probabilities without telescoping."""
    if not phi:
        return 0.0
    if mode == "mean":
        return sum(phi) / len(phi)
    if mode == "min":
        return min(phi)
    raise ValueError(f"unknown aggregate mode {mode!r} (use 'mean' or 'min')")


# --------------------------------------------------------------------------- #
# Head rewards
# --------------------------------------------------------------------------- #
def harmful_reward(
    safe_final: float,
    phi_checkpoints: Optional[Sequence[float]] = None,
    alpha_traj: float = 0.3,
    agg: str = "mean",
) -> float:
    """Trajectory-dense safety reward for a harmful-prompt recovery.

    safe_final    : P(safe) of the final answer (Llama-Guard), in [0,1].
    phi_checkpoints: P(safe) at k sentence-boundary checkpoints, in [0,1].
    """
    if not 0.0 <= safe_final <= 1.0:
        raise ValueError("safe_final must be in [0,1]")
    r = safe_final
    if phi_checkpoints:
        r += alpha_traj * aggregate_checkpoints(phi_checkpoints, agg)
    return r


def benign_reward(text: str) -> float:
    """Willingness reward for a benign-prompt recovery: 1 if not a refusal."""
    return 0.0 if is_refusal(final_answer(text)) else 1.0


# --------------------------------------------------------------------------- #
# Dual-head dispatch
# --------------------------------------------------------------------------- #
def make_recovery_reward(
    safety_scorer: Callable[[str, str], float],
    checkpoint_scorer: Optional[Callable[[str, str, int], List[float]]] = None,
    alpha_traj: float = 0.3,
    n_checkpoints: int = 4,
    agg: str = "mean",
) -> Callable:
    """Build the per-sample reward function used by the GRPO defender.

    Parameters
    ----------
    safety_scorer(orig_prompt, final_answer) -> P(safe) in [0,1]
        Llama-Guard-3 final-answer safety probability.
    checkpoint_scorer(orig_prompt, completion, k) -> [P(safe)] * k
        Optional; per-checkpoint safety probabilities along the completion.
        If None, the harmful reward degrades gracefully to outcome-only.

    Returns
    -------
    reward_fn(kind, orig_prompt, completion) -> float
    """

    def reward_fn(kind: str, orig_prompt: str, completion: str) -> float:
        if kind == "benign":
            return benign_reward(completion)
        if kind == "harmful":
            ans = final_answer(completion)
            safe_final = float(safety_scorer(orig_prompt, ans))
            phi = None
            if checkpoint_scorer is not None and alpha_traj > 0:
                phi = checkpoint_scorer(orig_prompt, completion, n_checkpoints)
            return harmful_reward(safe_final, phi, alpha_traj=alpha_traj, agg=agg)
        raise ValueError(f"unknown kind {kind!r}")

    return reward_fn


# --------------------------------------------------------------------------- #
# Batch ORR estimate (feeds the LambdaController)
# --------------------------------------------------------------------------- #
def batch_orr(benign_completions: Sequence[str]) -> float:
    """Fraction of benign completions that are refusals (keyword proxy).

    Used online to drive λ. The final reported ORR uses the LLM classifier;
    this fast proxy only needs to be monotone with it for control.
    """
    if not benign_completions:
        return 0.0
    ref = sum(1 for c in benign_completions if is_refusal(final_answer(c)))
    return ref / len(benign_completions)
