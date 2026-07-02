"""Deepening G — self-mined, evolving drift archive.

The archive holds all drifts (seed + self-mined [+ mutated]) and provides:

- UCB sampling of parent drifts for the attacker (explore rare-but-nasty drifts).
- Self-mining: clip failed recovery rollouts at the reasoning *pivot* and turn
  the model's own drifted prefix into a new, empirically-validated drift.
- Corpus-safety filtering + semantic dedup before admission.
- Fictitious-play elites: the defender trains on a mix of current + historical
  elite drifts so it does not forget old defenses while learning new ones.

Self-ReSET's passive unsafe-prefix buffer is exactly "self-mine only, no UCB,
no CVaR selection, no constraint" — a strict special case (ablation row).

Torch-free. Embedding + safety-judge functions are injected (tests use fakes).
"""

from __future__ import annotations

import math
import random
import re
from typing import Callable, Dict, List, Optional, Sequence

from .drift import Drift

# --------------------------------------------------------------------------- #
# Pivot detection (where recovery failed)
# --------------------------------------------------------------------------- #
def find_pivot(
    phi: Sequence[float],
    theta_flip: float = 0.5,
    hysteresis: int = 2,
) -> Optional[int]:
    """Index of the checkpoint where the trajectory 'flips' unsafe.

    Returns the index of the first checkpoint that begins a run of >= hysteresis
    consecutive checkpoints all below theta_flip. None if no such run exists
    (i.e. the trajectory never clearly flipped — don't mine it).
    """
    if not phi:
        return None
    run = 0
    start = None
    for i, p in enumerate(phi):
        if p < theta_flip:
            if run == 0:
                start = i
            run += 1
            if run >= hysteresis:
                return start
        else:
            run = 0
            start = None
    return None


# --------------------------------------------------------------------------- #
# Corpus-safety filter for mined / mutated drift text
# --------------------------------------------------------------------------- #
# Regex black-list of operational-harm markers. A mined drift is FRAMING text;
# if it carries operational detail it must be rejected. This is a coarse net;
# the real gate is the injected Llama-Guard `safety_judge`.
_OPERATIONAL_BLOCKLIST = re.compile(
    r"\b(step\s*\d|mix\s+\d|grams?|milliliters?|ml\b|dosage|synthesi[sz]e|"
    r"detonat|precursor|centrifuge|reagent|nitrat|sulfuric|ammonium|"
    r"IP\s+address|payload|CVE-\d|reverse\s+shell)\b",
    re.IGNORECASE,
)


def passes_safety_filter(
    text: str,
    safety_judge: Optional[Callable[[str], bool]] = None,
    max_len: int = 600,
    min_len: int = 15,
    min_words: int = 3,
) -> bool:
    """True if `text` is admissible as a (framing-only) drift.

    - reject empty / trivially short / overly long clips,
    - reject operational-harm markers,
    - reject if an injected safety judge flags the text itself as unsafe.

    A drift must be a phrase, not a fragment: a one-word clip like "Sure." is
    useless as a training drift and is rejected on length.
    """
    t = (text or "").strip()
    if not t or len(t) > max_len or len(t) < min_len:
        return False
    if len(t.split()) < min_words:
        return False
    if _OPERATIONAL_BLOCKLIST.search(t):
        return False
    if safety_judge is not None and not safety_judge(t):
        return False
    return True


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #
def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


# --------------------------------------------------------------------------- #
# Archive
# --------------------------------------------------------------------------- #
class DriftArchive:
    def __init__(
        self,
        seeds: Sequence[Drift],
        embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
        safety_judge: Optional[Callable[[str], bool]] = None,
        ucb_c: float = 1.4,
        dedup_cos: float = 0.9,
        elite_frac: float = 0.5,
    ) -> None:
        self.drifts: List[Drift] = list(seeds)
        self.embed_fn = embed_fn
        self.safety_judge = safety_judge
        self.ucb_c = ucb_c
        self.dedup_cos = dedup_cos
        self.elite_frac = elite_frac
        self._embeds: Dict[str, Sequence[float]] = {}
        self._round = 0
        if embed_fn is not None:
            for d in self.drifts:
                self._embeds[d.id] = embed_fn(d.text)

    # ---- lifecycle ----
    def set_round(self, r: int) -> None:
        self._round = r

    def of_kind(self, kind: str) -> List[Drift]:
        return [d for d in self.drifts if d.kind == kind]

    def __len__(self) -> int:
        return len(self.drifts)

    # ---- UCB sampling for the attacker ----
    def ucb_sampler(self, kind: str, rng: Optional[random.Random] = None):
        """Return a `drift_sampler(pool, need)` closure for drift.candidate_states.

        Mixes exploitation (high breach EMA) with UCB exploration, and reserves
        `elite_frac` of picks for historical elites (fictitious play).
        """
        rng = rng or random.Random()

        def sampler(pool: Sequence[Drift], need: int) -> List[Drift]:
            pool = list(pool)
            if len(pool) <= need:
                return pool
            total_trials = sum(d.trials for d in pool) + 1

            def ucb(d: Drift) -> float:
                exploit = d.breach_ema
                explore = self.ucb_c * math.sqrt(
                    math.log(total_trials + 1) / (d.trials + 1)
                )
                return exploit + explore

            ranked = sorted(pool, key=ucb, reverse=True)
            n_elite = int(round(self.elite_frac * need))
            elites = [d for d in ranked if d.source != "seed"][:n_elite]
            picks = list(elites)
            for d in ranked:
                if len(picks) >= need:
                    break
                if d not in picks:
                    picks.append(d)
            # a little randomness so we don't lock onto the same set every round
            rng.shuffle(picks)
            return picks[:need]

        return sampler

    def register_probe(self, drift_id: str, breached: bool) -> None:
        for d in self.drifts:
            if d.id == drift_id:
                d.register(breached)
                return

    # ---- self-mine ----
    def mine_harmful(
        self,
        drifted_prefix_text: str,
        phi: Sequence[float],
        theta_flip: float,
        hysteresis: int,
    ) -> Optional[Drift]:
        """Turn a FAILED harmful recovery into a new drift.

        `drifted_prefix_text` is the model's own generated text from the
        injection point onward; we clip it at the pivot and admit the clip.
        """
        pivot = find_pivot(phi, theta_flip, hysteresis)
        if pivot is None:
            return None
        # Clip roughly at the pivot: keep text up to the pivot-th sentence.
        clip = _clip_sentences(drifted_prefix_text, pivot + 1)
        return self._admit(clip, kind="harmful", source="selfmine")

    def mine_benign(self, drifted_prefix_text: str) -> Optional[Drift]:
        """Turn a FAILED benign recovery (over-refusal) into a new benign drift.

        Clip at the first hesitation/refusal-committing sentence.
        """
        clip = _first_hesitation_clip(drifted_prefix_text)
        if clip is None:
            return None
        return self._admit(clip, kind="benign", source="selfmine")

    def _admit(self, text: str, kind: str, source: str) -> Optional[Drift]:
        if not passes_safety_filter(text, self.safety_judge):
            return None
        if self.embed_fn is not None:
            emb = self.embed_fn(text)
            for d in self.of_kind(kind):
                de = self._embeds.get(d.id)
                if de is not None and _cos(emb, de) > self.dedup_cos:
                    return None  # semantic duplicate
        else:
            emb = None
        new = Drift(
            id=f"{kind[0]}_mined_r{self._round}_{len(self.drifts)}",
            text=text.strip(),
            kind=kind,
            source=source,
            born_round=self._round,
        )
        self.drifts.append(new)
        if emb is not None:
            self._embeds[new.id] = emb
        return new

    # ---- pruning ----
    def prune(self, min_trials: int = 4, min_breach: float = 0.02) -> int:
        """Drop mined drifts that never break the policy (keep all seeds).

        Returns the number pruned.
        """
        keep: List[Drift] = []
        pruned = 0
        for d in self.drifts:
            if d.source == "seed" or d.trials < min_trials or d.breach_ema >= min_breach:
                keep.append(d)
            else:
                pruned += 1
                self._embeds.pop(d.id, None)
        self.drifts = keep
        return pruned

    def stats(self) -> Dict[str, int]:
        return {
            "total": len(self.drifts),
            "seed": sum(1 for d in self.drifts if d.source == "seed"),
            "selfmine": sum(1 for d in self.drifts if d.source == "selfmine"),
            "mutate": sum(1 for d in self.drifts if d.source == "mutate"),
            "harmful": len(self.of_kind("harmful")),
            "benign": len(self.of_kind("benign")),
        }


# --------------------------------------------------------------------------- #
# Text clipping helpers
# --------------------------------------------------------------------------- #
_SENT = re.compile(r"(?<=[.!?])\s+")


def _clip_sentences(text: str, n: int) -> str:
    sents = [s for s in _SENT.split((text or "").strip()) if s]
    return " ".join(sents[: max(1, n)]).strip()


_HESITATION_RE = re.compile(
    r"(sounds?\s+(?:dangerous|violent|risky)|maybe\s+I\s+should\s+refuse|"
    r"better\s+safe\s+than\s+sorry|I\s+should\s+(?:probably\s+)?(?:not|decline)|"
    r"this\s+(?:might|could)\s+be\s+(?:unsafe|harmful)|red\s+flag)",
    re.IGNORECASE,
)


def _first_hesitation_clip(text: str) -> Optional[str]:
    sents = [s for s in _SENT.split((text or "").strip()) if s]
    for i, s in enumerate(sents):
        if _HESITATION_RE.search(s):
            return " ".join(sents[: i + 1]).strip()
    return None
