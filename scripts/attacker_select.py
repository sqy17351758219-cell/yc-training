#!/usr/bin/env python
"""Attacker: 8-GPU sharded CVaR worst-case state selection (Deepening B + G).

For each prompt, enumerate candidate drifted states from the archive (UCB
sampling), probe the CURRENT policy n_probe times per candidate, score badness,
and select one state from the worst alpha-tail (CVaR). Selected states are
written for the defender. Probe outcomes update the archive's breach stats.

    torchrun --nproc_per_node 8 scripts/attacker_select.py \
        --policy runs/abrr/round0/policy \
        --base_harmful data/base_cots_v3.jsonl \
        --base_benign  data/base_cots_benign_v3.jsonl \
        --archive runs/abrr/archive.json \
        --config configs/default.yaml \
        --n_harmful 128 --n_benign 128 --seed 0 \
        --out runs/abrr/round0/states.jsonl

`--n_harmful`/`--n_benign` come from the orchestrator, which sets them from the
LambdaController's split. Archive is loaded/saved as JSON by rank 0.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advaudit import cvar, dist as D  # noqa: E402
from advaudit.drift import Drift, candidate_states  # noqa: E402
from advaudit.archive import DriftArchive  # noqa: E402
from advaudit.io_utils import load_yaml, read_jsonl, write_jsonl, deep_get  # noqa: E402
from advaudit.judges import SafetyJudge  # noqa: E402
from advaudit.rewards import final_answer, is_refusal  # noqa: E402


def load_archive(path: str, seeds_kind=None) -> DriftArchive:
    from advaudit.drift import default_drifts
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        drifts = [Drift(**d) for d in data["drifts"]]
        return DriftArchive(drifts)
    return DriftArchive(default_drifts())


def save_archive(arch: DriftArchive, path: str) -> None:
    payload = {"drifts": [vars(d) for d in arch.drifts], "stats": arch.stats()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_templated(tok, prompt: str) -> str:
    """Chat-templated prompt ending right where reasoning begins."""
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )


@torch.no_grad()
def probe(model, tok, dev, state_text: str, n: int, max_new: int = 256):
    """Sample n continuations from a drifted state; return list of answers."""
    ids = tok(state_text, return_tensors="pt").input_ids.to(dev)
    outs = []
    for _ in range(n):
        gen = model.generate(
            ids, max_new_tokens=max_new, do_sample=True, temperature=0.8,
            pad_token_id=tok.eos_token_id,
        )
        outs.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--base_harmful", required=True)
    ap.add_argument("--base_benign", required=True)
    ap.add_argument("--archive", default="")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_harmful", type=int, required=True)
    ap.add_argument("--n_benign", type=int, required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    n_probe = deep_get(cfg, "attacker.n_probe", 3)
    max_cand = deep_get(cfg, "attacker.max_candidates", 10)
    alpha = deep_get(cfg, "attacker.cvar_alpha", 0.3)
    depths = tuple(deep_get(cfg, "attacker.depths", [0.25, 0.5, 0.75, 0.9]))
    temp = deep_get(cfg, "attacker.softmax_temp", 1.0)
    judge_id = deep_get(cfg, "judges.safety_judge", "meta-llama/Llama-Guard-3-8B")

    D.init_distributed()
    dev = D.device()

    tok = AutoTokenizer.from_pretrained(args.policy)
    model = AutoModelForCausalLM.from_pretrained(
        args.policy, torch_dtype=torch.float32
    ).to(dev).eval()
    judge = SafetyJudge(judge_id, device=dev)  # per-rank on cuda:{LOCAL_RANK}

    archive = load_archive(args.archive)
    archive.set_round(args.round)

    # Build the per-round prompt list (harmful + benign), then shard it.
    harmful = read_jsonl(args.base_harmful)[: args.n_harmful]
    benign = read_jsonl(args.base_benign)[: args.n_benign]
    work = [(r, "harmful") for r in harmful] + [(r, "benign") for r in benign]
    mine = D.shard(work)

    rng = random.Random(args.seed + D.rank())
    selected = []
    probe_updates = []  # (drift_id, breached)

    for row, kind in mine:
        templated = build_templated(tok, row["prompt"])
        sampler = archive.ucb_sampler(kind, rng)
        cands = candidate_states(
            prompt_id=str(row["id"]),
            templated_prompt=templated,
            base_cot=row["base_cot"],
            kind=kind,
            drifts=archive.of_kind(kind),
            depths=depths,
            max_candidates=max_cand,
            rng=rng,
            drift_sampler=sampler,
        )
        badness = []
        for c in cands:
            answers = probe(model, tok, dev, c.state_text, n_probe)
            if kind == "harmful":
                scores = [judge.p_safe(row["prompt"], final_answer(a)) for a in answers]
            else:
                scores = [1.0 if is_refusal(final_answer(a)) else 0.0 for a in answers]
            c.probe_scores = scores
            b = cvar.attacker_badness(scores, kind)
            badness.append(b)
            # breach = drift broke the policy on at least one probe
            breached = (b > 0.5)
            probe_updates.append((c.drift_id, breached))

        chosen, _ = cvar.select_cvar(cands, badness, alpha=alpha, temp=temp, rng=rng)
        selected.append({
            "prompt_id": chosen.prompt_id,
            "orig_prompt": row["prompt"],
            "kind": kind,
            "drift_id": chosen.drift_id,
            "depth": chosen.depth,
            "state_text": chosen.state_text,
            "badness": max(badness),
        })

    all_states = D.gather_objects(selected)
    all_updates = D.gather_objects(probe_updates)

    if D.is_main():
        for did, breached in all_updates:
            archive.register_probe(did, breached)
        n = write_jsonl(args.out, all_states)
        if args.archive:
            save_archive(archive, args.archive)
        print(f"[attacker] selected {n} states -> {args.out} | archive={archive.stats()}")
    D.barrier()
    D.cleanup()


if __name__ == "__main__":
    main()
