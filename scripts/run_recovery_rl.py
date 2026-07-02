#!/usr/bin/env python
"""Orchestrator: rounds of attacker -> defender -> archive/λ update.

Single-process controller that launches the distributed attacker and defender
via `torchrun --nproc_per_node 8` as subprocesses, and threads the state between
rounds:

    for r in rounds:
        n_h, n_b = lambda_ctrl.split_counts(per_side_limit * 2)   # λ-driven mix
        torchrun attacker_select.py  (writes states.jsonl, updates archive.json)
        torchrun grpo_defender.py    (writes policy_out, mined drifts)
        fold mined drifts into archive; measure ORR; lambda_ctrl.update(orr)

Run directly (NOT under torchrun) on the 8-GPU node:

    python scripts/run_recovery_rl.py \
        --sft saves/advchain-v3-r1-distill-qwen-7b \
        --base_harmful data/base_cots_v3.jsonl \
        --base_benign  data/base_cots_benign_v3.jsonl \
        --config configs/default.yaml \
        --out runs/abrr
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advaudit.lambda_ctrl import LambdaController  # noqa: E402
from advaudit.archive import DriftArchive  # noqa: E402
from advaudit.drift import Drift, default_drifts  # noqa: E402
from advaudit.io_utils import load_yaml, read_jsonl, deep_get  # noqa: E402
from advaudit.rewards import batch_orr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def torchrun(script: str, nproc: int, extra):
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}",
           os.path.join(HERE, script), *extra]
    print("[orchestrator] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def init_archive(path: str) -> None:
    if os.path.exists(path):
        return
    arch = DriftArchive(default_drifts())
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"drifts": [vars(d) for d in arch.drifts], "stats": arch.stats()},
                  f, ensure_ascii=False, indent=2)


def fold_mined(archive_path: str, mined_path: str, cfg) -> None:
    """Merge mined drifts (already safety-filtered at creation) into the archive,
    applying dedup via the archive's admission path."""
    if not (mined_path and os.path.exists(mined_path)):
        return
    with open(archive_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    arch = DriftArchive([Drift(**d) for d in data["drifts"]],
                        dedup_cos=deep_get(cfg, "archive.dedup_cos", 0.9))
    added = 0
    for row in read_jsonl(mined_path):
        # re-admit through the safety/dedup gate (embed_fn None -> no semantic dedup here)
        d = arch._admit(row["text"], kind=row["kind"], source="selfmine")
        if d is not None:
            added += 1
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump({"drifts": [vars(d) for d in arch.drifts], "stats": arch.stats()},
                  f, ensure_ascii=False, indent=2)
    print(f"[orchestrator] folded {added} mined drifts | archive={arch.stats()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="AdvChain SFT ckpt = start policy + KL ref")
    ap.add_argument("--base_harmful", required=True)
    ap.add_argument("--base_benign", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    rounds = deep_get(cfg, "rl.rounds", 1)
    steps = deep_get(cfg, "rl.steps_per_round", 80)
    per_side = deep_get(cfg, "rl.per_side_limit", 256)
    nproc = deep_get(cfg, "launch.nproc", 8)

    lam = LambdaController(
        tau=deep_get(cfg, "lambda_ctrl.tau", 0.04),
        eta=deep_get(cfg, "lambda_ctrl.eta", 0.05),
        lam=deep_get(cfg, "lambda_ctrl.lam_init", 1.0),
        lam_min=deep_get(cfg, "lambda_ctrl.lam_min", 0.05),
        lam_max=deep_get(cfg, "lambda_ctrl.lam_max", 20.0),
        ema=deep_get(cfg, "lambda_ctrl.ema", 0.9),
        pb_lo=deep_get(cfg, "lambda_ctrl.pb_clamp", [0.2, 0.8])[0],
        pb_hi=deep_get(cfg, "lambda_ctrl.pb_clamp", [0.2, 0.8])[1],
    )

    os.makedirs(args.out, exist_ok=True)
    archive_path = os.path.join(args.out, "archive.json")
    init_archive(archive_path)

    policy_in = args.sft  # round 0 starts from SFT
    for r in range(rounds):
        rdir = os.path.join(args.out, f"round{r}")
        os.makedirs(rdir, exist_ok=True)
        states_path = os.path.join(rdir, "states.jsonl")
        mined_path = os.path.join(rdir, "mined.jsonl")
        policy_out = os.path.join(rdir, "policy")

        total = per_side * 2
        n_h, n_b = lam.split_counts(total)
        print(f"[orchestrator] round {r}: λ={lam.lam:.3f} p_benign={lam.benign_ratio():.3f} "
              f"-> n_harmful={n_h} n_benign={n_b}", flush=True)

        torchrun("attacker_select.py", nproc, [
            "--policy", policy_in,
            "--base_harmful", args.base_harmful,
            "--base_benign", args.base_benign,
            "--archive", archive_path,
            "--config", args.config,
            "--n_harmful", str(n_h), "--n_benign", str(n_b),
            "--round", str(r), "--seed", str(r),
            "--out", states_path,
        ])

        torchrun("grpo_defender.py", nproc, [
            "--policy", policy_in,
            "--sft_ref", args.sft,     # KL anchor is always pi_SFT
            "--states", states_path,
            "--archive", archive_path,
            "--config", args.config,
            "--steps", str(steps),
            "--out", policy_out,
            "--mined_out", mined_path,
        ])

        fold_mined(archive_path, mined_path, cfg)

        # measure ORR on the benign states' final rollouts (written by defender if present)
        orr = _measure_round_orr(states_path, rdir)
        lam.update(orr)
        print(f"[orchestrator] round {r} done | measured ORR≈{orr:.3f} | next λ={lam.lam:.3f}",
              flush=True)

        with open(os.path.join(rdir, "lambda_history.json"), "w") as f:
            json.dump({"orr": orr, "history": lam.history}, f, indent=2)

        policy_in = policy_out  # chain rounds

    print(f"[orchestrator] finished {rounds} round(s). Final policy: {policy_in}")


def _measure_round_orr(states_path: str, rdir: str) -> float:
    """Best-effort ORR estimate from any benign rollouts the defender dumped.

    Falls back to the base-model target if no rollouts are available (keeps λ
    from moving on missing data).
    """
    roll = os.path.join(rdir, "benign_rollouts.jsonl")
    if os.path.exists(roll):
        comps = [r.get("completion", "") for r in read_jsonl(roll)]
        return batch_orr(comps)
    return 0.04  # neutral: equals default tau, leaves λ ~unchanged


if __name__ == "__main__":
    main()
