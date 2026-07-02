#!/usr/bin/env python
"""Defender: 8-GPU GRPO on selected drifted states (Deepening A + G, dual-head).

Loads the selected states from the attacker, runs TRL GRPOTrainer with a
trajectory-dense dual-head reward, and mines failed rollouts into the archive.
Launched via torchrun / accelerate across 8 GPUs; the safety judge lives on
cuda:{LOCAL_RANK} to avoid OOM.

    torchrun --nproc_per_node 8 scripts/grpo_defender.py \
        --policy runs/abrr/round0/policy_in \
        --sft_ref saves/advchain-v3-r1-distill-qwen-7b \
        --states runs/abrr/round0/states.jsonl \
        --archive runs/abrr/archive.json \
        --config configs/default.yaml \
        --steps 80 --out runs/abrr/round0/policy_out

Reward is dispatched by each state's `kind`. The harmful head adds
alpha_traj * mean(checkpoint P(safe)); the benign head is 1 - refusal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advaudit import dist as D  # noqa: E402
from advaudit.io_utils import load_yaml, read_jsonl, deep_get  # noqa: E402
from advaudit.judges import SafetyJudge  # noqa: E402
from advaudit.rewards import final_answer, is_refusal, harmful_reward  # noqa: E402
from advaudit.archive import DriftArchive, find_pivot  # noqa: E402
from advaudit.drift import Drift  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="policy to update (SFT ckpt on round 0)")
    ap.add_argument("--sft_ref", required=True, help="frozen KL reference = pi_SFT")
    ap.add_argument("--states", required=True)
    ap.add_argument("--archive", default="")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mined_out", default="", help="jsonl of newly mined drifts")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    num_gen = deep_get(cfg, "rl.num_generations", 8)
    pdbs = deep_get(cfg, "rl.per_device_train_batch_size", 4)
    max_comp = deep_get(cfg, "rl.max_completion_length", 1024)
    beta = deep_get(cfg, "rl.beta", 0.04)
    lr = deep_get(cfg, "rl.lr", 1e-6)
    alpha_traj = deep_get(cfg, "reward.alpha_traj", 0.3)
    n_ckpt = deep_get(cfg, "reward.n_checkpoints", 4)
    theta_flip = deep_get(cfg, "archive.theta_flip", 0.5)
    hysteresis = deep_get(cfg, "archive.flip_hysteresis", 2)
    enable_selfmine = deep_get(cfg, "archive.enable_selfmine", True)
    judge_id = deep_get(cfg, "judges.safety_judge", "meta-llama/Llama-Guard-3-8B")
    grad_ckpt = deep_get(cfg, "rl.grad_ckpt", True)

    D.init_distributed()
    dev = D.device()
    judge = SafetyJudge(judge_id, device=dev)  # per-rank

    states = read_jsonl(args.states)
    # index prompt/kind by the state text so the reward fn can look them up
    meta: Dict[str, Dict] = {s["state_text"]: s for s in states}
    ds = Dataset.from_dict({"prompt": [s["state_text"] for s in states]})

    mined: List[Drift] = []

    def reward_fn(prompts, completions, **kw) -> List[float]:
        """TRL calls this per generation. We score with the dual-head reward and
        opportunistically self-mine failed rollouts."""
        rewards: List[float] = []
        for prompt, completion in zip(prompts, completions):
            m = meta.get(prompt, {})
            kind = m.get("kind", "harmful")
            orig = m.get("orig_prompt", "")
            if kind == "benign":
                ans = final_answer(completion)
                refused = is_refusal(ans)
                rewards.append(0.0 if refused else 1.0)
                if enable_selfmine and refused:
                    d = _mine_benign(completion)
                    if d:
                        mined.append(d)
                continue
            # harmful head: trajectory-dense
            phi = judge.checkpoint_scores(orig, completion, n_ckpt)
            safe_final = judge.p_safe(orig, final_answer(completion))
            rewards.append(harmful_reward(safe_final, phi, alpha_traj=alpha_traj))
            if enable_selfmine and safe_final < 0.5:
                d = _mine_harmful(completion, phi, theta_flip, hysteresis)
                if d:
                    mined.append(d)
        return rewards

    grpo_cfg = GRPOConfig(
        output_dir=args.out,
        num_generations=num_gen,
        per_device_train_batch_size=pdbs,
        max_completion_length=max_comp,
        max_steps=args.steps,
        learning_rate=lr,
        beta=beta,  # KL to the reference model (pi_SFT)
        gradient_checkpointing=grad_ckpt,
        bf16=False, fp16=False,  # fp32 REQUIRED
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )

    policy = AutoModelForCausalLM.from_pretrained(args.policy, torch_dtype=torch.float32)
    ref = AutoModelForCausalLM.from_pretrained(args.sft_ref, torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(args.policy)

    trainer = GRPOTrainer(
        model=policy,
        ref_model=ref,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=ds,
        processing_class=tok,
    )
    trainer.train()

    if D.is_main():
        trainer.save_model(args.out)
        tok.save_pretrained(args.out)
        # persist mined drifts for the orchestrator to fold into the archive
        if args.mined_out and mined:
            with open(args.mined_out, "w", encoding="utf-8") as f:
                for d in mined:
                    f.write(json.dumps(vars(d), ensure_ascii=False) + "\n")
        print(f"[defender] saved {args.out} | mined {len(mined)} drifts")
    D.barrier()
    D.cleanup()


# module-level archive used only for its mining helpers (no state needed)
_MINER = DriftArchive([])


def _mine_harmful(completion, phi, theta_flip, hysteresis):
    return _MINER.mine_harmful(final_answer(completion), phi, theta_flip, hysteresis)


def _mine_benign(completion):
    return _MINER.mine_benign(final_answer(completion))


if __name__ == "__main__":
    main()
