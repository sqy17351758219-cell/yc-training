# yc-training — Adversarial Bidirectional Recovery RL (ABRR / DRACO)

Distributed training + evaluation framework for **Adversarial Bidirectional
Recovery RL for Safe Reasoning**. Builds on an already-trained AdvChain-style
SFT checkpoint and hardens it with a constrained, distributionally-robust,
self-mining attacker–defender RL loop.

The algorithm follows `FINAL-ALGO.md` (v3):

- **Component 1 — Constraint-balanced GRPO (λ controller).** Over-refusal is a
  constraint (`ORR ≤ τ`), not a hand-tuned data ratio. A Lagrange multiplier
  `λ` drives the harmful/benign **state-mixing ratio** each step.
- **Deepening B — CVaR tail selection.** The attacker samples the worst
  α-tail of candidate drifted states instead of a single noisy argmin.
- **Deepening G — Self-mined drift archive.** Failed recovery rollouts are
  clipped at the reasoning pivot and re-injected as new, empirically-validated
  drifts. Self-ReSET's passive buffer is a strict special case (an ablation
  row).
- **Deepening A — Trajectory-dense recovery reward.** Per-checkpoint guard
  scores are aggregated with a **non-telescoping** mean, attacking the
  `frac_reward_zero_std ≈ 0.9` reward-saturation problem.

## Hardware assumption

8 × 141 GB GPUs (e.g. H200 / H100-NVL-141G). Every stage is distributed via
`torchrun` and shards across all 8 cards. A single card holds the 7B policy +
Llama-Guard-3-8B judge + Qwen2.5-7B ORR classifier in fp32 (~60 GB), so 141 GB
is comfortable.

## Layout

```
advaudit/            # torch-free algorithm core (unit-tested) + torch model wrappers
  drift.py           # drift pool, splice, candidate enumeration
  cvar.py            # [B] CVaR worst-tail selection
  lambda_ctrl.py     # [Component 1] constraint controller
  rewards.py         # [A] trajectory-dense + dual-head reward, refusal/answer parsing
  archive.py         # [G] self-mine, UCB sampling, dedup, fictitious-play elites
  judges.py          # Llama-Guard-3 safety scorer + Qwen2.5 ORR classifier (torch)
  dist.py            # torch.distributed helpers (init, shard, gather)
scripts/
  gen_base_cots.py     # 8-GPU sharded on-policy base-CoT generation
  attacker_select.py   # 8-GPU sharded CVaR worst-case state selection
  grpo_defender.py     # 8-GPU GRPO defender (per-rank judge, traj reward, self-mine)
  run_recovery_rl.py   # orchestrator: rounds of attacker -> defender -> archive update
  run_eval.py          # 8-shard ASR/ORR eval (direct/inject) + merge
  eval_capability.py   # 8-shard MATH-500 / AIME
  download_benchmarks.py
configs/
  ds_zero3.json        # DeepSpeed ZeRO-3, fp32
  default.yaml         # all hyperparameters (algorithm + launch)
tests/                 # stdlib unittest for the torch-free core
run.sh                 # end-to-end reproduction commands
```

## Critical operational rules (from the blueprint)

- **fp32 everywhere.** bf16 makes R1-distill emit NaN.
- **Eval `max_new_tokens = 4096`.** Truncated reasoning inflates ASR because the
  extracted "answer" falls inside the still-drifted reasoning.
- **Multi-GPU judge → `cuda:{LOCAL_RANK}`** to avoid OOM piling onto one card.
- **λ enters via the state-mixing ratio, never as a reward multiplier** — GRPO's
  in-group normalization would cancel a constant reward scale.

## Quick start

```bash
conda activate claude
export ARIS_MODEL_HUB=modelscope HF_ENDPOINT=https://hf-mirror.com
# 1) benchmarks
python scripts/download_benchmarks.py --out_dir data
# 2) base CoTs from the SFT model (8-GPU)
bash run.sh gen_base_cots  /path/to/advchain-sft
# 3) recovery RL (8-GPU, rounds of attacker+defender)
bash run.sh train          /path/to/advchain-sft
# 4) eval (8-shard ASR/ORR)
bash run.sh eval           runs/abrr/round0
```

## Self-test

```bash
python -m pytest tests/ -q        # or: python -m unittest discover -s tests
python tools/compile_check.py     # py_compile every module
```

The torch-free core (drift/cvar/lambda_ctrl/rewards/archive) is fully unit
tested without a GPU. Distributed scripts are import/compile-checked; run them
on the 8-GPU cluster.
