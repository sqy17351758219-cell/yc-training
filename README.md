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

## Cluster runbook (8×141GB)

Full end-to-end command sequence. See `RUNBOOK.md` for the annotated version
(per-command "equivalent manual" torchrun lines, ablation switches, pitfalls).

### 0. Variables — edit to your paths

```bash
export REPO=/root/yc-training
export SFT=/root/grpo/saves/advchain-v3-r1-distill-qwen-7b   # your trained AdvChain SFT model
```

### 1. Code + environment

```bash
git clone https://github.com/sqy17351758219-cell/yc-training.git $REPO
cd $REPO
conda activate claude                    # base env transformers is broken
export ARIS_MODEL_HUB=modelscope
export HF_ENDPOINT=https://hf-mirror.com
pip install -r requirements.txt          # skip if trl0.19.1/transformers4.52.4 already installed
```

### 2. Self-test (no GPU)

```bash
python -m unittest discover -s tests     # expect 52 passing
python tools/compile_check.py            # expect all compile
```

### 3. Data — reuse the already-downloaded grpo/data

If benchmarks, `base_cots_v3.jsonl`, `base_cots_benign_v3.jsonl`, and
`math500.jsonl` are already under `/root/grpo/data`, just symlink it in; run.sh
reads `./data` by default. No re-download, no re-gen.

```bash
ln -s /root/grpo/data $REPO/data
ls -1 $REPO/data/ | grep -E 'base_cots|bench_|math500|aime'   # verify names below
```

The scripts expect these exact filenames (symlink or rename mismatches, or edit
`HARMFUL_BENCH`/`BENIGN_BENCH` at the top of `scripts/run_eval.py`):

| Purpose | Expected filename |
|---|---|
| attacker base CoTs | `base_cots_v3.jsonl`, `base_cots_benign_v3.jsonl` |
| ASR harmful (5) | `bench_harmbench.jsonl`, `bench_strongreject.jsonl`, `bench_safeunlearning.jsonl`, `bench_wj_vani_harm.jsonl`, `bench_wj_adv_harm.jsonl` |
| ORR benign (3) | `bench_xstest.jsonl`, `bench_wj_vani_benign.jsonl`, `bench_wj_adv_benign.jsonl` |
| capability | `math500.jsonl` (`{"problem","answer"}` per line), `aime2024.jsonl` |

Each `bench_*.jsonl` line needs `{"id","prompt"}` (inject mode also uses
`base_cot` if present; absent → falls back to direct concat).

If the data is NOT already present, download + generate instead:

```bash
python scripts/download_benchmarks.py --out_dir data --wj_train_tsv /path/to/wj_train.tsv
bash run.sh gen_base_cots $SFT           # 8-GPU on-policy base CoTs from the SFT model
```

### 4. Smoke test (real distributed link — run this FIRST, stops after 2 steps)

```bash
cp configs/default.yaml configs/smoke.yaml   # edit: rl.steps_per_round:2  rl.per_side_limit:8
python scripts/run_recovery_rl.py --sft $SFT \
  --base_harmful data/base_cots_v3.jsonl --base_benign data/base_cots_benign_v3.jsonl \
  --config configs/smoke.yaml --out runs/smoke 2>&1 | tee logs/smoke.log
```

Watch for: all 8 ranks start; `runs/smoke/round0/states.jsonl` non-empty; judge
sits on `cuda:{LOCAL_RANK}` (not piling on card 0); TRL 0.19.1 accepts the
`reward_funcs=reward_fn` signature; `policy` + `mined.jsonl` saved.

### 5. Full training (round 0)

```bash
python scripts/run_recovery_rl.py --sft $SFT \
  --base_harmful data/base_cots_v3.jsonl --base_benign data/base_cots_benign_v3.jsonl \
  --config configs/default.yaml --out runs/abrr 2>&1 | tee logs/abrr_round0.log
# -> runs/abrr/round0/policy , archive.json , round0/lambda_history.json
```

### 6. Evaluation (8-shard + merge)

```bash
bash run.sh eval runs/abrr/round0/policy direct   # main table ASR + ORR
bash run.sh eval runs/abrr/round0/policy inject   # Experiment B/B' OOD injection
bash run.sh eval $SFT direct                       # SFT baseline column
bash run.sh eval $SFT inject
```

### 7. Capability (Experiment C)

```bash
bash run.sh capability runs/abrr/round0/policy
bash run.sh capability $SFT
```

## Self-test

```bash
python -m pytest tests/ -q        # or: python -m unittest discover -s tests
python tools/compile_check.py     # py_compile every module
```

The torch-free core (drift/cvar/lambda_ctrl/rewards/archive) is fully unit
tested without a GPU. Distributed scripts are import/compile-checked; run them
on the 8-GPU cluster.
