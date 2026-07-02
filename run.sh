#!/usr/bin/env bash
# End-to-end reproduction commands for 8x141GB. Usage: bash run.sh <stage> [args]
set -euo pipefail

NPROC=8
CONFIG=configs/default.yaml
DATA=data
OUT=runs/abrr

export ARIS_MODEL_HUB=${ARIS_MODEL_HUB:-modelscope}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# DeepSpeed needs env propagated to all ranks:
printf 'ARIS_MODEL_HUB=%s\nHF_ENDPOINT=%s\n' "$ARIS_MODEL_HUB" "$HF_ENDPOINT" > .deepspeed_env

stage="${1:-help}"

case "$stage" in
  download)
    python scripts/download_benchmarks.py --out_dir "$DATA" "${@:2}"
    ;;

  gen_base_cots)
    SFT="${2:?usage: run.sh gen_base_cots <sft_ckpt>}"
    torchrun --standalone --nproc_per_node=$NPROC scripts/gen_base_cots.py \
      --model "$SFT" --prompts "$DATA/advchain_traces.jsonl" --kind harmful \
      --out "$DATA/base_cots_v3.jsonl" --max_new_tokens 2048
    torchrun --standalone --nproc_per_node=$NPROC scripts/gen_base_cots.py \
      --model "$SFT" --prompts "$DATA/rl_benign_prompts.jsonl" --kind benign \
      --out "$DATA/base_cots_benign_v3.jsonl" --max_new_tokens 2048
    ;;

  train)
    SFT="${2:?usage: run.sh train <sft_ckpt>}"
    python scripts/run_recovery_rl.py \
      --sft "$SFT" \
      --base_harmful "$DATA/base_cots_v3.jsonl" \
      --base_benign  "$DATA/base_cots_benign_v3.jsonl" \
      --config "$CONFIG" --out "$OUT"
    ;;

  eval)
    MODEL="${2:?usage: run.sh eval <model_dir> [direct|inject]}"
    MODE="${3:-direct}"
    RES="results/$(basename "$MODEL")_${MODE}.json"
    mkdir -p results
    torchrun --standalone --nproc_per_node=$NPROC scripts/run_eval.py \
      --model "$MODEL" --data_dir "$DATA" --mode "$MODE" \
      --orr_mode llm --config "$CONFIG" --out "$RES"
    python scripts/run_eval.py --merge --out "$RES"
    ;;

  capability)
    MODEL="${2:?usage: run.sh capability <model_dir>}"
    mkdir -p results
    for ds in math500 aime2024; do
      [ -f "$DATA/$ds.jsonl" ] || { echo "skip $ds (missing)"; continue; }
      RES="results/$(basename "$MODEL")_${ds}.json"
      torchrun --standalone --nproc_per_node=$NPROC scripts/eval_capability.py \
        --model "$MODEL" --data "$DATA/$ds.jsonl" --out "$RES"
      python scripts/eval_capability.py --merge --out "$RES"
    done
    ;;

  selftest)
    python -m unittest discover -s tests -v
    python tools/compile_check.py
    ;;

  *)
    echo "stages: download | gen_base_cots <sft> | train <sft> | eval <model> [direct|inject] | capability <model> | selftest"
    ;;
esac
