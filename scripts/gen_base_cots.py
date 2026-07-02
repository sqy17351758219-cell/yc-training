#!/usr/bin/env python
"""Generate on-policy base reasoning chains from the SFT model (8-GPU sharded).

The attacker splices drift into these chains, so they must come from the SAME
policy we start RL from. Launch with torchrun; rank r handles prompts[r::8] and
results are gathered to rank 0.

    torchrun --nproc_per_node 8 scripts/gen_base_cots.py \
        --model saves/advchain-v3-r1-distill-qwen-7b \
        --prompts data/advchain_traces.jsonl \
        --kind harmful \
        --out data/base_cots_v3.jsonl --max_new_tokens 2048

Each input row needs at least {"id", "prompt"}. Output rows:
{"id", "prompt", "kind", "base_cot"} where base_cot is the text BEFORE </think>.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advaudit import dist as D  # noqa: E402
from advaudit.io_utils import read_jsonl, write_jsonl  # noqa: E402

THINK_CLOSE = "</think>"


def extract_cot(text: str) -> str:
    """Keep only the reasoning up to (and excluding) the first </think>."""
    return text.split(THINK_CLOSE, 1)[0].strip() if THINK_CLOSE in text else text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--kind", choices=["harmful", "benign"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    D.init_distributed()
    dev = D.device()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32  # fp32 REQUIRED
    ).to(dev).eval()

    rows = read_jsonl(args.prompts)
    mine = D.shard(rows)

    out_local = []
    for r in mine:
        chat = [{"role": "user", "content": r["prompt"]}]
        input_ids = tok.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt"
        ).to(dev)
        with torch.no_grad():
            gen = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)
        out_local.append({
            "id": r["id"],
            "prompt": r["prompt"],
            "kind": args.kind,
            "base_cot": extract_cot(text),
        })

    all_rows = D.gather_objects(out_local)
    if D.is_main():
        n = write_jsonl(args.out, all_rows)
        print(f"[gen_base_cots] wrote {n} rows -> {args.out}")
    D.barrier()
    D.cleanup()


if __name__ == "__main__":
    main()
