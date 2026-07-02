#!/usr/bin/env python
"""8-shard capability eval: MATH-500 / AIME 2024 (Experiment C).

Greedy decoding, exact match on \\boxed{} or a trailing number. Two-phase
shard+merge like run_eval.py.

    torchrun --nproc_per_node 8 scripts/eval_capability.py \
        --model runs/abrr/round0/policy --data data/math500.jsonl \
        --out results/cap_rl0_math500.json
    python scripts/eval_capability.py --merge --out results/cap_rl0_math500.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advaudit.io_utils import read_jsonl  # noqa: E402

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUM = re.compile(r"(-?\d+(?:\.\d+)?)")


def extract_answer(text: str) -> str:
    m = list(_BOXED.finditer(text))
    if m:
        return m[-1].group(1).strip()
    nums = _NUM.findall(text)
    return nums[-1].strip() if nums else ""


def norm(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z.\-]", "", (s or "").strip()).lower()


def merge(out_path: str) -> None:
    base = out_path[:-5] if out_path.endswith(".json") else out_path
    shards = sorted(glob.glob(f"{base}.shard*.json"))
    if not shards:
        raise SystemExit(f"no shards for {base}")
    n = correct = 0
    for sf in shards:
        with open(sf) as f:
            c = json.load(f)
        n += c["n"]; correct += c["correct"]
    acc = round(100.0 * correct / max(n, 1), 2)
    with open(out_path, "w") as f:
        json.dump({"accuracy": acc, "n": n, "correct": correct}, f, indent=2)
    print(json.dumps({"accuracy": acc, "n": n}, indent=2))


def run_shard(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from advaudit import dist as D

    D.init_distributed()
    dev = D.device()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(dev).eval()

    rows = read_jsonl(args.data)
    mine = rows[D.rank():: D.world_size()]
    n = correct = 0
    for r in mine:
        chat = [{"role": "user", "content": r["problem"] if "problem" in r else r["prompt"]}]
        ids = tok.apply_chat_template(chat, add_generation_prompt=True,
                                      return_tensors="pt").to(dev)
        with torch.no_grad():
            gen = model.generate(ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        out = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(out)
        gt = str(r.get("answer", r.get("solution", "")))
        n += 1
        if norm(pred) and norm(pred) == norm(extract_answer(gt) or gt):
            correct += 1

    base = args.out[:-5] if args.out.endswith(".json") else args.out
    with open(f"{base}.shard{D.rank()}.json", "w") as f:
        json.dump({"n": n, "correct": correct}, f)
    print(f"[cap] rank {D.rank()}: {correct}/{n}")
    D.barrier(); D.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    if args.merge:
        merge(args.out); return
    if not (args.model and args.data):
        raise SystemExit("--model and --data required unless --merge")
    run_shard(args)


if __name__ == "__main__":
    main()
