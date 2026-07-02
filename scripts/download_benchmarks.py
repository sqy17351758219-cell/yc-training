#!/usr/bin/env python
"""Download the AdvChain evaluation suite into data/bench_*.jsonl (HF datasets).

Single-process (rank 0 of the pipeline). Uses HF_ENDPOINT mirror if set. Writes
one jsonl per benchmark with a uniform {"id", "prompt"} (+ "answer" for math).
WildJailbreak vanilla needs a locally-provided tsv (gated); pass --wj_train_tsv.

    export HF_ENDPOINT=https://hf-mirror.com
    python scripts/download_benchmarks.py --out_dir data \
        --wj_train_tsv data/wildjailbreak_train.tsv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advaudit.io_utils import write_jsonl  # noqa: E402


def _rows(prompts, prefix):
    return [{"id": f"{prefix}_{i}", "prompt": p} for i, p in enumerate(prompts) if p]


def load_hf(name, config=None, split="train"):
    from datasets import load_dataset
    return load_dataset(name, config, split=split)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--wj_train_tsv", default="", help="local WildJailbreak train.tsv (gated)")
    ap.add_argument("--wj_sample", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(0)

    def dump(name, rows):
        path = os.path.join(args.out_dir, f"bench_{name}.jsonl")
        n = write_jsonl(path, rows)
        print(f"  {name}: {n} -> {path}")

    print("[download] harmful sets")
    try:
        hb = load_hf("walledai/HarmBench", "standard", split="train")
        dump("harmbench", _rows([r["prompt"] for r in hb], "hb"))
    except Exception as e:
        print("  harmbench FAILED:", e)
    try:
        sr = load_hf("walledai/StrongREJECT", split="train")
        dump("strongreject", _rows([r["prompt"] for r in sr], "sr"))
    except Exception as e:
        print("  strongreject FAILED:", e)
    try:
        su = load_hf("thu-coai/SafeUnlearning", split="train")
        prompts = [r.get("question") or r.get("prompt") for r in su][:100]
        dump("safeunlearning", _rows(prompts, "su"))
    except Exception as e:
        print("  safeunlearning FAILED (try github mirror):", e)

    print("[download] benign / over-refusal sets")
    try:
        xs = load_hf("walledai/XSTest", split="test")
        safe = [r["prompt"] for r in xs if str(r.get("label", "")).lower() == "safe"]
        dump("xstest", _rows(safe[:250], "xs"))
    except Exception as e:
        print("  xstest FAILED:", e)

    # WildJailbreak: adversarial via HF eval config; vanilla via local gated tsv
    try:
        wj = load_hf("allenai/wildjailbreak", "eval", split="train")
        adv_h = [r["adversarial"] for r in wj if r.get("data_type") == "adversarial_harmful"]
        adv_b = [r["adversarial"] for r in wj if r.get("data_type") == "adversarial_benign"]
        dump("wj_adv_harm", _rows(rng.sample(adv_h, min(300, len(adv_h))), "wjah"))
        dump("wj_adv_benign", _rows(adv_b[:210], "wjab"))
    except Exception as e:
        print("  wildjailbreak(eval) FAILED:", e)

    if args.wj_train_tsv and os.path.exists(args.wj_train_tsv):
        with open(args.wj_train_tsv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            vh, vb = [], []
            for row in reader:
                dt = row.get("data_type", "")
                if dt == "vanilla_harmful":
                    vh.append(row.get("vanilla", ""))
                elif dt == "vanilla_benign":
                    vb.append(row.get("vanilla", ""))
        dump("wj_vani_harm", _rows(rng.sample(vh, min(args.wj_sample, len(vh))), "wjvh"))
        dump("wj_vani_benign", _rows(rng.sample(vb, min(300, len(vb))), "wjvb"))
    else:
        print("  WJ vanilla skipped (no --wj_train_tsv; gated dataset)")

    print("[download] capability sets")
    try:
        m5 = load_hf("HuggingFaceH4/MATH-500", split="test")
        rows = [{"id": f"m5_{i}", "problem": r["problem"], "answer": r["answer"]}
                for i, r in enumerate(m5)]
        write_jsonl(os.path.join(args.out_dir, "math500.jsonl"), rows)
        print(f"  math500: {len(rows)}")
    except Exception as e:
        print("  math500 FAILED:", e)

    print("[download] done")


if __name__ == "__main__":
    main()
