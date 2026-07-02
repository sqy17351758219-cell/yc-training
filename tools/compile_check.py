#!/usr/bin/env python
"""py_compile every .py in the repo (syntax check without importing torch)."""
from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".venv", "env", "__pycache__", ".git"}


def main() -> int:
    failed = []
    for p in ROOT.rglob("*.py"):
        if any(part in SKIP for part in p.parts):
            continue
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as e:
            failed.append((p, str(e)))
    if failed:
        for p, e in failed:
            print(f"FAIL {p.relative_to(ROOT)}\n     {e}")
        print(f"\n{len(failed)} file(s) failed to compile")
        return 1
    print(f"OK: all python files compile ({ROOT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
