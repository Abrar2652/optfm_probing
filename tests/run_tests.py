#!/usr/bin/env python3
"""
Lightweight test runner (pytest not required). Discovers every `test_*`
function in tests/test_*.py, runs it, prints PASS/FAIL, writes a log to
results/tests.txt, and exits nonzero on any failure.
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODULES = ["tests.test_pairs", "tests.test_encoders"]


def main():
    lines = []
    n_pass = n_fail = 0
    for modname in MODULES:
        mod = importlib.import_module(modname)
        for fname in sorted(d for d in dir(mod) if d.startswith("test_")):
            fn = getattr(mod, fname)
            if not callable(fn):
                continue
            try:
                fn()
                msg = f"PASS {modname}.{fname}"
                n_pass += 1
            except Exception:
                msg = f"FAIL {modname}.{fname}\n{traceback.format_exc()}"
                n_fail += 1
            print(msg)
            lines.append(msg)
    summary = f"\n{n_pass} passed, {n_fail} failed"
    print(summary)
    lines.append(summary)
    out = ROOT / "results" / "tests.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
