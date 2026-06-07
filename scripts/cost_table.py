#!/usr/bin/env python3
"""
C3 — Wall-clock / parameter-count honesty table.

Reports parameter count and mean forward-pass wall-clock per architecture, so the
"fancy architecture, same expressiveness" point lands with cost context: the
hierarchical multi-view cross-attention costs an order of magnitude more parameters
and compute than a plain GCN yet yields the identical (bit-identical) embedding on
1-WL-equivalent inputs.

Wall-clock note: timings use time.perf_counter; absolute numbers are machine- and
load-dependent (CPU). We pass a fixed timestamp-free seed; the runtime forbids
Date.now in workflows but perf_counter is fine in a plain script.

Output: results/cost_table.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs_v2 import construct_bipartite_cycle_pair
from models.extra_architectures import build_extra_models
from scripts._common import build_models, embed, n_params, write_csv

REPS = 50


def time_model(m, milp):
    # warmup
    for _ in range(5):
        embed(m, milp)
    t = time.perf_counter()
    for _ in range(REPS):
        embed(m, milp)
    return (time.perf_counter() - t) / REPS * 1e3   # ms per forward


def main():
    milp = construct_bipartite_cycle_pair(10).milp_a    # 40-node graph
    models = build_models(seed=0)
    models.update(build_extra_models(seed=0))
    rows = []
    for name, m in models.items():
        ms = time_model(m, milp)
        rows.append([name, n_params(m), float(ms)])
        print(f"  {name:32s} params={n_params(m):>9,}  {ms:7.3f} ms/forward")
    # sort by params for the table
    rows.sort(key=lambda r: r[1])
    write_csv(ROOT / "results" / "cost_table.csv",
              ["model", "params", "ms_per_forward"], rows)
    print("\nWrote results/cost_table.csv")


if __name__ == "__main__":
    main()
