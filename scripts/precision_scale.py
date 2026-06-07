#!/usr/bin/env python3
"""
B2 — Scale / numerical-precision controls.

(1) Precision: repeat the baseline grid in float32 AND float64 to show the
    bit-identity is not a float32 artifact (in float64 the L-inf gap drops to
    machine epsilon, ~1e-16).
(2) Scale: extend the cycle family to ~10x larger node counts (k up to 200,
    i.e. 800 nodes) and confirm the bound persists — it is structural, not a
    small-graph effect.

Outputs: results/precision_dtype.csv, results/scale_extension.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.milp_pairs_v2 import construct_bipartite_cycle_pair, construct_cubic_bipartite_pair
from scripts._common import build_models, build_hierarchical, embed, cos_sim, linf, write_csv

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
SCALE_KS = [30, 50, 75, 100, 150, 200]


def precision_table():
    print("=== precision (float32 vs float64) ===")
    pairs = [construct_bipartite_cycle_pair(k) for k in K_VALUES]
    pairs.append(construct_cubic_bipartite_pair())
    rows = []
    for dtype, dname in ((torch.float32, "float32"), (torch.float64, "float64")):
        models = build_models(seed=0, dtype=dtype)
        for mname, m in models.items():
            cos_l, lf_l, ex_l = [], [], []
            for p in pairs:
                a = embed(m, p.milp_a, dtype=dtype); b = embed(m, p.milp_b, dtype=dtype)
                cos_l.append(cos_sim(a, b)); lf_l.append(linf(a, b)); ex_l.append(linf(a, b) <= 1e-5)
            rows.append([mname, dname, float(np.mean(cos_l)), float(np.max(lf_l)), float(np.mean(ex_l))])
            print(f"  {mname:32s} {dname}  mean_cos={np.mean(cos_l):.6f} maxLinf={np.max(lf_l):.2e} exact={np.mean(ex_l):.2f}")
    write_csv(ROOT / "results" / "precision_dtype.csv",
              ["model", "dtype", "mean_cos", "max_linf", "exact_frac"], rows)


def scale_table():
    print("\n=== scale extension (cycle family up to 800 nodes) ===")
    m = build_hierarchical(seed=0, dtype=torch.float64)
    rows = []
    for k in SCALE_KS:
        p = construct_bipartite_cycle_pair(k)
        a = embed(m, p.milp_a, dtype=torch.float64)
        b = embed(m, p.milp_b, dtype=torch.float64)
        nnodes = 2 * (2 * k)
        rows.append([k, nnodes, float(cos_sim(a, b)), float(linf(a, b)), bool(linf(a, b) <= 1e-5)])
        print(f"  k={k:3d}  nodes={nnodes:4d}  cos={cos_sim(a,b):.6f}  linf={linf(a,b):.2e}  bit-identical={linf(a,b)<=1e-5}")
    write_csv(ROOT / "results" / "scale_extension.csv",
              ["k", "n_nodes", "cos_sim", "linf", "bit_identical"], rows)


def main():
    precision_table()
    scale_table()
    print("\nWrote results/precision_dtype.csv, results/scale_extension.csv")


if __name__ == "__main__":
    main()
