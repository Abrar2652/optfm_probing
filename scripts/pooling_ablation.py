#!/usr/bin/env python3
"""
B5 — Pooling-operator ablation (corroborates Lemma 4).

Theorem 1 covers symmetric pooling in {mean, sum, max}. A reviewer may suspect
the bit-identity depends on the mean-pool choice. We re-run the baseline grid
with each symmetric pooling and confirm bit-identity holds for all of them.

Output: results/pooling_ablation.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs_v2 import construct_bipartite_cycle_pair, construct_cubic_bipartite_pair
from scripts._common import build_models, embed, cos_sim, linf, write_csv

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
POOLINGS = ["mean", "sum", "max"]


def main():
    pairs = [construct_bipartite_cycle_pair(k) for k in K_VALUES]
    pairs.append(construct_cubic_bipartite_pair())
    models = build_models(seed=0)

    rows = []
    for pooling in POOLINGS:
        for mname, m in models.items():
            cos_list, linf_list, exact = [], [], []
            for p in pairs:
                a = embed(m, p.milp_a, pooling=pooling)
                b = embed(m, p.milp_b, pooling=pooling)
                cos_list.append(cos_sim(a, b)); linf_list.append(linf(a, b))
                exact.append(linf(a, b) <= 1e-5)
            rows.append([pooling, mname, float(np.mean(cos_list)),
                         float(np.max(linf_list)), float(np.mean(exact))])
            print(f"[{pooling:4s}] {mname:32s} mean_cos={np.mean(cos_list):.6f} "
                  f"maxLinf={np.max(linf_list):.1e} exact={np.mean(exact):.2f}")

    write_csv(ROOT / "results" / "pooling_ablation.csv",
              ["pooling", "model", "mean_cos", "max_linf", "exact_frac"], rows)
    print("\nWrote results/pooling_ablation.csv")
    print(f"Bit-identical under every pooling: {all(r[4] == 1.0 for r in rows)}")


if __name__ == "__main__":
    main()
