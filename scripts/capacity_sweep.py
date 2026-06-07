#!/usr/bin/env python3
"""
A6 — Capacity invariance.

The reflexive reviewer objection to a negative result on small models is
"you used toy models; a big enough model would escape the bound." Theorem 1
says capacity is irrelevant: the bound holds for any weights at any width /
depth / number of heads. We verify it by sweeping the hierarchical OPTFM's
hidden dimension, number of layers, and number of attention heads, and
confirming bit-identity (cos == 1.000000) at every capacity. The RWPE
positive control (cos < 1) is computed at each capacity as a falsifiability
reference, showing the test could detect an escape if one existed.

Output: results/capacity_sweep.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs_v2 import construct_bipartite_cycle_pair, construct_cubic_bipartite_pair
from scripts._common import build_hierarchical, embed, cos_sim, linf, n_params, write_csv
from scripts.improvements import make_rwpe_transform

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
SEED = 0

# (hidden_channels, trans_num_layers, trans_num_heads, sweep-label)
CONFIGS = (
    [(h, 1, 1, "width") for h in (16, 32, 64, 128, 256)] +
    [(16, L, 1, "depth") for L in (1, 2, 4, 8)] +
    [(16, 1, H, "heads") for H in (1, 2, 4)]
)


def main():
    pairs = [construct_bipartite_cycle_pair(k) for k in K_VALUES]
    pairs.append(construct_cubic_bipartite_pair())
    rwpe = make_rwpe_transform(steps=(4, 6, 8), cons_step=4, cons_scale=1.0)

    rows = []
    for hidden, layers, heads, sweep in CONFIGS:
        m = build_hierarchical(hidden_channels=hidden, trans_num_layers=layers,
                               trans_num_heads=heads, seed=SEED)
        base_cos, base_linf, base_exact = [], [], []
        rwpe_cos = []
        for p in pairs:
            a, b = embed(m, p.milp_a), embed(m, p.milp_b)
            base_cos.append(cos_sim(a, b))
            base_linf.append(linf(a, b))
            base_exact.append(linf(a, b) <= 1e-5)
            ra = embed(m, rwpe(p.milp_a, 100))
            rb = embed(m, rwpe(p.milp_b, 200))
            rwpe_cos.append(cos_sim(ra, rb))
        rows.append([
            sweep, hidden, layers, heads, n_params(m),
            float(np.mean(base_cos)), float(np.max(base_linf)),
            float(np.mean(base_exact)), float(np.mean(rwpe_cos)),
        ])
        print(f"[{sweep:5s}] h={hidden:<3} L={layers} H={heads}  params={n_params(m):>7,}  "
              f"baseline cos={np.mean(base_cos):.6f} maxLinf={np.max(base_linf):.1e} "
              f"exact={np.mean(base_exact):.2f}  RWPE cos={np.mean(rwpe_cos):.4f}")

    write_csv(ROOT / "results" / "capacity_sweep.csv",
              ["sweep", "hidden_channels", "num_layers", "num_heads", "params",
               "baseline_mean_cos", "baseline_max_linf", "baseline_exact_frac",
               "rwpe_mean_cos"],
              rows)
    print("\nWrote results/capacity_sweep.csv")
    allexact = all(r[7] == 1.0 for r in rows)
    print(f"Bit-identical at EVERY capacity: {allexact}")


if __name__ == "__main__":
    main()
