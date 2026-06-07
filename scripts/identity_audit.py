#!/usr/bin/env python3
"""
T3 — Numerical identity audit.

Beyond reporting cos == 1.0, quantify *how* identical the embeddings of a
1-WL-equivalent non-isomorphic pair are. We collect every per-coordinate
|Phi(G_A) - Phi(G_B)| across all pairs x seeds x architectures, in both
float32 and float64, and show the distribution sits at floating-point epsilon
(far below the 1e-5 bit-identity threshold). This pre-empts the reviewer
objection that "1.000000" is a rounding artifact.

Outputs:
  results/identity_audit.csv          per-(model,dtype) summary statistics
  results/identity_audit_diffs.npz    raw |diff| arrays for the histogram (Fig 8)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.milp_pairs_v2 import construct_bipartite_cycle_pair, construct_cubic_bipartite_pair
from scripts._common import build_models, embed, write_csv

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
SEEDS = [0, 1, 2, 3, 4]
ATOL = 1e-5


def main():
    pairs = [construct_bipartite_cycle_pair(k) for k in K_VALUES]
    pairs.append(construct_cubic_bipartite_pair())

    rows = []
    diff_store = {}
    for dtype, dname in ((torch.float32, "float32"), (torch.float64, "float64")):
        # accumulate abs diffs per model name across seeds+pairs
        per_model = {}
        for seed in SEEDS:
            models = build_models(seed=seed, dtype=dtype, include_pretrained=(seed == 0))
            for mname, m in models.items():
                for p in pairs:
                    a = embed(m, p.milp_a, dtype=dtype)
                    b = embed(m, p.milp_b, dtype=dtype)
                    d = (a - b).abs().cpu().numpy().astype(np.float64)
                    per_model.setdefault(mname, []).append(d)
        for mname, chunks in per_model.items():
            alld = np.concatenate(chunks)
            key = f"{mname}|{dname}"
            diff_store[key] = alld
            rows.append([
                mname, dname, int(alld.size),
                float(alld.max()), float(alld.mean()),
                float(np.quantile(alld, 0.5)), float(np.quantile(alld, 0.99)),
                float(np.mean(alld <= ATOL)),
            ])
            print(f"{mname:32s} {dname}  max={alld.max():.2e}  mean={alld.mean():.2e}  "
                  f"frac<=1e-5={np.mean(alld <= ATOL):.4f}")

    write_csv(ROOT / "results" / "identity_audit.csv",
              ["model", "dtype", "n_coords", "max_abs_diff", "mean_abs_diff",
               "median_abs_diff", "p99_abs_diff", "frac_within_atol"],
              rows)
    # Save raw diffs (capped per key to keep file small) for the histogram figure.
    np.savez_compressed(ROOT / "results" / "identity_audit_diffs.npz",
                        **{k.replace("|", "__"): v[:5000] for k, v in diff_store.items()})
    print(f"\nWrote results/identity_audit.csv and results/identity_audit_diffs.npz")
    print(f"Overall max |diff| across all cells: "
          f"{max(v.max() for v in diff_store.values()):.2e} (threshold {ATOL})")


if __name__ == "__main__":
    main()
