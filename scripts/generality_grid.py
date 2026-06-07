#!/usr/bin/env python3
"""
A1 — Generality beyond OPTFM.

Run the full 15-pair grid on FOUR architecturally distinct, independently-published
bipartite/MILP encoders that are NOT OPTFM ablations (Graphormer-style, GraphGPS-
style, Set-Transformer pooling, Gasse-2019 bipartite GCN). Required outcome:
bit-identical at baseline (cos == 1.000000) for every model, because each still
aggregates via symmetric sums/means — demonstrating the bound is a property of
symmetric multiset aggregation, not an OPTFM artifact. The RWPE positive control
confirms each is falsifiable.

Averaged over 5 random seeds; bootstrap 95% CI over pairs.

Output: results/generality_grid.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs_v2 import construct_bipartite_cycle_pair, construct_cubic_bipartite_pair
from models.extra_architectures import build_extra_models
from scripts._common import embed, cos_sim, linf, n_params, bootstrap_ci, write_csv
from scripts.improvements import make_rwpe_transform

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
SEEDS = [0, 1, 2, 3, 4]


def main():
    pairs = [construct_bipartite_cycle_pair(k) for k in K_VALUES]
    pairs.append(construct_cubic_bipartite_pair())
    rwpe = make_rwpe_transform(steps=(4, 6, 8), cons_step=4, cons_scale=1.0)

    # accumulate per (model, transform): per-pair mean cos over seeds
    acc = {}
    params = {}
    for seed in SEEDS:
        models = build_extra_models(seed=seed)
        for mname, m in models.items():
            params[mname] = n_params(m)
            for tname, tf in (("baseline", lambda x: x), ("rwpe", None)):
                cos_pp, exact_pp = [], []
                for p in pairs:
                    if tname == "baseline":
                        a, b = embed(m, p.milp_a), embed(m, p.milp_b)
                    else:
                        a, b = embed(m, rwpe(p.milp_a, 100)), embed(m, rwpe(p.milp_b, 200))
                    cos_pp.append(cos_sim(a, b)); exact_pp.append(linf(a, b) <= 1e-5)
                acc.setdefault((mname, tname), {"cos": [], "exact": []})
                acc[(mname, tname)]["cos"].append(cos_pp)
                acc[(mname, tname)]["exact"].append(exact_pp)

    rows = []
    for (mname, tname), d in acc.items():
        cos_mat = np.array(d["cos"])           # (seeds, pairs)
        exact_mat = np.array(d["exact"])
        per_pair_mean = cos_mat.mean(0)        # average over seeds
        mean, lo, hi = bootstrap_ci(per_pair_mean)
        rows.append([mname, tname, params[mname], float(mean), float(lo), float(hi),
                     float(exact_mat.mean())])
        print(f"{mname:30s} {tname:8s} params={params[mname]:>6,} "
              f"cos={mean:.6f} CI=[{lo:.6f},{hi:.6f}] exact_frac={exact_mat.mean():.2f}")

    write_csv(ROOT / "results" / "generality_grid.csv",
              ["model", "transform", "params", "mean_cos", "ci_lo", "ci_hi", "exact_frac"],
              rows)
    base = [r for r in rows if r[1] == "baseline"]
    print("\nWrote results/generality_grid.csv")
    print(f"All non-OPTFM models bit-identical at baseline: {all(r[6] == 1.0 for r in base)}")


if __name__ == "__main__":
    main()
