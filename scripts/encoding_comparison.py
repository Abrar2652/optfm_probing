#!/usr/bin/env python3
"""
B4 — Input-encoding comparison: which encodings escape the 1-WL bound, at what cost.

RWPE is not the only encoding-level escape. We compare, on the 15-pair grid,
several standard input encodings as positive controls:

  baseline   — no encoding (the 1-WL bound; cos == 1)
  RNI        — random node init (rnf sigma=1.0), Abboud et al. 2020 / Sato 2021
  RWPE       — random-walk return probabilities (deterministic), Dwivedi 2022
  LapPE      — Laplacian eigenvectors (sign/basis ambiguity), Dwivedi & Bresson 2021
  LP-primal  — LP relaxation primal solution injected into a feature slot

For each (model, encoding) we report mean cos_sim and exact_frac (bit-identity).
Stochastic encodings (RNI) are averaged over K seeds. The contrast shows the
limitation lives in the *encoding*, and which encodings give a detectable vs.
provable escape (tie to A3).

Output: results/encoding_comparison.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs_v2 import construct_bipartite_cycle_pair, construct_cubic_bipartite_pair
from scripts._common import build_models, embed, cos_sim, linf, write_csv
from scripts.improvements import (
    make_rnf_transform, make_rwpe_transform, make_lappe_transform,
    lp_primal_transform,
)

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
RNF_SAMPLES = 20

ENCODINGS = {
    "baseline":  (lambda m, s: m, False, "none"),
    "RNI":       (make_rnf_transform(1.0), True, "stochastic; no convergence guarantee"),
    "RWPE":      (make_rwpe_transform(steps=(4, 6, 8)), False, "deterministic; < 2-FWL"),
    "LapPE":     (make_lappe_transform(k_eigs=3), False, "sign/basis ambiguity"),
    "LP-primal": (lp_primal_transform, False, "needs an LP solve"),
}


def run(model, pair, transform, stochastic):
    n = RNF_SAMPLES if stochastic else 1
    cs, lf = [], []
    for s in range(n):
        a = embed(model, transform(pair.milp_a, 100 + s))
        b = embed(model, transform(pair.milp_b, 200 + s))
        cs.append(cos_sim(a, b)); lf.append(linf(a, b))
    return float(np.mean(cs)), float(np.mean([x <= 1e-5 for x in lf]))


def main():
    pairs = [construct_bipartite_cycle_pair(k) for k in K_VALUES]
    pairs.append(construct_cubic_bipartite_pair())
    models = build_models(seed=0)

    rows = []
    for ename, (transform, stochastic, note) in ENCODINGS.items():
        for mname, m in models.items():
            cos_list, exact_list = [], []
            for p in pairs:
                c, e = run(m, p, transform, stochastic)
                cos_list.append(c); exact_list.append(e)
            rows.append([ename, mname, float(np.mean(cos_list)),
                         float(np.mean(exact_list)), note])
        sub = [r for r in rows if r[0] == ename]
        print(f"[{ename:9s}] mean_cos(min over models)={min(r[2] for r in sub):.4f}  "
              f"exact_frac(mean)={np.mean([r[3] for r in sub]):.2f}  ({note})")

    write_csv(ROOT / "results" / "encoding_comparison.csv",
              ["encoding", "model", "mean_cos", "exact_frac", "note"], rows)
    print("\nWrote results/encoding_comparison.csv")


if __name__ == "__main__":
    main()
