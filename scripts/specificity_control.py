#!/usr/bin/env python3
"""
A7 — Non-uniform-feature pairs + specificity (negative) control.

Two soft spots in a bit-identity result:
  (i)  the collision might be an artifact of all-zero / uniform node features;
  (ii) the encoders might be near-constant maps, making cos==1 vacuous.

This script addresses both:
  * Non-uniform pairs: heterogeneous (type-uniform, non-zero) and gadget-union
    (genuinely node-varied) 1-WL-equivalent non-isomorphic pairs. We verify each
    is 1-WL equivalent AND non-isomorphic, then confirm bit-identity persists.
  * Specificity control: 1-WL-DISTINGUISHABLE pairs must yield DIFFERENT
    embeddings (linf > 1e-5) for the expressive architectures.

Outputs: results/nonuniform_pairs.csv, results/specificity_control.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs import verify_1wl_equivalence
from data.milp_pairs_v2 import _bipartite_components, are_isomorphic_bipartite
from data.construct_pairs import (
    construct_heterogeneous_cycle_pair, construct_gadget_union_pair,
    construct_distinguishable_pairs,
)
from scripts._common import build_models, embed, cos_sim, linf, write_csv

K_VALUES = [2, 3, 4, 5, 6, 8, 10]


def verify(pair):
    a, b = pair.milp_a, pair.milp_b
    eq = verify_1wl_equivalence(a, b)
    ca, cb = _bipartite_components(a.A), _bipartite_components(b.A)
    # for small graphs also run full iso check; else rely on component invariant
    if a.A.shape[0] <= 6 and a.A.shape[1] <= 6:
        noniso = not are_isomorphic_bipartite(a, b)
    else:
        noniso = ca != cb
    return eq, noniso, ca, cb


def main():
    # ---- Part 1: non-uniform pairs ------------------------------------------
    families = {
        "heterogeneous": [construct_heterogeneous_cycle_pair(k) for k in K_VALUES],
        "gadget_union":  [construct_gadget_union_pair(k) for k in K_VALUES],
    }
    models = build_models(seed=0)

    nonuniform_rows = []
    for fam, pairs in families.items():
        for p in pairs:
            eq, noniso, ca, cb = verify(p)
            assert eq, f"{p.description}: NOT 1-WL equivalent — construction invalid"
            assert noniso, f"{p.description}: isomorphic — construction invalid"
            for mname, m in models.items():
                a, b = embed(m, p.milp_a), embed(m, p.milp_b)
                nonuniform_rows.append([
                    fam, p.description, mname, ca, cb,
                    float(cos_sim(a, b)), float(linf(a, b)),
                    bool(linf(a, b) <= 1e-5),
                ])
        # summary print
        for mname in models:
            sub = [r for r in nonuniform_rows if r[0] == fam and r[2] == mname]
            mc = np.mean([r[5] for r in sub]); ex = np.mean([r[7] for r in sub])
            print(f"[{fam:13s}] {mname:32s} mean_cos={mc:.6f} exact_frac={ex:.2f}")

    write_csv(ROOT / "results" / "nonuniform_pairs.csv",
              ["family", "pair", "model", "A_components", "B_components",
               "cos_sim", "linf", "bit_identical"], nonuniform_rows)

    # ---- Part 2: specificity (negative) control -----------------------------
    dpairs = construct_distinguishable_pairs()
    spec_rows = []
    for p in dpairs:
        n_distinguish = 0
        for mname, m in models.items():
            a, b = embed(m, p.milp_a), embed(m, p.milp_b)
            distinguished = linf(a, b) > 1e-5
            n_distinguish += int(distinguished)
            spec_rows.append([p.description, mname, float(cos_sim(a, b)),
                              float(linf(a, b)), bool(distinguished)])
        print(f"[specificity] {p.description:38s} models_distinguishing={n_distinguish}/{len(models)}")
        assert n_distinguish >= 1, f"VACUOUS: no model separates {p.description}"

    write_csv(ROOT / "results" / "specificity_control.csv",
              ["pair", "model", "cos_sim", "linf", "distinguished"], spec_rows)

    print("\nWrote results/nonuniform_pairs.csv, results/specificity_control.csv")
    all_bit = all(r[7] for r in nonuniform_rows)
    print(f"Non-uniform pairs all bit-identical: {all_bit}")


if __name__ == "__main__":
    main()
