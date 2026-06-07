#!/usr/bin/env python3
"""
A2 — Naturalistic / standard-family pairs + a pair-construction algorithm + a
natural-collision prevalence scan.

Part 1 (construction): data.construct_pairs.construct_folded_pair is the general
pair-construction algorithm (connected d-regular vs k disjoint K_{d,d}) with proved
guarantees. We instantiate it dressed as four standard MILP families (set cover,
combinatorial auction, matching, multi-knapsack-style blocks), verify each is 1-WL
equivalent AND non-isomorphic, and run the full architecture grid: required outcome
bit-identical.

Part 2 (prevalence): generate random instances of two standard families and measure
how often natural 1-WL degeneracy arises (nodes sharing a 1-WL colour class), and
how many random instance pairs are 1-WL equivalent. This honestly characterizes when
the bound bites in the wild.

Outputs: results/naturalistic_pairs.csv, results/naturalistic_prevalence.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs import verify_1wl_equivalence
from data.milp_pairs_v2 import _bipartite_components
from data.construct_pairs import (
    construct_standard_family_pair, gen_random_set_cover, gen_random_comb_auction,
    wl_color_class_count,
)
from scripts._common import build_models, embed, cos_sim, linf, write_csv

FAMILIES = ["set_cover", "comb_auction", "matching", "knapsack_blocks"]
K_VALUES = [2, 3, 4, 5, 6, 8, 10]


def main():
    models = build_models(seed=0)

    # ---- Part 1: standard-family pairs --------------------------------------
    rows = []
    for fam in FAMILIES:
        for k in K_VALUES:
            p = construct_standard_family_pair(fam, k)
            eq = verify_1wl_equivalence(p.milp_a, p.milp_b)
            ca, cb = _bipartite_components(p.milp_a.A), _bipartite_components(p.milp_b.A)
            assert eq and ca != cb, f"{fam} k={k}: construction invalid (eq={eq}, {ca}vs{cb})"
            for mname, m in models.items():
                a, b = embed(m, p.milp_a), embed(m, p.milp_b)
                rows.append([fam, k, mname, ca, cb,
                             float(cos_sim(a, b)), float(linf(a, b)),
                             bool(linf(a, b) <= 1e-5)])
        for mname in models:
            sub = [r for r in rows if r[0] == fam and r[2] == mname]
            print(f"[{fam:15s}] {mname:32s} mean_cos={np.mean([r[5] for r in sub]):.6f} "
                  f"exact={np.mean([r[7] for r in sub]):.2f}")

    write_csv(ROOT / "results" / "naturalistic_pairs.csv",
              ["family", "k", "model", "A_components", "B_components",
               "cos_sim", "linf", "bit_identical"], rows)

    # ---- Part 2: natural-collision prevalence scan --------------------------
    print("\n=== prevalence scan over random standard-family instances ===")
    rng = np.random.default_rng(0)
    prev_rows = []
    N = 300
    for fam, gen, sizes in (
        ("set_cover", gen_random_set_cover, (10, 12)),
        ("comb_auction", gen_random_comb_auction, (10, 12)),
    ):
        insts = [gen(sizes[0], sizes[1], 0.3, rng) for _ in range(N)]
        # degeneracy: fraction of instances with WL-twin nodes (classes < nodes)
        degen = []
        for inst in insts:
            n_nodes = inst.A.shape[0] + inst.A.shape[1]
            cc = wl_color_class_count(inst)
            degen.append(cc < n_nodes)
        # natural 1-WL-equivalent pairs among random samples
        n_pairs, n_equiv = 0, 0
        for i in range(0, N - 1, 2):
            n_pairs += 1
            if insts[i].A.shape == insts[i + 1].A.shape and \
               verify_1wl_equivalence(insts[i], insts[i + 1]):
                n_equiv += 1
        frac_degen = float(np.mean(degen))
        frac_equiv = n_equiv / max(n_pairs, 1)
        prev_rows.append([fam, N, sizes[0], sizes[1], frac_degen, n_equiv, n_pairs])
        print(f"  {fam:13s}: {frac_degen*100:.1f}% instances have WL-twin nodes; "
              f"{n_equiv}/{n_pairs} random pairs are 1-WL-equivalent")

    write_csv(ROOT / "results" / "naturalistic_prevalence.csv",
              ["family", "n_instances", "m", "n", "frac_with_wl_twins",
               "n_equiv_pairs", "n_pairs"], prev_rows)
    print("\nWrote results/naturalistic_pairs.csv, results/naturalistic_prevalence.csv")
    base_ok = all(r[7] for r in rows)
    print(f"All standard-family pairs bit-identical: {base_ok}")


if __name__ == "__main__":
    main()
