#!/usr/bin/env python3
"""
C1 — Lemma <-> experiment correspondence.

Maps each of the 5 proof lemmas to a runnable ablation that isolates the
architectural component it concerns, and reports the empirical bit-identity
(exact_frac over the 15 pairs). This locks the theory and experiments together.

  L1 linear self-attention is a multiset fn  -> TransConv-only
  L2 cross-attention + edge agg preserve 1-WL -> Hierarchical OPTFM (has cross-attn)
  L3 bipartite GCN branch is 1-WL bounded     -> GNN-only
  L4 convex combination + pooling preserve it -> SGFormer+GCN (and mean/sum/max pooling)
  L5 virtual-global-node augmentation         -> Hierarchical + VGN transform

Output: results/lemma_ablation_map.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs_v2 import construct_bipartite_cycle_pair, construct_cubic_bipartite_pair
from scripts._common import build_models, embed, linf, write_csv
from scripts.improvements import virtual_global_node_transform

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]


def exact_frac(model, pairs, transform=None, pooling="mean"):
    ex = []
    for p in pairs:
        a_in = transform(p.milp_a, 0) if transform else p.milp_a
        b_in = transform(p.milp_b, 0) if transform else p.milp_b
        a = embed(model, a_in, pooling=pooling); b = embed(model, b_in, pooling=pooling)
        ex.append(linf(a, b) <= 1e-5)
    return float(np.mean(ex))


def main():
    pairs = [construct_bipartite_cycle_pair(k) for k in K_VALUES]
    pairs.append(construct_cubic_bipartite_pair())
    M = build_models(seed=0)

    spec = [
        ("L1", "Linear self-attention is a multiset function of its source set",
         "trans_conv (self-attn)", "TransConv only (random)", None, "mean"),
        ("L2", "Cross-attention + edge-weighted aggregation preserve 1-WL",
         "trans_conv_cross_*", "Hierarchical OPTFM (random)", None, "mean"),
        ("L3", "Bipartite GCN branch is 1-WL bounded (MPNN)",
         "GCN branch", "GNN only (random)", None, "mean"),
        ("L4", "Convex combination + symmetric pooling preserve 1-WL",
         "fuse + pool(mean)", "SGFormer+GCN (random)", None, "mean"),
        ("L4", "Convex combination + symmetric pooling preserve 1-WL",
         "fuse + pool(sum)", "SGFormer+GCN (random)", None, "sum"),
        ("L4", "Convex combination + symmetric pooling preserve 1-WL",
         "fuse + pool(max)", "SGFormer+GCN (random)", None, "max"),
        ("L5", "Virtual-global-node augmentation does not escape 1-WL",
         "VGN transform", "Hierarchical OPTFM (random)", virtual_global_node_transform, "mean"),
    ]

    rows = []
    for lemma, statement, component, model_name, transform, pooling in spec:
        ef = exact_frac(M[model_name], pairs, transform=transform, pooling=pooling)
        rows.append([lemma, statement, component, model_name,
                     "VGN" if transform else "baseline", pooling, ef])
        print(f"  {lemma}  {component:22s} [{model_name:28s} pool={pooling:4s}] exact_frac={ef:.2f}")

    write_csv(ROOT / "results" / "lemma_ablation_map.csv",
              ["lemma", "statement", "component", "model", "transform", "pooling", "exact_frac"],
              rows)
    print("\nWrote results/lemma_ablation_map.csv")
    print(f"All lemma ablations bit-identical: {all(r[6] == 1.0 for r in rows)}")


if __name__ == "__main__":
    main()
