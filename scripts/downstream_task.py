#!/usr/bin/env python3
"""
A4 — Data-mining consequence: downstream-task accuracy ceiling, corpus prevalence,
and the encoding-level remedy.

The data-mining angle: GFM embeddings of MILPs are used as mined features for
downstream tasks (clustering, hardness prediction, retrieval). We show the bound's
*practical* consequence on a downstream classification task, measure how prevalent
1-WL degeneracy is across standard-family corpora, and show the ceiling is lifted by
a separating input encoding.

  (1) Downstream ceiling: a graph-level binary task — predict whether the instance's
      bipartite graph is CONNECTED (1 component) vs disconnected. Every 1-WL pair has
      one connected and one disconnected member that map to the SAME embedding, so any
      embedding-only classifier is capped at chance (~0.5). RWPE-augmented embeddings
      lift the ceiling toward 1.0 (the remedy).
  (2) Corpus prevalence: fraction of random standard-family instances with WL-twin
      nodes, and natural exact-collision counts at fixed shape.
  (3) Remedy: part (1) already contrasts baseline vs RWPE per model.

Outputs: results/downstream_task.csv, results/corpus_prevalence.csv,
         results/benchmark_collisions.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs import verify_1wl_equivalence
from data.milp_pairs_v2 import (
    construct_bipartite_cycle_pair, construct_cubic_bipartite_pair, _bipartite_components,
)
from data.construct_pairs import (
    construct_standard_family_pair, gen_random_set_cover, gen_random_comb_auction,
    wl_color_class_count,
)
from scripts._common import build_models, write_csv
from scripts.improvements import make_rwpe_transform
from scripts.probe_primal_dual import build_embedding_matrix, train_classification_probe, LinearProbe, MLPProbe

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
SEEDS = [0, 1, 2, 3, 4]


def downstream_dataset():
    """instances labeled connected(0)/disconnected(1); balanced by construction."""
    insts, y = [], []
    for k in K_VALUES:
        p = construct_bipartite_cycle_pair(k)
        insts += [p.milp_a, p.milp_b]; y += [0, 1]
    for fam in ("set_cover", "comb_auction", "matching"):
        for k in (3, 5, 8):
            p = construct_standard_family_pair(fam, k)
            insts += [p.milp_a, p.milp_b]; y += [0, 1]
    cp = construct_cubic_bipartite_pair()
    insts += [cp.milp_a, cp.milp_b]; y += [0, 1]
    return insts, np.array(y)


def downstream_ceiling():
    print("=== (1) downstream-task ceiling + remedy ===")
    insts, y = downstream_dataset()
    rwpe = make_rwpe_transform(steps=(4, 6, 8), cons_step=4, cons_scale=1.0)
    insts_rwpe = [rwpe(m, 0) for m in insts]
    models = build_models(seed=0)
    rows = []
    for mname, m in models.items():
        Xb = build_embedding_matrix(m, insts)
        Xr = build_embedding_matrix(m, insts_rwpe)
        accs_b, accs_r = [], []
        for s in SEEDS:
            accs_b.append(train_classification_probe(Xb, y, MLPProbe, epochs=400, seed=s)["acc_test"])
            accs_r.append(train_classification_probe(Xr, y, MLPProbe, epochs=400, seed=s)["acc_test"])
        rows.append([mname, float(np.mean(accs_b)), float(np.std(accs_b)),
                     float(np.mean(accs_r)), float(np.std(accs_r))])
        print(f"  {mname:32s} baseline acc={np.mean(accs_b):.3f}  RWPE acc={np.mean(accs_r):.3f}")
    write_csv(ROOT / "results" / "downstream_task.csv",
              ["model", "baseline_acc", "baseline_std", "rwpe_acc", "rwpe_std"], rows)


def corpus_prevalence():
    print("\n=== (2) corpus prevalence ===")
    rng = np.random.default_rng(0)
    rows = []
    N = 300
    for fam, gen in (("set_cover", gen_random_set_cover), ("comb_auction", gen_random_comb_auction)):
        insts = [gen(10, 12, 0.3, rng) for _ in range(N)]
        ratios, twins = [], []
        for inst in insts:
            nn = inst.A.shape[0] + inst.A.shape[1]
            cc = wl_color_class_count(inst)
            ratios.append(cc / nn); twins.append(cc < nn)
        rows.append([fam, N, float(np.mean(twins)), float(np.mean(ratios))])
        print(f"  {fam:13s}: {np.mean(twins)*100:.1f}% have WL-twins; "
              f"mean #classes/#nodes={np.mean(ratios):.3f}")
    write_csv(ROOT / "results" / "corpus_prevalence.csv",
              ["family", "n_instances", "frac_with_wl_twins", "mean_class_node_ratio"], rows)


def benchmark_collisions():
    print("\n=== (3) natural exact-collision counts (same shape) ===")
    rng = np.random.default_rng(1)
    rows = []
    N = 120
    for fam, gen, shape in (("set_cover", gen_random_set_cover, (8, 8)),
                            ("comb_auction", gen_random_comb_auction, (8, 8))):
        insts = [gen(shape[0], shape[1], 0.35, rng) for _ in range(N)]
        coll = 0; pairs = 0
        for i in range(N):
            for j in range(i + 1, N):
                pairs += 1
                if verify_1wl_equivalence(insts[i], insts[j], iterations=3):
                    coll += 1
        rows.append([fam, N, shape[0], shape[1], coll, pairs, coll / max(pairs, 1)])
        print(f"  {fam:13s}: {coll}/{pairs} random same-shape pairs are 1-WL-equivalent "
              f"({100*coll/max(pairs,1):.2f}%)")
    write_csv(ROOT / "results" / "benchmark_collisions.csv",
              ["family", "n_instances", "m", "n", "n_collisions", "n_pairs", "collision_rate"], rows)


def main():
    downstream_ceiling()
    corpus_prevalence()
    benchmark_collisions()
    print("\nWrote results/downstream_task.csv, results/corpus_prevalence.csv, results/benchmark_collisions.csv")


if __name__ == "__main__":
    main()
