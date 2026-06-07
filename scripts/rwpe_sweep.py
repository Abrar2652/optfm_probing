#!/usr/bin/env python3
"""
A3 — RWPE is not a universal escape: scope it precisely.

Three parts:
  1. Analytic return-probability check. Verify numerically the claimed gap
     P^4[i,i] = 3/8 on C_{4k} vs 1/2 on k*C_4, and tabulate the per-family gap.
  2. Walk-length L-sweep. Sweep RWPE step L over {2,3,4,6,8,12,16} and report,
     per family, the MINIMUM L at which bit-identity breaks for the hierarchical
     OPTFM. (Odd L give zero return probability on bipartite graphs; the first
     even L that distinguishes is L=4.)
  3. Failure cases. Search small regular bipartite graphs for a pair that is
     1-WL equivalent AND RWPE-equivalent (identical return-probability profile at
     every L) but non-isomorphic — i.e., a cospectral mate. Such a pair needs
     higher-order (e.g. 3-WL) structure and demonstrates RWPE's boundary,
     converting a potential overclaim into a scoped contribution.

Outputs: results/rwpe_Lsweep.csv, results/rwpe_failure_cases.csv
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs_v2 import (
    adjacency_C4k, adjacency_k_times_C4, construct_bipartite_cycle_pair,
    construct_cubic_bipartite_pair, are_isomorphic_bipartite,
)
from data.construct_pairs import build_instance
from data.milp_pairs import MILPPair
from scripts._common import build_hierarchical, embed, cos_sim, linf, write_csv
from scripts.improvements import make_rwpe_transform

K_VALUES = [2, 3, 4, 5, 6, 8, 10]
L_GRID = [2, 3, 4, 6, 8, 12, 16]


def return_prob_diag(A_full: np.ndarray, L: int) -> np.ndarray:
    deg = A_full.sum(1)
    deg = np.where(deg > 0, deg, 1.0)
    P = A_full / deg[:, None]
    Pk = np.linalg.matrix_power(P, L)
    return np.diag(Pk)


def full_adj_from_bip(A):
    nc, nv = A.shape
    N = nc + nv
    F = np.zeros((N, N))
    F[:nc, nc:] = (A != 0)
    F[nc:, :nc] = F[:nc, nc:].T
    return F


# ---------------------------------------------------------------------------
# Part 1 — analytic check
# ---------------------------------------------------------------------------

def analytic_check():
    print("=== Part 1: analytic return-probability check (P^4[i,i]) ===")
    rows = []
    for k in K_VALUES:
        pa = return_prob_diag(full_adj_from_bip(adjacency_C4k(k)), 4).mean()
        pb = return_prob_diag(full_adj_from_bip(adjacency_k_times_C4(k)), 4).mean()
        rows.append(["cycle", k, float(pa), float(pb), float(abs(pa - pb))])
        print(f"  k={k:2d}  C_4k P^4={pa:.4f}  kC4 P^4={pb:.4f}  gap={abs(pa-pb):.4f}")
    # cubic family
    cp = construct_cubic_bipartite_pair()
    pa = return_prob_diag(full_adj_from_bip(cp.milp_a.A), 4).mean()
    pb = return_prob_diag(full_adj_from_bip(cp.milp_b.A), 4).mean()
    rows.append(["cubic", 6, float(pa), float(pb), float(abs(pa - pb))])
    print(f"  cubic     A P^4={pa:.4f}  B P^4={pb:.4f}  gap={abs(pa-pb):.4f}")
    # The textbook values: 3/8 = 0.375 on C_4k, 1/2 = 0.5 on k*C_4
    assert abs(pa - pb) >= 0 and abs(return_prob_diag(full_adj_from_bip(adjacency_C4k(3)), 4).mean() - 0.375) < 1e-9
    assert abs(return_prob_diag(full_adj_from_bip(adjacency_k_times_C4(3)), 4).mean() - 0.5) < 1e-9
    print("  analytic values verified: C_4k -> 3/8, k*C_4 -> 1/2")
    return rows


# ---------------------------------------------------------------------------
# Part 2 — L-sweep
# ---------------------------------------------------------------------------

def l_sweep():
    print("\n=== Part 2: RWPE walk-length L-sweep (hierarchical OPTFM) ===")
    m = build_hierarchical(seed=0)
    rows = []
    families = [("cycle_k%d" % k, construct_bipartite_cycle_pair(k)) for k in K_VALUES]
    families.append(("cubic", construct_cubic_bipartite_pair()))
    for fam, pair in families:
        min_break = None
        for L in L_GRID:
            tf = make_rwpe_transform(steps=(L,), cons_step=L, cons_scale=1.0)
            a = embed(m, tf(pair.milp_a, 0)); b = embed(m, tf(pair.milp_b, 0))
            broke = linf(a, b) > 1e-5
            rows.append([fam, L, float(cos_sim(a, b)), float(linf(a, b)), bool(broke)])
            if broke and min_break is None:
                min_break = L
        print(f"  {fam:10s}  min L that breaks bit-identity = {min_break}")
    return rows


# ---------------------------------------------------------------------------
# Part 3 — failure-case search (cospectral => RWPE-blind)
# ---------------------------------------------------------------------------

def circulant_bip(m, offsets):
    A = np.zeros((m, m), np.float32)
    for i in range(m):
        for s in offsets:
            A[i, (i + s) % m] = 1.0
    return A


def rwpe_profile(A, steps):
    """Sorted return-probability profile over the given step set (the RWPE 'view')."""
    F = full_adj_from_bip(A)
    feats = [np.sort(return_prob_diag(F, L)) for L in steps]
    return np.round(np.concatenate(feats), 8)


def _search(steps, m_range, ds, scope_label):
    """Find non-isomorphic circulant bipartite pairs with identical RWPE profile."""
    rows = []
    for m in m_range:
        for d in ds:
            cand = {}
            for offs in itertools.combinations(range(1, m), d - 1):
                offsets = (0,) + offs
                A = circulant_bip(m, offsets)
                if not (A.sum(1) == d).all():
                    continue
                prof = tuple(rwpe_profile(A, steps).tolist())
                cand.setdefault(prof, []).append((offsets, A))
            for prof, graphs in cand.items():
                if len(graphs) < 2:
                    continue
                for (o1, A1), (o2, A2) in itertools.combinations(graphs, 2):
                    if not are_isomorphic_bipartite(
                            build_instance("a", A1, np.ones(m, np.float32)),
                            build_instance("b", A2, np.ones(m, np.float32))):
                        rows.append([m, d, str(o1), str(o2),
                                     f"1-WL+RWPE-equiv({scope_label}) NON-iso"])
                        print(f"  FOUND [{scope_label}] m={m} d={d}: {o1} vs {o2} "
                              f"RWPE-blind & non-isomorphic")
                        if len(rows) >= 6:
                            return rows
    return rows


def failure_search():
    print("\n=== Part 3: search for RWPE-blind non-isomorphic pairs ===")
    rows = []
    # (a) deployed RWPE(4,6,8): the encoding actually used in the paper
    print("  [a] deployed RWPE(4,6,8) blind spot:")
    rows += _search((4, 6, 8), range(5, 13), (3, 4), "deployed_4_6_8")
    # (b) all even L<=16: stricter (closer to full cospectrality)
    print("  [b] all even L<=16:")
    rows += _search(tuple(range(2, 17, 2)), range(5, 13), (3, 4), "allL<=16")
    if not rows:
        print("  No RWPE-blind non-isomorphic pair found in search range "
              "(m<=12, d in {3,4}); RWPE separated all candidate 1-WL-equivalent "
              "pairs here. A guaranteed blind pair requires cospectral graphs "
              "(smallest connected cospectral regular pairs have >=14 vertices).")
        rows.append(["-", "-", "-", "-",
                     "none in m<=12,d in{3,4}; needs cospectral (>=14 vtx)"])
    return rows


def main():
    arows = analytic_check()
    lrows = l_sweep()
    frows = failure_search()

    write_csv(ROOT / "results" / "rwpe_Lsweep.csv",
              ["family", "L", "cos_sim", "linf", "breaks_bit_identity"], lrows)
    # combine analytic + failure into the failure-cases file
    write_csv(ROOT / "results" / "rwpe_failure_cases.csv",
              ["m_or_family", "d_or_k", "a", "b", "note"],
              [["analytic_" + r[0], r[1], r[2], r[3], f"P4_gap={r[4]:.4f}"] for r in arows]
              + [["search"] + r for r in frows])
    print("\nWrote results/rwpe_Lsweep.csv, results/rwpe_failure_cases.csv")


if __name__ == "__main__":
    main()
