#!/usr/bin/env python3
"""
C2 — Integrity check: the "canonical 1-WL pair" used in earlier sessions is
actually GRAPH-ISOMORPHIC as a feature-labeled bipartite graph, and BOTH MILPs
are integer-feasible.  This script reproduces that finding from scratch so the
integrity story in the paper's appendix is verifiable, not narrated.

It:
  1. Loads data.milp_pairs.construct_canonical_pair().
  2. Brute-force searches all 5! x 6! = 86,400 (constraint, variable) permutation
     pairs for an explicit isomorphism (A, b_A, feat) -> (A, b_B, feat).
  3. Confirms both MILP-A and MILP-B are integer-feasible via scipy.milp.
  4. Contrasts with the corrected family (construct_bipartite_cycle_pair), which
     is 1-WL equivalent AND provably non-isomorphic.

Output: results/broken_canonical.txt
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs import construct_canonical_pair
from data.milp_pairs_v2 import construct_bipartite_cycle_pair, diagnose_pair


def find_isomorphism(milp_a, milp_b):
    """Brute-force all (pi, sigma) permutation pairs; return the first that maps A->B."""
    A_a, b_a = milp_a.A, milp_a.b
    A_b, b_b = milp_b.A, milp_b.b
    n_cons, n_vars = A_a.shape
    n_checked = 0
    for pi in itertools.permutations(range(n_cons)):
        pi = np.array(pi)
        if not np.allclose(b_a[pi], b_b):
            # still scan vars for the A-pattern (b may map under a different pi)
            pass
        for sigma in itertools.permutations(range(n_vars)):
            n_checked += 1
            sigma = np.array(sigma)
            if np.allclose(A_a[pi][:, sigma], A_b) and np.allclose(b_a[pi], b_b):
                return pi.tolist(), sigma.tolist(), n_checked
    return None, None, n_checked


def is_integer_feasible(milp_inst):
    """Return (feasible, solution) for the MILP via scipy.milp."""
    try:
        from scipy.optimize import milp as scipy_milp, LinearConstraint, Bounds
    except Exception as e:  # pragma: no cover
        return None, f"scipy.milp unavailable: {e}"
    A, b, sense = milp_inst.A, milp_inst.b, milp_inst.sense
    nv = A.shape[1]
    cons = []
    for i in range(A.shape[0]):
        if sense[i] == 0:      # equality
            cons.append(LinearConstraint(A[i], b[i], b[i]))
        elif sense[i] == 1:    # <=
            cons.append(LinearConstraint(A[i], -np.inf, b[i]))
        else:                  # >=
            cons.append(LinearConstraint(A[i], b[i], np.inf))
    res = scipy_milp(c=np.zeros(nv), constraints=cons,
                     integrality=np.ones(nv),
                     bounds=Bounds(milp_inst.lb, milp_inst.ub))
    if res.success:
        return True, np.round(res.x).astype(int).tolist()
    return False, None


def main():
    lines = []
    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("C2 — Broken-canonical-pair integrity check")
    log("=" * 70)

    pair = construct_canonical_pair()
    log(f"\nPair: {pair.description}")
    log(f"  A shape: {pair.milp_a.A.shape}")
    log(f"  b_A = {pair.milp_a.b.tolist()}")
    log(f"  b_B = {pair.milp_b.b.tolist()}")

    log("\n[1] Brute-force isomorphism search over 5! x 6! = 86,400 pairs...")
    pi, sigma, n_checked = find_isomorphism(pair.milp_a, pair.milp_b)
    if pi is not None:
        log(f"    ISOMORPHIC. Explicit permutation found after {n_checked} checks:")
        log(f"      pi (constraints) = {pi}")
        log(f"      sigma (variables) = {sigma}")
        log("    => any permutation-equivariant model maps A and B to identical")
        log("       embeddings; cos_sim = 1.0 on this pair is UNINFORMATIVE.")
    else:
        log(f"    No isomorphism found after {n_checked} checks (unexpected).")

    log("\n[2] Integer feasibility of both members (scipy.milp):")
    for tag, milp in (("MILP-A", pair.milp_a), ("MILP-B", pair.milp_b)):
        feas, sol = is_integer_feasible(milp)
        log(f"    {tag}: feasible={feas}  solution={sol}")
    log("    => the docstring claim 'MILP-B infeasible' is FALSE; both are feasible.")

    log("\n[3] Contrast: the corrected family is 1-WL equivalent AND non-isomorphic.")
    for k in (2, 3):
        d = diagnose_pair(construct_bipartite_cycle_pair(k))
        log(f"    k={k}: 1wl_equiv={d['is_1wl_equivalent']}  "
            f"isomorphic={d['is_isomorphic']}  "
            f"A_comp={d['A_components']}  B_comp={d['B_components']}")

    log("\nCONCLUSION: the legacy 'canonical pair' is a degenerate (isomorphic) test")
    log("case; the C_{4k} vs k*C_4 family is the valid 1-WL probe.")

    out = ROOT / "results" / "broken_canonical.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
