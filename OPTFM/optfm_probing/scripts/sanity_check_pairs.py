#!/usr/bin/env python3
"""
Sanity-check the existing milp_pairs.py 'canonical' pair.

We suspect the pair currently used in data/milp_pairs.py is actually
GRAPH-ISOMORPHIC (not just 1-WL equivalent) as feature-labeled bipartite
graphs, which would make the existing probe uninformative: any
permutation-equivariant model will trivially map isomorphic inputs to the
same embedding, regardless of 1-WL expressivity.

We verify three things:
  1. Both "MILP-A" and "MILP-B" are feasible (testing the docstring claim
     that B is infeasible)
  2. 1-WL color refinement treats them as equivalent (sanity check)
  3. There exists a constraint-variable permutation that maps (A, b_A, features)
     to (A, b_B, features) — i.e., they are isomorphic as feature-labeled
     bipartite graphs (sanity check the suspicion).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import itertools
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

from scripts._tee_log import start_logging
from data.milp_pairs import (
    construct_canonical_pair, verify_1wl_equivalence
)


def solve_feasibility(A, b, sense, lb, ub, integer=True):
    """Solve the MILP feasibility problem. Returns True/False and a point."""
    n = A.shape[1]
    # Equality constraints (sense == 0) with A x = b
    eq_rows = np.where(sense == 0)[0]
    leq_rows = np.where(sense == 1)[0]
    geq_rows = np.where(sense == 2)[0]

    constraints = []
    if len(eq_rows) > 0:
        constraints.append(LinearConstraint(A[eq_rows], lb=b[eq_rows], ub=b[eq_rows]))
    if len(leq_rows) > 0:
        constraints.append(LinearConstraint(A[leq_rows], lb=-np.inf, ub=b[leq_rows]))
    if len(geq_rows) > 0:
        constraints.append(LinearConstraint(A[geq_rows], lb=b[geq_rows], ub=np.inf))

    integrality = np.ones(n) if integer else np.zeros(n)
    bounds = Bounds(lb=lb, ub=ub)
    res = milp(
        c=np.zeros(n),
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
    )
    return bool(res.success), (res.x if res.success else None)


def check_isomorphism_by_brute_force(milp_a, milp_b):
    """
    Brute-force search for a permutation (pi_cons, sigma_vars) that maps
    (A_a, b_a, cons_features_a, var_features_a) to
    (A_b, b_b, cons_features_b, var_features_b).

    Returns None if no such permutation exists, else (pi, sigma).
    Only tractable for small instances (n_cons, n_vars <= 7).
    """
    n_cons, n_vars = milp_a.A.shape
    assert (n_cons, n_vars) == milp_b.A.shape, "different shapes"
    if n_cons > 7 or n_vars > 7:
        print("  skipping brute force (too large)")
        return None

    for pi in itertools.permutations(range(n_cons)):
        for sigma in itertools.permutations(range(n_vars)):
            pi_arr = np.array(pi)
            sigma_arr = np.array(sigma)
            A_mapped = milp_a.A[pi_arr][:, sigma_arr]
            b_mapped = milp_a.b[pi_arr]
            cons_mapped = milp_a.cons_features[pi_arr]
            var_mapped = milp_a.var_features[sigma_arr]
            if (np.allclose(A_mapped, milp_b.A) and
                np.allclose(b_mapped, milp_b.b) and
                np.allclose(cons_mapped, milp_b.cons_features) and
                np.allclose(var_mapped, milp_b.var_features)):
                return pi, sigma
    return None


def main():
    start_logging("sanity_check_pairs")
    print("=" * 78)
    print("SANITY CHECK on existing 'canonical' 1-WL pair in data/milp_pairs.py")
    print("=" * 78)

    pair = construct_canonical_pair()
    ma, mb = pair.milp_a, pair.milp_b

    print(f"\npair.description           : {pair.description}")
    print(f"pair.expected_distinguishable: {pair.expected_distinguishable}")
    print(f"pair.is_1wl_equivalent      : {pair.is_1wl_equivalent}")
    print(f"MILP-A claimed feasible    : {ma.is_feasible}")
    print(f"MILP-B claimed infeasible  : {not mb.is_feasible}")
    print(f"A matrix:\n{ma.A}")
    print(f"b_A = {ma.b}")
    print(f"b_B = {mb.b}")

    print("\n--- 1. Feasibility (ground truth via scipy.milp) ---")
    fa, xa = solve_feasibility(ma.A, ma.b, ma.sense, ma.lb, ma.ub, integer=True)
    fb, xb = solve_feasibility(mb.A, mb.b, mb.sense, mb.lb, mb.ub, integer=True)
    print(f"MILP-A feasible? {fa}  solution: {xa}")
    print(f"MILP-B feasible? {fb}  solution: {xb}")
    if fa and fb:
        print("!! BOTH are feasible. The 'canonical' pair does NOT have the "
              "feasibility-differs property claimed in its docstring.")

    print("\n--- 2. 1-WL color refinement ---")
    is_1wl_eq = verify_1wl_equivalence(ma, mb)
    print(f"1-WL equivalent? {is_1wl_eq}")

    print("\n--- 3. Graph isomorphism (feature-labeled bipartite) ---")
    iso = check_isomorphism_by_brute_force(ma, mb)
    if iso is not None:
        pi, sigma = iso
        print(f"ISOMORPHIC. constraint permutation pi = {pi}, variable permutation sigma = {sigma}")
        print("!! The pair is graph-isomorphic. Any permutation-equivariant model "
              "will trivially map them to the same embedding, REGARDLESS of 1-WL "
              "expressivity. The 'canonical pair' as currently constructed is "
              "uninformative for a 1-WL expressiveness probe.")
    else:
        print("NOT isomorphic. Pair is 1-WL equivalent but non-isomorphic (good "
              "for a real 1-WL probe).")

    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    if fa and fb and iso is not None:
        print("The existing probe is UNINFORMATIVE:")
        print("  * Both MILPs are feasible (not A-feasible/B-infeasible as claimed)")
        print("  * They are GRAPH ISOMORPHIC (a strictly stronger property than")
        print("    1-WL equivalence). Equal embeddings are the CORRECT behavior")
        print("    for any GNN, not a 1-WL failure.")
        print("  * To test 1-WL boundedness we need 1-WL equivalent but NON-")
        print("    isomorphic MILP pairs (e.g. Cai-Fürer-Immerman construction).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
