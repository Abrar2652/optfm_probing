"""
Proper 1-WL equivalent but non-isomorphic bipartite MILP pairs.

The previous construction in milp_pairs.py produced pairs that are graph-
ISOMORPHIC as feature-labeled bipartite graphs (verified in
scripts/sanity_check_pairs.py). Isomorphic pairs are trivially mapped to
the same embedding by any permutation-equivariant model — their equal
embeddings tell us nothing about 1-WL expressiveness.

This module constructs genuinely 1-WL equivalent but non-isomorphic pairs
using the cycle / disjoint-4-cycle construction:

  G_A(k) = C_{4k} as a bipartite graph
           (2k constraints, 2k variables, connected 4k-cycle)
  G_B(k) = k disjoint copies of C_4 as a bipartite graph
           (2k constraints, 2k variables, k disconnected components)

Both graphs are 2-regular on both sides, so when all constraint features
and all variable features are uniform, 1-WL color refinement never
introduces any distinction: at every iteration every node's neighborhood
is a multiset of two identically-colored neighbors, so all nodes stay in
the same color class. Therefore the two graphs are 1-WL equivalent at
every refinement depth.

They are non-isomorphic because G_A(k) is connected (one component) while
G_B(k) has k components. Connectivity is a graph isomorphism invariant
that 1-WL cannot detect.

When k = 2 the pair is the smallest non-trivial 1-WL equivalent non-
isomorphic bipartite MILP pair.

We also expose:
  * construct_bipartite_cycle_pair(k)       — one pair at scale k
  * generate_cycle_pair_family(k_values)    — a family of pairs
  * feature-labeled bipartite isomorphism check via networkx (brute force
    for small n, certificate via graph hashing for larger n)
  * LP relaxation value / solution / duals for each instance, used later
    by the primal-dual probe battery

References
----------
Chen, Liu, Gao, Wang 2023. "On Representing Mixed-Integer Linear Programs
by Graph Neural Networks." ICLR 2023.
    — general framework for reasoning about MILP GNN expressivity via WL.
Cai, Fürer, Immerman 1992. "An optimal lower bound on the number of
variables for graph identification." Combinatorica.
    — canonical source for 1-WL-equivalent non-isomorphic graphs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from data.milp_pairs import (
    MILPInstance, MILPPair, create_ecole_features, milp_to_tensors,
    verify_1wl_equivalence,
)


# ---------------------------------------------------------------------------
# Bipartite adjacency constructions
# ---------------------------------------------------------------------------

def adjacency_C4k(k: int) -> np.ndarray:
    """
    Adjacency of the bipartite graph whose underlying graph is C_{4k}.

    Layout: alternating cons-var around the cycle.
        cons nodes: c_0, c_1, ..., c_{2k-1}
        var  nodes: v_0, v_1, ..., v_{2k-1}

    Cycle edges (cycling through 4k vertices):
        c_0 - v_0 - c_1 - v_1 - ... - c_{2k-1} - v_{2k-1} - c_0
    In particular constraint c_i is adjacent to variables v_{i-1 mod 2k}
    and v_i, making each row have exactly two 1s and each column have
    exactly two 1s.
    """
    assert k >= 1
    n = 2 * k
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        A[i, i] = 1.0                # c_i -- v_i
        A[i, (i - 1) % n] = 1.0      # c_i -- v_{i-1}
    return A


def adjacency_k_times_C4(k: int) -> np.ndarray:
    """
    Adjacency of the bipartite graph whose underlying graph is k disjoint
    copies of C_4.

    Layout: for each of the k components, there are 2 cons and 2 vars,
    with all 4 cons-var edges (a K_{2,2}).

    Row/column sums are exactly 2, matching adjacency_C4k(k).
    """
    assert k >= 1
    n = 2 * k
    A = np.zeros((n, n), dtype=np.float32)
    for j in range(k):
        r = 2 * j
        c = 2 * j
        A[r:r+2, c:c+2] = 1.0
    return A


# ---------------------------------------------------------------------------
# Cubic (3-regular) bipartite pair family — structurally independent of
# the 2-regular C_{4k} vs k*C_4 family. Both families test the same
# 1-WL bound, but from different regularity classes.
# ---------------------------------------------------------------------------

def adjacency_cubic_connected(m: int) -> np.ndarray:
    """
    Connected 3-regular bipartite graph on m + m vertices, m >= 3.

    Construction: constraint i is adjacent to variables (i, i+1, i+3) mod m.
    Each constraint has degree 3 (three distinct offsets). Each variable
    j is adjacent to constraints i for which j - i in {0, 1, 3} mod m,
    i.e. i in {j, j-1, j-3} mod m — also three distinct values, so each
    variable also has degree 3. The offsets {0, 1, 3} generate Z/mZ
    whenever m is not a proper multiple of any divisor dividing the
    differences, which holds for m >= 4; for m = 6 the resulting graph
    is connected (verified by _bipartite_components).
    """
    assert m >= 4, "cubic construction needs m >= 4"
    A = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        A[i, i % m] = 1.0
        A[i, (i + 1) % m] = 1.0
        A[i, (i + 3) % m] = 1.0
    return A


def adjacency_two_K33() -> np.ndarray:
    """
    Disconnected 3-regular bipartite graph on 6 + 6 vertices.

    Two disjoint copies of K_{3,3}. Each K_{3,3} is a 3-regular bipartite
    graph (complete bipartite between its 3 constraints and 3 variables),
    so the union is a 3-regular bipartite graph on 12 vertices with
    exactly two connected components.
    """
    A = np.zeros((6, 6), dtype=np.float32)
    A[0:3, 0:3] = 1.0
    A[3:6, 3:6] = 1.0
    return A


def construct_cubic_bipartite_pair() -> MILPPair:
    """
    A structurally independent second 1-WL equivalent non-isomorphic pair.

      G_A = connected 3-regular bipartite graph on 6 cons + 6 vars
            (via adjacency_cubic_connected(6)).
      G_B = K_{3,3} disjoint union K_{3,3} on 6 cons + 6 vars
            (via adjacency_two_K33()).

    Both graphs are 3-regular on both sides, so with uniform cons features
    and uniform var features, 1-WL color refinement collapses to a single
    color class after iteration 1 (every node has three neighbors of the
    same color), and the histograms are identical at every depth.

    They are NON-isomorphic because G_A is connected (1 component) while
    G_B has exactly 2 components.

    Uniform b = 1 keeps cons features uniform at 1 / sqrt(3) per row.
    """
    A_a = adjacency_cubic_connected(6)
    A_b = adjacency_two_K33()
    assert A_a.shape == A_b.shape == (6, 6)
    # Verify row / col sums are all 3
    assert np.all(A_a.sum(axis=1) == 3) and np.all(A_a.sum(axis=0) == 3)
    assert np.all(A_b.sum(axis=1) == 3) and np.all(A_b.sum(axis=0) == 3)

    b = np.ones(6, dtype=np.float32)
    milp_a = _make_instance("cubic_connected_6", A_a, b)
    milp_b = _make_instance("two_K33", A_b, b)

    return MILPPair(
        milp_a=milp_a, milp_b=milp_b,
        description="Cubic bipartite 1-WL pair: connected vs 2*K_{3,3}",
        expected_distinguishable=False,
        is_1wl_equivalent=True,
    )


# ---------------------------------------------------------------------------
# MILP pair builder with uniform features
# ---------------------------------------------------------------------------

def _make_instance(name: str, A: np.ndarray, b: np.ndarray) -> MILPInstance:
    n_cons, n_vars = A.shape
    c = np.zeros(n_vars, dtype=np.float32)
    sense = np.zeros(n_cons, dtype=np.float32)     # all equality (=)
    lb = np.zeros(n_vars, dtype=np.float32)
    ub = np.ones(n_vars, dtype=np.float32)
    vtype = np.ones(n_vars, dtype=np.float32)      # binary
    var_feat, cons_feat, edge_idx, edge_attr = create_ecole_features(
        c, A, b, sense, lb, ub, vtype
    )
    return MILPInstance(
        name=name, var_features=var_feat, cons_features=cons_feat,
        edge_index=edge_idx, edge_attr=edge_attr,
        c=c, A=A, b=b, sense=sense, lb=lb, ub=ub, vtype=vtype,
    )


def construct_bipartite_cycle_pair(k: int, b_value: float = 1.0) -> MILPPair:
    """
    Build the C_{4k}-bipartite vs k*C_4-bipartite MILP pair at scale k.

    All constraint RHS entries are set to `b_value`, which keeps cons
    features uniform across all rows in both graphs (because each row of A
    has norm sqrt(2)). This uniformity is what forces 1-WL refinement to
    stay at a single color class throughout all iterations.
    """
    assert k >= 2, "k=1 degenerates: C_4 == 1 * C_4 (trivially isomorphic)"

    A_a = adjacency_C4k(k)
    A_b = adjacency_k_times_C4(k)
    n_cons = 2 * k
    b = np.full(n_cons, b_value, dtype=np.float32)

    milp_a = _make_instance(f"C_{4*k}_bipartite_k{k}", A_a, b)
    milp_b = _make_instance(f"{k}xC_4_bipartite_k{k}", A_b, b)

    return MILPPair(
        milp_a=milp_a, milp_b=milp_b,
        description=f"1-WL equivalent non-iso pair: C_{4*k} vs {k}*C_4 (k={k})",
        expected_distinguishable=False,
        is_1wl_equivalent=True,
    )


def generate_cycle_pair_family(k_values: List[int]) -> List[MILPPair]:
    return [construct_bipartite_cycle_pair(k) for k in k_values]


# ---------------------------------------------------------------------------
# Isomorphism check — brute force for tiny graphs, NetworkX for larger
# ---------------------------------------------------------------------------

def are_isomorphic_bipartite(milp_a: MILPInstance, milp_b: MILPInstance) -> bool:
    """
    Check if two MILPInstances are feature-labeled-bipartite-graph-isomorphic.

    Strategy:
      - Shapes must match.
      - For n <= 6 each side, use brute force permutation.
      - For larger, use NetworkX VF2 with node attributes.
    """
    if milp_a.A.shape != milp_b.A.shape:
        return False
    n_cons, n_vars = milp_a.A.shape
    if n_cons <= 6 and n_vars <= 6:
        return _iso_brute_force(milp_a, milp_b)
    return _iso_networkx(milp_a, milp_b)


def _iso_brute_force(milp_a: MILPInstance, milp_b: MILPInstance) -> bool:
    n_cons, n_vars = milp_a.A.shape
    for pi in itertools.permutations(range(n_cons)):
        pi_arr = np.array(pi)
        if not np.allclose(milp_a.cons_features[pi_arr], milp_b.cons_features):
            continue
        for sigma in itertools.permutations(range(n_vars)):
            sigma_arr = np.array(sigma)
            if not np.allclose(milp_a.var_features[sigma_arr], milp_b.var_features):
                continue
            A_mapped = milp_a.A[pi_arr][:, sigma_arr]
            if np.allclose(A_mapped, milp_b.A):
                return True
    return False


def _iso_networkx(milp_a: MILPInstance, milp_b: MILPInstance) -> bool:
    try:
        import networkx as nx
        from networkx.algorithms import isomorphism
    except ImportError:
        raise RuntimeError("NetworkX required for larger isomorphism checks")

    def to_graph(milp: MILPInstance):
        g = nx.Graph()
        n_cons, n_vars = milp.A.shape
        for i in range(n_cons):
            g.add_node(("c", i), side="c", feat=tuple(np.round(milp.cons_features[i], 4)))
        for j in range(n_vars):
            g.add_node(("v", j), side="v", feat=tuple(np.round(milp.var_features[j], 4)))
        rows, cols = np.nonzero(milp.A)
        for r, c in zip(rows, cols):
            g.add_edge(("c", int(r)), ("v", int(c)), w=float(milp.A[r, c]))
        return g

    g_a = to_graph(milp_a)
    g_b = to_graph(milp_b)
    gm = isomorphism.GraphMatcher(
        g_a, g_b,
        node_match=lambda u, v: u.get("side") == v.get("side") and u.get("feat") == v.get("feat"),
        edge_match=lambda e, f: np.isclose(e.get("w", 1.0), f.get("w", 1.0)),
    )
    return gm.is_isomorphic()


# ---------------------------------------------------------------------------
# Graph diagnostics — connectivity, degree sequences
# ---------------------------------------------------------------------------

def _bipartite_components(A: np.ndarray) -> int:
    """Count connected components of the bipartite graph with adjacency A."""
    n_cons, n_vars = A.shape
    total = n_cons + n_vars
    parent = list(range(total))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    rows, cols = np.nonzero(A)
    for r, c in zip(rows, cols):
        union(int(r), int(n_cons + c))
    roots = {find(i) for i in range(total)}
    return len(roots)


def diagnose_pair(pair: MILPPair) -> dict:
    """Return a dictionary of diagnostic properties for a pair."""
    return {
        "description": pair.description,
        "shape": tuple(pair.milp_a.A.shape),
        "A_nnz": int(np.count_nonzero(pair.milp_a.A)),
        "B_nnz": int(np.count_nonzero(pair.milp_b.A)),
        "A_components": _bipartite_components(pair.milp_a.A),
        "B_components": _bipartite_components(pair.milp_b.A),
        "row_sums_equal": np.array_equal(
            np.sort(pair.milp_a.A.sum(axis=1)),
            np.sort(pair.milp_b.A.sum(axis=1))),
        "col_sums_equal": np.array_equal(
            np.sort(pair.milp_a.A.sum(axis=0)),
            np.sort(pair.milp_b.A.sum(axis=0))),
        "is_1wl_equivalent": verify_1wl_equivalence(pair.milp_a, pair.milp_b),
        "is_isomorphic": are_isomorphic_bipartite(pair.milp_a, pair.milp_b),
    }


# ---------------------------------------------------------------------------
# LP relaxation — for primal-dual probe battery
# ---------------------------------------------------------------------------

def solve_lp_relaxation(milp: MILPInstance, objective: Optional[np.ndarray] = None):
    """
    Solve the LP relaxation of the MILP with a given objective.

    If objective is None, uses a deterministic non-symmetric objective
    c_j = j + 1 to ensure the optimal vertex is non-degenerate.

    Returns a dict with keys:
        x_star (n_vars,)
        obj_value  (float)
        duals (n_cons,)
        reduced_costs (n_vars,)
        feasible (bool)
    """
    from scipy.optimize import linprog

    n_vars = milp.A.shape[1]
    if objective is None:
        objective = np.arange(1, n_vars + 1, dtype=float)

    A_eq_rows = np.where(milp.sense == 0)[0]
    A_leq_rows = np.where(milp.sense == 1)[0]
    A_geq_rows = np.where(milp.sense == 2)[0]

    A_eq = milp.A[A_eq_rows] if len(A_eq_rows) else None
    b_eq = milp.b[A_eq_rows] if len(A_eq_rows) else None

    A_ub_parts = []
    b_ub_parts = []
    if len(A_leq_rows):
        A_ub_parts.append(milp.A[A_leq_rows])
        b_ub_parts.append(milp.b[A_leq_rows])
    if len(A_geq_rows):
        A_ub_parts.append(-milp.A[A_geq_rows])
        b_ub_parts.append(-milp.b[A_geq_rows])

    A_ub = np.vstack(A_ub_parts) if A_ub_parts else None
    b_ub = np.concatenate(b_ub_parts) if b_ub_parts else None

    bounds = list(zip(milp.lb, milp.ub))
    res = linprog(objective, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    out = {
        "feasible": bool(res.success),
        "obj_value": float(res.fun) if res.success else float("nan"),
        "x_star": res.x.astype(np.float32) if res.success else np.zeros(n_vars, dtype=np.float32),
        "duals": np.zeros(milp.A.shape[0], dtype=np.float32),
        "reduced_costs": np.zeros(n_vars, dtype=np.float32),
    }
    if res.success and hasattr(res, "ineqlin") and res.ineqlin is not None:
        # Translate HiGHS duals / reduced costs back
        if hasattr(res, "eqlin") and res.eqlin is not None and len(A_eq_rows) > 0:
            out["duals"][A_eq_rows] = -res.eqlin.marginals.astype(np.float32)
        if hasattr(res, "ineqlin") and res.ineqlin is not None:
            ineq_marginals = -res.ineqlin.marginals.astype(np.float32)
            # ineq rows are [leq rows..., -geq rows...]
            m_leq = len(A_leq_rows)
            m_geq = len(A_geq_rows)
            if m_leq:
                out["duals"][A_leq_rows] = ineq_marginals[:m_leq]
            if m_geq:
                out["duals"][A_geq_rows] = -ineq_marginals[m_leq:m_leq + m_geq]
        if hasattr(res, "lower") and res.lower is not None:
            out["reduced_costs"] = res.lower.marginals.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing bipartite-cycle pair family...")
    for k in [2, 3, 4, 5]:
        pair = construct_bipartite_cycle_pair(k)
        diag = diagnose_pair(pair)
        print(f"\nk={k}  {diag['description']}")
        for key, val in diag.items():
            if key != "description":
                print(f"   {key:20s}: {val}")
        # Sanity: 1-WL equivalent AND not isomorphic
        assert diag["is_1wl_equivalent"], f"k={k}: not 1-WL equivalent"
        assert not diag["is_isomorphic"], f"k={k}: is isomorphic (bad)"
        assert diag["A_components"] == 1, f"k={k}: C_{4*k} should be connected"
        assert diag["B_components"] == k, f"k={k}: {k}*C_4 should have {k} components"
    print("\nAll pairs: 1-WL equivalent, non-isomorphic, correct component counts.")
