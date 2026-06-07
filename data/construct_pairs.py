"""
Constructors for ICDM experiments beyond the original cycle / cubic pairs:

  * Non-uniform-feature 1-WL-equivalent non-isomorphic pairs (A7), to refute the
    objection that the collision only arises because all node features are zero.
  * A general MILP builder and standard MILP-family generators (A2): set cover,
    knapsack, combinatorial auctions.
  * A pair-construction algorithm that, given a base MILP, produces a
    1-WL-equivalent non-isomorphic partner by component refolding (A2).
  * Distinguishable (negative-control) pairs (A7 specificity).

Every constructor returns the existing MILPPair / MILPInstance dataclasses, and
the pairs are designed to pass data.milp_pairs_v2.diagnose_pair verification
(1-WL equivalent AND non-isomorphic). Callers SHOULD verify before probing.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from data.milp_pairs import (
    MILPInstance, MILPPair, create_ecole_features, verify_1wl_equivalence,
)
from data.milp_pairs_v2 import (
    adjacency_C4k, adjacency_k_times_C4, _bipartite_components,
    are_isomorphic_bipartite,
)


# ---------------------------------------------------------------------------
# General MILP builder
# ---------------------------------------------------------------------------

def build_instance(name: str, A: np.ndarray, b: np.ndarray,
                   c: Optional[np.ndarray] = None,
                   sense: Optional[np.ndarray] = None,
                   lb: Optional[np.ndarray] = None,
                   ub: Optional[np.ndarray] = None,
                   vtype: Optional[np.ndarray] = None) -> MILPInstance:
    n_cons, n_vars = A.shape
    c = np.zeros(n_vars, np.float32) if c is None else c.astype(np.float32)
    sense = np.zeros(n_cons, np.float32) if sense is None else sense.astype(np.float32)
    lb = np.zeros(n_vars, np.float32) if lb is None else lb.astype(np.float32)
    ub = np.ones(n_vars, np.float32) if ub is None else ub.astype(np.float32)
    vtype = np.ones(n_vars, np.float32) if vtype is None else vtype.astype(np.float32)
    vf, cf, ei, ea = create_ecole_features(c, A.astype(np.float32), b.astype(np.float32),
                                           sense, lb, ub, vtype)
    return MILPInstance(name=name, var_features=vf, cons_features=cf,
                        edge_index=ei, edge_attr=ea, c=c, A=A.astype(np.float32),
                        b=b.astype(np.float32), sense=sense, lb=lb, ub=ub, vtype=vtype)


def _diag(pair: MILPPair) -> dict:
    a, b = pair.milp_a, pair.milp_b
    return {
        "is_1wl_equivalent": verify_1wl_equivalence(a, b),
        "A_components": _bipartite_components(a.A),
        "B_components": _bipartite_components(b.A),
        "non_isomorphic_by_components": _bipartite_components(a.A) != _bipartite_components(b.A),
    }


# ---------------------------------------------------------------------------
# A7 (i) — non-uniform but type-uniform features
# ---------------------------------------------------------------------------

def construct_heterogeneous_cycle_pair(k: int) -> MILPPair:
    """C_{4k} vs k*C_4 but with NON-trivial (non-zero) node features.

    Variables are integer with finite bounds and a non-zero objective coefficient,
    so the Ecole feature vector populates slots 0 (obj), 1 (type), 4/5 (bounds) —
    not the all-zero var features of the original family. Features are identical
    per node-type across both graphs, so 1-WL equivalence is preserved while the
    "the collision only works because features are zero" objection is refuted.
    """
    A_a, A_b = adjacency_C4k(k), adjacency_k_times_C4(k)
    n = 2 * k
    b = np.full(n, 1.0, np.float32)
    c = np.full(n, 1.5, np.float32)          # non-zero objective
    vtype = np.full(n, 2.0, np.float32)      # integer
    lb = np.full(n, 0.0, np.float32)
    ub = np.full(n, 5.0, np.float32)         # finite, non-binary bound
    ma = build_instance(f"het_C{4*k}_k{k}", A_a, b, c, None, lb, ub, vtype)
    mb = build_instance(f"het_{k}C4_k{k}", A_b, b, c, None, lb, ub, vtype)
    return MILPPair(ma, mb, f"Heterogeneous-feature 1-WL pair (k={k})",
                    expected_distinguishable=False, is_1wl_equivalent=True)


# ---------------------------------------------------------------------------
# A7 (ii) — node-varied features via a feature-rich gadget union
# ---------------------------------------------------------------------------

def _gadget():
    """A small fixed MILP with genuinely node-varying features (a 3x3 path)."""
    A = np.array([[1, 1, 0],
                  [0, 1, 1],
                  [1, 0, 1]], np.float32)
    b = np.array([1.0, 2.0, 3.0], np.float32)          # distinct RHS -> distinct cons feats
    c = np.array([0.5, 1.0, 2.0], np.float32)          # distinct obj -> distinct var feats
    vtype = np.array([0.0, 1.0, 2.0], np.float32)      # continuous / binary / integer
    return A, b, c, vtype


def construct_gadget_union_pair(k: int) -> MILPPair:
    """(C_{4k} ⊔ gadget) vs (k*C_4 ⊔ gadget): node-varied features, still 1-WL equiv.

    The gadget contributes nodes whose features differ from the (uniform) core
    nodes, so the combined instance has genuinely heterogeneous node features.
    The WL colour histogram of the union is the (multiset) union of the core and
    gadget histograms; since the two cores are 1-WL equivalent and the gadget is
    identical, the unions are 1-WL equivalent. Non-isomorphism is preserved
    because the cores differ in component count.
    """
    core_a, core_b = adjacency_C4k(k), adjacency_k_times_C4(k)
    n = 2 * k
    gA, gb, gc, gvt = _gadget()
    ng_c, ng_v = gA.shape

    def union(core, name):
        nc, nv = core.shape
        A = np.zeros((nc + ng_c, nv + ng_v), np.float32)
        A[:nc, :nv] = core
        A[nc:, nv:] = gA
        b = np.concatenate([np.full(nc, 1.0, np.float32), gb])
        c = np.concatenate([np.zeros(nv, np.float32), gc])
        vtype = np.concatenate([np.ones(nv, np.float32), gvt])
        ub = np.concatenate([np.ones(nv, np.float32), np.full(ng_v, 3.0, np.float32)])
        return build_instance(name, A, b, c, None, None, ub, vtype)

    ma = union(core_a, f"gadget_C{4*k}_k{k}")
    mb = union(core_b, f"gadget_{k}C4_k{k}")
    return MILPPair(ma, mb, f"Gadget-union node-varied 1-WL pair (k={k})",
                    expected_distinguishable=False, is_1wl_equivalent=True)


# ---------------------------------------------------------------------------
# A2 — pair-construction algorithm + standard MILP families
# ---------------------------------------------------------------------------

def _circulant_bip(m: int, offsets) -> np.ndarray:
    A = np.zeros((m, m), np.float32)
    for i in range(m):
        for s in offsets:
            A[i, (i + s) % m] = 1.0
    return A


def _disjoint_Kdd(k: int, d: int) -> np.ndarray:
    """k disjoint copies of K_{d,d} (d-regular, k components)."""
    A = np.zeros((k * d, k * d), np.float32)
    for j in range(k):
        A[j*d:(j+1)*d, j*d:(j+1)*d] = 1.0
    return A


def construct_folded_pair(k: int, d: int = 3, sense_val: float = 0.0,
                          obj_value: float = 0.0, name: str = "folded") -> MILPPair:
    """
    PAIR-CONSTRUCTION ALGORITHM (general form of the cycle family).

    Given a degree d, a constraint sense, and a (uniform) objective value, produce a
    1-WL-equivalent non-isomorphic bipartite-MILP pair at scale k:

      G_A = a CONNECTED d-regular bipartite graph on (kd cons + kd vars), built as a
            circulant with offsets {0,1,...,d-2, d} chosen to be connected.
      G_B = k DISJOINT copies of K_{d,d} on the same (kd + kd) vertices.

    GUARANTEES (proved in the paper, verified by data.milp_pairs_v2.diagnose_pair):
      * both are d-regular on both sides with identical (uniform) features, so 1-WL
        colour refinement collapses to one class per side -> 1-WL EQUIVALENT;
      * G_A is connected (1 component), G_B has k components -> NON-ISOMORPHIC.

    The sense / objective arguments dress the structural collision in the feature
    distribution of a given MILP family (set cover: >=, costs; auctions: <=, prices),
    so the collision is shown inside realistic instance encodings, not only the bare
    uniform case. Uniformity across nodes is required to preserve 1-WL equivalence.
    """
    m = k * d
    offsets = tuple(range(d))                       # {0,1,...,d-1}: offset 1 => connected
    A_a = _circulant_bip(m, offsets)
    assert (A_a.sum(1) == d).all() and (A_a.sum(0) == d).all(), "G_A not d-regular"
    assert _bipartite_components(A_a) == 1, "G_A not connected"
    A_b = _disjoint_Kdd(k, d)
    b = np.ones(m, np.float32)
    c = np.full(m, obj_value, np.float32)
    sense = np.full(m, sense_val, np.float32)
    ma = build_instance(f"{name}_connected_k{k}", A_a, b, c, sense)
    mb = build_instance(f"{name}_disjoint_k{k}", A_b, b, c, sense)
    return MILPPair(ma, mb, f"{name} 1-WL pair (k={k}, d={d})",
                    expected_distinguishable=False, is_1wl_equivalent=True)


def construct_standard_family_pair(family: str, k: int) -> MILPPair:
    """1-WL-equivalent non-isomorphic pair dressed as a standard MILP family."""
    if family == "set_cover":      # cover constraints (>=), positive costs
        return construct_folded_pair(k, d=3, sense_val=2.0, obj_value=1.0, name="set_cover")
    if family == "comb_auction":   # packing constraints (<=), positive prices
        return construct_folded_pair(k, d=3, sense_val=1.0, obj_value=2.0, name="comb_auction")
    if family == "matching":       # 2-regular packing (<=)
        return construct_folded_pair(k, d=2, sense_val=1.0, obj_value=1.0, name="matching")
    if family == "knapsack_blocks":  # multi-knapsack-style 4-regular blocks (<=)
        return construct_folded_pair(k, d=4, sense_val=1.0, obj_value=3.0, name="knapsack")
    raise ValueError(family)


# ---- random instance generators (for the prevalence scan) -------------------

def gen_random_set_cover(m_elems, n_sets, density, rng):
    A = (rng.random((m_elems, n_sets)) < density).astype(np.float32)
    for i in range(m_elems):                          # ensure each element coverable
        if A[i].sum() == 0:
            A[i, rng.integers(n_sets)] = 1.0
    b = np.ones(m_elems, np.float32)
    c = rng.integers(1, 10, n_sets).astype(np.float32)
    return build_instance("rand_sc", A, b, c, np.full(m_elems, 2.0, np.float32))


def gen_random_comb_auction(n_goods, n_bids, density, rng):
    A = (rng.random((n_goods, n_bids)) < density).astype(np.float32)
    for j in range(n_bids):
        if A[:, j].sum() == 0:
            A[rng.integers(n_goods), j] = 1.0
    b = np.ones(n_goods, np.float32)
    c = rng.integers(1, 10, n_bids).astype(np.float32)
    return build_instance("rand_ca", A, b, c, np.full(n_goods, 1.0, np.float32))


def wl_color_class_count(milp, iterations: int = 5) -> int:
    """Number of distinct 1-WL colour classes at the fixed point (node-level)."""
    A = milp.A
    n_cons, n_vars = A.shape
    var_c = [tuple(milp.var_features[j].round(4)) for j in range(n_vars)]
    cons_c = [tuple(milp.cons_features[i].round(4)) for i in range(n_cons)]
    for _ in range(iterations):
        nv = [(var_c[j], tuple(sorted(cons_c[i] for i in range(n_cons) if A[i, j] != 0)))
              for j in range(n_vars)]
        nc = [(cons_c[i], tuple(sorted(var_c[j] for j in range(n_vars) if A[i, j] != 0)))
              for i in range(n_cons)]
        var_c, cons_c = nv, nc
    return len(set(var_c) | set(cons_c))


# ---------------------------------------------------------------------------
# A7 specificity — distinguishable (negative-control) pairs
# ---------------------------------------------------------------------------

def construct_distinguishable_pairs() -> List[MILPPair]:
    """Pairs that 1-WL CAN tell apart (different colour histograms). Used to
    prove the encoders are not constant maps — the bit-identity result is
    non-vacuous only if these produce DIFFERENT embeddings."""
    pairs = []

    # (1) chain vs star on 5x6 (different degree sequences)
    A_chain = np.array([[1,1,0,0,0,0],[0,1,1,0,0,0],[0,0,1,1,0,0],
                        [0,0,0,1,1,0],[0,0,0,0,1,1]], np.float32)
    A_star = np.array([[1,1,0,0,0,0],[1,0,1,0,0,0],[1,0,0,1,0,0],
                       [1,0,0,0,1,0],[1,0,0,0,0,1]], np.float32)
    b = np.ones(5, np.float32)
    pairs.append(MILPPair(build_instance("chain", A_chain, b),
                          build_instance("star", A_star, b),
                          "Distinguishable: chain vs star",
                          expected_distinguishable=True, is_1wl_equivalent=False))

    # (2) C_8 (single 8-cycle, 2-regular) vs path P (degree sequence differs)
    A_cyc = adjacency_C4k(2)                       # 4x4, 2-regular, connected
    A_path = np.array([[1,1,0,0],[0,1,1,0],[0,0,1,1],[1,0,0,1]], np.float32)
    A_path[3] = np.array([0,0,0,1], np.float32)    # break regularity (one deg-1 var)
    b4 = np.ones(4, np.float32)
    pairs.append(MILPPair(build_instance("cyc4", A_cyc, b4),
                          build_instance("irregular", A_path, b4),
                          "Distinguishable: 2-regular vs irregular",
                          expected_distinguishable=True, is_1wl_equivalent=False))

    # (3) different RHS (breaks constraint-feature uniformity)
    A = np.array([[1,1,0,0],[0,0,1,1],[1,0,1,0]], np.float32)
    pairs.append(MILPPair(build_instance("rhs_a", A, np.array([1,1,1], np.float32)),
                          build_instance("rhs_b", A, np.array([1,2,3], np.float32)),
                          "Distinguishable: different RHS",
                          expected_distinguishable=True, is_1wl_equivalent=False))
    return pairs
