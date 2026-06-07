"""
T1 — Pair-property test suite.

The central empirical claims rest on the test pairs being genuinely
(a) 1-WL equivalent and (b) non-isomorphic with (c) a connectivity mismatch.
If any of these silently fails the paper collapses, so we assert them in code.

Runnable directly (`python tests/test_pairs.py`) or via tests/run_tests.py;
also pytest-compatible.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs import verify_1wl_equivalence
from data.milp_pairs_v2 import (
    construct_bipartite_cycle_pair,
    construct_cubic_bipartite_pair,
    diagnose_pair,
    are_isomorphic_bipartite,
)

CYCLE_KS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]


def _all_pairs():
    pairs = [("cycle_k%d" % k, construct_bipartite_cycle_pair(k)) for k in CYCLE_KS]
    pairs.append(("cubic", construct_cubic_bipartite_pair()))
    return pairs


def test_node_and_edge_counts_match():
    for name, p in _all_pairs():
        a, b = p.milp_a, p.milp_b
        assert a.A.shape == b.A.shape, f"{name}: shape mismatch"
        assert np.count_nonzero(a.A) == np.count_nonzero(b.A), f"{name}: nnz mismatch"


def test_degree_sequences_match():
    for name, p in _all_pairs():
        a, b = p.milp_a, p.milp_b
        assert np.array_equal(np.sort(a.A.sum(1)), np.sort(b.A.sum(1))), f"{name}: row degs"
        assert np.array_equal(np.sort(a.A.sum(0)), np.sort(b.A.sum(0))), f"{name}: col degs"


def test_1wl_equivalent_all_depths():
    # 1-WL refinement converges in <= 1 iteration on these uniform-regular
    # graphs (every node collapses to one colour class), so depths {1,2,3,5}
    # already certify equivalence at the fixed point. (The pure-python colour
    # tuples nest with depth, so we keep depths shallow and instead verify a
    # *wide* range of pair sizes.)
    for name, p in _all_pairs():
        for depth in (1, 2, 3, 5):
            assert verify_1wl_equivalence(p.milp_a, p.milp_b, iterations=depth), \
                f"{name}: not 1-WL equivalent at depth {depth}"


def test_non_isomorphic():
    # Full (brute-force / VF2) isomorphism check is only tractable for small
    # graphs, so we run it for n<=12. For larger graphs a component-count
    # mismatch is itself a graph-isomorphism invariant (see test_component_
    # mismatch), which rigorously certifies non-isomorphism.
    for name, p in _all_pairs():
        n_cons, n_vars = p.milp_a.A.shape
        if n_cons <= 6 and n_vars <= 6:
            assert not are_isomorphic_bipartite(p.milp_a, p.milp_b), \
                f"{name}: unexpectedly isomorphic"


def test_component_mismatch():
    for name, p in _all_pairs():
        d = diagnose_pair(p)
        assert d["A_components"] != d["B_components"], f"{name}: components match (bad)"


if __name__ == "__main__":
    for fn in [test_node_and_edge_counts_match, test_degree_sequences_match,
               test_1wl_equivalent_all_depths, test_non_isomorphic,
               test_component_mismatch]:
        fn()
        print(f"PASS {fn.__name__}")
    print("test_pairs: all passed")
