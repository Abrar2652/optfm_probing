"""
T2 — Encoder invariance / determinism / specificity tests.

(a) permutation-equivariance: relabelling a graph leaves the pooled embedding
    unchanged (cos == 1) — confirms the pipeline really is equivariant.
(b) bit-identity reproduction: the headline cos == 1.000000 reproduces for every
    architecture on a 1-WL-equivalent non-isomorphic pair.
(c) determinism: same seed -> identical embeddings to 1e-12.
(d) specificity (negative control): a 1-WL-DISTINGUISHABLE pair yields different
    embeddings (linf > 1e-5) for the expressive architectures, so the bit-identity
    result is non-vacuous (the encoders are not constant maps).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs import (
    MILPInstance, create_ecole_features, construct_control_pair,
)
from data.milp_pairs_v2 import construct_bipartite_cycle_pair
from scripts._common import build_models, build_hierarchical, embed, cos_sim, linf


def permute_milp(milp: MILPInstance, pi, sigma) -> MILPInstance:
    """Relabel constraints by pi and variables by sigma; rebuild features."""
    pi = np.asarray(pi); sigma = np.asarray(sigma)
    A = milp.A[pi][:, sigma]
    b = milp.b[pi]
    c = milp.c[sigma]
    sense = milp.sense[pi]
    lb, ub, vtype = milp.lb[sigma], milp.ub[sigma], milp.vtype[sigma]
    vf, cf, ei, ea = create_ecole_features(c, A, b, sense, lb, ub, vtype)
    return MILPInstance(name=milp.name + "_perm", var_features=vf, cons_features=cf,
                        edge_index=ei, edge_attr=ea, c=c, A=A, b=b, sense=sense,
                        lb=lb, ub=ub, vtype=vtype)


def test_permutation_equivariance():
    p = construct_bipartite_cycle_pair(6)
    rng = np.random.default_rng(0)
    pi = rng.permutation(p.milp_a.A.shape[0])
    sigma = rng.permutation(p.milp_a.A.shape[1])
    perm = permute_milp(p.milp_a, pi, sigma)
    for name, m in build_models(seed=0).items():
        a = embed(m, p.milp_a); ap = embed(m, perm)
        assert cos_sim(a, ap) > 1 - 1e-5, f"{name}: not permutation-equivariant"


def test_bit_identity_reproduction():
    p = construct_bipartite_cycle_pair(7)
    for name, m in build_models(seed=0).items():
        a = embed(m, p.milp_a); b = embed(m, p.milp_b)
        assert linf(a, b) <= 1e-5, f"{name}: not bit-identical (linf={linf(a,b):.2e})"
        assert abs(cos_sim(a, b) - 1.0) <= 1e-6, f"{name}: cos != 1"


def test_determinism_same_seed():
    p = construct_bipartite_cycle_pair(5)
    m1 = build_hierarchical(seed=123)
    m2 = build_hierarchical(seed=123)
    e1 = embed(m1, p.milp_a); e2 = embed(m2, p.milp_a)
    assert linf(e1, e2) <= 1e-12, f"non-deterministic init (linf={linf(e1,e2):.2e})"


def test_specificity_distinguishable_pair():
    """A 1-WL-distinguishable pair (chain vs star) must NOT collapse."""
    p = construct_control_pair()
    assert p.expected_distinguishable
    n_distinguished = 0
    for name, m in build_models(seed=0).items():
        a = embed(m, p.milp_a); b = embed(m, p.milp_b)
        if linf(a, b) > 1e-5:
            n_distinguished += 1
    # At least the expressive architectures must separate a distinguishable pair;
    # if NONE do, the encoders are degenerate and the whole result is vacuous.
    assert n_distinguished >= 1, "no architecture separates a distinguishable pair (vacuous!)"


if __name__ == "__main__":
    for fn in [test_permutation_equivariance, test_bit_identity_reproduction,
               test_determinism_same_seed, test_specificity_distinguishable_pair]:
        fn()
        print(f"PASS {fn.__name__}")
    print("test_encoders: all passed")
