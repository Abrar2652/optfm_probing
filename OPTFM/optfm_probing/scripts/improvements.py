"""
Input-only improvements that attempt to break the 1-WL limit without
retraining or modifying the model architecture. All improvements work by
injecting additional information into the three currently-zeroed slots
of the Ecole variable feature vector (indices 6, 7, 8 = basis_status,
reduced_cost, solution_value). The pre-trained model was trained with
these slots set to zero, so small perturbations should not destroy the
learned representation.

Provided transforms:

  rnf            — random node features (independent noise per pair,
                   averaged over K samples). Standard "RNF-GIN" trick
                   from Abboud et al. 2020 / Sato et al. 2021.

  lp_primal      — LP-relaxation solution value x*[j] placed in slot 8.
                   Deterministic per MILP. Distinguishes any pair where
                   the LP optimum differs between the two MILPs.

  lp_reduced     — LP-relaxation reduced cost r*[j] in slot 7.

  lp_primal_dual — LP x* in slot 8 AND dual y*[i] applied to slot 0 of
                   cons features (but we cannot change cons features
                   directly because cons_x has only 1 slot; we instead
                   inject via an additive perturbation to cons_x).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict

import numpy as np

from data.milp_pairs import MILPInstance
from data.milp_pairs_v2 import solve_lp_relaxation


Transform = Callable[[MILPInstance, int], MILPInstance]


# ---------------------------------------------------------------------------
# Random node features
# ---------------------------------------------------------------------------

def make_rnf_transform(sigma: float = 0.3) -> Transform:
    def rnf(milp: MILPInstance, seed: int) -> MILPInstance:
        rng = np.random.default_rng(seed)
        new_var = milp.var_features.copy()
        n = new_var.shape[0]
        new_var[:, 6] = rng.normal(0.0, sigma, size=n)
        new_var[:, 7] = rng.normal(0.0, sigma, size=n)
        new_var[:, 8] = rng.normal(0.0, sigma, size=n)
        return replace(milp, var_features=new_var.astype(np.float32))
    return rnf


# ---------------------------------------------------------------------------
# LP-feature transforms
# ---------------------------------------------------------------------------

def lp_primal_transform(milp: MILPInstance, seed: int = 0) -> MILPInstance:
    sol = solve_lp_relaxation(milp)
    new_var = milp.var_features.copy()
    new_var[:, 8] = sol["x_star"]
    return replace(milp, var_features=new_var.astype(np.float32))


def lp_reduced_transform(milp: MILPInstance, seed: int = 0) -> MILPInstance:
    sol = solve_lp_relaxation(milp)
    new_var = milp.var_features.copy()
    new_var[:, 7] = sol["reduced_costs"]
    new_var[:, 8] = sol["x_star"]
    return replace(milp, var_features=new_var.astype(np.float32))


def lp_primal_dual_transform(milp: MILPInstance, seed: int = 0) -> MILPInstance:
    """Inject LP primal into var slot 8 AND LP dual as an additive
    perturbation to the (single) cons feature (normalized RHS + σ·dual)."""
    sol = solve_lp_relaxation(milp)
    new_var = milp.var_features.copy()
    new_var[:, 8] = sol["x_star"]
    new_cons = milp.cons_features.copy()
    new_cons[:, 0] = new_cons[:, 0] + 0.1 * sol["duals"]
    return replace(
        milp,
        var_features=new_var.astype(np.float32),
        cons_features=new_cons.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Random Walk Positional Encoding (RWPE)
# ---------------------------------------------------------------------------

def make_rwpe_transform(steps=(4, 6, 8), cons_step: int = 4, cons_scale: float = 1.0) -> Transform:
    """
    Random Walk Positional Encoding using selected return-probability
    steps. Dwivedi et al. 2022 ("Graph Neural Networks with Learnable
    Structural and Positional Representations") show RWPE is strictly
    more expressive than 1-WL because return probabilities encode
    multi-step neighborhood information that 1-WL collapses.

    IMPORTANT: for *bipartite* graphs all odd-step return probabilities
    are zero, and the first even step (P^2[i,i] = 1/deg(i)) is the same
    for any regular bipartite graph. The first step at which C_{4k}
    differs from k*C_4 is L = 4 (C_4 has P^4[i,i] = 1/2 whereas C_8 has
    P^4[i,i] = 3/8). We therefore use steps (4, 6, 8) by default.

    Injection:
      var_features[:, 6:6+K]  <- return_probs[var_nodes, steps]
      cons_features[:, 0]     <- cons_features[:, 0] + cons_scale * return_probs[cons, cons_step]
    """
    assert 1 <= len(steps) <= 3, "at most 3 steps (var slots 6,7,8)"
    max_step = max(max(steps), cons_step)

    def transform(milp: MILPInstance, seed: int = 0) -> MILPInstance:
        n_cons, n_vars = milp.A.shape
        N = n_cons + n_vars
        A_full = np.zeros((N, N), dtype=np.float64)
        A_full[:n_cons, n_cons:] = (milp.A != 0).astype(np.float64)
        A_full[n_cons:, :n_cons] = A_full[:n_cons, n_cons:].T
        deg = A_full.sum(axis=1)
        deg_safe = np.where(deg > 0, deg, 1.0)
        P = A_full / deg_safe[:, None]

        Pk = np.eye(N)
        diags = [None]  # 1-indexed lookup
        for _ in range(max_step):
            Pk = Pk @ P
            diags.append(np.diag(Pk).copy())
        # diags[L] is the vector of P^L[i,i] for i in 0..N-1

        var_pe = np.stack([diags[L][n_cons:] for L in steps], axis=1)
        cons_pe = diags[cons_step][:n_cons]

        new_var = milp.var_features.copy()
        new_var[:, 6:6 + var_pe.shape[1]] = var_pe
        new_cons = milp.cons_features.copy()
        new_cons[:, 0] = new_cons[:, 0] + cons_scale * cons_pe
        return replace(
            milp,
            var_features=new_var.astype(np.float32),
            cons_features=new_cons.astype(np.float32),
        )

    return transform


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def virtual_global_node_transform(milp: MILPInstance, seed: int = 0) -> MILPInstance:
    """
    Append two virtual global nodes — one on the constraint side and one
    on the variable side — exactly as OPTFM's pretraining pipeline does
    at OPTFM/node_pretrain/main_mip.py lines 148-171. The global cons
    node carries the mean of the per-constraint features and is connected
    to every variable node with unit edge weight; symmetrically for the
    global var node.

    This transform has no effect on graph-level expressiveness: the new
    nodes carry features that are themselves a symmetric function of the
    existing node multisets, and the extra edges they introduce are a
    fixed pattern that depends only on node counts, not on the identity
    of individual nodes. So if two inputs were 1-WL equivalent before
    the transform, they remain 1-WL equivalent after. This is the
    empirical verification of Lemma 5 in paper/proof_sketch.md.

    Implementation: the virtual nodes are *appended* to the feature
    matrices and the edge list is extended. All existing features and
    edges are preserved.
    """
    n_cons, _ = milp.cons_features.shape
    n_vars, _ = milp.var_features.shape

    # Global cons features = mean of cons features; global var = mean of var
    global_cons_feat = milp.cons_features.mean(axis=0, keepdims=True)
    new_cons_feat = np.concatenate([milp.cons_features, global_cons_feat], axis=0)

    global_var_feat = milp.var_features.mean(axis=0, keepdims=True)
    new_var_feat = np.concatenate([milp.var_features, global_var_feat], axis=0)

    # New edges:
    #   * global cons (index n_cons in the cons block) -- every variable
    #   * every constraint -- global var (index n_vars in the var block)
    new_cons_idx = n_cons          # the global cons is at row n_cons
    new_var_idx = n_vars           # the global var is at col n_vars

    rows = list(milp.edge_index[0]) + [new_cons_idx] * n_vars + list(range(n_cons))
    cols = list(milp.edge_index[1]) + list(range(n_vars)) + [new_var_idx] * n_cons
    new_edge_index = np.array([rows, cols], dtype=np.int64)

    n_new_edges = n_vars + n_cons
    new_edge_attr = np.concatenate([
        milp.edge_attr,
        np.ones((n_new_edges, 1), dtype=np.float32),
    ], axis=0)

    # Extend A (only for bookkeeping / diagnostics downstream)
    new_A = np.zeros((n_cons + 1, n_vars + 1), dtype=np.float32)
    new_A[:n_cons, :n_vars] = milp.A
    new_A[new_cons_idx, :n_vars] = 1.0
    new_A[:n_cons, new_var_idx] = 1.0

    new_b = np.concatenate([milp.b, [float(milp.b.mean())]])
    new_sense = np.concatenate([milp.sense, [0.0]])
    new_c = np.concatenate([milp.c, [0.0]])
    new_lb = np.concatenate([milp.lb, [0.0]])
    new_ub = np.concatenate([milp.ub, [1.0]])
    new_vtype = np.concatenate([milp.vtype, [1.0]])

    return replace(
        milp,
        var_features=new_var_feat.astype(np.float32),
        cons_features=new_cons_feat.astype(np.float32),
        edge_index=new_edge_index,
        edge_attr=new_edge_attr.astype(np.float32),
        c=new_c.astype(np.float32),
        A=new_A,
        b=new_b.astype(np.float32),
        sense=new_sense.astype(np.float32),
        lb=new_lb.astype(np.float32),
        ub=new_ub.astype(np.float32),
        vtype=new_vtype.astype(np.float32),
    )


TRANSFORMS: Dict[str, Transform] = {
    "baseline":       lambda m, s: m,
    "rnf_sigma_0.3":  make_rnf_transform(0.3),
    "rnf_sigma_1.0":  make_rnf_transform(1.0),
    "lp_primal":      lp_primal_transform,
    "lp_reduced":     lp_reduced_transform,
    "lp_primal_dual": lp_primal_dual_transform,
    "rwpe_steps_4_6_8": make_rwpe_transform(steps=(4, 6, 8), cons_step=4, cons_scale=1.0),
    "virtual_global_node": virtual_global_node_transform,
}
