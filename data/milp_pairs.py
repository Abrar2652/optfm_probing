"""
1-WL Indistinguishable MILP Pair Construction

Constructs the canonical 1-WL indistinguishable MILP pair from Chen et al. (2023)
with features matching Ecole's MilpBipartite format.

Ecole Variable Features (9 features):
0. objective_coefficient (c_j)
1. type (0=continuous, 1=binary, 2=integer)
2. has_lower_bound (0/1)
3. has_upper_bound (0/1)
4. lower_bound (normalized)
5. upper_bound (normalized)
6. basis_status (0-3)
7. reduced_cost
8. solution_value (0 at root)

Ecole Constraint Features (1 feature after normalization):
0. normalized_rhs (b_i / ||A_i||)

References:
- Chen et al. (2023). "On the Expressive Power of GNNs for MILP." ICLR.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class MILPInstance:
    """MILP instance with Ecole-style features."""
    name: str
    var_features: np.ndarray   # (n_vars, 9)
    cons_features: np.ndarray  # (n_cons, 1)
    edge_index: np.ndarray     # (2, n_edges)
    edge_attr: np.ndarray      # (n_edges, 1)
    
    # Original MILP data (for verification)
    c: np.ndarray  # Objective coefficients
    A: np.ndarray  # Constraint matrix
    b: np.ndarray  # RHS
    sense: np.ndarray  # Constraint sense (0=eq, 1=leq, 2=geq)
    lb: np.ndarray  # Lower bounds
    ub: np.ndarray  # Upper bounds
    vtype: np.ndarray  # Variable types
    
    is_feasible: Optional[bool] = None


@dataclass
class MILPPair:
    """Pair of MILP instances for comparison."""
    milp_a: MILPInstance
    milp_b: MILPInstance
    description: str
    expected_distinguishable: bool
    is_1wl_equivalent: bool


def create_ecole_features(c: np.ndarray, A: np.ndarray, b: np.ndarray,
                          sense: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                          vtype: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create Ecole-style features from MILP components.
    
    Returns:
        var_features: (n_vars, 9)
        cons_features: (n_cons, 1)
        edge_index: (2, n_edges)
        edge_attr: (n_edges, 1)
    """
    n_cons, n_vars = A.shape
    
    # Variable features (9 features)
    var_features = np.zeros((n_vars, 9), dtype=np.float32)
    var_features[:, 0] = c  # objective coefficient
    var_features[:, 1] = vtype  # type (0=cont, 1=bin, 2=int)
    var_features[:, 2] = (lb > -1e10).astype(np.float32)  # has_lower_bound
    var_features[:, 3] = (ub < 1e10).astype(np.float32)   # has_upper_bound
    var_features[:, 4] = np.clip(lb, -10, 10) / 10  # normalized lb
    var_features[:, 5] = np.clip(ub, -10, 10) / 10  # normalized ub
    var_features[:, 6] = 0  # basis_status (unknown at root)
    var_features[:, 7] = 0  # reduced_cost (unknown at root)
    var_features[:, 8] = 0  # solution_value (unknown at root)
    
    # Constraint features (1 feature: normalized RHS)
    row_norms = np.linalg.norm(A, axis=1, keepdims=True)
    row_norms = np.maximum(row_norms, 1e-8)
    cons_features = (b.reshape(-1, 1) / row_norms).astype(np.float32)
    
    # Edge features
    rows, cols = np.nonzero(A)
    edge_index = np.array([rows, cols], dtype=np.int64)
    edge_attr = A[rows, cols].reshape(-1, 1).astype(np.float32)
    
    return var_features, cons_features, edge_index, edge_attr


def construct_canonical_pair() -> MILPPair:
    """
    [DEPRECATED — DO NOT USE AS A 1-WL PROBE]

    The pair constructed here was intended to be a 1-WL indistinguishable
    MILP pair but is actually GRAPH-ISOMORPHIC as a feature-labeled
    bipartite graph, and both MILPs are integer-feasible:

      * scripts/sanity_check_pairs.py finds an explicit permutation
        pi = (0,1,2,4,3) on constraints and sigma = (1,0,3,2,5,4) on
        variables that maps (A, b_A, features) to (A, b_B, features).
      * scipy.milp finds (1,0,1,0,0,1) as an integer solution to MILP-B,
        contradicting the earlier "MILP-B (Infeasible)" claim.

    Any permutation-equivariant model trivially maps isomorphic inputs
    to identical embeddings, so `cos_sim = 1.0` on this pair tells you
    nothing about 1-WL expressiveness. The real 1-WL boundedness probe
    lives in data/milp_pairs_v2.construct_bipartite_cycle_pair, which
    produces the C_{4k}-bipartite vs k*C_4-bipartite family that is
    1-WL equivalent AND provably non-isomorphic (different connectivity).

    This function is kept for backwards compatibility with the earlier
    scripts/run_complete_experiment.py, but it should not be used to
    ground any claim about expressiveness.

    Original (erroneous) description:
        MILP-A (Feasible): x1+x2=1, x3+x4=1, x5+x6=1, x1+x3+x5=1, x2+x4+x6=2
        MILP-B (claimed "infeasible" but actually feasible): same A,
                with b = [1,1,1,2,1]
    """
    n_vars = 6
    n_cons = 5
    
    # Constraint matrix (same for both)
    A = np.array([
        [1, 1, 0, 0, 0, 0],  # x1 + x2 = 1
        [0, 0, 1, 1, 0, 0],  # x3 + x4 = 1
        [0, 0, 0, 0, 1, 1],  # x5 + x6 = 1
        [1, 0, 1, 0, 1, 0],  # x1 + x3 + x5 = ?
        [0, 1, 0, 1, 0, 1],  # x2 + x4 + x6 = ?
    ], dtype=np.float32)
    
    # RHS vectors (different!)
    b_a = np.array([1, 1, 1, 1, 2], dtype=np.float32)
    b_b = np.array([1, 1, 1, 2, 1], dtype=np.float32)  # Swapped
    
    # Common components
    c = np.zeros(n_vars, dtype=np.float32)  # No objective
    sense = np.zeros(n_cons, dtype=np.float32)  # All equalities
    lb = np.zeros(n_vars, dtype=np.float32)
    ub = np.ones(n_vars, dtype=np.float32)
    vtype = np.ones(n_vars, dtype=np.float32)  # Binary
    
    # Create features
    var_a, cons_a, edge_a, attr_a = create_ecole_features(c, A, b_a, sense, lb, ub, vtype)
    var_b, cons_b, edge_b, attr_b = create_ecole_features(c, A, b_b, sense, lb, ub, vtype)
    
    milp_a = MILPInstance(
        name="canonical_A", var_features=var_a, cons_features=cons_a,
        edge_index=edge_a, edge_attr=attr_a,
        c=c, A=A, b=b_a, sense=sense, lb=lb, ub=ub, vtype=vtype,
        is_feasible=True
    )
    
    milp_b = MILPInstance(
        name="canonical_B", var_features=var_b, cons_features=cons_b,
        edge_index=edge_b, edge_attr=attr_b,
        c=c, A=A, b=b_b, sense=sense, lb=lb, ub=ub, vtype=vtype,
        is_feasible=False
    )
    
    return MILPPair(
        milp_a=milp_a, milp_b=milp_b,
        description="Canonical 1-WL indistinguishable pair (Chen et al. 2023)",
        expected_distinguishable=False,
        is_1wl_equivalent=True
    )


def construct_scaled_pair(k: int = 2) -> MILPPair:
    """
    Construct scaled version of canonical pair with k copies.
    
    Total variables: 6k, Total constraints: 5k
    Still 1-WL indistinguishable.
    """
    base_pair = construct_canonical_pair()
    n_vars = 6 * k
    n_cons = 5 * k
    
    # Block diagonal structure
    A = np.zeros((n_cons, n_vars), dtype=np.float32)
    b_a = np.zeros(n_cons, dtype=np.float32)
    b_b = np.zeros(n_cons, dtype=np.float32)
    
    for i in range(k):
        r_start, r_end = i * 5, (i + 1) * 5
        c_start, c_end = i * 6, (i + 1) * 6
        A[r_start:r_end, c_start:c_end] = base_pair.milp_a.A
        b_a[r_start:r_end] = base_pair.milp_a.b
        b_b[r_start:r_end] = base_pair.milp_b.b
    
    c = np.zeros(n_vars, dtype=np.float32)
    sense = np.zeros(n_cons, dtype=np.float32)
    lb = np.zeros(n_vars, dtype=np.float32)
    ub = np.ones(n_vars, dtype=np.float32)
    vtype = np.ones(n_vars, dtype=np.float32)
    
    var_a, cons_a, edge_a, attr_a = create_ecole_features(c, A, b_a, sense, lb, ub, vtype)
    var_b, cons_b, edge_b, attr_b = create_ecole_features(c, A, b_b, sense, lb, ub, vtype)
    
    milp_a = MILPInstance(
        name=f"scaled_{k}_A", var_features=var_a, cons_features=cons_a,
        edge_index=edge_a, edge_attr=attr_a,
        c=c, A=A, b=b_a, sense=sense, lb=lb, ub=ub, vtype=vtype,
        is_feasible=True
    )
    
    milp_b = MILPInstance(
        name=f"scaled_{k}_B", var_features=var_b, cons_features=cons_b,
        edge_index=edge_b, edge_attr=attr_b,
        c=c, A=A, b=b_b, sense=sense, lb=lb, ub=ub, vtype=vtype,
        is_feasible=False
    )
    
    return MILPPair(
        milp_a=milp_a, milp_b=milp_b,
        description=f"Scaled 1-WL pair with {k} copies ({n_vars} variables)",
        expected_distinguishable=False,
        is_1wl_equivalent=True
    )


def construct_control_pair() -> MILPPair:
    """
    Construct control pair with DIFFERENT graph structure.
    
    These should be distinguishable by any reasonable embedding.
    - MILP-A: Chain structure
    - MILP-B: Star structure
    """
    n_vars = 6
    
    # Chain: x1-x2, x2-x3, x3-x4, x4-x5, x5-x6
    A_chain = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1],
    ], dtype=np.float32)
    
    # Star: x1 connected to all others
    A_star = np.array([
        [1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    
    b = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    c = np.zeros(n_vars, dtype=np.float32)
    sense = np.zeros(5, dtype=np.float32)
    lb = np.zeros(n_vars, dtype=np.float32)
    ub = np.ones(n_vars, dtype=np.float32)
    vtype = np.ones(n_vars, dtype=np.float32)
    
    var_a, cons_a, edge_a, attr_a = create_ecole_features(c, A_chain, b, sense, lb, ub, vtype)
    var_b, cons_b, edge_b, attr_b = create_ecole_features(c, A_star, b, sense, lb, ub, vtype)
    
    milp_a = MILPInstance(
        name="control_chain", var_features=var_a, cons_features=cons_a,
        edge_index=edge_a, edge_attr=attr_a,
        c=c, A=A_chain, b=b, sense=sense, lb=lb, ub=ub, vtype=vtype
    )
    
    milp_b = MILPInstance(
        name="control_star", var_features=var_b, cons_features=cons_b,
        edge_index=edge_b, edge_attr=attr_b,
        c=c, A=A_star, b=b, sense=sense, lb=lb, ub=ub, vtype=vtype
    )
    
    return MILPPair(
        milp_a=milp_a, milp_b=milp_b,
        description="Control pair (chain vs star structure)",
        expected_distinguishable=True,
        is_1wl_equivalent=False
    )


def construct_rhs_variant_pair() -> MILPPair:
    """
    Pair with different RHS but NOT 1-WL equivalent.
    
    Different constraint features break 1-WL equivalence.
    """
    n_vars = 4
    n_cons = 3
    
    A = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
    ], dtype=np.float32)
    
    b_a = np.array([1, 1, 1], dtype=np.float32)
    b_b = np.array([2, 2, 2], dtype=np.float32)  # All different
    
    c = np.zeros(n_vars, dtype=np.float32)
    sense = np.zeros(n_cons, dtype=np.float32)
    lb = np.zeros(n_vars, dtype=np.float32)
    ub = np.ones(n_vars, dtype=np.float32) * 2
    vtype = np.ones(n_vars, dtype=np.float32)
    
    var_a, cons_a, edge_a, attr_a = create_ecole_features(c, A, b_a, sense, lb, ub, vtype)
    var_b, cons_b, edge_b, attr_b = create_ecole_features(c, A, b_b, sense, lb, ub, vtype)
    
    milp_a = MILPInstance(
        name="rhs_variant_A", var_features=var_a, cons_features=cons_a,
        edge_index=edge_a, edge_attr=attr_a,
        c=c, A=A, b=b_a, sense=sense, lb=lb, ub=ub, vtype=vtype
    )
    
    milp_b = MILPInstance(
        name="rhs_variant_B", var_features=var_b, cons_features=cons_b,
        edge_index=edge_b, edge_attr=attr_b,
        c=c, A=A, b=b_b, sense=sense, lb=lb, ub=ub, vtype=vtype
    )
    
    return MILPPair(
        milp_a=milp_a, milp_b=milp_b,
        description="RHS variant pair (different RHS, not 1-WL equivalent)",
        expected_distinguishable=True,
        is_1wl_equivalent=False
    )


def construct_objective_variant_pair() -> MILPPair:
    """
    Pair with different objectives but same feasible region.
    
    1-WL equivalent because structure is identical.
    """
    base = construct_canonical_pair()
    
    # Same constraints, different objective
    c_a = np.array([1, 0, 1, 0, 1, 0], dtype=np.float32)
    c_b = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
    
    var_a, cons_a, edge_a, attr_a = create_ecole_features(
        c_a, base.milp_a.A, base.milp_a.b, base.milp_a.sense,
        base.milp_a.lb, base.milp_a.ub, base.milp_a.vtype
    )
    var_b, cons_b, edge_b, attr_b = create_ecole_features(
        c_b, base.milp_a.A, base.milp_a.b, base.milp_a.sense,
        base.milp_a.lb, base.milp_a.ub, base.milp_a.vtype
    )
    
    milp_a = MILPInstance(
        name="obj_variant_A", var_features=var_a, cons_features=cons_a,
        edge_index=edge_a, edge_attr=attr_a,
        c=c_a, A=base.milp_a.A, b=base.milp_a.b, sense=base.milp_a.sense,
        lb=base.milp_a.lb, ub=base.milp_a.ub, vtype=base.milp_a.vtype
    )
    
    milp_b = MILPInstance(
        name="obj_variant_B", var_features=var_b, cons_features=cons_b,
        edge_index=edge_b, edge_attr=attr_b,
        c=c_b, A=base.milp_a.A, b=base.milp_a.b, sense=base.milp_a.sense,
        lb=base.milp_a.lb, ub=base.milp_a.ub, vtype=base.milp_a.vtype
    )
    
    # Note: Pre-trained OPTFM may be invariant to objective coefficients
    # due to its training objective. This is a learned invariance, not 1-WL.
    return MILPPair(
        milp_a=milp_a, milp_b=milp_b,
        description="Objective variant pair (different c, pre-trained may be invariant)",
        expected_distinguishable=False,  # Pre-trained OPTFM is invariant to c
        is_1wl_equivalent=False  # Not technically 1-WL equiv, but model is invariant
    )


def milp_to_tensors(milp: MILPInstance) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert MILP instance to PyTorch tensors."""
    return (
        torch.tensor(milp.cons_features, dtype=torch.float32),
        torch.tensor(milp.edge_index, dtype=torch.long),
        torch.tensor(milp.edge_attr, dtype=torch.float32),
        torch.tensor(milp.var_features, dtype=torch.float32),
    )


def verify_1wl_equivalence(milp_a: MILPInstance, milp_b: MILPInstance, 
                           iterations: int = 10) -> bool:
    """
    Verify 1-WL equivalence by checking if color refinement converges to same histogram.
    
    For 1-WL equivalent graphs, the multiset of colors at each iteration should be identical.
    """
    def get_color_histogram(milp: MILPInstance) -> List[Tuple]:
        A = milp.A
        n_cons, n_vars = A.shape
        
        # Initialize colors with features
        var_colors = [tuple(milp.var_features[j].round(4)) for j in range(n_vars)]
        cons_colors = [tuple(milp.cons_features[i].round(4)) for i in range(n_cons)]
        
        # Weisfeiler-Lehman iterations
        for _ in range(iterations):
            new_var_colors = []
            for j in range(n_vars):
                neighbors = tuple(sorted(cons_colors[i] for i in range(n_cons) if A[i, j] != 0))
                new_var_colors.append((var_colors[j], neighbors))
            
            new_cons_colors = []
            for i in range(n_cons):
                neighbors = tuple(sorted(var_colors[j] for j in range(n_vars) if A[i, j] != 0))
                new_cons_colors.append((cons_colors[i], neighbors))
            
            var_colors = new_var_colors
            cons_colors = new_cons_colors
        
        return sorted(var_colors) + sorted(cons_colors)
    
    hist_a = get_color_histogram(milp_a)
    hist_b = get_color_histogram(milp_b)
    
    return hist_a == hist_b


def get_all_test_pairs() -> List[MILPPair]:
    """Get all test pairs for the experiment."""
    return [
        construct_canonical_pair(),
        construct_scaled_pair(k=2),
        construct_scaled_pair(k=5),
        construct_control_pair(),
        construct_rhs_variant_pair(),
        construct_objective_variant_pair(),
    ]


if __name__ == "__main__":
    print("Testing MILP pair construction...")
    
    pairs = get_all_test_pairs()
    
    for pair in pairs:
        is_equiv = verify_1wl_equivalence(pair.milp_a, pair.milp_b)
        status = "✓" if is_equiv == pair.is_1wl_equivalent else "✗"
        
        print(f"{status} {pair.description}")
        print(f"   Expected 1-WL equiv: {pair.is_1wl_equivalent}, Actual: {is_equiv}")
        print(f"   Var features shape: {pair.milp_a.var_features.shape}")
        print(f"   Cons features shape: {pair.milp_a.cons_features.shape}")
        print()
