#!/usr/bin/env python3
"""
Structural and primal-dual probe battery on frozen OPTFM embeddings.

Question addressed: Does the pre-trained OPTFM/SGFormer learned
embedding implicitly encode either (a) primal-dual / Lagrangian
information or (b) structural graph properties that 1-WL cannot compute?

Freeze the backbone and train linear / MLP probes on top to predict:

  * n_components       — number of connected components in the bipartite
                         graph. For the 1-WL equivalent pair family the
                         two graphs differ by exactly this invariant
                         (C_{4k} is connected, k*C_4 has k components),
                         and 1-WL CANNOT compute it because at every
                         iteration every node's neighborhood multiset
                         is identical in both graphs.
  * girth_le_4         — binary: "is there a 4-cycle in the underlying
                         bipartite graph". True for k*C_4 (k >= 1),
                         false for C_{4k} with k >= 2. 1-WL cannot
                         compute this either.
  * lp_value           — LP relaxation optimal value with a per-instance
                         random objective. For random MILPs this varies
                         by construction; for 1-WL equivalent pairs it
                         can differ between MILP-A and MILP-B (and does,
                         for certain c vectors).
  * feasible           — is the MILP integer-feasible. Mixed in the
                         random dataset.

Two datasets:

  D_random:  random regular bipartite MILPs at (n_cons, n_vars) = (10, 10)
             with n_components / girth / feasibility / lp_value varying
             by construction. Positive control for "probes work at all".

  D_1wl:     C_{4k} bipartite and k*C_4 bipartite for k in k_values_1wl.
             Every k produces two instances (A and B) that are 1-WL
             equivalent. If the probe can still predict n_components
             here, the embedding encodes something beyond 1-WL. If not,
             the embedding is 1-WL bounded in a probe-detectable sense.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sgformer_mip import create_model
from data.milp_pairs import MILPInstance, create_ecole_features, milp_to_tensors
from data.milp_pairs_v2 import (
    _bipartite_components, construct_bipartite_cycle_pair, solve_lp_relaxation,
)
from scripts._tee_log import start_logging

CKPT = "D:/GitHub/OPTFM/node_pretrain/Models_SCIP/model_params_epoch_31.pth"


# ---------------------------------------------------------------------------
# Graph invariants
# ---------------------------------------------------------------------------

def bipartite_girth_le_k(A: np.ndarray, k: int = 4) -> bool:
    """Return True iff the bipartite graph contains a cycle of length <= k.
    Bipartite graphs have only even cycles, so this is trivially True
    for k >= 4 iff there's a 4-cycle. We use BFS from every node."""
    n_cons, n_vars = A.shape
    N = n_cons + n_vars
    adj = [[] for _ in range(N)]
    rows, cols = np.nonzero(A)
    for r, c in zip(rows, cols):
        adj[int(r)].append(int(n_cons + c))
        adj[int(n_cons + c)].append(int(r))
    for start in range(N):
        dist = [-1] * N
        parent = [-1] * N
        dist[start] = 0
        queue = [start]
        while queue:
            u = queue.pop(0)
            if dist[u] >= k // 2:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Cycle of length dist[u] + dist[v] + 1
                    if dist[u] + dist[v] + 1 <= k:
                        return True
    return False


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def random_bipartite_milp(n_cons: int, n_vars: int, seed: int) -> MILPInstance:
    """Random sparse bipartite MILP with mixed equality/inequality
    constraints. Feasibility is non-trivial."""
    rng = np.random.default_rng(seed)
    density = 2.5 / n_vars
    A = (rng.random((n_cons, n_vars)) < density).astype(np.float32)
    # Make sure no empty rows or columns
    for i in range(n_cons):
        if A[i].sum() == 0:
            A[i, rng.integers(0, n_vars)] = 1.0
    for j in range(n_vars):
        if A[:, j].sum() == 0:
            A[rng.integers(0, n_cons), j] = 1.0
    b = rng.integers(1, 4, size=n_cons).astype(np.float32)
    c = rng.normal(0, 1, size=n_vars).astype(np.float32)
    # 50/50 equality/leq for a harder feasibility mix
    sense = (rng.random(n_cons) < 0.5).astype(np.float32)   # 0 eq, 1 leq
    lb = np.zeros(n_vars, dtype=np.float32)
    ub = np.ones(n_vars, dtype=np.float32)
    vtype = np.ones(n_vars, dtype=np.float32)
    var_feat, cons_feat, edge_idx, edge_attr = create_ecole_features(
        c, A, b, sense, lb, ub, vtype
    )
    return MILPInstance(
        name=f"rand_{seed}", var_features=var_feat, cons_features=cons_feat,
        edge_index=edge_idx, edge_attr=edge_attr,
        c=c, A=A, b=b, sense=sense, lb=lb, ub=ub, vtype=vtype,
    )


def build_1wl_dataset_with_varying_c(k_values: list, n_objectives: int = 5):
    """
    For every k, create n_objectives random objective vectors, and for
    each of them build both the C_{4k}-bipartite and k*C_4-bipartite MILP.
    These remain 1-WL equivalent (var features stay zero -> uniform)
    because we don't propagate c into var_features.

    That is: we keep var_features = zeros, but store c in the MILPInstance
    so that the LP solver uses it. The embedding therefore cannot
    possibly know about c — but the LP value DOES depend on (A, c),
    which is what makes the probe informative.
    """
    instances = []
    rng = np.random.default_rng(42)
    for k in k_values:
        pair_base = construct_bipartite_cycle_pair(k)
        n_vars = pair_base.milp_a.A.shape[1]
        for obj_idx in range(n_objectives):
            c = rng.normal(0, 1, size=n_vars).astype(np.float32)
            ma = replace(pair_base.milp_a, c=c)
            mb = replace(pair_base.milp_b, c=c)
            instances.append(ma)
            instances.append(mb)
    return instances


def build_1wl_dataset_uniform(k_values: list):
    """One instance per (k, side) with zero objective. LP value is uninteresting
    but n_components still varies between A and B."""
    instances = []
    for k in k_values:
        pair = construct_bipartite_cycle_pair(k)
        instances.append(pair.milp_a)
        instances.append(pair.milp_b)
    return instances


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def compute_targets(instances: list) -> dict:
    n = len(instances)
    n_comp   = np.zeros(n, dtype=np.float32)
    girth4   = np.zeros(n, dtype=np.int64)
    lp_val   = np.zeros(n, dtype=np.float32)
    feas     = np.zeros(n, dtype=np.int64)

    for i, m in enumerate(instances):
        n_comp[i] = float(_bipartite_components(m.A))
        girth4[i] = int(bipartite_girth_le_k(m.A, k=4))
        sol = solve_lp_relaxation(m, objective=m.c)
        feas[i] = int(sol["feasible"])
        lp_val[i] = float(sol["obj_value"]) if sol["feasible"] else 0.0
    return {
        "n_components": n_comp,
        "girth_le_4":   girth4,
        "lp_value":     lp_val,
        "feasible":     feas,
    }


# ---------------------------------------------------------------------------
# Embedding extraction (handles instances with different sizes)
# ---------------------------------------------------------------------------

def extract_embedding(model: nn.Module, milp: MILPInstance) -> torch.Tensor:
    cons_x, ei, ea, var_x = milp_to_tensors(milp)
    with torch.no_grad():
        return model.get_graph_embedding(cons_x, ei, ea, var_x)


def build_embedding_matrix(model: nn.Module, instances: list) -> np.ndarray:
    return torch.stack([extract_embedding(model, m) for m in instances]).cpu().numpy()


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.fc(x)


class MLPProbe(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")   # degenerate target
    return float(1 - ss_res / ss_tot)


def train_regression_probe(X, y, probe_cls, epochs=800, lr=1e-2, train_frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_train = max(2, int(train_frac * n))
    idx_tr, idx_te = perm[:n_train], perm[n_train:]
    if len(idx_te) == 0:
        idx_te = idx_tr
    y_tr = y[idx_tr].reshape(-1, 1).astype(np.float32)
    y_te = y[idx_te].reshape(-1, 1).astype(np.float32)
    X_tr_t = torch.as_tensor(X[idx_tr], dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_tr,       dtype=torch.float32)
    X_te_t = torch.as_tensor(X[idx_te], dtype=torch.float32)

    probe = probe_cls(X.shape[1], 1)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        loss = F.mse_loss(probe(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        pred_tr = probe(X_tr_t).cpu().numpy()
        pred_te = probe(X_te_t).cpu().numpy()
    return {
        "r2_train": r2_score(y_tr, pred_tr),
        "r2_test":  r2_score(y_te, pred_te),
    }


def train_classification_probe(X, y, probe_cls, epochs=800, lr=1e-2, train_frac=0.7, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_train = max(2, int(train_frac * n))
    idx_tr, idx_te = perm[:n_train], perm[n_train:]
    if len(idx_te) == 0:
        idx_te = idx_tr
    y_int = y.astype(np.int64)
    n_classes = int(max(y_int.max() + 1, 2))
    X_tr_t = torch.as_tensor(X[idx_tr], dtype=torch.float32)
    y_tr_t = torch.as_tensor(y_int[idx_tr], dtype=torch.long)
    X_te_t = torch.as_tensor(X[idx_te], dtype=torch.float32)
    y_te_t = torch.as_tensor(y_int[idx_te], dtype=torch.long)

    probe = probe_cls(X.shape[1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        logits = probe(X_tr_t)
        loss = F.cross_entropy(logits, y_tr_t)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        pred_tr = probe(X_tr_t).argmax(1)
        pred_te = probe(X_te_t).argmax(1)
    return {
        "acc_train": float((pred_tr == y_tr_t).float().mean().item()),
        "acc_test":  float((pred_te == y_te_t).float().mean().item()),
        "majority_rate_train": float(torch.bincount(y_tr_t).max().item() / len(y_tr_t)),
        "majority_rate_test":  float(torch.bincount(y_te_t).max().item() / len(y_te_t)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_random", type=int, default=600)
    parser.add_argument("--rand_n_cons", type=int, default=10)
    parser.add_argument("--rand_n_vars", type=int, default=10)
    parser.add_argument("--k_values", type=int, nargs="+",
                        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20])
    parser.add_argument("--n_objectives", type=int, default=10,
                        help="number of random objectives per k in 1-WL dataset")
    parser.add_argument("--output_dir", type=str, default="results/probes")
    parser.add_argument("--n_seeds", type=int, default=3)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir) / ts
    out.mkdir(parents=True, exist_ok=True)

    start_logging("probe_primal_dual", log_dir=out)
    print("=" * 78)
    print("STRUCTURAL & PRIMAL-DUAL PROBE BATTERY")
    print("=" * 78)

    # --- datasets ---
    print("\n[1/5] building random-MILP dataset (positive control)...")
    D_rand = [random_bipartite_milp(args.rand_n_cons, args.rand_n_vars, s)
              for s in range(args.n_random)]
    print(f"  {len(D_rand)} random MILPs  (n_cons={args.rand_n_cons} n_vars={args.rand_n_vars})")

    print("\n[2/5] building 1-WL equivalent dataset (varying c)...")
    D_1wl = build_1wl_dataset_with_varying_c(args.k_values, args.n_objectives)
    print(f"  {len(D_1wl)} instances  k={args.k_values}  n_objectives={args.n_objectives}")

    print("\n[3/5] computing targets...")
    T_rand = compute_targets(D_rand)
    T_1wl  = compute_targets(D_1wl)
    print(f"  random  n_components:   range [{T_rand['n_components'].min()}, {T_rand['n_components'].max()}]  mean {T_rand['n_components'].mean():.2f}")
    print(f"  random  girth_le_4 rate:{T_rand['girth_le_4'].mean():.3f}")
    print(f"  random  feasible rate : {T_rand['feasible'].mean():.3f}")
    print(f"  random  lp_value range: [{T_rand['lp_value'].min():.3f}, {T_rand['lp_value'].max():.3f}]")
    print(f"  1-WL    n_components:   range [{T_1wl['n_components'].min()}, {T_1wl['n_components'].max()}]  mean {T_1wl['n_components'].mean():.2f}")
    print(f"  1-WL    girth_le_4 rate:{T_1wl['girth_le_4'].mean():.3f}")
    print(f"  1-WL    lp_value range: [{T_1wl['lp_value'].min():.3f}, {T_1wl['lp_value'].max():.3f}]")

    # --- embeddings ---
    print("\n[4/5] extracting embeddings with pretrained OPTFM/SGFormer...")
    model = create_model("optfm", pretrained_path=CKPT)
    model.eval()
    X_rand = build_embedding_matrix(model, D_rand)
    X_1wl  = build_embedding_matrix(model, D_1wl)
    print(f"  X_rand shape: {X_rand.shape}   X_1wl shape: {X_1wl.shape}")

    # --- probes ---
    print("\n[5/5] training probes (averaging over random seeds)...")
    reg_targets = ["n_components", "lp_value"]
    cls_targets = ["girth_le_4", "feasible"]

    def run_all(X, T):
        res = {}
        for t in reg_targets:
            y = T[t]
            for probe_cls, pname in [(LinearProbe, "linear"), (MLPProbe, "mlp")]:
                r2s = []
                for s in range(args.n_seeds):
                    r = train_regression_probe(X, y, probe_cls, seed=s)
                    r2s.append(r["r2_test"])
                res[f"{t}__{pname}"] = {
                    "r2_test_mean": float(np.nanmean(r2s)),
                    "r2_test_std":  float(np.nanstd(r2s)),
                }
        for t in cls_targets:
            y = T[t]
            for probe_cls, pname in [(LinearProbe, "linear"), (MLPProbe, "mlp")]:
                accs = []
                majorities = []
                for s in range(args.n_seeds):
                    r = train_classification_probe(X, y, probe_cls, seed=s)
                    accs.append(r["acc_test"])
                    majorities.append(r["majority_rate_test"])
                res[f"{t}__{pname}"] = {
                    "acc_test_mean": float(np.mean(accs)),
                    "acc_test_std":  float(np.std(accs)),
                    "majority_rate": float(np.mean(majorities)),
                }
        return res

    print("\n  -- random dataset --")
    results_rand = run_all(X_rand, T_rand)
    for key, val in results_rand.items():
        label = list(val.keys())[0]
        val_str = " ".join(f"{k}={v:+.4f}" for k, v in val.items())
        print(f"     {key:<28} {val_str}")

    print("\n  -- 1-WL equivalent dataset --")
    results_1wl = run_all(X_1wl, T_1wl)
    for key, val in results_1wl.items():
        val_str = " ".join(f"{k}={v:+.4f}" for k, v in val.items())
        print(f"     {key:<28} {val_str}")

    all_results = {
        "config": vars(args),
        "random":          results_rand,
        "1wl_equivalent":  results_1wl,
        "target_stats": {
            "random":  {k: {"mean": float(v.mean()), "std": float(v.std())}
                        for k, v in T_rand.items()},
            "1wl":     {k: {"mean": float(v.mean()), "std": float(v.std())}
                        for k, v in T_1wl.items()},
        },
    }
    (out / "results.json").write_text(
        json.dumps(all_results, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
    )

    # --- human-readable report ---
    lines = [
        "=" * 78,
        "STRUCTURAL & PRIMAL-DUAL PROBE BATTERY — REPORT",
        "=" * 78,
        f"generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Setup: frozen pretrained OPTFM/SGFormer backbone, linear + MLP probes",
        "       trained on top to predict graph-level targets.",
        "",
        "Targets:",
        "  n_components        regression  (# connected components in bipartite A)",
        "  lp_value            regression  (LP relaxation objective value w/ c)",
        "  girth_le_4          binary      (is there a 4-cycle in the bipartite?)",
        "  feasible            binary      (is the MILP integer feasible?)",
        "",
        f"Random MILP dataset : {len(D_rand)} samples, shape ({args.rand_n_cons}, {args.rand_n_vars})",
        f"1-WL equivalent set : {len(D_1wl)} samples, k in {args.k_values}, {args.n_objectives} objectives per k",
        "",
        "RANDOM DATASET TARGET STATISTICS:",
        f"  n_components : range [{T_rand['n_components'].min():.0f}, {T_rand['n_components'].max():.0f}]  mean {T_rand['n_components'].mean():.2f}",
        f"  girth_le_4 rate: {T_rand['girth_le_4'].mean():.3f}",
        f"  feasible rate  : {T_rand['feasible'].mean():.3f}",
        f"  lp_value range : [{T_rand['lp_value'].min():.3f}, {T_rand['lp_value'].max():.3f}]",
        "",
        "1-WL DATASET TARGET STATISTICS:",
        f"  n_components : range [{T_1wl['n_components'].min():.0f}, {T_1wl['n_components'].max():.0f}]  mean {T_1wl['n_components'].mean():.2f}",
        f"  girth_le_4 rate: {T_1wl['girth_le_4'].mean():.3f}",
        f"  lp_value range : [{T_1wl['lp_value'].min():.3f}, {T_1wl['lp_value'].max():.3f}]",
        "",
        "=" * 78,
        "(a) RANDOM MILPS — positive control",
        "=" * 78,
    ]
    for key, val in results_rand.items():
        if "r2_test_mean" in val:
            lines.append(f"  {key:<30}  R²_test = {val['r2_test_mean']:+.4f} ± {val['r2_test_std']:.4f}")
        else:
            lines.append(f"  {key:<30}  acc = {val['acc_test_mean']:.4f} ± {val['acc_test_std']:.4f}  (majority rate = {val['majority_rate']:.4f})")
    lines.append("")
    lines.append("=" * 78)
    lines.append("(b) 1-WL EQUIVALENT NON-ISOMORPHIC MILPS — the target population")
    lines.append("=" * 78)
    for key, val in results_1wl.items():
        if "r2_test_mean" in val:
            lines.append(f"  {key:<30}  R²_test = {val['r2_test_mean']:+.4f} ± {val['r2_test_std']:.4f}")
        else:
            lines.append(f"  {key:<30}  acc = {val['acc_test_mean']:.4f} ± {val['acc_test_std']:.4f}  (majority rate = {val['majority_rate']:.4f})")
    lines.append("")
    lines.append("=" * 78)
    lines.append("INTERPRETATION")
    lines.append("=" * 78)
    lines.append("")
    lines.append("If the pretrained embedding implicitly encodes structural or primal-dual")
    lines.append("information, probes should fit ALL targets on the 1-WL equivalent dataset.")
    lines.append("In particular, n_components varies explicitly between MILP-A (1) and MILP-B")
    lines.append("(= k) within each pair, so a probe that achieves high R² on random MILPs")
    lines.append("should also achieve high R² here — IF the embedding is aware of graph")
    lines.append("structure beyond 1-WL.")
    lines.append("")
    lines.append("Collapsing R² / chance-level accuracy on the 1-WL dataset despite")
    lines.append("non-trivial target variance = the embedding does NOT encode the target,")
    lines.append("which confirms that OPTFM's learned representation is effectively")
    lines.append("1-WL bounded in a probe-detectable sense.")
    (out / "report.txt").write_text("\n".join(lines))
    print(f"\nresults: {out / 'results.json'}")
    print(f"report:  {out / 'report.txt'}")


if __name__ == "__main__":
    main()
