#!/usr/bin/env python3
"""
B3 — Extended probe battery + capacity control.

Extends scripts/probe_primal_dual.py with:
  * more structural targets that 1-WL cannot compute: number of 4-cycles and the
    spectral gap (algebraic connectivity), in addition to n_components / girth /
    lp_value / feasible;
  * per-target VARIANCE reporting (to substantiate "larger target variance yet R^2
    collapses");
  * a PROBE-CAPACITY CONTROL: the SAME probe, trained on embeddings of the SAME
    instances but with an RWPE-augmented (separating) encoding, recovers the target
    — proving the collapse is a property of the embedding, not probe weakness.

Multi-seed (5 seeds); frozen pretrained backbone.

Output: results/probe_battery_extended.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from models.sgformer_mip import create_model
from data.milp_pairs_v2 import _bipartite_components
from scripts._common import CKPT, write_csv
from scripts.improvements import make_rwpe_transform
from scripts.probe_primal_dual import (
    random_bipartite_milp, build_1wl_dataset_with_varying_c, bipartite_girth_le_k,
    LinearProbe, MLPProbe, train_regression_probe, train_classification_probe,
    build_embedding_matrix,
)

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
N_RANDOM = 300
N_OBJ = 8
SEEDS = [0, 1, 2, 3, 4]


def num_4cycles(A):
    M = A @ A.T                       # cons x cons common-neighbour counts
    n = A.shape[0]
    tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = int(M[i, j])
            tot += c * (c - 1) // 2
    return float(tot)


def spectral_gap(A):
    nc, nv = A.shape
    N = nc + nv
    F = np.zeros((N, N))
    F[:nc, nc:] = (A != 0); F[nc:, :nc] = F[:nc, nc:].T
    L = np.diag(F.sum(1)) - F
    ev = np.sort(np.linalg.eigvalsh(L))
    return float(ev[1])               # algebraic connectivity (0 if disconnected)


def targets(insts):
    from data.milp_pairs_v2 import solve_lp_relaxation
    n = len(insts)
    out = {"n_components": np.zeros(n), "num_4cycles": np.zeros(n),
           "spectral_gap": np.zeros(n), "girth_le_4": np.zeros(n),
           "lp_value": np.zeros(n), "feasible": np.zeros(n)}
    for i, m in enumerate(insts):
        out["n_components"][i] = _bipartite_components(m.A)
        out["num_4cycles"][i] = num_4cycles(m.A)
        out["spectral_gap"][i] = spectral_gap(m.A)
        out["girth_le_4"][i] = bipartite_girth_le_k(m.A, 4)
        sol = solve_lp_relaxation(m, objective=m.c)
        out["feasible"][i] = int(sol["feasible"])
        out["lp_value"][i] = float(sol["obj_value"]) if sol["feasible"] else 0.0
    return out


def probe_target(X, y, kind):
    """Mean test score over seeds for linear+mlp probes."""
    res = {}
    for cls, pname in ((LinearProbe, "linear"), (MLPProbe, "mlp")):
        scores = []
        for s in SEEDS:
            if kind == "reg":
                scores.append(train_regression_probe(X, y, cls, epochs=400, seed=s)["r2_test"])
            else:
                scores.append(train_classification_probe(X, y, cls, epochs=400, seed=s)["acc_test"])
        scores = np.array(scores, dtype=float)
        res[pname] = (float(np.nanmean(scores)), float(np.nanstd(scores)))
    return res


def main():
    print("Building datasets...")
    D_rand = [random_bipartite_milp(10, 10, s) for s in range(N_RANDOM)]
    D_1wl = build_1wl_dataset_with_varying_c(K_VALUES, N_OBJ)
    T_rand, T_1wl = targets(D_rand), targets(D_1wl)

    model = create_model("optfm", pretrained_path=CKPT); model.eval()
    print("Extracting embeddings (baseline)...")
    X_rand = build_embedding_matrix(model, D_rand)
    X_1wl = build_embedding_matrix(model, D_1wl)

    print("Extracting embeddings (RWPE capacity control on 1-WL set)...")
    rwpe = make_rwpe_transform(steps=(4, 6, 8), cons_step=4, cons_scale=1.0)
    D_1wl_rwpe = [rwpe(m, 0) for m in D_1wl]
    X_1wl_rwpe = build_embedding_matrix(model, D_1wl_rwpe)

    reg = ["n_components", "num_4cycles", "spectral_gap", "lp_value"]
    cls = ["girth_le_4", "feasible"]

    rows = []
    for t in reg + cls:
        kind = "reg" if t in reg else "cls"
        var_rand = float(np.var(T_rand[t])); var_1wl = float(np.var(T_1wl[t]))
        r_rand = probe_target(X_rand, T_rand[t], kind)
        r_1wl = probe_target(X_1wl, T_1wl[t], kind)
        r_cap = probe_target(X_1wl_rwpe, T_1wl[t], kind) if kind == "reg" else \
            probe_target(X_1wl_rwpe, T_1wl[t], kind)
        for probe in ("linear", "mlp"):
            rows.append([t, kind, probe, var_rand, var_1wl,
                         r_rand[probe][0], r_1wl[probe][0], r_cap[probe][0]])
        print(f"{t:14s} var(rand)={var_rand:8.3f} var(1wl)={var_1wl:8.3f} | "
              f"mlp: rand={r_rand['mlp'][0]:+.3f} 1wl={r_1wl['mlp'][0]:+.3f} "
              f"1wl+RWPE(capacity)={r_cap['mlp'][0]:+.3f}")

    write_csv(ROOT / "results" / "probe_battery_extended.csv",
              ["target", "kind", "probe", "var_random", "var_1wl",
               "score_random", "score_1wl", "score_1wl_rwpe_capacity"], rows)
    print("\nWrote results/probe_battery_extended.csv")


if __name__ == "__main__":
    main()
