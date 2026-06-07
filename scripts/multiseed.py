#!/usr/bin/env python3
"""
B1 — Multi-seed stability of the headline numbers.

Single-seed numbers read as cherry-picked. We re-run the two headline results
that involve randomness across >=5 seeds and report mean +/- std and bootstrap CIs:

  (1) RWPE break: mean cos_sim of each architecture under RWPE(4,6,8), over the
      15-pair set, for 5 random initializations. (Baseline is bit-identical for
      every seed by construction; we record exact_frac to confirm.)
  (2) ln 2 plateau: jointly train the hierarchical OPTFM + 2-class head on the
      15-pair population for 5 seeds; baseline final cross-entropy must sit at
      ln 2 ~= 0.6931 within a tight CI, while RWPE-augmented training drops well
      below it.

(The probe-battery R^2 collapse is re-run across 5 seeds in
scripts/probe_battery_extended.py, which reports its own multi-seed mean +/- std.)

Outputs: results/multiseed/rwpe_break.csv, results/multiseed/finetune_plateau.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs_v2 import construct_bipartite_cycle_pair, construct_cubic_bipartite_pair
from scripts._common import build_models, embed, cos_sim, linf, bootstrap_ci, write_csv
from scripts.improvements import make_rwpe_transform
from scripts.finetune_pair_classifier import train_one, LN2

K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
SEEDS = [0, 1, 2, 3, 4]


def rwpe_break_multiseed():
    print("=== (1) RWPE break across seeds ===")
    pairs = [construct_bipartite_cycle_pair(k) for k in K_VALUES]
    pairs.append(construct_cubic_bipartite_pair())
    rwpe = make_rwpe_transform(steps=(4, 6, 8), cons_step=4, cons_scale=1.0)
    # per model: list over seeds of (mean rwpe cos over pairs, baseline exact_frac)
    per_model = {}
    for seed in SEEDS:
        models = build_models(seed=seed, include_pretrained=(seed == 0))
        for mname, m in models.items():
            base_exact, rcos = [], []
            for p in pairs:
                a, b = embed(m, p.milp_a), embed(m, p.milp_b)
                base_exact.append(linf(a, b) <= 1e-5)
                ra, rb = embed(m, rwpe(p.milp_a, 100)), embed(m, rwpe(p.milp_b, 200))
                rcos.append(cos_sim(ra, rb))
            per_model.setdefault(mname, {"rwpe": [], "base_exact": []})
            per_model[mname]["rwpe"].append(float(np.mean(rcos)))
            per_model[mname]["base_exact"].append(float(np.mean(base_exact)))
    rows = []
    for mname, d in per_model.items():
        arr = np.array(d["rwpe"])
        mean, lo, hi = bootstrap_ci(arr)
        rows.append([mname, len(arr), float(arr.mean()), float(arr.std()),
                     float(lo), float(hi), float(np.mean(d["base_exact"]))])
        print(f"  {mname:32s} RWPE cos={arr.mean():.6f}+/-{arr.std():.6f} "
              f"CI=[{lo:.6f},{hi:.6f}]  baseline exact_frac={np.mean(d['base_exact']):.2f}")
    write_csv(ROOT / "results" / "multiseed" / "rwpe_break.csv",
              ["model", "n_seeds", "mean_rwpe_cos", "std", "ci_lo", "ci_hi",
               "baseline_exact_frac"], rows)


def finetune_multiseed(epochs=200):
    print("\n=== (2) ln 2 plateau across seeds ===")
    rows = []
    for transform in ("baseline", "rwpe_steps_4_6_8"):
        finals, identities = [], []
        for seed in SEEDS:
            r = train_one(transform, n_epochs=epochs, lr=1e-3, hidden=16, seed=seed)
            finals.append(r["losses"][-1])
            identities.append(float(np.mean([i["exact"] for i in r["identity"]])))
            print(f"  {transform:20s} seed={seed} final_loss={r['losses'][-1]:.6f} "
                  f"frac_identical={identities[-1]:.2f}")
        arr = np.array(finals)
        mean, lo, hi = bootstrap_ci(arr)
        rows.append([transform, len(arr), float(arr.mean()), float(arr.std()),
                     float(lo), float(hi), LN2, float(np.mean(identities))])
        print(f"  -> {transform:20s} final={arr.mean():.6f}+/-{arr.std():.6f} "
              f"(ln2={LN2:.6f})")
    write_csv(ROOT / "results" / "multiseed" / "finetune_plateau.csv",
              ["transform", "n_seeds", "mean_final_loss", "std", "ci_lo", "ci_hi",
               "ln2", "mean_frac_identical_after"], rows)


def main():
    rwpe_break_multiseed()
    finetune_multiseed()
    print("\nWrote results/multiseed/rwpe_break.csv, results/multiseed/finetune_plateau.csv")


if __name__ == "__main__":
    main()
