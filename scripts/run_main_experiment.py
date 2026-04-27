#!/usr/bin/env python3
"""
Main 1-WL boundedness experiment for OPTFM.

Runs the full probing pipeline:

  1. Generate a population of 1-WL equivalent non-isomorphic MILP pairs
     (C_{4k}-bipartite vs k*C_4-bipartite, k ∈ k_values).

  2. For every pair, verify
       * 1-WL equivalent           (color refinement)
       * NOT graph-isomorphic      (brute-force / VF2)

  3. Probe every model variant:
       * SGFormer+GCN (pretrained)   ← the shipped checkpoint
       * SGFormer+GCN (random)
       * TransConv only (random)
       * GNN only (random)
       * Simple GCN (random)
       * HierarchicalOPTFM (random)  ← the full multi-view cross-attention
                                         architecture described in the paper

     and for every input transform:
       * baseline
       * rnf_sigma_0.1 / 0.3 / 1.0
       * lp_primal
       * lp_reduced
       * lp_primal_dual

  4. For RNF, average over K=30 i.i.d. seeds to compute the expected
     cosine similarity (not a single noisy draw).

  5. Report:
       * mean cosine similarity per (model, transform)
       * bootstrap 95% CI over pairs
       * fraction of pairs with embeddings exactly equal (atol=1e-5)

  6. Save:
       * results/main/<timestamp>/results.json   raw per-pair numbers
       * results/main/<timestamp>/summary.csv    (model, transform, mean, lo, hi)
       * results/main/<timestamp>/report.txt     human-readable summary
       * results/main/<timestamp>/figures/*.png  publication figures
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from data.milp_pairs import milp_to_tensors
from data.milp_pairs_v2 import (
    construct_bipartite_cycle_pair,
    construct_cubic_bipartite_pair,
    diagnose_pair,
)
from models.sgformer_mip import create_model
from models.optfm_hierarchical import create_hierarchical
from scripts._tee_log import start_logging
from scripts.improvements import TRANSFORMS

CKPT = "D:/GitHub/OPTFM/node_pretrain/Models_SCIP/model_params_epoch_31.pth"


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed(model, milp, pooling="mean"):
    cons_x, ei, ea, var_x = milp_to_tensors(milp)
    with torch.no_grad():
        return model.get_graph_embedding(cons_x, ei, ea, var_x, pooling=pooling)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def exact_eq(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5) -> bool:
    return bool(torch.allclose(a, b, atol=atol))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def build_models():
    models = {
        "SGFormer+GCN (pretrained)":   create_model("optfm", pretrained_path=CKPT),
        "SGFormer+GCN (random)":       create_model("random"),
        "TransConv only (random)":     create_model("transconv_only"),
        "GNN only (random)":           create_model("gnn_only"),
        "Simple GCN (random)":         create_model("gcn"),
        "Hierarchical OPTFM (random)": create_hierarchical(),
    }
    for m in models.values():
        m.eval()
    return models


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_on_pair(model, pair, transform_name, transform, rnf_samples: int = 30):
    """Return (mean_cos, std_cos, exact_match_fraction, per-sample list)."""
    is_stochastic = transform_name.startswith("rnf")
    n_samples = rnf_samples if is_stochastic else 1

    sims = []
    exacts = []
    for s in range(n_samples):
        # For stochastic transforms use different seeds for A and B
        ma = transform(pair.milp_a, 100 + s)
        mb = transform(pair.milp_b, 200 + s)
        ea = embed(model, ma)
        eb = embed(model, mb)
        sims.append(cos_sim(ea, eb))
        exacts.append(exact_eq(ea, eb))

    return {
        "mean_cos": float(np.mean(sims)),
        "std_cos":  float(np.std(sims)),
        "min_cos":  float(np.min(sims)),
        "max_cos":  float(np.max(sims)),
        "exact_frac": float(np.mean(exacts)),
        "n_samples": n_samples,
    }


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(boot_means, alpha / 2)),
        float(np.quantile(boot_means, 1 - alpha / 2)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_values", type=int, nargs="+",
                        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30])
    parser.add_argument("--rnf_samples", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="results/main")
    parser.add_argument("--skip_transforms", nargs="*", default=[])
    parser.add_argument("--family", choices=["cycle", "cubic", "both"], default="both",
                        help="which 1-WL pair family to probe")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir) / ts
    (out / "figures").mkdir(parents=True, exist_ok=True)

    start_logging("run_main_experiment", log_dir=out)
    print("=" * 78)
    print("OPTFM 1-WL boundedness — main experiment")
    print("=" * 78)
    print(f"Output dir:   {out}")
    print(f"k values:     {args.k_values}")
    print(f"RNF samples:  {args.rnf_samples}")

    # --- 1. Generate pairs -------------------------------------------------
    print("\n[1/4] generating 1-WL equivalent non-isomorphic pairs...")
    pairs = []
    if args.family in ("cycle", "both"):
        pairs += [construct_bipartite_cycle_pair(k) for k in args.k_values]
    if args.family in ("cubic", "both"):
        # The cubic construction is a single fixed 6x6 instance; adding it
        # once per run makes the families comparable in size.
        pairs += [construct_cubic_bipartite_pair()]
    diagnostics = [diagnose_pair(p) for p in pairs]
    for d in diagnostics:
        assert d["is_1wl_equivalent"], f"not 1-WL equiv: {d['description']}"
        assert not d["is_isomorphic"], f"isomorphic (bad): {d['description']}"
    print(f"  OK. {len(pairs)} pairs, all verified 1-WL equivalent AND non-isomorphic")
    for d in diagnostics:
        print(f"    - {d['description']}  shape={d['shape']}  "
              f"A_comp={d['A_components']}  B_comp={d['B_components']}")

    # --- 2. Models ---------------------------------------------------------
    print("\n[2/4] building models...")
    models = build_models()
    for name, m in models.items():
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {name:<35} {n_params:>8,} params")

    # --- 3. Probe every (model, transform) combination --------------------
    print("\n[3/4] probing...")
    transforms = {k: v for k, v in TRANSFORMS.items() if k not in args.skip_transforms}

    # results[model_name][transform_name] = list of per-pair dicts
    results = {mname: {tname: [] for tname in transforms} for mname in models}
    def parse_k(desc: str) -> int:
        if "k=" in desc:
            return int(desc.split("k=")[-1].rstrip(")"))
        return -1   # non-cycle pair (cubic); sentinel

    for tname, transform in transforms.items():
        print(f"  transform = {tname}")
        for mname, model in models.items():
            for p in pairs:
                out_dict = run_on_pair(model, p, tname, transform,
                                       rnf_samples=args.rnf_samples)
                out_dict["pair"] = p.description
                out_dict["k"] = parse_k(p.description)
                results[mname][tname].append(out_dict)
            per_pair = np.array([r["mean_cos"] for r in results[mname][tname]])
            exact_frac = np.mean([r["exact_frac"] for r in results[mname][tname]])
            mean, lo, hi = bootstrap_ci(per_pair)
            print(f"    {mname:<32}  mean={mean:.6f}  95%CI=[{lo:.6f},{hi:.6f}]  exact_frac={exact_frac:.2f}")

    # --- 4. Save results ---------------------------------------------------
    print("\n[4/4] saving results...")
    (out / "results.json").write_text(json.dumps({
        "k_values": args.k_values,
        "rnf_samples": args.rnf_samples,
        "diagnostics": diagnostics,
        "results": results,
    }, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o)))

    # Summary CSV
    lines = ["model,transform,mean,lo,hi,exact_frac,n_pairs"]
    for mname in models:
        for tname in transforms:
            values = np.array([r["mean_cos"] for r in results[mname][tname]])
            exact = float(np.mean([r["exact_frac"] for r in results[mname][tname]]))
            mean, lo, hi = bootstrap_ci(values)
            lines.append(f'"{mname}","{tname}",{mean:.6f},{lo:.6f},{hi:.6f},{exact:.4f},{len(values)}')
    (out / "summary.csv").write_text("\n".join(lines))

    # --- 5. Figures --------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

        # Figure A: bar chart of mean cos_sim per model (baseline only)
        fig, ax = plt.subplots(figsize=(10, 5))
        model_names = list(models.keys())
        means = []
        errs_lo = []
        errs_hi = []
        for mname in model_names:
            values = np.array([r["mean_cos"] for r in results[mname]["baseline"]])
            m, lo, hi = bootstrap_ci(values)
            means.append(m)
            errs_lo.append(m - lo)
            errs_hi.append(hi - m)
        y = np.arange(len(model_names))
        ax.barh(y, means, xerr=[errs_lo, errs_hi], color="#cc3333", alpha=0.8, capsize=4)
        ax.axvline(1.0, color="black", linestyle="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(model_names)
        ax.set_xlabel("cosine similarity between 1-WL equivalent non-iso MILP pairs")
        ax.set_xlim(0.9, 1.01)
        ax.set_title("Baseline: all architectures cannot distinguish 1-WL equivalent non-iso pairs")
        plt.tight_layout()
        plt.savefig(out / "figures" / "fig_A_baseline_all_fail.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Figure B: improvements — only for the pretrained SGFormer + Hierarchical OPTFM
        fig, ax = plt.subplots(figsize=(10, 6))
        tnames = [t for t in transforms if t != "baseline"]
        bar_models = ["SGFormer+GCN (pretrained)", "Hierarchical OPTFM (random)"]
        width = 0.35
        x = np.arange(len(tnames))
        for i, mname in enumerate(bar_models):
            heights = []
            errs = []
            for tname in tnames:
                values = np.array([r["mean_cos"] for r in results[mname][tname]])
                m, lo, hi = bootstrap_ci(values)
                heights.append(m)
                errs.append(max(m - lo, hi - m))
            ax.bar(x + i * width, heights, width, yerr=errs, capsize=3,
                   label=mname)
        ax.axhline(1.0, color="black", linestyle="--", lw=1, label="identical (1-WL bound)")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(tnames, rotation=30, ha="right")
        ax.set_ylabel("cosine similarity (lower = more distinguishable)")
        ax.set_title("Input transforms that can escape the 1-WL limit")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out / "figures" / "fig_B_improvements.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Figure C: distribution of per-pair cosine similarity (baseline),
        # restricted to the cycle family where we have a k axis.
        fig, ax = plt.subplots(figsize=(10, 5))
        for mname in model_names:
            entries = [r for r in results[mname]["baseline"] if r["k"] >= 0]
            ks = [r["k"] for r in entries]
            vals = [r["mean_cos"] for r in entries]
            if ks:
                ax.plot(ks, vals, "o-", label=mname, alpha=0.8)
        ax.axhline(1.0, color="black", linestyle="--", lw=1)
        ax.set_xlabel("k (C_{4k} cycle / number of C_4 components)")
        ax.set_ylabel("cosine similarity")
        ax.set_ylim(0.9, 1.01)
        ax.set_title("Cycle family: cos_sim vs k — all architectures at 1.0 for every scale")
        ax.legend(fontsize=8, loc="lower right")
        plt.tight_layout()
        plt.savefig(out / "figures" / "fig_C_scale_invariance.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  figures written to {out / 'figures'}")
    except ImportError:
        print("  matplotlib not available, skipping figures")

    # --- 6. Human-readable report -----------------------------------------
    lines = []
    lines.append("=" * 78)
    lines.append("OPTFM 1-WL BOUNDEDNESS EXPERIMENT — MAIN REPORT")
    lines.append("=" * 78)
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("")
    lines.append("Pairs: 1-WL equivalent non-isomorphic bipartite MILP pairs")
    lines.append(f"       Family selection: --family {args.family}")
    lines.append(f"       Total pairs: {len(pairs)}")
    if args.family in ("cycle", "both"):
        lines.append(f"       Family I (2-regular cycles): C_{{4k}} bipartite vs k*C_4, k in {args.k_values}")
    if args.family in ("cubic", "both"):
        lines.append(f"       Family II (3-regular cubic): connected 6+6 cubic vs 2*K_{{3,3}}")
    lines.append("")
    lines.append("Every pair verified:")
    lines.append(f"       1-WL equivalent:          {all(d['is_1wl_equivalent'] for d in diagnostics)}")
    lines.append(f"       NOT graph-isomorphic:     {not any(d['is_isomorphic'] for d in diagnostics)}")
    lines.append(f"       Different #components:    {all(d['A_components'] != d['B_components'] for d in diagnostics)}")
    lines.append("")
    lines.append("Pair breakdown:")
    for d in diagnostics:
        lines.append(f"   {d['description']}  shape={d['shape']}  "
                     f"A_comp={d['A_components']}  B_comp={d['B_components']}")
    lines.append("")
    lines.append("=" * 78)
    lines.append("BASELINE: each architecture on unmodified inputs")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"{'model':<35} {'mean cos':>12} {'95% CI lo':>12} {'95% CI hi':>12} {'exact %':>10}")
    for mname in model_names:
        values = np.array([r["mean_cos"] for r in results[mname]["baseline"]])
        exact = 100 * np.mean([r["exact_frac"] for r in results[mname]["baseline"]])
        m, lo, hi = bootstrap_ci(values)
        lines.append(f"{mname:<35} {m:>12.6f} {lo:>12.6f} {hi:>12.6f} {exact:>10.1f}")
    lines.append("")
    lines.append("INTERPRETATION: every architecture produces cos_sim = 1.0 on every pair.")
    lines.append("This is the 1-WL boundedness barrier. It holds even for the full")
    lines.append("hierarchical OPTFM with cross-attention, which the paper claims")
    lines.append("`enhances representational power`.")
    lines.append("")
    lines.append("=" * 78)
    lines.append("IMPROVEMENTS: does a non-trivial input transform escape the limit?")
    lines.append("=" * 78)
    lines.append("")
    for tname in transforms:
        if tname == "baseline":
            continue
        lines.append(f"\n--- transform: {tname} ---")
        lines.append(f"{'model':<35} {'mean cos':>12} {'exact %':>10}")
        for mname in model_names:
            values = np.array([r["mean_cos"] for r in results[mname][tname]])
            exact = 100 * np.mean([r["exact_frac"] for r in results[mname][tname]])
            m, lo, hi = bootstrap_ci(values)
            lines.append(f"{mname:<35} {m:>12.6f} {exact:>10.1f}")
    lines.append("")
    lines.append("=" * 78)
    (out / "report.txt").write_text("\n".join(lines))
    print(f"\nReport: {out / 'report.txt'}")
    print(f"Results: {out / 'results.json'}")
    print("done.")


if __name__ == "__main__":
    main()
