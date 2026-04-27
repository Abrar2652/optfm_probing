"""Figure 2 — main empirical result.

Two panels:
  (a) log-scale heatmap of (1 - cos_sim) across 6 architectures and 8 input
      transforms, 15 pairs per cell. Darker = more distinguishable. The
      baseline + virtual-global-node columns are exactly zero (1-WL bound);
      transforms with genuine expressive power (RWPE) produce visible gaps,
      with the hierarchical OPTFM showing the largest gap.
  (b) cos_sim vs k on the cycle family for the baseline: all 6 architectures
      pin at 1.0 for every scale k ∈ {2, ..., 30}, overlaid with the
      hierarchical OPTFM RWPE curve as a falsifiability positive control.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization.neurips_style import (
    apply_style, save_figure, PALETTE, MODEL_COLORS, TEXT_WIDTH,
)


RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "main"


def latest_run() -> Path:
    runs = sorted(RESULTS_ROOT.glob("2026*"))
    if not runs:
        raise FileNotFoundError(f"no runs found in {RESULTS_ROOT}")
    return runs[-1]


MODELS = [
    "SGFormer+GCN (pretrained)",
    "SGFormer+GCN (random)",
    "TransConv only (random)",
    "GNN only (random)",
    "Simple GCN (random)",
    "Hierarchical OPTFM (random)",
]

TRANSFORMS_ORDER = [
    "baseline",
    "virtual_global_node",
    "lp_primal",
    "lp_reduced",
    "lp_primal_dual",
    "rnf_sigma_0.3",
    "rnf_sigma_1.0",
    "rwpe_steps_4_6_8",
]

TRANSFORM_LABELS = {
    "baseline":            "base",
    "virtual_global_node": "VGN",
    "lp_primal":           "LP$_{\\mathrm{prim}}$",
    "lp_reduced":          "LP$_{\\mathrm{red}}$",
    "lp_primal_dual":      "LP$_{\\mathrm{p{+}d}}$",
    "rnf_sigma_0.3":       "RNF$_{0.3}$",
    "rnf_sigma_1.0":       "RNF$_{1.0}$",
    "rwpe_steps_4_6_8":    "RWPE",
}

MODEL_LABELS = {
    "SGFormer+GCN (pretrained)":   "SGFormer+GCN (pre)",
    "SGFormer+GCN (random)":       "SGFormer+GCN (rnd)",
    "TransConv only (random)":     "TransConv only",
    "GNN only (random)":           "GNN only",
    "Simple GCN (random)":         "Simple GCN",
    "Hierarchical OPTFM (random)": "Hierarchical OPTFM",
}


def load_summary(run_dir: Path):
    """Parse summary.csv into dict[(model, transform)] = mean."""
    table = {}
    with open(run_dir / "summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table[(row["model"], row["transform"])] = {
                "mean": float(row["mean"]),
                "lo":   float(row["lo"]),
                "hi":   float(row["hi"]),
                "exact_frac": float(row["exact_frac"]),
            }
    return table


def load_results_json(run_dir: Path):
    with open(run_dir / "results.json") as f:
        return json.load(f)


def panel_a_heatmap(ax, summary):
    """Heatmap of (1 - mean cos_sim), log-scaled to expose structure."""
    data = np.zeros((len(MODELS), len(TRANSFORMS_ORDER)))
    exact = np.zeros_like(data, dtype=bool)
    for i, m in enumerate(MODELS):
        for j, t in enumerate(TRANSFORMS_ORDER):
            cell = summary[(m, t)]
            # Clamp floating-point noise near zero.
            v = max(0.0, 1.0 - cell["mean"])
            data[i, j] = v
            exact[i, j] = cell["exact_frac"] >= 0.999

    # Use a floor so zero cells render at the color-map's lightest end.
    floor = 1e-6
    display = np.where(data <= 0, floor, np.maximum(data, floor))

    norm = LogNorm(vmin=floor, vmax=1e-1)
    im = ax.imshow(display, cmap="magma_r", norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(TRANSFORMS_ORDER)))
    ax.set_xticklabels([TRANSFORM_LABELS[t] for t in TRANSFORMS_ORDER],
                       fontsize=7.5)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=7.5)

    # Annotate exact-match cells with a dot, and non-trivial cells with their value
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if v <= 1e-5:
                ax.text(j, i, "=", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            else:
                # put value in scientific: e.g. 8.1e-2
                color = "white" if v > 3e-3 else "black"
                ax.text(j, i, f"{v:.1e}", ha="center", va="center",
                        fontsize=6.2, color=color)

    # Grid between cells
    ax.set_xticks(np.arange(-0.5, len(TRANSFORMS_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    cb = plt.colorbar(im, ax=ax, pad=0.015, fraction=0.04, aspect=18)
    cb.set_label(r"$1 - \overline{\cos}(G_A, G_B)$", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.text(-0.01, 1.02, "(a)", transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="left", va="bottom")


def panel_b_scale(ax, results_json):
    """cos_sim vs k on the cycle family, baseline + RWPE overlays."""
    k_values = results_json["k_values"]

    # Baseline curves
    for mname in MODELS:
        baseline_entries = [
            r for r in results_json["results"][mname]["baseline"]
            if r.get("k", -1) >= 2
        ]
        ks = [r["k"] for r in baseline_entries]
        vs = [r["mean_cos"] for r in baseline_entries]
        ax.plot(ks, vs, "o-", color=MODEL_COLORS[mname], linewidth=1.0,
                markersize=3, label=mname, alpha=0.9)

    # RWPE (4,6,8) for hierarchical OPTFM — the strongest escape signal
    rwpe_entries = [
        r for r in results_json["results"]["Hierarchical OPTFM (random)"]["rwpe_steps_4_6_8"]
        if r.get("k", -1) >= 2
    ]
    ks = [r["k"] for r in rwpe_entries]
    vs = [r["mean_cos"] for r in rwpe_entries]
    ax.plot(ks, vs, "s--", color=MODEL_COLORS["Hierarchical OPTFM (random)"],
            linewidth=1.0, markersize=3.5, alpha=0.95,
            markerfacecolor="white",
            label="Hier. OPTFM + RWPE (falsifiability)")

    # Reference line
    ax.axhline(1.0, color=PALETTE["gray"], linestyle=":", linewidth=0.6,
               alpha=0.8)
    ax.text(ks[-1] * 0.98, 1.004, "1-WL bound",
            fontsize=6.8, color=PALETTE["gray"], ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel(r"scale $k$ (cycle family $C_{4k}$ vs $k{\cdot}C_4$)")
    ax.set_ylabel("mean cosine similarity")
    ax.set_xticks([2, 3, 5, 10, 20, 30])
    ax.set_xticklabels([2, 3, 5, 10, 20, 30])
    ax.set_ylim(0.88, 1.012)
    leg = ax.legend(
        loc="lower left", fontsize=6.2, ncol=1,
        handlelength=1.6, borderaxespad=0.3,
        frameon=True, framealpha=0.0, edgecolor="black",
        fancybox=False,
    )
    leg.get_frame().set_linewidth(0.5)
    ax.text(-0.01, 1.02, "(b)", transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="left", va="bottom")


def make_figure(out_dir: Path):
    apply_style()
    run_dir = latest_run()
    summary = load_summary(run_dir)
    results_json = load_results_json(run_dir)

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.8))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.45, 1.0], wspace=0.38,
        left=0.14, right=0.985, top=0.94, bottom=0.16,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    panel_a_heatmap(ax_a, summary)
    panel_b_scale(ax_b, results_json)

    return save_figure(fig, "fig2_main_result", out_dir)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "results" / "figures"
    for p in make_figure(out):
        print(p)
