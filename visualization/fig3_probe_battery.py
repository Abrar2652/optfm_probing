"""Figure 3 — probe battery result (the load-bearing structural evidence).

Frozen-backbone linear + MLP probes are trained to predict four graph-level
targets on two populations: (i) random MILPs (positive control) and
(ii) the 1-WL equivalent non-isomorphic family. If OPTFM's pretrained
embedding implicitly encoded primal-dual / structural info beyond 1-WL,
R² should transfer. It does not: n_components MLP R² collapses from 0.396
to −0.010 despite 6x more variance on the 1-WL population.

Single panel: grouped bar chart. For each target:
  - left pair of bars : random MILPs  (linear, MLP), shown as R² or acc_over_majority
  - right pair of bars: 1-WL MILPs    (linear, MLP)

Majority-rate baseline drawn as a horizontal mark for classification targets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization.neurips_style import (
    apply_style, save_figure, PALETTE, TEXT_WIDTH,
)

PROBE_ROOT = Path(__file__).resolve().parent.parent / "results" / "probes"


def latest_run() -> Path:
    runs = sorted(PROBE_ROOT.glob("2026*"))
    if not runs:
        raise FileNotFoundError(f"no probe runs in {PROBE_ROOT}")
    return runs[-1]


# Targets to display (display-name, json-key, is_classification)
TARGETS = [
    ("n_components",           "n_components",     False, "R²"),
    ("lp_value",               "lp_value",         False, "R²"),
    ("girth ≤ 4",              "girth_le_4",       True,  "acc"),
    ("feasible",               "feasible",         True,  "acc"),
]


def get_score(run, pop, target_key, probe_kind, is_clf):
    entry = run[pop][f"{target_key}__{probe_kind}"]
    if is_clf:
        return entry["acc_test_mean"], entry["acc_test_std"], entry.get("majority_rate")
    return entry["r2_test_mean"], entry["r2_test_std"], None


def make_figure(out_dir: Path):
    apply_style()
    run_path = latest_run() / "results.json"
    with open(run_path) as f:
        run = json.load(f)

    target_stats = run["target_stats"]

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.9))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.55, 1.0], wspace=0.32,
        left=0.09, right=0.985, top=0.93, bottom=0.14,
    )
    ax_main = fig.add_subplot(gs[0, 0])
    ax_var = fig.add_subplot(gs[0, 1])

    # --- Panel A: R² / accuracy grouped bars ---
    group_x = np.arange(len(TARGETS))
    bar_w = 0.19

    # Colors
    c_rand_lin = PALETTE["blue"]
    c_rand_mlp = "#66B2D6"  # lighter blue
    c_1wl_lin  = PALETTE["red"]
    c_1wl_mlp  = "#E8877A"  # lighter red

    for i, (label, key, is_clf, metric) in enumerate(TARGETS):
        # random/linear, random/mlp, 1wl/linear, 1wl/mlp
        rm, rs, rmaj = get_score(run, "random",         key, "linear", is_clf)
        rM, rS, rmaj = get_score(run, "random",         key, "mlp",    is_clf)
        wm, ws, wmaj = get_score(run, "1wl_equivalent", key, "linear", is_clf)
        wM, wS, wmaj = get_score(run, "1wl_equivalent", key, "mlp",    is_clf)

        # For classification, show acc - majority (positive = learned something)
        if is_clf:
            rm_d, rM_d = rm - rmaj, rM - rmaj
            wm_d, wM_d = wm - wmaj, wM - wmaj
            bars = [rm_d, rM_d, wm_d, wM_d]
            errs = [rs, rS, ws, wS]
        else:
            bars = [rm, rM, wm, wM]
            errs = [rs, rS, ws, wS]

        xs = [group_x[i] - 1.5 * bar_w, group_x[i] - 0.5 * bar_w,
              group_x[i] + 0.5 * bar_w, group_x[i] + 1.5 * bar_w]
        colors = [c_rand_lin, c_rand_mlp, c_1wl_lin, c_1wl_mlp]
        edges = ["black"] * 4

        ax_main.bar(xs, bars, bar_w, yerr=errs, capsize=1.8,
                    color=colors, edgecolor=edges, linewidth=0.4,
                    error_kw={"elinewidth": 0.5, "ecolor": PALETTE["gray"]})

        # Annotate only regression MLP bars with R² value (load-bearing).
        # Classification bars already show acc - majority directly; annotating
        # zero-height degenerate-majority bars just adds visual noise.
        if not is_clf:
            for xv, yv, err in [(xs[1], rM, rS), (xs[3], wM, wS)]:
                top = max(yv + err, 0) + 0.018
                ax_main.text(xv, top, f"{yv:+.2f}", ha="center", va="bottom",
                             fontsize=6.2, color="black")

    ax_main.axhline(0, color="black", linewidth=0.5)
    ax_main.set_xticks(group_x)
    ax_main.set_xticklabels([t[0] for t in TARGETS], fontsize=8)
    ax_main.set_ylabel("probe score\n(R² for regression;  acc − majority for classification)",
                       fontsize=8)
    ax_main.set_ylim(-0.2, 0.55)
    ax_main.yaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax_main.text(-0.01, 1.02, "(a)", transform=ax_main.transAxes,
                 fontsize=9, fontweight="bold", ha="left", va="bottom")

    # Legend — boxed, transparent, top-right corner (empty region above feasible bars)
    legend_handles = [
        mpatches.Patch(facecolor=c_rand_lin, edgecolor="black",
                       linewidth=0.5, label="random · linear"),
        mpatches.Patch(facecolor=c_rand_mlp, edgecolor="black",
                       linewidth=0.5, label="random · MLP"),
        mpatches.Patch(facecolor=c_1wl_lin,  edgecolor="black",
                       linewidth=0.5, label="1-WL · linear"),
        mpatches.Patch(facecolor=c_1wl_mlp,  edgecolor="black",
                       linewidth=0.5, label="1-WL · MLP"),
    ]
    leg = ax_main.legend(
        handles=legend_handles, loc="upper right",
        fontsize=6.8, handlelength=1.0, columnspacing=0.8, ncol=2,
        frameon=True, framealpha=0.0, edgecolor="black", fancybox=False,
    )
    leg.get_frame().set_linewidth(0.5)

    # --- Panel B: variance sanity check ---
    # Show that n_components has MORE variance on 1-WL set, so the
    # probe collapse cannot be explained by target being degenerate.
    targets = ["n_components", "lp_value"]
    x = np.arange(len(targets))
    rand_std = [target_stats["random"][t]["std"] for t in targets]
    wl_std   = [target_stats["1wl"][t]["std"]    for t in targets]

    w = 0.32
    ax_var.bar(x - w/2, rand_std, w, color=c_rand_mlp, edgecolor="black",
               linewidth=0.4, label="random MILPs")
    ax_var.bar(x + w/2, wl_std,  w, color=c_1wl_mlp,  edgecolor="black",
               linewidth=0.4, label="1-WL equivalent")
    for i, (rs, ws) in enumerate(zip(rand_std, wl_std)):
        ax_var.text(i - w/2, rs + 0.12, f"{rs:.2f}", ha="center",
                    fontsize=6.5, color="black")
        ax_var.text(i + w/2, ws + 0.12, f"{ws:.2f}", ha="center",
                    fontsize=6.5, color="black")
    ax_var.set_xticks(x)
    ax_var.set_xticklabels(targets, fontsize=8)
    ax_var.set_ylabel("target std. deviation")
    ax_var.set_ylim(0, max(rand_std + wl_std) * 1.25)
    var_handles = [
        mpatches.Patch(facecolor=c_rand_mlp, edgecolor="black",
                       linewidth=0.5, label="random"),
        mpatches.Patch(facecolor=c_1wl_mlp, edgecolor="black",
                       linewidth=0.5, label="1-WL equivalent"),
    ]
    leg2 = ax_var.legend(
        handles=var_handles, loc="upper left", fontsize=7,
        frameon=True, framealpha=0.0, edgecolor="black", fancybox=False,
    )
    leg2.get_frame().set_linewidth(0.5)
    ax_var.text(-0.01, 1.02, "(b)", transform=ax_var.transAxes,
                fontsize=9, fontweight="bold", ha="left", va="bottom")

    return save_figure(fig, "fig3_probe_battery", out_dir)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "results" / "figures"
    for p in make_figure(out):
        print(p)
