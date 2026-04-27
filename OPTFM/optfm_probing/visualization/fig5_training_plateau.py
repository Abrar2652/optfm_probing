"""Figure 5 — training cannot overcome the 1-WL bound.

Loads results from the latest results/finetune/<timestamp>/results.json
produced by scripts/finetune_pair_classifier.py and plots per-epoch
cross-entropy loss for:

  * baseline inputs  — theorem predicts loss = ln(2) forever, regardless
    of training effort, because the pooled embeddings of G_A and G_B
    are bit-identical so head logits are bit-identical.
  * RWPE(4,6,8) inputs — theorem places no such bound; the classifier
    can converge.

The y-axis is log-scaled below ln(2) to expose the multi-decade drop of
the RWPE curve while keeping the baseline plateau on a readable scale.
A dashed horizontal line at ln(2) marks the theoretical floor.
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
    apply_style, save_figure, PALETTE, TEXT_WIDTH, COL_WIDTH,
)

LN2 = float(np.log(2.0))
FT_ROOT = Path(__file__).resolve().parent.parent / "results" / "finetune"


def latest_run() -> Path:
    runs = sorted(FT_ROOT.glob("2026*"))
    if not runs:
        raise FileNotFoundError(f"no fine-tune runs in {FT_ROOT}")
    return runs[-1]


def make_figure(out_dir: Path):
    apply_style()
    run_dir = latest_run()
    with open(run_dir / "results.json") as f:
        data = json.load(f)

    runs = data["runs"]
    baseline_runs = [r for r in runs if r["transform"] == "baseline"]
    rwpe_runs     = [r for r in runs if r["transform"] == "rwpe_steps_4_6_8"]

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.9))
    gs = fig.add_gridspec(
        1, 1, left=0.10, right=0.985, top=0.94, bottom=0.17,
    )
    ax = fig.add_subplot(gs[0, 0])

    # Plot baseline curves — all pinned at ln(2)
    for r in baseline_runs:
        epochs = np.arange(len(r["losses"]))
        ax.plot(epochs, r["losses"], color=PALETTE["blue"],
                linewidth=0.9, alpha=0.85,
                label="baseline (Theorem 1 $\\Rightarrow$ plateau at $\\ln 2$)"
                      if r is baseline_runs[0] else None,
                zorder=4)

    # Plot RWPE curves — drop multiple decades
    for r in rwpe_runs:
        epochs = np.arange(len(r["losses"]))
        ax.plot(epochs, r["losses"], "--", color=PALETTE["red"],
                linewidth=0.9, alpha=0.85,
                label="RWPE (4,6,8) (no bound $\\Rightarrow$ converges)"
                      if r is rwpe_runs[0] else None,
                zorder=3)

    # Theoretical-floor reference line at ln(2)
    ax.axhline(LN2, color=PALETTE["gray"], linestyle=":", linewidth=0.7,
               zorder=2)
    ax.text(
        len(baseline_runs[0]["losses"]) - 1, LN2 * 1.08,
        r"$\ln 2 \approx 0.693$", fontsize=7.2, color=PALETTE["gray"],
        ha="right", va="bottom",
    )

    # Log-scaled y-axis to span the dynamic range
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.0)
    ax.set_xlim(0, len(baseline_runs[0]["losses"]) - 1)
    ax.set_xlabel("training epoch (Adam, lr $=10^{-3}$, full-batch)")
    ax.set_ylabel("cross-entropy loss")

    # Annotate final plateau value with 6-decimal precision
    final_base = np.mean([r["losses"][-1] for r in baseline_runs])
    ax.annotate(
        f"final loss = {final_base:.6f} $= \\ln 2$",
        xy=(len(baseline_runs[0]["losses"]) - 1, final_base),
        xytext=(-10, 20), textcoords="offset points",
        ha="right", va="bottom", fontsize=7.5, color=PALETTE["blue"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["blue"],
                        linewidth=0.5, shrinkA=0, shrinkB=2),
    )
    final_rwpe = np.mean([r["losses"][-1] for r in rwpe_runs])
    ax.annotate(
        f"final loss $\\approx {final_rwpe:.0e}$",
        xy=(len(rwpe_runs[0]["losses"]) - 1, final_rwpe),
        xytext=(-10, 25), textcoords="offset points",
        ha="right", va="bottom", fontsize=7.5, color=PALETTE["red"],
        arrowprops=dict(arrowstyle="->", color=PALETTE["red"],
                        linewidth=0.5, shrinkA=0, shrinkB=2),
    )

    # Legend in upper-right (empty corner — all curves drop or stay flat)
    leg = ax.legend(
        loc="lower left", fontsize=7, handlelength=2.0,
        frameon=True, framealpha=0.0, edgecolor="black", fancybox=False,
    )
    leg.get_frame().set_linewidth(0.5)

    # Sub-note beneath plot
    ax.text(
        0.99, 0.02,
        f"{len(baseline_runs)} random seeds per curve, 15 1-WL-equivalent pairs, 30 training examples per epoch",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.3, color=PALETTE["gray"],
    )

    return save_figure(fig, "fig5_training_plateau", out_dir)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "results" / "figures"
    for p in make_figure(out):
        print(p)
