"""ICDM figures 6-12, generated from the results/*.csv produced by the new
experiment scripts. Run: python -m visualization.figs_icdm
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from visualization import ieee_style as S

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "figures"
S.apply_style()


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _legend(ax, **kw):
    """Framed, slightly-opaque, black-edged legend placed in empty space."""
    kw.setdefault("loc", "best")
    leg = ax.legend(**kw)
    frame = leg.get_frame()
    frame.set_edgecolor("black")
    frame.set_facecolor("white")
    frame.set_alpha(0.85)
    frame.set_linewidth(0.6)
    return leg


# --- Fig 6: WL-hierarchy placement schematic -------------------------------
def fig6():
    fig, ax = plt.subplots(figsize=(S.TEXT_WIDTH * 0.6, 1.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.5)
    yline = 0.55
    # axis as a left-to-right arrow (drawn, not an axhline, so nothing overlaps it)
    ax.annotate("", xy=(9.9, yline), xytext=(0.2, yline),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
    pts = [(1.2, "constant\n(below 1-WL)", S.PALETTE["gray"]),
           (3.6, "OPTFM &\nall encoders here\n(≤1-WL)", S.PALETTE["red"]),
           (6.2, "RWPE / RNI / LapPE\n(>1-WL, <2-FWL)", S.PALETTE["green"]),
           (8.6, "2-FWL / 3-WL\n(higher order)", S.PALETTE["blue"])]
    for x, label, c in pts:
        ax.plot(x, yline, "o", color=c, ms=8, zorder=3)
        ax.annotate(label, (x, yline), (x, yline + 0.42), ha="center", va="bottom",
                    fontsize=6.6, color=c, arrowprops=dict(arrowstyle="-", color=c, lw=0.6))
    # axis caption centered BELOW the line (no longer overlapping any marker)
    ax.text(5.0, 0.12, r"increasing expressive power $\rightarrow$",
            ha="center", va="center", fontsize=7, color=S.PALETTE["gray"])
    ax.axis("off")
    ax.set_title("Where the encoders sit in the WL hierarchy")
    return S.save_figure(fig, "fig6_wl_hierarchy", OUT)


# --- Fig 7: capacity invariance --------------------------------------------
def fig7():
    rows = _read(RES / "capacity_sweep.csv")
    params = np.array([float(r["params"]) for r in rows])
    base = np.array([float(r["baseline_mean_cos"]) for r in rows])
    rwpe = np.array([float(r["rwpe_mean_cos"]) for r in rows])
    order = np.argsort(params)
    fig, ax = plt.subplots(figsize=(S.COL_WIDTH, 2.2))
    ax.plot(params[order], base[order], "o-", color=S.PALETTE["red"],
            markeredgecolor="black", markeredgewidth=0.5, label="baseline (1-WL bound)")
    ax.plot(params[order], rwpe[order], "s--", color=S.PALETTE["green"],
            markeredgecolor="black", markeredgewidth=0.5, label="RWPE (positive control)")
    ax.axhline(1.0, color="black", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("parameters"); ax.set_ylabel("mean cos(G$_A$,G$_B$)")
    ax.set_ylim(0.97, 1.005)
    ax.set_title("Capacity invariance of the 1-WL bound")
    _legend(ax)
    return S.save_figure(fig, "fig7_capacity_invariance", OUT)


# --- Fig 8: identity audit histogram ---------------------------------------
def fig8():
    data = np.load(RES / "identity_audit_diffs.npz")
    f32 = np.concatenate([data[k] for k in data.files if k.endswith("float32")])
    f64 = np.concatenate([data[k] for k in data.files if k.endswith("float64")])
    f32 = f32[f32 > 0]; f64 = f64[f64 > 0]
    fig, ax = plt.subplots(figsize=(S.COL_WIDTH, 2.2))
    bins = np.logspace(-18, -4, 40)
    ax.hist(f32, bins=bins, color=S.PALETTE["orange"], alpha=0.7,
            edgecolor="black", linewidth=0.4, label="float32")
    ax.hist(f64, bins=bins, color=S.PALETTE["blue"], alpha=0.7,
            edgecolor="black", linewidth=0.4, label="float64")
    ax.axvline(1e-5, color=S.PALETTE["red"], lw=1.0, ls="--", label="bit-identity threshold")
    ax.set_xscale("log")
    ax.set_xlabel(r"per-coordinate $|\Phi(G_A)-\Phi(G_B)|$")
    ax.set_ylabel("count")
    ax.set_title("Embedding differences sit at machine epsilon")
    _legend(ax)
    return S.save_figure(fig, "fig8_identity_audit", OUT)


# --- Fig 9: encoding comparison --------------------------------------------
def fig9():
    rows = _read(RES / "encoding_comparison.csv")
    encs = []
    for r in rows:
        if r["encoding"] not in encs:
            encs.append(r["encoding"])
    # mean over models of (1 - mean_cos), i.e. distance from identity
    gap = {e: np.mean([1 - float(r["mean_cos"]) for r in rows if r["encoding"] == e]) for e in encs}
    fig, ax = plt.subplots(figsize=(S.COL_WIDTH, 2.2))
    colors = [S.PALETTE["gray"] if e == "baseline" else S.PALETTE["green"] for e in encs]
    ax.bar(range(len(encs)), [gap[e] for e in encs], color=colors,
           edgecolor="black", linewidth=0.5)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_xticks(range(len(encs))); ax.set_xticklabels(encs, rotation=30, ha="right")
    ax.set_ylabel(r"$1-\overline{\cos}$ (escape size)")
    ax.set_title("Which input encodings escape the bound")
    return S.save_figure(fig, "fig9_encoding_comparison", OUT)


# --- Fig 10: downstream ceiling + remedy -----------------------------------
def fig10():
    rows = _read(RES / "downstream_task.csv")
    names = [r["model"].replace(" (random)", "").replace(" (pretrained)", "*") for r in rows]
    base = [float(r["baseline_acc"]) for r in rows]
    rwpe = [float(r["rwpe_acc"]) for r in rows]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(S.TEXT_WIDTH * 0.62, 2.3))
    ax.bar(x - w/2, base, w, color=S.PALETTE["red"], edgecolor="black",
           linewidth=0.5, label="frozen embedding")
    ax.bar(x + w/2, rwpe, w, color=S.PALETTE["green"], edgecolor="black",
           linewidth=0.5, label="+RWPE (remedy)")
    ax.axhline(0.5, color="black", lw=0.6, ls=":", label="chance")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("downstream test accuracy")
    ax.set_title("Connected-vs-disconnected: embedding ceiling and remedy")
    _legend(ax)
    return S.save_figure(fig, "fig10_downstream_ceiling", OUT)


# --- Fig 11: diagnostic workflow schematic ---------------------------------
def fig11():
    fig, ax = plt.subplots(figsize=(S.TEXT_WIDTH * 0.62, 1.7))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    boxes = [(0.3, "your MILP\nencoder", S.PALETTE["blue"]),
             (3.2, "1-WL-equivalent\nnon-iso pairs", S.PALETTE["orange"]),
             (6.4, "diagnose():\ncos / L∞ / exact", S.PALETTE["gray"]),
             (9.5, "verdict:\n1-WL bounded?", S.PALETTE["red"])]
    for x, label, c in boxes:
        ax.add_patch(plt.Rectangle((x, 1.0), 2.2, 1.0, fill=False, edgecolor=c, lw=1.2))
        ax.text(x + 1.1, 1.5, label, ha="center", va="center", fontsize=6.8, color=c)
    for x in (2.5, 5.7, 8.8):
        ax.annotate("", (x + 0.7, 1.5), (x, 1.5), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_title("The reusable 1-WL diagnostic (scripts/wl_diagnostic.py)")
    return S.save_figure(fig, "fig11_diagnostic_workflow", OUT)


# --- Fig 12: corpus prevalence ---------------------------------------------
def fig12():
    rows = _read(RES / "corpus_prevalence.csv")
    fams = [r["family"] for r in rows]
    twins = [100 * float(r["frac_with_wl_twins"]) for r in rows]
    fig, ax = plt.subplots(figsize=(S.COL_WIDTH, 2.2))
    ax.bar(range(len(fams)), twins, color=S.PALETTE["purple"],
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams, rotation=20, ha="right")
    ax.set_ylabel("% instances with WL-twin nodes")
    ax.set_title("Prevalence of 1-WL degeneracy in random instances")
    return S.save_figure(fig, "fig12_corpus_prevalence", OUT)


FIGS = [fig6, fig7, fig8, fig9, fig10, fig11, fig12]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in FIGS:
        try:
            for p in f():
                print(f"  {p}")
        except Exception as e:
            print(f"  SKIP {f.__name__}: {e}")
    print(f"\nICDM figures written to {OUT}")


if __name__ == "__main__":
    main()
