"""IEEE-conference matplotlib configuration (IEEEtran, 2-column).

Same public API as visualization.neurips_style (apply_style, save_figure, PALETTE,
MODEL_COLORS, COL_WIDTH, TEXT_WIDTH) but with IEEE column widths and serif fonts so
figures sit cleanly in the IEEEtran two-column layout for the ICDM submission.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib import pyplot as plt

# IEEEtran text widths, in inches: single column ~3.5", full text width ~7.16".
COL_WIDTH = 3.5
TEXT_WIDTH = 7.16

PALETTE = {
    "blue":   "#0173B2",
    "orange": "#DE8F05",
    "green":  "#029E73",
    "red":    "#CC3311",
    "purple": "#7F3C8D",
    "brown":  "#8B4513",
    "pink":   "#CC79A7",
    "gray":   "#555555",
    "lightgray": "#BBBBBB",
}

MODEL_COLORS = {
    "SGFormer+GCN (pretrained)":   PALETTE["blue"],
    "SGFormer+GCN (random)":       PALETTE["orange"],
    "TransConv only (random)":     PALETTE["green"],
    "GNN only (random)":           PALETTE["purple"],
    "Simple GCN (random)":         PALETTE["pink"],
    "Hierarchical OPTFM (random)": PALETTE["red"],
}


def apply_style():
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "font.size":          8,
        "axes.titlesize":     8.5,
        "axes.labelsize":     8,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    6.8,
        "legend.frameon":     True,
        "legend.framealpha":  0.85,
        "legend.edgecolor":   "black",
        "legend.facecolor":   "white",
        "legend.fancybox":    False,
        "axes.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "lines.linewidth":    1.0,
        "lines.markersize":   3.2,
        "grid.linewidth":     0.4,
        "grid.alpha":         0.35,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi":        400,
        "figure.dpi":         120,
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def save_figure(fig, stem, out_dir, formats=("pdf", "png")):
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, format=fmt)
        paths.append(p)
    plt.close(fig)
    return paths
