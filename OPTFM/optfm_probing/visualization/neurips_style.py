"""NeurIPS-style matplotlib configuration.

Applies font, sizing, and color conventions consistent with the
neurips_2024.sty template: Times body, ~9pt axis labels, ~8pt ticks,
thin rules, constrained layout. Figure widths match the single-column
(3.25in) and two-column (6.75in) text widths of the template.
"""
from __future__ import annotations

import matplotlib as mpl
from matplotlib import pyplot as plt

# NeurIPS (neurips_2024.sty) text widths, in inches.
COL_WIDTH = 3.25
TEXT_WIDTH = 6.75

# Palette: colorblind-safe, adapted from Wong 2011 / tableau-colorblind-10.
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
    """Install NeurIPS-style rcParams globally."""
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "font.size":          9,
        "axes.titlesize":     9,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    7.5,
        "legend.frameon":     False,
        "legend.handlelength": 1.4,
        "legend.borderaxespad": 0.4,
        "axes.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.titlepad":      4.0,
        "axes.labelpad":      2.5,
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "xtick.major.size":   2.5,
        "ytick.major.size":   2.5,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "lines.linewidth":    1.0,
        "lines.markersize":   3.5,
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
    """Save a figure under out_dir/<stem>.<fmt> for each requested format."""
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
