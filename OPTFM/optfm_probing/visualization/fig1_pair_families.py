"""Figure 1 — the two 1-WL-equivalent non-isomorphic MILP pair families.

Three visual claims in one figure:
  (a) Family I: C_{4k}-bipartite vs k*C_4-bipartite (k=3), both 2-regular.
  (b) Family II: connected 3-regular bipartite vs 2*K_{3,3}, both 3-regular.
  (c) 1-WL color refinement: histogram of node-color counts coincides across
      refinement depth for both members of each pair.

Nodes are plotted with constraints as circles, variables as squares. The
stable-color assignment used for coloring is the 1-WL fixed-point color
(identical across A and B within each pair, even though the graphs differ).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.milp_pairs_v2 import (
    adjacency_C4k, adjacency_k_times_C4,
    adjacency_cubic_connected, adjacency_two_K33,
)
from visualization.neurips_style import (
    apply_style, save_figure, PALETTE, TEXT_WIDTH,
)


# ----------------------------------------------------------------------
# 1-WL refinement for bipartite graphs with uniform features
# ----------------------------------------------------------------------

def wl_refinement(A: np.ndarray, n_iter: int = 4):
    """Run 1-WL refinement on a bipartite graph with uniform node features.

    Returns `colors_by_iter`, a list of length n_iter+1: each entry is
    a length-(n_cons + n_vars) integer array of node color classes at
    that refinement depth (cons first, then vars).
    """
    n_cons, n_vars = A.shape
    # Initial colors: 0 for cons, 1 for vars.
    c = np.concatenate([np.zeros(n_cons, dtype=int), np.ones(n_vars, dtype=int)])
    colors_by_iter = [c.copy()]
    # Neighbor lookup
    neigh = [[] for _ in range(n_cons + n_vars)]
    rows, cols = np.nonzero(A)
    for r, col in zip(rows, cols):
        neigh[int(r)].append(n_cons + int(col))
        neigh[n_cons + int(col)].append(int(r))
    for _ in range(n_iter):
        sigs = []
        for i in range(n_cons + n_vars):
            sigs.append((c[i], tuple(sorted(c[j] for j in neigh[i]))))
        # Relabel
        mapping = {}
        new_c = np.zeros_like(c)
        for i, s in enumerate(sigs):
            if s not in mapping:
                mapping[s] = len(mapping)
            new_c[i] = mapping[s]
        c = new_c
        colors_by_iter.append(c.copy())
    return colors_by_iter


def color_histogram(c: np.ndarray):
    """Sorted multiset representation of 1-WL colors (canonical signature)."""
    vals, counts = np.unique(c, return_counts=True)
    order = np.argsort(-counts)
    return tuple(sorted(int(x) for x in counts[order]))


# ----------------------------------------------------------------------
# Graph drawing helpers
# ----------------------------------------------------------------------

def bipartite_graph(A: np.ndarray) -> nx.Graph:
    n_cons, n_vars = A.shape
    g = nx.Graph()
    for i in range(n_cons):
        g.add_node(("c", i), side="c")
    for j in range(n_vars):
        g.add_node(("v", j), side="v")
    rows, cols = np.nonzero(A)
    for r, col in zip(rows, cols):
        g.add_edge(("c", int(r)), ("v", int(col)))
    return g


def circular_positions(n_cons, n_vars, r_c=1.0, r_v=0.55):
    """Two concentric rings — constraints outer, variables inner."""
    pos = {}
    for i in range(n_cons):
        t = 2 * np.pi * i / n_cons + np.pi / n_cons
        pos[("c", i)] = (r_c * np.cos(t), r_c * np.sin(t))
    for j in range(n_vars):
        t = 2 * np.pi * j / n_vars - np.pi / n_vars
        pos[("v", j)] = (r_v * np.cos(t), r_v * np.sin(t))
    return pos


def alternating_cycle_positions(n_cons, radius=1.0):
    """Position C_{4k} as a single ring alternating c_0, v_0, c_1, v_1, ...

    adjacency_C4k puts c_i adjacent to v_i and v_{i-1}, which forms the
    single cycle c_0 - v_0 - c_1 - v_1 - ... - c_{n-1} - v_{n-1} - c_0.
    """
    pos = {}
    total = 2 * n_cons
    for i in range(n_cons):
        t_c = 2 * np.pi * (2 * i) / total + np.pi / 2
        t_v = 2 * np.pi * (2 * i + 1) / total + np.pi / 2
        pos[("c", i)] = (radius * np.cos(t_c), radius * np.sin(t_c))
        pos[("v", i)] = (radius * np.cos(t_v), radius * np.sin(t_v))
    return pos


def component_positions(A: np.ndarray, comps, layout="auto"):
    """Lay out each connected component in its own small tile.

    layout="auto": for 2-cons-2-var components (C_4), use a square
    (cons on left, vars on right). Otherwise use concentric rings.
    """
    n_cons, _ = A.shape
    k = len(comps)
    cols = int(np.ceil(np.sqrt(k)))
    rows = int(np.ceil(k / cols))
    pos = {}
    for idx, (c_nodes, v_nodes) in enumerate(comps):
        gx = idx % cols
        gy = idx // cols
        cx, cy = (gx - (cols - 1) / 2) * 1.1, ((rows - 1) / 2 - gy) * 1.1
        nc, nv = len(c_nodes), len(v_nodes)
        if nc == 2 and nv == 2:
            # Square-layout K_{2,2}: cons on left column, vars on right column
            dx, dy = 0.28, 0.28
            pos[("c", c_nodes[0])] = (cx - dx, cy + dy)
            pos[("c", c_nodes[1])] = (cx - dx, cy - dy)
            pos[("v", v_nodes[0])] = (cx + dx, cy + dy)
            pos[("v", v_nodes[1])] = (cx + dx, cy - dy)
        elif nc == 3 and nv == 3:
            # K_{3,3}: cons in a column left, vars in a column right
            for rank, i in enumerate(c_nodes):
                pos[("c", i)] = (cx - 0.35, cy + 0.34 - 0.34 * rank)
            for rank, j in enumerate(v_nodes):
                pos[("v", j)] = (cx + 0.35, cy + 0.34 - 0.34 * rank)
        else:
            for rank, i in enumerate(c_nodes):
                t = 2 * np.pi * rank / nc + np.pi / nc
                pos[("c", i)] = (cx + 0.45 * np.cos(t), cy + 0.45 * np.sin(t))
            for rank, j in enumerate(v_nodes):
                t = 2 * np.pi * rank / nv - np.pi / nv
                pos[("v", j)] = (cx + 0.22 * np.cos(t), cy + 0.22 * np.sin(t))
    return pos


def split_components(A: np.ndarray):
    """Return a list of (cons_list, vars_list) per connected component."""
    n_cons, n_vars = A.shape
    parent = list(range(n_cons + n_vars))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    rows, cols = np.nonzero(A)
    for r, c in zip(rows, cols):
        union(int(r), n_cons + int(c))
    comps = {}
    for i in range(n_cons):
        comps.setdefault(find(i), ([], []))[0].append(i)
    for j in range(n_vars):
        comps.setdefault(find(n_cons + j), ([], []))[1].append(j)
    return list(comps.values())


def draw_bipartite(ax, A, pos, wl_colors, title, cons_palette, var_palette):
    """Render a bipartite graph with 1-WL colors on nodes."""
    n_cons, n_vars = A.shape
    g = bipartite_graph(A)
    # Edges first
    for (u, v) in g.edges():
        xu, yu = pos[u]
        xv, yv = pos[v]
        ax.plot([xu, xv], [yu, yv], color=PALETTE["lightgray"], linewidth=0.7, zorder=1)
    # Nodes
    for i in range(n_cons):
        x, y = pos[("c", i)]
        c = cons_palette[wl_colors[i] % len(cons_palette)]
        ax.scatter([x], [y], s=55, marker="o", facecolor=c, edgecolor="black",
                   linewidth=0.5, zorder=2)
    for j in range(n_vars):
        x, y = pos[("v", j)]
        c = var_palette[wl_colors[n_cons + j] % len(var_palette)]
        ax.scatter([x], [y], s=45, marker="s", facecolor=c, edgecolor="black",
                   linewidth=0.5, zorder=2)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")


# ----------------------------------------------------------------------
# Main figure
# ----------------------------------------------------------------------

def make_figure(out_dir: Path):
    apply_style()

    # Pair I: k = 3 so the difference between connected and 3 components is visible.
    k = 3
    A_I_a = adjacency_C4k(k)
    A_I_b = adjacency_k_times_C4(k)

    # Pair II: cubic connected vs 2 * K_{3,3}
    A_II_a = adjacency_cubic_connected(6)
    A_II_b = adjacency_two_K33()

    # 1-WL refinement: uniform features make 1-WL collapse everything into
    # one cons color and one var color at every depth.
    wl_I_a = wl_refinement(A_I_a, n_iter=3)
    wl_I_b = wl_refinement(A_I_b, n_iter=3)
    wl_II_a = wl_refinement(A_II_a, n_iter=3)
    wl_II_b = wl_refinement(A_II_b, n_iter=3)

    # Signatures match at every iteration — sanity check.
    for it in range(len(wl_I_a)):
        assert color_histogram(wl_I_a[it]) == color_histogram(wl_I_b[it])
    for it in range(len(wl_II_a)):
        assert color_histogram(wl_II_a[it]) == color_histogram(wl_II_b[it])

    # Use fixed-point colors (last iteration) for drawing.
    cons_palette = [PALETTE["blue"]]        # cons stay one color class
    var_palette  = [PALETTE["orange"]]      # vars stay one color class

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.7))
    gs = fig.add_gridspec(
        1, 4, width_ratios=[1, 1, 1, 1], wspace=0.05,
        left=0.02, right=0.98, top=0.88, bottom=0.12,
    )

    ax_Ia = fig.add_subplot(gs[0, 0])
    ax_Ib = fig.add_subplot(gs[0, 1])
    ax_IIa = fig.add_subplot(gs[0, 2])
    ax_IIb = fig.add_subplot(gs[0, 3])

    label_kwargs = dict(fontsize=8.5, loc="left", pad=2)

    # Family I — A: C_{4k} as a single alternating ring
    n_cons_I = A_I_a.shape[0]
    pos_Ia = alternating_cycle_positions(n_cons_I, radius=1.0)
    draw_bipartite(ax_Ia, A_I_a, pos_Ia, wl_I_a[-1], "", cons_palette, var_palette)

    # Family I — B: k disjoint C_4's in a row
    comps_Ib = split_components(A_I_b)
    pos_Ib = component_positions(A_I_b, comps_Ib)
    draw_bipartite(ax_Ib, A_I_b, pos_Ib, wl_I_b[-1], "", cons_palette, var_palette)

    # Family II — A: circulant {0,1,3} mod 6
    pos_IIa = circular_positions(6, 6, r_c=1.0, r_v=0.55)
    draw_bipartite(ax_IIa, A_II_a, pos_IIa, wl_II_a[-1], "", cons_palette, var_palette)

    # Family II — B: 2 * K_{3,3}
    comps_IIb = split_components(A_II_b)
    pos_IIb = component_positions(A_II_b, comps_IIb)
    draw_bipartite(ax_IIb, A_II_b, pos_IIb, wl_II_b[-1], "", cons_palette, var_palette)

    # Panel labels — placed above each axis in the "title slot", not overlapping nodes
    for ax, tag in [
        (ax_Ia,  r"(a) $G^{\mathrm{I}}_A\!:\ C_{4k},\ k{=}3$"),
        (ax_Ib,  r"(b) $G^{\mathrm{I}}_B\!:\ k\cdot C_4$"),
        (ax_IIa, r"(c) $G^{\mathrm{II}}_A\!:$ cubic connected"),
        (ax_IIb, r"(d) $G^{\mathrm{II}}_B\!:\ K_{3,3}\sqcup K_{3,3}$"),
    ]:
        ax.text(0.5, 1.02, tag, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8)

    # Vertical divider between families
    fig.add_artist(plt.Line2D(
        [0.51, 0.51], [0.08, 0.96], transform=fig.transFigure,
        color=PALETTE["lightgray"], linewidth=0.5, linestyle="--",
    ))

    fig.text(0.265, 0.04, "Family I (2-regular)", ha="center",
             fontsize=8.5, style="italic", color=PALETTE["gray"])
    fig.text(0.75, 0.04, "Family II (3-regular)", ha="center",
             fontsize=8.5, style="italic", color=PALETTE["gray"])

    paths = save_figure(fig, "fig1_pair_families", out_dir)
    return paths


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "results" / "figures"
    for p in make_figure(out):
        print(p)
