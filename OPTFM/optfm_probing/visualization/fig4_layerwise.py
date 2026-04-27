"""Figure 4 — layer-wise cos_sim trajectory through the hierarchical OPTFM.

This figure turns the four proof lemmas of `paper/proof_sketch.md` into
an empirical observable: if the theorem is correct, cos_sim(G_A, G_B)
must equal 1.0 at the output of *every* sub-layer of the encoder, not
just end-to-end. We walk a 1-WL-equivalent non-isomorphic pair through
the hierarchical OPTFM, pool at each stage, and plot cos_sim against
stage.

As a negative control on the same figure, we run the same pair with
RWPE-augmented variable features through the same encoder. RWPE is
known to be strictly more expressive than 1-WL, so it should produce a
detectable gap at the input-embedding stage that propagates through the
architecture.

Design principles:
  * single panel, NeurIPS text-width
  * one line per (k, transform) combination
  * stage boundaries annotated with the specific lemma (L0-L4) that each
    segment corroborates
  * legend boxed, transparent, in the empty upper-left corner (the
    baseline line sits at the top-right)
  * no title — description belongs in the caption
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Callable

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.milp_pairs import milp_to_tensors, MILPInstance
from data.milp_pairs_v2 import construct_bipartite_cycle_pair
from models.optfm_hierarchical import HierarchicalOPTFM
from scripts.improvements import make_rwpe_transform, TRANSFORMS
from visualization.neurips_style import (
    apply_style, save_figure, PALETTE, TEXT_WIDTH,
)


# ---------------------------------------------------------------------------
# Instrumented encoder
# ---------------------------------------------------------------------------
#
# Stages (matched to lemmas in paper/proof_sketch.md):
#   S0 input               — identity on inputs (by construction: cos = 1)
#   S1 linear embed        — Wv, Wc linear maps  (L2: per-node)
#   S2 self-attn (V)       — TransConvSelf on V  (L1, L2)
#   S3 self-attn (C)       — TransConvSelf on C  (L1, L2)
#   S4 cross-attn C->V     — TransConvCross + edge agg  (L2)
#   S5 cross-attn V->C     — TransConvCross + edge agg  (L2)
#   S6 GCN branch          — GNNPolicy (var_h2, cons_h2)  (L3)
#   S7 fuse (add)          — α·GCN + (1-α)·attn  (L4)
#   S8 mean-pool           — graph-level pooling  (L4)

STAGES = [
    ("S0", "input",         "L0"),
    ("S1", "linear embed",  "L2"),
    ("S2", "self-attn V",   "L1,L2"),
    ("S3", "self-attn C",   "L1,L2"),
    ("S4", "cross C$\\to$V",  "L2"),
    ("S5", "cross V$\\to$C",  "L2"),
    ("S6", "GCN branch",    "L3"),
    ("S7", "fuse (add)",    "L4"),
    ("S8", "mean-pool",     "L4"),
]


def _graph_embed(cons_h: torch.Tensor, var_h: torch.Tensor) -> torch.Tensor:
    """Pool node embeddings into a graph-level vector (mean over all nodes)."""
    x = torch.cat([cons_h, var_h], dim=0)
    return x.mean(dim=0)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def _run_stages(model: HierarchicalOPTFM,
                cons_x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, var_x: torch.Tensor
                ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Run the hierarchical encoder manually, returning (cons_h, var_h)
    snapshots at each of the 9 stages.

    This is a verbatim copy of HierarchicalOPTFM._encode with an extra
    snapshot list. Any refactor that changes _encode must be mirrored
    here.
    """
    snapshots: List[Tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        # S0: input (cos_x has 1 channel, var_x has 9 — not directly
        # comparable dims, so we return them as-is and handle separately)
        snapshots.append((cons_x, var_x))

        # S1: linear embed
        var_h  = model.var_embedding(var_x)
        cons_h = model.cons_embedding(cons_x)
        snapshots.append((cons_h.clone(), var_h.clone()))

        # S2: self-attn on V
        var_h = model.trans_conv_var(var_h)
        snapshots.append((cons_h.clone(), var_h.clone()))

        # S3: self-attn on C
        cons_h = model.trans_conv_cons(cons_h)
        snapshots.append((cons_h.clone(), var_h.clone()))

        # S4+S5: cross-attention stage
        n_cons = cons_x.shape[0]
        n_vars = var_x.shape[0]
        edge_mat_cv = model._build_dense_edge_matrix(n_cons, n_vars, edge_index, edge_attr)
        edge_mat_vc = edge_mat_cv.t()

        var_h  = model.trans_conv_cross_constovar(var_h, cons_h, edge_mat_vc)
        snapshots.append((cons_h.clone(), var_h.clone()))

        cons_h = model.trans_conv_cross_vartocons(cons_h, var_h, edge_mat_cv)
        snapshots.append((cons_h.clone(), var_h.clone()))

        # S6: GCN branch
        var_h2, cons_h2 = model.GCN(
            model.cons_embedding(cons_x), edge_index, edge_attr,
            model.var_embedding(var_x),
        )
        snapshots.append((cons_h2.clone(), var_h2.clone()))

        # S7: fuse
        alpha = model.graph_weight
        var_h  = alpha * var_h2  + (1 - alpha) * var_h
        cons_h = alpha * cons_h2 + (1 - alpha) * cons_h
        snapshots.append((cons_h.clone(), var_h.clone()))

        # S8: mean-pool — stored as a single vector packaged as (cons_h, var_h)
        #   where both halves are the same pooled vector (so _graph_embed
        #   returns the same thing). Cleanest: just return the graph vec.
        snapshots.append((cons_h.clone(), var_h.clone()))

    return snapshots


def _stage_cos(snapshots_a, snapshots_b, is_pool: List[bool]) -> np.ndarray:
    """Given per-stage (cons, var) snapshots for G_A and G_B, compute
    cos_sim between pooled graph embeddings at each stage.

    For the input stage (S0), we pool cons_x and var_x separately and
    compute a combined cosine similarity: cos_cons * w_c + cos_var * w_v,
    weighted by their respective channel count. By construction uniform
    features give cos_sim = 1.0 at S0.
    """
    cos = np.zeros(len(snapshots_a))
    for i, ((ca, va), (cb, vb)) in enumerate(zip(snapshots_a, snapshots_b)):
        if i == 0:
            # Input stage: cons has 1 channel, var has 9 — pool each side.
            za_c = ca.mean(dim=0)
            zb_c = cb.mean(dim=0)
            za_v = va.mean(dim=0)
            zb_v = vb.mean(dim=0)
            # Concatenate (they're disjoint channel spaces)
            za = torch.cat([za_c, za_v])
            zb = torch.cat([zb_c, zb_v])
        elif is_pool[i]:
            za = torch.cat([ca, va], dim=0).mean(dim=0)
            zb = torch.cat([cb, vb], dim=0).mean(dim=0)
        else:
            za = _graph_embed(ca, va)
            zb = _graph_embed(cb, vb)
        cos[i] = _cos(za, zb)
    return cos


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _apply_rwpe(pair, rwpe_fn) -> Tuple[MILPInstance, MILPInstance]:
    return rwpe_fn(pair.milp_a), rwpe_fn(pair.milp_b)


def compute_trajectories(k: int = 10, seed: int = 0):
    """Compute stage trajectories for the k-th cycle pair under three
    input regimes on the hierarchical OPTFM with fixed random init:

      * baseline              — raw uniform features (theorem predicts cos=1)
      * virtual_global_node   — pretraining augmentation, Lemma 5 (cos=1)
      * RWPE(4,6,8)           — strictly-more-expressive positional signal
                                 (positive control; should produce a gap)
    """
    torch.manual_seed(seed)
    model = HierarchicalOPTFM()
    model.eval()

    pair = construct_bipartite_cycle_pair(k)
    is_pool = [False] * 8 + [True]

    regimes = {
        "baseline":            TRANSFORMS["baseline"],
        "virtual_global_node": TRANSFORMS["virtual_global_node"],
        "rwpe":                TRANSFORMS["rwpe_steps_4_6_8"],
    }

    out = {}
    for name, fn in regimes.items():
        ma, mb = fn(pair.milp_a, seed), fn(pair.milp_b, seed)
        snap_a = _run_stages(model, *milp_to_tensors(ma))
        snap_b = _run_stages(model, *milp_to_tensors(mb))
        out[name] = _stage_cos(snap_a, snap_b, is_pool)

    return out


def compute_trajectory_seeds(k: int = 10, seeds=(0, 1, 2, 3, 4)) -> dict:
    """Run `compute_trajectories` over multiple random initializations of
    the hierarchical OPTFM and stack the results.

    Returns dict[regime_name] = ndarray(n_seeds, n_stages).
    """
    regimes = ("baseline", "virtual_global_node", "rwpe")
    stacks = {r: [] for r in regimes}
    for s in seeds:
        t = compute_trajectories(k=k, seed=s)
        for r in regimes:
            stacks[r].append(t[r])
    return {r: np.stack(stacks[r]) for r in regimes}


def make_figure(out_dir: Path):
    apply_style()
    seeds = (0, 1, 2, 3, 4)
    trajs = compute_trajectory_seeds(k=10, seeds=seeds)

    fig = plt.figure(figsize=(TEXT_WIDTH, 3.1))
    gs = fig.add_gridspec(
        1, 1, left=0.08, right=0.985, top=0.92, bottom=0.30,
    )
    ax = fig.add_subplot(gs[0, 0])

    xs = np.arange(len(STAGES))

    def mean_minmax(arr):
        return arr.mean(axis=0), arr.min(axis=0), arr.max(axis=0)

    # Baseline — mean over seeds, sits at 1.0 up to ~1e-7 float noise.
    # The min-max band is too narrow to show; mark it in the caption.
    b_mean, _, _ = mean_minmax(trajs["baseline"])
    ax.plot(
        xs, b_mean, "o-",
        color=PALETTE["blue"], linewidth=1.2, markersize=4.0,
        markeredgecolor="black", markeredgewidth=0.4,
        label="baseline (Theorem 1)", zorder=4,
    )

    # Virtual-global-node — same story as baseline; diamond markers.
    v_mean, _, _ = mean_minmax(trajs["virtual_global_node"])
    ax.plot(
        xs, v_mean, "D-",
        color=PALETTE["green"], linewidth=1.0, markersize=3.5,
        markeredgecolor="black", markeredgewidth=0.4,
        label="virtual-global-node (Lemma 5)", zorder=3,
    )

    # RWPE — genuine seed variance. Mean line + shaded min-max envelope.
    r_mean, r_min, r_max = mean_minmax(trajs["rwpe"])
    ax.fill_between(
        xs, r_min, r_max,
        color=PALETTE["red"], alpha=0.15, linewidth=0, zorder=2,
    )
    ax.plot(
        xs, r_mean, "s--",
        color=PALETTE["red"], linewidth=1.3, markersize=4.0,
        markerfacecolor="white", markeredgecolor=PALETTE["red"],
        markeredgewidth=1.0,
        label=f"RWPE (4,6,8) — positive control (mean, min–max over {len(seeds)} seeds)",
        zorder=5,
    )

    # Reference line at cos = 1.0
    ax.axhline(1.0, color=PALETTE["gray"], linestyle=":", linewidth=0.6,
               zorder=1)
    ax.text(len(STAGES) - 1 + 0.05, 1.0, "1-WL bound",
            fontsize=6.8, color=PALETTE["gray"], ha="left", va="center")

    # Lemma bracket annotations below the x-axis.
    # Group stages by their lemma label; draw a horizontal bracket with the
    # lemma tag centered beneath.
    groups = [
        (0, 0, "L0"),   # input — identity
        (1, 1, "embed (L2)"),
        (2, 3, "self-attn (L1, L2)"),
        (4, 5, "cross-attn (L2)"),
        (6, 6, "GCN (L3)"),
        (7, 8, "fuse + pool (L4)"),
    ]
    y_bracket = -0.21  # axes-coord y-position for brackets (below x-ticks)
    y_label = y_bracket - 0.055
    for a, b, name in groups:
        # Bracket line in axes coordinates
        ax.annotate(
            "", xy=(a - 0.25, y_bracket), xycoords=("data", "axes fraction"),
            xytext=(b + 0.25, y_bracket), textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="-", color=PALETTE["gray"],
                            linewidth=0.6, shrinkA=0, shrinkB=0),
            annotation_clip=False,
        )
        # Small tick down at each end
        for x0 in (a - 0.25, b + 0.25):
            ax.annotate(
                "", xy=(x0, y_bracket + 0.015),
                xycoords=("data", "axes fraction"),
                xytext=(x0, y_bracket),
                textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="-", color=PALETTE["gray"],
                                linewidth=0.6, shrinkA=0, shrinkB=0),
                annotation_clip=False,
            )
        ax.text(
            (a + b) / 2, y_label, name,
            transform=ax.get_xaxis_transform(),
            fontsize=6.8, color=PALETTE["gray"],
            ha="center", va="top",
            clip_on=False,
        )

    # X-axis: stage labels
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{s[0]}\n{s[1]}" for s in STAGES], fontsize=7)
    ax.tick_params(axis="x", pad=2)

    # Y-axis tightened to the actual dynamic range. RWPE's min-max band
    # reaches its lowest near S4-S5; baselines stay at 1.0.
    y_min_val = float(min(arr.min() for arr in trajs.values()))
    y_min = np.floor((y_min_val - 0.005) * 100) / 100
    ax.set_ylim(y_min, 1.005)
    ax.set_ylabel("cosine similarity  $\\overline{\\cos}(G_A, G_B)$")
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))

    # Legend: boxed, transparent, in the empty lower-right corner — both
    # baseline (pinned at 1.0) and RWPE (drops to ~0.97) leave the
    # upper-right and lower-left comparatively crowded; lower-right stays
    # empty because by S8 the curves have either returned to 1 or sit at
    # their shared attractor near 0.98.
    leg = ax.legend(
        loc="lower left", fontsize=6.8, ncol=2, handlelength=1.8,
        columnspacing=1.0, handletextpad=0.5,
        frameon=True, framealpha=0.0, edgecolor="black", fancybox=False,
    )
    leg.get_frame().set_linewidth(0.5)

    return save_figure(fig, "fig4_layerwise", out_dir)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "results" / "figures"
    for p in make_figure(out):
        print(p)
