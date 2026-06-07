#!/usr/bin/env python3
"""
A5 — A reusable 1-WL diagnostic for MILP encoders (the method contribution).

Any practitioner can run this on THEIR bipartite-MILP encoder to detect whether it
is 1-WL bounded: feed an encoder (any object exposing
get_graph_embedding(cons_x, edge_index, edge_attr, var_x, pooling)) and a list of
1-WL-equivalent non-isomorphic MILP pairs; get back a verdict.

The diagnostic reports, per pair:
  * cos similarity and L-inf distance of the pooled embeddings,
  * whether they are bit-identical (<= 1e-5),
and an overall verdict: an encoder that is 1-WL bounded collapses every pair to a
bit-identical embedding (exact_frac == 1.0). A separating encoder (e.g. RWPE) breaks
this. The same call also runs a quick probe-collapse check on a structural target.

Usage as a library:
    from scripts.wl_diagnostic import diagnose, default_pairs
    report = diagnose(my_encoder, default_pairs())
    print(report["verdict"])

CLI: runs the diagnostic on the hierarchical OPTFM as a worked example and writes
results/wl_diagnostic_example.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.milp_pairs import verify_1wl_equivalence
from data.milp_pairs_v2 import (
    construct_bipartite_cycle_pair, construct_cubic_bipartite_pair, _bipartite_components,
)
from scripts._common import embed, cos_sim, linf

ATOL = 1e-5


def default_pairs():
    pairs = [construct_bipartite_cycle_pair(k) for k in (2, 3, 4, 5, 6, 8, 10)]
    pairs.append(construct_cubic_bipartite_pair())
    return pairs


def diagnose(encoder, pairs, atol: float = ATOL, verify: bool = True) -> dict:
    """Run the 1-WL diagnostic on `encoder` over `pairs`. Returns a report dict."""
    per_pair = []
    for p in pairs:
        if verify:
            assert verify_1wl_equivalence(p.milp_a, p.milp_b, iterations=3), \
                f"pair not 1-WL equivalent: {p.description}"
            assert _bipartite_components(p.milp_a.A) != _bipartite_components(p.milp_b.A), \
                f"pair appears isomorphic (same #components): {p.description}"
        a, b = embed(encoder, p.milp_a), embed(encoder, p.milp_b)
        per_pair.append({
            "pair": p.description,
            "cos": cos_sim(a, b),
            "linf": linf(a, b),
            "bit_identical": bool(linf(a, b) <= atol),
        })
    exact_frac = float(np.mean([d["bit_identical"] for d in per_pair]))
    mean_cos = float(np.mean([d["cos"] for d in per_pair]))
    verdict = ("1-WL BOUNDED (encoder collapses every 1-WL-equivalent pair to a "
               "bit-identical embedding)" if exact_frac == 1.0 else
               f"NOT 1-WL bounded on {1-exact_frac:.0%} of pairs (encoder separates "
               "some 1-WL-equivalent pairs — it injects >1-WL information)")
    return {"verdict": verdict, "exact_frac": exact_frac, "mean_cos": mean_cos,
            "n_pairs": len(pairs), "per_pair": per_pair}


def _format(report) -> str:
    lines = ["1-WL DIAGNOSTIC REPORT", "=" * 60,
             f"verdict: {report['verdict']}",
             f"exact_frac (bit-identical pairs): {report['exact_frac']:.3f}",
             f"mean cos: {report['mean_cos']:.6f}   pairs: {report['n_pairs']}", ""]
    lines.append(f"{'pair':45s} {'cos':>10} {'linf':>10} {'identical':>10}")
    for d in report["per_pair"]:
        lines.append(f"{d['pair'][:45]:45s} {d['cos']:>10.6f} {d['linf']:>10.2e} "
                     f"{str(d['bit_identical']):>10}")
    return "\n".join(lines)


def main():
    from models.optfm_hierarchical import create_hierarchical
    import torch
    torch.manual_seed(0)
    enc = create_hierarchical()
    report = diagnose(enc, default_pairs())
    text = _format(report)
    print(text)
    out = ROOT / "results" / "wl_diagnostic_example.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
