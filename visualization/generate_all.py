"""Regenerate every paper figure into results/figures/.

Run from the repo root:

    python visualization/generate_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from visualization import (
    fig1_pair_families, fig2_main_result, fig3_probe_battery,
    fig4_layerwise, fig5_training_plateau, figs_icdm,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["neurips", "ieee"], default="ieee",
                    help="figure style; ieee for the ICDM submission")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Original figures 1-5 (each module manages its own style).
    for mod in (fig1_pair_families, fig2_main_result,
                fig3_probe_battery, fig4_layerwise, fig5_training_plateau):
        print(f"--- {mod.__name__} ---")
        try:
            for p in mod.make_figure(out_dir):
                print(f"  {p}")
        except Exception as e:
            print(f"  SKIP {mod.__name__}: {e}")
    # New ICDM figures 6-12 (always IEEE-styled, read from results/*.csv).
    print("--- figs_icdm (6-12) ---")
    figs_icdm.main()
    print(f"\nAll figures written to: {out_dir}")


if __name__ == "__main__":
    main()
