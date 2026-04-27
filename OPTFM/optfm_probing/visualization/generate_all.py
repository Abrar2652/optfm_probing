"""Regenerate every paper figure into results/figures/.

Run from the repo root:

    python visualization/generate_all.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization import (
    fig1_pair_families, fig2_main_result, fig3_probe_battery,
    fig4_layerwise, fig5_training_plateau,
)


def main():
    out_dir = Path(__file__).resolve().parent.parent / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mod in (fig1_pair_families, fig2_main_result,
                fig3_probe_battery, fig4_layerwise, fig5_training_plateau):
        print(f"--- {mod.__name__} ---")
        for p in mod.make_figure(out_dir):
            print(f"  {p}")
    print(f"\nAll figures written to: {out_dir}")


if __name__ == "__main__":
    main()
