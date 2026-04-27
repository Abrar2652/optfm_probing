"""NeurIPS-style figure generation for the 1-WL expressiveness study.

Figures (in paper order):
    fig1_pair_families     — the two 1-WL-equivalent non-isomorphic pair families
    fig2_main_result       — cos-sim heatmap (6 models x 8 transforms) + scale plot
    fig3_probe_battery     — frozen-backbone probe R² collapse on 1-WL population
    fig4_layerwise         — layer-wise cos-sim trajectory through hierarchical OPTFM,
                              one line per proof regime (baseline, VGN, RWPE)
    fig5_training_plateau  — full fine-tuning: baseline CE loss plateaus at ln(2),
                              RWPE converges. Training cannot escape the bound.

All outputs land under results/figures/ as both .pdf and .png.
"""
