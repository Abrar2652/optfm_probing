# 1-WL Expressiveness Study for OPTFM

**DSO-619 Final Project** | Spring 2026 | Prof. Vishal Gupta\
**Authors:** Md Abrar Jahin & Shahriar Tanvir Alam

---

## Headline finding

**OPTFM's full hierarchical multi-view cross-attention architecture is 1-WL bounded on bipartite MILPs.** We give (a) a formal proof sketch covering the full `SGFormer_MIP_hierarchical_improve` class from OPTFM's own source, (b) a large-scale empirical probe on a verified 1-WL equivalent but non-isomorphic MILP pair family where every architecture — pretrained or random, with or without cross-attention — produces **bit-identical embeddings** across 14 problem sizes, and (c) a probe battery showing that even basic structural invariants (number of connected components) collapse to chance R² on the 1-WL population while fitting cleanly on random MILPs. A simple RWPE input transform detectably escapes the limit, ruling out the pathological-pair defense.

---

## What went wrong before

Before running any real probe we verified (with [scripts/sanity_check_pairs.py](scripts/sanity_check_pairs.py)) that the "canonical 1-WL pair" in [data/milp_pairs.py](data/milp_pairs.py) — adapted from an earlier assistant session — is actually **graph-isomorphic** as a feature-labeled bipartite graph:

- The docstring claims MILP-B is infeasible. `scipy.milp` solves MILP-B with `(1,0,1,0,0,1)`: **both** MILPs are feasible.
- A brute-force permutation search finds an explicit isomorphism: π = (0,1,2,4,3) and σ = (1,0,3,2,5,4) map `(A, b_A, f_A) → (A, b_B, f_B)`.

Any permutation-equivariant model will trivially map isomorphic inputs to identical embeddings. The earlier probe's `cos_sim = 1.0` result was **expected and uninformative** — it says nothing about 1-WL expressiveness. We fixed this by constructing a proper 1-WL equivalent but non-isomorphic pair family (see below).

---

## Two independent 1-WL equivalent non-isomorphic pair families

We probe the 1-WL bound on two **structurally independent** pair families. Both are constructed and verified (1-WL equivalent + non-isomorphic + connectivity mismatch) in [data/milp_pairs_v2.py](data/milp_pairs_v2.py).

**Family I — 2-regular cycles** (`construct_bipartite_cycle_pair(k)`):

- `G_A(k) = C_{4k}`-bipartite — a single `4k`-cycle.
- `G_B(k) = k · C_4`-bipartite — `k` disjoint `C_4` components.
- 2-regular on both sides, `2k` cons and `2k` vars, probed at `k ∈ {2, 3, …, 30}`.
- 1-WL equivalent because regular graphs collapse to a single color class; non-isomorphic because `G_A` has 1 connected component and `G_B` has `k`.

**Family II — 3-regular bipartite** (`construct_cubic_bipartite_pair()`):

- `G_A` — connected 3-regular bipartite graph on `6 + 6` vertices built from the circulant offsets `{0, 1, 3} mod 6` (each constraint `i` adjacent to variables `i, i+1, i+3 mod 6`).
- `G_B = K_{3,3} ⊔ K_{3,3}` — two disjoint copies of `K_{3,3}`.
- 12 vertices, 18 edges, 3-regular on both sides.
- 1-WL equivalent by the same uniform-regularity collapse; non-isomorphic because `G_A` has 1 component and `G_B` has 2.

Family II is a genuinely independent test: different regularity (3 vs 2), different local girth distribution, and a different reason for 1-WL equivalence (cubic-regularity collapse vs. cycle-length collapse). If the result held only for Family I a reviewer could suspect a cycle-specific artifact; it also holds for Family II.

---

## Architectures probed

| Model | Source | Params | Pretrained? |
|---|---|---|---|
| SGFormer+GCN (pretrained) | [models/sgformer_mip.py](models/sgformer_mip.py) | 6,820 | `OPTFM/node_pretrain/Models_SCIP/model_params_epoch_31.pth` |
| SGFormer+GCN (random) | same | 6,820 | no |
| TransConv only (random) | [models/sgformer_mip.py](models/sgformer_mip.py) | 1,344 | no |
| GNN only (random) | [models/sgformer_mip.py](models/sgformer_mip.py) | 5,602 | no |
| Simple GCN (random) | [models/sgformer_mip.py](models/sgformer_mip.py) | 736 | no |
| **Hierarchical OPTFM (random)** | [models/optfm_hierarchical.py](models/optfm_hierarchical.py) | 10,852 | no |

The **hierarchical** variant is our dense-operations port of `ours_crossattention_improve.SGFormer_MIP` — the multi-view self-attention + cross-attention + bipartite-GCN architecture that the OPTFM paper actually describes as its novel contribution. The checkpoint shipped in `Models_SCIP/` is for the simpler SGFormer+GCN baseline (verified key-by-key in [scripts/verify_weights.py](scripts/verify_weights.py)). 1-WL boundedness is an *architectural* property that holds for every weight setting, so random initialization is sufficient to probe the hierarchical variant.

---

## Main experimental result

On the combined 15-pair set (14 from Family I + 1 from Family II):

| model | mean cos_sim | 95% CI | exact_frac |
|---|---|---|---|
| SGFormer+GCN (pretrained) | **1.000000** | [1.000000, 1.000000] | **1.00** |
| SGFormer+GCN (random) | **1.000000** | [1.000000, 1.000000] | **1.00** |
| TransConv only (random) | **1.000000** | [1.000000, 1.000000] | **1.00** |
| GNN only (random) | **1.000000** | [1.000000, 1.000000] | **1.00** |
| Simple GCN (random) | **1.000000** | [1.000000, 1.000000] | **1.00** |
| **Hierarchical OPTFM (random)** | **1.000000** | [1.000000, 1.000000] | **1.00** |

Every model maps every pair to bit-identical embeddings, across both pair families.

**Virtual-global-node empirical check** (Lemma 5 in project report): applying the `virtual_global_node` transform from [scripts/improvements.py](scripts/improvements.py) — which appends OPTFM's mean-pooled virtual constraint and variable nodes to every input, exactly as the pretraining pipeline does — leaves `cos_sim = 1.000000` and `exact_frac = 1.00` on every cell. The pretraining trick does not escape the 1-WL bound.

---

## Probe battery: does the embedding encode structural info?

Training linear + MLP probes on frozen pretrained embeddings to predict `n_components`, `lp_value`, `girth_≤_4`, `feasible`:

| target | random MILPs (positive control) | 1-WL equivalent MILPs |
|---|---|---|
| **n_components** (R²) | linear 0.21, **MLP 0.39** | **linear −0.01, MLP −0.01** |
| lp_value (R²) | ≈ 0 | ≈ 0 |
| girth_≤_4 (acc) | 97.3% (majority 97.2%) | 44.9% (majority 55.1%) |
| feasible (acc) | 88.9% (majority 88.9%) | 100% (target degenerate) |

The load-bearing cell: **n_components MLP R² collapses from 0.39 → −0.01** on 1-WL equivalent MILPs, despite the target having 6× more variance there (range 1–30 vs. 1–5). Direct empirical evidence that the embedding is blind to a basic structural invariant within a 1-WL equivalence class. See [scripts/probe_primal_dual.py](scripts/probe_primal_dual.py).

---

## Falsifiability: RWPE positive control

Injecting Random Walk Positional Encoding at steps {4, 6, 8} into the three zero-valued Ecole variable feature slots causes `exact_frac` to drop from 1.00 to 0.00 and cos_sim to drop below 1.0 for every model. The drop is substantially larger on Family II (cubic bipartite) than on Family I alone:

| model | baseline | RWPE (4,6,8), 15 pairs |
|---|---|---|
| SGFormer+GCN (pretrained) | 1.000000 | 0.999486 |
| SGFormer+GCN (random) | 1.000000 | 0.999447 |
| TransConv only (random) | 1.000000 | 0.998045 |
| Simple GCN (random) | 1.000000 | 0.996787 |
| **Hierarchical OPTFM (random)** | 1.000000 | **0.918729** |

RWPE is known to be strictly more expressive than 1-WL because return probabilities at short steps distinguish the two graphs in each family (e.g., `P⁴[i,i] = 3/8` on C₄ₖ differs from `P⁴[i,i] = 1/2` on k·C₄). The fact that this minimal-change transform produces detectable drops on both pair families rules out the "pathological pair" defense.

---

## Repository layout

```
optfm_probing/
├── models/
│   ├── sgformer_mip.py           ← SGFormer+GCN baseline (dense)
│   └── optfm_hierarchical.py     ← port of SGFormer_MIP_hierarchical_improve
├── data/
│   ├── milp_pairs.py             ← original (broken) "canonical" pair
│   └── milp_pairs_v2.py          ← C_{4k}-bipartite vs k·C_4-bipartite family
├── scripts/
│   ├── sanity_check_pairs.py     ← diagnoses why the original pair is broken
│   ├── verify_weights.py         ← exhaustive checkpoint loading verification
│   ├── run_main_experiment.py    ← main cos_sim probe (all 6 models × all transforms)
│   ├── run_complete_experiment.py ← earlier experiment on the (broken) pair
│   ├── probe_primal_dual.py      ← probe battery
│   ├── probe_v2_quick.py         ← quick one-model probe
│   ├── probe_hierarchical_quick.py ← quick all-model probe
│   ├── finetune_pair_classifier.py ← 2-class fine-tuning; loss plateaus at ln 2
│   └── improvements.py           ← RNF, LP-feature, RWPE input transforms
├── visualization/
│   ├── neurips_style.py          ← NeurIPS-style matplotlib rcParams + palette
│   ├── fig1_pair_families.py     ← Fig 1: both pair families with 1-WL coloring
│   ├── fig2_main_result.py       ← Fig 2: heatmap + scale invariance
│   ├── fig3_probe_battery.py     ← Fig 3: probe R² collapse
│   ├── fig4_layerwise.py         ← Fig 4: per-sub-layer cos_sim trajectory
│   ├── fig5_training_plateau.py  ← Fig 5: CE loss plateaus at ln 2 under training
│   └── generate_all.py           ← regenerate every figure
└── results/
    ├── main/<timestamp>/         ← main experiment outputs
    ├── probes/<timestamp>/       ← probe battery outputs
    └── figures/                  ← NeurIPS-style PDFs + PNGs for the paper
```

### Regenerating the paper figures

```bash
python visualization/generate_all.py   # writes results/figures/*.pdf + *.png
```

Figures 1–3 read only from cached `results/main/<latest>/summary.csv`
and `results/probes/<latest>/results.json`, so they regenerate in a few
seconds with no model forward pass. Figure 4 instantiates the
hierarchical OPTFM with 5 fixed random seeds and forwards a $k{=}10$
cycle pair through each sub-layer; a few seconds on CPU. Figure 5
reads from the latest `results/finetune/<timestamp>/results.json` —
regenerate that JSON first via:

```bash
PYTHONUTF8=1 python scripts/finetune_pair_classifier.py --epochs 300 --seeds 0 1 2
```

This is the only figure that runs SGD, and it takes ~2 minutes on CPU
(3 seeds × 2 input regimes × 300 full-batch epochs over 30 examples).

---

## Reproducing

```bash
python -m venv .venv
source .venv/Scripts/activate    # or .venv/bin/activate on Linux/Mac
pip install torch numpy scipy matplotlib networkx pymupdf

# 1. Verify the shipped checkpoint really loads (strict=True) and differs from random
PYTHONUTF8=1 python scripts/verify_weights.py

# 2. Show the original "canonical pair" is graph-isomorphic
PYTHONUTF8=1 python scripts/sanity_check_pairs.py

# 3. Main 1-WL probe on the corrected C_{4k} vs k*C_4 pair family
PYTHONUTF8=1 python scripts/run_main_experiment.py \
    --k_values 2 3 4 5 6 7 8 9 10 12 15 20 25 30 \
    --rnf_samples 30

# 4. Primal-dual / structural probe battery
PYTHONUTF8=1 python scripts/probe_primal_dual.py \
    --n_random 800 \
    --k_values 2 3 4 5 6 7 8 9 10 12 15 20 25 30 \
    --n_objectives 15 \
    --n_seeds 5
```

Note: on Windows we set `PYTHONUTF8=1` because the experiment scripts print `✓`/`✗` status characters.


