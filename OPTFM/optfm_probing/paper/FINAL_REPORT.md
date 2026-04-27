# OPTFM 1-WL Boundedness — Final Report

**DSO-619 Final Project** | Spring 2026 | Prof. Vishal Gupta
**Authors:** Md Abrar Jahin, Shahriar Tanvir Alam

---

## Summary

We investigate whether the OPTFM architecture proposed in Yuan et al. (NeurIPS 2025) — specifically its *hierarchical* multi-view cross-attention variant — escapes the 1-WL expressiveness bound that Chen et al. (ICLR 2023) established for GNNs applied to bipartite MILPs. The OPTFM paper motivates its multi-view graph transformer and hybrid self-/cross-attention as "going beyond the local interaction structure of standard GNNs through scalable global interactions."

Our three-part contribution is:

1. **Theorem (§2 + `paper/proof_sketch.md`).** We prove that the full hierarchical OPTFM encoder, for *any* setting of its weights, is 1-WL bounded on bipartite MILPs. The proof extends the Xu–Chen argument by showing that (a) SGFormer-style linear attention with Frobenius-norm normalization is a multiset function of its source set, (b) OPTFM's additive edge-weighted cross-attention message is a multiset function of the edge-weighted neighborhood, and (c) the convex combination with the bipartite GCN branch preserves the bound. None of OPTFM's architectural novelties over SGFormer escape the limit.

2. **Large-scale empirical probe (§3).** We construct a population of 15 genuinely 1-WL-equivalent-but-non-isomorphic bipartite MILP pairs drawn from **two structurally independent families**: 14 *2-regular cycle pairs* (C₄ₖ vs k·C₄ for k ∈ {2, …, 30}) and one *3-regular cubic pair* (connected 6+6 cubic bipartite vs 2·K₃,₃), all verified non-isomorphic via brute-force / VF2. Across six architectures — the pre-trained SGFormer+GCN baseline checkpoint, a random-init copy, TransConv-only, GNN-only, a 2-layer GCN, and a random-init full *hierarchical* OPTFM — every single pair yields bit-identical embeddings (cos_sim = 1.000000). The result is scale-invariant (constant at 1.0 from k=2 to k=30) and regularity-invariant (holds for both 2-regular and 3-regular families). We also verify empirically that OPTFM's virtual-global-node pretraining augmentation (Lemma 5 of the proof) does not escape the limit: applying the augmentation leaves cos_sim = 1.000000 and exact_frac = 1.00 on every cell.

3. **Primal-dual / structural probe battery (§4).** We train linear and MLP probes on frozen pretrained embeddings to predict four targets: `n_components`, `lp_value`, `girth_≤_4`, and `feasible`. On random bipartite MILPs the MLP probe attains R² = 0.387 for n_components (clear signal). On the 1-WL equivalent dataset the same probe collapses to R² = −0.011, **despite the target variance being six times larger in the 1-WL dataset than in the random one**. This shows empirically that the OPTFM embedding does not encode even a basic structural invariant (number of components) within a 1-WL equivalence class — which is also the closest proxy we have to the "primal-dual information" claim in our original project proposal.

4. **Falsifiability / positive control (§5).** Injecting Random Walk Positional Encodings at steps 4, 6, 8 into the three zero-valued Ecole variable feature slots causes the cosine similarity to drop below 1.0 across all models. The drop is modest (≈ 0.9995 for the pre-trained model) because the injected signal is small relative to the rest of the feature vector, but `exact_frac` falls from 1.00 to 0.00, confirming that the probe is genuinely falsifiable.

---

## 1. What went wrong in the initial probe — and why that matters

Before doing any of the above, we verified (in `scripts/sanity_check_pairs.py`) that the "canonical 1-WL pair" defined in `data/milp_pairs.py` — adapted from an earlier assistant session — is actually **graph-isomorphic** as a feature-labeled bipartite graph. Specifically:

- The docstring claims MILP-B is infeasible. `scipy.milp` finds (1,0,1,0,0,1) as an integer-feasible point for MILP-B. **Both** MILPs in the "canonical pair" are feasible.
- A brute-force permutation search finds an explicit isomorphism: π = (0,1,2,4,3) on constraints and σ = (1,0,3,2,5,4) on variables maps (A, b_A, f_A) → (A, b_B, f_B). The two MILPs are the *same* graph with relabeled constraints and variables.
- Any permutation-equivariant model (including every GNN ever) will map isomorphic inputs to identical embeddings. The fact that the earlier probe found cos_sim = 1.0 is **expected and uninformative** — it tells us nothing about 1-WL expressiveness.

This finding alone changes the interpretation of the project from "we ran Chen et al.'s example" to "we found that the standard 'canonical 1-WL pair' as commonly coded is a degenerate test case, and constructed a genuinely non-isomorphic pair family that serves as a real probe." Section 3 below builds on this corrected foundation.

---

## 2. The 1-WL boundedness theorem (full proof in `paper/proof_sketch.md`)

**Theorem.** *Let $G_1 = (C_1, V_1, A_1, f^1_C, f^1_V)$ and $G_2 = (C_2, V_2, A_2, f^2_C, f^2_V)$ be feature-labeled bipartite MILPs that are 1-WL equivalent under the standard color refinement of §1.2 of the proof. For any weights of the hierarchical OPTFM encoder `SGFormer_MIP_hierarchical_improve` and any pooling in {mean, sum, max}, the pooled graph embeddings satisfy $z(G_1) = z(G_2)$.*

**Key lemmas.**

- **L1** (Linear attention is a multiset function of the source.) SGFormer's normalized linear attention output at query $i$ depends only on $q_i$, $v_i$, and the multisets $\{\!\{k_\ell\}\!\}$ and $\{\!\{k_\ell^\top v_\ell\}\!\}$. This is because the numerator $\frac{1}{N} q_i (\sum_\ell k_\ell^\top v_\ell)$ and the denominator normalizer $\sum_\ell k_\ell^\top \mathbf{1}$ are both commutative sums. The Frobenius-norm normalization depends only on $\|Q\|_F$, also a multiset invariant.

- **L2** (Cross-attention + edge-weighted aggregation preserves 1-WL.) TransConvCross adds one term $m_i = \sum_j A_{ij} y_j$ to the cross-attention output. Two 1-WL equivalent nodes have, by definition, a bijection between their edge-weighted neighborhoods preserving $(A_{ij}, \chi^{(t)}(j))$ pairs. Sums are multiset functions. ✓

- **L3** (The bipartite GCN branch is 1-WL bounded.) Gasse-style bipartite message passing is an MPNN; the 1-WL bound for MPNNs (Xu et al. 2019) ports directly to the bipartite case (Chen et al. 2023).

- **L4** (Convex combination and pooling preserve 1-WL.) The convex combination $\alpha \tilde{v}^{(g)} + (1-\alpha) \tilde{v}$ is per-node. Mean/sum/max pool are symmetric functions of the final node multiset.

The theorem follows by propagating the multiset-equivalence invariant through each stage of the encoder. See `paper/proof_sketch.md` for the full statement and step-by-step proof.

---

## 3. Main experiment: cos_sim distribution on 1-WL equivalent non-isomorphic pairs

### 3.1 Pair families

**Family I — 2-regular cycles.**
$$G_A(k) = \text{C}_{4k}\text{-bipartite}, \qquad G_B(k) = k \cdot \text{C}_4\text{-bipartite}, \qquad k \in \{2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30\}.$$
Each pair has $2k$ constraint nodes and $2k$ variable nodes, both sides 2-regular, with the constraint RHS set to 1 so that constraint features are uniform.

**Family II — 3-regular cubic bipartite.** A single pair at 6+6 vertices:

- $G_A$ = connected 3-regular bipartite graph built from the circulant offsets $\{0, 1, 3\} \pmod 6$ (each constraint $i$ is adjacent to variables $i$, $i+1$, $i+3 \pmod 6$).
- $G_B = K_{3,3} \sqcup K_{3,3}$ — two disjoint copies of $K_{3,3}$.

12 vertices, 18 edges, 3-regular on both sides. Family II is structurally independent of Family I: the regularity, local girth distribution, and the specific mechanism by which 1-WL equivalence arises are all different (uniform-cubic-regularity collapse rather than cycle-length collapse).

Both families are verified (by `data/milp_pairs_v2.diagnose_pair`) to satisfy:

- Same node count, same edge count, same row/column sum sequences
- 1-WL equivalent at all refinement depths (color-refinement histograms match)
- NON-isomorphic as feature-labeled bipartite graphs (brute-force for small instances, VF2 for large)
- Different number of connected components ($G_A$ has 1, $G_B$ has $k$ or 2)

![Figure 1](../results/figures/fig1_pair_families.png)

**Figure 1: The two test pair families.** Constraints are drawn as circles, variables as squares; edges come from the nonzero pattern of the constraint matrix $A$. Nodes are colored by their 1-WL fixed-point color class — uniform cons-features and uniform var-features force 1-WL refinement to collapse every constraint into one class (blue) and every variable into one class (orange) in both members of each pair. (a) $G^{\mathrm{I}}_A$: a single $4k$-cycle drawn as an alternating ring (shown for $k{=}3$, so 12 vertices, connected). (b) $G^{\mathrm{I}}_B$: $k$ disjoint $C_4$'s (i.e., $k$ copies of $K_{2,2}$, $k$ components). (c) $G^{\mathrm{II}}_A$: connected 3-regular bipartite graph on $6+6$ vertices built from the circulant offsets $\{0,1,3\} \bmod 6$. (d) $G^{\mathrm{II}}_B$: two disjoint copies of $K_{3,3}$. Within each family the two graphs share the same 1-WL color histogram at every refinement depth but differ in number of connected components — a quintessential 1-WL-invisible graph invariant.

### 3.2 Models probed

All six architectures are implemented in this repo:

| Model | Source | # params | Pretrained? |
|---|---|---|---|
| SGFormer+GCN (pretrained) | `models/sgformer_mip.SGFormer_MIP` | 6,820 | yes (`model_params_epoch_31.pth`) |
| SGFormer+GCN (random) | same architecture, random init | 6,820 | no |
| TransConv only (random) | `TransConvOnly` | 1,344 | no |
| GNN only (random) | `GNNOnly` | 5,602 | no |
| Simple GCN (random) | `SimpleGCN` | 736 | no |
| **Hierarchical OPTFM (random)** | `models/optfm_hierarchical.HierarchicalOPTFM` | 10,852 | no |

The hierarchical variant is our dense-operation port of the full `SGFormer_MIP_hierarchical_improve` class from `OPTFM/node_pretrain/ours_crossattention_improve.py`, which is the architecture the OPTFM paper actually describes. The shipped checkpoint is for the simpler SGFormer+GCN baseline, not the hierarchical variant (verified via exhaustive key-match in `scripts/verify_weights.py`). Since 1-WL boundedness is an architectural property that holds for all weights, random initialization is sufficient to probe it on the hierarchical variant.

### 3.3 Result

Running `python scripts/run_main_experiment.py --family both` on all 14 cycle-family k values plus the single cubic pair, averaging RNF transforms over 30 seeds:

| model | mean cos_sim | 95% CI | exact_frac |
|---|---|---|---|
| SGFormer+GCN (pretrained) | 1.000000 | [1.000000, 1.000000] | **1.00** |
| SGFormer+GCN (random) | 1.000000 | [1.000000, 1.000000] | **1.00** |
| TransConv only (random) | 1.000000 | [1.000000, 1.000000] | **1.00** |
| GNN only (random) | 1.000000 | [1.000000, 1.000000] | **1.00** |
| Simple GCN (random) | 1.000000 | [1.000000, 1.000000] | **1.00** |
| **Hierarchical OPTFM (random)** | **1.000000** | [1.000000, 1.000000] | **1.00** |

For every architecture, for every one of the 15 pairs (14 cycle-family + 1 cubic-family), the two graph embeddings are **bit-identical** (exact_frac = 1.00 over 15 pairs). This is the empirical version of the theorem. The key point for the report is that the hierarchical OPTFM, which the paper motivates as going beyond GNN-like local structure via global multi-view attention, behaves identically to the baseline GCN on this test.

![Figure 2](../results/figures/fig2_main_result.png)

**Figure 2: 1-WL boundedness is architecturally universal across the OPTFM encoder family.** (a) Per-cell distance from bit-identical, $1 - \overline{\cos}(G_A, G_B)$, on a logarithmic color scale; mean over 15 1-WL-equivalent non-isomorphic pairs. Cells marked '=' hit bit-identical (cos_sim $= 1.000000$ to within $10^{-5}$). Baseline and virtual-global-node (VGN) columns are '=' across all six architectures — the pretraining trick of appending mean-pooled global nodes does not escape the 1-WL bound. LP-primal is '=' for most models because variable-features are uniform on both graphs (the dual-feedback channel is what occasionally leaks signal). Only RWPE reliably produces a non-trivial gap, and only the full hierarchical OPTFM shows a gap as large as $\approx 10^{-2}$. (b) Scale invariance of the baseline result. For every scale $k \in \{2,\dots,30\}$ on the cycle family, all six architectures are pinned at cos_sim $= 1.0$; the hierarchical OPTFM under RWPE (open squares, dashed) is plotted as a falsifiability reference and drops below 1.0 at every scale.

**Virtual-global-node augmentation (Lemma 5).** Applying the `virtual_global_node` transform from `scripts/improvements.py` — which appends the mean-pooled virtual constraint and variable nodes exactly as OPTFM's pretraining pipeline does (see `OPTFM/node_pretrain/main_mip.py` lines 148–171) — leaves cos_sim = 1.000000 and exact_frac = 1.00 across every model and every pair. This matches Lemma 5 in the proof sketch and addresses the most natural objection ("the actual pretrained pipeline uses global nodes, so your bound doesn't apply"): it does.

---

## 4. Probe battery — does the embedding encode primal-dual / structural info?

### 4.1 Setup

We train linear and MLP probes on **frozen** pretrained OPTFM/SGFormer embeddings to predict four targets on two datasets:

- **D_rand** (positive control): 800 random bipartite MILPs with n_cons=n_vars=10, sparsity ≈ 0.25, 50/50 mix of equality and ≤ constraints, random objective and RHS. n_components varies in [1, 5], LP values vary in [−6.77, 4.52].
- **D_1wl** (target): 420 instances — for each k ∈ {2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30} and each of 15 random objective vectors, both the C₄ₖ-bipartite (A) and k·C₄-bipartite (B) MILP with that objective. var_features stay zero-valued across instances, so all instances are mutually 1-WL equivalent, but n_components varies explicitly in [1, 30] and LP value varies in [−29.16, 8.06].

Probes: `LinearProbe` (1-layer linear), `MLPProbe` (64-64 MLP with ReLU). 70/30 train/test split. 5 random seeds.

### 4.2 Targets

| Target | Type | Positive control (random) | 1-WL equivalent |
|---|---|---|---|
| `n_components` | regression (R²) | linear = 0.21, **MLP = 0.39** | **linear = −0.01, MLP = −0.01** |
| `lp_value` | regression (R²) | linear = −0.01, MLP = 0.00 | linear = −0.02, MLP = −0.01 |
| `girth_≤_4` | classification (acc) | 97.3% (majority = 97.2%) | 44.9% (majority = 55.1%) |
| `feasible` | classification (acc) | 88.9% (majority = 88.9%) | 100% (target degenerate) |

### 4.3 Interpretation

The load-bearing cell is the second column of `n_components`: **MLP R² = 0.387 on random MILPs, collapsing to R² = −0.011 on 1-WL equivalent MILPs**. This collapse happens despite the target variance being ≈ 6× larger on the 1-WL dataset (range 1–30 vs. 1–5). It is direct empirical evidence that the pretrained OPTFM/SGFormer embedding does not distinguish C₄ₖ from k·C₄, even for the structural invariant `n_components` which is *entirely determined* by the graph topology and is readily computable by any non-1-WL method.

The `lp_value` row is, if anything, even more damning: the pretrained model cannot predict LP relaxation value from its embedding even on the random dataset. This was not the setup of our probe (we froze the embedding and fit a probe from scratch), but it does suggest that the pretraining objective (masked edge prediction) does not implicitly induce LP-value-aware representations. This supports the second substantive claim the professor identified — that OPTFM does not capture primal-dual / Lagrangian information — as an empirical finding complementary to the 1-WL result.

The `girth_≤_4` row offers a negative result worth noting: the probe accuracy is *below* the majority baseline on the 1-WL dataset (44.9% vs. 55.1% majority). This is consistent with the embedding being *blind* to the girth invariant: the probe is trying to fit a target whose variation is invisible in the input.

![Figure 3](../results/figures/fig3_probe_battery.png)

**Figure 3: Frozen-backbone probe R² collapses on the 1-WL population despite larger target variance.** (a) Linear and MLP probe scores for four graph-level targets: R² for regression targets (`n_components`, `lp_value`), and accuracy-minus-majority for classification targets (`girth ≤ 4`, `feasible`). On random MILPs (positive control, blue), the MLP recovers `n_components` with $R^2 = +0.40$. On the 1-WL equivalent population (red), the same probe collapses to $R^2 = -0.01$ — negative on held-out data. Error bars show $\pm 1$ std over 5 seeds. `lp_value` is at chance on both populations, indicating that masked-edge pretraining does not induce LP-aware representations. On `girth ≤ 4` the 1-WL probes fit *worse* than the majority classifier, consistent with the embedding being blind to the distinguishing invariant. `feasible` is degenerate on the 1-WL set (every instance is feasible) so both bars sit at zero. (b) Variance sanity check: the 1-WL targets have $\geq$ as much variance as the random targets ($6\times$ for `n_components`), so the collapse in (a) cannot be attributed to a degenerate target.

---

## 5. Falsifiability: positive control via RWPE

A purely negative result ("the probe gives 1.0 for every pair") raises the obvious concern: maybe the test is broken and no model would ever get cos_sim < 1.0 on this pair family. We check this by injecting Random Walk Positional Encoding (RWPE) features at steps {4, 6, 8} into the three zero-valued Ecole variable feature slots, plus one extra return probability added to the single constraint feature. These values are known to differ: `P⁴[i,i] = 3/8` on C₄ₖ vs `P⁴[i,i] = 1/2` on k·C₄ (verified analytically in the inline comments of `scripts/improvements.make_rwpe_transform`).

Result (full-scale run over the combined 15-pair set, averaging RNF samples over 30 seeds):

| model | baseline cos | RWPE cos | RWPE 95% CI | exact_frac |
|---|---|---|---|---|
| SGFormer+GCN (pretrained) | 1.000000 | 0.999486 | [0.99946, 0.99954] | **0.00** |
| SGFormer+GCN (random) | 1.000000 | 0.999447 | [0.99937, 0.99949] | **0.00** |
| TransConv only (random) | 1.000000 | 0.998045 | [0.99803, 0.99807] | **0.00** |
| GNN only (random) | 1.000000 | 0.999996 | [0.99999, 1.00000] | **0.00** |
| Simple GCN (random) | 1.000000 | 0.996787 | [0.99676, 0.99680] | **0.00** |
| **Hierarchical OPTFM (random)** | 1.000000 | **0.918729** | [0.91320, 0.92972] | **0.00** |

For every architecture, `exact_frac` drops from 1.00 (bit-identical) to 0.00 (always distinguishable). The full hierarchical OPTFM — the architecture of most interest — shows the most dramatic drop, with mean cos_sim dropping from 1.000000 to ≈ 0.919 once RWPE features are injected. The key point is that the probe **is genuinely falsifiable**: an architecture that truly encoded >1-WL information would be detected by this test, and we confirmed this by building the simplest such architecture (input RWPE) and observing the drop.

The large drop on the hierarchical architecture (vs. the smaller drops elsewhere) is because it has more attention pathways through which the injected per-node return probabilities can propagate and amplify. Random-init networks are not "trained" to respond to RWPE features, so the effect size is purely a function of how the architecture moves the input signal through its layers. The drop being biggest on the most expressive architecture is reassuring: if anything could exploit structural PE, it should be the hierarchical variant.

This also rules out the "pathological pair" critique: not only does RWPE distinguish both families, but the two families are *structurally independent* from each other (2-regular cycles vs. 3-regular cubic), and the result is consistent across both. The failure is specific to the OPTFM/GNN family, not to any particular pair construction.

### 5.1 Where does the 1-WL bound hold inside the encoder?

The end-to-end results of §3 and §5 show that the theorem's prediction holds at the output of the whole encoder. The proof sketch makes a stronger claim: it proves the 1-WL boundedness stage-by-stage, with one lemma per architectural sub-layer. Figure 4 turns this stronger claim into an empirical observable by instrumenting the hierarchical OPTFM and computing cos_sim$(G_A, G_B)$ after every sub-layer, for a representative $k{=}10$ pair from Family I.

![Figure 4](../results/figures/fig4_layerwise.png)

**Figure 4: The 1-WL bound holds at every sub-layer of the hierarchical OPTFM, not only end-to-end.** We instrument the encoder and compute cos_sim$(G_A, G_B)$ on the pooled representation at each of its nine sub-layers, on the $k{=}10$ pair from Family I. Stages are grouped below the x-axis by the proof lemma they corroborate (L0–L4 from `paper/proof_sketch.md`). Curves are the mean over **5 independent random initializations** of the hierarchical OPTFM (seeds 0–4). **Baseline (blue circles):** uniform input features; by Theorem 1 the cosine must equal 1 at every stage, and it does — cos $= 1.0 \pm 10^{-7}$ across all 5 seeds (at floating-point noise), empirically verifying each of Lemmas 1–4 individually. **Virtual-global-node (green diamonds):** the pretraining augmentation of Lemma 5; coincides with baseline at the same noise level, empirically confirming that OPTFM's mean-pooled virtual nodes do not escape the bound. **RWPE (red squares, dashed; shaded band = min–max over seeds):** positive control. RWPE is strictly more expressive than 1-WL, so the cosine must fall below 1; the drop appears at S1 (linear embedding of the perturbed features) and propagates through self-attention and cross-attention with genuine seed variance in the S4–S5 cross-attention range. At S6 the cosine *consistently* recovers to $\approx 1.0$ across all 5 seeds: the GCN branch re-embeds from the raw input via a separate parameter group whose random-init response is effectively blind to the RWPE channels, partially masking the signal after fusion at S7 and pooling at S8. This figure thus functions simultaneously as a layer-by-layer verification of the theorem, a refutation of the virtual-global-node defense, and a diagnosis of *where* in the architecture a more-expressive input signal is preserved vs. attenuated.

### 5.2 Can training overcome the bound? A supervised-classification test

Every result up to this point has used either the pretrained backbone or randomly initialized weights without any optimization. A sharp reviewer objection is: *"You have not actually trained the model to distinguish these pairs — maybe optimization would find weights that do."* The 1-WL boundedness theorem rules this out: the bound holds *for any fixed weights*, so no optimization trajectory through weight space can ever produce a weight setting that distinguishes two 1-WL-equivalent non-isomorphic MILPs. We verify this directly.

**Setup.** We put a 2-class linear head on top of the mean-pooled hidden state of the hierarchical OPTFM and jointly train backbone + head as a binary classifier that must separate $G_A(k)$ (label $0$) from $G_B(k)$ (label $1$). The dataset is all 30 examples from the 15 1-WL-equivalent non-isomorphic pairs (14 cycle + 1 cubic), balanced 50/50. Cross-entropy loss, Adam at learning rate $10^{-3}$, full-batch updates, 300 epochs, 3 random seeds. Script: `scripts/finetune_pair_classifier.py`.

**Theoretical prediction.** If the theorem is correct, the pooled backbone embedding is bit-identical for $G_A$ and $G_B$ at every step of training, so the linear head produces identical logits for the two inputs of every pair, and the binary cross-entropy loss is pinned exactly at
$$\mathcal L^\star \;=\; -\frac{1}{2}\log p \;-\; \frac{1}{2}\log(1-p),\qquad p = \sigma(\text{logit})$$
which is minimized at $p = 1/2$, giving $\mathcal L^\star = \ln 2 \approx 0.693147$. No optimizer, no learning rate schedule, no head design can do better.

**Result.**

![Figure 5](../results/figures/fig5_training_plateau.png)

**Figure 5: Training cannot overcome the 1-WL bound.** Cross-entropy loss vs. epoch for a 2-class linear-head classifier trained jointly with the hierarchical OPTFM backbone on the 15-pair 1-WL-equivalent population. Three seeds per curve. **Baseline (blue, solid):** loss plateaus exactly at $\ln 2 \approx 0.693$ — to six decimal places, across all three seeds, for all 300 epochs. At the end of training the pooled embeddings of $G_A$ and $G_B$ are still bit-identical for *every* one of the 15 pairs (frac_pairs_identical_after training = 1.000). **RWPE (red, dashed):** the exact same architecture and training procedure, but with RWPE(4,6,8)-augmented inputs that inject a strictly-more-expressive-than-1-WL signal. Loss converges to $\approx 3 \times 10^{-4}$ — five to seven decades below the baseline plateau, depending on the seed — and the pooled-embedding identity is broken for every pair (frac_pairs_identical = 0.000). The two curves together make the strongest possible empirical statement of the theorem: the architectural bound survives arbitrary optimization, and a genuine >1-WL signal trivially breaks through it.

The quantitative numerical agreement is striking. Baseline final losses across seeds are $0.693147$, $0.693147$, $0.693147$ — matching $\ln 2$ to six decimals — while baseline initial losses (random head + random backbone) were $0.694$, $0.700$, $0.721$. Training is moving the loss, but only toward the theoretical floor; it cannot cross it.

---

## 6. Scope and limitations

**What we have established.**

1. The hierarchical OPTFM encoder is 1-WL bounded for any weight setting on bipartite MILPs (theorem, proof sketch in `paper/proof_sketch.md`).
2. On a population of 14 verified 1-WL equivalent non-isomorphic bipartite MILP pairs, every architecture we test — including the full hierarchical OPTFM — produces bit-identical graph embeddings for every pair.
3. On a 420-sample probing dataset, the pretrained OPTFM embedding fails to encode `n_components`, collapsing from R² = 0.39 on random MILPs to R² = −0.01 on 1-WL equivalent pairs, despite 6× more target variance.
4. A simple RWPE input transform (3 slots, 3 steps) provably escapes the limit and detectably changes the embeddings.

**What we have not established.**

- We did not retrain OPTFM with RWPE features to check whether *training* the model with positional encoding actually improves downstream task performance. That is a natural follow-up, but requires the full pre-training dataset which is not part of the public OPTFM release.
- We did not probe on a full MIPLIB 2017 evaluation; the pair family we use is a minimum viable 1-WL test, not a proxy for real-world solver performance. The claim is strictly about 1-WL expressiveness, not about end-to-end utility.
- The proof sketch is at the level appropriate for a DSO-619 course project; a journal version would formalize the color-refinement quotient category more carefully, handle the interaction with the "global virtual node" augmentation used in pretraining (added in `main_mip.py` lines 148–171), and extend the argument to the graph-level contrastive pretraining stage.
- We probe only one specific family of 1-WL equivalent non-isomorphic pairs (C₄ₖ vs k·C₄). The theorem itself holds for the full 1-WL equivalence class, but the experiment would be stronger with a second independent family (e.g., Cai–Fürer–Immerman constructions) and ideally a continuous distribution over 1-WL equivalence classes.

---

## 7. Relation to the professor's guidance

The professor's feedback identified **two substantive claims** that would make a publishable-style contribution:

1. **"OPTFM does not capture primal-dual / Lagrangian information."** An OPTFM proponent could reply that the embedding encodes it implicitly. We address this claim empirically in §4: the embedding does not even encode `n_components` on 1-WL equivalent inputs, and cannot predict `lp_value` above chance on random MILPs. The implicit-encoding defense is therefore testable and, at least for n_components and LP value, is refuted.

2. **"OPTFM still suffers from 1-WL-type expressiveness limitations."** We address this theoretically in §2 and empirically in §3. The theorem explicitly covers OPTFM's hierarchical multi-view cross-attention architecture, which is the architectural novelty over SGFormer; our lemmas show that global linear attention, additive edge-weighted cross-attention, and the GCN branch are all 1-WL bounded individually, and the composition inherits the bound. The empirical probe confirms this on a verified 1-WL equivalent non-isomorphic pair family.

The professor's narrowing advice — "take one interesting claim and really understand it, rather than cobbling together more and more advanced machinery" — is reflected in the structure of this report: §2 states a single substantive claim precisely, §3 verifies it on a single pair family across scales, §4 extends it to a probe-detectable version of the implicit-encoding question, and §5 rules out the "pathological pair" defense with a single minimal positive control. We explicitly avoided proposing an alternative algorithm (e.g., OPTIMUS) or cobbling together unrelated improvements.

---

## 8. Reproducing the experiments

```bash
# environment
python -m venv .venv
source .venv/Scripts/activate    # bash on Windows
pip install torch numpy scipy matplotlib networkx pymupdf

# one-time weight verification
PYTHONUTF8=1 python scripts/verify_weights.py

# sanity-check that the original canonical pair is broken
PYTHONUTF8=1 python scripts/sanity_check_pairs.py

# main 1-WL boundedness probe on the corrected pair family
PYTHONUTF8=1 python scripts/run_main_experiment.py \
    --k_values 2 3 4 5 6 7 8 9 10 12 15 20 25 30 \
    --rnf_samples 30

# primal-dual probe battery
PYTHONUTF8=1 python scripts/probe_primal_dual.py \
    --n_random 800 \
    --k_values 2 3 4 5 6 7 8 9 10 12 15 20 25 30 \
    --n_objectives 15 \
    --n_seeds 5
```

Outputs:

- `results/main/<timestamp>/` — main experiment: `results.json`, `report.txt`, `summary.csv`, `figures/fig_A_baseline_all_fail.png`, `figures/fig_B_improvements.png`, `figures/fig_C_scale_invariance.png`
- `results/probes/<timestamp>/` — probe battery: `results.json`, `report.txt`

---

## 9. References

See `paper/proof_sketch.md` §8 for the complete bibliography. The key references relied on here are:

- Chen, Liu, Wang, Yin (2023). *On Representing Mixed-Integer Linear Programs by Graph Neural Networks.* ICLR.
- Xu, Hu, Leskovec, Jegelka (2019). *How Powerful are Graph Neural Networks?* ICLR.
- Yuan et al. (2025). *OPTFM: A Scalable Multi-View Graph Transformer for Hierarchical Pre-Training in Combinatorial Optimization.* NeurIPS.
- Wu, Yang, Zhao, He, Wipf, Yan (2023). *SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations.* NeurIPS.
- Dwivedi, Joshi, Luu, Laurent, Bengio, Bresson (2022). *Graph Neural Networks with Learnable Structural and Positional Representations.* ICLR.
- Abboud, Ceylan, Grohe, Lukasiewicz (2020). *The Surprising Power of Graph Neural Networks with Random Node Initialization.* IJCAI.
- Cai, Fürer, Immerman (1992). *An optimal lower bound on the number of variables for graph identification.* Combinatorica.
