# 1-WL Boundedness of OPTFM: Proof Sketch

**Project:** 1-WL expressiveness study for OPTFM (DSO-619, Spring 2026)
**Status:** draft — 1.5 page proof sketch supporting the empirical probe

---

## 1. Setup

### 1.1 Bipartite MILP as a feature-labeled graph

A MILP
$$\min_x \{\,c^\top x \;:\; Ax \lhd b,\; l \le x \le u,\; x \in \mathbb{Z}^p \times \mathbb{R}^{n-p}\,\}$$
is represented by the bipartite graph $G = (C \cup V, E)$ with $C = \{c_1,\ldots,c_m\}$ constraint nodes, $V = \{v_1,\ldots,v_n\}$ variable nodes, and edges $(c_i, v_j) \in E$ iff $A_{ij} \ne 0$. Each node carries a feature vector ($f_C(c_i) \in \mathbb{R}^{d_c}$, $f_V(v_j) \in \mathbb{R}^{d_v}$) and each edge carries a scalar weight $A_{ij}$. The OPTFM checkpoint uses Ecole's MilpBipartite features: $d_v = 9$ per-variable and $d_c = 1$ (normalized right-hand side) per constraint.

### 1.2 1-WL color refinement on bipartite MILPs

Let $\chi^{(0)}(u) = f(u)$ for every node $u \in C \cup V$. Iterate
$$\chi^{(t+1)}(u) \;=\; \mathrm{HASH}\Big(\chi^{(t)}(u),\; \{\!\!\{\,(\chi^{(t)}(w), A_{uw}) : w \sim u\,\}\!\!\}\Big)$$
where $\{\!\!\{\cdot\}\!\!\}$ denotes a multiset. Two graphs $G, G'$ are *1-WL equivalent* if for every $t \ge 0$ the multiset $\{\!\!\{\chi^{(t)}(u) : u \in G\}\!\!\}$ equals $\{\!\!\{\chi^{(t)}(u') : u' \in G'\}\!\!\}$. This equivalence is coarser than graph isomorphism: there exist non-isomorphic graphs whose 1-WL histograms coincide at every depth.

### 1.3 The OPTFM encoder

We consider the full hierarchical OPTFM from `OPTFM/node_pretrain/ours_crossattention_improve.py` (the `SGFormer_MIP_hierarchical_improve` class). The encoder maps $(C, V, A, f_C, f_V)$ to graph-level embedding $z \in \mathbb{R}^h$ via

$$
\begin{aligned}
&\text{(input)} && \tilde v_j = W_v f_V(v_j),\; \tilde c_i = W_c f_C(c_i)\\
&\text{(self attention)} && \tilde v \leftarrow \text{TransConvSelf}(\tilde v),\quad \tilde c \leftarrow \text{TransConvSelf}(\tilde c)\\
&\text{(cross C$\to$V)} && \tilde v \leftarrow \text{TransConvCross}(\tilde v, \tilde c, A^\top)\\
&\text{(cross V$\to$C)} && \tilde c \leftarrow \text{TransConvCross}(\tilde c, \tilde v, A)\\
&\text{(GCN branch)} && (\tilde v^{(g)}, \tilde c^{(g)}) = \text{GNNPolicy}(\tilde c_{\text{emb}}, \text{edges}, \tilde v_{\text{emb}})\\
&\text{(fuse)} && \tilde v \leftarrow \alpha \tilde v^{(g)} + (1-\alpha) \tilde v,\quad \tilde c \leftarrow \alpha \tilde c^{(g)} + (1-\alpha) \tilde c\\
&\text{(pool)} && z = \text{POOL}(\tilde c \;\Vert\; \tilde v),\; \text{POOL} \in \{\text{mean}, \text{sum}, \text{max}\}.
\end{aligned}
$$

where `TransConvSelf` is SGFormer-style linear attention (Wu et al., 2023, Eq. 2–3) and `TransConvCross` is OPTFM's cross-attention with an additive edge-weighted message $A_{ij} \tilde c_i$ concatenated to the cross-attention output and then linearly projected. The GCN branch `GNNPolicy` is Gasse et al.'s bipartite message-passing GNN.

---

## 2. The claim

**Theorem (OPTFM is 1-WL bounded).** *Fix any weights of the hierarchical OPTFM encoder. Let $(G_1, f_1)$ and $(G_2, f_2)$ be two feature-labeled bipartite MILPs that are 1-WL equivalent under the color refinement in §1.2. Then the pooled graph embedding is identical: $z(G_1, f_1) = z(G_2, f_2)$.*

**Corollary.** The function $z$ factors through the 1-WL equivalence class: it cannot be used to distinguish non-isomorphic graphs that nevertheless have the same 1-WL color histogram.

This is the MILP analogue of the GNN 1-WL bound established by Xu et al. (2019) for MPNNs and extended by Chen et al. (2023) to bipartite MILPs. The new content here is that **OPTFM's combination of intra-type self-attention, inter-type cross-attention with edge-weighted aggregation, and a bipartite GCN branch still does not escape the bound**, even though the paper motivates the architecture as going beyond "the local interaction structure of standard GNNs".

---

## 3. Lemmas

All lemmas are stated for fixed weights. We say an operator $\Phi$ is **1-WL bounded** if whenever two node sets $X, X'$ agree as multisets of refined colors, the multiset $\{\!\!\{\Phi(u): u \in X\}\!\!\}$ equals $\{\!\!\{\Phi(u'): u' \in X'\}\!\!\}$.

### Lemma 1 (Linear attention is a multiset function of its source)

Let $Q, K, V \in \mathbb{R}^{N \times d}$ arise from a node-wise linear map. SGFormer's normalized linear attention computes, for each query index $i$,
$$
\text{attn}(i) \;=\; D_i^{-1}\!\left(\frac{1}{N}\, q_i \!\left(\sum_{\ell=1}^{N} k_\ell^\top v_\ell\right) + v_i\right), \qquad
D_i = 1 + \frac{1}{N} q_i\!\left(\sum_{\ell=1}^{N} k_\ell^\top \mathbf{1}\right).
$$

The terms $\sum_\ell k_\ell^\top v_\ell$ and $\sum_\ell k_\ell^\top \mathbf{1}$ are commutative sums, i.e. functions of the multiset $\{\!\!\{(k_\ell, v_\ell)\}\!\!\}$. The output at position $i$ depends only on (a) $(q_i, v_i)$ at that position and (b) the multiset of source $(k_\ell, v_\ell)$ pairs. The Frobenius-norm normalization $q_i \mathrel{/}\mathrel{=} \|Q\|_F$ depends only on $\|Q\|_F = (\sum_j \|q_j\|_2^2)^{1/2}$, which is a multiset function of queries.

**Consequence.** If two attention inputs have the same multiset of source pairs (and the same value at position $i$), then $\text{attn}(i)$ is the same at both. Linear attention is therefore 1-WL bounded within the query set.

### Lemma 2 (TransConvSelf and TransConvCross preserve 1-WL equivalence)

`TransConvSelf` stacks per-node affine, LayerNorm, ReLU, dropout (inactive at eval time), residual, and a linear attention layer. The per-node operations are functions of the single-node feature, so they commute with multiset equivalence. Linear attention is 1-WL bounded by Lemma 1. Composition of 1-WL bounded maps is 1-WL bounded.

`TransConvCross` adds one edge-weighted message
$$
m_i \;=\; \sum_{j} A_{ij}\, y_j,
$$
where $y_j$ is the source embedding. Two 1-WL equivalent nodes $i \in G_1, i' \in G_2$ have, by definition, a bijection between their edge-weighted neighborhoods preserving $(A_{ij}, \chi^{(t)}(j))$ pairs. Since the sum is a multiset function of these pairs, $m_i = m_{i'}$. The subsequent concatenation with the linear-attention output and the final linear projection are per-node, so they preserve equivalence.

### Lemma 3 (The bipartite GCN branch is 1-WL bounded)

The GNNPolicy branch of OPTFM is a standard bipartite Gasse-style MPNN: it computes
$$
\text{msg}(c_i, v_j) = \phi\big(\tilde c_i, A_{ij}, \tilde v_j\big),\qquad
\tilde c_i \leftarrow \psi\!\left(\tilde c_i,\; \sum_{j \sim i} \text{msg}(c_i, v_j)\right),
$$
and symmetrically for $V$. This is exactly an MPNN, for which Xu et al. (2019) prove that no MPNN is strictly more expressive than 1-WL. Chen et al. (2023, Theorem 1) extend this to bipartite MILPs: two 1-WL equivalent MILPs are mapped to identical node-feature multisets by any MPNN.

### Lemma 4 (Convex combination and pooling)

For $\alpha \in [0,1]$, the fuse step $\tilde v \mapsto \alpha \tilde v^{(g)} + (1-\alpha) \tilde v$ is per-node, hence 1-WL bounded (it acts on aligned node multisets). Mean, sum, and max pooling over a node set are symmetric functions of their input multiset; two graphs whose node-feature multisets agree have equal pooled outputs.

### Lemma 5 (Virtual global nodes preserve 1-WL equivalence)

OPTFM's pretraining pipeline (see `OPTFM/node_pretrain/main_mip.py` lines 148–171) appends two artificial global nodes to every subgraph before training: a constraint-side global node whose feature vector is the mean of the existing constraint features, and a variable-side global node whose feature vector is the mean of the existing variable features. The global constraint is connected to every original variable with unit edge weight, and symmetrically the global variable is connected to every original constraint. This modification is applied identically to every input, which is why the paper motivates it as providing "global variable and constraint representations."

**Claim.** *Applying the virtual-global-node transform to two 1-WL equivalent feature-labeled bipartite MILPs yields two feature-labeled bipartite MILPs that are still 1-WL equivalent.*

**Proof.** Let $G_1, G_2$ be 1-WL equivalent. By definition, at every iteration $t$, the multisets of refined colors agree:
$$\{\!\!\{\chi^{(t)}(u) : u \in G_1\}\!\!\} = \{\!\!\{\chi^{(t)}(u) : u \in G_2\}\!\!\}.$$
Split these multisets by node side:
$$\{\!\!\{\chi^{(t)}(c) : c \in C_1\}\!\!\} = \{\!\!\{\chi^{(t)}(c) : c \in C_2\}\!\!\}, \qquad \{\!\!\{\chi^{(t)}(v) : v \in V_1\}\!\!\} = \{\!\!\{\chi^{(t)}(v) : v \in V_2\}\!\!\}$$
(the side labels are themselves 1-WL-stable). The global-node features are
$$f_C^{\text{global}} = \frac{1}{|C|} \sum_{c \in C} f_C(c), \qquad f_V^{\text{global}} = \frac{1}{|V|} \sum_{v \in V} f_V(v),$$
which are symmetric functions of the respective feature multisets and are therefore equal across $G_1, G_2$. The edge pattern introduced by each global node is fixed: the global constraint is adjacent to every variable, and the global variable is adjacent to every constraint. This pattern depends only on the node counts $(|C|, |V|)$, which are invariants of 1-WL equivalence, and not on the identity of individual nodes.

Run color refinement on the augmented graphs. At iteration $t+1$, each *original* node's refined color becomes
$$\chi^{(t+1)}(u) = \mathrm{HASH}\Big(\chi^{(t)}(u),\;\{\!\!\{(\chi^{(t)}(w), A_{uw}) : w \sim u\}\!\!\} \cup \{(\chi^{(t)}(\text{global}), 1)\}\Big),$$
where the global-node color is a symmetric function of the existing feature multiset and is the same in $G_1$ and $G_2$. Since the new term is added uniformly to every original node's neighborhood, and the multisets it acts on were already equal, the refined multisets remain equal. The global nodes themselves are in the same 1-WL class in $G_1$ and $G_2$ because they see the entire (multiset-equal) opposite-side neighborhood. Hence the augmented graphs are 1-WL equivalent at every depth. $\blacksquare$

**Consequence.** The theorem holds *including* the virtual-global-node augmentation used in OPTFM's pretraining. An OPTFM proponent cannot reply "the actual pretrained pipeline escapes your bound because it uses global nodes." It does not.

**Empirical verification.** The `virtual_global_node` transform in `scripts/improvements.py` appends these nodes to every MILP instance before forward propagation. Running the main experiment with this transform on all 15 (cycle + cubic) 1-WL equivalent non-isomorphic pairs yields `cos_sim = 1.000000` and `exact_frac = 1.00` across every architecture (see `results/main/<timestamp>/report.txt`, row `transform = virtual_global_node`). This matches the lemma.

---

## 4. Proof of the theorem

Let $G_1, G_2$ be 1-WL equivalent feature-labeled bipartite MILPs. Apply the encoder to both, tracking feature multisets at each stage.

1. **(Optional) virtual-global-node augmentation.** If the pretraining pipeline is in use, augment each input with the global constraint and global variable nodes described in §3 / Lemma 5. By Lemma 5 the augmented inputs are still 1-WL equivalent.
2. **Input embedding.** $\tilde v = W_v f_V$ and $\tilde c = W_c f_C$ are per-node linear maps, so they act on the respective multisets. Equivalence is preserved.
3. **Self-attention.** By Lemma 2, applying `TransConvSelf` separately to $\tilde v$ and $\tilde c$ preserves multiset equivalence of both sets.
4. **Cross-attention with edge-weighted aggregation.** By Lemma 2, `TransConvCross` preserves equivalence: linear attention is a multiset function of its source, and the dense edge-weighted message $A y$ is determined by the multiset of $(A_{ij}, y_j)$ pairs around each node, which is exactly what 1-WL equivalence provides.
5. **GCN branch.** By Lemma 3, `GNNPolicy` is 1-WL bounded.
6. **Fuse.** The linear combination in step 5 is per-node; it acts on aligned node multisets that are already equal. By Lemma 4 it preserves equivalence.
7. **Pool.** The pooled output is a symmetric function of the final node-feature multiset; by Lemma 4 it is identical for the two graphs.

Hence $z(G_1, f_1) = z(G_2, f_2)$. $\blacksquare$

---

## 5. What the theorem does and does not say

The theorem says: *no amount of training of the OPTFM hierarchical architecture can produce a graph-level embedding that distinguishes two 1-WL equivalent MILPs.* This holds for any weights and any pooling in $\{$mean, sum, max$\}$.

The theorem does **not** say:
- OPTFM cannot distinguish non-equivalent MILPs. It can (and does) distinguish anything that 1-WL itself distinguishes. The probe results confirm this: on random bipartite MILPs the model separates them cleanly.
- Primal-dual information cannot be encoded *implicitly*. It cannot for 1-WL equivalent pairs (where the target varies but the embedding is constant), but for the generic case (where targets and 1-WL histograms co-vary) implicit encoding is still possible. Our probe battery tests this separately.
- The bound cannot be escaped. Injecting random node features (Abboud et al. 2020) or structural positional encodings (Dwivedi et al. 2022) provably escapes 1-WL because the initial colors become unique per node. Section 7 of the accompanying report reports empirical confirmation: with RWPE at steps 4, 6, 8 the pooled cosine similarity on C\_{4k} vs k·C\_4 drops below 1.0 for every model variant.

---

## 6. Connection to the Chen et al. (2023) / Joshi et al. (2024) results

Chen et al. (2023) prove that no MPNN can distinguish 1-WL equivalent MILPs, and give a specific construction where feasibility is undecidable from a 1-WL-bounded encoder. Joshi et al. (2024) port the argument to bipartite TSP and show GNNs' 1-WL bound persists there. Our contribution is to extend the argument to an architecture that was explicitly proposed to escape the "local interaction structure of standard GNNs" by adding full global attention. We show that:

1. Global linear attention alone is not enough: Lemma 1 makes it a multiset function of the entire node set, so it respects 1-WL equivalence.
2. Adding a bipartite GCN branch cannot help: Lemma 3 quotes the standard MPNN 1-WL bound.
3. The cross-attention with edge-weighted aggregation, which is the architectural novelty of OPTFM over SGFormer, is also 1-WL bounded because its edge-weighted sum is again a multiset function (Lemma 2).

The result is therefore robust to the specific attention mechanism used — any attention that aggregates over the source via a symmetric sum is 1-WL bounded, regardless of the query-side weighting.

---

## 7. Empirical verification

Empirically we verify the theorem on **two structurally independent families** of 1-WL equivalent non-isomorphic bipartite MILP pairs:

- **Family I — 2-regular bipartite (cycles vs disjoint 4-cycles).** $G_A(k) = C_{4k}$-bipartite (a single $4k$-cycle) and $G_B(k) = k\cdot C_4$-bipartite ($k$ disjoint 4-cycles), both with $2k$ constraints and $2k$ variables, all constraint RHS set to $1$. Both graphs are 2-regular on both sides. Their 1-WL color histograms match at every iteration because the regular-graph color refinement stabilizes at a single color class. They are non-isomorphic because connectivity differs (1 component vs. $k$ components), an invariant 1-WL cannot detect. Verified programmatically via brute-force permutation (for small $k$) and NetworkX VF2 feature-labeled bipartite isomorphism check. We probe $k \in \{2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30\}$.

- **Family II — 3-regular bipartite (connected vs $2 \cdot K_{3,3}$).** $G_A$ is the connected 3-regular bipartite graph on $6+6$ vertices constructed from the circulant offsets $\{0, 1, 3\} \pmod 6$ (each constraint $i$ is adjacent to variables $i, i+1, i+3 \pmod 6$). $G_B = K_{3,3} \sqcup K_{3,3}$ is two disjoint copies of $K_{3,3}$. Both graphs have 12 vertices, 18 edges, every node of degree exactly 3, so the standard 1-WL color refinement collapses both to a single color class after the first iteration. They are non-isomorphic because $G_A$ has 1 component and $G_B$ has 2. This family is *structurally* independent of family I: it differs in regularity, in local girth distribution, and in the specific mechanism by which the 1-WL bound is saturated (uniform-regularity collapse rather than cycle-length collapse).

- **Models.** We probe six architectures: the pre-trained SGFormer+GCN baseline (loaded from `Models_SCIP/model_params_epoch_31.pth`), a random-init copy of the same, random-init TransConv-only, random-init GNN-only, a 2-layer GCN baseline, and a random-init full hierarchical OPTFM (the multi-view cross-attention architecture of `ours_crossattention_improve.py`, dense-ported to avoid torch-sparse).

- **Result.** For every one of the 15 pairs (14 from family I + 1 from family II) and every one of the six models, the cosine similarity between the graph-level embeddings is exactly $1.000000$ and the embeddings are bit-identical ($\text{exact\_frac} = 1.00$ over 15 pairs). This is consistent with the theorem: no permutation-equivariant linear-attention + MPNN architecture can escape the 1-WL bound, regardless of weights, scale, or regularity class.

- **Virtual global nodes (empirical Lemma 5 check).** Applying the `virtual_global_node` transform from `scripts/improvements.py` — which appends OPTFM's mean-pooled virtual constraint and variable nodes to every input, exactly as done in the pretraining pipeline — leaves $\text{cos\_sim} = 1.000000$ and $\text{exact\_frac} = 1.00$ across every model and every pair. The 1-WL bound persists under the augmentation, matching Lemma 5.

- **Falsifiability.** Injecting Random Walk Positional Encodings at steps 4, 6, 8 (known to be $>1$-WL, Dwivedi et al. 2022) into the three zero-valued Ecole variable feature slots causes the cosine similarity to drop below 1.0 across all models and both families, with substantially larger drops on family II (e.g., the random-init hierarchical OPTFM drops to $\text{cos\_sim} \approx 0.919$ across the combined 15-pair set, vs. $\approx 0.9986$ on the cycle-only run). This confirms that the test is genuinely falsifiable — a model that truly escapes 1-WL *would* be detected by this probe — and that the signal is strongest precisely on the second (cubic) family, which a reviewer could have otherwise suspected of hiding behind family I's particular structure.

---

## 8. References

- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful are Graph Neural Networks?* ICLR.
- Chen, Z., Liu, J., Wang, X., & Yin, W. (2023). *On Representing Mixed-Integer Linear Programs by Graph Neural Networks.* ICLR.
- Gasse, M., Chételat, D., Ferroni, N., Charlin, L., & Lodi, A. (2019). *Exact Combinatorial Optimization with Graph Convolutional Neural Networks.* NeurIPS.
- Wu, Q., Yang, C., Zhao, W., He, Y., Wipf, D., & Yan, J. (2023). *SGFormer: Simplifying and Empowering Transformers for Large-Graph Representations.* NeurIPS.
- Dwivedi, V. P., Joshi, C. K., Luu, A. T., Laurent, T., Bengio, Y., & Bresson, X. (2022). *Graph Neural Networks with Learnable Structural and Positional Representations.* ICLR.
- Abboud, R., Ceylan, İ. İ., Grohe, M., & Lukasiewicz, T. (2020). *The Surprising Power of Graph Neural Networks with Random Node Initialization.* IJCAI.
- Cai, J., Fürer, M., & Immerman, N. (1992). *An optimal lower bound on the number of variables for graph identification.* Combinatorica.
- Yuan, H. et al. (2025). *OPTFM: A Scalable Multi-View Graph Transformer for Hierarchical Pre-Training in Combinatorial Optimization.* NeurIPS.
