# Expressiveness Limits of Graph Foundation Models for MILPs

A study of the Weisfeiler–Leman (1-WL) expressiveness ceiling of graph
foundation models that encode mixed-integer linear programs (MILPs) as
constraint–variable bipartite graphs.

## Summary

Hierarchical multi-view graph transformers that combine global linear attention
with bipartite message passing are **provably 1-WL bounded** on bipartite MILPs:
for any weights, two 1-WL-equivalent (but non-isomorphic) MILPs are mapped to
**bit-identical** pooled embeddings. We provide a proof (one theorem, five
lemmas) and broad empirical confirmation across ten architecturally distinct
encoders and several MILP families.

Key findings (all reproducible from `results/*.csv`):

- **Architectural universality.** Every encoder maps every 1-WL-equivalent pair
  to a bit-identical embedding (`‖Φ(G_A) − Φ(G_B)‖∞ ≤ 5.96e-8` in float32,
  `≤ 5.6e-17` in float64).
- **Capacity- and scale-invariant.** Holds from 10K to 2.5M parameters and up to
  800-node graphs.
- **Readout consequence.** A frozen-embedding probe of a structural invariant
  collapses to chance on the 1-WL population despite larger target variance; a
  jointly trained classifier's loss is pinned exactly at `ln 2`.
- **The limit is in the encoding.** Random-walk positional encodings (RWPE)
  break these pairs — but are not a universal fix (we exhibit pairs RWPE also
  fails to separate).
- **A reusable diagnostic** lets you test your own MILP encoder for
  1-WL-blindness.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only; no GPU required. All experiments run in minutes.

## Reproduce

Run everything, or a single target:

```bash
./repro.sh all          # Linux/Mac        (repro.ps1 on Windows)
./repro.sh capacity     # one experiment
```

`RESULTS_INDEX.md` maps every table and figure to its script and output CSV.
Each script is seeded, so reruns reproduce the CSVs. Figures:

```bash
python -m visualization.generate_all --style ieee
```

## Diagnostic

Test any encoder exposing
`get_graph_embedding(cons_x, edge_index, edge_attr, var_x, pooling)`:

```python
from scripts.wl_diagnostic import diagnose, default_pairs
report = diagnose(my_encoder, default_pairs())
print(report["verdict"])   # "1-WL BOUNDED ..." or "NOT 1-WL bounded ..."
```

## Tests

```bash
python tests/run_tests.py
```

Checks that the test pairs are 1-WL equivalent, non-isomorphic, and have the
expected connectivity mismatch, and that encoders are permutation-equivariant,
deterministic, and separate genuinely distinguishable pairs.

## Repository layout

```
models/            encoder architectures (baseline, hierarchical, and
                   four independent encoders for the generality study)
data/              MILP pair constructions + standard-family generators
scripts/           experiments, transforms, and the diagnostic
visualization/     figure generation
tests/             pair-property and encoder-invariance tests
results/           generated CSVs and figures
```

## Notes

- A pretrained checkpoint is optional; 1-WL boundedness is an architectural
  property that holds for any weights, so random initialization suffices.
- `data/milp_pairs.py` contains a deprecated "canonical pair" kept only to
  document, via `scripts/broken_canonical.py`, why it is an invalid 1-WL probe
  (it is graph-isomorphic).
