#!/usr/bin/env python3
"""Fine-tune a 2-class classifier head on top of the hierarchical OPTFM
to distinguish G_A from G_B on the 15-pair 1-WL-equivalent population.

The 1-WL boundedness theorem predicts that for baseline (uniform) inputs
the pooled graph embeddings are bit-identical between G_A and G_B.
A 2-class classifier head therefore produces identical logits for each
pair member, and the cross-entropy loss is PINNED at ln(2) ≈ 0.693
regardless of how long we train or how we initialize. No optimizer can
escape this — the gradient flow is identical for both classes, so head
updates affect them symmetrically.

Positive control: feeding RWPE(4,6,8)-augmented inputs (strictly more
expressive than 1-WL) breaks the pooled-embedding identity, so the
classifier CAN reduce loss below ln(2).

Observing loss(baseline) = ln(2) and loss(RWPE) < ln(2) simultaneously
is the most unambiguous empirical statement of the theorem available:
training cannot overcome an architectural bound. Conversely, if
loss(baseline) drops below ln(2), the theorem is wrong.

This script is deterministic (fixed seeds) and finishes in a few minutes
on CPU. Outputs JSON with per-epoch loss and a small summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.milp_pairs import milp_to_tensors
from data.milp_pairs_v2 import (
    construct_bipartite_cycle_pair, construct_cubic_bipartite_pair,
)
from models.optfm_hierarchical import HierarchicalOPTFM
from scripts.improvements import TRANSFORMS


K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]
LN2 = float(np.log(2.0))


class PairClassifier(nn.Module):
    """Hierarchical OPTFM backbone + 2-class linear head on mean-pool.

    The backbone is trained jointly with the head; all weights are updated.
    If the 1-WL boundedness theorem holds, the pooled embedding is
    identical for G_A and G_B on baseline inputs, so the logits are
    identical and the loss cannot drop below ln(2).
    """

    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        self.backbone = HierarchicalOPTFM(hidden_channels=hidden_channels)
        self.head = nn.Linear(hidden_channels, 2)

    def forward(self, cons_x, edge_index, edge_attr, var_x):
        z = self.backbone.get_graph_embedding(
            cons_x, edge_index, edge_attr, var_x, pooling="mean")
        return self.head(z)


def build_dataset(transform_name: str, seed: int = 0):
    """Return a list of (cons_x, edge_index, edge_attr, var_x, y) tuples
    covering all 15 1-WL equivalent pairs after applying `transform_name`.
    """
    transform = TRANSFORMS[transform_name]
    examples = []
    for k in K_VALUES:
        pair = construct_bipartite_cycle_pair(k)
        milp_a = transform(pair.milp_a, seed)
        milp_b = transform(pair.milp_b, seed)
        examples.append((*milp_to_tensors(milp_a), 0))
        examples.append((*milp_to_tensors(milp_b), 1))
    pair_cubic = construct_cubic_bipartite_pair()
    milp_a = transform(pair_cubic.milp_a, seed)
    milp_b = transform(pair_cubic.milp_b, seed)
    examples.append((*milp_to_tensors(milp_a), 0))
    examples.append((*milp_to_tensors(milp_b), 1))
    return examples


def check_pair_identity(model: PairClassifier, dataset, atol: float = 1e-5):
    """Check whether pooled embeddings of G_A and G_B are identical
    (within atol) for every pair in the dataset.
    """
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(dataset), 2):
            cx_a, ei_a, ea_a, vx_a, y_a = dataset[i]
            cx_b, ei_b, ea_b, vx_b, y_b = dataset[i + 1]
            assert y_a == 0 and y_b == 1
            z_a = model.backbone.get_graph_embedding(cx_a, ei_a, ea_a, vx_a)
            z_b = model.backbone.get_graph_embedding(cx_b, ei_b, ea_b, vx_b)
            cos = F.cosine_similarity(z_a.unsqueeze(0), z_b.unsqueeze(0)).item()
            exact = torch.allclose(z_a, z_b, atol=atol)
            results.append({"cos": cos, "exact": exact})
    return results


def train_one(transform_name: str, n_epochs: int, lr: float,
              hidden: int, seed: int, verbose: bool = False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PairClassifier(hidden_channels=hidden)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = build_dataset(transform_name, seed=seed)
    labels = torch.tensor([ex[-1] for ex in dataset], dtype=torch.long)

    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = []
        for cx, ei, ea, vx, _ in dataset:
            logits.append(model(cx, ei, ea, vx))
        logits = torch.stack(logits)  # (N, 2)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f"  [{transform_name:>20s}  seed={seed}]  epoch {epoch:4d}  loss {loss.item():.6f}")

    # Final identity check: are pooled embeddings still bit-identical after training?
    identity = check_pair_identity(model, dataset)
    return {
        "transform": transform_name,
        "seed":      seed,
        "losses":    losses,
        "identity":  identity,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--transforms", nargs="+",
                    default=["baseline", "rwpe_steps_4_6_8"])
    ap.add_argument("--output_dir", default="results/finetune")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir) / stamp
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINE-TUNING TEST: can training overcome the 1-WL bound?")
    print("=" * 70)
    print(f"  ln(2) plateau = {LN2:.6f}")
    print(f"  15 pairs, 30 training examples, {args.epochs} epochs, lr={args.lr}")
    print(f"  seeds: {args.seeds}")
    print(f"  transforms: {args.transforms}")
    print(f"  output: {out}")
    print()

    runs = []
    for transform in args.transforms:
        for seed in args.seeds:
            print(f"--- {transform}  seed={seed} ---")
            runs.append(train_one(transform, args.epochs, args.lr,
                                  args.hidden, seed, verbose=True))

    # Summary table
    print()
    print("=" * 70)
    print(f"FINAL LOSSES (ln 2 = {LN2:.4f}):")
    print("=" * 70)
    print(f"{'transform':<25} {'seed':>6} {'loss_init':>11} {'loss_final':>12} {'frac_pairs_identical_after':>30}")
    for r in runs:
        exact_frac = float(np.mean([i["exact"] for i in r["identity"]]))
        print(f"{r['transform']:<25} {r['seed']:>6} {r['losses'][0]:>11.6f} {r['losses'][-1]:>12.6f} {exact_frac:>30.3f}")

    # Write results JSON
    with open(out / "results.json", "w") as f:
        json.dump({
            "config": vars(args),
            "ln2":    LN2,
            "runs":   runs,
        }, f, indent=2)
    print(f"\nResults: {out / 'results.json'}")


if __name__ == "__main__":
    main()
