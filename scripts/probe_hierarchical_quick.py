#!/usr/bin/env python3
"""
Quick probe: does the hierarchical OPTFM (full cross-attention, random init)
also map proper 1-WL equivalent non-isomorphic pairs to identical embeddings?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from scripts._tee_log import start_logging
from models.sgformer_mip import create_model
from models.optfm_hierarchical import create_hierarchical
from data.milp_pairs_v2 import construct_bipartite_cycle_pair
from data.milp_pairs import milp_to_tensors

CKPT = "D:/GitHub/OPTFM/node_pretrain/Models_SCIP/model_params_epoch_31.pth"


def embed(model, milp):
    c_x, ei, ea, v_x = milp_to_tensors(milp)
    with torch.no_grad():
        return model.get_graph_embedding(c_x, ei, ea, v_x)


def main():
    start_logging("probe_hierarchical_quick")
    models = {
        "SGFormer+GCN (pretrained)": create_model("optfm", pretrained_path=CKPT),
        "SGFormer+GCN (random)":     create_model("random"),
        "TransConv only (random)":   create_model("transconv_only"),
        "GNN only (random)":         create_model("gnn_only"),
        "Simple GCN (random)":       create_model("gcn"),
        "Hierarchical OPTFM (random)": create_hierarchical(),
    }
    for m in models.values():
        m.eval()

    ks = [2, 3, 5, 10, 20]

    print(f"\n{'model':<32}" + "".join(f" k={k:<6}" for k in ks))
    print("-" * (32 + 8 * len(ks)))
    for name, m in models.items():
        row = f"{name:<32}"
        for k in ks:
            pair = construct_bipartite_cycle_pair(k)
            ea = embed(m, pair.milp_a)
            eb = embed(m, pair.milp_b)
            cos = F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item()
            row += f" {cos:8.6f}"
        print(row)


if __name__ == "__main__":
    main()
