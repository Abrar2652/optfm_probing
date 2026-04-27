#!/usr/bin/env python3
"""
Quick probe: does the pre-trained SGFormer checkpoint map the proper 1-WL
equivalent non-isomorphic pairs (C_{4k} vs k*C_4) to identical embeddings?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F

from scripts._tee_log import start_logging
from models.sgformer_mip import create_model
from data.milp_pairs_v2 import construct_bipartite_cycle_pair, diagnose_pair
from data.milp_pairs import milp_to_tensors

CKPT = "D:/GitHub/OPTFM/node_pretrain/Models_SCIP/model_params_epoch_31.pth"


def embed(model, milp, pooling="mean"):
    c_x, ei, ea, v_x = milp_to_tensors(milp)
    with torch.no_grad():
        return model.get_graph_embedding(c_x, ei, ea, v_x, pooling=pooling)


def main():
    start_logging("probe_v2_quick")
    model_pre = create_model("optfm", pretrained_path=CKPT)
    model_rnd = create_model("random")
    model_pre.eval()
    model_rnd.eval()

    print(f"{'k':>3} {'n':>4} {'||e_A|| pre':>12} {'||e_B|| pre':>12} "
          f"{'cos_pre':>10} {'exact_pre':>10} "
          f"{'cos_rnd':>10} {'exact_rnd':>10} {'iso?':>6}")
    print("-" * 98)

    for k in [2, 3, 4, 5, 8, 10, 15, 20]:
        pair = construct_bipartite_cycle_pair(k)
        diag = diagnose_pair(pair)
        ea_pre = embed(model_pre, pair.milp_a)
        eb_pre = embed(model_pre, pair.milp_b)
        ea_rnd = embed(model_rnd, pair.milp_a)
        eb_rnd = embed(model_rnd, pair.milp_b)

        cos_pre = F.cosine_similarity(ea_pre.unsqueeze(0), eb_pre.unsqueeze(0)).item()
        cos_rnd = F.cosine_similarity(ea_rnd.unsqueeze(0), eb_rnd.unsqueeze(0)).item()
        exact_pre = torch.allclose(ea_pre, eb_pre, atol=1e-6)
        exact_rnd = torch.allclose(ea_rnd, eb_rnd, atol=1e-6)

        print(f"{k:>3} {2*k:>4} {ea_pre.norm().item():>12.4f} {eb_pre.norm().item():>12.4f} "
              f"{cos_pre:>10.6f} {str(bool(exact_pre)):>10} "
              f"{cos_rnd:>10.6f} {str(bool(exact_rnd)):>10} "
              f"{str(diag['is_isomorphic']):>6}")


if __name__ == "__main__":
    main()
