"""
Shared utilities for the ICDM experiment scripts.

Centralizes the embedding / similarity / bootstrap / model-registry helpers
that several scripts need, so the new experiments (capacity sweep, specificity
control, pooling ablation, encoding comparison, RWPE sweep, generality grid,
identity audit, …) stay consistent with the original run_main_experiment.py.

Conventions
-----------
* Bit-identity criterion (claude.md Rule 4): ``||Phi(G_A) - Phi(G_B)||_inf <= 1e-5``
  evaluated in the model's working dtype.  ``linf`` returns the raw number so a
  cell can be flagged when cos == 1.0 but the L-inf gap exceeds the threshold.
* Bootstrap: 2,000 resamples, 95% CI, seeded for determinism.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from data.milp_pairs import milp_to_tensors
from models.sgformer_mip import create_model
from models.optfm_hierarchical import create_hierarchical, HierarchicalOPTFM

CKPT = "D:/GitHub/OPTFM/node_pretrain/Models_SCIP/model_params_epoch_31.pth"
BIT_IDENTITY_ATOL = 1e-5


# ---------------------------------------------------------------------------
# Embedding / similarity helpers
# ---------------------------------------------------------------------------

def embed(model, milp, pooling: str = "mean", dtype: torch.dtype = torch.float32):
    """Pooled graph embedding of a MILPInstance under `model`."""
    cons_x, ei, ea, var_x = milp_to_tensors(milp)
    if dtype == torch.float64:
        cons_x, ea, var_x = cons_x.double(), ea.double(), var_x.double()
    with torch.no_grad():
        return model.get_graph_embedding(cons_x, ei, ea, var_x, pooling=pooling)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def linf(a: torch.Tensor, b: torch.Tensor) -> float:
    """L-infinity distance between two embeddings (the bit-identity metric)."""
    return float((a - b).abs().max().item())


def exact_eq(a: torch.Tensor, b: torch.Tensor, atol: float = BIT_IDENTITY_ATOL) -> bool:
    return bool(linf(a, b) <= atol)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[idx].mean(axis=1)
    return (float(values.mean()),
            float(np.quantile(boot, alpha / 2)),
            float(np.quantile(boot, 1 - alpha / 2)))


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def build_models(seed: int | None = None, dtype: torch.dtype = torch.float32,
                 include_pretrained: bool = True):
    """The canonical 6-architecture registry used by the main experiment."""
    if seed is not None:
        torch.manual_seed(seed)
    models = {}
    if include_pretrained:
        models["SGFormer+GCN (pretrained)"] = create_model("optfm", pretrained_path=CKPT)
    models.update({
        "SGFormer+GCN (random)":       create_model("random"),
        "TransConv only (random)":     create_model("transconv_only"),
        "GNN only (random)":           create_model("gnn_only"),
        "Simple GCN (random)":         create_model("gcn"),
        "Hierarchical OPTFM (random)": create_hierarchical(),
    })
    for m in models.values():
        m.eval()
        if dtype == torch.float64:
            m.double()
    return models


def build_hierarchical(hidden_channels: int = 16, trans_num_layers: int = 1,
                       trans_num_heads: int = 1, seed: int | None = None,
                       dtype: torch.dtype = torch.float32) -> HierarchicalOPTFM:
    """A hierarchical OPTFM at an arbitrary capacity (for the A6 sweep)."""
    if seed is not None:
        torch.manual_seed(seed)
    m = HierarchicalOPTFM(hidden_channels=hidden_channels,
                          trans_num_layers=trans_num_layers,
                          trans_num_heads=trans_num_heads)
    m.eval()
    if dtype == torch.float64:
        m.double()
    return m


def n_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(path: Path, header: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)]
    for r in rows:
        cells = []
        for c in r:
            if isinstance(c, float):
                cells.append(f"{c:.8g}")
            else:
                s = str(c)
                cells.append(f'"{s}"' if ("," in s or " " in s) else s)
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")
    return path
