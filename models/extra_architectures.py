"""
A1 — Independently-designed bipartite/MILP encoders that are NOT OPTFM ablations.

The point of these is to show the 1-WL bound is a property of *symmetric multiset
aggregation*, not an OPTFM-specific artifact. We implement four architecturally
distinct, published encoder families, each in its pure symmetric-aggregation form:

  GraphormerStyle      — Graphormer-style global multi-head SOFTMAX self-attention
                         with degree-centrality encoding and an adjacency
                         attention bias (Ying et al. 2021).
  GPSStyle             — GraphGPS hybrid: local bipartite message passing + global
                         softmax attention per layer (Rampasek et al. 2022).
  SetTransformerPool   — Set Transformer (SAB + PMA) pooling over node features,
                         a pure permutation-invariant set function (Lee et al. 2019).
  GasseBipartiteGCN    — the canonical bipartite GCN for MILPs (Gasse et al. 2019).

IMPORTANT scoping note. A *faithful* Graphormer also injects shortest-path-distance
spatial encodings, which are strictly more expressive than 1-WL and would (like
RWPE) distinguish C_{4k} from k*C_4. That is an ENCODING-level escape, fully
consistent with this paper's thesis ("the limitation lives in the encoding"). To
demonstrate the *architectural* bound we therefore use degree + adjacency bias
(both 1-WL-visible) and exclude SPD. The SPD variant is discussed in prose as an
encoding-level escape, not an architectural one.

Every model exposes get_graph_embedding(cons_x, edge_index, edge_attr, var_x,
pooling) so it drops into the existing probing harness.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _degrees(n_cons, n_vars, edge_index):
    N = n_cons + n_vars
    deg = torch.zeros(N, dtype=torch.long)
    c = edge_index[0]
    v = edge_index[1] + n_cons
    deg.index_add_(0, c, torch.ones_like(c))
    deg.index_add_(0, v, torch.ones_like(v))
    return deg


def _adj_mask(n_cons, n_vars, edge_index, dtype):
    N = n_cons + n_vars
    M = torch.zeros(N, N, dtype=dtype)
    c = edge_index[0]; v = edge_index[1] + n_cons
    M[c, v] = 1.0; M[v, c] = 1.0
    return M


def _pool(x, pooling):
    if pooling == "mean":
        return x.mean(0)
    if pooling == "sum":
        return x.sum(0)
    if pooling == "max":
        return x.max(0).values
    raise ValueError(pooling)


# ---------------------------------------------------------------------------
# Graphormer-style
# ---------------------------------------------------------------------------

class _GraphormerLayer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.h = heads
        self.dk = hidden // heads
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                nn.Linear(2 * hidden, hidden))
        self.adj_bias = nn.Parameter(torch.zeros(heads))
        self.nonadj_bias = nn.Parameter(torch.zeros(heads))

    def forward(self, x, adj):
        N = x.shape[0]
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(N, 3, self.h, self.dk).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # (heads, N, dk)
        scores = torch.einsum("hnd,hmd->hnm", q, k) / (self.dk ** 0.5)
        bias = adj[None] * self.adj_bias[:, None, None] + \
               (1 - adj)[None] * self.nonadj_bias[:, None, None]
        attn = torch.softmax(scores + bias, dim=-1)
        out = torch.einsum("hnm,hmd->hnd", attn, v).permute(1, 0, 2).reshape(N, -1)
        x = x + self.proj(out)
        x = x + self.ff(self.ln2(x))
        return x


class GraphormerStyle(nn.Module):
    def __init__(self, in_channels_var=9, in_channels_cons=1, hidden_channels=32,
                 heads=4, layers=2, max_deg=128, **kw):
        super().__init__()
        self.var_emb = nn.Linear(in_channels_var, hidden_channels)
        self.cons_emb = nn.Linear(in_channels_cons, hidden_channels)
        self.deg_emb = nn.Embedding(max_deg + 1, hidden_channels)
        self.layers = nn.ModuleList([_GraphormerLayer(hidden_channels, heads)
                                     for _ in range(layers)])
        self.max_deg = max_deg

    def get_graph_embedding(self, cons_x, edge_index, edge_attr, var_x, pooling="mean"):
        nc, nv = cons_x.shape[0], var_x.shape[0]
        h = torch.cat([self.cons_emb(cons_x), self.var_emb(var_x)], 0)
        deg = _degrees(nc, nv, edge_index).clamp(max=self.max_deg)
        h = h + self.deg_emb(deg)
        adj = _adj_mask(nc, nv, edge_index, h.dtype)
        for layer in self.layers:
            h = layer(h, adj)
        return _pool(h, pooling)


# ---------------------------------------------------------------------------
# GraphGPS-style (local MPNN + global attention)
# ---------------------------------------------------------------------------

class _GPSLayer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.lin_self = nn.Linear(hidden, hidden)
        self.lin_neigh = nn.Linear(hidden, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ln = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                nn.Linear(2 * hidden, hidden))

    def forward(self, x, adj_norm):
        local = self.lin_self(x) + self.lin_neigh(adj_norm @ x)     # symmetric mean agg
        glob, _ = self.attn(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        h = self.ln(local + glob.squeeze(0))
        return h + self.ff(h)


class GPSStyle(nn.Module):
    def __init__(self, in_channels_var=9, in_channels_cons=1, hidden_channels=32,
                 heads=4, layers=2, **kw):
        super().__init__()
        self.var_emb = nn.Linear(in_channels_var, hidden_channels)
        self.cons_emb = nn.Linear(in_channels_cons, hidden_channels)
        self.layers = nn.ModuleList([_GPSLayer(hidden_channels, heads)
                                     for _ in range(layers)])

    def get_graph_embedding(self, cons_x, edge_index, edge_attr, var_x, pooling="mean"):
        nc, nv = cons_x.shape[0], var_x.shape[0]
        h = torch.cat([self.cons_emb(cons_x), self.var_emb(var_x)], 0)
        adj = _adj_mask(nc, nv, edge_index, h.dtype)
        deg = adj.sum(1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg
        for layer in self.layers:
            h = layer(h, adj_norm)
        return _pool(h, pooling)


# ---------------------------------------------------------------------------
# Set Transformer pooling (pure set function over node features)
# ---------------------------------------------------------------------------

class _MAB(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, q, kv):
        a, _ = self.attn(q, kv, kv)
        h = self.ln(q + a)
        return h + self.ff(h)


class SetTransformerPool(nn.Module):
    def __init__(self, in_channels_var=9, in_channels_cons=1, hidden_channels=32,
                 heads=4, **kw):
        super().__init__()
        self.var_emb = nn.Linear(in_channels_var, hidden_channels)
        self.cons_emb = nn.Linear(in_channels_cons, hidden_channels)
        self.sab = _MAB(hidden_channels, heads)
        self.seed = nn.Parameter(torch.randn(1, 1, hidden_channels))
        self.pma = _MAB(hidden_channels, heads)

    def get_graph_embedding(self, cons_x, edge_index, edge_attr, var_x, pooling="mean"):
        h = torch.cat([self.cons_emb(cons_x), self.var_emb(var_x)], 0).unsqueeze(0)
        h = self.sab(h, h)                                  # set self-attention
        out = self.pma(self.seed, h)                        # pooling by attention
        return out.squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Gasse et al. 2019 bipartite GCN (clean reimplementation)
# ---------------------------------------------------------------------------

class _GasseConv(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * hidden + 1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.upd = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))

    def forward(self, src, dst, edge_index_sd, edge_attr, n_dst):
        s = src[edge_index_sd[0]]
        d = dst[edge_index_sd[1]]
        m = self.msg(torch.cat([s, d, edge_attr], -1))
        agg = torch.zeros(n_dst, m.shape[1], dtype=m.dtype, device=m.device)
        agg.index_add_(0, edge_index_sd[1], m)              # symmetric sum aggregation
        return self.upd(torch.cat([dst, agg], -1))


class GasseBipartiteGCN(nn.Module):
    def __init__(self, in_channels_var=9, in_channels_cons=1, hidden_channels=32,
                 layers=2, **kw):
        super().__init__()
        self.var_emb = nn.Linear(in_channels_var, hidden_channels)
        self.cons_emb = nn.Linear(in_channels_cons, hidden_channels)
        self.v2c = nn.ModuleList([_GasseConv(hidden_channels) for _ in range(layers)])
        self.c2v = nn.ModuleList([_GasseConv(hidden_channels) for _ in range(layers)])

    def get_graph_embedding(self, cons_x, edge_index, edge_attr, var_x, pooling="mean"):
        nc, nv = cons_x.shape[0], var_x.shape[0]
        c = self.cons_emb(cons_x); v = self.var_emb(var_x)
        ei_cv = edge_index                                  # cons -> var indices
        ei_vc = torch.stack([edge_index[1], edge_index[0]], 0)
        for v2c, c2v in zip(self.v2c, self.c2v):
            c = v2c(v, c, ei_vc, edge_attr, nc)             # vars send to cons
            v = c2v(c, v, ei_cv, edge_attr, nv)             # cons send to vars
        return _pool(torch.cat([c, v], 0), pooling)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_extra_models(seed: int | None = None, hidden_channels: int = 32,
                       dtype: torch.dtype = torch.float32):
    if seed is not None:
        torch.manual_seed(seed)
    models = {
        "Graphormer-style (deg+adj)": GraphormerStyle(hidden_channels=hidden_channels),
        "GraphGPS-style (MPNN+attn)": GPSStyle(hidden_channels=hidden_channels),
        "Set-Transformer pooling":    SetTransformerPool(hidden_channels=hidden_channels),
        "Gasse-2019 bipartite GCN":   GasseBipartiteGCN(hidden_channels=hidden_channels),
    }
    for m in models.values():
        m.eval()
        if dtype == torch.float64:
            m.double()
    return models


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from data.milp_pairs_v2 import construct_bipartite_cycle_pair
    from scripts._common import embed, cos_sim, linf, n_params
    p = construct_bipartite_cycle_pair(5)
    for name, m in build_extra_models(seed=0).items():
        a, b = embed(m, p.milp_a), embed(m, p.milp_b)
        print(f"{name:30s} cos={cos_sim(a,b):.6f} linf={linf(a,b):.2e} params={n_params(m):,}")
