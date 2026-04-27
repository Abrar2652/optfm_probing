"""
Full hierarchical OPTFM architecture, ported from
  OPTFM/node_pretrain/ours_crossattention_improve.py

using DENSE operations instead of torch_sparse (since our probing inputs
are tiny — a few hundred nodes at most). The architecture's expressivity
properties are independent of sparse/dense implementation.

The hierarchical variant adds to SGFormer_MIP:

  * self-attention on variables only      (trans_conv_var)
  * self-attention on constraints only    (trans_conv_cons)
  * cross-attention C -> V (via edge adj) (trans_conv_cross_constovar)
  * cross-attention V -> C (via edge adj) (trans_conv_cross_vartocons)
  * bipartite GNNPolicy                   (GCN)
  * pairwise final MLP over (c_i, v_j)    (fc)

For our 1-WL probing we only need graph-level embeddings, so we pool the
variable and constraint embeddings AFTER the cross-attention stack and
before the pairwise FC head.

This architecture is NOT loaded from any checkpoint — the checkpoint
file model_params_epoch_31.pth shipped in the OPTFM repo is for the
SGFormer_MIP *baseline*, not for the hierarchical variant. This is fine
for our purposes: 1-WL boundedness is an *architectural* property that
must hold for *every* setting of the weights. Random initialization is
sufficient to demonstrate the limit.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Linear attention (SGFormer-style): used by every trans_conv module below
# ---------------------------------------------------------------------------

class TransConvLayer(nn.Module):
    """Linear attention with global Frobenius-norm normalization.

    Source-input size may differ from query-input size (used by cross-
    attention). When they differ, the "+N * vs" residual term is padded
    up to the larger size in `ours_crossattention_improve.py`; we
    replicate that logic here.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 num_heads: int = 1, use_weight: bool = True):
        super().__init__()
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    @staticmethod
    def _pad_first_dim(t1: torch.Tensor, t2: torch.Tensor):
        m = max(t1.shape[0], t2.shape[0])
        if t1.shape[0] < m:
            t1 = F.pad(t1, (0, 0, 0, 0, 0, m - t1.shape[0]))
        if t2.shape[0] < m:
            t2 = F.pad(t2, (0, 0, 0, 0, 0, m - t2.shape[0]))
        return t1, t2

    def forward(self, query_input: torch.Tensor, source_input: torch.Tensor) -> torch.Tensor:
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)

        qs = qs / (torch.norm(qs, p=2) + 1e-8)
        ks = ks / (torch.norm(ks, p=2) + 1e-8)
        N = qs.shape[0]

        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)

        attention_num, vs = self._pad_first_dim(attention_num, vs)
        attention_num = attention_num + N * vs
        attention_num = attention_num[:qs.shape[0]]

        all_ones = torch.ones([ks.shape[0]], device=ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)
        attention_normalizer = attention_normalizer.unsqueeze(-1) + N
        attn_output = attention_num / attention_normalizer
        return attn_output.mean(dim=1)


# ---------------------------------------------------------------------------
# Self-attention stack (used on V-only and on C-only)
# ---------------------------------------------------------------------------

class TransConvSelf(nn.Module):
    """Stack of self-attention layers over a single node set."""
    def __init__(self, hidden_channels: int, num_layers: int = 1,
                 num_heads: int = 1, dropout: float = 0.0,
                 use_bn: bool = True, use_residual: bool = True,
                 use_weight: bool = True, use_act: bool = True):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels)])
        self.bns = nn.ModuleList([nn.LayerNorm(hidden_channels)])
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransConvLayer(hidden_channels, hidden_channels,
                                             num_heads, use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        residual = x
        for i, conv in enumerate(self.convs):
            y = conv(x, x)
            if self.use_residual:
                y = (y + residual) / 2.0
            if self.use_bn:
                y = self.bns[i + 1](y)
            if self.use_act:
                y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
            residual = y
            x = y
        return x


# ---------------------------------------------------------------------------
# Cross-attention with edge-weighted aggregation ("with pos embedding")
# ---------------------------------------------------------------------------

class TransConvCross(nn.Module):
    """Cross-attention from x to y, then concatenates with edge_matrix @ y.

    Dense port of TransConvwithPosembedding from
    ours_crossattention_improve.py. `edge_matrix` is the dense adjacency
    matrix of edges (one-directional), with shape (N_x, N_y) where N_x is
    the number of query nodes and N_y is the number of source nodes.
    """
    def __init__(self, hidden_channels: int, num_layers: int = 1,
                 num_heads: int = 1, dropout: float = 0.0,
                 use_bn: bool = True, use_residual: bool = True,
                 use_weight: bool = True, use_act: bool = True):
        super().__init__()
        # Output projection takes [attn_out, edge_agg] (2*hidden) -> hidden
        self.fcs = nn.ModuleList([nn.Linear(2 * hidden_channels, hidden_channels)])
        self.bns = nn.ModuleList([nn.LayerNorm(hidden_channels),
                                  nn.LayerNorm(hidden_channels)])
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransConvLayer(hidden_channels, hidden_channels,
                                             num_heads, use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                edge_matrix: torch.Tensor) -> torch.Tensor:
        if self.use_bn:
            x = self.bns[0](x)
            y = self.bns[1](y)
        x = F.relu(x)
        y = F.relu(y)
        x = F.dropout(x, p=self.dropout, training=self.training)
        y = F.dropout(y, p=self.dropout, training=self.training)

        residual = x
        for i, conv in enumerate(self.convs):
            x = conv(x, y)
            if self.use_residual:
                x = (x + residual) / 2.0
            if self.use_bn:
                x = self.bns[i + 2](x)
            if self.use_act:
                x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual = x

        # Edge-weighted aggregation: for each query node, sum over source
        # nodes weighted by edge_matrix[i, j] (a dense A or A^T).
        edge_agg = edge_matrix @ y                  # (N_x, hidden)
        x = torch.cat((x, edge_agg), dim=1)         # (N_x, 2*hidden)
        x = self.fcs[0](x)                          # (N_x, hidden)
        return x


# ---------------------------------------------------------------------------
# Bipartite GCN (dense version, same as sgformer_mip.GNNPolicy)
# ---------------------------------------------------------------------------

class _BipartiteConv(nn.Module):
    def __init__(self, emb: int = 16):
        super().__init__()
        self.fm_left  = nn.Linear(emb, emb, bias=True)
        self.fm_edge  = nn.Linear(1, emb, bias=False)
        self.fm_right = nn.Linear(emb, emb, bias=False)
        self.fm_final = nn.Sequential(nn.ReLU(), nn.Linear(emb, emb))
        self.post     = nn.Sequential(nn.ReLU(), nn.Linear(emb, emb))
        self.output   = nn.Sequential(nn.Linear(2 * emb, emb), nn.ReLU(),
                                      nn.Linear(emb, emb))

    def forward(self, left_features, edge_indices, edge_features, right_features):
        n_right = right_features.shape[0]
        left_idx  = edge_indices[0]
        right_idx = edge_indices[1]
        msg = self.fm_final(
            self.fm_left(left_features[left_idx])
            + self.fm_edge(edge_features)
            + self.fm_right(right_features[right_idx])
        )
        aggregated = torch.zeros(n_right, msg.shape[1], device=msg.device)
        aggregated.index_add_(0, right_idx, msg)
        return self.output(torch.cat([self.post(aggregated), right_features], dim=-1))


class _GNNPolicy(nn.Module):
    def __init__(self, emb: int = 16):
        super().__init__()
        self.cons_embedding = nn.Sequential(
            nn.Linear(emb, emb), nn.ReLU(),
            nn.Linear(emb, emb), nn.ReLU())
        self.edge_embedding = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
        self.var_embedding = nn.Sequential(
            nn.Linear(emb, emb), nn.ReLU(),
            nn.Linear(emb, emb), nn.ReLU())
        self.conv_v_to_c = _BipartiteConv(emb)
        self.conv_c_to_v = _BipartiteConv(emb)
        self.output_module_var  = nn.Sequential(nn.Linear(emb, emb), nn.ReLU())
        self.output_module_cons = nn.Sequential(nn.Linear(emb, emb), nn.ReLU())

    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        reversed_edges = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        c = self.cons_embedding(constraint_features)
        e = self.edge_embedding(edge_features)
        v = self.var_embedding(variable_features)
        c = self.conv_v_to_c(v, reversed_edges, e, c)
        v = self.conv_c_to_v(c, edge_indices, e, v)
        return self.output_module_var(v), self.output_module_cons(c)


# ---------------------------------------------------------------------------
# Full hierarchical OPTFM
# ---------------------------------------------------------------------------

class HierarchicalOPTFM(nn.Module):
    """
    Full multi-view hierarchical OPTFM architecture (matching
    ours_crossattention_improve.SGFormer_MIP).
    """
    def __init__(self, in_channels_var: int = 9, in_channels_cons: int = 1,
                 hidden_channels: int = 16, out_channels: int = 2,
                 trans_num_layers: int = 1, trans_num_heads: int = 1,
                 trans_dropout: float = 0.0, trans_use_bn: bool = True,
                 trans_use_residual: bool = True, trans_use_weight: bool = True,
                 trans_use_act: bool = True, use_graph: bool = True,
                 graph_weight: float = 0.5, aggregate: str = "add"):
        super().__init__()
        self.var_embedding  = nn.Linear(in_channels_var, hidden_channels)
        self.cons_embedding = nn.Linear(in_channels_cons, hidden_channels)
        self.trans_conv_var  = TransConvSelf(hidden_channels, trans_num_layers,
                                             trans_num_heads, trans_dropout,
                                             trans_use_bn, trans_use_residual,
                                             trans_use_weight, trans_use_act)
        self.trans_conv_cons = TransConvSelf(hidden_channels, trans_num_layers,
                                             trans_num_heads, trans_dropout,
                                             trans_use_bn, trans_use_residual,
                                             trans_use_weight, trans_use_act)
        self.trans_conv_cross_constovar = TransConvCross(
            hidden_channels, trans_num_layers, trans_num_heads, trans_dropout,
            trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)
        self.trans_conv_cross_vartocons = TransConvCross(
            hidden_channels, trans_num_layers, trans_num_heads, trans_dropout,
            trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)
        self.GCN = _GNNPolicy(hidden_channels)
        self.hidden_channels = hidden_channels
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.aggregate = aggregate
        if aggregate == "add":
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            self.fc = nn.Linear(4 * hidden_channels, out_channels)

    # ------------------------------------------------------------------
    # Embedding extraction (main entry point for probing)
    # ------------------------------------------------------------------

    def _build_dense_edge_matrix(self, cons_n: int, var_n: int,
                                 edge_index: torch.Tensor,
                                 edge_attr: torch.Tensor) -> torch.Tensor:
        """Build the dense adjacency matrix (cons_n, var_n) from coo indices."""
        M = torch.zeros(cons_n, var_n, device=edge_index.device)
        M[edge_index[0], edge_index[1]] = edge_attr.squeeze(-1)
        return M

    def _encode(self, cons_x, edge_index, edge_attr, var_x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the hierarchical encoder and return (cons_emb, var_emb)."""
        n_cons = cons_x.shape[0]
        n_vars = var_x.shape[0]

        var_h  = self.var_embedding(var_x)
        cons_h = self.cons_embedding(cons_x)

        # Stage 1: intra-type self-attention
        var_h  = self.trans_conv_var(var_h)
        cons_h = self.trans_conv_cons(cons_h)

        # Stage 2: cross-attention with edge-weighted aggregation
        edge_mat_cv = self._build_dense_edge_matrix(n_cons, n_vars, edge_index, edge_attr)
        edge_mat_vc = edge_mat_cv.t()

        # C->V cross: var attends to cons, then edge-aggregates over cons
        var_h  = self.trans_conv_cross_constovar(var_h, cons_h, edge_mat_vc)
        # V->C cross: cons attends to var, then edge-aggregates over var
        cons_h = self.trans_conv_cross_vartocons(cons_h, var_h, edge_mat_cv)

        # Stage 3: the bipartite GCN branch (optional)
        if self.use_graph:
            var_h2, cons_h2 = self.GCN(self.cons_embedding(cons_x),
                                       edge_index, edge_attr,
                                       self.var_embedding(var_x))
            if self.aggregate == "add":
                var_h  = self.graph_weight * var_h2  + (1 - self.graph_weight) * var_h
                cons_h = self.graph_weight * cons_h2 + (1 - self.graph_weight) * cons_h
            else:
                var_h  = torch.cat((var_h,  var_h2),  dim=1)
                cons_h = torch.cat((cons_h, cons_h2), dim=1)
        return cons_h, var_h

    def get_graph_embedding(self, cons_x, edge_index, edge_attr, var_x,
                            pooling: str = "mean") -> torch.Tensor:
        cons_h, var_h = self._encode(cons_x, edge_index, edge_attr, var_x)
        x = torch.cat([cons_h, var_h], dim=0)
        if pooling == "mean":
            return x.mean(dim=0)
        if pooling == "sum":
            return x.sum(dim=0)
        if pooling == "max":
            return x.max(dim=0).values
        raise ValueError(f"unknown pooling: {pooling}")

    def get_node_embeddings(self, cons_x, edge_index, edge_attr, var_x):
        return self._encode(cons_x, edge_index, edge_attr, var_x)


def create_hierarchical(hidden_channels: int = 16) -> HierarchicalOPTFM:
    model = HierarchicalOPTFM(hidden_channels=hidden_channels)
    model.eval()
    return model


if __name__ == "__main__":
    # Smoke test
    model = create_hierarchical()
    n_cons, n_vars = 5, 6
    cons_x = torch.randn(n_cons, 1)
    var_x = torch.randn(n_vars, 9)
    ei = torch.stack([
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4], dtype=torch.long),
        torch.tensor([0, 1, 2, 3, 4, 5, 0, 2, 4, 1, 3, 5], dtype=torch.long),
    ])
    ea = torch.ones(ei.shape[1], 1)
    with torch.no_grad():
        emb = model.get_graph_embedding(cons_x, ei, ea, var_x)
    print("HierarchicalOPTFM smoke test: embedding shape =", emb.shape)
    print("params:", sum(p.numel() for p in model.parameters()))
