"""
OPTFM Model - Exact Architecture Match to Pre-trained Checkpoint

Architecture parameters (from checkpoint):
- in_channels_var = 9 (Ecole MilpBipartite variable features)
- in_channels_cons = 1 (normalized RHS)
- hidden_channels = 16
- trans_num_layers = 1
- trans_num_heads = 1
- out_channels = 2 (binary classification for pre-training)
- aggregate = 'add' (fc input = 32 = 2*16)
- graph_weight = 0.8 (default)

This implementation avoids torch-sparse by using dense operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class TransConvLayer(nn.Module):
    """Transformer layer with linear attention (from OPTFM paper)."""
    
    def __init__(self, in_channels: int, out_channels: int, num_heads: int = 1, use_weight: bool = True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight
    
    def forward(self, query_input: torch.Tensor, source_input: torch.Tensor) -> torch.Tensor:
        qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
        
        if self.use_weight:
            vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
        else:
            vs = source_input.reshape(-1, 1, self.out_channels)
        
        # Normalize
        qs = qs / (torch.norm(qs, p=2) + 1e-8)
        ks = ks / (torch.norm(ks, p=2) + 1e-8)
        N = qs.shape[0]
        
        # Linear attention: O(N) complexity
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)
        attention_num = attention_num + N * vs
        
        all_ones = torch.ones([ks.shape[0]], device=ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)
        attention_normalizer = attention_normalizer.unsqueeze(-1) + N
        
        attn_output = attention_num / attention_normalizer
        return attn_output.mean(dim=1)


class TransConv(nn.Module):
    """Multi-layer transformer with linear attention."""
    
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 1,
                 num_heads: int = 1, dropout: float = 0.5, use_bn: bool = True,
                 use_residual: bool = True, use_weight: bool = True, use_act: bool = True):
        super().__init__()
        
        self.fcs = nn.ModuleList([nn.Linear(in_channels, hidden_channels)])
        self.bns = nn.ModuleList([nn.LayerNorm(hidden_channels)])
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(TransConvLayer(hidden_channels, hidden_channels, num_heads, use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))
        
        self.dropout = dropout
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act
        self.activation = F.relu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_ = []
        
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        layer_.append(x)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, x)
            if self.use_residual:
                x = (x + layer_[i]) / 2.0
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class BipartiteGraphConvolution(nn.Module):
    """Bipartite message passing (dense implementation for small graphs)."""
    
    def __init__(self, emb_size: int = 16):
        super().__init__()
        
        self.feature_module_left = nn.Sequential(
            nn.Linear(emb_size, emb_size, bias=True)
        )
        self.feature_module_edge = nn.Sequential(
            nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = nn.Sequential(
            nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )
        self.post_conv_module = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )
        self.output_module = nn.Sequential(
            nn.Linear(2 * emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
        )
    
    def forward(self, left_features: torch.Tensor, edge_indices: torch.Tensor,
                edge_features: torch.Tensor, right_features: torch.Tensor) -> torch.Tensor:
        n_right = right_features.shape[0]
        
        left_idx = edge_indices[0]
        right_idx = edge_indices[1]
        
        msg_left = self.feature_module_left(left_features[left_idx])
        msg_edge = self.feature_module_edge(edge_features)
        msg_right = self.feature_module_right(right_features[right_idx])
        
        messages = self.feature_module_final(msg_left + msg_edge + msg_right)
        
        # Aggregate messages to right nodes (sum)
        aggregated = torch.zeros(n_right, messages.shape[1], device=messages.device)
        aggregated.index_add_(0, right_idx, messages)
        
        output = self.output_module(
            torch.cat([self.post_conv_module(aggregated), right_features], dim=-1)
        )
        return output


class GNNPolicy(nn.Module):
    """Bipartite GNN for MILP (variables <-> constraints)."""
    
    def __init__(self, emb_size: int = 16):
        super().__init__()
        
        self.cons_embedding = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )
        self.edge_embedding = nn.Sequential(
            nn.Linear(1, 1),
            nn.ReLU(),
        )
        self.var_embedding = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )
        
        self.conv_v_to_c = BipartiteGraphConvolution(emb_size)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size)
        
        self.output_module_var = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )
        self.output_module_cons = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
        )
    
    def forward(self, constraint_features: torch.Tensor, edge_indices: torch.Tensor,
                edge_features: torch.Tensor, variable_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
        
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)
        
        constraint_features = self.conv_v_to_c(
            variable_features, reversed_edge_indices, edge_features, constraint_features
        )
        variable_features = self.conv_c_to_v(
            constraint_features, edge_indices, edge_features, variable_features
        )
        
        return self.output_module_var(variable_features), self.output_module_cons(constraint_features)


class SGFormer_MIP(nn.Module):
    """
    OPTFM main architecture: SGFormer for MILP bipartite graphs.
    
    Combines TransConv (global attention) + GNNPolicy (local message passing).
    """
    
    def __init__(self, in_channels_var: int = 9, in_channels_cons: int = 1,
                 hidden_channels: int = 16, out_channels: int = 2,
                 trans_num_layers: int = 1, trans_num_heads: int = 1,
                 trans_dropout: float = 0.5, trans_use_bn: bool = True,
                 trans_use_residual: bool = True, trans_use_weight: bool = True,
                 trans_use_act: bool = True, use_graph: bool = True,
                 graph_weight: float = 0.8, aggregate: str = 'add'):
        super().__init__()
        
        self.var_embedding = nn.Linear(in_channels_var, hidden_channels)
        self.cons_embedding = nn.Linear(in_channels_cons, hidden_channels)
        
        self.trans_conv = TransConv(
            hidden_channels, hidden_channels, trans_num_layers, trans_num_heads,
            trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act
        )
        self.GCN = GNNPolicy(hidden_channels)
        
        self.use_graph = use_graph
        self.graph_weight = graph_weight
        self.aggregate = aggregate
        self.hidden_channels = hidden_channels
        
        if aggregate == 'add':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            self.fc = nn.Linear(4 * hidden_channels, out_channels)
    
    def forward(self, cons_x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, var_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        var_emb = self.var_embedding(var_x)
        cons_emb = self.cons_embedding(cons_x)
        
        concatenated_x = torch.cat((cons_emb, var_emb), dim=0)
        x1 = self.trans_conv(concatenated_x)
        
        if self.use_graph:
            x2_var, x2_cons = self.GCN(cons_emb, edge_index, edge_attr, var_emb)
            x2 = torch.cat((x2_cons, x2_var), dim=0)
            
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        
        n_cons = cons_emb.shape[0]
        return x, x[n_cons:], x[:n_cons]
    
    def get_graph_embedding(self, cons_x: torch.Tensor, edge_index: torch.Tensor,
                            edge_attr: torch.Tensor, var_x: torch.Tensor,
                            pooling: str = 'mean') -> torch.Tensor:
        """Get graph-level embedding by pooling node embeddings."""
        x, var_emb, cons_emb = self.forward(cons_x, edge_index, edge_attr, var_x)
        
        if pooling == 'mean':
            return x.mean(dim=0)
        elif pooling == 'sum':
            return x.sum(dim=0)
        elif pooling == 'max':
            return x.max(dim=0)[0]
        else:
            return x.mean(dim=0)
    
    def get_node_embeddings(self, cons_x: torch.Tensor, edge_index: torch.Tensor,
                            edge_attr: torch.Tensor, var_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get variable and constraint embeddings separately."""
        _, var_emb, cons_emb = self.forward(cons_x, edge_index, edge_attr, var_x)
        return var_emb, cons_emb


class TransConvOnly(nn.Module):
    """Ablation: TransConv only (no GNN)."""
    
    def __init__(self, in_channels_var: int = 9, in_channels_cons: int = 1,
                 hidden_channels: int = 16, **kwargs):
        super().__init__()
        self.var_embedding = nn.Linear(in_channels_var, hidden_channels)
        self.cons_embedding = nn.Linear(in_channels_cons, hidden_channels)
        self.trans_conv = TransConv(hidden_channels, hidden_channels, num_layers=1)
        self.hidden_channels = hidden_channels
    
    def get_graph_embedding(self, cons_x, edge_index, edge_attr, var_x, pooling='mean'):
        var_emb = self.var_embedding(var_x)
        cons_emb = self.cons_embedding(cons_x)
        x = torch.cat((cons_emb, var_emb), dim=0)
        x = self.trans_conv(x)
        return x.mean(dim=0) if pooling == 'mean' else x.sum(dim=0)


class GNNOnly(nn.Module):
    """Ablation: GNN only (no TransConv)."""
    
    def __init__(self, in_channels_var: int = 9, in_channels_cons: int = 1,
                 hidden_channels: int = 16, **kwargs):
        super().__init__()
        self.var_embedding = nn.Linear(in_channels_var, hidden_channels)
        self.cons_embedding = nn.Linear(in_channels_cons, hidden_channels)
        self.GCN = GNNPolicy(hidden_channels)
        self.hidden_channels = hidden_channels
    
    def get_graph_embedding(self, cons_x, edge_index, edge_attr, var_x, pooling='mean'):
        var_emb = self.var_embedding(var_x)
        cons_emb = self.cons_embedding(cons_x)
        x2_var, x2_cons = self.GCN(cons_emb, edge_index, edge_attr, var_emb)
        x = torch.cat((x2_cons, x2_var), dim=0)
        return x.mean(dim=0) if pooling == 'mean' else x.sum(dim=0)


class SimpleGCN(nn.Module):
    """Baseline: Simple 2-layer GCN."""
    
    def __init__(self, in_channels_var: int = 9, in_channels_cons: int = 1,
                 hidden_channels: int = 16, **kwargs):
        super().__init__()
        self.var_embedding = nn.Linear(in_channels_var, hidden_channels)
        self.cons_embedding = nn.Linear(in_channels_cons, hidden_channels)
        self.conv1 = nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, hidden_channels)
        self.hidden_channels = hidden_channels
    
    def get_graph_embedding(self, cons_x, edge_index, edge_attr, var_x, pooling='mean'):
        var_emb = self.var_embedding(var_x)
        cons_emb = self.cons_embedding(cons_x)
        
        # Simple message passing using adjacency
        n_cons, n_vars = cons_x.shape[0], var_x.shape[0]
        n_total = n_cons + n_vars
        
        x = torch.cat([cons_emb, var_emb], dim=0)
        
        # Build adjacency (dense for small graphs)
        adj = torch.zeros(n_total, n_total, device=x.device)
        cons_idx, var_idx = edge_index[0], edge_index[1] + n_cons
        adj[cons_idx, var_idx] = 1.0
        adj[var_idx, cons_idx] = 1.0
        
        # Normalize
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj = adj / deg
        
        # GCN layers
        x = F.relu(self.conv1(adj @ x))
        x = self.conv2(adj @ x)
        
        return x.mean(dim=0) if pooling == 'mean' else x.sum(dim=0)


def load_pretrained_weights(model: SGFormer_MIP, checkpoint_path: str) -> bool:
    """Load pre-trained OPTFM weights."""
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing:
            print(f"Missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
        
        print(f"Loaded pre-trained weights from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False


def create_model(model_type: str = 'optfm', pretrained_path: str = None,
                 in_channels_var: int = 9, in_channels_cons: int = 1,
                 hidden_channels: int = 16) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'optfm', 'transconv_only', 'gnn_only', 'gcn', 'random'
        pretrained_path: Path to checkpoint (only for 'optfm')
        
    Returns:
        Model instance
    """
    kwargs = {
        'in_channels_var': in_channels_var,
        'in_channels_cons': in_channels_cons,
        'hidden_channels': hidden_channels,
    }
    
    if model_type == 'optfm':
        model = SGFormer_MIP(**kwargs)
        if pretrained_path:
            load_pretrained_weights(model, pretrained_path)
    elif model_type == 'transconv_only':
        model = TransConvOnly(**kwargs)
    elif model_type == 'gnn_only':
        model = GNNOnly(**kwargs)
    elif model_type == 'gcn':
        model = SimpleGCN(**kwargs)
    elif model_type == 'random':
        model = SGFormer_MIP(**kwargs)
        # Random weights (no loading)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    return model


if __name__ == "__main__":
    # Test model creation and forward pass
    print("Testing OPTFM model...")
    
    model = create_model('optfm')
    
    # Test input (5 constraints, 6 variables)
    n_cons, n_vars = 5, 6
    cons_x = torch.randn(n_cons, 1)  # 1 constraint feature
    var_x = torch.randn(n_vars, 9)   # 9 variable features
    
    n_edges = 15
    edge_index = torch.stack([
        torch.randint(0, n_cons, (n_edges,)),
        torch.randint(0, n_vars, (n_edges,))
    ])
    edge_attr = torch.randn(n_edges, 1)
    
    with torch.no_grad():
        graph_emb = model.get_graph_embedding(cons_x, edge_index, edge_attr, var_x)
    
    print(f"Graph embedding shape: {graph_emb.shape}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print("Test passed!")
