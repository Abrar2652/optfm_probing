import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
import torch_geometric

class GraphConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_weight=True, use_init=False):
        super(GraphConvLayer, self).__init__()

        self.use_init = use_init
        self.use_weight = use_weight
        if self.use_init:
            in_channels_ = 2 * in_channels
        else:
            in_channels_ = in_channels
        self.W = nn.Linear(in_channels_, out_channels)

    def reset_parameters(self):
        self.W.reset_parameters()

    def forward(self, x, edge_index, x0):
        N = x.shape[0]
        row, col = edge_index
        d = degree(col, N).float()
        d_norm_in = (1. / d[col]).sqrt()
        d_norm_out = (1. / d[row]).sqrt()
        value = torch.ones_like(row) * d_norm_in * d_norm_out
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))
        x = matmul(adj, x)  # [N, D]

        if self.use_init:
            x = torch.cat([x, x0], 1)
            x = self.W(x)
        elif self.use_weight:
            x = self.W(x)

        return x


class GraphConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_init=False, use_act=True):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers):
            self.convs.append(
                GraphConvLayer(hidden_channels, hidden_channels, use_weight, use_init))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, edge_index):
        layer_ = []

        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        layer_.append(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, layer_[0])
            if self.use_bn:
                x = self.bns[i+1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.use_residual:
                x = x + layer_[-1]
        return x

# class TransConvLayer(nn.Module):
#     '''
#     transformer with fast attention
#     '''

#     def __init__(self, in_channels,
#                  out_channels,
#                  num_heads,
#                  use_weight=True):
#         super().__init__()
#         self.Wk = nn.Linear(in_channels, out_channels * num_heads)
#         self.Wq = nn.Linear(in_channels, out_channels * num_heads)
#         if use_weight:
#             self.Wv = nn.Linear(in_channels, out_channels * num_heads)

#         self.out_channels = out_channels
#         self.num_heads = num_heads
#         self.use_weight = use_weight

#     def reset_parameters(self):
#         self.Wk.reset_parameters()
#         self.Wq.reset_parameters()
#         if self.use_weight:
#             self.Wv.reset_parameters()
    
#     def pad_first_dim(self, t1, t2):
#         """Pad the first dimension of tensors with zeros to make them the same size."""
#         # 获取两个张量的形状
#         shape1 = t1.shape
#         shape2 = t2.shape
        
#         # 找到第一维的最大值
#         max_size_0 = max(shape1[0], shape2[0])
        
#         # 创建目标形状
#         target_shape = (max_size_0, *shape1[1:])
        
#         # 创建填充后的张量
#         padded_t1 = F.pad(t1, (0, 0, 0, 0, 0, max_size_0 - shape1[0])) if shape1[0] < max_size_0 else t1
#         padded_t2 = F.pad(t2, (0, 0, 0, 0, 0, max_size_0 - shape2[0])) if shape2[0] < max_size_0 else t2
        
#         return padded_t1, padded_t2

#     def forward(self, query_input, source_input, output_attn=False):
#         # feature transformation
#         qs = self.Wq(query_input).reshape(-1, self.num_heads, self.out_channels)
#         ks = self.Wk(source_input).reshape(-1, self.num_heads, self.out_channels)
#         if self.use_weight:
#             vs = self.Wv(source_input).reshape(-1, self.num_heads, self.out_channels)
#         else:
#             vs = source_input.reshape(-1, 1, self.out_channels)

#         # normalize input
#         qs = qs / torch.norm(qs, p=2)  # [N, H, M]
#         ks = ks / torch.norm(ks, p=2)  # [L, H, M]
#         N = qs.shape[0]

#         # numerator
#         kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
#         attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        
#         attention_num, vs = self.pad_first_dim(attention_num, vs)
        
#         attention_num += N * vs
#         attention_num = attention_num[:qs.shape[0]]

#         # denominator
#         all_ones = torch.ones([ks.shape[0]]).to(ks.device)
#         ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
#         attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

#         # attentive aggregated results
#         attention_normalizer = torch.unsqueeze(
#             attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
#         attention_normalizer += torch.ones_like(attention_normalizer) * N
#         attn_output = attention_num / attention_normalizer  # [N, H, D]

#         # compute attention for visualization if needed
#         if output_attn:
#             attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
#             normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
#             attention = attention / normalizer

#         final_output = attn_output.mean(dim=1)

#         if output_attn:
#             return final_output, attention
#         else:
#             return final_output


class TransConvLayer(nn.Module):
    '''
    Transformer with fast attention supporting batched graph input.
    '''

    def __init__(self, in_channels,
                 out_channels,
                 num_heads,
                 use_weight=True):
        super().__init__()
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        if use_weight:
            self.Wv = nn.Linear(in_channels, out_channels * num_heads)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_weight = use_weight

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        if self.use_weight:
            self.Wv.reset_parameters()

    def forward(self, query_input, source_input, output_attn=False):
        # 获取 batch size 和节点数
        batch_size, N, _ = query_input.size()
        _, L, _ = source_input.size()

        # feature transformation
        qs = self.Wq(query_input).reshape(batch_size, N, self.num_heads, self.out_channels)
        ks = self.Wk(source_input).reshape(batch_size, L, self.num_heads, self.out_channels)
        if self.use_weight:
            vs = self.Wv(source_input).reshape(batch_size, L, self.num_heads, self.out_channels)
        else:
            vs = source_input.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        # normalize input
        qs = qs / torch.norm(qs, p=2, dim=-1, keepdim=True)  # [B, N, H, M]
        ks = ks / torch.norm(ks, p=2, dim=-1, keepdim=True)  # [B, L, H, M]

        # numerator
        kvs = torch.einsum("blhm,blhd->bhmd", ks, vs)  # [B, H, M, D]
        attention_num = torch.einsum("bnhm,bhmd->bnhd", qs, kvs)  # [B, N, H, D]
        
        # Add N * vs to attention_num
        N_tensor = torch.tensor(N, dtype=torch.float32, device=query_input.device)
        vs_expanded = vs.sum(dim=1, keepdim=True)  # [B, 1, H, D]
        attention_num += N_tensor * vs_expanded  # [B, N, H, D]

        # denominator
        all_ones = torch.ones([batch_size, L], device=query_input.device)
        ks_sum = torch.einsum("blhm,bl->bhm", ks, all_ones)  # [B, H, M]
        attention_normalizer = torch.einsum("bnhm,bhm->bnh", qs, ks_sum)  # [B, N, H]

        # attentive aggregated results
        attention_normalizer = attention_normalizer.unsqueeze(-1) + N  # [B, N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N_tensor  # [B, N, H, 1]
        attn_output = attention_num / attention_normalizer  # [B, N, H, D]

        # compute attention for visualization if needed
        if output_attn:
            attention = torch.einsum("bnhm,blhm->bnlh", qs, ks).mean(dim=-1)  # [B, N, L]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [B, N, 1]
            attention = attention / normalizer

        final_output = attn_output.mean(dim=2)  # [B, N, D]

        if output_attn:
            return final_output, attention
        else:
            return final_output


class TransConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x):
        layer_ = []

        # input MLP layer
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, x)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class TransConvwithPosembedding(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, num_heads=1,
                 dropout=0.5, use_bn=True, use_residual=True, use_weight=True, use_act=True):
        super().__init__()

        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels * 2, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels))
        self.bns.append(nn.LayerNorm(hidden_channels))
        for i in range(num_layers):
            self.convs.append(
                TransConvLayer(hidden_channels, hidden_channels, num_heads=num_heads, use_weight=use_weight))
            self.bns.append(nn.LayerNorm(hidden_channels))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.use_residual = use_residual
        self.use_act = use_act

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()

    def forward(self, x, y, edge_matrix):
        
        # x, y: input cross-attention matrix
        # edge_matrix: edge embedding
        
        layer_ = []

        # input MLP layer
        # x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
            y = self.bns[1](y)
        x = self.activation(x)
        y = self.activation(y)
        x = F.dropout(x, p=self.dropout, training=self.training)
        y = F.dropout(y, p=self.dropout, training=self.training)

        # store as residual link
        layer_.append(x)

        for i, conv in enumerate(self.convs):
            # graph convolution with full attention aggregation
            x = conv(x, y)
            if self.use_residual:
                x = (x + layer_[i]) / 2.            
            if self.use_bn:
                x = self.bns[i + 2](x)
            if self.use_act:
                x = self.activation(x)
            
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_.append(x)
        
        x = torch.cat((x, torch.matmul(edge_matrix, y)), dim=1)
        
        x = self.fcs[0](x)

        return x

    def get_attentions(self, x):
        layer_, attentions = [], []
        x = self.fcs[0](x)
        if self.use_bn:
            x = self.bns[0](x)
        x = self.activation(x)
        layer_.append(x)
        for i, conv in enumerate(self.convs):
            x, attn = conv(x, x, output_attn=True)
            attentions.append(attn)
            if self.use_residual:
                x = (x + layer_[i]) / 2.
            if self.use_bn:
                x = self.bns[i + 1](x)
            if self.use_act:
                x = self.activation(x)
            layer_.append(x)
        return torch.stack(attentions, dim=0)  # [layer num, N, N]


class SGFormer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True, trans_use_weight=True, trans_use_act=True,
                 gnn_num_layers=1, gnn_dropout=0.5, gnn_use_weight=True, gnn_use_init=False, gnn_use_bn=True, gnn_use_residual=True, gnn_use_act=True,
                 use_graph=True, graph_weight=0.8, aggregate='add'):
        super().__init__()
        self.trans_conv = TransConv(in_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)
        self.graph_conv = GraphConv(in_channels, hidden_channels, gnn_num_layers, gnn_dropout, gnn_use_bn, gnn_use_residual, gnn_use_weight, gnn_use_init, gnn_use_act)
        self.use_graph = use_graph
        self.graph_weight = graph_weight

        self.aggregate = aggregate

        if aggregate == 'add':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif aggregate == 'cat':
            self.fc = nn.Linear(2 * hidden_channels, out_channels)
        else:
            raise ValueError(f'Invalid aggregate type:{aggregate}')

        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.graph_conv.parameters()) if self.graph_conv is not None else []
        self.params2.extend(list(self.fc.parameters()))

    def forward(self, x, edge_index):
        x1 = self.trans_conv(x)
        if self.use_graph:
            x2 = self.graph_conv(x, edge_index)
            if self.aggregate == 'add':
                x = self.graph_weight * x2 + (1 - self.graph_weight) * x1
            else:
                x = torch.cat((x1, x2), dim=1)
        else:
            x = x1
        x = self.fc(x)
        return x
    
    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x) # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.trans_conv.reset_parameters()
        if self.use_graph:
            self.graph_conv.reset_parameters()


class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    Class Description:
    Based on graph convolution, define the bipartite graph semi-convolution process.
    """

    def __init__(self):
        '''
        Function Description:
        Define the size of the encoding space, and implement the semi-convolution layer and output layer.
        '''
        super().__init__("add")
        emb_size = 16

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        '''
        Function Description:
        Based on the given node and edge features, output the result of forward propagation after semi-convolution.

        Parameters:
        - left_features: Features of the nodes on the left side of the bipartite graph.
        - edge_indices: Edge information.
        - edge_features: Edge features.
        - right_features: Features of the nodes on the right side of the bipartite graph.

        Return: The result after forward propagation.
        '''
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )

    def message(self, node_features_i, node_features_j, edge_features):
        '''
        Function Description:
        This method sends the messages, computed in the message method.
        
        Parameters:
        - node_features_i: Features of the nodes on the left side of the bipartite graph.
        - node_features_j: Features of the nodes on the right side of the bipartite graph.
        - edge_features: Edge features.

        Return: The result after the message passing in the semi-convolution.
        '''
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output


class GNNPolicy(torch.nn.Module):
    """
    Class Description:
    Based on the semi-convolutional layer, define the entire GNN network structure.
    """
    def __init__(self, emb_size):
        '''
        Function Description:
        Define the size of the encoding space, and define the layers for decision variable encoding, edge feature encoding, and constraint feature encoding.
        Define two semi-convolutional layers and the final output layer.
        '''
        super().__init__()
        edge_nfeats = 1

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.Linear(edge_nfeats, edge_nfeats),
            torch.nn.ReLU(),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        self.conv_v_to_c = BipartiteGraphConvolution()
        self.conv_c_to_v = BipartiteGraphConvolution()

        self.output_module_var = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            # torch.nn.Linear(emb_size, 1, bias=False),
        )
        
        self.output_module_cons = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            # torch.nn.Linear(emb_size, 1, bias=False),
        )


    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        '''
        Function Description:
        Based on the given constraint, edge, and variable features, output the result of forward propagation after GNN.

        Parameters:
        - constraint_features: Features of the constraint points.
        - edge_indices: Edge information.
        - edge_features: Edge features.
        - variable_features: Features of the variable points.

        Return: The result after forward propagation.
        '''
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
        output_var = self.output_module_var(variable_features).squeeze(-1)
        output_cons = self.output_module_cons(constraint_features).squeeze(-1)
        return output_var, output_cons


class SGFormer_MIP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 trans_num_layers=1, trans_num_heads=1, trans_dropout=0.5, trans_use_bn=True, trans_use_residual=True, trans_use_weight=True, trans_use_act=True,
                 use_graph=True, graph_weight=0.8, aggregate='add'):
        super().__init__()
        self.embedding = nn.Linear(in_channels, hidden_channels)
        self.trans_conv = TransConv(hidden_channels, hidden_channels, trans_num_layers, trans_num_heads, trans_dropout, trans_use_bn, trans_use_residual, trans_use_weight, trans_use_act)

        self.fc = nn.Linear(hidden_channels, out_channels)

        self.params0 = list(self.embedding.parameters())
        self.params1 = list(self.trans_conv.parameters())
        self.params2 = list(self.fc.parameters())

    def forward(self, data, lengths):
        data = self.embedding(data)
        
        encodings = self.trans_conv(data)
        output = self.fc(encodings)

        # 创建掩码矩阵，形状为 [B, N]
        mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # 将掩码扩展到与特征张量相同的形状 [B, N, D]
        mask = mask.unsqueeze(-1).expand_as(output).float()
        
        # 应用掩码，使填充部分的特征值为 0
        masked_features = output * mask
        
        # 对有效部分求和
        sum_features = masked_features.sum(dim=1)
        
        # 使用 lengths 进行除法运算以得到平均值
        # 注意: 需要将 lengths 转换为 float 类型，并且扩展成 [B, 1] 形状以便广播
        pooled_features = sum_features / lengths.unsqueeze(1).float()
        
        return pooled_features
    
    def get_attentions(self, x):
        attns = self.trans_conv.get_attentions(x) # [layer num, N, N]

        return attns

    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.trans_conv.reset_parameters()