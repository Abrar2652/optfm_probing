import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_scatter import scatter

from logger import Logger, save_result
from dataset import load_dataset_bipartite, split_bipartite_graph
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_recall, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from eval_mip import evaluate_mip
from parse import parser_add_main_args, parse_method_mip

import time
import pickle

import random
import math

import warnings
warnings.filterwarnings('ignore')

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    """计算并输出模型的总参数量"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    return total_params

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Edge Identification')
parser_add_main_args(parser)
args = parser.parse_args()

args.data_dir = args.data_dir.split("/")[0] + "_SCIP/"
args.valid_dir = args.valid_dir.split("/")[0] + "_SCIP/"
args.test_dir = args.test_dir.split("/")[0] + "_SCIP/"

if args.all_data:
    args.data_dir = None
    args.valid_dir = None
    args.test_dir = None

print(args)

fix_seed(args.seed)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

### Load and preprocess data ###
dataset_loader = load_dataset_bipartite(args.data_dir, args.batch_size)

# Label to the device
# for label in label_loader:
#     label = label.to(device)
#     c = label.values().max().item() + 1

# Extract some basic information
for train_data in dataset_loader:
    var_d = train_data.variable_features.shape[1]
    con_d = train_data.constraint_features.shape[1]
    break

# Label count, 二分类问题
c = 2

### Load method ###
model = parse_method_mip(args, c, var_d, con_d, device)

# criterion = nn.NLLLoss()  # 负对数似然损失（Negative Log Likelihood Loss
lass_weights = torch.tensor([0.2, 15], dtype=torch.float)
criterion = nn.CrossEntropyLoss()  # 负对数似然损失（Negative Log Likelihood Loss）

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
elif args.metric == 'acc':
    eval_func = eval_acc
else:
    eval_func = eval_recall

model.train()
print('MODEL:', model)

# 验证集逐个instance进行评估
valid_loader = load_dataset_bipartite(args.valid_dir, 1)

Maximum_batch_size = 1000
Maximum_batch_size_valid = 2000

best_acc = -float("inf")
model_save_dir = "Models_SCIP/"
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)

optimizer = torch.optim.Adam([
            {'params': model.params0, 'weight_decay': args.trans_weight_decay},
            {'params': model.params1, 'weight_decay': args.trans_weight_decay},
            {'params': model.params2, 'weight_decay': args.gnn_weight_decay}
        ],
            lr=args.lr)

for epoch in range(args.epochs):
    print("Current epoch: ", epoch)
    batch_id = 0
    for batch in dataset_loader:
        start_time = time.time()
        print("Batch ID: ", batch_id)
        
        ### Basic information of datasets ###
        n = batch.constraint_features.shape[0] + batch.variable_features.shape[0]
        e = batch.edge_index.shape[1]
        # infer the number of classes for non one-hot and one-hot labels
        
        c = 2
        var_d = batch.variable_features.shape[1]
        con_d = batch.constraint_features.shape[1]
        edge_d = batch.edge_attr.shape[1]

        print(f" num nodes {n} | num edge {e} | num var feats {var_d} | num cons feats {con_d} | num classes {c}")
        
        batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features = \
            batch.constraint_features.to(device), batch.edge_index.to(device), batch.edge_attr.to(device), batch.variable_features.to(device)
        
        if n > Maximum_batch_size:
            subgraphs = split_bipartite_graph(batch.variable_features, batch.constraint_features, batch.edge_index, batch.edge_attr, Maximum_batch_size, device)
            batch_acc_list = []
            for mini_batch_id in range(len(subgraphs)):
                constraint_indices, variable_indices, batch_edge_indices = subgraphs[mini_batch_id]
                
                mini_batch_constraint_features = batch.constraint_features[constraint_indices]
                mini_batch_variable_features = batch.variable_features[variable_indices]
                
                # Append global variable and constraint features
                mini_batch_constraint_features = torch.cat((mini_batch_constraint_features, torch.mean(mini_batch_constraint_features, dim=0, keepdim=True)), dim=0)
                mini_batch_variable_features = torch.cat((mini_batch_variable_features, torch.mean(mini_batch_variable_features, dim=0, keepdim=True)), dim=0)
                M = mini_batch_constraint_features.shape[0] - 1
                N = mini_batch_variable_features.shape[0] - 1
                
                mini_batch_edge_index = batch.edge_index[:, batch_edge_indices]
                mini_batch_edge_features = batch.edge_attr[batch_edge_indices]
                
                con_node_idx = torch.zeros(batch.constraint_features.shape[0], dtype=torch.long,
                                device=device)
                var_node_idx = torch.zeros(batch.variable_features.shape[0], dtype=torch.long,
                                device=device)
                con_node_idx[constraint_indices] = torch.arange(len(constraint_indices), device=device)
                var_node_idx[variable_indices] = torch.arange(len(variable_indices), device=device)
                
                mini_batch_edge_index[0] = con_node_idx[mini_batch_edge_index[0]]
                mini_batch_edge_index[1] = var_node_idx[mini_batch_edge_index[1]]
                
                # Append edge indices and edge features
                left_to_right_edges = torch.stack([torch.full((N,), M, dtype=torch.long, device=device),torch.arange(N, dtype=torch.long, device=device)], dim=0)
                right_to_left_edges = torch.stack([torch.arange(M, dtype=torch.long, device=device),torch.full((M,), N, dtype=torch.long, device=device)], dim=0)
                mini_batch_edge_index = torch.cat([mini_batch_edge_index, left_to_right_edges, right_to_left_edges], dim=1)
                
                mini_batch_edge_features = torch.cat([mini_batch_edge_features, torch.ones((M + N, 1), dtype=mini_batch_edge_features.dtype, device=device)], dim=0)
                
                # Extract the labels
                size = (len(mini_batch_constraint_features), len(mini_batch_variable_features))
                value = torch.tensor([1 for i in range(len(mini_batch_edge_index[0]))], device=device)
                mini_batch_label = torch.sparse_coo_tensor(mini_batch_edge_index, value, size)
                mini_batch_label = mini_batch_label.coalesce()
                
                model.train()
                optimizer.zero_grad()
                
                # Random mask some edges
                total_edges = len(mini_batch_edge_features)
                num_edges_to_select = math.ceil(total_edges * (1 - args.masked_edge_ratio))
                random_indices = torch.randperm(total_edges)[:num_edges_to_select]
                
                selected_edge_index = mini_batch_edge_index[:, random_indices]
                selected_edge_features = mini_batch_edge_features[random_indices]
                
                out, _, _ = model(mini_batch_constraint_features,
                        selected_edge_index,
                        selected_edge_features,
                        mini_batch_variable_features)

                label = mini_batch_label.to_dense().flatten()
                
                if len(label.shape) == 1:
                    label_ = label.unsqueeze(1)
                
                train_acc = eval_func(label_, out)
                print(train_acc)
                batch_acc_list.append(train_acc)

                # out = F.log_softmax(out, dim=1)
                loss = criterion(
                    out, label)
                loss.backward()
                optimizer.step()

            print("Current training ACC: ", sum(batch_acc_list)/len(batch_acc_list))
        else:
            model.train()
            optimizer.zero_grad()
            
            # Append global variable and constraint features
            constraint_features = torch.cat((batch.constraint_features, torch.mean(batch.constraint_features, dim=0, keepdim=True)), dim=0)
            variable_features = torch.cat((batch.variable_features, torch.mean(batch.variable_features, dim=0, keepdim=True)), dim=0)
            M = constraint_features.shape[0] - 1
            N = variable_features.shape[0] - 1
            
            # Append edge indices and edge features
            left_to_right_edges = torch.stack([torch.full((N,), M, dtype=torch.long, device=device),torch.arange(N, dtype=torch.long, device=device)], dim=0)
            right_to_left_edges = torch.stack([torch.arange(M, dtype=torch.long, device=device),torch.full((M,), N, dtype=torch.long, device=device)], dim=0)
            edge_indices = torch.cat([batch.edge_index, left_to_right_edges, right_to_left_edges], dim=1)
            
            edge_features = torch.cat([batch.edge_attr, torch.ones((M + N, 1), dtype=batch.edge_attr.dtype, device=device)], dim=0)
                        
            # Random mask some edges
            total_edges = len(edge_features)
            num_edges_to_select = math.ceil(total_edges * (1 - args.masked_edge_ratio))
            random_indices = torch.randperm(total_edges)[:num_edges_to_select]
            
            selected_edge_index = edge_indices[:, random_indices]
            selected_edge_features = edge_features[random_indices]
            
            out, _, _ = model(constraint_features,
                    selected_edge_index,
                    selected_edge_features,
                    variable_features)
            
            label_size = (len(constraint_features), len(variable_features))
            label_value = torch.tensor([1 for i in range(len(edge_indices[0]))], device=device)
            batch_label = torch.sparse_coo_tensor(edge_indices, label_value, label_size)
            batch_label = batch_label.coalesce()
            batch_label = batch_label.to(device)

            label = batch_label.to_dense().flatten()
            label = label.to(device)
            
            if len(label.shape) == 1:
                label_ = label.unsqueeze(1)
            
            train_acc = eval_func(label_, out)
            
            # out = F.log_softmax(out, dim=1)
            loss = criterion(
                out, label)
            
            print("Current training ACC: ", train_acc)
            loss.backward()
            optimizer.step()
            
        batch_id += 1

    if epoch % args.eval_step == 0:
        valid_acc, valid_loss = evaluate_mip(model, valid_loader, eval_func, criterion, args, device, Maximum_batch_size_valid)

        if epoch % args.display_step == 0:
            print_str = f'Epoch: {epoch:02d}, ' + \
                        f'Loss: {loss:.4f}, ' + \
                        f'Valid: {100 * valid_acc:.2f}%'
            print(print_str)
        
        if valid_acc > best_acc:
            best_acc = valid_acc
            save_path = model_save_dir + "model_params_epoch_" + str(epoch) + ".pth"
            torch.save(model.state_dict(), save_path)