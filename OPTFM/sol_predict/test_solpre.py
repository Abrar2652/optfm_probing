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
from dataset import load_dataset_bipartite, split_bipartite_graph, BipartiteNodeData
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, eval_recall, to_sparse_tensor, \
    load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from eval_mip import generate_embedding
from parse import parser_add_main_args, parse_method_mip

from generate_graph_scip import bipartite_graph_generation_scip_for_test

from LNS import optimize

import time
import pickle

import random
import math

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Testing Pipeline for Edge Identification')
parser_add_main_args(parser)
args = parser.parse_args()

data_dir = "instances/SC/hard"
instance_list = os.listdir(data_dir)
print(instance_list)

bipartite_graph_generation = bipartite_graph_generation_scip_for_test
var_d = 9
con_d = 1

print(args)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# Label to the device
# for label in label_loader:
#     label = label.to(device)
#     c = label.values().max().item() + 1

# Label count, 二分类问题
c = 2

### Load method ###
model = parse_method_mip(args, c, var_d, con_d, device)

### Load model
model_save_dir = "../node_pretrain/Models_SCIP/"

test_epoch = 0
model_path = model_save_dir + "model_params_epoch_" + str(test_epoch) + ".pth"
model.load_state_dict(torch.load(model_path))

Maximum_batch_size_test = 1000

for instance in instance_list:
    if "mps" not in instance:
        continue
    
    # Extract the features
    var_feas, cons_feas, edge_indices, edge_features, graph_feas, col_bounds, row_bounds = bipartite_graph_generation(data_dir, instance)
    
    graph = BipartiteNodeData(
            torch.FloatTensor(cons_feas),
            torch.LongTensor(edge_indices),
            torch.FloatTensor(edge_features),
            torch.FloatTensor(var_feas)
        )
    
    embeddings = generate_embedding(model, graph, args, device, Maximum_batch_size_test)
    print(embeddings)
    
    set_time = 100
    rate = 0.7
    compare_with_base = False
    
    time, obj, base_time, base_obj = optimize(data_dir + "/" + instance, graph_feas, col_bounds, row_bounds, embeddings, set_time, rate, model_save_dir + 'GBDT.pickle', compare_with_base)